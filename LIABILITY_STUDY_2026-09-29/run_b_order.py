"""Same jobs as run_b.phase_rl (identical code, settings and outputs), submitted in a different order: the NE market
first, seed-major, so that one seed of every NE condition finishes before the next seed starts. run_b.py is unchanged
(its pre-registered hash stays valid); only the submission order differs. Resume-safe: finished jobs are skipped.

    python run_b_order.py --workers 7
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from envs_b import LIABILITY
from run_b import ALGOS, REGIONS, SEEDS, STEPS, _init_worker, job_path, rl_job


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    jobs = [(l, mk, al, s, g) for mk in ("NE", "RM") for s in SEEDS for al in ("DQN", "PPO") for l in LIABILITY for g in REGIONS]
    todo = [j for j in jobs if not job_path(*j).exists()]
    print(f"{len(jobs)} RL jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers (NE first, seed-major)", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(rl_job, *j, STEPS, False): j for j in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:300], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

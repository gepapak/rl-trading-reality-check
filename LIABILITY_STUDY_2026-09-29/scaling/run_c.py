"""Study C runner (PREREGISTRATION_C.md): does learned gambling grow with training? One training run of 1.6M steps per
job, with snapshots at fixed milestones; at each snapshot the policy is evaluated in the no-edge market (20 sign draws of
the test period) and probed at fixed equity levels, exactly as in Study B (run_b.evaluate, run_b.probe).

    python run_c.py --workers 7
    python run_c.py --smoke        # development check: tiny budgets, TRAIN split only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
OUT = HERE / "results"
LIABS = ["FL", "ZF"]
ALGOS = ["PPO", "DQN"]
SEEDS = [10, 11, 12]            # fresh seeds (Study B used 0-4)
REGIONS = ["DK1", "DK2"]
MILESTONES = [100_000, 200_000, 400_000, 800_000, 1_600_000]


def job_path(liab, algo, seed, region, smoke=False) -> Path:
    return OUT / ("jobs_smoke" if smoke else "jobs") / f"{liab}__NE__{algo}__s{seed}__{region}.json"


def c_job(liab: str, algo: str, seed: int, region: str, milestones: list[int], smoke: bool) -> str:
    out = job_path(liab, algo, seed, region, smoke)
    if out.exists():
        return f"skip {out.name}"
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from envs_b import LiabilityEnv, load_market
    from run_b import evaluate, greedy, network_layers, probe
    t0 = time.time()
    mtr = load_market(region, "train", "NE")
    meval = mtr if smoke else load_market(region, "test", "NE")
    snaps = []

    def snapshot(model, step):
        layers = network_layers(model, algo)
        act = lambda env_, obs: int(greedy(layers, obs[None, :])[0])  # noqa: E731
        res = evaluate(act, liab, region, "NE", smoke)
        snaps.append(dict(step=step, **{k: v for k, v in res.items() if not k.startswith("draws_")},
                          **probe(layers, meval), seconds=round(time.time() - t0, 1)))
        out.with_suffix(".partial.json").write_text(json.dumps(snaps, indent=1))

    class Milestones(BaseCallback):
        def __init__(self):
            super().__init__()
            self.i = 0

        def _on_step(self) -> bool:
            while self.i < len(milestones) and self.num_timesteps >= milestones[self.i]:
                snapshot(self.model, milestones[self.i])
                self.i += 1
            return True

    env = LiabilityEnv(mtr, liab, train=True, seed=seed)
    Algo = {"PPO": PPO, "DQN": DQN}[algo]
    model = Algo("MlpPolicy", env, seed=seed, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.learn(total_timesteps=milestones[-1], callback=Milestones())
    if len(snaps) < len(milestones):
        snapshot(model, milestones[-1])
    # exactness check of the NumPy forward pass at the final snapshot
    layers = network_layers(model, algo)
    chk = LiabilityEnv(mtr, liab, train=True, seed=seed + 1000)
    o, _ = chk.reset()
    for _ in range(300):
        assert int(greedy(layers, o[None, :])[0]) == int(model.predict(o, deterministic=True)[0]), "fast policy mismatch"
        o, _, d, _, _ = chk.step(chk.action_space.sample())
        if d:
            o, _ = chk.reset()
    res = dict(liability=liab, market="NE", algo=algo, seed=seed, region=region, eval_split="train(SMOKE)" if smoke else "test",
               milestones=milestones, snapshots=snaps, seconds=round(time.time() - t0, 1))
    out.write_text(json.dumps(res, indent=1))
    out.with_suffix(".partial.json").unlink(missing_ok=True)
    return f"done {out.name} ({res['seconds']}s)"


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ms = [1000, 2000, 3000] if a.smoke else MILESTONES
    jobs = [(l, al, s, g) for al in ALGOS for l in LIABS for s in SEEDS for g in REGIONS]   # PPO (longest) first
    if a.smoke:
        jobs = [j for j in jobs if j[2] == SEEDS[0] and j[3] == "DK1"]
    todo = [j for j in jobs if not job_path(*j, smoke=a.smoke).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(c_job, *j, ms, a.smoke): j for j in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:300], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

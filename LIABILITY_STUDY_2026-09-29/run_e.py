"""Study E runner (PREREGISTRATION_E.md): reward curvature x liability x leverage with equity-scaled positions, placebo
market only. Training and evaluation follow Study B (run_b.py): SB3 defaults, 64-64, gamma 0.99, 400k steps, seeds 0-4,
DK1/DK2, evaluation on 20 sign draws of the test period from equity K0. Resume-safe; one JSON per job.

    python run_e.py --workers 7
    python run_e.py --smoke         # development check: tiny budgets, TRAIN split only
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_e"
ALGOS = ["PPO", "DQN"]
SEEDS = [0, 1, 2, 3, 4]
REGIONS = ["DK1", "DK2"]
STEPS = 400_000
N_DRAWS = 20
PROBE_E = (0.1, 0.25, 1.0, 2.0)


def job_path(rule, levset, algo, seed, region, smoke=False) -> Path:
    return OUT / ("jobs_smoke" if smoke else "jobs") / f"{rule}__{levset}__{algo}__s{seed}__{region}.json"


def run_policy(env, act) -> dict:
    from envs_e import outcome
    obs, _ = env.reset()
    done = False
    while not done:
        obs, _, done, _, _ = env.step(act(env, obs))
    return outcome(env)


def evaluate(act, rule, levset, region, smoke) -> dict:
    from envs_e import CurvatureEnv, load_market
    split = "train" if smoke else "test"
    draws = [run_policy(CurvatureEnv(load_market(region, split, "NE", draw=k), rule, levset, train=False), act)
             for k in range(3 if smoke else N_DRAWS)]
    res = {}
    for key in ("reported_return_pct", "ledger_return_pct", "mean_abs_lev_open", "share_top_lev_open"):
        res[f"draws_{key}"] = [d[key] for d in draws]
        res[f"drawmean_{key}"] = float(np.mean(res[f"draws_{key}"]))
    res["drawshare_bust"] = float(np.mean([d["bust"] for d in draws]))
    return res


def probe(layers, m, LEV) -> dict:
    from envs_b import obs_matrix
    from run_b import greedy
    ts = np.arange(200, m.n, 8)
    return {f"probe_e{e:g}": float(np.abs(LEV[greedy(layers, obs_matrix(m, ts, e))]).mean()) for e in PROBE_E}


def e_job(rule, levset, algo, seed, region, steps, smoke) -> str:
    out = job_path(rule, levset, algo, seed, region, smoke)
    if out.exists():
        return f"skip {out.name}"
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import DQN, PPO
    from envs_e import LEVSETS, CurvatureEnv, load_market
    from run_b import greedy, network_layers
    t0 = time.time()
    mtr = load_market(region, "train", "NE")
    meval = mtr if smoke else load_market(region, "test", "NE")
    env = CurvatureEnv(mtr, rule, levset, train=True, seed=seed)
    model = {"PPO": PPO, "DQN": DQN}[algo]("MlpPolicy", env, seed=seed, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    layers = network_layers(model, algo)
    act = lambda env_, obs: int(greedy(layers, obs[None, :])[0])  # noqa: E731
    chk = CurvatureEnv(mtr, rule, levset, train=True, seed=seed + 1000)
    o, _ = chk.reset()
    for _ in range(300):
        assert act(None, o) == int(model.predict(o, deterministic=True)[0]), "fast policy mismatch"
        o, _, d, _, _ = chk.step(chk.action_space.sample())
        if d:
            o, _ = chk.reset()
    res = dict(rule=rule, levset=levset, algo=algo, seed=seed, region=region, steps=steps,
               eval_split="train(SMOKE)" if smoke else "test", **evaluate(act, rule, levset, region, smoke),
               **probe(layers, meval, LEVSETS[levset]), seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return f"done {out.name} ({res['seconds']}s)"


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    from envs_e import LEVSETS, RULES
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    jobs = [(r, l, al, s, g) for s in SEEDS for al in ("PPO", "DQN") for l in LEVSETS for r in RULES for g in REGIONS]  # seed-major
    if a.smoke:
        jobs = [j for j in jobs if j[3] == 0 and j[4] == "DK1"]
    steps = 3000 if a.smoke else STEPS
    todo = [j for j in jobs if not job_path(*j, smoke=a.smoke).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(e_job, *j, steps, a.smoke): j for j in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:300], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

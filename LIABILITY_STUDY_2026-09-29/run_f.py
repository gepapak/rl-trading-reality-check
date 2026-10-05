"""Study F runner (PREREGISTRATION_F.md): do learning agents reach the theoretical liability policy when the learning
problem is well posed? Placebo market, fixed sizing (Study B environment), three liability rules, two observation sets,
PPO with standard best practice (8 parallel environments, reward normalization), 1M steps, seeds 0-4, DK1/DK2.

    python run_f.py --workers 7
    python run_f.py --smoke        # development check: tiny budget, TRAIN split only
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
OUT = HERE / "results_f"
RULES = ["FL", "LL", "ZF"]
OBSSETS = ["EQ", "FULL"]          # EQ: equity is the only input; FULL: Study B's 11 inputs
SEEDS = [0, 1, 2, 3, 4]
REGIONS = ["DK1", "DK2"]
STEPS = 1_000_000
N_ENVS = 8
N_DRAWS = 20
EGRID = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0)


def job_path(rule, obsset, seed, region, smoke=False) -> Path:
    return OUT / ("jobs_smoke" if smoke else "jobs") / f"{rule}__{obsset}__PPO__s{seed}__{region}.json"


def f_job(rule: str, obsset: str, seed: int, region: str, steps: int, smoke: bool) -> str:
    out = job_path(rule, obsset, seed, region, smoke)
    if out.exists():
        return f"skip {out.name}"
    import gymnasium as gym
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_b import LEV, LiabilityEnv, load_market, obs_matrix, outcome
    from run_b import greedy, network_layers
    t0 = time.time()
    mtr = load_market(region, "train", "NE")
    split = "train" if smoke else "test"

    class EquityOnly(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(0, 5, shape=(1,), dtype=np.float32)

        def observation(self, obs):
            return np.array([obs[6]], dtype=np.float32)   # reported equity / K0, clipped to [0, 5]

    wrap = (lambda e: EquityOnly(e)) if obsset == "EQ" else (lambda e: e)
    venv = DummyVecEnv([(lambda i: (lambda: wrap(LiabilityEnv(mtr, rule, train=True, seed=seed * 100 + i))))(i) for i in range(N_ENVS)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, policy_kwargs=dict(net_arch=[64, 64]),
                device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    layers = network_layers(model, "PPO")
    pick = (lambda o: o[6:7]) if obsset == "EQ" else (lambda o: o)
    act = lambda env_, o: int(greedy(layers, pick(o)[None, :])[0])  # noqa: E731
    # exactness check of the NumPy forward pass
    chk = wrap(LiabilityEnv(mtr, rule, train=True, seed=seed + 1000))
    o, _ = chk.reset()
    for _ in range(300):
        assert int(greedy(layers, np.asarray(o)[None, :])[0]) == int(model.predict(o, deterministic=True)[0]), "fast policy mismatch"
        o, _, d, _, _ = chk.step(chk.action_space.sample())
        if d:
            o, _ = chk.reset()
    # leverage at fixed equity levels (EQ: exact; FULL: mean over every 8th test quarter-hour, as in Study B)
    meval = load_market(region, split, "NE")
    lev = {}
    for e in EGRID:
        if obsset == "EQ":
            lev[f"{e:g}"] = float(abs(LEV[greedy(layers, np.array([[min(e, 5.0)]]))[0]]))
        else:
            lev[f"{e:g}"] = float(np.abs(LEV[greedy(layers, obs_matrix(meval, np.arange(200, meval.n, 8), e))]).mean())
    # placebo-market evaluation from equity K0 over sign draws of the evaluation period
    draws = []
    for k in range(3 if smoke else N_DRAWS):
        env = LiabilityEnv(load_market(region, split, "NE", draw=k), rule, train=False)
        o, _ = env.reset(options={"e0": 1.0})
        done = False
        while not done:
            o, _, done, _, _ = env.step(act(env, o))
        draws.append(outcome(env))
    res = dict(rule=rule, obsset=obsset, algo="PPO", seed=seed, region=region, steps=steps, eval_split=split, lev=lev,
               drawmean_reported_return_pct=float(np.mean([d["reported_return_pct"] for d in draws])),
               drawmean_ledger_return_pct=float(np.mean([d["ledger_return_pct"] for d in draws])),
               drawmean_mean_abs_lev_open=float(np.mean([d["mean_abs_lev_open"] for d in draws])),
               drawshare_floor_bound=float(np.mean([d["floor_bound"] for d in draws])),
               drawshare_closed=float(np.mean([d["closed"] for d in draws])),
               draws_reported_return_pct=[d["reported_return_pct"] for d in draws],
               seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return f"done {out.name} ({res['seconds']}s): L(0.1)={lev['0.1']:g} L(2)={lev['2']:g}"


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    jobs = [(r, o, s, g) for s in SEEDS for o in OBSSETS for r in RULES for g in REGIONS]
    if a.smoke:
        jobs = [j for j in jobs if j[2] == 0 and j[3] == "DK1"]
    steps = 16_384 if a.smoke else STEPS
    todo = [j for j in jobs if not job_path(*j, smoke=a.smoke).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(f_job, *j, steps, a.smoke): j for j in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:300], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Study I runner (PREREGISTRATION_I.md): is the well-posed result of Study F specific to PPO?

Study F's EQ condition (placebo market, fixed sizing, equity as the only input, 8 parallel environments, reward
normalization, 1M steps, seeds 0-4, DK1/DK2) with two further learners: A2C (on-policy actor-critic; two
configurations) and DQN (off-policy, value-based; library defaults). Evaluation identical to run_f.py.

    python run_i.py --workers 7
    python run_i.py --smoke        # development check: tiny budget, TRAIN split only
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
OUT = HERE / "results_i"
CONFIGS = ["A2C-n256", "A2C-n64", "DQN-default"]   # learner-configuration; chosen from the pilot (PREREGISTRATION_I.md)
RULES = ["FL", "LL", "ZF"]
SEEDS = [0, 1, 2, 3, 4]
REGIONS = ["DK1", "DK2"]
STEPS = 1_000_000
N_ENVS = 8
N_DRAWS = 20
EGRID = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0)
KW = {"A2C-n256": dict(n_steps=256, gae_lambda=0.95, normalize_advantage=True),
      "A2C-n64": dict(n_steps=64, gae_lambda=0.95, normalize_advantage=True),
      "DQN-default": {}}


def job_path(cfg, rule, seed, region, smoke=False) -> Path:
    return OUT / ("jobs_smoke" if smoke else "jobs") / f"{rule}__EQ__{cfg}__s{seed}__{region}.json"


def i_job(cfg: str, rule: str, seed: int, region: str, steps: int, smoke: bool) -> str:
    out = job_path(cfg, rule, seed, region, smoke)
    algo = cfg.split("-")[0]
    if out.exists():
        return f"skip {out.name}"
    import gymnasium as gym
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import A2C, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_b import LEV, LiabilityEnv, load_market, outcome
    from run_b import greedy
    t0 = time.time()
    mtr = load_market(region, "train", "NE")
    split = "train" if smoke else "test"

    class EquityOnly(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(0, 5, shape=(1,), dtype=np.float32)

        def observation(self, obs):
            return np.array([obs[6]], dtype=np.float32)   # reported equity / K0, clipped to [0, 5]

    venv = DummyVecEnv([(lambda i: (lambda: EquityOnly(LiabilityEnv(mtr, rule, train=True, seed=seed * 100 + i))))(i)
                        for i in range(N_ENVS)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)
    Algo = A2C if algo == "A2C" else DQN
    model = Algo("MlpPolicy", venv, seed=seed, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0, **KW[cfg])
    model.learn(total_timesteps=steps)
    layers = network_layers_any(model, algo)
    act = lambda env_, o: int(greedy(layers, np.asarray(o)[6:7][None, :])[0])  # noqa: E731
    # exactness check of the NumPy forward pass
    chk = EquityOnly(LiabilityEnv(mtr, rule, train=True, seed=seed + 1000))
    o, _ = chk.reset()
    for _ in range(300):
        assert int(greedy(layers, np.asarray(o)[None, :])[0]) == int(model.predict(o, deterministic=True)[0]), "fast policy mismatch"
        o, _, d, _, _ = chk.step(chk.action_space.sample())
        if d:
            o, _ = chk.reset()
    lev = {f"{e:g}": float(abs(LEV[greedy(layers, np.array([[min(e, 5.0)]]))[0]])) for e in EGRID}
    draws = []
    for k in range(3 if smoke else N_DRAWS):
        env = LiabilityEnv(load_market(region, split, "NE", draw=k), rule, train=False)
        o, _ = env.reset(options={"e0": 1.0})
        done = False
        while not done:
            o, _, done, _, _ = env.step(act(env, o))
        draws.append(outcome(env))
    res = dict(rule=rule, obsset="EQ", algo=algo, config=cfg, kwargs=KW[cfg], seed=seed, region=region, steps=steps,
               eval_split=split, lev=lev,
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


def network_layers_any(model, algo: str):
    """NumPy layers of the greedy network: A2C as PPO (policy MLP + action head), DQN its Q-network."""
    from run_b import network_layers
    return network_layers(model, "PPO" if algo == "A2C" else "DQN")


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    jobs = [(c, r, s, g) for s in SEEDS for c in CONFIGS for r in RULES for g in REGIONS]
    if a.smoke:
        jobs = [j for j in jobs if j[2] == 0 and j[3] == "DK1"]
    steps = 16_384 if a.smoke else STEPS
    todo = [j for j in jobs if not job_path(*j, smoke=a.smoke).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(i_job, *j, steps, a.smoke): j for j in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:300], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""EXPLORATORY PILOT for Study I (training split only, not pre-registered): do learners other than PPO reach the
theoretical liability policy in Study F's well-posed set-up?

Same set-up as Study F's EQ condition (equity as the only input, 8 parallel environments, reward normalization,
1M steps), with A2C (on-policy actor-critic without clipping or epochs) and DQN (off-policy, value-based) at
Stable-Baselines3 defaults. DQN is also run with 8 gradient steps per update (variant "g8"), because with 8 parallel
environments the default performs one gradient step per 32 collected transitions.

    python pilot_i.py --workers 6
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
OUT = HERE / "results_pilot_i"
EGRID = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0)
N_ENVS = 8


def job(rule: str, algo: str, variant: str, seed: int, region: str) -> str:
    import gymnasium as gym
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import A2C, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_b import LEV, LiabilityEnv, load_market
    tag = f"{rule}__{algo}__{variant}__s{seed}__{region}"
    out = OUT / f"{tag}.json"
    if out.exists():
        return f"skip {tag}"
    t0 = time.time()
    m = load_market(region, "train", "NE")

    class EquityOnly(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(0, 5, shape=(1,), dtype=np.float32)

        def observation(self, obs):
            return np.array([obs[6]], dtype=np.float32)   # reported equity / K0, clipped to [0, 5]

    def mk(i):
        return lambda: EquityOnly(LiabilityEnv(m, rule, train=True, seed=seed * 100 + i))
    venv = VecNormalize(DummyVecEnv([mk(i) for i in range(N_ENVS)]), norm_obs=False, norm_reward=True, gamma=0.99)
    if algo == "A2C":
        # second pilot round: larger rollouts with GAE and advantage normalization (defaults collapsed, round 1)
        kw = {"default": {}, "n64": dict(n_steps=64, gae_lambda=0.95, normalize_advantage=True),
              "n256": dict(n_steps=256, gae_lambda=0.95, normalize_advantage=True)}[variant]
        model = A2C("MlpPolicy", venv, seed=seed, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0, **kw)
    else:
        kw = dict(gradient_steps=8) if variant == "g8" else {}
        model = DQN("MlpPolicy", venv, seed=seed, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0, **kw)
    snaps, done_steps = [], 0
    for target in (250_000, 500_000, 1_000_000):
        model.learn(total_timesteps=target - done_steps, reset_num_timesteps=False)
        done_steps = target
        acts = model.predict(np.array(EGRID, dtype=np.float32)[:, None], deterministic=True)[0]
        snaps.append(dict(step=target, lev={f"{e:g}": float(abs(LEV[int(a)])) for e, a in zip(EGRID, acts)},
                          seconds=round(time.time() - t0, 1)))
    OUT.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(rule=rule, algo=algo, variant=variant, seed=seed, region=region, snapshots=snaps,
                                   seconds=round(time.time() - t0, 1)), indent=1))
    return f"done {tag} ({time.time() - t0:.0f}s): " + str(snaps[-1]["lev"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--round", type=int, default=1)
    a = ap.parse_args()
    sets = {1: (("A2C", "default"), ("DQN", "default"), ("DQN", "g8")), 2: (("A2C", "n64"), ("A2C", "n256"))}[a.round]
    jobs = [(r, algo, v, 0, "DK1") for algo, v in sets for r in ("FL", "ZF")]
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(job, *j): j for j in jobs}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print("FAILED", futs[f], repr(e)[:300], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

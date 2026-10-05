"""EXPLORATORY PILOT (training split only, not pre-registered): can learning agents reach the theoretical liability
policy when the learning problem is well posed?

In the placebo market every input except equity is noise by construction. The pilot gives the agent equity as its only
input and trains with standard best practice (8 parallel environments, reward normalization), then reads the greedy
leverage at fixed equity levels. Compare with the optimum (dp_theory.py): full liability -> 0 everywhere; zero floor ->
16 up to one allocation, 0 at two.

    python pilot_f.py --workers 4
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
OUT = HERE / "results_pilot_f"
EGRID = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0)


def job(rule: str, algo: str, norm_reward: bool, seed: int, region: str, steps: int) -> str:
    import gymnasium as gym
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs_b import LEV, LiabilityEnv, load_market
    tag = f"{rule}__{algo}__{'nr' if norm_reward else 'raw'}__s{seed}__{region}"
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
    n_envs = 8 if algo == "PPO" else 1
    venv = DummyVecEnv([mk(i) for i in range(n_envs)])
    if norm_reward:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)
    Algo = {"PPO": PPO, "DQN": DQN}[algo]
    kw = dict(n_steps=1024, batch_size=256) if algo == "PPO" else {}
    model = Algo("MlpPolicy", venv, seed=seed, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0, **kw)
    snaps = []
    done_steps = 0
    for target in (250_000, 500_000, 1_000_000)[: {250_000: 1, 500_000: 2, 1_000_000: 3}[steps]]:
        model.learn(total_timesteps=target - done_steps, reset_num_timesteps=False)
        done_steps = target
        acts = model.predict(np.array(EGRID, dtype=np.float32)[:, None], deterministic=True)[0]
        snaps.append(dict(step=target, lev={f"{e:g}": float(abs(LEV[int(a)])) for e, a in zip(EGRID, acts)}))
    OUT.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(rule=rule, algo=algo, norm_reward=norm_reward, seed=seed, region=region,
                                   snapshots=snaps, seconds=round(time.time() - t0, 1)), indent=1))
    return f"done {tag} ({time.time() - t0:.0f}s): " + str(snaps[-1]["lev"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1_000_000)
    a = ap.parse_args()
    jobs = [(r, "PPO", nr, 0, "DK1") for r in ("FL", "ZF") for nr in (True, False)]
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(job, *j, a.steps): j for j in jobs}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print("FAILED", futs[f], repr(e)[:300], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

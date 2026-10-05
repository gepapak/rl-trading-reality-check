"""Study G runner (PREREGISTRATION_G.md): PPO trained in gym-mtsim 2.0.0 (published accounting, ORIG) vs the same code
without the zero floor on the balance (FLPATCH), on synthetic placebo forex prices. Run with the study venv:

    .venv\\Scripts\\python run_g.py --workers 7
    .venv\\Scripts\\python run_g.py --smoke
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
OUT = HERE / "results_g"
CONDS = ["ORIG", "FLPATCH"]
SEEDS = list(range(10))
STEPS = 1_000_000
N_ENVS = 8
TRAIN_PRICE_SEED, TEST_PRICE_SEED0, N_TEST, TEST_HOURS = 1000, 2000, 20, 2000
EGRID = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


def job_path(cond, seed, smoke=False) -> Path:
    return OUT / ("jobs_smoke" if smoke else "jobs") / f"{cond}__PPO__s{seed}.json"


def exposure(model, e: float) -> float:
    """Fraction of the margin-feasible maximum position opened by the deterministic policy at equity e x 10k
    (no open order, no margin): 0 if it holds or the order would be refused."""
    a = model.predict(np.array([[e, e, 0.0, 0.0, 0.0]], dtype=np.float32), deterministic=True)[0][0]
    hold = 1.0 / (1.0 + np.exp(-a[1])) > 0.5
    if hold:
        return 0.0
    lots = float(np.clip(abs(a[2]), 0.01, 500.0))
    feasible = e * 1e4 * 100.0 / (1e5 * 1.10)          # leverage 100, contract 100k, price ~1.10
    return float(min(lots / feasible, 1.0)) if lots <= feasible else 0.0


def evaluate(model, vecnorm, smoke: bool) -> dict:
    """Deterministic policy on placebo test paths from balance 10k, in ORIG and FLPATCH accounting (same actions until
    trading stops): reported (ORIG) vs booked (FLPATCH) return."""
    from mtsim_common import make_env, placebo_prices
    rep, led, wiped = [], [], []
    for k in range(2 if smoke else N_TEST):
        p = placebo_prices(TEST_PRICE_SEED0 + k, hours=300 if smoke else TEST_HOURS)
        out = {}
        for cond in ("ORIG", "FLPATCH"):
            env = make_env(cond, p, train=False)
            o, info = env.reset(options={"balance": 1e4})
            done, minbal = False, 1e4
            while not done:
                a = model.predict(vecnorm.normalize_obs(o[None, :]), deterministic=True)[0][0]
                o, r, te, tr, info = env.step(a)
                minbal = min(minbal, info["balance"])
                done = te or tr
            out[cond] = (info["equity"] / 1e4 - 1.0, minbal)
        rep.append(100 * out["ORIG"][0]); led.append(100 * out["FLPATCH"][0]); wiped.append(out["ORIG"][1] <= 1e-9)
    return dict(reported_return_pct=float(np.mean(rep)), booked_return_pct=float(np.mean(led)),
                draws_reported=rep, draws_booked=led, wiped_share=float(np.mean(wiped)))


def g_job(cond: str, seed: int, steps: int, smoke: bool) -> str:
    out = job_path(cond, seed, smoke)
    if out.exists():
        return f"skip {out.name}"
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from mtsim_common import make_env, placebo_prices
    t0 = time.time()
    prices = placebo_prices(TRAIN_PRICE_SEED, hours=4000 if smoke else 30_000)
    venv = DummyVecEnv([(lambda i: (lambda: make_env(cond, prices, train=True, seed=seed * 100 + i)))(i) for i in range(N_ENVS)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=0.99)
    model = PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, policy_kwargs=dict(net_arch=[64, 64]),
                device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    venv.training = False

    class Normed:   # probe through the frozen observation normalization
        def predict(self, x, deterministic=True):
            return model.predict(venv.normalize_obs(x), deterministic=deterministic)
    expo = {f"{e:g}": exposure(Normed(), e) for e in EGRID}
    res = dict(condition=cond, algo="PPO", seed=seed, steps=steps, exposure=expo, **evaluate(model, venv, smoke),
               seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return f"done {out.name} ({res['seconds']}s): x(0.1)={expo['0.1']:.2f} x(1)={expo['1']:.2f} rep={res['reported_return_pct']:+.1f} booked={res['booked_return_pct']:+.1f}"


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    jobs = [(c, s) for s in SEEDS for c in CONDS]
    if a.smoke:
        jobs = [j for j in jobs if j[1] == 0]
    steps = 16_384 if a.smoke else STEPS
    todo = [j for j in jobs if not job_path(*j, smoke=a.smoke).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(g_job, *j, steps, a.smoke): j for j in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:400], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

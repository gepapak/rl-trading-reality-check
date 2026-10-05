"""Study B runner (PREREGISTRATION_B.md): train PPO and DQN under each liability rule and market, evaluate on the test
period from full equity, and probe the leverage each trained policy chooses at fixed equity levels. Resume-safe; one
JSON per job.

    python run_b.py --phase rules            # rule baselines (descriptive); minutes
    python run_b.py --phase rl --workers 6   # 120 trainings + evaluations + probes
    python run_b.py --smoke                  # development check: tiny budgets, evaluation on the TRAIN split only
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ALGOS = ["PPO", "DQN"]
SEEDS = [0, 1, 2, 3, 4]
REGIONS = ["DK1", "DK2"]
STEPS = 400_000
PROBE_E = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)  # equity as a multiple of K0
PROBE_STRIDE = 8


def run_policy(env, act, e0: float = 1.0) -> dict:
    from envs_b import outcome
    obs, _ = env.reset(options={"e0": e0})
    done = False
    while not done:
        obs, _, done, _, _ = env.step(act(env, obs))
    return outcome(env)


def evaluate(act, liab: str, region: str, market: str, smoke: bool) -> dict:
    """Main evaluation path; for NE also the mean over N_DRAWS independent sign draws of the evaluation period."""
    from envs_b import N_DRAWS, LiabilityEnv, load_market
    split = "train" if smoke else "test"
    res = run_policy(LiabilityEnv(load_market(region, split, market), liab, train=False), act)
    if market == "NE":
        draws = [res] + [run_policy(LiabilityEnv(load_market(region, split, market, draw=k), liab, train=False), act)
                         for k in range(1, 3 if smoke else N_DRAWS)]
        for key in ("reported_return_pct", "ledger_return_pct", "mean_abs_lev_open", "max_gap_pp"):
            res[f"draws_{key}"] = [d[key] for d in draws]
            res[f"drawmean_{key}"] = float(np.mean(res[f"draws_{key}"]))
        res["drawshare_floor_bound"] = float(np.mean([d["floor_bound"] for d in draws]))
        res["drawshare_closed"] = float(np.mean([d["closed"] for d in draws]))
    return res


# ----------------------------------------------------------------------------------------------------------- rules
def rule_candidates() -> dict:
    """Index into envs_b.LEV = [-16, -4, -1, -0.25, 0, 0.25, 1, 4, 16]."""
    from envs_b import LEV
    idx = {float(v): i for i, v in enumerate(LEV)}
    c = {"flat": ("flat", lambda env, obs: idx[0.0])}
    for L in (0.25, 1.0, 4.0, 16.0):
        c[f"persistence_x{L:g}"] = ("persistence",
                                    (lambda L: lambda env, obs: idx[L] if env.m.spread[env.t - 2] > 0 else (idx[-L] if env.m.spread[env.t - 2] < 0 else idx[0.0]))(L))
        c[f"long_x{L:g}"] = ("constant", (lambda L: lambda env, obs: idx[L])(L))
        c[f"short_x{L:g}"] = ("constant", (lambda L: lambda env, obs: idx[-L])(L))
    return c


def phase_rules(smoke: bool = False) -> None:
    from envs_b import LIABILITY, MARKETS, LiabilityEnv, load_market
    rows = []
    cands = rule_candidates()
    for region in REGIONS:
        for market in MARKETS:
            mtr = load_market(region, "train", market)
            for liab in LIABILITY:
                chosen = {"flat": cands["flat"][1]}
                for fam in ("persistence", "constant"):
                    fc = {k: f for k, (g, f) in cands.items() if g == fam}
                    score = {k: run_policy(LiabilityEnv(mtr, liab, train=False), f)["reported_return_pct"] for k, f in fc.items()}
                    best = max(score, key=score.get)
                    chosen[f"{fam}_tuned:{best}"] = fc[best]
                for name, f in chosen.items():
                    res = evaluate(f, liab, region, market, smoke)
                    rows.append(dict(liability=liab, market=market, region=region, policy=name,
                                     **{k: v for k, v in res.items() if not k.startswith("draws_")}))
            print(f"rules done: {region} {market}", flush=True)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / ("rules_SMOKE_train_split.csv" if smoke else "rules.csv"), index=False)


# -------------------------------------------------------------------------------------------------------------- RL
def job_path(liab, market, algo, seed, region, smoke=False) -> Path:
    return OUT / ("jobs_smoke" if smoke else "jobs") / f"{liab}__{market}__{algo}__s{seed}__{region}.json"


def network_layers(model, algo: str):
    import torch.nn as nn
    mods = (list(model.policy.mlp_extractor.policy_net) + [model.policy.action_net]) if algo == "PPO" else list(model.policy.q_net.q_net)
    layers = []
    for m in mods:
        if isinstance(m, nn.Linear):
            layers.append(("lin", m.weight.detach().cpu().numpy().astype(np.float64), m.bias.detach().cpu().numpy().astype(np.float64)))
        elif isinstance(m, nn.Tanh):
            layers.append(("tanh", None, None))
        elif isinstance(m, nn.ReLU):
            layers.append(("relu", None, None))
        else:
            raise TypeError(f"unsupported layer {m}")
    return layers


def greedy(layers, X: np.ndarray) -> np.ndarray:
    """Deterministic greedy actions for a batch of observations (rows), identical to model.predict(deterministic=True)."""
    x = np.asarray(X, dtype=np.float64)
    for kind, w, b in layers:
        x = x @ w.T + b if kind == "lin" else (np.tanh(x) if kind == "tanh" else np.maximum(x, 0.0))
    return np.argmax(x, axis=1)


def probe(layers, m) -> dict:
    """Mean |leverage| chosen over test quarters when the equity input is set to each level in PROBE_E."""
    from envs_b import LEV, obs_matrix
    ts = np.arange(200, m.n, PROBE_STRIDE)
    return {f"probe_e{e:g}": float(np.abs(LEV[greedy(layers, obs_matrix(m, ts, e))]).mean()) for e in PROBE_E}


def rl_job(liab: str, market: str, algo: str, seed: int, region: str, steps: int, smoke: bool) -> str:
    out = job_path(liab, market, algo, seed, region, smoke)
    if out.exists():
        return f"skip {out.name}"
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import DQN, PPO
    from envs_b import LiabilityEnv, load_market
    t0 = time.time()
    mtr = load_market(region, "train", market)
    meval = mtr if smoke else load_market(region, "test", market)
    env = LiabilityEnv(mtr, liab, train=True, seed=seed)
    Algo = {"PPO": PPO, "DQN": DQN}[algo]
    model = Algo("MlpPolicy", env, seed=seed, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    layers = network_layers(model, algo)
    act = lambda env_, obs: int(greedy(layers, obs[None, :])[0])  # noqa: E731
    # exactness check: the NumPy forward pass must reproduce model.predict on sampled observations
    chk = LiabilityEnv(mtr, liab, train=True, seed=seed + 1000)
    o, _ = chk.reset()
    for _ in range(300):
        assert act(None, o) == int(model.predict(o, deterministic=True)[0]), "fast policy mismatch"
        o, _, d, _, _ = chk.step(chk.action_space.sample())
        if d:
            o, _ = chk.reset()
    res = dict(liability=liab, market=market, algo=algo, seed=seed, region=region, steps=steps,
               eval_split="train(SMOKE)" if smoke else "test",
               **evaluate(act, liab, region, market, smoke), **probe(layers, meval),
               seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return f"done {out.name} ({res['seconds']}s)"


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def phase_rl(workers: int, smoke: bool = False) -> None:
    from envs_b import LIABILITY, MARKETS
    jobs = [(l, mk, a, s, g) for mk in MARKETS for l in LIABILITY for a in ALGOS for s in SEEDS for g in REGIONS]
    if smoke:
        jobs = [j for j in jobs if j[3] == 0 and j[4] == "DK1"]
    steps = 3000 if smoke else STEPS
    todo = [j for j in jobs if not job_path(*j, smoke=smoke).exists()]
    print(f"{len(jobs)} RL jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futs = {ex.submit(rl_job, *j, steps, smoke): j for j in todo}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e)[:300], flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["rules", "rl", "all"], default="all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.phase in ("rules", "all"):
        phase_rules(a.smoke)
    if a.phase in ("rl", "all"):
        phase_rl(a.workers, a.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

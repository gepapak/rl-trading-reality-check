"""Study A runner (PREREGISTRATION_A.md): train RL agents and tune rules within each regime, then evaluate every policy
over the test period under its own regime ("reported") and under S0 ("reality"). Resume-safe; one JSON per job.

    python run_study.py --phase rules            # rule tuning (train) + evaluation (test); minutes
    python run_study.py --phase rl --workers 6   # 280 RL trainings + evaluations
    python run_study.py --smoke                  # development check: tiny budgets, evaluation on the TRAIN split only
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
STEPS = 200_000


def load_markets(region: str):
    from envs import Market
    panel = pd.read_pickle(HERE / "data_panel" / f"panel_{region}.pkl")
    return Market(panel, "train"), Market(panel, "test")


def run_policy(env, act) -> dict:
    """Evaluation pass; act(env, obs) -> action index."""
    from envs import outcome
    obs, _ = env.reset()
    done = False
    while not done:
        obs, _, done, _, _ = env.step(act(env, obs))
    return outcome(env)


def headline(env_name: str, res: dict) -> float:
    return res["profit_eur"] if env_name.startswith("E1") else (res["reported_return_pct"] if env_name == "E2" else res["revenue_eur"])


# ----------------------------------------------------------------------------------------------------------- rules
def rule_threshold(lo: float, hi: float):
    def act(env, obs):
        sig = env.m.imb_lags[env.t, 4] * 100.0      # mean imbalance price t-2 .. t-9
        return 0 if sig < lo else (4 if sig > hi else 2)
    return act


def rule_const(a: int):
    return lambda env, obs: a


def rule_persistence(env, obs):
    s = env.m.spread[env.t - 2]
    return 3 if s > 0 else (1 if s < 0 else 2)


def rule_candidates(env_name: str, mtrain) -> dict:
    if env_name.startswith("E1"):
        sig = mtrain.imb_lags[:, 4] * 100.0
        c = {"idle": rule_const(2)}
        for ql in (0.1, 0.2, 0.3, 0.4):
            for qh in (0.6, 0.7, 0.8, 0.9):
                c[f"threshold_q{ql}_q{qh}"] = (rule_threshold(float(np.quantile(sig, ql)), float(np.quantile(sig, qh))), "threshold")
        return c
    if env_name == "E2":
        return {"persistence": rule_persistence, "flat": rule_const(2)}
    return {"bid_forecast": rule_const(2), **{f"shade_{a}": (rule_const(a), "shade") for a in range(5)}}


def phase_rules(smoke: bool = False) -> None:
    from envs import ENV_CONFIGS, make_env
    rows = []
    for region in REGIONS:
        mtr, mte = load_markets(region)
        meval = mtr if smoke else mte
        for env_name, cfg in ENV_CONFIGS.items():
            cands = rule_candidates(env_name, mtr)
            for regime in cfg["regimes"]:
                # tune each tunable family on the TRAIN split within this regime; fixed rules pass through
                chosen = {}
                for fam in ("threshold", "shade"):
                    fam_c = {k: v[0] for k, v in cands.items() if isinstance(v, tuple) and v[1] == fam}
                    if fam_c:
                        score = {k: headline(env_name, run_policy(make_env(env_name, regime, mtr, train=False), f)) for k, f in fam_c.items()}
                        best = max(score, key=score.get)
                        chosen[f"{fam}_tuned"] = (fam_c[best], best)
                for k, v in cands.items():
                    if not isinstance(v, tuple):
                        chosen[k] = (v, k)
                for name, (f, param) in chosen.items():
                    rep = run_policy(make_env(env_name, regime, meval, train=False), f)
                    real = run_policy(make_env(env_name, "S0", meval, train=False), f)
                    rows.append(dict(env=env_name, regime=regime, region=region, policy=name, param=param,
                                     reported=headline(env_name, rep), reality=headline(env_name, real),
                                     **{f"rep_{k}": v for k, v in rep.items()}, **{f"real_{k}": v for k, v in real.items()}))
            print(f"rules done: {region} {env_name}", flush=True)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / ("rules_SMOKE_train_split.csv" if smoke else "rules.csv"), index=False)


# -------------------------------------------------------------------------------------------------------------- RL
def job_path(env_name, regime, algo, seed, region, smoke=False) -> Path:
    return OUT / ("jobs_smoke" if smoke else "jobs") / f"{env_name}__{regime}__{algo}__s{seed}__{region}.json"


def rl_job(env_name: str, regime: str, algo: str, seed: int, region: str, steps: int, smoke: bool) -> str:
    out = job_path(env_name, regime, algo, seed, region, smoke)
    if out.exists():
        return f"skip {out.name}"
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import DQN, PPO
    from envs import make_env
    t0 = time.time()
    mtr, mte = load_markets(region)
    meval = mtr if smoke else mte
    env = make_env(env_name, regime, mtr, train=True, seed=seed)
    Algo = {"PPO": PPO, "DQN": DQN}[algo]
    model = Algo("MlpPolicy", env, seed=seed, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)
    model.learn(total_timesteps=steps)
    act = fast_policy(model, algo)
    # exactness check: the NumPy forward pass must reproduce model.predict on sampled observations
    probe = make_env(env_name, regime, mtr, train=True, seed=seed + 1000)
    o, _ = probe.reset()
    for _ in range(300):
        assert act(None, o) == int(model.predict(o, deterministic=True)[0]), "fast policy mismatch"
        o, _, d, _, _ = probe.step(probe.action_space.sample())
        if d:
            o, _ = probe.reset()
    rep = run_policy(make_env(env_name, regime, meval, train=False), act)
    real = run_policy(make_env(env_name, "S0", meval, train=False, proxy=(regime == "S4")), act)
    res = dict(env=env_name, regime=regime, algo=algo, seed=seed, region=region, steps=steps, seconds=round(time.time() - t0, 1),
               eval_split="train(SMOKE)" if smoke else "test", reported=headline(env_name, rep), reality=headline(env_name, real),
               **{f"rep_{k}": v for k, v in rep.items()}, **{f"real_{k}": v for k, v in real.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return f"done {out.name} ({res['seconds']}s)"


def fast_policy(model, algo: str):
    """Deterministic greedy action via a NumPy forward pass of the trained network (identical to model.predict)."""
    import torch.nn as nn
    if algo == "PPO":
        mods = list(model.policy.mlp_extractor.policy_net) + [model.policy.action_net]
    else:
        mods = list(model.policy.q_net.q_net)
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

    def act(env, obs):
        x = np.asarray(obs, dtype=np.float64)
        for kind, w, b in layers:
            x = w @ x + b if kind == "lin" else (np.tanh(x) if kind == "tanh" else np.maximum(x, 0.0))
        return int(np.argmax(x))
    return act


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def phase_rl(workers: int, smoke: bool = False) -> None:
    from envs import ENV_CONFIGS
    jobs = [(e, r, a, s, g) for e, cfg in ENV_CONFIGS.items() for r in cfg["regimes"] for a in ALGOS for s in SEEDS for g in REGIONS]
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

"""Study I exploratory analysis (not pre-registered): per-run curves, shape of the zero-floor policies, and the pooled
comparison with Study F's PPO agents.

    python explore_i.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(20261005)
I = pd.read_csv(HERE / "results_i" / "runs_i.csv")
F = pd.read_csv(HERE / "results_f" / "runs_f.csv")
F = F[F.obsset == "EQ"].assign(config="PPO")
R = pd.concat([F, I], ignore_index=True)
E = ["0", "0.02", "0.05", "0.1", "0.25", "0.5", "1", "2", "3"]


def boot(x: np.ndarray) -> tuple[float, float]:
    m = x[rng.integers(0, len(x), (10_000, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def boot_diff(a, b):
    d = a[rng.integers(0, len(a), (10_000, len(a)))].mean(axis=1) - b[rng.integers(0, len(b), (10_000, len(b)))].mean(axis=1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


for cfg in ("PPO", "A2C-n256", "A2C-n64", "DQN-default"):
    g = R[R.config == cfg]
    print(f"\n=== {cfg}")
    for rule in ("ZF", "FL", "LL"):
        h = g[g.rule == rule].sort_values(["region", "seed"])
        print(f"  {rule} per run (region seed: L at e={','.join(E)}):")
        for _, r in h.iterrows():
            print(f"    {r.region} s{r.seed}: {[r[f'L{e}'] for e in E]}")
    z = g[g.rule == "ZF"]
    const = int(((z[[f"L{e}" for e in E]].nunique(axis=1)) == 1).sum())
    drop = (z["L0.1"] - z["L2"]).to_numpy()
    lo, hi = boot(drop)
    print(f"  ZF: constant policy (no equity dependence) in {const}/{len(z)} runs; L(0.1) >= L(2) in {int((drop >= 0).sum())}/{len(z)};"
          f" mean L(0.1)-L(2) = {drop.mean():.2f} [{lo:.2f}, {hi:.2f}]")
    for rule in ("FL", "LL"):
        a, b = z["L0.1"].to_numpy(), g[g.rule == rule]["L0.1"].to_numpy()
        lo, hi = boot_diff(a, b)
        print(f"  L_ZF(0.1) - L_{rule}(0.1) = {a.mean() - b.mean():.2f} [{lo:.2f}, {hi:.2f}]")
    rep, led = z.drawmean_reported_return_pct.to_numpy(), z.drawmean_ledger_return_pct.to_numpy()
    print(f"  ZF placebo: reported {rep.mean():.1f} {boot(rep)}, ledger {led.mean():.1f} {boot(led)}, gap {np.mean(rep - led):.1f}")

print("\n=== pooled over the three non-PPO configurations (30 runs per rule)")
g = R[R.config != "PPO"]
z, f, l = (g[g.rule == r]["L0.1"].to_numpy() for r in ("ZF", "FL", "LL"))
for name, b in (("FL", f), ("LL", l)):
    lo, hi = boot_diff(z, b)
    print(f"  L_ZF(0.1) - L_{name}(0.1) = {z.mean() - b.mean():.2f} [{lo:.2f}, {hi:.2f}]")
print(f"  ZF L(0.1)=16 in {int((z == 16).sum())}/{len(z)}; FL L(0.1)<=1 in {int((f <= 1).sum())}/{len(f)}")
print("  mean curves (e = 0.1, 0.5, 1, 2, 3):")
for rule in ("ZF", "FL", "LL"):
    print(f"    {rule}:", g[g.rule == rule][["L0.1", "L0.5", "L1", "L2", "L3"]].mean().round(2).tolist())

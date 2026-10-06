"""EXPLORATORY (robustness of the DP benchmark): the zero-floor optimal policy from the FULL training distribution.

dp_theory.py and dp_j.py solve the DP on 3,000 sampled quarter-hours. With heavy-tailed spreads the optimal gambling
region depends on whether the sample contains the largest spikes. Here each leverage level's one-period P&L magnitude
q * |s| (liquidity cap applied) over ALL training quarter-hours is compressed into equal-probability quantile bins
(bin means, so the mean is preserved and the extreme bins hold the spikes), and the DP is solved once,
deterministically, with dp_theory.solve on an equity grid to 6 allocations.

    python dp_full.py [--bins 2000] [--gammas 0.99]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import dp_theory
from envs_b import FEE, K0, Q1, load_market
from envs_j import load_market_j
from theory_k import dp_threshold

HERE = Path(__file__).resolve().parent
dp_theory.GRID = np.concatenate([np.linspace(0.0, 0.5, 101), np.linspace(0.51, 6.0, 550)]) * K0


def binned_pnl(zone: str, bins: int) -> np.ndarray:
    m = load_market(zone, "train", "RM") if zone in ("DK1", "DK2") else load_market_j(zone, "train", "RM")
    s, cap = np.abs(m.spread[200:]), m.cap_mwh[200:]
    rows = []
    for L in dp_theory.LEVELS:
        q = np.minimum(L * Q1, cap)
        z, f = q * s, FEE * q
        order = np.argsort(z)
        parts = np.array_split(order, bins)
        zb = np.array([z[p].mean() for p in parts])
        fb = np.array([f[p].mean() for p in parts])
        rows.append(np.concatenate([zb - fb, -zb - fb]))
    return np.array(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=2000)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.99])
    ap.add_argument("--zones", nargs="+", default=["DK1", "DK2", "FI", "NO2"])
    a = ap.parse_args()
    rows = []
    for zone in a.zones:
        X = binned_pnl(zone, a.bins)
        for g in a.gammas:
            dp_theory.GAMMA = g
            V, best, it, Q = dp_theory.solve("ZF", X, iters=20000)
            lev = dp_theory.LEVELS[best]
            gain = Q.max(axis=1) - Q[:, 0]
            idx = {e: int(np.argmin(np.abs(dp_theory.GRID - e * K0))) for e in (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0)}
            probe = {f"opt_{e:g}": float(lev[i]) for e, i in idx.items()}
            gains = {f"gain_{e:g}": float(gain[i]) for e, i in idx.items()}
            rows.append(dict(zone=zone, gamma=g, bins=a.bins, threshold_K=dp_threshold(best) / K0, **probe, **gains))
            print(rows[-1], flush=True)
    out = HERE / "results_k" / f"dp_full_b{a.bins}_g{len(a.gammas)}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

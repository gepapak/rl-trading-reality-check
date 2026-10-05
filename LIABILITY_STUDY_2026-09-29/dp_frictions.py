"""EXPLORATORY design calculation for Study H: the risk-neutral optimal policy of the fixed-sizing placebo sleeve under
realistic frictions (fee 1.0 EUR/MWh, liquidity cap 10% of balancing volume, quadratic impact kappa*q^2/cap with
kappa = 50 EUR/MWh), by the same value iteration as dp_theory.py. Writes results/dp_frictions.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import dp_theory as T
from envs_b import K0, Q1, load_market

FEE_H, CAPF_H, KAPPA_H = 1.0, 0.10, 50.0


def pnl_samples_h(region: str, rng: np.random.Generator) -> np.ndarray:
    m = load_market(region, "train", "RM")
    pick = rng.choice(np.arange(200, m.n), size=T.N_SAMPLES, replace=False)
    s, cap = np.abs(m.spread[pick]), m.cap_mwh[pick] * CAPF_H / 0.25
    out = []
    for L in T.LEVELS:
        q = np.minimum(L * Q1, cap)
        cost = FEE_H * q + KAPPA_H * q ** 2 / np.maximum(cap, 0.05)
        out.append(np.concatenate([q * s - cost, -q * s - cost]))
    return np.array(out)


def main() -> int:
    rng = np.random.default_rng(20261003)
    rows = []
    for region in ("DK1", "DK2"):
        X = pnl_samples_h(region, rng)
        for rule in ("FL", "ZF"):
            V, best, it, Q = T.solve(rule, X)
            gain = Q.max(axis=1) - Q[:, 0]
            for e in T.PROBE + (3.0,):
                i = int(np.argmin(np.abs(T.GRID - e * K0)))
                rows.append(dict(region=region, rule=rule, equity=e, opt_abs_lev=float(T.LEVELS[best[i]]), gain_over_flat_eur=float(gain[i])))
    D = pd.DataFrame(rows)
    D.to_csv("results/dp_frictions.csv", index=False)
    print(D.pivot_table(index=["region", "rule"], columns="equity", values="opt_abs_lev").to_string())
    print(D.pivot_table(index=["region", "rule"], columns="equity", values="gain_over_flat_eur").round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

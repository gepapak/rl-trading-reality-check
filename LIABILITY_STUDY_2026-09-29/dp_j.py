"""Study J design check (run before registration): the risk-neutral optimal policy of the no-edge sleeve in FI and NO2,
with dp_theory.py's model and solver unchanged (only the training-period spread and cap distributions differ).

    python dp_j.py     # writes results_j/dp_j.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import dp_theory
from envs_b import FEE, K0, Q1
from envs_j import load_market_j

HERE = Path(__file__).resolve().parent
PROBE = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0)


def pnl_samples_j(region: str, rng: np.random.Generator) -> np.ndarray:
    m = load_market_j(region, "train", "RM")
    pick = rng.choice(np.arange(200, m.n), size=dp_theory.N_SAMPLES, replace=False)
    s, cap = np.abs(m.spread[pick]), m.cap_mwh[pick]
    out = []
    for L in dp_theory.LEVELS:
        q = np.minimum(L * Q1, cap)
        out.append(np.concatenate([q * s - FEE * q, -q * s - FEE * q]))
    return np.array(out)


def main() -> int:
    rng = np.random.default_rng(20261005)
    rows = []
    for region in ("FI", "NO2"):
        X = pnl_samples_j(region, rng)
        for rule in ("FL", "LL", "ZF"):
            V, best, it, Q = dp_theory.solve(rule, X)
            gain = Q.max(axis=1) - Q[:, 0]
            for e in PROBE:
                i = int(np.argmin(np.abs(dp_theory.GRID - e * K0)))
                rows.append(dict(region=region, rule=rule, equity=e, opt_abs_lev=float(dp_theory.LEVELS[best[i]]),
                                 gain_over_flat_eur=float(gain[i]), iterations=it + 1))
    D = pd.DataFrame(rows)
    (HERE / "results_j").mkdir(exist_ok=True)
    D.to_csv(HERE / "results_j" / "dp_j.csv", index=False)
    print(D.pivot_table(index=["region", "rule"], columns="equity", values="opt_abs_lev").to_string())
    print(D.pivot_table(index=["region", "rule"], columns="equity", values="gain_over_flat_eur").round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""EXPLORATORY (not pre-registered): gambling region under the zero floor, exact DP versus the closed-form reflected-
diffusion barrier, for DK1, DK2, FI and NO2 at gamma in {0.95, 0.98, 0.99, 0.995}.

The DP uses 3,000 sampled quarter-hours of the empirical (heavy-tailed) training distribution; its threshold varies
with the sample, so it is computed for three independent samples (median and range reported). The closed form uses the
effective volatility at the discount horizon (reflection_barrier.py). Equity grid to 6 allocations.

    python theory_table.py     # writes results_k/theory_table.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import dp_theory
import reflection_barrier as rb
from dp_j import pnl_samples_j
from envs_b import K0, LEV, Q1, load_market
from envs_j import load_market_j
from theory_k import dp_threshold

HERE = Path(__file__).resolve().parent
GAMMAS = (0.95, 0.98, 0.99, 0.995)
ZONES = ("DK1", "DK2", "FI", "NO2")
SEEDS = (101, 202, 303)
dp_theory.GRID = np.concatenate([np.linspace(0.0, 0.5, 101), np.linspace(0.51, 6.0, 550)]) * K0
QMAX = float(np.abs(LEV).max() * Q1)


def main() -> int:
    rows = []
    for zone in ZONES:
        m = load_market(zone, "train", "RM") if zone in ("DK1", "DK2") else load_market_j(zone, "train", "RM")
        q = np.minimum(QMAX, m.cap_mwh[200:])
        z = q * np.abs(m.spread[200:])
        for g in GAMMAS:
            dp_theory.GAMMA = g
            rb.LAM = 1 - g
            thr = []
            for s in SEEDS:
                rng = np.random.default_rng(s)
                X = dp_theory.pnl_samples(zone, rng) if zone in ("DK1", "DK2") else pnl_samples_j(zone, rng)
                _, best, _, _ = dp_theory.solve("ZF", X, iters=20000)
                thr.append(dp_threshold(best) / K0)
            rng = np.random.default_rng(7)
            H = int(round(1 / (1 - g)))
            sims = np.array([np.sum(rng.choice([-1.0, 1.0], H) * z[i:i + H]) for i in rng.integers(0, len(z) - H, 20000)])
            sigma_h = float(np.mean(np.abs(sims)) / np.sqrt(2 / np.pi) / np.sqrt(H))
            b, _ = rb.barrier(sigma_h, float(np.mean(q)))
            rows.append(dict(zone=zone, gamma=g, dp_median=float(np.median(thr)), dp_min=min(thr), dp_max=max(thr),
                             closed_form=b / K0, length_scale=sigma_h / np.sqrt(2 * (1 - g)) / K0))
            print(rows[-1], flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(HERE / "results_k" / "theory_table.csv", index=False)
    print(D.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Study K design check (run before registration): how the optimal gambling region under the zero floor depends on the
discount factor gamma, by the exact DP (dp_theory.solve, empirical training distribution) and by the closed-form
reflected-diffusion barrier (reflection_barrier.barrier). Equity grid extended to 6 allocations.

    python theory_k.py     # writes results_k/theory_k.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import dp_theory
import reflection_barrier as rb
from envs_b import K0, LEV, Q1, load_market

HERE = Path(__file__).resolve().parent
GAMMAS = (0.95, 0.98, 0.99, 0.995)
dp_theory.GRID = np.concatenate([np.linspace(0.0, 0.5, 101), np.linspace(0.51, 6.0, 550)]) * K0
QMAX = float(np.abs(LEV).max() * Q1)


def dp_threshold(V_best: np.ndarray) -> float:
    """Largest equity below which maximal leverage is optimal at every grid point (the gambling region)."""
    lev = dp_theory.LEVELS[V_best]
    ok = lev == lev.max()
    if not ok[0]:
        return 0.0
    first_bad = np.argmin(ok) if not ok.all() else len(ok)
    return float(dp_theory.GRID[first_bad - 1])


def main() -> int:
    rows = []
    for region in ("DK1", "DK2"):
        rng = np.random.default_rng(20261005)
        X = dp_theory.pnl_samples(region, rng)
        m = load_market(region, "train", "RM")
        q = np.minimum(QMAX, m.cap_mwh[200:])
        z = q * np.abs(m.spread[200:])
        for g in GAMMAS:
            dp_theory.GAMMA = g
            rb.LAM = 1 - g
            V, best, it, Q = dp_theory.solve("ZF", X, iters=20000)
            thr = dp_threshold(best)
            H = int(round(1 / (1 - g)))
            sims = np.array([np.sum(rng.choice([-1.0, 1.0], H) * z[i:i + H]) for i in rng.integers(0, len(z) - H, 20000)])
            sigma_h = float(np.mean(np.abs(sims)) / np.sqrt(2 / np.pi) / np.sqrt(H))
            b, _ = rb.barrier(sigma_h, float(np.mean(q)))
            rows.append(dict(region=region, gamma=g, dp_threshold_K=thr / K0, closed_form_barrier_K=b / K0,
                             length_scale_K=sigma_h / np.sqrt(2 * (1 - g)) / K0, iterations=it + 1))
            print(rows[-1], flush=True)
    D = pd.DataFrame(rows)
    (HERE / "results_k").mkdir(exist_ok=True)
    D.to_csv(HERE / "results_k" / "theory_k.csv", index=False)
    print(D.round(3).to_string(index=False))
    for region in ("DK1", "DK2"):
        d = D[D.region == region].set_index("gamma")
        print(region, "DP ratio 0.995/0.95:", round(d.dp_threshold_K[0.995] / d.dp_threshold_K[0.95], 2),
              "| closed-form ratio:", round(d.closed_form_barrier_K[0.995] / d.closed_form_barrier_K[0.95], 2),
              "| 1/sqrt(1-gamma) ratio:", round(np.sqrt(0.05 / 0.005), 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

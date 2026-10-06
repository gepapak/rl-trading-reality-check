"""EXPLORATORY (not pre-registered): the zero floor reports the best moment in hindsight.

A fixed position (leverage 16, liquidity cap applied) is held from zero equity under the zero floor in the placebo
market (real spread magnitudes, independent random signs), for N sign draws of the test period. Checks:
  1. pathwise: reported equity W_T = L_T - min(0, min_s L_s) (Proposition 6, drawup of the ledger);
  2. in distribution: W_T versus the running maximum M_T = max(0, max_k L_k) of the same ledger
     (Lindley-Loynes duality; exact for exchangeable increments);
  3. diffusion limit: M_T ~ |B_T| (Levy), so CV(W_T) -> sqrt(pi/2 - 1) = 0.756 and E[W_T] -> sigma sqrt(2T/pi);
  4. detection: the share of n-draw placebo tests (one-sided t-test, 5%) that flag the floor, n = 2..10.

    python best_moment.py
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from envs_b import FEE, K0, Q1, load_market
from envs_j import load_market_j

N = 1000
rng = np.random.default_rng(20261005)


def main() -> int:
    print(f"universal CV sqrt(pi/2 - 1) = {np.sqrt(np.pi / 2 - 1):.3f}")
    for region in ("DK1", "DK2", "FI", "NO2"):
        lm = load_market if region in ("DK1", "DK2") else load_market_j
        m = lm(region, "test", "RM")
        q = np.minimum(16 * Q1, m.cap_mwh[200:])
        mag = q * np.abs(m.spread[200:])
        fee = FEE * q
        T = len(mag)
        W = np.zeros(N)            # reported equity (Lindley)
        L = np.zeros(N)            # ledger
        Lmin = np.zeros(N)
        Lmax = np.zeros(N)
        for t in range(T):
            x = rng.choice([-1.0, 1.0], N) * mag[t] - fee[t]
            W = np.maximum(W + x, 0.0)
            L = L + x
            Lmin = np.minimum(Lmin, L)
            Lmax = np.maximum(Lmax, L)
        drawup = L - Lmin
        assert np.allclose(W, drawup), "pathwise identity failed"
        ks = stats.ks_2samp(W, Lmax)
        sigma = float(np.sqrt(np.mean(mag ** 2)))
        pred = sigma * np.sqrt(2 * T / np.pi)
        cv = float(W.std() / W.mean())
        # detection power of an n-draw placebo test on the reported return (one-sided t-test of mean > 0 at 5%)
        R = W / K0
        power = {}
        for n in (2, 3, 4, 5, 10):
            idx = rng.integers(0, N, (5000, n))
            S = R[idx]
            t = S.mean(1) / (S.std(1, ddof=1) / np.sqrt(n))
            power[n] = float(np.mean(t > stats.t.ppf(0.95, n - 1)))
        print(f"{region}: pathwise identity OK | KS(W, running max) D={ks.statistic:.3f} p={ks.pvalue:.2f} | "
              f"mean reported {W.mean() / K0:6.2f}K, predicted sigma*sqrt(2T/pi) {pred / K0:6.2f}K | CV {cv:.3f} | "
              f"booked mean {L.mean() / K0:+.2f}K | power by n draws {power}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

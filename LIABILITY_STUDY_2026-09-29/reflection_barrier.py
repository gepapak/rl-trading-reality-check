"""EXPLORATORY (not pre-registered): optimal gambling barrier of the zero floor in the diffusion limit, against the DP.

Model. Under the zero floor, reported equity is the ledger reflected at zero (Lindley / Skorokhod). In the diffusion
limit, gambling at leverage l gives dE = mu dt + sigma dB + dL with sigma = q_l * sd(spread) per step, mu = -fee * q_l,
and L the reflection regulator (the forgiven loss); staying flat freezes equity. The agent maximizes
E int e^{-lam t} dE_t. Because sigma and mu are both linear in q while the value is convex, the Hamiltonian is convex in
q and the optimal control is bang-bang: maximal leverage below a barrier b, flat above. On (0, b):

    (sigma^2/2) V'' + mu V' - lam V + mu = 0,   V'(0) = -1 (reflection),   V(b) = 0,   V'(b) = 0 (smooth fit)

V = mu/lam + A e^{r1 e} + B e^{r2 e}; eliminating A gives B = -1 / (r2 (1 - e^{(r2-r1) b})) and the barrier equation
B e^{r2 b} (1 - r2/r1) = -mu/lam, solved here numerically. The DP benchmark is dp_theory.solve on the empirical
(discrete-time, heavy-tailed) distribution.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

import dp_theory
from dp_j import pnl_samples_j
from envs_b import FEE, K0, LEV, Q1, load_market
from envs_j import load_market_j

LAM = 1 - dp_theory.GAMMA
QMAX = float(np.abs(LEV).max() * Q1)
PROBE = (0.02, 0.1, 0.25, 0.5, 1.0, 2.0)


def market(region):
    return load_market(region, "train", "RM") if region in ("DK1", "DK2") else load_market_j(region, "train", "RM")


def barrier(sigma: float, q: float) -> tuple[float, callable]:
    mu = -FEE * q
    disc = np.sqrt(mu ** 2 + 2 * LAM * sigma ** 2)
    r1, r2 = (-mu + disc) / sigma ** 2, (-mu - disc) / sigma ** 2

    def coeffs(b):
        B = -1.0 / (r2 * (1 - np.exp((r2 - r1) * b)))
        A = -B * r2 * np.exp((r2 - r1) * b) / r1
        return A, B

    def f(b):
        A, B = coeffs(b)
        return mu / LAM + A * np.exp(r1 * b) + B * np.exp(r2 * b)
    hi = 1e-6 * K0
    while np.isfinite(f(hi * 2)) and f(hi * 2) > 0 and hi < 50 * K0:
        hi *= 2
    b = brentq(f, 1e-6 * K0, hi * 2) if np.isfinite(f(hi * 2)) else brentq(f, 1e-6 * K0, hi)
    A, B = coeffs(b)
    V = lambda e: np.where(e < b, mu / LAM + A * np.exp(r1 * e) + B * np.exp(r2 * e), 0.0)  # noqa: E731
    return b, V


def main() -> int:
    rng = np.random.default_rng(20261005)
    print(f"{'zone':5s} {'sd(q s)':>8s} {'b* diffusion':>13s} {'DP threshold':>14s} | value V(e) diffusion vs DP at e = {PROBE}")
    for region in ("DK1", "DK2", "FI", "NO2"):
        m = market(region)
        q = np.minimum(QMAX, m.cap_mwh[200:])
        z = q * np.abs(m.spread[200:])
        sigma = float(np.sqrt(np.mean(z ** 2)))          # per-step sd of the sign-randomized P&L
        # effective volatility at the discount horizon H = 1/lam: sd implied by E|S_H| of sign-randomized H-step sums
        H = int(round(1 / LAM))
        sims = np.array([np.sum(rng.choice([-1.0, 1.0], H) * z[i:i + H]) for i in rng.integers(0, len(z) - H, 20000)])
        sigma_h = float(np.mean(np.abs(sims)) / np.sqrt(2 / np.pi) / np.sqrt(H))
        b0, _ = barrier(sigma, float(np.mean(q)))
        b, V = barrier(sigma_h, float(np.mean(q)))
        X = dp_theory.pnl_samples(region, rng) if region in ("DK1", "DK2") else pnl_samples_j(region, rng)
        Vdp, best, _, _ = dp_theory.solve("ZF", X)
        lev = dp_theory.LEVELS[best]
        gamble = dp_theory.GRID[lev == lev.max()]
        thr = float(gamble.max()) if len(gamble) else 0.0
        idx = [int(np.argmin(np.abs(dp_theory.GRID - e * K0))) for e in PROBE]
        vd = [round(float(V(np.array(e * K0))), 0) for e in PROBE]
        vp = [round(float(Vdp[i]), 0) for i in idx]
        print(f"{region:5s} {sigma:8.0f} {sigma_h:8.0f} b*(sd) {b0 / K0:6.3f} b*(horizon sd) {b / K0:6.3f} DP {thr / K0:6.3f} | diffusion {vd} | DP {vp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""EXPLORATORY (not pre-registered): the risk-neutral optimal leverage policy of the Study B no-edge (NE) sleeve under
each liability rule, by value iteration over equity, for comparison with the learned probe curves.

Model = the Study B environment with the state reduced to equity (in the NE market no feature predicts the payoff):
    per quarter: q = min(|L| * Q1, cap_t), pnl = eps_t * q * |s_t| - FEE * q, eps_t = +-1 with probability 1/2
    (|s_t|, cap_t) drawn from their empirical joint distribution on the TRAINING period
    FL: e' = e + pnl, reward pnl, terminal if e' <= 0
    LL: e' = max(e + pnl, 0), reward e' - e, terminal if e + pnl <= 0
    ZF: e' = max(e + pnl, 0), reward e' - e, never terminal
    discount 0.99 per quarter (the learners' discount); direction of L is irrelevant (symmetric payoff)

    python dp_theory.py        # writes results/dp_theory.csv and prints the optimal |L| at the probe equity levels
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from envs_b import FEE, K0, LEV, Q1, load_market

GAMMA = 0.99
GRID = np.concatenate([np.linspace(0.0, 0.5, 101), np.linspace(0.51, 3.0, 250)]) * K0  # equity grid (EUR)
LEVELS = np.unique(np.abs(LEV))                                                          # 0, 0.25, 1, 4, 16
PROBE = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
N_SAMPLES = 3000
TOL = 1e-5


def pnl_samples(region: str, rng: np.random.Generator) -> np.ndarray:
    """Matrix (levels x 2*N) of equally likely one-quarter P&L outcomes for each |L| (both signs of eps)."""
    m = load_market(region, "train", "RM")
    idx = np.arange(200, m.n)
    pick = rng.choice(idx, size=N_SAMPLES, replace=False)
    s, cap = np.abs(m.spread[pick]), m.cap_mwh[pick]
    out = []
    for L in LEVELS:
        q = np.minimum(L * Q1, cap)
        out.append(np.concatenate([q * s - FEE * q, -q * s - FEE * q]))
    return np.array(out)


def solve(rule: str, X: np.ndarray, iters: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Value iteration on GRID; returns V and the optimal |L| index per grid point (ties -> the smaller leverage)."""
    V = np.zeros_like(GRID)  # V = expected discounted future reward (not equity); flat forever gives 0
    E = GRID[:, None, None]
    nxt = E + X[None, :, :]                                   # (grid, levels, samples)
    if rule == "FL":
        reward, e2, alive = nxt - E, nxt, nxt > 0
    else:
        floored = np.maximum(nxt, 0.0)
        reward, e2 = floored - E, floored
        alive = (nxt > 0) if rule == "LL" else np.ones_like(nxt, dtype=bool)
    r_mean = reward.mean(axis=2)
    # interpolation weights are fixed (e2 does not depend on V): precompute once, gather each iteration
    top = GRID[-1]
    ec = np.clip(e2, GRID[0], top)
    i1 = np.clip(np.searchsorted(GRID, ec, side="right"), 1, len(GRID) - 1)
    i0 = i1 - 1
    w1 = (ec - GRID[i0]) / (GRID[i1] - GRID[i0])
    live = alive.astype(float)                            # above the grid V is held constant (flat extension)
    del nxt, reward, ec
    tie = 1e-9 * K0 * (-np.arange(len(LEVELS)))[None, :]  # ties -> the lower leverage
    rows = np.arange(len(GRID))

    def q_all(V):
        return r_mean + GAMMA * ((V[i0] * (1 - w1) + V[i1] * w1) * live).mean(axis=2)

    # modified policy iteration: greedy step over all actions, then SWEEPS evaluation sweeps of the greedy policy
    SWEEPS = 40
    for it in range(iters):
        Q = q_all(V)
        best = np.argmax(Q + tie, axis=1)
        V_new = Q[rows, best]
        a0, a1, aw, al, ar = i0[rows, best], i1[rows, best], w1[rows, best], live[rows, best], r_mean[rows, best]
        for _ in range(SWEEPS):
            V_new = ar + GAMMA * ((V_new[a0] * (1 - aw) + V_new[a1] * aw) * al).mean(axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        if it % 10 == 0:
            print(f"    {rule} iter {it} max change {delta:.4f} EUR", flush=True)
        if delta < TOL * K0:
            break
    Q = q_all(V)
    best = np.argmax(Q + tie, axis=1)
    return V, best, it, Q


def main() -> int:
    rng = np.random.default_rng(20260930)
    rows = []
    for region in ("DK1", "DK2"):
        X = pnl_samples(region, rng)
        for rule in ("FL", "LL", "ZF"):
            V, best, it, Q = solve(rule, X)
            gain = Q.max(axis=1) - Q[:, 0]    # value of the best action over staying flat (EUR)
            for e in PROBE:
                i = int(np.argmin(np.abs(GRID - e * K0)))
                rows.append(dict(region=region, rule=rule, equity=e, opt_abs_lev=float(LEVELS[best[i]]),
                                 gain_over_flat_eur=float(gain[i]), iterations=it + 1))
            print(f"{region} {rule}: converged after {it + 1} iterations", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv("results/dp_theory.csv", index=False)
    print(D.pivot_table(index=["region", "rule"], columns="equity", values="opt_abs_lev").to_string())
    print(D.pivot_table(index=["region", "rule"], columns="equity", values="gain_over_flat_eur").round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

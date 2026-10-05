"""Placebo-market test for trading agents (the check proposed in the paper, Section "Implications for engineering practice").

Idea: rebuild the market with the SIGN of the traded payoff randomized independently in every period, keeping its
magnitudes (heavy tails, volatility clustering in |payoff|) and every other input. No policy can profit there in
expectation, so a positive expected REPORTED return of an agent trained and evaluated in the placebo market is a false
positive of the simulator (e.g. loss forgiveness) or of the evaluation, by construction.

Usage in your own pipeline:
    draws = placebo_payoffs(payoff, n_draws=20, seed=0)        # list of arrays, same shape as payoff
    for p in draws: rebuild lagged features from p, train/evaluate the agent, record reported and booked returns
    verdict = placebo_verdict(reported_returns)                 # mean and bootstrap interval over draws or seeds

Example with this study's environment (envs_b.NoEdgeMarket already implements the placebo market):
    python placebo_market.py --demo
"""
from __future__ import annotations

import argparse

import numpy as np


def placebo_payoffs(payoff: np.ndarray, n_draws: int = 20, seed: int = 0) -> list[np.ndarray]:
    """Independent random-sign copies of a payoff series (E[sign * payoff | past] = 0 for every policy)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(payoff, dtype=float)
    return [x * rng.choice([-1.0, 1.0], size=x.shape) for _ in range(n_draws)]


def placebo_verdict(reported_returns: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Mean reported return with a percentile bootstrap interval; FLAG if the lower bound is above zero."""
    r = np.asarray(reported_returns, dtype=float)
    rng = np.random.default_rng(seed)
    means = r[rng.integers(0, len(r), (n_boot, len(r)))].mean(axis=1)
    lo, hi = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    return {"mean": float(r.mean()), "ci95": (lo, hi), "false_positive_flag": lo > 0}


def _demo() -> None:
    """A constant maximum-leverage long policy in the placebo DK1 market under full liability and under a zero floor."""
    from envs_b import LEV, LiabilityEnv, load_market
    idx = int(np.argmax(LEV))
    for rule in ("FL", "ZF"):
        rep = []
        for k in range(20):
            env = LiabilityEnv(load_market("DK1", "test", "NE", draw=k), rule, train=False)
            env.reset()
            done = False
            while not done:
                _, _, done, _, _ = env.step(idx)
            rep.append(100 * (env.reported / env.e0 - 1))
        print(rule, placebo_verdict(np.array(rep)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    if ap.parse_args().demo:
        _demo()

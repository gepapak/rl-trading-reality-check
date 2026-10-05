"""Study B environment (PREREGISTRATION_B.md): a leverage-choosing imbalance-spread sleeve under three liability rules.

The agent chooses a signed leverage level each quarter-hour; the position is sized from a fixed allocation K0 (not from
current equity), as in the audited engine and in any account whose trading limit does not shrink with its losses.

Liability rules (identical market, sizing, cap, fee and observation; only the accounting of losses beyond equity differs)
    FL  full liability: every loss is booked; the sleeve closes when equity reaches zero (the overshoot is booked too)
    LL  limited liability: the loss that would take equity below zero is capped at the remaining equity; the sleeve closes
    ZF  zero floor: reported equity is floored at zero every quarter and trading continues (the audited engine's rule)
The ledger always books every loss in full; "reported" is what the environment reports and rewards.

Markets
    RM  real Danish imbalance spreads (imbalance price minus day-ahead price)
    NE  no edge: the same spreads with an independent random sign each quarter, eps_t in {-1, +1}; lagged features use the
        sign-flipped series, so no policy can predict the payoff (E[s'_t | past] = 0 exactly)

Data and causal features are those of Study A (GENERALIZATION_STUDY_2026-09-29/envs.py, data_panel/).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

STUDY_A = Path(__file__).resolve().parents[1] / "GENERALIZATION_STUDY_2026-09-29"
sys.path.insert(0, str(STUDY_A))
from envs import EPISODE, PSCALE, Market, clipobs, lag  # noqa: E402

LEV = np.array([-16.0, -4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0, 16.0])
K0 = 20_000.0          # EUR allocation of the sleeve
Q1 = 0.25              # MWh per quarter-hour at leverage 1 (maximum position 4 MWh)
FEE = 0.10             # EUR per MWh traded (as in Study A)
E0_TRAIN = (0.05, 2.0) # training episodes start at an equity drawn log-uniformly in this range (multiples of K0)
LIABILITY = ("FL", "LL", "ZF")
MARKETS = ("RM", "NE")
NE_SEED = {("DK1", "train"): 9101, ("DK1", "test"): 9102, ("DK2", "train"): 9201, ("DK2", "test"): 9202}


class NoEdgeMarket(Market):
    """Real spreads with an independent random sign per quarter; lag features rebuilt from the sign-flipped series."""

    def __init__(self, panel: pd.DataFrame, split: str, seed: int):
        super().__init__(panel, split)
        eps = np.random.default_rng(seed).choice([-1.0, 1.0], size=self.n)
        self.spread = self.spread * eps
        spr_l = np.c_[[lag(self.spread, k) for k in range(2, 10)]].T
        self.spr_lags = np.c_[spr_l[:, [0, 1, 2, 6]], spr_l.mean(axis=1)] / PSCALE


N_DRAWS = 20          # independent sign draws of the NE test period (draw 0 is the main one)


@lru_cache(maxsize=None)
def _panel(region: str) -> pd.DataFrame:
    return pd.read_pickle(STUDY_A / "data_panel" / f"panel_{region}.pkl")


def load_market(region: str, split: str, market: str, draw: int = 0) -> Market:
    panel = _panel(region)
    return Market(panel, split) if market == "RM" else NoEdgeMarket(panel, split, NE_SEED[(region, split)] + 1000 * draw)


def obs_matrix(m: Market, ts: np.ndarray, equity_multiple: float) -> np.ndarray:
    """Observations at quarters ts with the equity slot set to a given multiple of K0 (policy probe)."""
    n = len(ts)
    X = np.c_[m.spot[ts] / PSCALE, m.spr_lags[ts], np.full(n, np.clip(equity_multiple, 0, 5)), m.time[ts]]
    return np.clip(X, -10.0, 10.0).astype(np.float32)


class LiabilityEnv(gym.Env):
    metadata: dict = {}

    def __init__(self, market: Market, liability: str, train: bool, seed: int = 0):
        super().__init__()
        assert liability in LIABILITY
        self.m, self.liab, self.train = market, liability, train
        self.action_space = gym.spaces.Discrete(len(LEV))
        self.observation_space = gym.spaces.Box(-10, 10, shape=(11,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.train:
            self.t0 = int(self.rng.integers(200, self.m.n - EPISODE))
            self.t_end = self.t0 + EPISODE
            e0 = K0 * float(np.exp(self.rng.uniform(np.log(E0_TRAIN[0]), np.log(E0_TRAIN[1]))))
        else:
            self.t0, self.t_end = 200, self.m.n
            e0 = K0 * float((options or {}).get("e0", 1.0))
        self.t = self.t0
        self.e0 = e0
        self.ledger = e0      # booked equity, every loss included
        self.reported = e0    # equity the environment reports and rewards
        self.closed = False
        self.floor_bound = False
        self.max_gap_pp = 0.0
        self.lev_abs_sum = 0.0
        self.open_q = 0
        self.max_lev_q = 0
        return self._obs(), {}

    def _obs(self):
        t, m = self.t, self.m
        return clipobs(np.array([m.spot[t] / PSCALE, *m.spr_lags[t], np.clip(self.reported / K0, 0, 5), *m.time[t]]))

    def step(self, action):
        t, m = self.t, self.m
        lev = 0.0 if self.closed else float(LEV[int(action)])
        if not self.closed:
            self.open_q += 1
            self.lev_abs_sum += abs(lev)
            self.max_lev_q += int(abs(lev) == LEV.max())
        q = float(np.clip(lev * Q1, -m.cap_mwh[t], m.cap_mwh[t]))
        pnl = q * m.spread[t] - FEE * abs(q)
        self.ledger += pnl
        before = self.reported
        if self.liab == "FL":
            self.reported = self.ledger
            if self.reported <= 0.0:
                self.closed = True
        else:
            new = self.reported + pnl
            if new < 0.0:
                self.floor_bound = True
            self.reported = max(new, 0.0)
            if self.liab == "LL" and new <= 0.0:
                self.closed = True
        reward = (self.reported - before) / K0 * 100.0
        self.max_gap_pp = max(self.max_gap_pp, 100.0 * abs(self.reported - self.ledger) / K0)
        self.t += 1
        done = self.t >= self.t_end or (self.train and self.closed)
        return (self._obs() if not done else np.zeros(11, np.float32)), float(reward), done, False, {"pnl": pnl}


def outcome(env: LiabilityEnv) -> dict:
    return {"reported_return_pct": 100 * (env.reported / env.e0 - 1), "ledger_return_pct": 100 * (env.ledger / env.e0 - 1),
            "closed": bool(env.closed), "floor_bound": bool(env.floor_bound), "max_gap_pp": env.max_gap_pp,
            "mean_abs_lev_open": env.lev_abs_sum / max(env.open_q, 1), "share_max_lev_open": env.max_lev_q / max(env.open_q, 1),
            "open_quarters": env.open_q}

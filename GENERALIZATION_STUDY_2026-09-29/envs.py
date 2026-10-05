"""Study A environments (PREREGISTRATION_A.md). Own implementations of three common environment designs.

E1 battery arbitrage (asset-backed), E2 capital-account spread sleeve (finance-style), E3 wind bidding (asset-backed).
Protocol flags (S0 = all False except the defaults below):
    price_taker   no liquidity cap (S1)
    no_solvency   sizing from K0, no ruin stop, cash floored at zero (S2; E2 only)
    lookahead     current-quarter imbalance price (E1, E2) / actual production + imbalance price (E3) in the observation (S4)
    pct_payoff    percent-of-price payoff (S5; E2 only)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import pandas as pd

LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
SHADES = np.array([-0.3, -0.15, 0.0, 0.15, 0.3])
EPISODE = 7 * 96
PSCALE = 100.0  # EUR/MWh feature scale
CAP_FRAC = 0.25
DT = 0.25  # hours per quarter

REGIMES = {
    "S0": {},
    "S1": {"price_taker": True},
    "S2": {"no_solvency": True},
    "S3": {"price_taker": True, "no_solvency": True},
    "S4": {"lookahead": True},
    "S5": {"pct_payoff": True},
}


@dataclass
class Flags:
    price_taker: bool = False
    no_solvency: bool = False
    lookahead: bool = False
    pct_payoff: bool = False
    lookahead_proxy: bool = False  # reality evaluation of an S4-trained agent: leaked slots hold the causal proxy

    @property
    def extra_slots(self) -> bool:
        return self.lookahead or self.lookahead_proxy

    @classmethod
    def of(cls, regime: str) -> "Flags":
        return cls(**REGIMES[regime])


def time_features(idx: pd.DatetimeIndex) -> np.ndarray:
    h = (idx.hour + idx.minute / 60.0).values
    d = idx.dayofweek.values
    return np.c_[np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)]


def lag(x: np.ndarray, k: int) -> np.ndarray:
    out = np.empty_like(x)
    out[:k] = x[0]
    out[k:] = x[:-k]
    return out


class Market:
    """Arrays for one region and one split, plus causal lag features."""

    def __init__(self, panel: pd.DataFrame, split: str):
        p = panel[panel.split == split]
        self.idx = p.index
        self.imb, self.spot = p.imb.values.astype(float), p.spot.values.astype(float)
        self.bal = p.bal_mw.values.astype(float)
        self.wind_act, self.wind_fc = p.wind_act.values.astype(float), p.wind_fc.values.astype(float)
        self.spread = self.imb - self.spot
        self.n = len(p)
        self.time = time_features(p.index)
        imb_l = np.c_[[lag(self.imb, k) for k in range(2, 10)]].T              # t-2 .. t-9
        spr_l = np.c_[[lag(self.spread, k) for k in range(2, 10)]].T
        self.imb_lags = np.c_[imb_l[:, [0, 1, 2, 6]], imb_l.mean(axis=1)] / PSCALE
        self.spr_lags = np.c_[spr_l[:, [0, 1, 2, 6]], spr_l.mean(axis=1)] / PSCALE
        d1, d2 = lag(self.spread, 96), lag(self.spread, 192)                    # >= 1 day old (E3)
        day_mean = pd.Series(self.spread).rolling(96, min_periods=1).mean().values
        self.e3_lags = np.c_[d1, d2, lag(day_mean, 96)] / PSCALE
        self.cap_mwh = CAP_FRAC * self.bal * DT                                 # executable energy per quarter


def clipobs(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -10.0, 10.0).astype(np.float32)


class BaseEnv(gym.Env):
    metadata: dict = {}

    def __init__(self, market: Market, flags: Flags, train: bool, seed: int = 0):
        super().__init__()
        self.m, self.f, self.train = market, flags, train
        self.action_space = gym.spaces.Discrete(5)
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-10, 10, shape=(self.obs_dim(),), dtype=np.float32)

    def obs_dim(self) -> int:
        raise NotImplementedError

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.train:
            self.t0 = int(self.rng.integers(200, self.m.n - EPISODE))
            self.t_end = self.t0 + EPISODE
        else:
            self.t0, self.t_end = 200, self.m.n  # evaluation: whole split (first 200 quarters are lag warm-up)
        self.t = self.t0
        self._reset_state()
        return self._obs(), {}

    def _reset_state(self):
        pass


class BatteryEnv(BaseEnv):
    """E1: power level x P MW; settles charged/discharged energy at the imbalance price."""

    def __init__(self, market, flags, train, p_mw: float, e_mwh: float, seed: int = 0):
        self.p, self.e = p_mw, e_mwh
        super().__init__(market, flags, train, seed)

    def obs_dim(self):
        return 1 + 5 + 1 + 4 + (1 if self.f.extra_slots else 0)

    def _reset_state(self):
        self.soc = 0.5 * self.e
        self.profit = 0.0

    def _obs(self):
        t, m = self.t, self.m
        o = [m.spot[t] / PSCALE, *m.imb_lags[t], self.soc / self.e, *m.time[t]]
        if self.f.lookahead:
            o.append(m.imb[t] / PSCALE)
        elif self.f.lookahead_proxy:
            o.append(m.imb[t - 2] / PSCALE)
        return clipobs(np.array(o))

    def step(self, action):
        t, m = self.t, self.m
        want = LEVELS[int(action)] * self.p * DT                      # + discharge / - charge, MWh at the grid side
        if want > 0:
            e = min(want, self.soc * 0.9)                              # 90% efficiency on discharge
            if not self.f.price_taker:
                e = min(e, m.cap_mwh[t])
            self.soc -= e / 0.9
            pnl = e * m.imb[t]
        elif want < 0:
            e = min(-want, self.e - self.soc)
            if not self.f.price_taker:
                e = min(e, m.cap_mwh[t])
            self.soc += e
            pnl = -e * m.imb[t]
        else:
            pnl = 0.0
        self.profit += pnl
        self.t += 1
        done = self.t >= self.t_end
        return (self._obs() if not done else np.zeros(self.observation_space.shape, np.float32),
                float(pnl / (self.p * PSCALE)), done, False, {"pnl": pnl})


class SleeveEnv(BaseEnv):
    """E2: capital-account sleeve trading q = a*Q MWh of (imbalance - spot) per quarter."""
    K0 = 1_000_000.0
    Q_PER_M = 20.0
    FEE = 0.10

    def obs_dim(self):
        return 1 + 5 + 1 + 4 + (1 if self.f.extra_slots else 0)

    def _reset_state(self):
        self.ledger = self.K0        # booked equity, never floored
        self.reported = self.K0      # equity the environment reports (floored under no_solvency)
        self.ruined = False
        self.max_gap_pp = 0.0
        self.floor_bound = False

    def _obs(self):
        t, m = self.t, self.m
        o = [m.spot[t] / PSCALE, *m.spr_lags[t], np.clip(self.reported / self.K0, 0, 5), *m.time[t]]
        if self.f.lookahead:
            o.append(m.spread[t] / PSCALE)
        elif self.f.lookahead_proxy:
            o.append(m.spread[t - 2] / PSCALE)
        return clipobs(np.array(o))

    def step(self, action):
        t, m, a = self.t, self.m, LEVELS[int(action)]
        sizing = self.K0 if self.f.no_solvency else max(self.ledger, 0.0)
        if self.ruined:
            a, sizing = 0.0, 0.0
        q_des = a * self.Q_PER_M * sizing / 1e6
        q = q_des if self.f.price_taker else float(np.clip(q_des, -m.cap_mwh[t], m.cap_mwh[t]))
        fill = (q / q_des) if q_des != 0 else 0.0
        if self.f.pct_payoff:
            spot = m.spot[t] if abs(m.spot[t]) >= 1.0 else 1.0
            pnl = a * fill * sizing * float(np.clip(m.imb[t] / spot - 1.0, -0.5, 0.5)) - self.FEE * abs(q)
        else:
            pnl = q * m.spread[t] - self.FEE * abs(q)
        self.ledger += pnl
        if self.f.no_solvency:
            self.reported = max(self.reported + pnl, 0.0)
            if self.reported <= 0.0 and self.ledger < 0.0:
                self.floor_bound = True
        else:
            self.reported = self.ledger
            if self.ledger <= 0.05 * self.K0:
                self.ruined = True
        self.max_gap_pp = max(self.max_gap_pp, 100.0 * abs(self.reported - self.ledger) / self.K0)
        self.t += 1
        done = self.t >= self.t_end or (self.train and self.ruined)
        return (self._obs() if not done else np.zeros(self.observation_space.shape, np.float32),
                float(pnl / self.K0 * 1000.0), done, False, {"pnl": pnl})


class WindEnv(BaseEnv):
    """E3: schedule = clip(forecast*(1+s), 0, 100); revenue = schedule*spot + (actual - schedule)*imbalance."""
    CAP = 100.0

    def obs_dim(self):
        return 2 + 3 + 4 + (2 if self.f.extra_slots else 0)

    def _reset_state(self):
        self.revenue = 0.0
        self.revenue_fc = 0.0

    def _obs(self):
        t, m = self.t, self.m
        o = [m.wind_fc[t] / PSCALE, m.spot[t] / PSCALE, *m.e3_lags[t], *m.time[t]]
        if self.f.lookahead:
            o += [m.wind_act[t] / PSCALE, m.imb[t] / PSCALE]
        elif self.f.lookahead_proxy:
            o += [m.wind_fc[t] / PSCALE, m.imb[t - 96] / PSCALE]
        return clipobs(np.array(o))

    def step(self, action):
        t, m = self.t, self.m
        sched = float(np.clip(m.wind_fc[t] * (1 + SHADES[int(action)]), 0, self.CAP)) * DT
        sched0 = float(np.clip(m.wind_fc[t], 0, self.CAP)) * DT
        act = m.wind_act[t] * DT
        rev = sched * m.spot[t] + (act - sched) * m.imb[t]
        rev0 = sched0 * m.spot[t] + (act - sched0) * m.imb[t]
        self.revenue += rev
        self.revenue_fc += rev0
        self.t += 1
        done = self.t >= self.t_end
        return (self._obs() if not done else np.zeros(self.observation_space.shape, np.float32),
                float((rev - rev0) / PSCALE), done, False, {"pnl": rev})


ENV_CONFIGS = {
    "E1-small": dict(cls=BatteryEnv, kw=dict(p_mw=1.0, e_mwh=2.0), regimes=["S0", "S1", "S4"]),
    "E1-large": dict(cls=BatteryEnv, kw=dict(p_mw=50.0, e_mwh=100.0), regimes=["S0", "S1", "S4"]),
    "E2": dict(cls=SleeveEnv, kw={}, regimes=["S0", "S1", "S2", "S3", "S4", "S5"]),
    "E3": dict(cls=WindEnv, kw={}, regimes=["S0", "S4"]),
}


def make_env(env_name: str, regime: str, market: Market, train: bool, seed: int = 0, proxy: bool = False):
    """proxy=True: S0 dynamics with the look-ahead slots filled by causal proxies (reality check of S4-trained agents)."""
    c = ENV_CONFIGS[env_name]
    flags = Flags.of(regime)
    if proxy:
        flags.lookahead_proxy = True
    return c["cls"](market, flags, train, seed=seed, **c["kw"])


def outcome(env) -> dict:
    """Headline outcome after an evaluation pass."""
    if isinstance(env, BatteryEnv):
        return {"profit_eur": env.profit}
    if isinstance(env, SleeveEnv):
        return {"reported_return_pct": 100 * (env.reported / env.K0 - 1), "ledger_return_pct": 100 * (env.ledger / env.K0 - 1),
                "ruined": bool(env.ruined), "floor_bound": bool(env.floor_bound), "max_gap_pp": env.max_gap_pp}
    return {"revenue_eur": env.revenue, "shading_value_eur": env.revenue - env.revenue_fc}

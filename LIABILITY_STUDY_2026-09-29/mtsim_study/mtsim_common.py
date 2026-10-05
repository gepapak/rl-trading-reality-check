"""Study G common code: gym-mtsim (v2.0.0, unchanged) on synthetic placebo forex prices.

- Synthetic market: one symbol with EURUSD's contract specification (from the package's bundled symbol info), hourly
  bars, log-price a driftless random walk with Student-t(3) returns scaled to 0.15% per hour; no policy can earn a
  positive expected P&L beyond chance (placebo market).
- Two accounting conditions:
    ORIG    gym-mtsim as published: after a stop-out the balance is floored at zero (loss beyond the balance forgiven)
    FLPATCH identical, except that the floor is removed (the full loss is booked; negative balance stays)
  FLPATCH subclasses MtSimulator and overrides tick() with the published code minus the two floor lines.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import gymnasium as gym

from gym_mtsim import FOREX_DATA_PATH, MtEnv, MtSimulator

SYMBOL = "EURUSD"
HOURS = 30_000
VOL = 0.0015


class MtSimulatorFL(MtSimulator):
    """gym-mtsim's tick() without the zero floor on the balance (full liability)."""

    def tick(self, delta_time: timedelta = timedelta()) -> None:
        self._check_current_time()
        self.current_time += delta_time
        self.equity = self.balance
        for order in self.orders:
            order.exit_time = self.current_time
            order.exit_price = self.price_at(order.symbol, order.exit_time)["Close"]
            self._update_order_profit(order)
            self.equity += order.profit
        while self.margin_level < self.stop_out_level and len(self.orders) > 0:
            most_unprofitable_order = min(self.orders, key=lambda order: order.profit)
            self.close_order(most_unprofitable_order)
        # published code here: if self.balance < 0.: self.balance = 0.; self.equity = self.balance  (removed)


def placebo_prices(seed: int, hours: int = HOURS, start_price: float = 1.10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.standard_t(3, size=hours) / np.sqrt(3.0) * VOL          # unit-variance t(3) scaled to VOL per hour
    close = start_price * np.exp(np.cumsum(r))
    open_ = np.r_[start_price, close[:-1]]
    idx = pd.date_range(datetime(2030, 1, 1), periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({"Open": open_, "Close": close, "Low": np.minimum(open_, close), "High": np.maximum(open_, close),
                         "Volume": np.ones(hours)}, index=idx)


def make_simulator(condition: str, prices: pd.DataFrame, balance: float = 10_000.0) -> MtSimulator:
    base = MtSimulator(symbols_filename=FOREX_DATA_PATH, hedge=True)
    cls = MtSimulator if condition == "ORIG" else MtSimulatorFL
    sim = cls(unit="USD", balance=balance, leverage=100.0, stop_out_level=0.2, hedge=True)
    sim.symbols_info = {SYMBOL: copy.deepcopy(base.symbols_info[SYMBOL])}
    sim.symbols_data = {SYMBOL: prices}
    return sim


class EquityObs(gym.Wrapper):
    """Flat observation for a standard MLP policy: [equity/10k, balance/10k, margin/10k, open volume, open profit/10k]
    (price features are uninformative in the placebo market). Episodes start at a random window and a random initial
    balance (log-uniform 0.05-2 x 10k) so that low-equity states are visited; the accounting is gym-mtsim's own.
    Also tracks the booked ledger (sum of equity changes without the floor) for reconciliation."""

    def __init__(self, env: MtEnv, train: bool, episode: int = 500, seed: int = 0):
        super().__init__(env)
        self.train, self.episode = train, episode
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-10, 10, shape=(5,), dtype=np.float32)

    def _flat(self, o) -> np.ndarray:
        return np.clip(np.array([o["equity"][0] / 1e4, o["balance"][0] / 1e4, o["margin"][0] / 1e4,
                                 o["orders"][0, 0, 1], o["orders"][0, 0, 2] / 1e4], dtype=np.float32), -10, 10)

    def reset(self, *, seed=None, options=None):
        e = self.env.unwrapped
        n = len(e.time_points)
        if self.train:
            start = int(self.rng.integers(e.window_size, n - self.episode - 1))
            e._start_tick, e._end_tick = start, start + self.episode
            e.original_simulator.balance = e.original_simulator.equity = \
                1e4 * float(np.exp(self.rng.uniform(np.log(0.05), np.log(2.0))))
        else:
            b = float((options or {}).get("balance", 1e4))
            e._start_tick, e._end_tick = e.window_size - 1, n - 1
            e.original_simulator.balance = e.original_simulator.equity = b
        o, info = self.env.reset()
        self.e0 = float(info["equity"])
        self.ledger = self.e0
        return self._flat(o), info

    def step(self, action):
        before_bal = self.env.unwrapped.simulator.balance
        o, r, term, trunc, info = self.env.step(action)
        self.ledger += r
        return self._flat(o), float(r) / 100.0, term, trunc, info


def make_env(condition: str, prices: pd.DataFrame, train: bool, seed: int = 0) -> EquityObs:
    sim = make_simulator(condition, prices)
    env = MtEnv(original_simulator=sim, trading_symbols=[SYMBOL], window_size=2, symbol_max_orders=1)  # published default fee
    return EquityObs(env, train=train, seed=seed)

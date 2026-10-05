"""Study E environment (PREREGISTRATION_E.md): equity-scaled positions under three reward/liability conventions and two
leverage sets, in the placebo (no-edge) market of Study B.

Position: q_t = clip(lev * Q1 * E_t / K0, -cap_t, cap_t)  -- scaled to current reported equity (Study B: fixed allocation).
P&L: pnl = q s - FEE |q|; the ledger always books it. A step is a "bust" when E_t + pnl <= 0; the episode/sleeve ends.

Rules
    LINFL  linear reward = equity change; the bust step books the full loss, overshoot included (full liability)
    LINLL  linear reward = equity change with the loss capped at the remaining equity (limited liability at bankruptcy)
    LOGNF  log reward = log(E_{t+1}/E_t); on the bust step the reward is never computed and stays 0 (Gym-Trading-Env rule)
Leverage sets (9 levels each)
    LOW   {0, +-0.25, +-1, +-4, +-16}   single-quarter busts are (almost) impossible in the data
    HIGH  {0, +-1, +-4, +-16, +-64}     single-quarter busts become possible
Reported equity is what the environment reports (LINLL floors it at 0; LINFL and LOGNF report the booked value).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from envs_b import E0_TRAIN, EPISODE, FEE, K0, PSCALE, Q1, clipobs, load_market  # noqa: F401  (load_market re-exported)

LEVSETS = {"LOW": np.array([-16.0, -4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0, 16.0]),
           "HIGH": np.array([-64.0, -16.0, -4.0, -1.0, 0.0, 1.0, 4.0, 16.0, 64.0])}
RULES = ("LINFL", "LINLL", "LOGNF")
LOG_SCALE = 100.0   # reward scaling of the log reward (as the linear reward is scaled by 100 / K0)


class CurvatureEnv(gym.Env):
    metadata: dict = {}

    def __init__(self, market, rule: str, levset: str, train: bool, seed: int = 0):
        super().__init__()
        assert rule in RULES and levset in LEVSETS
        self.m, self.rule, self.LEV, self.train = market, rule, LEVSETS[levset], train
        self.action_space = gym.spaces.Discrete(len(self.LEV))
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
        self.t, self.e0 = self.t0, e0
        self.ledger = self.reported = e0
        self.closed = self.bust = False
        self.lev_abs_sum, self.open_q, self.top_q = 0.0, 0, 0
        return self._obs(), {}

    def _obs(self):
        t, m = self.t, self.m
        return clipobs(np.array([m.spot[t] / PSCALE, *m.spr_lags[t], np.clip(self.reported / K0, 0, 5), *m.time[t]]))

    def step(self, action):
        t, m = self.t, self.m
        lev = 0.0 if self.closed else float(self.LEV[int(action)])
        if not self.closed:
            self.open_q += 1
            self.lev_abs_sum += abs(lev)
            self.top_q += int(abs(lev) == self.LEV.max())
        q = float(np.clip(lev * Q1 * max(self.reported, 0.0) / K0, -m.cap_mwh[t], m.cap_mwh[t]))
        pnl = q * m.spread[t] - FEE * abs(q)
        self.ledger += pnl
        before = self.reported
        new = before + pnl
        if new <= 0.0 and not self.closed:
            self.bust = self.closed = True
        if self.rule == "LINFL":
            self.reported = new
            reward = (new - before) / K0 * 100.0
        elif self.rule == "LINLL":
            self.reported = max(new, 0.0)
            reward = (self.reported - before) / K0 * 100.0
        else:  # LOGNF
            self.reported = new
            reward = float(np.log(new / before)) * LOG_SCALE if (new > 0.0 and before > 0.0) else 0.0
        self.t += 1
        done = self.t >= self.t_end or (self.train and self.closed)
        return (self._obs() if not done else np.zeros(11, np.float32)), float(reward), done, False, {"pnl": pnl}


def outcome(env: CurvatureEnv) -> dict:
    return {"reported_return_pct": 100 * (env.reported / env.e0 - 1), "ledger_return_pct": 100 * (env.ledger / env.e0 - 1),
            "bust": bool(env.bust), "mean_abs_lev_open": env.lev_abs_sum / max(env.open_q, 1),
            "share_top_lev_open": env.top_q / max(env.open_q, 1), "open_quarters": env.open_q}

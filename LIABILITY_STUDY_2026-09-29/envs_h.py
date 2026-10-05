"""Study H environment (PREREGISTRATION_H.md): Study B's fixed-sizing sleeve with realistic frictions.

Changes relative to envs_b.LiabilityEnv (everything else identical):
    fee            1.0 EUR/MWh (Study B: 0.10)
    liquidity cap  10% of the balancing volume (Study B: 25%)
    price impact   quadratic, kappa * q^2 / cap with kappa = 50 EUR/MWh (Study B: none)
All costs are part of the quarter-hour P&L, so the liability rules treat them like trading losses.
"""
from __future__ import annotations

import numpy as np

from envs_b import K0, LEV, Q1, LiabilityEnv, load_market, obs_matrix, outcome  # noqa: F401 (re-exported)

FEE_H, CAPF_H, KAPPA_H = 1.0, 0.10, 50.0


class FrictionEnv(LiabilityEnv):
    def step(self, action):
        t, m = self.t, self.m
        lev = 0.0 if self.closed else float(LEV[int(action)])
        if not self.closed:
            self.open_q += 1
            self.lev_abs_sum += abs(lev)
            self.max_lev_q += int(abs(lev) == LEV.max())
        cap = m.cap_mwh[t] * CAPF_H / 0.25
        q = float(np.clip(lev * Q1, -cap, cap))
        pnl = q * m.spread[t] - FEE_H * abs(q) - KAPPA_H * q * q / max(cap, 0.05)
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

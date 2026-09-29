"""Ledger reconciliation check for trading-simulator evaluations.

Rebuilds the equity path from booked trading P&L and booked costs, and compares it with the equity the simulator
reports. The two must agree whenever the simulator's accounting is sound. A large divergence means the reported
number is not the money the strategy made: for example losses forgiven by a zero floor, recapitalisation, or a payoff
channel that bypasses the ledger.

Usage:
    python ledger_check.py STEP_LOG.csv[.gz] [--reported-col trading_sleeve_value_dkk] [--pnl-col mtm_pnl]
                           [--cost-cols cumulative_volume_transaction_fees_dkk,...] [--distributions-col total_distributions_dkk]
                           [--tol-pp 5]

Exit code 0 = reconciled within tolerance, 1 = divergence above tolerance, 2 = input error.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

DEFAULT_COSTS = ["cumulative_volume_transaction_fees_dkk", "cumulative_market_access_fees_dkk",
                 "cumulative_impact_costs_dkk", "cumulative_collateral_funding_costs_dkk"]


def reconcile(d: pd.DataFrame, reported_col: str, pnl_col: str, cost_cols: list[str], distributions_col: str | None,
              tol_pp: float) -> dict:
    reported = d[reported_col].astype(float).values
    if distributions_col and distributions_col in d:  # distributions leave the account but are part of the return
        reported = reported + d[distributions_col].fillna(0).astype(float).values
    e0 = float(reported[0])
    cum_costs = d[[c for c in cost_cols if c in d]].fillna(0).sum(axis=1).values
    step = d[pnl_col].fillna(0).astype(float).values - np.diff(np.r_[0.0, cum_costs])
    ledger = e0 + np.cumsum(step)
    gap_pp = 100.0 * (reported - ledger) / e0
    worst = int(np.argmax(np.abs(gap_pp)))
    floor_steps = int(np.sum((d[reported_col].astype(float).values <= 1e-6 * abs(e0)) & (ledger < 0)))
    out = dict(steps=int(len(d)), initial_equity=e0,
               reported_return_pct=float(100.0 * (reported[-1] / e0 - 1)), ledger_return_pct=float(100.0 * (ledger[-1] / e0 - 1)),
               ledger_min_equity_pct=float(100.0 * (ledger.min() / e0 - 1)), final_gap_pp=float(gap_pp[-1]),
               max_abs_gap_pp=float(abs(gap_pp[worst])), max_gap_step=worst,
               steps_reported_at_zero_while_ledger_negative=floor_steps, tolerance_pp=tol_pp)
    out["reconciled"] = bool(out["max_abs_gap_pp"] <= tol_pp)
    if not out["reconciled"]:
        out["diagnosis"] = ("reported equity sits at zero while the ledger is negative: losses beyond equity are being "
                            "forgiven (zero-floor accounting)" if floor_steps else
                            "reported equity diverges from booked P&L: check payoff channels, recapitalisation or "
                            "costs missing from the ledger")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--reported-col", default="trading_sleeve_value_dkk")
    ap.add_argument("--pnl-col", default="mtm_pnl")
    ap.add_argument("--cost-cols", default=",".join(DEFAULT_COSTS))
    ap.add_argument("--distributions-col", default="total_distributions_dkk")
    ap.add_argument("--tol-pp", type=float, default=5.0)
    a = ap.parse_args()
    try:
        d = pd.read_csv(a.log)
        for c in (a.reported_col, a.pnl_col):
            if c not in d:
                raise KeyError(f"column {c!r} not in {a.log}")
    except Exception as e:  # noqa: BLE001
        print(f"input error: {e}", file=sys.stderr)
        return 2
    res = reconcile(d, a.reported_col, a.pnl_col, [c for c in a.cost_cols.split(",") if c], a.distributions_col, a.tol_pp)
    print(json.dumps(res, indent=1))
    return 0 if res["reconciled"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

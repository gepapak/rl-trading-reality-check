#!/usr/bin/env python3
"""Regression checks for final-campaign reporting contracts."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Ablations.mechanism_ablations import ARM_SPECS, prior_args_for_arm as campaign_args
from scripts.evaluate_temporal_price_baselines import _daily_sharpe_stats
from scripts.generate_detailed_metrics import (
    _rerun_command_for_missing,
    prior_args_for_arm as reporting_args,
)


def _test_arm_reconstruction() -> None:
    for arm in ARM_SPECS:
        if reporting_args(arm) != campaign_args(arm):
            raise AssertionError(f"Reporting arguments differ from campaign for {arm}")
        command = _rerun_command_for_missing(
            {
                "category": "mechanism_ablation",
                "needs_sleeve_rerun": True,
                "arm": arm,
                "seed": "7",
                "region": "original",
            },
            "python",
        )
        if not command or "--tier1_dir" not in command:
            raise AssertionError(f"Missing reporting rerun command for {arm}")


def _test_temporal_sharpe_conversion() -> None:
    initial = 100.0
    daily_returns = np.asarray([0.01, -0.004, 0.007, 0.002, -0.001], dtype=float)
    daily_equity = initial * np.cumprod(1.0 + daily_returns)
    ten_minute_equity = np.repeat(daily_equity, 144)
    stats = _daily_sharpe_stats(
        ten_minute_equity,
        initial_sleeve=initial,
        annual_risk_free_rate=0.02,
    )

    hac365 = float(stats["daily_hac7_sharpe"])
    nonannual = hac365 / math.sqrt(365.25)
    expected252 = nonannual * math.sqrt(252.0)
    if not math.isclose(
        float(stats["daily_hac7_nonannual_365"]),
        nonannual,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("Nonannual HAC Sharpe conversion is incorrect")
    if not math.isclose(
        float(stats["daily_hac7_annualized_252"]),
        expected252,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("252-day HAC Sharpe conversion is incorrect")


def main() -> int:
    _test_arm_reconstruction()
    _test_temporal_sharpe_conversion()
    print("CAMPAIGN_REPORTING_CONTRACT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

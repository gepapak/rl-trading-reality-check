#!/usr/bin/env python3
"""Harness 3 - risk-envelope-matched MAPPO controls.

Both methods use the identical symmetric, forecast-independent tail/liquidity/
collateral action support. The only difference is whether the actor observes
forecast features. Ten seeds support a fair comparison with plain MAPPO.

  1. cfm_zero_trust              x 10 seeds  (forecast-blind risk-limited MAPPO)
  2. cfm_zero_trust_forecast_obs x 10 seeds  (same support + forecast state)

Run:      python harness3.py
Dry run:  python harness3.py --dry_run
"""

import sys

from harness_common import SEEDS_FULL, run_jobs

ABL = "Ablations/mechanism_ablations.py"

JOBS = [
    (
        "cfm_zero_trust x10",
        [
            ABL,
            "--arms",
            "cfm_zero_trust",
            "--seeds",
            *SEEDS_FULL,
            "--skip_training_if_complete",
        ],
        True,
    ),
    (
        "cfm_zero_trust_forecast_obs x10",
        [
            ABL,
            "--arms",
            "cfm_zero_trust_forecast_obs",
            "--seeds",
            *SEEDS_FULL,
            "--skip_training_if_complete",
        ],
        True,
    ),
]


if __name__ == "__main__":
    sys.exit(run_jobs("harness3", JOBS))

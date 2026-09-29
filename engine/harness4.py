#!/usr/bin/env python3
"""Harness 4 - learned forecast-integration comparators.

  1. feasible_action_full             x 10 seeds  (headline learned integration)
  2. forecast_observation_only        x 10 seeds  (ordinary MAPPO + forecast state)
  3. unconditioned_capacity           x 3 seeds   (capacity ablation)
  4. feasible_action_no_paired_reward x 3 seeds   (paired-risk objective ablation)

Run:      python harness4.py
Dry run:  python harness4.py --dry_run
"""

import sys

from harness_common import SEEDS_FULL, SEEDS_MECH, run_jobs

ABL = "Ablations/mechanism_ablations.py"

JOBS = [
    (
        "feasible_action_full x10",
        [
            ABL,
            "--arms",
            "feasible_action_full",
            "--seeds",
            *SEEDS_FULL,
            "--skip_training_if_complete",
        ],
        True,
    ),
    (
        "forecast_observation_only x10",
        [
            ABL,
            "--arms",
            "forecast_observation_only",
            "--seeds",
            *SEEDS_FULL,
            "--skip_training_if_complete",
        ],
        True,
    ),
    (
        "unconditioned_capacity x3",
        [
            ABL,
            "--arms",
            "unconditioned_capacity",
            "--seeds",
            *SEEDS_MECH,
            "--skip_training_if_complete",
        ],
        True,
    ),
    (
        "feasible_action_no_paired_reward x3",
        [
            ABL,
            "--arms",
            "feasible_action_no_paired_reward",
            "--seeds",
            *SEEDS_MECH,
            "--skip_training_if_complete",
        ],
        True,
    ),
]


if __name__ == "__main__":
    sys.exit(run_jobs("harness4", JOBS))

#!/usr/bin/env python3
"""Harness 2 - plain MAPPO and protocol-matched classical baselines.

Static baselines run first. Plain MAPPO then runs over all ten training seeds.
Temporal rules are evaluated last and matched region-by-region to the completed
plain-MAPPO and headline-anchor exposures. The temporal step may wait for
Harness 1's focal_anchor evaluation when all harnesses start in parallel.

Run:      python harness2.py
Dry run:  python harness2.py --dry_run
"""

import sys

from harness_common import SEEDS_FULL, run_jobs

JOBS = [
    (
        "static baselines (both regions)",
        ["baselines.py", "--skip_temporal_baselines"],
        True,
    ),
    (
        "plain MAPPO x10",
        ["marl.py", "--seeds", *SEEDS_FULL, "--skip_training_if_complete"],
        True,
    ),
    (
        "wait for focal_anchor evaluation used by temporal matching",
        [
            "scripts/wait_for_campaign_results.py",
            "--suite_dir",
            (
                "Ablations/batch_tier_phase_runs/"
                "prototype5_mechanism_ablations_final_v1/focal_anchor"
            ),
            "--tier_dir",
            "tier1_forecast_utilization",
            "--seeds",
            "7",
            "--timeout_hours",
            "24",
        ],
        True,
    ),
    (
        "temporal price baselines matched to MAPPO and focal_anchor",
        ["baselines.py", "--skip_baselines"],
        True,
    ),
]


if __name__ == "__main__":
    sys.exit(run_jobs("harness2", JOBS))

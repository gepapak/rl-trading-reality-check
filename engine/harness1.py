#!/usr/bin/env python3
"""Harness 1 - deterministic anchor mechanism and forecast placebos.

Each deterministic arm runs once. Repeating it under training seeds is not
statistical evidence because the investor path is seed-invariant after battery
cash is removed. The corruption battery evaluates the headline anchor itself.

  1. focal_anchor          x 1 seed  (headline deterministic controller)
  2. anchor_gated          x 1 seed  (forecast-head agreement gate)
  3. global_evidence       x 1 seed  (pooled calibration)
  4. no_confidence_weight  x 1 seed  (remove confidence sizing)
  5. fixed_direction_cap   x 1 seed  (fixed-cap forecast direction)
  6. zero/shuffle/sign-flip placebos on focal_anchor, both regions

Run:      python harness1.py
Dry run:  python harness1.py --dry_run
"""

import sys

from harness_common import SEEDS_DETERMINISTIC, run_jobs

ABL = "Ablations/mechanism_ablations.py"

JOBS = [
    (
        "focal_anchor x1",
        [
            ABL,
            "--arms",
            "focal_anchor",
            "--seeds",
            *SEEDS_DETERMINISTIC,
            "--skip_training_if_complete",
        ],
        True,
    ),
    (
        "anchor_gated x1",
        [
            ABL,
            "--arms",
            "anchor_gated",
            "--seeds",
            *SEEDS_DETERMINISTIC,
            "--skip_training_if_complete",
        ],
        True,
    ),
    (
        "global_evidence x1",
        [
            ABL,
            "--arms",
            "global_evidence",
            "--seeds",
            *SEEDS_DETERMINISTIC,
            "--skip_training_if_complete",
        ],
        True,
    ),
    (
        "no_confidence_weight x1",
        [
            ABL,
            "--arms",
            "no_confidence_weight",
            "--seeds",
            *SEEDS_DETERMINISTIC,
            "--skip_training_if_complete",
        ],
        True,
    ),
    (
        "fixed_direction_cap x1",
        [
            ABL,
            "--arms",
            "fixed_direction_cap",
            "--seeds",
            *SEEDS_DETERMINISTIC,
            "--skip_training_if_complete",
        ],
        True,
    ),
    (
        "focal_anchor forecast-corruption battery x1",
        [
            "Ablations/forecast_corruption_ablations.py",
            "--arm",
            "focal_anchor",
            "--seeds",
            *SEEDS_DETERMINISTIC,
            "--modes",
            "zero_edge",
            "shuffle",
            "sign_flip",
            "--regions",
            "original",
            "v2",
            "--rebuild_caches",
        ],
        True,
    ),
]


if __name__ == "__main__":
    sys.exit(run_jobs("harness1", JOBS))

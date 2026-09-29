#!/usr/bin/env python3
"""Wait until required original and v2 evaluation JSONs exist.

This dependency gate lets final-campaign harnesses run in parallel without
silently falling back to guessed exposure-matching values.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_completed_result(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    tiers = payload.get("tiers")
    if not isinstance(tiers, dict) or not tiers:
        return False
    return any(
        isinstance(metrics, dict)
        and str(metrics.get("status", "")).strip().lower() == "completed"
        for metrics in tiers.values()
    )


def _missing(suite_dir: Path, tier_dir: str, seeds: list[int]) -> list[Path]:
    missing: list[Path] = []
    for seed in seeds:
        for eval_dir in ("evaluations_2025", "evaluations_2025_v2"):
            folder = suite_dir / f"seed{seed}" / eval_dir / tier_dir
            if not any(
                _is_completed_result(path)
                for path in folder.glob("evaluation_tiers_*.json")
            ):
                missing.append(folder)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite_dir", required=True)
    parser.add_argument("--tier_dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--poll_seconds", type=float, default=60.0)
    parser.add_argument("--timeout_hours", type=float, default=168.0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    if not suite_dir.is_absolute():
        suite_dir = PROJECT_ROOT / suite_dir
    if args.dry_run:
        print(
            "[DRY-RUN] wait for original and v2 evaluations:",
            suite_dir,
            "seeds=",
            args.seeds,
        )
        return 0

    deadline = time.monotonic() + max(float(args.timeout_hours), 0.0) * 3600.0
    last_count: int | None = None
    while True:
        missing = _missing(suite_dir, str(args.tier_dir), list(args.seeds))
        if not missing:
            print(f"[OK] Required evaluations are available under {suite_dir}")
            return 0
        if len(missing) != last_count:
            print(
                f"[WAIT] {len(missing)} evaluation folder(s) incomplete; "
                f"next check in {float(args.poll_seconds):g}s",
                flush=True,
            )
            last_count = len(missing)
        if time.monotonic() >= deadline:
            print("[ERROR] Timed out waiting for:")
            for path in missing:
                print(" ", path)
            return 1
        time.sleep(max(float(args.poll_seconds), 1.0))


if __name__ == "__main__":
    raise SystemExit(main())

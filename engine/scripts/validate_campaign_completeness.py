#!/usr/bin/env python3
"""Verify that every final-campaign result required by the paper exists."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_contract import engine_file_hashes


CURRENT_ENGINE_HASHES = engine_file_hashes(str(ROOT))
SEEDS_FULL = [7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202]
SEEDS_MECH = [7, 42, 123]
REGIONS = [
    ("evaluations_2025", "original"),
    ("evaluations_2025_v2", "v2"),
]


def _completed_json(folder: Path) -> Path | None:
    for path in sorted(folder.glob("evaluation_tiers_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        tiers = payload.get("tiers")
        if not isinstance(tiers, dict):
            continue
        for metrics in tiers.values():
            if not isinstance(metrics, dict):
                continue
            if str(metrics.get("status", "")).strip().lower() != "completed":
                continue
            if not isinstance(metrics.get("sleeve_metrics"), dict):
                continue
            if not bool(metrics.get("log_sleeve", False)):
                continue
            env_log = Path(str(metrics.get("env_debug_log", "")))
            if not env_log.is_absolute():
                env_log = ROOT / env_log
            if not env_log.is_file():
                continue
            train_hash = str(metrics.get("train_runtime_contract_hash", ""))
            policy_match_hash = str(
                metrics.get("eval_policy_match_contract_hash", "")
            )
            if not train_hash or train_hash != policy_match_hash:
                continue
            if metrics.get("engine_file_hashes") != CURRENT_ENGINE_HASHES:
                continue
            return path
    return None


def _expect_suite(
    missing: list[str],
    *,
    root: Path,
    seeds: list[int],
    tier_name: str,
) -> int:
    count = 0
    for seed in seeds:
        for eval_dir, region in REGIONS:
            folder = root / f"seed{seed}" / eval_dir / tier_name
            if _completed_json(folder) is None:
                missing.append(f"{root.name}: seed={seed} region={region} ({folder})")
            else:
                count += 1
    return count


def main() -> int:
    missing: list[str] = []
    counts: dict[str, int] = {}

    marl_root = (
        ROOT
        / "batch_tier_phase_runs"
        / "prototype5_mappo_marl_final_v1"
    )
    counts["mappo_marl"] = _expect_suite(
        missing,
        root=marl_root,
        seeds=SEEDS_FULL,
        tier_name="tier1",
    )

    mechanism_root = (
        ROOT
        / "Ablations"
        / "batch_tier_phase_runs"
        / "prototype5_mechanism_ablations_final_v1"
    )
    expected_arms = {
        "focal_anchor": [7],
        "anchor_gated": [7],
        "global_evidence": [7],
        "no_confidence_weight": [7],
        "fixed_direction_cap": [7],
        "cfm_zero_trust": SEEDS_FULL,
        "cfm_zero_trust_forecast_obs": SEEDS_FULL,
        "feasible_action_full": SEEDS_FULL,
        "forecast_observation_only": SEEDS_FULL,
        "unconditioned_capacity": SEEDS_MECH,
        "feasible_action_no_paired_reward": SEEDS_MECH,
    }
    for arm, seeds in expected_arms.items():
        counts[arm] = _expect_suite(
            missing,
            root=mechanism_root / arm,
            seeds=seeds,
            tier_name="tier1_forecast_utilization",
        )

    corruption_root = (
        ROOT
        / "Ablations"
        / "results"
        / "prototype5_forecast_corruption_final_v1"
    )
    corruption_count = 0
    for mode in ("zero_edge", "shuffle", "sign_flip"):
        for region in ("original", "v2"):
            folder = corruption_root / mode / region / "seed7"
            if _completed_json(folder) is None:
                missing.append(
                    f"forecast_corruption: mode={mode} region={region} ({folder})"
                )
            else:
                corruption_count += 1
    counts["forecast_corruption"] = corruption_count

    static_summary = (
        ROOT
        / "baseline_results"
        / "prototype5_final_paper_baselines_v1"
        / "final_paper_baseline_summary_latest.csv"
    )
    static_ok = False
    if not static_summary.is_file():
        missing.append(f"static_baselines ({static_summary})")
    else:
        with static_summary.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        region_counts = {
            region: sum(1 for row in rows if row.get("region") == region)
            for region in ("original", "unseendata_v2")
        }
        static_ok = all(count >= 5 for count in region_counts.values())
        if not static_ok:
            missing.append(
                "static_baselines: expected at least five completed rows per "
                f"region, found {region_counts} ({static_summary})"
            )
    counts["static_baseline_summary"] = int(static_ok)

    temporal_root = (
        ROOT
        / "baseline_results"
        / "prototype5_temporal_baselines_v1"
    )
    temporal_count = 0
    for profile in ("focal_matched", "marl_matched"):
        for region in ("original", "unseendata_v2"):
            path = (
                temporal_root
                / profile
                / region
                / "temporal_price_baseline_summary.csv"
            )
            temporal_ok = False
            if not path.is_file():
                missing.append(
                    f"temporal_baseline: profile={profile} region={region} ({path})"
                )
            else:
                with path.open("r", newline="", encoding="utf-8") as handle:
                    temporal_rows = list(csv.DictReader(handle))
                temporal_ok = len(temporal_rows) >= 3
                if not temporal_ok:
                    missing.append(
                        "temporal_baseline: expected three rule rows for "
                        f"profile={profile} region={region}, found "
                        f"{len(temporal_rows)} ({path})"
                    )
            if temporal_ok:
                temporal_count += 1
    counts["temporal_baseline_summaries"] = temporal_count

    print("Final campaign result counts:")
    for label, count in counts.items():
        print(f"  {label:38s} {count}")
    if missing:
        print(f"\nINCOMPLETE: {len(missing)} required result item(s) missing")
        for item in missing:
            print(f"  - {item}")
        return 1
    print("\nCAMPAIGN_COMPLETENESS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

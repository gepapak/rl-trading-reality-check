#!/usr/bin/env python3
"""Freeze data provenance and create the causal evaluation handoff.

This script does not regenerate stochastic physical trajectories. It hashes the
artifacts actually used by the experiments, creates history_020 from the final
2024-H2 training scenario, and writes project-relative manifests suitable for
the paper/reproduction package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIRS = (
    "training_dataset_ffill",
    "rolling_past_history_dataset_ffill",
    "forecast_training_dataset_settlement_hourly_v2",
    "settlement_price_dataset_real_v2",
    "liquidity_volume_dataset_real_v1",
    "evaluation_dataset_ffill",
    "forecast_cache_input_settlement_v2",
    "forecast_models_settlement_hourly_v2",
    "forecast_cache_settlement_hourly_v2",
)
CORE_FILES = (
    "real_settlement_protocol_v2_manifest.json",
    "real_liquidity_volume_protocol_v1_manifest.json",
)

PORTABLE_ARTIFACT_ROOTS = set(CORE_DIRS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _csv_record(path: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "path": _relative(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }
    try:
        header = pd.read_csv(path, nrows=0).columns
        if "timestamp" in header:
            frame = pd.read_csv(path, usecols=["timestamp"])
            timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
            record.update({
                "rows": int(len(frame)),
                "start": str(timestamps.iloc[0]) if len(frame) else "",
                "end": str(timestamps.iloc[-1]) if len(frame) else "",
                "timestamps_monotonic": bool(timestamps.is_monotonic_increasing),
                "timestamps_unique": bool(not timestamps.duplicated().any()),
            })
            if len(timestamps) > 1:
                deltas = timestamps.diff().dropna()
                record["ten_minute_regular"] = bool((deltas == pd.Timedelta(minutes=10)).all())
    except Exception as exc:
        record["inspection_error"] = str(exc)
    return record


def _portable_metadata_value(value: object, key: str = "") -> object:
    if isinstance(value, dict):
        return {
            str(child_key): _portable_metadata_value(child_value, str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_portable_metadata_value(item, key) for item in value]
    if not isinstance(value, str):
        return value
    if key == "project_root":
        return "."
    normalized = value.replace("\\", "/")
    for root_name in PORTABLE_ARTIFACT_ROOTS:
        marker = f"/{root_name}/"
        index = normalized.lower().find(marker.lower())
        if index >= 0:
            return normalized[index + 1 :]
        terminal = f"/{root_name}"
        if normalized.lower().endswith(terminal.lower()):
            return root_name
    return value


def _normalize_cache_metadata() -> None:
    cache_root = PROJECT_ROOT / "forecast_cache_settlement_hourly_v2"
    if not cache_root.is_dir():
        raise FileNotFoundError(cache_root)
    for path in sorted(cache_root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        portable = _portable_metadata_value(payload)
        path.write_text(json.dumps(portable, indent=2, sort_keys=True), encoding="utf-8")


def _ensure_eval_history(overwrite: bool) -> Path:
    source = PROJECT_ROOT / "training_dataset_ffill" / "scenario_019.csv"
    dest = PROJECT_ROOT / "rolling_past_history_dataset_ffill" / "history_020.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    if dest.exists() and not overwrite:
        if _sha256(source) != _sha256(dest):
            raise RuntimeError(f"Existing evaluation handoff differs from {source}: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def _write_history_manifest() -> Path:
    history_dir = PROJECT_ROOT / "rolling_past_history_dataset_ffill"
    training_dir = PROJECT_ROOT / "training_dataset_ffill"
    rows = []
    for history in sorted(history_dir.glob("history_[0-9][0-9][0-9].csv")):
        episode = int(history.stem.split("_")[-1])
        source = training_dir / f"scenario_{episode - 1:03d}.csv" if episode > 0 else None
        source_matches = bool(
            source is not None
            and source.is_file()
            and _sha256(source) == _sha256(history)
        )
        record = _csv_record(history)
        rows.append({
            "marl_episode": f"{episode:03d}",
            "history_file": history.name,
            "source_file": _relative(source) if source_matches and source is not None else "frozen_pre-2015_bootstrap_artifact",
            "source_type": "previous_marl_episode_full" if source_matches else "frozen_bootstrap_artifact",
            "rows": record.get("rows", ""),
            "start": record.get("start", ""),
            "end": record.get("end", ""),
            "output_sha256": record["sha256"],
            "source_hash_verified": source_matches,
        })
    path = history_dir / "history_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_eval_metadata(csv_path: Path, region: str) -> Path:
    record = _csv_record(csv_path)
    metadata = {
        "script": _relative(Path(__file__)),
        "output": _relative(csv_path),
        "rows": record.get("rows"),
        "start": record.get("start"),
        "end": record.get("end"),
        "sha256": record["sha256"],
        "price_area": region,
        "price_fill": "ffill_only",
        "resolution": "10min",
        "physical_trajectory": "shared_2025_physical_template",
        "evaluation_role": (
            "primary_price_settlement_liquidity_region"
            if region == "DK1"
            else "cross_price_region_stress_test_with_shared_physical_trajectory"
        ),
        "note": (
            "The DK2 evaluation changes day-ahead price, settlement price, and "
            "liquidity region while preserving the DK1 physical trajectory."
        ),
    }
    path = csv_path.with_suffix(csv_path.suffix + ".metadata.json")
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite_history_020", action="store_true")
    parser.add_argument("--require_complete_eval_day", action="store_true")
    args = parser.parse_args()

    history_020 = _ensure_eval_history(bool(args.overwrite_history_020))
    history_manifest = _write_history_manifest()
    _normalize_cache_metadata()

    eval_files = [
        (PROJECT_ROOT / "evaluation_dataset_ffill" / "unseendata.csv", "DK1"),
        (PROJECT_ROOT / "evaluation_dataset_ffill" / "unseendata_v2.csv", "DK2"),
    ]
    for path, region in eval_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        record = _csv_record(path)
        if not record.get("ten_minute_regular", False):
            raise RuntimeError(f"Evaluation grid is not a regular 10-minute series: {path}")
        end = pd.Timestamp(str(record["end"]))
        if args.require_complete_eval_day and (end.hour, end.minute) != (23, 50):
            raise RuntimeError(
                f"Evaluation grid ends at {end}, not a complete delivery day. "
                "Run scripts/complete_evaluation_window.py first."
            )
        _write_eval_metadata(path, region)

    files = []
    for dirname in CORE_DIRS:
        base = PROJECT_ROOT / dirname
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".keras", ".pkl"}:
                files.append(_csv_record(path) if path.suffix.lower() == ".csv" else {
                    "path": _relative(path),
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                })
    for filename in CORE_FILES:
        path = PROJECT_ROOT / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({
            "path": _relative(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        })

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "Prototype4_same_delivery_causal_v3_MWh_real_settlement_liquidity_margin",
        "generator_seed_recoverable_for_frozen_physical_artifacts": False,
        "reproducibility_basis": (
            "The exact experiment inputs are frozen by SHA-256. Future stochastic "
            "regeneration scripts use explicit deterministic seeds, but are not claimed "
            "to recreate the historical frozen trajectories without their original seed."
        ),
        "cross_region_design": (
            "DK2 is a price/settlement/liquidity stress test sharing the DK1 physical trajectory."
        ),
        "history_eval_handoff": _relative(history_020),
        "history_manifest": _relative(history_manifest),
        "files": files,
    }
    output = PROJECT_ROOT / "protocol_data_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] evaluation history: {history_020}")
    print(f"[OK] history manifest: {history_manifest}")
    print(f"[OK] protocol manifest: {output} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

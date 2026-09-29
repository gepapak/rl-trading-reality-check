"""Fail-fast checks for the frozen same-delivery causal data contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamps(path: Path) -> pd.Series:
    if not path.is_file():
        raise RuntimeError(f"[PROTOCOL_PREFLIGHT] Missing required file: {path}")
    frame = pd.read_csv(path, usecols=["timestamp"])
    values = pd.to_datetime(frame["timestamp"], errors="raise")
    if values.duplicated().any() or not values.is_monotonic_increasing:
        raise RuntimeError(f"[PROTOCOL_PREFLIGHT] Timestamps are not unique and sorted: {path}")
    if len(values) > 1 and not (values.diff().dropna() == pd.Timedelta(minutes=10)).all():
        raise RuntimeError(f"[PROTOCOL_PREFLIGHT] Expected a regular 10-minute grid: {path}")
    return values


def _verify_frozen_manifest(root: Path) -> None:
    manifest_path = root / "protocol_data_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"[PROTOCOL_PREFLIGHT] Missing frozen data manifest: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"[PROTOCOL_PREFLIGHT] Invalid data manifest {manifest_path}: {exc}"
        ) from exc

    records = manifest.get("files", [])
    if not isinstance(records, list) or not records:
        raise RuntimeError("[PROTOCOL_PREFLIGHT] Data manifest contains no file records")
    expected_protocol = "Prototype4_same_delivery_causal_v3_MWh_real_settlement_liquidity_margin"
    if str(manifest.get("protocol", "")) != expected_protocol:
        raise RuntimeError(
            "[PROTOCOL_PREFLIGHT] Frozen data manifest has the wrong protocol label: "
            f"expected={expected_protocol}, actual={manifest.get('protocol')}"
        )
    declared = {str(record.get("path", "")) for record in records if isinstance(record, dict)}
    required_prefixes = (
        "forecast_models_settlement_hourly_v2/",
        "forecast_cache_settlement_hourly_v2/",
    )
    for prefix in required_prefixes:
        if not any(path.startswith(prefix) for path in declared):
            raise RuntimeError(
                f"[PROTOCOL_PREFLIGHT] Data manifest omits required artifact tree: {prefix}"
            )
    for required_file in (
        "real_settlement_protocol_v2_manifest.json",
        "real_liquidity_volume_protocol_v1_manifest.json",
    ):
        if required_file not in declared:
            raise RuntimeError(
                f"[PROTOCOL_PREFLIGHT] Data manifest omits source manifest: {required_file}"
            )

    errors: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            errors.append("non-object manifest record")
            continue
        relative = str(record.get("path", "") or "").strip()
        expected_hash = str(record.get("sha256", "") or "").strip().lower()
        expected_bytes = record.get("bytes")
        if not relative or not expected_hash:
            errors.append(f"incomplete manifest record: {record!r}")
            continue
        path = root / Path(relative)
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
            errors.append(
                f"size mismatch {relative}: expected={expected_bytes}, actual={path.stat().st_size}"
            )
            continue
        actual_hash = _sha256(path)
        if actual_hash.lower() != expected_hash:
            errors.append(
                f"hash mismatch {relative}: expected={expected_hash}, actual={actual_hash}"
            )
        if len(errors) >= 20:
            break
    if errors:
        raise RuntimeError(
            "[PROTOCOL_PREFLIGHT] Frozen artifact verification failed:\n  "
            + "\n  ".join(errors)
        )


def _verify_forecast_artifact_contract(root: Path) -> None:
    expected_version = "3.0.0"
    expected_alignment = "same_delivery_causal_v3"
    errors: list[str] = []

    model_root = root / "forecast_models_settlement_hourly_v2"
    for episode in range(21):
        meta_path = (
            model_root
            / f"episode_{episode}"
            / "price_short_experts"
            / "ANN"
            / "ann_metadata.json"
        )
        if not meta_path.is_file():
            errors.append(f"missing model metadata {meta_path.relative_to(root)}")
            continue
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            errors.append(f"invalid model metadata {meta_path.relative_to(root)}: {exc}")
            continue
        if str(metadata.get("version", "")) != expected_version:
            errors.append(
                f"stale model {meta_path.relative_to(root)}: version={metadata.get('version')}"
            )
        if not bool(metadata.get("ann_input_includes_entry_price", False)):
            errors.append(f"model lacks causal entry input: {meta_path.relative_to(root)}")
        if str(metadata.get("target_delivery_alignment", "")) != "same_delivery_hour":
            errors.append(f"model target is not same-delivery: {meta_path.relative_to(root)}")

    cache_root = root / "forecast_cache_settlement_hourly_v2"
    cache_metadata = sorted(cache_root.rglob("precomputed_forecasts_*_metadata.json"))
    if not cache_metadata:
        errors.append("forecast cache contains no metadata files")
    for meta_path in cache_metadata:
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            errors.append(f"invalid cache metadata {meta_path.relative_to(root)}: {exc}")
            continue
        if str(metadata.get("price_short_expert_version", "")) != expected_version:
            errors.append(
                f"stale cache {meta_path.relative_to(root)}: "
                f"expert_version={metadata.get('price_short_expert_version')}"
            )
        if str(metadata.get("target_alignment_version", "")) != expected_alignment:
            errors.append(
                f"stale cache alignment {meta_path.relative_to(root)}: "
                f"alignment={metadata.get('target_alignment_version')}"
            )
        if str(metadata.get("forecast_information_set", "")) != (
            "settlement_history_strictly_before_origin_plus_current_day_ahead_entry"
        ):
            errors.append(f"cache information set is not causal: {meta_path.relative_to(root)}")

    if errors:
        raise RuntimeError(
            "[PROTOCOL_PREFLIGHT] Forecast artifacts do not satisfy the causal v3 contract:\n  "
            + "\n  ".join(errors[:20])
            + ("\n  ..." if len(errors) > 20 else "")
            + "\nRetrain with scripts/train_hourly_forecast_models_v2.py --force_retrain, "
            "rebuild caches with scripts/precompute_hourly_forecast_cache_v2.py "
            "--overwrite_cache, then refresh protocol_data_manifest.json."
        )


def assert_final_data_contract(project_root: str | Path) -> None:
    root = Path(project_root).resolve()
    _verify_forecast_artifact_contract(root)
    _verify_frozen_manifest(root)
    history_dir = root / "rolling_past_history_dataset_ffill"
    history_019 = history_dir / "history_019.csv"
    history_020 = history_dir / "history_020.csv"
    scenario_019 = root / "training_dataset_ffill" / "scenario_019.csv"
    h19 = _timestamps(history_019)
    h20 = _timestamps(history_020)
    if _sha256(history_020) != _sha256(scenario_019):
        raise RuntimeError(
            "[PROTOCOL_PREFLIGHT] history_020 must exactly match "
            "training_dataset_ffill/scenario_019.csv"
        )
    history = pd.concat([h19, h20], ignore_index=True).drop_duplicates().sort_values()
    if history.iloc[0] > pd.Timestamp("2024-01-02") or history.iloc[-1] != pd.Timestamp("2024-12-31 23:50:00"):
        raise RuntimeError(
            "[PROTOCOL_PREFLIGHT] Evaluation rolling history must cover the trailing "
            f"2024 delivery year; got {history.iloc[0]}..{history.iloc[-1]}"
        )

    regions = {
        "original": (
            root / "evaluation_dataset_ffill/unseendata.csv",
            root / "evaluation_dataset_ffill/unseendata_settlement_real_v2.csv",
            root / "evaluation_dataset_ffill/unseendata_liquidity_volume_real_v1.csv",
            root / "forecast_cache_input_settlement_v2/unseendata.csv",
        ),
        "v2": (
            root / "evaluation_dataset_ffill/unseendata_v2.csv",
            root / "evaluation_dataset_ffill/unseendata_v2_settlement_real_v2.csv",
            root / "evaluation_dataset_ffill/unseendata_v2_liquidity_volume_real_v1.csv",
            root / "forecast_cache_input_settlement_v2/unseendata_v2.csv",
        ),
    }
    eval_frames = {}
    for region, paths in regions.items():
        timestamp_sets = [_timestamps(path) for path in paths]
        reference = timestamp_sets[0]
        if reference.iloc[-1] != pd.Timestamp("2025-09-30 23:50:00"):
            raise RuntimeError(
                f"[PROTOCOL_PREFLIGHT] {region} evaluation ends at {reference.iloc[-1]}; "
                "complete the evaluation window and rebuild settlement, liquidity, and forecast caches."
            )
        for path, values in zip(paths[1:], timestamp_sets[1:]):
            if len(values) != len(reference) or not np.array_equal(values.to_numpy(), reference.to_numpy()):
                raise RuntimeError(
                    f"[PROTOCOL_PREFLIGHT] Timestamp mismatch for {region}: {paths[0]} vs {path}"
                )
        eval_frames[region] = pd.read_csv(paths[0], usecols=["wind", "solar", "hydro", "load"])

    original = eval_frames["original"].to_numpy(dtype=float)
    v2 = eval_frames["v2"].to_numpy(dtype=float)
    if original.shape != v2.shape or not np.allclose(original, v2, rtol=0.0, atol=1e-8):
        raise RuntimeError(
            "[PROTOCOL_PREFLIGHT] DK2 stress-test physical trajectory must match DK1 exactly."
        )

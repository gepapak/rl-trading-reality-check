"""Precompute v2 hourly forecast cache and expand it to the 10-minute grid.

The forecast models are trained on hourly data, but FoCAL consumes one cache row
per MARL environment row.  This script predicts once per hourly origin and then
holds that forecast constant across the six 10-minute rows inside the hour.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EnhancedConfig
from forecast_engine import get_episode_forecast_dirs
from forecast_price_experts import PRICE_SHORT_EXPERT_VERSION, price_short_expert_bank_exists
from generator import MultiHorizonForecastGenerator


REQUIRED_FORECAST_COLS = [
    "price_short_expert_ann_pred_return",
    "price_short_expert_ann_direction_prob",
    "price_short_expert_ann_direction_margin",
    "price_short_expert_ann_uncertainty",
    "price_short_expert_ann_quality",
]

OPTIONAL_FORECAST_COLS = [
    "price_forecast_short",
    "price_short_expert_ann",
    "price_short_expert_ann_latent_norm",
    "price_short_expert_ann_latent_0",
    "price_short_expert_ann_latent_1",
    "price_short_expert_ann_latent_2",
    "price_short_expert_ann_latent_3",
]


def _as_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _hourly_v2_config(look_back_hours: int, horizon_hours: int) -> EnhancedConfig:
    cfg = EnhancedConfig()
    cfg.forecast_look_back = int(look_back_hours)
    cfg.forecast_horizons = dict(getattr(cfg, "forecast_horizons", {}) or {})
    cfg.forecast_horizons["short"] = int(horizon_hours)
    cfg.forecast_targets = ["price"]
    cfg.investment_freq = 1
    return cfg


def _parse_timestamps(df: pd.DataFrame, path: Path) -> pd.Series:
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing required column 'timestamp'")
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    bad = int(ts.isna().sum())
    if bad:
        raise ValueError(f"{path} has {bad:,} unparsable timestamp rows")
    return ts


def _validate_env_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in ("timestamp", "price", "wind", "solar", "hydro", "load") if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    _parse_timestamps(df, path)
    return df


def _apply_price_clip_guard(
    df: pd.DataFrame,
    *,
    clip_min: float,
    clip_max: float,
    disabled: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if disabled:
        return df, {"price_clip_guard_enabled": False}
    if clip_min >= clip_max:
        raise ValueError("--price_clip_min must be smaller than --price_clip_max")
    out = df.copy()
    raw = pd.to_numeric(out["price"], errors="coerce").astype(float)
    raw_np = raw.to_numpy(dtype=float)
    clipped = raw.clip(lower=float(clip_min), upper=float(clip_max))
    clipped_np = clipped.to_numpy(dtype=float)
    out["price"] = clipped_np
    if "settlement_price" in out.columns:
        out["settlement_price"] = pd.to_numeric(out["settlement_price"], errors="coerce").clip(
            lower=float(clip_min),
            upper=float(clip_max),
        )
    finite = np.isfinite(raw_np)
    clipped_mask = finite & (np.abs(raw_np - clipped_np) > 1e-12)
    stats = {
        "price_clip_guard_enabled": True,
        "price_clip_min": float(clip_min),
        "price_clip_max": float(clip_max),
        "raw_price_min": float(np.nanmin(raw_np)) if raw_np.size else None,
        "raw_price_max": float(np.nanmax(raw_np)) if raw_np.size else None,
        "clipped_price_min": float(np.nanmin(clipped_np)) if clipped_np.size else None,
        "clipped_price_max": float(np.nanmax(clipped_np)) if clipped_np.size else None,
        "price_clip_fraction": float(np.mean(clipped_mask)) if raw_np.size else 0.0,
        "price_clip_count": int(np.sum(clipped_mask)),
    }
    return out, stats


def _hourly_origin_rows(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    ts = _parse_timestamps(df, path)
    work = df.copy()
    work["_ts"] = ts
    work["_hour"] = ts.dt.floor("h")
    work["_is_exact_hour"] = (
        ts.dt.minute.eq(0)
        & ts.dt.second.eq(0)
        & ts.dt.microsecond.eq(0)
        & ts.dt.nanosecond.eq(0)
    )
    work = work.sort_values(
        ["_hour", "_is_exact_hour", "_ts"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    hourly = work.drop_duplicates("_hour", keep="first").sort_values("_hour", kind="mergesort")
    out = hourly.drop(columns=["_ts", "_hour", "_is_exact_hour"]).copy()
    out["timestamp"] = hourly["_hour"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    info = {
        "environment_rows": int(len(df)),
        "hourly_origin_rows": int(len(out)),
        "exact_hour_origin_rows": int(hourly["_is_exact_hour"].sum()),
        "fallback_hours_without_exact_hh00": int(len(hourly) - int(hourly["_is_exact_hour"].sum())),
    }
    return out, info


def _episode_cache_dir(cache_root: Path, episode_num: int) -> Path:
    return cache_root / f"episode_{int(episode_num)}"


def _eval_cache_dir(cache_root: Path, eval_data: Path) -> Path:
    eval_basename = eval_data.stem
    if eval_basename.lower() in {"unseendata", "unseen", "evaluation", "eval"}:
        eval_basename = "full"
    return (
        cache_root
        / "forecast_cache_eval_episode20_2025"
        / f"forecast_cache_eval_episode20_2025-{eval_basename}"
    )


def _cache_has_csv_and_metadata(cache_dir: Path) -> bool:
    if not cache_dir.is_dir():
        return False
    names = [p.name for p in cache_dir.iterdir() if p.is_file()]
    return any(name.startswith("precomputed_forecasts_") and name.endswith(".csv") for name in names) and any(
        name.startswith("precomputed_forecasts_") and name.endswith("_metadata.json") for name in names
    )


def _remove_cache_dir(cache_dir: Path, cache_root: Path) -> None:
    if not cache_dir.exists():
        return
    resolved = cache_dir.resolve()
    root_resolved = cache_root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Refusing to remove cache outside cache root: {resolved}")
    shutil.rmtree(resolved)


def _cache_stem(df: pd.DataFrame) -> str:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.notna().any():
        start = ts.min().strftime("%Y%m%d")
        end = ts.max().strftime("%Y%m%d")
        return f"precomputed_forecasts_{start}_to_{end}_{len(df)}rows"
    return f"precomputed_forecasts_{len(df)}rows"


def _validate_model_bank(forecast_base_dir: Path, episode_num: int) -> None:
    paths = get_episode_forecast_dirs(int(episode_num), str(forecast_base_dir))
    episode_dir = Path(paths["episode_dir"])
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"Missing forecast model episode dir: {episode_dir}")
    if not price_short_expert_bank_exists(str(episode_dir)):
        raise FileNotFoundError(f"Incomplete ANN short forecast bank for episode {episode_num}: {episode_dir}")


def _build_forecaster(
    forecast_base_dir: Path,
    episode_num: int,
    *,
    look_back_hours: int,
    horizon_hours: int,
) -> MultiHorizonForecastGenerator:
    _validate_model_bank(forecast_base_dir, episode_num)
    paths = get_episode_forecast_dirs(int(episode_num), str(forecast_base_dir))
    cfg = _hourly_v2_config(look_back_hours=look_back_hours, horizon_hours=horizon_hours)
    return MultiHorizonForecastGenerator(
        model_dir=paths["model_dir"],
        scaler_dir=paths["scaler_dir"],
        metadata_dir=paths["metadata_dir"],
        look_back=int(look_back_hours),
        expert_refresh_stride=1,
        verbose=False,
        fallback_mode=False,
        config=cfg,
    )


def _expand_hourly_cache_to_environment(
    env_df: pd.DataFrame,
    hourly_cache: pd.DataFrame,
    env_path: Path,
) -> pd.DataFrame:
    env_ts = _parse_timestamps(env_df, env_path)
    if "timestamp" not in hourly_cache.columns:
        raise ValueError("Hourly forecast cache is missing timestamp column")
    cache_ts = pd.to_datetime(hourly_cache["timestamp"], errors="coerce")
    if cache_ts.isna().any():
        raise ValueError("Hourly forecast cache has unparsable timestamp rows")

    missing_required = [c for c in REQUIRED_FORECAST_COLS if c not in hourly_cache.columns]
    if missing_required:
        raise ValueError(f"Hourly forecast cache missing required columns: {missing_required}")

    forecast_cols = [c for c in OPTIONAL_FORECAST_COLS + REQUIRED_FORECAST_COLS if c in hourly_cache.columns]
    hourly = hourly_cache[["timestamp"] + forecast_cols].copy()
    hourly["_hour"] = cache_ts.dt.floor("h")
    hourly = hourly.drop(columns=["timestamp"]).drop_duplicates("_hour", keep="last")

    expanded = pd.DataFrame(
        {
            "timestamp": env_ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "_hour": env_ts.dt.floor("h"),
        }
    )
    expanded = expanded.merge(hourly, on="_hour", how="left", sort=False)

    missing_rows = expanded[REQUIRED_FORECAST_COLS].isna().any(axis=1)
    if bool(missing_rows.any()):
        missing_count = int(missing_rows.sum())
        sample_hours = expanded.loc[missing_rows, "_hour"].drop_duplicates().head(5).astype(str).tolist()
        raise RuntimeError(
            f"Missing hourly forecast values for {missing_count:,} environment rows from {env_path}. "
            f"Sample missing hours: {sample_hours}"
        )

    for col in forecast_cols:
        expanded[col] = pd.to_numeric(expanded[col], errors="coerce")
    if expanded[forecast_cols].isna().any().any():
        raise RuntimeError(f"Non-finite forecast values after expansion for {env_path}")

    out_cols = ["timestamp"] + forecast_cols
    return expanded[out_cols].copy()


def _precompute_one(
    *,
    episode_num: int,
    data_path: Path,
    cache_dir: Path,
    forecast_base_dir: Path,
    cache_root: Path,
    look_back_hours: int,
    horizon_hours: int,
    overwrite_cache: bool,
    skip_existing: bool,
    dry_run: bool,
    price_clip_min: float,
    price_clip_max: float,
    disable_price_clip: bool,
) -> dict[str, Any]:
    env_df = _validate_env_dataset(data_path)
    env_df, price_clip_stats = _apply_price_clip_guard(
        env_df,
        clip_min=float(price_clip_min),
        clip_max=float(price_clip_max),
        disabled=bool(disable_price_clip),
    )
    hourly_df, hourly_info = _hourly_origin_rows(env_df, data_path)

    entry: dict[str, Any] = {
        "episode": int(episode_num),
        "data_path": str(data_path),
        "cache_dir": str(cache_dir),
        **hourly_info,
        **price_clip_stats,
    }

    if skip_existing and _cache_has_csv_and_metadata(cache_dir):
        entry["status"] = "skipped_existing"
        print(f"[SKIP] episode_{episode_num}: existing cache at {cache_dir}")
        return entry

    if overwrite_cache:
        _remove_cache_dir(cache_dir, cache_root)

    if dry_run:
        _validate_model_bank(forecast_base_dir, episode_num)
        entry["status"] = "dry_run"
        print(
            f"[DRY-RUN] episode_{episode_num}: {len(env_df):,} env rows, "
            f"{len(hourly_df):,} hourly origins -> {cache_dir}"
        )
        return entry

    cache_dir.mkdir(parents=True, exist_ok=True)
    forecaster = _build_forecaster(
        forecast_base_dir,
        episode_num,
        look_back_hours=look_back_hours,
        horizon_hours=horizon_hours,
    )
    with tempfile.TemporaryDirectory(prefix=f"hourly_v2_ep{episode_num}_") as tmp:
        hourly_csv = forecaster.precompute_offline(
            df=hourly_df,
            timestamp_col="timestamp",
            batch_size=8192,
            cache_dir=tmp,
        )
        hourly_cache = pd.read_csv(hourly_csv)

    expanded = _expand_hourly_cache_to_environment(env_df, hourly_cache, data_path)
    stem = _cache_stem(env_df)
    csv_path = cache_dir / f"{stem}.csv"
    meta_path = cache_dir / f"{stem}_metadata.json"
    expanded.to_csv(csv_path, index=False)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "forecast_grid": "hourly_v2",
        "environment_grid": "10min_forward_filled",
        "expanded_to_environment_rows": True,
        "expansion_rule": "hold each hourly-origin forecast constant across environment rows with the same floored hour",
        "forecast_alignment": "origin_timestamp",
        "forecast_information_set": "settlement_history_strictly_before_origin_plus_current_day_ahead_entry",
        "target_delivery_alignment": "same_delivery_hour",
        "target_alignment_version": "same_delivery_causal_v3",
        "price_short_expert_version": PRICE_SHORT_EXPERT_VERSION,
        "look_back": int(look_back_hours),
        "look_back_unit": "hours",
        "horizons": ["short"],
        "horizon_offsets": {"short": int(horizon_hours)},
        "horizon_unit": "hours",
        "prior_calibration_horizon_steps_on_10min_grid": int(horizon_hours) * 6,
        "rows": int(len(expanded)),
        "hourly_origin_rows": int(len(hourly_df)),
        "data_path": str(data_path),
        "forecast_model_episode": int(episode_num),
        "forecast_model_dir": str(forecast_base_dir / f"episode_{int(episode_num)}"),
        "cache_csv": str(csv_path),
        "required_forecast_columns": REQUIRED_FORECAST_COLS,
        **price_clip_stats,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    entry.update(
        {
            "status": "completed",
            "cache_csv": str(csv_path),
            "metadata": str(meta_path),
            "expanded_rows": int(len(expanded)),
        }
    )
    print(
        f"[OK] episode_{episode_num}: {len(hourly_df):,} hourly forecasts expanded "
        f"to {len(expanded):,} env rows -> {cache_dir}"
    )
    return entry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute hourly v2 forecast caches and expand them to the 10-minute MARL grid."
    )
    parser.add_argument("--forecast_base_dir", default="forecast_models_settlement_hourly_v2")
    parser.add_argument("--forecast_cache_dir", default="forecast_cache_settlement_hourly_v2")
    parser.add_argument("--episode_data_dir", default="forecast_cache_input_settlement_v2")
    parser.add_argument("--eval_data", default="forecast_cache_input_settlement_v2/unseendata.csv")
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--end_episode", type=int, default=19)
    parser.add_argument("--look_back_hours", type=int, default=24)
    parser.add_argument("--horizon_hours", type=int, default=1)
    parser.add_argument("--price_clip_min", type=float, default=-111750.0)
    parser.add_argument("--price_clip_max", type=float, default=111750.0)
    parser.add_argument(
        "--disable_price_clip",
        action="store_true",
        help="Disable the settlement price support guard. Not recommended for real-settlement protocol runs.",
    )
    parser.add_argument("--skip_training_episodes", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.start_episode) < 0 or int(args.end_episode) < int(args.start_episode):
        raise ValueError("Invalid episode range")
    if int(args.end_episode) >= 20:
        raise ValueError("Training episode range must stop at 19; episode 20 is reserved for evaluation")
    if bool(args.overwrite_cache) and bool(args.skip_existing):
        raise ValueError("--overwrite_cache and --skip_existing are mutually exclusive")

    forecast_base_dir = _as_path(args.forecast_base_dir)
    forecast_cache_dir = _as_path(args.forecast_cache_dir)
    episode_data_dir = _as_path(args.episode_data_dir)
    eval_data = _as_path(args.eval_data)
    forecast_cache_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "project_root": str(PROJECT_ROOT),
        "forecast_base_dir": str(forecast_base_dir),
        "forecast_cache_dir": str(forecast_cache_dir),
        "episode_data_dir": str(episode_data_dir),
        "eval_data": str(eval_data),
        "look_back_hours": int(args.look_back_hours),
        "horizon_hours": int(args.horizon_hours),
        "price_clip_min": float(args.price_clip_min),
        "price_clip_max": float(args.price_clip_max),
        "price_clip_guard_enabled": not bool(args.disable_price_clip),
        "training_episode_results": [],
        "eval_result": None,
    }

    if not bool(args.skip_training_episodes):
        for episode_num in range(int(args.start_episode), int(args.end_episode) + 1):
            data_path = episode_data_dir / f"scenario_{episode_num:03d}.csv"
            result = _precompute_one(
                episode_num=episode_num,
                data_path=data_path,
                cache_dir=_episode_cache_dir(forecast_cache_dir, episode_num),
                forecast_base_dir=forecast_base_dir,
                cache_root=forecast_cache_dir,
                look_back_hours=int(args.look_back_hours),
                horizon_hours=int(args.horizon_hours),
                overwrite_cache=bool(args.overwrite_cache),
                skip_existing=bool(args.skip_existing),
                dry_run=bool(args.dry_run),
                price_clip_min=float(args.price_clip_min),
                price_clip_max=float(args.price_clip_max),
                disable_price_clip=bool(args.disable_price_clip),
            )
            manifest["training_episode_results"].append(result)

    if not bool(args.skip_eval):
        result = _precompute_one(
            episode_num=20,
            data_path=eval_data,
            cache_dir=_eval_cache_dir(forecast_cache_dir, eval_data),
            forecast_base_dir=forecast_base_dir,
            cache_root=forecast_cache_dir,
            look_back_hours=int(args.look_back_hours),
            horizon_hours=int(args.horizon_hours),
            overwrite_cache=bool(args.overwrite_cache),
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
            price_clip_min=float(args.price_clip_min),
            price_clip_max=float(args.price_clip_max),
            disable_price_clip=bool(args.disable_price_clip),
        )
        manifest["eval_result"] = result

    manifest_path = forecast_cache_dir / "hourly_forecast_cache_v2_manifest.json"
    if not bool(args.dry_run):
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("\nDone.")
    print(f"Forecast cache root: {forecast_cache_dir}")
    print(f"Evaluation cache expected by evaluation.py: {_eval_cache_dir(forecast_cache_dir, eval_data)}")
    if not bool(args.dry_run):
        print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main(sys.argv[1:])

"""Build transformed forecast-cache directories for placebo ablations."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "price_short_expert_ann_pred_return",
    "price_short_expert_ann_direction_prob",
    "price_short_expert_ann_direction_margin",
    "price_short_expert_ann_uncertainty",
    "price_short_expert_ann_quality",
]

PRICE_FORECAST_COLS = [
    "price_short_expert_ann",
    "price_forecast_short",
]

OPTIONAL_COLS = [
    "price_short_expert_ann_latent_norm",
    "price_short_expert_ann_latent_0",
    "price_short_expert_ann_latent_1",
    "price_short_expert_ann_latent_2",
    "price_short_expert_ann_latent_3",
]


def forecast_csv_paths(root: Path) -> List[Path]:
    paths = []
    for path in root.rglob("precomputed_forecasts_*.csv"):
        if "_metadata" not in path.name:
            paths.append(path)
    return sorted(paths)


def forecast_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in PRICE_FORECAST_COLS + REQUIRED_COLS + OPTIONAL_COLS if c in df.columns]


def _timestamp_keys(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, errors="coerce")
    return ts.dt.strftime("%Y-%m-%d %H:%M:%S")


def _compute_entry_prices(
    prices: np.ndarray,
    *,
    entry_price_mode: str,
    steps_per_day: int,
) -> np.ndarray:
    clipped = np.clip(
        pd.to_numeric(pd.Series(prices), errors="coerce").to_numpy(dtype=np.float64),
        -1000.0,
        1e9,
    )
    out = np.empty_like(clipped, dtype=np.float64)
    mode = str(entry_price_mode or "current_price").strip().lower()
    steps = max(int(steps_per_day), 1)

    for idx, current in enumerate(clipped):
        if not np.isfinite(current):
            current = 0.0
        if mode == "current_price":
            out[idx] = current
            continue

        prev_idx = idx - steps
        if mode == "same_hour_prev_day" and prev_idx >= 0:
            prev = clipped[prev_idx]
            if np.isfinite(prev):
                out[idx] = float(prev)
                continue

        values = []
        cursor = prev_idx
        while cursor >= 0 and len(values) < 30:
            value = clipped[cursor]
            if np.isfinite(value):
                values.append(float(value))
            cursor -= steps
        out[idx] = float(np.median(np.asarray(values, dtype=np.float64))) if values else float(current)
    return out


def load_price_context(args) -> Optional[Dict[str, object]]:
    if args.price_data is None:
        return None

    price_path = Path(args.price_data).resolve()
    if not price_path.is_file():
        raise SystemExit(f"price_data not found: {price_path}")

    df = pd.read_csv(price_path)
    if args.price_col not in df.columns:
        raise SystemExit(f"price_data missing price column '{args.price_col}': {price_path}")
    if args.timestamp_col not in df.columns:
        raise SystemExit(f"price_data missing timestamp column '{args.timestamp_col}': {price_path}")

    prices = pd.to_numeric(df[args.price_col], errors="coerce").to_numpy(dtype=np.float64)
    entry = _compute_entry_prices(
        prices,
        entry_price_mode=args.entry_price_mode,
        steps_per_day=int(args.steps_per_day),
    )
    current = np.clip(prices, -1000.0, 1e9)
    current[~np.isfinite(current)] = 0.0

    keys = _timestamp_keys(df[args.timestamp_col])
    valid = keys.notna()
    return {
        "price_path": str(price_path),
        "keys": keys,
        "entry": entry,
        "current": current,
        "entry_by_ts": dict(zip(keys[valid], entry[valid.to_numpy()])),
        "current_by_ts": dict(zip(keys[valid], current[valid.to_numpy()])),
    }


def align_price_context(
    df: pd.DataFrame,
    context: Optional[Dict[str, object]],
    *,
    timestamp_col: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if context is None:
        return None, None

    entry = np.asarray(context["entry"], dtype=np.float64)
    current = np.asarray(context["current"], dtype=np.float64)
    if timestamp_col in df.columns:
        keys = _timestamp_keys(df[timestamp_col])
        entry_map = context["entry_by_ts"]
        current_map = context["current_by_ts"]
        aligned_entry = keys.map(entry_map).to_numpy(dtype=np.float64)
        aligned_current = keys.map(current_map).to_numpy(dtype=np.float64)
        missing = ~np.isfinite(aligned_entry) | ~np.isfinite(aligned_current)
        if missing.any() and len(df) == len(entry):
            aligned_entry[missing] = entry[missing]
            aligned_current[missing] = current[missing]
        if np.isfinite(aligned_entry).all() and np.isfinite(aligned_current).all():
            return aligned_entry, aligned_current

    if len(df) == len(entry):
        return entry.copy(), current.copy()

    raise ValueError(
        "could not align forecast cache rows with price_data. "
        "Use matching eval data or include a timestamp column in both files."
    )


def transform_frame(
    df: pd.DataFrame,
    mode: str,
    rng: np.random.Generator,
    lag_steps: int,
    *,
    price_context: Optional[Dict[str, object]],
    timestamp_col: str,
) -> pd.DataFrame:
    out = df.copy()
    cols = forecast_columns(out)
    missing = [c for c in REQUIRED_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"missing required forecast columns: {missing}")

    pred = "price_short_expert_ann_pred_return"
    prob = "price_short_expert_ann_direction_prob"
    margin = "price_short_expert_ann_direction_margin"
    uncertainty = "price_short_expert_ann_uncertainty"
    quality = "price_short_expert_ann_quality"
    price_cols = [c for c in PRICE_FORECAST_COLS if c in out.columns]
    needs_price_alignment = bool(price_cols) and mode in {"sign_flip", "zero_edge"}
    if needs_price_alignment and price_context is None:
        raise ValueError(
            f"mode '{mode}' requires --price_data when price-level forecast columns are present: {price_cols}"
        )
    entry_prices, current_prices = align_price_context(
        out,
        price_context,
        timestamp_col=timestamp_col,
    )

    if mode == "sign_flip":
        out[pred] = -pd.to_numeric(out[pred], errors="coerce").fillna(0.0)
        out[margin] = -pd.to_numeric(out[margin], errors="coerce").fillna(0.0)
        out[prob] = 1.0 - pd.to_numeric(out[prob], errors="coerce").fillna(0.5).clip(0.0, 1.0)
        if price_cols and entry_prices is not None:
            for col in price_cols:
                old_price = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float64)
                missing_price = ~np.isfinite(old_price)
                old_price[missing_price] = entry_prices[missing_price]
                out[col] = (2.0 * entry_prices) - old_price
    elif mode == "shuffle":
        if len(out) > 1:
            perm = rng.permutation(len(out))
            out.loc[:, cols] = out.loc[:, cols].iloc[perm].to_numpy()
    elif mode == "lag":
        shift = max(int(lag_steps), 1)
        out.loc[:, cols] = out.loc[:, cols].shift(shift)
        fill_values = {
            pred: 0.0,
            prob: 0.5,
            margin: 0.0,
            uncertainty: 1.0,
            quality: 0.0,
        }
        for col in OPTIONAL_COLS:
            if col in out.columns:
                fill_values[col] = 0.0
        if entry_prices is not None:
            for col in price_cols:
                out[col] = out[col].fillna(pd.Series(entry_prices, index=out.index))
        out.loc[:, cols] = out.loc[:, cols].fillna(fill_values)
    elif mode == "zero_edge":
        out[pred] = 0.0
        out[prob] = 0.5
        out[margin] = 0.0
        out[uncertainty] = 1.0
        out[quality] = 0.0
        if price_cols and entry_prices is not None:
            for col in price_cols:
                out[col] = entry_prices
        for col in OPTIONAL_COLS:
            if col in out.columns:
                out[col] = 0.0
    else:
        raise ValueError(f"unknown mode: {mode}")

    return out


def copy_or_transform_file(
    src: Path,
    dst: Path,
    args,
    rng: np.random.Generator,
    price_context: Optional[Dict[str, object]],
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".csv" and src.name.startswith("precomputed_forecasts_"):
        df = pd.read_csv(src)
        out = transform_frame(
            df,
            args.mode,
            rng,
            args.lag_steps,
            price_context=price_context,
            timestamp_col=args.timestamp_col,
        )
        out.to_csv(dst, index=False)
        return

    if src.suffix.lower() == ".json" and src.name.endswith("_metadata.json"):
        try:
            meta = json.loads(src.read_text(encoding="utf-8"))
            meta["forecast_cache_ablation"] = {
                "mode": args.mode,
                "source_root": str(args.source),
                "seed": int(args.seed),
                "lag_steps": int(args.lag_steps),
                "price_data": str(args.price_data.resolve()) if args.price_data else None,
                "entry_price_mode": str(args.entry_price_mode),
                "steps_per_day": int(args.steps_per_day),
                "transforms_price_forecast_columns": True,
            }
            dst.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create forecast-cache placebo ablations.")
    parser.add_argument("--source", type=Path, default=Path("forecast_cache"))
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--mode", choices=["sign_flip", "shuffle", "lag", "zero_edge"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lag_steps", type=int, default=288)
    parser.add_argument("--price_data", type=Path, default=None)
    parser.add_argument("--timestamp_col", default="timestamp")
    parser.add_argument("--price_col", default="price")
    parser.add_argument(
        "--entry_price_mode",
        choices=["same_hour_prev_day", "current_price"],
        default="current_price",
    )
    parser.add_argument("--steps_per_day", type=int, default=144)
    args = parser.parse_args()

    source = args.source.resolve()
    dest = args.dest.resolve()
    if not source.is_dir():
        raise SystemExit(f"source cache directory not found: {source}")
    if source == dest or source in dest.parents:
        raise SystemExit("destination must not be the source directory or inside it")

    csvs = forecast_csv_paths(source)
    if not csvs:
        raise SystemExit(f"no forecast CSVs found under {source}")

    rng = np.random.default_rng(int(args.seed))
    price_context = load_price_context(args)
    for src in iter_files(source):
        rel = src.relative_to(source)
        copy_or_transform_file(src, dest / rel, args, rng, price_context)

    print(f"created {args.mode} forecast cache: {dest}")
    print(f"transformed forecast CSVs: {len(csvs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

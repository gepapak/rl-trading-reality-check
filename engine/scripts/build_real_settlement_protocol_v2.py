#!/usr/bin/env python3
"""Build the real-settlement Option 2 protocol datasets.

This script creates aligned settlement-price files for the MTM payoff and
separate forecast-target datasets where ``price`` means the realized settlement
index.  The MARL environment still trains on the original day-ahead/index
datasets; only the financial settlement and FoCAL forecast target use the
realized settlement series.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://api.energidataservice.dk/dataset"
AUTO_DATASETS = ["ImbalancePrice", "RegulatingBalancePowerdata", "RealtimeMarket"]

TIMESTAMP_CANDIDATES = [
    "HourDK",
    "HourUTC",
    "Minutes5DK",
    "Minutes5UTC",
    "Minutes15DK",
    "Minutes15UTC",
    "TimeDK",
    "TimeUTC",
    "timestamp",
]

PRICE_SINGLE_CANDIDATES = [
    "ImbalancePriceDKK",
    "ImbalancePrice",
    "BalancingPowerPriceDKK",
    "BalancePowerPriceDKK",
    "RegulatingPowerPriceDKK",
    "RealtimePriceDKK",
    "ActivationPriceDKK",
    "PriceDKK",
    "Price",
]

CONSUMER_CANDIDATES = [
    "ConsumptionImbalancePriceDKK",
    "ConsumerImbalancePriceDKK",
    "ConsumptionImbalancePrice",
    "ConsumerImbalancePrice",
]

PRODUCER_CANDIDATES = [
    "ProductionImbalancePriceDKK",
    "ProducerImbalancePriceDKK",
    "ProductionImbalancePrice",
    "ProducerImbalancePrice",
]


def _as_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _api_cache_path(
    *,
    cache_dir: str | Path,
    dataset: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    price_area: str,
) -> Path:
    payload = {
        "dataset": str(dataset),
        "start": start.strftime("%Y-%m-%dT%H:%M"),
        "end": end.strftime("%Y-%m-%dT%H:%M"),
        "price_area": str(price_area),
        "timezone": "dk",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    safe_name = (
        f"{str(dataset)}_{str(price_area)}_"
        f"{start.strftime('%Y%m%d%H%M')}_{end.strftime('%Y%m%d%H%M')}_{digest}.json"
    )
    return _as_path(cache_dir) / safe_name


def _parse_timestamps(values: pd.Series, column_name: str = "") -> pd.Series:
    name = str(column_name or "").lower()
    if "utc" in name:
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        try:
            return parsed.dt.tz_convert("Europe/Copenhagen").dt.tz_localize(None)
        except Exception:
            return parsed.dt.tz_localize(None)
    parsed = pd.to_datetime(values, errors="coerce")
    try:
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert("Europe/Copenhagen").dt.tz_localize(None)
    except Exception:
        pass
    return parsed


def _template_timestamps(template: pd.DataFrame, path: Path) -> pd.Series:
    if "timestamp" not in template.columns:
        raise ValueError(f"{path} is missing required column 'timestamp'")
    ts = _parse_timestamps(template["timestamp"], "timestamp")
    bad = int(ts.isna().sum())
    if bad:
        raise ValueError(f"{path} has {bad:,} unparsable timestamps")
    return ts


def _canonical_columns(columns: Iterable[str]) -> dict[str, str]:
    return {str(c).strip().lower(): str(c) for c in columns}


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = _canonical_columns(df.columns)
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def _price_columns_for_side(df: pd.DataFrame, side: str) -> list[str]:
    side = str(side or "average").strip().lower()
    consumer = _first_existing(df, CONSUMER_CANDIDATES)
    producer = _first_existing(df, PRODUCER_CANDIDATES)

    if side in {"consumer", "consumption", "load"} and consumer is not None:
        return [consumer]
    if side in {"producer", "production", "generation"} and producer is not None:
        return [producer]
    if side == "average" and consumer is not None and producer is not None:
        return [consumer, producer]

    single = _first_existing(df, PRICE_SINGLE_CANDIDATES)
    if single is not None:
        return [single]

    # Last-resort heuristic: use a DKK price-like numeric column, but avoid
    # EUR/MW volume fields. If this branch is used, the manifest records it.
    for col in df.columns:
        lower = str(col).lower()
        if "price" in lower and "eur" not in lower and "mwh" not in lower:
            return [str(col)]
    return []


def _extract_settlement_frame(raw: pd.DataFrame, *, dataset: str, price_area: str, side: str) -> pd.DataFrame:
    ts_col = _first_existing(raw, TIMESTAMP_CANDIDATES)
    if ts_col is None:
        raise ValueError(f"{dataset} response has no supported timestamp column. Columns: {list(raw.columns)}")

    price_cols = _price_columns_for_side(raw, side)
    if not price_cols:
        raise ValueError(f"{dataset} response has no supported settlement price column. Columns: {list(raw.columns)}")

    prices = []
    for col in price_cols:
        prices.append(pd.to_numeric(raw[col], errors="coerce"))
    if len(prices) == 1:
        settlement_price = prices[0]
        price_rule = price_cols[0]
    else:
        settlement_price = pd.concat(prices, axis=1).mean(axis=1, skipna=True)
        price_rule = "average(" + ",".join(price_cols) + ")"

    out = pd.DataFrame(
        {
            "timestamp": _parse_timestamps(raw[ts_col], ts_col),
            "settlement_price": settlement_price,
            "settlement_source": str(dataset),
            "settlement_price_area": str(price_area),
            "settlement_price_rule": price_rule,
        }
    ).dropna(subset=["timestamp", "settlement_price"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if out.empty:
        raise ValueError(f"{dataset} returned no usable settlement rows after parsing")
    return out


def _fetch_dataset(
    dataset: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    price_area: str,
    timeout: float,
    retries: int,
    pause_seconds: float,
    cache_dir: str,
    use_cache: bool,
) -> pd.DataFrame:
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M"),
        "end": end.strftime("%Y-%m-%dT%H:%M"),
        "filter": json.dumps({"PriceArea": [str(price_area)]}),
        "timezone": "dk",
        "limit": 0,
    }

    cache_path = _api_cache_path(
        cache_dir=cache_dir,
        dataset=dataset,
        start=start,
        end=end,
        price_area=price_area,
    )
    if use_cache and cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        if not records:
            raise RuntimeError(
                f"{dataset} cached response has no records for {price_area} "
                f"from {start} to {end}: {cache_path}"
            )
        return pd.DataFrame.from_records(records)

    last_error: Exception | None = None
    max_attempts = max(1, int(retries) + 1)
    for attempt in range(max_attempts):
        if attempt > 0:
            wait = max(float(pause_seconds), 0.0) * (2 ** (attempt - 1))
            time.sleep(min(wait, 120.0))
        elif float(pause_seconds) > 0.0:
            time.sleep(float(pause_seconds))

        response = requests.get(f"{API_ROOT}/{dataset}", params=params, timeout=float(timeout))
        try:
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = None
                if retry_after is not None:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = None
                if attempt < max_attempts - 1:
                    time.sleep(min(max(wait or 0.0, float(pause_seconds) * (2 ** attempt)), 180.0))
                    continue
            response.raise_for_status()
            records = response.json().get("records", [])
            break
        except Exception as exc:
            last_error = exc
            status = getattr(response, "status_code", None)
            if status in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                continue
            raise
    else:
        raise RuntimeError(f"{dataset} request failed after {max_attempts} attempts: {last_error}")

    if not records:
        raise RuntimeError(f"{dataset} returned no records for {price_area} from {start} to {end}")
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_payload = {
            "dataset": str(dataset),
            "price_area": str(price_area),
            "start": start.strftime("%Y-%m-%dT%H:%M"),
            "end": end.strftime("%Y-%m-%dT%H:%M"),
            "records": records,
        }
        cache_path.write_text(json.dumps(cache_payload), encoding="utf-8")
    return pd.DataFrame.from_records(records)


def _fetch_settlement_prices(
    *,
    dataset_arg: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    price_area: str,
    side: str,
    timeout: float,
    retries: int,
    pause_seconds: float,
    cache_dir: str,
    use_cache: bool,
) -> pd.DataFrame:
    datasets = AUTO_DATASETS if str(dataset_arg).strip().lower() == "auto" else [str(dataset_arg)]
    errors: list[str] = []
    for dataset in datasets:
        try:
            raw = _fetch_dataset(
                dataset,
                start=start,
                end=end,
                price_area=price_area,
                timeout=timeout,
                retries=int(retries),
                pause_seconds=float(pause_seconds),
                cache_dir=str(cache_dir),
                use_cache=bool(use_cache),
            )
            return _extract_settlement_frame(raw, dataset=dataset, price_area=price_area, side=side)
        except Exception as exc:
            errors.append(f"{dataset}: {exc}")
    raise RuntimeError(
        "Could not fetch usable settlement prices. Tried:\n  "
        + "\n  ".join(errors)
        + "\nUse --settlement_dataset with a specific official dataset if auto detection is wrong."
    )


def _align_settlement_to_template(
    template: pd.DataFrame,
    template_path: Path,
    settlement: pd.DataFrame,
    *,
    tolerance_minutes: float,
) -> pd.DataFrame:
    ts = _template_timestamps(template, template_path)
    source = settlement.copy()
    source["timestamp"] = _parse_timestamps(source["timestamp"], "timestamp")
    source = source.dropna(subset=["timestamp", "settlement_price"]).sort_values("timestamp")
    source = source.groupby("timestamp", as_index=False).last()
    source["_matched_source_timestamp"] = source["timestamp"]

    target = pd.DataFrame({"timestamp": ts, "_row": np.arange(len(template), dtype=np.int64)})
    target = target.sort_values("timestamp")
    aligned = pd.merge_asof(
        target,
        source,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(minutes=float(tolerance_minutes)),
    ).sort_values("_row")

    coverage = float(aligned["settlement_price"].notna().mean())
    missing_rows = int(aligned["settlement_price"].isna().sum())
    if coverage < 1.0:
        raise RuntimeError(
            f"Settlement alignment is incomplete for {template_path}: "
            f"coverage={coverage:.6%}, missing_rows={missing_rows}. "
            "No forward/backward filling is permitted."
        )

    for col in ["settlement_price", "settlement_source", "settlement_price_area", "settlement_price_rule"]:
        if col not in aligned.columns:
            aligned[col] = ""
    aligned["settlement_price"] = pd.to_numeric(aligned["settlement_price"], errors="coerce")

    required = ["settlement_price", "settlement_source", "settlement_price_area", "settlement_price_rule"]
    if aligned[required].isna().any().any():
        missing = {col: int(aligned[col].isna().sum()) for col in required}
        raise RuntimeError(f"Settlement alignment left missing values for {template_path}: {missing}")

    age_minutes = (
        aligned["timestamp"] - aligned["_matched_source_timestamp"]
    ).dt.total_seconds() / 60.0
    if (age_minutes < -1e-9).any():
        raise RuntimeError(f"Settlement alignment used a future source row for {template_path}")
    max_age_minutes = float(age_minutes.max()) if len(age_minutes) else 0.0

    result = pd.DataFrame(
        {
            "timestamp": template["timestamp"].to_numpy(),
            "settlement_price": aligned["settlement_price"].to_numpy(dtype=float),
            "settlement_source": aligned["settlement_source"].astype(str).to_numpy(),
            "settlement_price_area": aligned["settlement_price_area"].astype(str).to_numpy(),
            "settlement_price_rule": aligned["settlement_price_rule"].astype(str).to_numpy(),
        }
    )
    result.attrs["alignment_coverage"] = coverage
    result.attrs["alignment_missing_rows"] = missing_rows
    result.attrs["alignment_max_source_age_minutes"] = max_age_minutes
    return result


def _build_forecast_input(template: pd.DataFrame, settlement: pd.DataFrame) -> pd.DataFrame:
    out = template.copy()
    if "price" not in out.columns:
        raise ValueError("Template must contain price column")
    if "energy_index_price" not in out.columns:
        out["energy_index_price"] = pd.to_numeric(out["price"], errors="coerce")
    out["entry_price_index"] = pd.to_numeric(out["price"], errors="coerce")
    if "settlement_price_raw" in settlement.columns:
        out["settlement_price_raw"] = pd.to_numeric(settlement["settlement_price_raw"], errors="coerce")
    out["settlement_price"] = pd.to_numeric(settlement["settlement_price"], errors="coerce")
    out["settlement_source"] = settlement["settlement_source"].astype(str).to_numpy()
    out["settlement_price_area"] = settlement["settlement_price_area"].astype(str).to_numpy()
    out["settlement_price_rule"] = settlement["settlement_price_rule"].astype(str).to_numpy()
    out["price"] = out["settlement_price"]
    return out


def _settlement_clip_bounds(args: argparse.Namespace) -> tuple[float, float]:
    clip_min = float(getattr(args, "settlement_clip_min", -111750.0))
    clip_max = float(getattr(args, "settlement_clip_max", 111750.0))
    if clip_min >= clip_max:
        raise ValueError("--settlement_clip_min must be smaller than --settlement_clip_max")
    return clip_min, clip_max


def _apply_settlement_clip(aligned: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use the same settlement-price support for forecast inputs and PnL."""
    clip_min, clip_max = _settlement_clip_bounds(args)
    out = aligned.copy()
    raw = pd.to_numeric(out["settlement_price"], errors="coerce").astype(float)
    clipped = raw.clip(lower=clip_min, upper=clip_max)
    raw_np = raw.to_numpy(dtype=float)
    clipped_np = clipped.to_numpy(dtype=float)
    out["settlement_price_raw"] = raw_np
    out["settlement_price"] = clipped_np
    finite_raw = raw[np.isfinite(raw)]
    finite_clipped = clipped[np.isfinite(clipped)]
    clipped_mask = np.isfinite(raw_np) & (np.abs(raw_np - clipped_np) > 1e-12)
    stats = {
        "settlement_clip_min": clip_min,
        "settlement_clip_max": clip_max,
        "raw_settlement_price_min": float(finite_raw.min()) if len(finite_raw) else None,
        "raw_settlement_price_max": float(finite_raw.max()) if len(finite_raw) else None,
        "clipped_settlement_price_min": float(finite_clipped.min()) if len(finite_clipped) else None,
        "clipped_settlement_price_max": float(finite_clipped.max()) if len(finite_clipped) else None,
        "settlement_clip_fraction": float(np.mean(clipped_mask)) if len(raw) else 0.0,
        "settlement_clip_count": int(np.sum(clipped_mask)),
    }
    return out, stats


def _hourly_forecast_rows(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    ts = _template_timestamps(df, path)
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
    return out


def _fetch_for_template(
    template: pd.DataFrame,
    path: Path,
    *,
    price_area: str,
    args: argparse.Namespace,
    dataset_arg: str | None = None,
) -> pd.DataFrame:
    ts = _template_timestamps(template, path)
    start = ts.min().floor("h") - pd.Timedelta(hours=2)
    end = ts.max().ceil("h") + pd.Timedelta(hours=3)
    return _fetch_settlement_prices(
        dataset_arg=str(dataset_arg if dataset_arg is not None else args.settlement_dataset),
        start=start,
        end=end,
        price_area=str(price_area),
        side=str(args.imbalance_side),
        timeout=float(args.api_timeout),
        retries=int(args.api_retries),
        pause_seconds=float(args.api_pause_seconds),
        cache_dir=str(args.api_cache_dir),
        use_cache=not bool(args.no_api_cache),
    )


def _fetch_and_align_settlement_to_template(
    template: pd.DataFrame,
    path: Path,
    *,
    price_area: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    dataset_arg = str(args.settlement_dataset).strip()
    datasets = AUTO_DATASETS if dataset_arg.lower() == "auto" else [dataset_arg]
    errors: list[str] = []
    extracted_frames: list[pd.DataFrame] = []
    for dataset in datasets:
        try:
            fetched = _fetch_for_template(
                template,
                path,
                price_area=price_area,
                args=args,
                dataset_arg=dataset,
            )
            extracted = fetched.copy()
            extracted["_source_priority"] = int(len(extracted_frames))
            extracted_frames.append(extracted)
            return _align_settlement_to_template(
                template,
                path,
                extracted.drop(columns=["_source_priority"], errors="ignore"),
                tolerance_minutes=float(args.alignment_tolerance_minutes),
            )
        except Exception as exc:
            errors.append(f"{dataset}: {exc}")
    if dataset_arg.lower() == "auto" and extracted_frames:
        try:
            combined = pd.concat(extracted_frames, ignore_index=True)
            combined["timestamp"] = _parse_timestamps(combined["timestamp"], "timestamp")
            combined = combined.dropna(subset=["timestamp", "settlement_price"])
            # Prefer earlier AUTO_DATASETS entries on exact timestamp overlap. In
            # practice this uses ImbalancePrice when available and falls back to
            # RegulatingBalancePowerdata for the early-2025 gap.
            combined = combined.sort_values(
                ["timestamp", "_source_priority"],
                ascending=[True, True],
                kind="mergesort",
            ).drop_duplicates("timestamp", keep="first")
            combined["settlement_source"] = "auto_composite:" + combined["settlement_source"].astype(str)
            return _align_settlement_to_template(
                template,
                path,
                combined.drop(columns=["_source_priority"], errors="ignore"),
                tolerance_minutes=float(args.alignment_tolerance_minutes),
            )
        except Exception as exc:
            errors.append(f"auto_composite: {exc}")
    raise RuntimeError(
        f"Could not fetch and align settlement prices for {path}. Tried:\n  "
        + "\n  ".join(errors)
        + "\nUse --settlement_dataset with a specific official dataset if auto detection is wrong."
    )


def _write_one_template(
    *,
    source_path: Path,
    settlement_dest: Path | None,
    forecast_input_dest: Path | None,
    price_area: str,
    args: argparse.Namespace,
    hourly_forecast_dest: Path | None = None,
) -> dict[str, Any]:
    template = pd.read_csv(source_path)
    requested_outputs = [
        path
        for path in (settlement_dest, forecast_input_dest, hourly_forecast_dest)
        if path is not None
    ]
    if (
        requested_outputs
        and bool(getattr(args, "skip_existing", False))
        and all(path.is_file() for path in requested_outputs)
    ):
        ts = _template_timestamps(template, source_path)
        clip_min, clip_max = _settlement_clip_bounds(args)
        return {
            "source": str(source_path),
            "settlement_dest": str(settlement_dest) if settlement_dest is not None else "",
            "forecast_input_dest": str(forecast_input_dest) if forecast_input_dest is not None else "",
            "hourly_forecast_dest": str(hourly_forecast_dest) if hourly_forecast_dest is not None else "",
            "rows": int(len(template)),
            "hourly_rows": None,
            "price_area": str(price_area),
            "settlement_source": "skipped_existing",
            "settlement_price_std": None,
            "tracking_error_std_vs_entry_price": None,
            "settlement_clip_min": clip_min,
            "settlement_clip_max": clip_max,
            "settlement_clip_fraction": None,
            "settlement_clip_count": None,
            "start": str(ts.min()),
            "end": str(ts.max()),
            "status": "skipped_existing",
        }
    aligned = _fetch_and_align_settlement_to_template(
        template,
        source_path,
        price_area=price_area,
        args=args,
    )
    alignment_stats = {
        "alignment_coverage": float(aligned.attrs.get("alignment_coverage", 1.0)),
        "alignment_missing_rows": int(aligned.attrs.get("alignment_missing_rows", 0)),
        "alignment_max_source_age_minutes": float(
            aligned.attrs.get("alignment_max_source_age_minutes", 0.0)
        ),
    }
    aligned, clip_stats = _apply_settlement_clip(aligned, args)
    forecast_input = _build_forecast_input(template, aligned)

    if not bool(args.dry_run):
        if settlement_dest is not None:
            settlement_dest.parent.mkdir(parents=True, exist_ok=True)
            if settlement_dest.exists() and bool(getattr(args, "skip_existing", False)):
                pass
            elif settlement_dest.exists() and not bool(args.overwrite):
                raise FileExistsError(f"Refusing to overwrite {settlement_dest}; use --overwrite")
            else:
                aligned.to_csv(settlement_dest, index=False)
        if forecast_input_dest is not None:
            forecast_input_dest.parent.mkdir(parents=True, exist_ok=True)
            if forecast_input_dest.exists() and bool(getattr(args, "skip_existing", False)):
                pass
            elif forecast_input_dest.exists() and not bool(args.overwrite):
                raise FileExistsError(f"Refusing to overwrite {forecast_input_dest}; use --overwrite")
            else:
                forecast_input.to_csv(forecast_input_dest, index=False)
        if hourly_forecast_dest is not None:
            hourly = _hourly_forecast_rows(forecast_input, source_path)
            hourly_forecast_dest.parent.mkdir(parents=True, exist_ok=True)
            if hourly_forecast_dest.exists() and bool(getattr(args, "skip_existing", False)):
                pass
            elif hourly_forecast_dest.exists() and not bool(args.overwrite):
                raise FileExistsError(f"Refusing to overwrite {hourly_forecast_dest}; use --overwrite")
            else:
                hourly.to_csv(hourly_forecast_dest, index=False)
    else:
        hourly = _hourly_forecast_rows(forecast_input, source_path) if hourly_forecast_dest is not None else None

    settlement_std = float(np.nanstd(aligned["settlement_price"].to_numpy(dtype=float)))
    basis_std = float(
        np.nanstd(
            aligned["settlement_price"].to_numpy(dtype=float)
            - pd.to_numeric(template["price"], errors="coerce").to_numpy(dtype=float)
        )
    )
    return {
        "source": str(source_path),
        "settlement_dest": str(settlement_dest) if settlement_dest is not None else "",
        "forecast_input_dest": str(forecast_input_dest) if forecast_input_dest is not None else "",
        "hourly_forecast_dest": str(hourly_forecast_dest) if hourly_forecast_dest is not None else "",
        "rows": int(len(template)),
        "hourly_rows": int(len(hourly)) if hourly_forecast_dest is not None else None,
        "price_area": str(price_area),
        "settlement_source": str(aligned["settlement_source"].iloc[0]) if len(aligned) else "",
        "settlement_price_std": settlement_std,
        "tracking_error_std_vs_entry_price": basis_std,
        **alignment_stats,
        **clip_stats,
        "start": str(_template_timestamps(template, source_path).min()),
        "end": str(_template_timestamps(template, source_path).max()),
    }


def _iter_training_files(source_dir: Path, pattern: str) -> list[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Training template directory not found: {source_dir}")
    files = sorted(source_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern!r} under {source_dir}")
    return files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settlement_dataset", default="auto")
    parser.add_argument(
        "--imbalance_side",
        choices=["average", "consumer", "producer", "consumption", "production"],
        default="average",
        help="Which imbalance side to use when both consumption and production prices are available.",
    )
    parser.add_argument("--training_template_dir", default="training_dataset_ffill")
    parser.add_argument("--training_pattern", default="scenario_*.csv")
    parser.add_argument("--training_price_area", default="DK1")
    parser.add_argument("--settlement_dest_dir", default="settlement_price_dataset_real_v2")
    parser.add_argument("--forecast_cache_input_dest_dir", default="forecast_cache_input_settlement_v2")
    parser.add_argument("--forecast_training_source_dir", default="forecast_training_dataset_ffill")
    parser.add_argument("--forecast_training_dest_dir", default="forecast_training_dataset_settlement_hourly_v2")
    parser.add_argument("--forecast_training_pattern", default="forecast_scenario_*.csv")
    parser.add_argument("--forecast_training_price_area", default="DK1")
    parser.add_argument("--eval_original_template", default="evaluation_dataset_ffill/unseendata.csv")
    parser.add_argument("--eval_original_settlement_dest", default="evaluation_dataset_ffill/unseendata_settlement_real_v2.csv")
    parser.add_argument("--eval_original_forecast_input_dest", default="forecast_cache_input_settlement_v2/unseendata.csv")
    parser.add_argument("--eval_original_price_area", default="DK1")
    parser.add_argument("--eval_v2_template", default="evaluation_dataset_ffill/unseendata_v2.csv")
    parser.add_argument("--eval_v2_settlement_dest", default="evaluation_dataset_ffill/unseendata_v2_settlement_real_v2.csv")
    parser.add_argument("--eval_v2_forecast_input_dest", default="forecast_cache_input_settlement_v2/unseendata_v2.csv")
    parser.add_argument("--eval_v2_price_area", default="DK2")
    parser.add_argument("--alignment_tolerance_minutes", type=float, default=75.0)
    parser.add_argument(
        "--settlement_clip_min",
        type=float,
        default=-111750.0,
        help="Lower bound for the settlement price process used in PnL and forecast targets.",
    )
    parser.add_argument(
        "--settlement_clip_max",
        type=float,
        default=111750.0,
        help="Upper bound for the settlement price process used in PnL and forecast targets.",
    )
    parser.add_argument("--api_timeout", type=float, default=90.0)
    parser.add_argument("--api_retries", type=int, default=8)
    parser.add_argument("--api_pause_seconds", type=float, default=3.0)
    parser.add_argument("--api_cache_dir", default=".cache/energidataservice_settlement_v2")
    parser.add_argument("--no_api_cache", action="store_true")
    parser.add_argument("--skip_training_settlement", action="store_true")
    parser.add_argument("--skip_forecast_training", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_eval_v2", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "project_root": str(PROJECT_ROOT),
        "settlement_dataset": str(args.settlement_dataset),
        "imbalance_side": str(args.imbalance_side),
        "alignment_tolerance_minutes": float(args.alignment_tolerance_minutes),
        "settlement_clip_min": float(args.settlement_clip_min),
        "settlement_clip_max": float(args.settlement_clip_max),
        "api_retries": int(args.api_retries),
        "api_pause_seconds": float(args.api_pause_seconds),
        "api_cache_dir": str(args.api_cache_dir),
        "api_cache_enabled": not bool(args.no_api_cache),
        "skip_existing": bool(args.skip_existing),
        "training_settlement": [],
        "forecast_training": [],
        "evaluation": [],
    }

    if not bool(args.skip_training_settlement):
        training_dir = _as_path(args.training_template_dir)
        settlement_dest_dir = _as_path(args.settlement_dest_dir)
        forecast_input_dest_dir = _as_path(args.forecast_cache_input_dest_dir)
        for source in _iter_training_files(training_dir, str(args.training_pattern)):
            entry = _write_one_template(
                source_path=source,
                settlement_dest=settlement_dest_dir / source.name,
                forecast_input_dest=forecast_input_dest_dir / source.name,
                price_area=str(args.training_price_area),
                args=args,
            )
            manifest["training_settlement"].append(entry)
            print(
                f"[OK] env {source.name}: rows={entry['rows']:,} "
                f"settlement={entry['settlement_source']} area={entry['price_area']}"
            )

    if not bool(args.skip_forecast_training):
        forecast_source_dir = _as_path(args.forecast_training_source_dir)
        forecast_dest_dir = _as_path(args.forecast_training_dest_dir)
        for source in _iter_training_files(forecast_source_dir, str(args.forecast_training_pattern)):
            entry = _write_one_template(
                source_path=source,
                settlement_dest=None,
                forecast_input_dest=None,
                hourly_forecast_dest=forecast_dest_dir / source.name,
                price_area=str(args.forecast_training_price_area),
                args=args,
            )
            manifest["forecast_training"].append(entry)
            hourly_rows_label = (
                f"{int(entry['hourly_rows']):,}"
                if entry.get("hourly_rows") is not None
                else "skipped_existing"
            )
            print(
                f"[OK] forecast {source.name}: rows={entry['rows']:,} -> "
                f"hourly={hourly_rows_label} settlement={entry['settlement_source']}"
            )

    if not bool(args.skip_eval):
        eval_jobs = [
            (
                "original",
                _as_path(args.eval_original_template),
                _as_path(args.eval_original_settlement_dest),
                _as_path(args.eval_original_forecast_input_dest),
                str(args.eval_original_price_area),
            )
        ]
        if not bool(args.skip_eval_v2):
            eval_jobs.append(
                (
                    "unseendata_v2",
                    _as_path(args.eval_v2_template),
                    _as_path(args.eval_v2_settlement_dest),
                    _as_path(args.eval_v2_forecast_input_dest),
                    str(args.eval_v2_price_area),
                )
            )
        for label, template, settlement_dest, forecast_input_dest, area in eval_jobs:
            entry = _write_one_template(
                source_path=template,
                settlement_dest=settlement_dest,
                forecast_input_dest=forecast_input_dest,
                price_area=area,
                args=args,
            )
            entry["label"] = label
            manifest["evaluation"].append(entry)
            print(
                f"[OK] eval {label}: rows={entry['rows']:,} "
                f"settlement={entry['settlement_source']} area={entry['price_area']}"
            )

    manifest_path = _as_path(args.settlement_dest_dir).with_name("real_settlement_protocol_v2_manifest.json")
    if not bool(args.dry_run):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nManifest: {manifest_path}")
    else:
        print("\nDry run complete; no files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Build official real-liquidity volume files for the same-delivery protocol.

The current financial simulator can consume an external hourly MWh
volume series for market-impact scaling and hard participation caps. This
script builds that series from Energinet/Energi Data Service
``RegulatingBalancePowerdata`` records, aligned onto the existing 10-minute
training/evaluation templates.

The default is the sum of observed balancing-activation energy. Historical
``RegulatingBalancePowerdata`` rows use mFRR MWh fields. After that dataset's
2025 transition, ``ImbalancePrice`` rows use available aFRR MW fields converted
to MWh. The source columns and exact formula are persisted on every row.

``ImbalanceMWh`` is preserved as a diagnostic column but is not the default
market-depth proxy. Activation volume is a conservative participation proxy,
not an order-book depth measurement.
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
DEFAULT_DATASET = "auto"
AUTO_DATASETS = ["RegulatingBalancePowerdata", "ImbalancePrice"]

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

UP_BAL_CANDIDATES = ["mFRRUpActBal", "UpActBal", "UpActivatedBalancingMWh"]
DOWN_BAL_CANDIDATES = ["mFRRDownActBal", "DownActBal", "DownActivatedBalancingMWh"]
UP_SPEC_CANDIDATES = ["mFRRUpActSpec", "UpActSpec", "UpActivatedSpecialMWh"]
DOWN_SPEC_CANDIDATES = ["mFRRDownActSpec", "DownActSpec", "DownActivatedSpecialMWh"]
IMBALANCE_CANDIDATES = ["ImbalanceMWh", "SystemImbalanceMWh", "TotalImbalanceMWh"]
UP_MW_CANDIDATES = ["aFRRUpMW", "mFRRUpMW", "UpMW"]
DOWN_MW_CANDIDATES = ["aFRRDownMW", "mFRRDownMW", "DownMW"]
SATISFIED_DEMAND_CANDIDATES = ["SatisfiedDemand"]


def _as_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _canonical_columns(columns: Iterable[str]) -> dict[str, str]:
    return {str(c).strip().lower(): str(c) for c in columns}


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = _canonical_columns(df.columns)
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in lookup:
            return lookup[key]
    return None


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
        "purpose": "liquidity_volume_v1",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    safe_name = (
        f"{str(dataset)}_{str(price_area)}_"
        f"{start.strftime('%Y%m%d%H%M')}_{end.strftime('%Y%m%d%H%M')}_{digest}.json"
    )
    return _as_path(cache_dir) / safe_name


def _fetch_dataset(
    *,
    dataset: str,
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
        payload = json.loads(cache_path.read_text(encoding="utf-8-sig"))
        records = payload.get("records", [])
        if not records:
            raise RuntimeError(f"Cached response has no records: {cache_path}")
        return pd.DataFrame.from_records(records)

    last_error: Exception | None = None
    max_attempts = max(1, int(retries) + 1)
    records: list[dict[str, Any]] = []
    for attempt in range(max_attempts):
        if attempt > 0:
            wait = max(float(pause_seconds), 0.0) * (2 ** (attempt - 1))
            time.sleep(min(wait, 180.0))
        elif float(pause_seconds) > 0.0:
            time.sleep(float(pause_seconds))

        response = requests.get(f"{API_ROOT}/{dataset}", params=params, timeout=float(timeout))
        try:
            if response.status_code == 429 and attempt < max_attempts - 1:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_429 = float(retry_after) if retry_after is not None else None
                except ValueError:
                    wait_429 = None
                time.sleep(min(max(wait_429 or 0.0, float(pause_seconds) * (2 ** attempt)), 300.0))
                continue
            response.raise_for_status()
            records = response.json().get("records", [])
            break
        except Exception as exc:
            last_error = exc
            status = getattr(response, "status_code", None)
            if status == 429 and attempt < max_attempts - 1:
                continue
            if attempt >= max_attempts - 1:
                raise

    if not records:
        if last_error is not None:
            raise RuntimeError(f"{dataset} returned no records after retries: {last_error}") from last_error
        raise RuntimeError(f"{dataset} returned no records for {price_area} from {start} to {end}")

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "start": start.strftime("%Y-%m-%dT%H:%M"),
                    "end": end.strftime("%Y-%m-%dT%H:%M"),
                    "price_area": price_area,
                    "records": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return pd.DataFrame.from_records(records)


def _numeric_or_zero(raw: pd.DataFrame, candidates: Iterable[str]) -> tuple[pd.Series, str, bool]:
    col = _first_existing(raw, candidates)
    if col is None:
        return pd.Series(0.0, index=raw.index, dtype=float), "", False
    return pd.to_numeric(raw[col], errors="coerce").fillna(0.0), col, True


def _extract_volume_frame(
    raw: pd.DataFrame,
    *,
    dataset: str,
    price_area: str,
    volume_mode: str,
) -> pd.DataFrame:
    ts_col = _first_existing(raw, TIMESTAMP_CANDIDATES)
    if ts_col is None:
        raise ValueError(f"{dataset} response has no supported timestamp column. Columns: {list(raw.columns)}")

    parsed_ts = _parse_timestamps(raw[ts_col], ts_col)
    diffs = parsed_ts.dropna().sort_values().diff().dropna().dt.total_seconds() / 3600.0
    median_step_hours = float(diffs[diffs > 0.0].median()) if not diffs.empty else 1.0
    interval_hours = float(np.clip(median_step_hours if np.isfinite(median_step_hours) else 1.0, 1.0 / 12.0, 1.0))

    up_bal, up_bal_col, up_bal_ok = _numeric_or_zero(raw, UP_BAL_CANDIDATES)
    down_bal, down_bal_col, down_bal_ok = _numeric_or_zero(raw, DOWN_BAL_CANDIDATES)
    up_spec, up_spec_col, up_spec_ok = _numeric_or_zero(raw, UP_SPEC_CANDIDATES)
    down_spec, down_spec_col, down_spec_ok = _numeric_or_zero(raw, DOWN_SPEC_CANDIDATES)
    up_mw, up_mw_col, up_mw_ok = _numeric_or_zero(raw, UP_MW_CANDIDATES)
    down_mw, down_mw_col, down_mw_ok = _numeric_or_zero(raw, DOWN_MW_CANDIDATES)
    imbalance, imbalance_col, imbalance_ok = _numeric_or_zero(raw, IMBALANCE_CANDIDATES)
    satisfied_demand, satisfied_demand_col, satisfied_demand_ok = _numeric_or_zero(raw, SATISFIED_DEMAND_CANDIDATES)

    if not any([up_bal_ok, down_bal_ok, up_spec_ok, down_spec_ok, up_mw_ok, down_mw_ok, imbalance_ok, satisfied_demand_ok]):
        raise ValueError(f"{dataset} response has no supported liquidity volume columns. Columns: {list(raw.columns)}")

    up_activation = up_bal.abs() + up_spec.abs() + up_mw.abs() * interval_hours
    down_activation = down_bal.abs() + down_spec.abs() + down_mw.abs() * interval_hours
    total_activation = up_activation + down_activation
    imbalance_abs = imbalance.abs()
    if not imbalance_ok and satisfied_demand_ok:
        imbalance_abs = satisfied_demand.abs() * interval_hours

    up_activation_terms: list[str] = []
    down_activation_terms: list[str] = []
    for terms, available, column in (
        (up_activation_terms, up_bal_ok, up_bal_col),
        (down_activation_terms, down_bal_ok, down_bal_col),
        (up_activation_terms, up_spec_ok, up_spec_col),
        (down_activation_terms, down_spec_ok, down_spec_col),
    ):
        if available and column:
            terms.append(f"abs({column})")
    for terms, available, column in (
        (up_activation_terms, up_mw_ok, up_mw_col),
        (down_activation_terms, down_mw_ok, down_mw_col),
    ):
        if available and column:
            terms.append(f"abs({column})*{interval_hours:g}h")
    up_activation_rule = "+".join(up_activation_terms) or "0"
    down_activation_rule = "+".join(down_activation_terms) or "0"
    activation_rule = f"({up_activation_rule})+({down_activation_rule})"
    imbalance_rule = (
        f"abs({imbalance_col})"
        if imbalance_ok and imbalance_col
        else (
            f"abs({satisfied_demand_col})*{interval_hours:g}h"
            if satisfied_demand_ok and satisfied_demand_col
            else "0"
        )
    )

    mode = str(volume_mode or "activation_sum").strip().lower()
    if mode == "activation_sum":
        market_volume = total_activation
        rule = activation_rule
    elif mode == "max_activation_side":
        market_volume = pd.concat([up_activation, down_activation], axis=1).max(axis=1)
        rule = f"max(({up_activation_rule}),({down_activation_rule}))"
    elif mode == "activation_plus_imbalance":
        market_volume = total_activation + imbalance_abs
        rule = f"({activation_rule})+({imbalance_rule})"
    elif mode == "max_activation_or_imbalance":
        market_volume = pd.concat([total_activation, imbalance_abs], axis=1).max(axis=1)
        rule = f"max(({activation_rule}),({imbalance_rule}))"
    elif mode == "imbalance_abs":
        market_volume = imbalance_abs
        rule = imbalance_rule
    else:
        raise ValueError(f"Unsupported volume_mode={volume_mode!r}")

    out = pd.DataFrame(
        {
            "timestamp": parsed_ts,
            "market_volume_mwh": pd.to_numeric(market_volume, errors="coerce"),
            "up_activation_volume_mwh": pd.to_numeric(up_activation, errors="coerce"),
            "down_activation_volume_mwh": pd.to_numeric(down_activation, errors="coerce"),
            "total_activation_volume_mwh": pd.to_numeric(total_activation, errors="coerce"),
            "imbalance_volume_mwh": pd.to_numeric(imbalance_abs, errors="coerce"),
            "liquidity_source": str(dataset),
            "liquidity_price_area": str(price_area),
            "liquidity_volume_rule": rule,
            "source_up_bal_column": up_bal_col,
            "source_down_bal_column": down_bal_col,
            "source_up_spec_column": up_spec_col,
            "source_down_spec_column": down_spec_col,
            "source_up_mw_column": up_mw_col,
            "source_down_mw_column": down_mw_col,
            "source_imbalance_column": imbalance_col,
            "source_satisfied_demand_column": satisfied_demand_col,
            "source_interval_hours": interval_hours,
        }
    ).dropna(subset=["timestamp", "market_volume_mwh"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if out.empty:
        raise ValueError(f"{dataset} returned no usable liquidity rows after parsing")
    for col in [
        "market_volume_mwh",
        "up_activation_volume_mwh",
        "down_activation_volume_mwh",
        "total_activation_volume_mwh",
        "imbalance_volume_mwh",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["_hour"] = out["timestamp"].dt.floor("h")
    numeric_cols = [
        "market_volume_mwh",
        "up_activation_volume_mwh",
        "down_activation_volume_mwh",
        "total_activation_volume_mwh",
        "imbalance_volume_mwh",
    ]
    text_cols = [c for c in out.columns if c not in numeric_cols and c not in {"timestamp", "_hour"}]
    grouped = out.groupby("_hour", as_index=False)[numeric_cols].sum()
    text = out.sort_values("timestamp").groupby("_hour", as_index=False)[text_cols].first()
    hourly = grouped.merge(text, on="_hour", how="left")
    hourly = hourly.rename(columns={"_hour": "timestamp"}).sort_values("timestamp")
    return hourly


def _fetch_for_template(template: pd.DataFrame, path: Path, *, price_area: str, args: argparse.Namespace) -> pd.DataFrame:
    ts = _template_timestamps(template, path)
    start = ts.min().floor("h") - pd.Timedelta(hours=2)
    end = ts.max().ceil("h") + pd.Timedelta(hours=3)
    dataset_arg = str(args.dataset).strip()
    datasets = AUTO_DATASETS if dataset_arg.lower() == "auto" else [dataset_arg]
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for priority, dataset in enumerate(datasets):
        try:
            raw = _fetch_dataset(
                dataset=str(dataset),
                start=start,
                end=end,
                price_area=str(price_area),
                timeout=float(args.api_timeout),
                retries=int(args.api_retries),
                pause_seconds=float(args.api_pause_seconds),
                cache_dir=str(args.api_cache_dir),
                use_cache=not bool(args.no_api_cache),
            )
            extracted = _extract_volume_frame(
                raw,
                dataset=str(dataset),
                price_area=str(price_area),
                volume_mode=str(args.volume_mode),
            )
            extracted["_source_priority"] = int(priority)
            frames.append(extracted)
        except Exception as exc:
            errors.append(f"{dataset}: {exc}")
            if dataset_arg.lower() != "auto":
                raise
    if not frames:
        raise RuntimeError("Could not fetch usable liquidity volume data. Tried:\n  " + "\n  ".join(errors))
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = _parse_timestamps(combined["timestamp"], "timestamp")
    combined = combined.dropna(subset=["timestamp", "market_volume_mwh"])
    combined = combined.sort_values(["timestamp", "_source_priority"], ascending=[True, True], kind="mergesort")
    combined = combined.drop_duplicates("timestamp", keep="first")
    if dataset_arg.lower() == "auto":
        combined["liquidity_source"] = "auto_composite:" + combined["liquidity_source"].astype(str)
    return combined.drop(columns=["_source_priority"], errors="ignore")


def _align_volume_to_template(
    template: pd.DataFrame,
    path: Path,
    volume: pd.DataFrame,
    *,
    tolerance_minutes: float,
    min_coverage_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_ts = _template_timestamps(template, path)
    target = pd.DataFrame({"_row": np.arange(len(template), dtype=np.int64), "timestamp": target_ts})
    source = volume.copy()
    source["timestamp"] = _parse_timestamps(source["timestamp"], "timestamp")
    source = source.dropna(subset=["timestamp"]).sort_values("timestamp")
    target_sorted = target.sort_values("timestamp")
    aligned = pd.merge_asof(
        target_sorted,
        source,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(minutes=float(tolerance_minutes)),
    ).sort_values("_row")

    missing = int(aligned["market_volume_mwh"].isna().sum())
    coverage = float(1.0 - missing / max(len(aligned), 1))
    if coverage < float(min_coverage_fraction):
        raise RuntimeError(
            f"Liquidity volume coverage too low for {path}: {coverage:.2%}; "
            f"missing rows={missing:,}, required={float(min_coverage_fraction):.2%}"
        )
    if missing:
        raise RuntimeError(
            f"Liquidity volume has {missing:,} missing/stale rows for {path}. "
            "Increase --alignment_tolerance_minutes only if this is justified."
        )

    out = aligned.drop(columns=["_row"]).copy()
    out["timestamp"] = target_ts.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    numeric_cols = [
        "market_volume_mwh",
        "up_activation_volume_mwh",
        "down_activation_volume_mwh",
        "total_activation_volume_mwh",
        "imbalance_volume_mwh",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    values = out["market_volume_mwh"].to_numpy(dtype=float)
    stats = {
        "coverage_fraction": coverage,
        "missing_rows": missing,
        "market_volume_min_mwh": float(np.nanmin(values)) if values.size else 0.0,
        "market_volume_p05_mwh": float(np.nanpercentile(values, 5)) if values.size else 0.0,
        "market_volume_median_mwh": float(np.nanmedian(values)) if values.size else 0.0,
        "market_volume_mean_mwh": float(np.nanmean(values)) if values.size else 0.0,
        "market_volume_p95_mwh": float(np.nanpercentile(values, 95)) if values.size else 0.0,
        "market_volume_max_mwh": float(np.nanmax(values)) if values.size else 0.0,
        "market_volume_zero_fraction": float(np.mean(values <= 0.0)) if values.size else 0.0,
    }
    return out, stats


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_one_template(
    *,
    source_path: Path,
    dest_path: Path,
    price_area: str,
    label: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    template = pd.read_csv(source_path)
    if bool(args.skip_existing) and dest_path.is_file():
        meta_path = dest_path.with_suffix(dest_path.suffix + ".metadata.json")
        if meta_path.is_file():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8-sig"))
                metadata["status"] = "skipped_existing"
                return metadata
            except Exception:
                pass
        ts = _template_timestamps(template, source_path)
        return {
            "label": label,
            "source": str(source_path),
            "dest": str(dest_path),
            "rows": int(len(template)),
            "price_area": str(price_area),
            "start": str(ts.min()),
            "end": str(ts.max()),
            "status": "skipped_existing",
        }

    volume = _fetch_for_template(template, source_path, price_area=str(price_area), args=args)
    aligned, stats = _align_volume_to_template(
        template,
        source_path,
        volume,
        tolerance_minutes=float(args.alignment_tolerance_minutes),
        min_coverage_fraction=float(args.min_coverage_fraction),
    )

    ts = _template_timestamps(template, source_path)
    metadata = {
        "label": label,
        "source": str(source_path),
        "dest": str(dest_path),
        "rows": int(len(aligned)),
        "price_area": str(price_area),
        "dataset": str(args.dataset),
        "volume_mode": str(args.volume_mode),
        "alignment_tolerance_minutes": float(args.alignment_tolerance_minutes),
        "start": str(ts.min()),
        "end": str(ts.max()),
        "columns": list(aligned.columns),
        **stats,
    }
    if not bool(args.dry_run):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists() and not bool(args.overwrite):
            raise FileExistsError(f"Refusing to overwrite {dest_path}; use --overwrite")
        aligned.to_csv(dest_path, index=False)
        _write_metadata(dest_path, metadata)
    return metadata


def _iter_training_files(source_dir: Path, pattern: str) -> list[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Training template directory not found: {source_dir}")
    files = sorted(source_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern!r} under {source_dir}")
    return files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--volume_mode",
        choices=[
            "activation_sum",
            "max_activation_side",
            "activation_plus_imbalance",
            "max_activation_or_imbalance",
            "imbalance_abs",
        ],
        default="activation_sum",
    )
    parser.add_argument("--training_template_dir", default="training_dataset_ffill")
    parser.add_argument("--training_pattern", default="scenario_*.csv")
    parser.add_argument("--training_price_area", default="DK1")
    parser.add_argument("--liquidity_dest_dir", default="liquidity_volume_dataset_real_v1")
    parser.add_argument("--eval_original_template", default="evaluation_dataset_ffill/unseendata.csv")
    parser.add_argument("--eval_original_dest", default="evaluation_dataset_ffill/unseendata_liquidity_volume_real_v1.csv")
    parser.add_argument("--eval_original_price_area", default="DK1")
    parser.add_argument("--eval_v2_template", default="evaluation_dataset_ffill/unseendata_v2.csv")
    parser.add_argument("--eval_v2_dest", default="evaluation_dataset_ffill/unseendata_v2_liquidity_volume_real_v1.csv")
    parser.add_argument("--eval_v2_price_area", default="DK2")
    parser.add_argument(
        "--alignment_tolerance_minutes",
        type=float,
        default=180.0,
        help=(
            "Builder-only alignment tolerance. The wider default handles Danish "
            "DST spring-forward gaps in historical local-hour records; generated "
            "10-minute runtime files still align exactly afterward."
        ),
    )
    parser.add_argument("--min_coverage_fraction", type=float, default=0.999)
    parser.add_argument("--api_timeout", type=float, default=90.0)
    parser.add_argument("--api_retries", type=int, default=12)
    parser.add_argument("--api_pause_seconds", type=float, default=5.0)
    parser.add_argument("--api_cache_dir", default=".cache/energidataservice_liquidity_v1")
    parser.add_argument("--no_api_cache", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
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
        "dataset": str(args.dataset),
        "volume_mode": str(args.volume_mode),
        "alignment_tolerance_minutes": float(args.alignment_tolerance_minutes),
        "min_coverage_fraction": float(args.min_coverage_fraction),
        "api_cache_dir": str(args.api_cache_dir),
        "api_cache_enabled": not bool(args.no_api_cache),
        "training": [],
        "evaluation": [],
    }

    if not bool(args.skip_training):
        training_dir = _as_path(args.training_template_dir)
        dest_dir = _as_path(args.liquidity_dest_dir)
        for source in _iter_training_files(training_dir, str(args.training_pattern)):
            entry = _write_one_template(
                source_path=source,
                dest_path=dest_dir / source.name,
                price_area=str(args.training_price_area),
                label=f"training:{source.name}",
                args=args,
            )
            manifest["training"].append(entry)
            print(
                f"[OK] train {source.name}: rows={entry['rows']:,} "
                f"area={entry['price_area']} median_volume={entry.get('market_volume_median_mwh')}"
            )

    if not bool(args.skip_eval):
        entry = _write_one_template(
            source_path=_as_path(args.eval_original_template),
            dest_path=_as_path(args.eval_original_dest),
            price_area=str(args.eval_original_price_area),
            label="eval:original",
            args=args,
        )
        manifest["evaluation"].append(entry)
        print(
            f"[OK] eval original: rows={entry['rows']:,} area={entry['price_area']} "
            f"median_volume={entry.get('market_volume_median_mwh')}"
        )

    if not bool(args.skip_eval_v2):
        entry = _write_one_template(
            source_path=_as_path(args.eval_v2_template),
            dest_path=_as_path(args.eval_v2_dest),
            price_area=str(args.eval_v2_price_area),
            label="eval:unseendata_v2",
            args=args,
        )
        manifest["evaluation"].append(entry)
        print(
            f"[OK] eval unseendata_v2: rows={entry['rows']:,} area={entry['price_area']} "
            f"median_volume={entry.get('market_volume_median_mwh')}"
        )

    if not bool(args.dry_run):
        manifest_path = PROJECT_ROOT / "real_liquidity_volume_protocol_v1_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nManifest: {manifest_path}")
    else:
        print("\nDry run complete; no files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

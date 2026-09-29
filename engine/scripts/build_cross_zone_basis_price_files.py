#!/usr/bin/env python3
"""Build peer-zone price files for cross-zone settlement basis risk.

The main training/evaluation datasets keep their physical-asset columns.  This
script only creates aligned ``timestamp,price`` CSVs for the peer price area.
Those files are consumed by ``--mtm_settlement_price_mode cross_zone_basis``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.energidataservice.dk/dataset/Elspotprices"


def _as_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _parse_timestamps(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    try:
        return parsed.dt.tz_convert(None)
    except Exception:
        return parsed


def _fetch_elspot(start: pd.Timestamp, end: pd.Timestamp, price_area: str) -> pd.DataFrame:
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M"),
        "end": end.strftime("%Y-%m-%dT%H:%M"),
        "filter": json.dumps({"PriceArea": [price_area]}),
        "sort": "HourDK asc",
        "timezone": "dk",
        "limit": 0,
    }
    response = requests.get(API_URL, params=params, timeout=90)
    response.raise_for_status()
    records = response.json().get("records", [])
    if not records:
        raise RuntimeError(f"Elspotprices returned no records for {price_area} from {start} to {end}")
    raw = pd.DataFrame.from_records(records)
    if "HourDK" not in raw.columns or "SpotPriceDKK" not in raw.columns:
        raise RuntimeError(f"Unexpected Elspotprices columns: {list(raw.columns)}")
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["HourDK"], errors="coerce"),
            "price": pd.to_numeric(raw["SpotPriceDKK"], errors="coerce"),
        }
    ).dropna(subset=["timestamp", "price"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if out.empty:
        raise RuntimeError(f"No valid Elspot price rows for {price_area}")
    return out


def _aligned_peer_prices(template: pd.DataFrame, peer_prices: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in template.columns:
        raise ValueError("Template CSV must contain a timestamp column")
    timestamps = _parse_timestamps(template["timestamp"])
    if timestamps.isna().any():
        raise ValueError("Template timestamps could not all be parsed")

    hourly = peer_prices.copy()
    hourly["timestamp"] = _parse_timestamps(hourly["timestamp"])
    hourly = hourly.dropna(subset=["timestamp", "price"]).sort_values("timestamp")
    hourly = hourly.groupby("timestamp", as_index=False)["price"].last()

    target = pd.DataFrame({"timestamp": timestamps, "_row": range(len(timestamps))}).sort_values("timestamp")
    aligned = pd.merge_asof(
        target,
        hourly,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("70min"),
    ).sort_values("_row")
    aligned["price"] = pd.to_numeric(aligned["price"], errors="coerce").ffill().bfill()
    if aligned["price"].isna().any():
        raise RuntimeError("Peer price alignment left missing values")
    return pd.DataFrame(
        {
            "timestamp": template["timestamp"].to_numpy(),
            "price": aligned["price"].to_numpy(dtype=float),
        }
    )


def _template_files(source_dir: Path, pattern: str) -> Iterable[Path]:
    files = sorted(source_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} under {source_dir}")
    return files


def build_folder(source_dir: Path, dest_dir: Path, pattern: str, price_area: str, overwrite: bool) -> list[dict[str, object]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for source in _template_files(source_dir, pattern):
        dest = dest_dir / source.name
        if dest.exists() and not overwrite:
            rows.append({"source": str(source), "dest": str(dest), "status": "exists"})
            continue

        template = pd.read_csv(source)
        timestamps = _parse_timestamps(template["timestamp"])
        start = timestamps.min().floor("h")
        end = (timestamps.max() + pd.Timedelta(hours=2)).ceil("h")
        peer = _fetch_elspot(start, end, price_area)
        out = _aligned_peer_prices(template, peer)
        out.to_csv(dest, index=False)
        rows.append(
            {
                "source": str(source),
                "dest": str(dest),
                "status": "written",
                "rows": int(len(out)),
                "start": str(out["timestamp"].iloc[0]),
                "end": str(out["timestamp"].iloc[-1]),
                "price_area": price_area,
            }
        )
        print(f"Wrote {dest} rows={len(out)} price_area={price_area}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_template_dir", default="training_dataset_ffill")
    parser.add_argument("--training_dest_dir", default="basis_price_dataset_dk2_ffill")
    parser.add_argument("--training_pattern", default="scenario_*.csv")
    parser.add_argument("--eval_template", default="evaluation_dataset_ffill/unseendata.csv")
    parser.add_argument("--eval_dest", default="evaluation_dataset_ffill/unseendata_basis_dk2.csv")
    parser.add_argument("--price_area", default="DK2")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest: dict[str, object] = {
        "price_area": args.price_area,
        "training": [],
        "evaluation": None,
    }

    if not args.skip_training:
        manifest["training"] = build_folder(
            _as_path(args.training_template_dir),
            _as_path(args.training_dest_dir),
            str(args.training_pattern),
            str(args.price_area),
            bool(args.overwrite),
        )

    if not args.skip_eval:
        eval_template = _as_path(args.eval_template)
        eval_dest = _as_path(args.eval_dest)
        eval_dest.parent.mkdir(parents=True, exist_ok=True)
        template = pd.read_csv(eval_template)
        timestamps = _parse_timestamps(template["timestamp"])
        peer = _fetch_elspot(timestamps.min().floor("h"), (timestamps.max() + pd.Timedelta(hours=2)).ceil("h"), str(args.price_area))
        out = _aligned_peer_prices(template, peer)
        if eval_dest.exists() and not args.overwrite:
            status = "exists"
        else:
            out.to_csv(eval_dest, index=False)
            status = "written"
        manifest["evaluation"] = {
            "source": str(eval_template),
            "dest": str(eval_dest),
            "status": status,
            "rows": int(len(out)),
            "price_area": args.price_area,
        }
        print(f"{status.title()} {eval_dest} rows={len(out)} price_area={args.price_area}")

    manifest_path = _as_path(args.training_dest_dir).with_name(f"{Path(args.training_dest_dir).name}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

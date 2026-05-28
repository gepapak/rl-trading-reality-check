"""
Template-preserving cross-zone evaluation dataset generator for 2025.

This is a companion to datagenerationforevaluationyear2025.py. It is intended for
cross-market-zone robustness checks, starting with DK2, while preserving the exact
10-minute timestamp grid and physical/synthetic template of the current forward-filled
evaluation dataset.

Default behavior:
- read robustness_ffill_retrained/evaluation_dataset/unseendata.csv
- fetch hourly Energi Data Service Elspot prices for --price_area DK2
- forward-fill the hourly price onto the existing 10-minute grid
- recompute price-dependent columns: revenue, battery_energy, npv, risk, profit_label
- write robustness_ffill_retrained_dk2/evaluation_dataset/unseendata_2.csv

For FoCAL/prior evaluation, build a matching forecast cache for this output before
treating it as a clean same-zone prior evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datagenerationforevaluationyear2025 import fetch_elspot_prices, simulate_battery_energy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = PROJECT_ROOT / "robustness_ffill_retrained" / "evaluation_dataset" / "unseendata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "robustness_ffill_retrained_dk2" / "evaluation_dataset"

REQUIRED_TEMPLATE_COLUMNS = [
    "timestamp",
    "wind",
    "solar",
    "hydro",
    "load",
    "price",
    "scenario",
    "revenue",
    "battery_energy",
    "npv",
    "risk",
    "profit_label",
]


def _parse_timestamps(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, errors="raise")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    return ts


def _format_like_template(original: pd.Series, parsed: pd.Series) -> pd.Series:
    if original.dtype == object:
        return original.astype(str)
    return parsed.dt.strftime("%Y-%m-%d %H:%M:%S")


def _load_template(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Template evaluation dataset not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_TEMPLATE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    timestamps = _parse_timestamps(df["timestamp"])
    if timestamps.isna().any():
        raise ValueError(f"{path} contains unparsable timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"{path} timestamps must be sorted increasing")
    return df.copy(), timestamps


def _load_local_prices(path: Path, timestamp_col: str, price_col: str, freq: str, price_fill: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Price CSV not found: {path}")
    df = pd.read_csv(path)
    for col in (timestamp_col, price_col):
        if col not in df.columns:
            raise ValueError(f"{path} missing column '{col}'. cols={list(df.columns)}")
    out = pd.DataFrame(
        {
            "timestamp": _parse_timestamps(df[timestamp_col]),
            "price": pd.to_numeric(df[price_col], errors="coerce"),
        }
    ).dropna(subset=["timestamp", "price"])
    if out.empty:
        raise ValueError(f"No usable price rows in {path}")
    out = out.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    out = out.resample(freq).asfreq()
    if price_fill == "interpolate_then_ffill":
        out["price"] = out["price"].interpolate(method="time").ffill().bfill()
    else:
        out["price"] = out["price"].ffill().bfill()
    return out.reset_index()


def _fetch_prices_for_template(
    timestamps: pd.Series,
    *,
    price_area: str,
    freq: str,
    price_fill: str,
    price_csv: Path | None,
    price_timestamp_column: str,
    price_column: str,
) -> pd.Series:
    start = timestamps.iloc[0]
    end = timestamps.iloc[-1]
    if price_csv is not None:
        prices = _load_local_prices(price_csv, price_timestamp_column, price_column, freq, price_fill)
    else:
        prices = fetch_elspot_prices(
            start_iso=start.strftime("%Y-%m-%dT%H:%M"),
            end_iso=end.strftime("%Y-%m-%dT%H:%M"),
            price_area=price_area,
            freq=freq,
            price_fill=price_fill,
        )
    if prices.empty:
        raise ValueError(f"No prices returned for price_area={price_area}, start={start}, end={end}")
    prices = prices.copy()
    prices["timestamp"] = _parse_timestamps(prices["timestamp"])
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices = prices.dropna(subset=["timestamp", "price"]).drop_duplicates("timestamp").sort_values("timestamp")
    aligned = prices.set_index("timestamp").reindex(timestamps, method="ffill")["price"]
    if aligned.isna().any():
        missing = int(aligned.isna().sum())
        raise ValueError(
            f"Price series did not cover the template grid ({missing} missing rows). "
            "Check --price_area, --price_csv, date range, and --freq."
        )
    return aligned.reset_index(drop=True)


def _recompute_price_dependent_columns(df: pd.DataFrame, *, wind_capacity_mw: float) -> pd.DataFrame:
    out = df.copy()
    for col in ("wind", "solar", "hydro", "load", "price"):
        out[col] = pd.to_numeric(out[col], errors="raise")
    out["revenue"] = (out["wind"] + out["solar"] + out["hydro"]) * out["price"]
    out["battery_energy"] = simulate_battery_energy(out)
    npv = float(out["revenue"].sum() - (float(wind_capacity_mw) * 1000.0 + 600000.0 + 1200000.0))
    risk = float(out["revenue"].std() / max(out["revenue"].mean(), 1e-8))
    out["npv"] = npv
    out["risk"] = risk
    out["profit_label"] = int(npv > 0.0)
    return out


def _write_metadata(path: Path, payload: dict) -> None:
    meta_path = path.with_suffix(path.suffix + ".metadata.json")
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote metadata: {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a forward-filled 2025 cross-zone evaluation dataset from the existing template."
    )
    parser.add_argument("--template", type=str, default=str(DEFAULT_TEMPLATE), help="Template ffill eval CSV.")
    parser.add_argument("--price_area", type=str, default="DK2", help="Energi Data Service PriceArea, default DK2.")
    parser.add_argument("--freq", type=str, default="10min", help="Template resolution, default 10min.")
    parser.add_argument(
        "--price_fill",
        type=str,
        default="ffill_only",
        choices=["ffill_only", "interpolate_then_ffill"],
        help="How to fill hourly prices onto the 10-minute grid. Use ffill_only for the paper protocol.",
    )
    parser.add_argument(
        "--price_csv",
        type=str,
        default="",
        help="Optional local price CSV. If omitted, fetches Elspot prices from Energi Data Service.",
    )
    parser.add_argument("--price_timestamp_column", type=str, default="timestamp")
    parser.add_argument("--price_column", type=str, default="price")
    parser.add_argument("--wind_capacity_mw", type=float, default=1500.0)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output_name", type=str, default="unseendata_2.csv")
    parser.add_argument("--scenario_suffix", type=str, default="", help="Optional suffix appended to scenario labels.")
    parser.add_argument("--no_metadata", action="store_true", help="Do not write sidecar metadata JSON.")
    args = parser.parse_args()

    template_path = Path(args.template)
    out_dir = Path(args.output_dir)
    out_path = out_dir / args.output_name

    template, timestamps = _load_template(template_path)
    price_csv = Path(args.price_csv) if str(args.price_csv).strip() else None
    new_price = _fetch_prices_for_template(
        timestamps,
        price_area=str(args.price_area),
        freq=str(args.freq),
        price_fill=str(args.price_fill),
        price_csv=price_csv,
        price_timestamp_column=str(args.price_timestamp_column),
        price_column=str(args.price_column),
    )

    output = template.copy()
    output["timestamp"] = _format_like_template(template["timestamp"], timestamps)
    output["price"] = new_price.to_numpy(dtype=float)
    if str(args.scenario_suffix).strip():
        output["scenario"] = output["scenario"].astype(str) + str(args.scenario_suffix)
    output = _recompute_price_dependent_columns(output, wind_capacity_mw=float(args.wind_capacity_mw))

    out_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    start = timestamps.iloc[0]
    end = timestamps.iloc[-1]
    print(
        f"Wrote: {out_path} rows={len(output):,} start={start} end={end} "
        f"price_area={args.price_area} price_fill={args.price_fill}"
    )

    if not args.no_metadata:
        _write_metadata(
            out_path,
            {
                "script": Path(__file__).name,
                "template": str(template_path),
                "output": str(out_path),
                "rows": int(len(output)),
                "start": str(start),
                "end": str(end),
                "price_area": str(args.price_area),
                "price_fill": str(args.price_fill),
                "price_csv": str(price_csv) if price_csv is not None else "",
                "columns": list(output.columns),
                "note": (
                    "Cross-zone evaluation dataset preserving the existing 10-minute physical template. "
                    "Build a matching forecast cache before using this as a clean prior-enabled FoCAL evaluation."
                ),
            },
        )


if __name__ == "__main__":
    main()

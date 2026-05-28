"""
Build a second-zone 2023-2024 forecast-training dataset from an existing forecast scenario.

This is intentionally template-preserving: it keeps the existing timestamp grid,
physical/synthetic columns, and scenario labels, replaces only the spot-price series
with another Energi Data Service price area, and recomputes price-dependent columns.

Default use case:
- source: robustness_ffill_retrained/forecast_training_dataset/forecast_scenario_20.csv
  (2023-01-01 through 2024-12-31, ffill-only, used to train the 2025 evaluation forecaster)
- price area: DK2
- output: forecast_training_dataset_2/forecast_scenario_20_2.csv

The output is suitable for training a separate evaluation forecaster into
forecast_models/episode_20_2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datagenerationforevaluationyear2025 import fetch_elspot_prices, simulate_battery_energy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "robustness_ffill_retrained"
    / "forecast_training_dataset"
    / "forecast_scenario_20.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forecast_training_dataset_2"

REQUIRED_COLUMNS = [
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
        raise FileNotFoundError(f"Forecast-training scenario not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    timestamps = _parse_timestamps(df["timestamp"])
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


def _resolve_prices(
    timestamps: pd.Series,
    *,
    price_area: str,
    freq: str,
    price_fill: str,
    price_csv: Path | None,
    price_timestamp_column: str,
    price_column: str,
) -> pd.Series:
    if price_csv is not None:
        prices = _load_local_prices(price_csv, price_timestamp_column, price_column, freq, price_fill)
    else:
        prices = fetch_elspot_prices(
            start_iso=timestamps.iloc[0].strftime("%Y-%m-%dT%H:%M"),
            end_iso=timestamps.iloc[-1].strftime("%Y-%m-%dT%H:%M"),
            price_area=price_area,
            freq=freq,
            price_fill=price_fill,
        )
    if prices.empty:
        raise ValueError(f"No prices returned for price_area={price_area}")
    prices = prices.copy()
    prices["timestamp"] = _parse_timestamps(prices["timestamp"])
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices = prices.dropna(subset=["timestamp", "price"]).drop_duplicates("timestamp").sort_values("timestamp")
    aligned = prices.set_index("timestamp").reindex(timestamps, method="ffill")["price"]
    if aligned.isna().any():
        missing = int(aligned.isna().sum())
        raise ValueError(f"Price series did not cover the template grid ({missing} missing rows).")
    return aligned.reset_index(drop=True)


def _recompute(df: pd.DataFrame, *, wind_capacity_mw: float) -> pd.DataFrame:
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
        description="Create a second-zone forecast-training scenario by replacing the price series."
    )
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE), help="Source forecast scenario CSV.")
    parser.add_argument("--price_area", type=str, default="DK2", help="Energi Data Service PriceArea.")
    parser.add_argument("--freq", type=str, default="10min")
    parser.add_argument(
        "--price_fill",
        type=str,
        default="ffill_only",
        choices=["ffill_only", "interpolate_then_ffill"],
        help="Use ffill_only for the forward-filled paper protocol.",
    )
    parser.add_argument("--price_csv", type=str, default="", help="Optional local price CSV instead of API fetch.")
    parser.add_argument("--price_timestamp_column", type=str, default="timestamp")
    parser.add_argument("--price_column", type=str, default="price")
    parser.add_argument("--wind_capacity_mw", type=float, default=1500.0)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output_name", type=str, default="forecast_scenario_20_2.csv")
    parser.add_argument("--scenario_suffix", type=str, default="")
    parser.add_argument("--no_metadata", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.source)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name
    price_csv = Path(args.price_csv) if str(args.price_csv).strip() else None

    template, timestamps = _load_template(source_path)
    price = _resolve_prices(
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
    output["price"] = price.to_numpy(dtype=float)
    if str(args.scenario_suffix).strip():
        output["scenario"] = output["scenario"].astype(str) + str(args.scenario_suffix)
    output = _recompute(output, wind_capacity_mw=float(args.wind_capacity_mw))

    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(
        f"Wrote: {output_path} rows={len(output):,} "
        f"start={timestamps.iloc[0]} end={timestamps.iloc[-1]} "
        f"price_area={args.price_area} price_fill={args.price_fill}"
    )

    if not args.no_metadata:
        _write_metadata(
            output_path,
            {
                "script": Path(__file__).name,
                "source": str(source_path),
                "output": str(output_path),
                "rows": int(len(output)),
                "start": str(timestamps.iloc[0]),
                "end": str(timestamps.iloc[-1]),
                "price_area": str(args.price_area),
                "price_fill": str(args.price_fill),
                "price_csv": str(price_csv) if price_csv is not None else "",
                "preserved_columns": ["timestamp", "wind", "solar", "hydro", "load", "scenario"],
                "recomputed_columns": ["price", "revenue", "battery_energy", "npv", "risk", "profit_label"],
                "note": "Template-preserving second-zone forecast-training scenario for episode_20_2.",
            },
        )


if __name__ == "__main__":
    main()

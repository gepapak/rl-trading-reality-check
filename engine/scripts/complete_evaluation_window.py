#!/usr/bin/env python3
"""Complete the final delivery hour of the Prototype3 evaluation datasets.

The frozen evaluation files currently stop at 22:50 on 2025-09-30. This tool
adds the six missing 10-minute rows through 23:50 using the same documented
sources as the existing data: ERA5 100-m wind, the H2 physical template, and
official DK1/DK2 hourly day-ahead prices held constant on the environment grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.energidataservice.dk/dataset/Elspotprices"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _wind_power_series(netcdf_path: Path) -> pd.Series:
    with xr.open_dataset(netcdf_path) as dataset:
        timestamps = pd.to_datetime(dataset["valid_time"].values)
        u100 = np.asarray(dataset["u100"].values, dtype=float).reshape(len(timestamps), -1)[:, 0]
        v100 = np.asarray(dataset["v100"].values, dtype=float).reshape(len(timestamps), -1)[:, 0]
    speed = pd.Series(np.sqrt(u100 * u100 + v100 * v100), index=timestamps)
    speed = speed.resample("10min").interpolate("time")
    values = speed.to_numpy(dtype=float)
    power = np.zeros_like(values)
    ramp = (values >= 3.0) & (values <= 12.0)
    power[ramp] = (values[ramp] - 3.0) / 9.0
    power[(values > 12.0) & (values < 25.0)] = 1.0
    return pd.Series(np.clip(power, 0.0, 1.0) * 1500.0, index=speed.index)


def _fetch_hourly_prices(start: pd.Timestamp, end: pd.Timestamp, area: str) -> pd.Series:
    # Energi Data Service uses an exclusive end bound for hourly records.
    request_end = end.floor("h") + pd.Timedelta(hours=1)
    params = {
        "start": start.floor("h").strftime("%Y-%m-%dT%H:%M"),
        "end": request_end.strftime("%Y-%m-%dT%H:%M"),
        "filter": json.dumps({"PriceArea": [area]}),
        "columns": "HourDK,SpotPriceDKK",
        "sort": "HourDK ASC",
        "timezone": "dk",
    }
    response = None
    for attempt in range(1, 9):
        response = requests.get(API_URL, params=params, timeout=60)
        if response.status_code != 429:
            response.raise_for_status()
            break
        retry_after = float(response.headers.get("Retry-After", 0) or 0)
        time.sleep(max(retry_after, min(5.0 * attempt, 30.0)))
    if response is None or response.status_code == 429:
        raise RuntimeError(f"Energi Data Service rate limit persisted for {area} after 8 retries")
    records = response.json().get("records", [])
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError(f"No Elspot prices returned for {area}: {start}..{end}")
    timestamps = pd.to_datetime(frame["HourDK"], errors="raise")
    prices = pd.to_numeric(frame["SpotPriceDKK"], errors="raise")
    return pd.Series(prices.to_numpy(dtype=float), index=timestamps).sort_index()


def _simulate_battery(frame: pd.DataFrame) -> np.ndarray:
    energy = 0.0
    average_price = float(frame["price"].mean())
    result = []
    for row in frame.itertuples(index=False):
        surplus = max(0.0, float(row.wind + row.solar + row.hydro - row.load))
        charge = min(5.0, surplus) * 0.9
        if float(row.price) < average_price:
            energy += charge
        else:
            energy -= min(5.0, energy)
        energy = min(max(energy, 0.0), 10.0)
        result.append(energy)
    return np.asarray(result, dtype=float)


def _validate_existing_sources(frame: pd.DataFrame, wind: pd.Series, template: pd.DataFrame) -> None:
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise", format="mixed")
    tail = frame.tail(min(len(frame), 144)).copy()
    tail_ts = pd.to_datetime(tail["timestamp"], errors="raise", format="mixed")
    wind_error = np.max(np.abs(tail["wind"].to_numpy(dtype=float) - wind.reindex(tail_ts).to_numpy(dtype=float)))
    if not np.isfinite(wind_error) or wind_error > 1e-3:
        raise RuntimeError(f"Existing evaluation wind does not match the ERA5 source (max error={wind_error})")

    h2_start = pd.Timestamp("2025-07-01 00:00:00")
    h2 = frame.loc[timestamps >= h2_start].copy()
    h2_ts = pd.to_datetime(h2["timestamp"], errors="raise", format="mixed")
    indices = ((h2_ts - h2_start) / pd.Timedelta(minutes=10)).astype(int).to_numpy()
    for column in ("solar", "hydro", "load"):
        observed = h2[column].to_numpy(dtype=float)
        expected = template.iloc[indices][column].to_numpy(dtype=float)
        error = float(np.max(np.abs(observed - expected)))
        if not np.isfinite(error) or error > 1e-5:
            raise RuntimeError(f"Existing evaluation {column} does not match the H2 template (max error={error})")


def _complete_one(
    path: Path,
    area: str,
    target_end: pd.Timestamp,
    wind: pd.Series,
    template: pd.DataFrame,
    overwrite: bool,
) -> dict[str, object]:
    frame = pd.read_csv(path)
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise", format="mixed")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise RuntimeError(f"Evaluation timestamps are not unique and sorted: {path}")
    _validate_existing_sources(frame, wind, template)
    current_end = timestamps.iloc[-1]
    if current_end == target_end:
        if overwrite:
            frame["timestamp"] = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S")
            temporary = path.with_suffix(path.suffix + ".tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(path)
        return {"path": _relative(path), "status": "already_complete", "rows": len(frame), "sha256": _sha256(path)}
    if current_end >= target_end:
        raise RuntimeError(f"Unexpected evaluation endpoint {current_end} for {path}; target={target_end}")
    missing_ts = pd.date_range(current_end + pd.Timedelta(minutes=10), target_end, freq="10min")
    if len(missing_ts) == 0:
        raise RuntimeError(f"No missing rows found for {path}")

    h2_start = pd.Timestamp("2025-07-01 00:00:00")
    indices = ((missing_ts - h2_start) / pd.Timedelta(minutes=10)).astype(int)
    if int(indices.min()) < 0 or int(indices.max()) >= len(template):
        raise RuntimeError("H2 template does not cover the missing evaluation timestamps")
    hourly_prices = _fetch_hourly_prices(missing_ts[0], missing_ts[-1], area)
    price = hourly_prices.reindex(missing_ts.floor("h"), method="ffill").to_numpy(dtype=float)
    if not np.all(np.isfinite(price)):
        raise RuntimeError(f"Official price coverage is incomplete for {area}")

    template_rows = template.iloc[np.asarray(indices, dtype=int)]
    extra = pd.DataFrame({column: np.nan for column in frame.columns}, index=range(len(missing_ts)))
    extra["timestamp"] = missing_ts.strftime("%Y-%m-%d %H:%M:%S")
    extra["wind"] = wind.reindex(missing_ts).to_numpy(dtype=float)
    for column in ("solar", "hydro", "load"):
        extra[column] = template_rows[column].to_numpy(dtype=float)
    extra["price"] = price
    extra["scenario"] = str(frame["scenario"].iloc[-1])

    output = pd.concat([frame, extra], ignore_index=True)
    output["timestamp"] = pd.to_datetime(
        output["timestamp"], errors="raise", format="mixed"
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    for column in ("wind", "solar", "hydro", "load", "price"):
        output[column] = pd.to_numeric(output[column], errors="raise")
    output["revenue"] = (output["wind"] + output["solar"] + output["hydro"]) * output["price"]
    output["battery_energy"] = _simulate_battery(output)
    npv = float(output["revenue"].sum() - (1500.0 * 1000.0 + 600000.0 + 1200000.0))
    mean_revenue = float(output["revenue"].mean())
    output["npv"] = npv
    output["risk"] = float(output["revenue"].std() / mean_revenue) if abs(mean_revenue) > 1e-8 else 0.0
    output["profit_label"] = int(npv > 0.0)
    output = output.loc[:, frame.columns]

    if not overwrite:
        raise RuntimeError(f"Refusing to replace {path} without --overwrite")
    temporary = path.with_suffix(path.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(path)
    return {
        "path": _relative(path),
        "status": "completed",
        "area": area,
        "rows_before": int(len(frame)),
        "rows_after": int(len(output)),
        "start": str(pd.to_datetime(output["timestamp"], format="%Y-%m-%d %H:%M:%S").iloc[0]),
        "end": str(pd.to_datetime(output["timestamp"], format="%Y-%m-%d %H:%M:%S").iloc[-1]),
        "sha256": _sha256(path),
        "missing_rows_added": int(len(missing_ts)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_end", default="2025-09-30 23:50:00")
    parser.add_argument("--wind_netcdf", default="dataset generation/wind speed data/2025.nc")
    parser.add_argument("--h2_template", default="training_dataset_ffill/scenario_001.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    netcdf = PROJECT_ROOT / args.wind_netcdf
    template_path = PROJECT_ROOT / args.h2_template
    wind = _wind_power_series(netcdf)
    template = pd.read_csv(template_path)
    target_end = pd.Timestamp(args.target_end)
    results = [
        _complete_one(PROJECT_ROOT / "evaluation_dataset_ffill/unseendata.csv", "DK1", target_end, wind, template, args.overwrite),
        _complete_one(PROJECT_ROOT / "evaluation_dataset_ffill/unseendata_v2.csv", "DK2", target_end, wind, template, args.overwrite),
    ]
    manifest = {
        "script": _relative(Path(__file__)),
        "target_end": str(target_end),
        "wind_source": _relative(netcdf),
        "wind_source_sha256": _sha256(netcdf),
        "physical_template": _relative(template_path),
        "physical_template_sha256": _sha256(template_path),
        "price_source": "Energi Data Service Elspotprices, timezone=dk",
        "results": results,
    }
    output = PROJECT_ROOT / "evaluation_window_completion_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    for result in results:
        print(f"[OK] {result}")
    print(f"Manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

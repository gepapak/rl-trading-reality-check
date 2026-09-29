#!/usr/bin/env python3
"""Build the SECOND out-of-sample evaluation window (2025 Q4) for both zones.

The frozen first window spans 2025-01-01 -> 2025-09-30 (files ``unseendata*``).
This tool produces a *separate*, non-overlapping second window covering the
remaining months of 2025 (2025-10-01 00:00 -> 2025-12-31 23:50) for both price
zones (DK1 and DK2), with all three evaluation-data products per zone:

    1. main physical/price eval CSV
    2. real strict-settlement product (settlement_real_v2)
    3. real balancing-activation liquidity product (liquidity_volume_real_v1)

Everything reuses the *same documented sources and tested code paths* as the
first window so provenance is identical:
  * ERA5 100-m wind from ``dataset generation/wind speed data/2025.nc``
  * the 2025-H2 physical template ``training_dataset_ffill/scenario_001.csv``
    for solar/hydro/load (index-aligned to 2025-07-01; it fully covers Q4)
  * official DK1/DK2 hourly day-ahead prices (Energi Data Service Elspotprices),
    forward-filled onto the 10-minute grid
The wind/price/battery primitives are imported directly from
``complete_evaluation_window.py`` so the numbers match the first window exactly.

Stage 1 writes the two main eval CSVs. Stages 2-3 drive the existing, tested
settlement and liquidity builders (``--skip_training*``, eval-only) pointed at
the new Q4 main files, so the real settlement/activation data is fetched for the
Q4 period and aligned to the Q4 timestamps. Optional Stage 4 precomputes the
FoCAL hourly-forecast cache for the Q4 window using the *already-trained* bank
(no retraining), which is what the forecast-authority arms need to evaluate on
window 2.

Output files (all under ``evaluation_dataset_ffill/``):
    DK1  unseendata_2025q4.csv
         unseendata_2025q4_settlement_real_v2.csv
         unseendata_2025q4_liquidity_volume_real_v1.csv
    DK2  unseendata_v2_2025q4.csv
         unseendata_v2_2025q4_settlement_real_v2.csv
         unseendata_v2_2025q4_liquidity_volume_real_v1.csv

Network: Stages 2-3 (and the price fetch in Stage 1) call the Energi Data
Service API. Use --dry_run to print the plan without any network / file writes.

Example:
    python scripts/build_second_window_2025q4.py --overwrite
    python scripts/build_second_window_2025q4.py --overwrite --with_forecast_cache
    python scripts/build_second_window_2025q4.py --dry_run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Reuse the exact, tested primitives that built the first window so wind and
# battery numbers are byte-for-byte consistent across both windows. The price
# fetch is reimplemented locally (same dataset/columns/timezone/ffill) with a
# resilient month-chunked retry loop, because the first-window helper was only
# battle-tested on tiny ranges and is fragile under API rate-limiting.
from complete_evaluation_window import (  # noqa: E402
    _simulate_battery,
    _wind_power_series,
)

API_URL = "https://api.energidataservice.dk/dataset/Elspotprices"

EVAL_DIR = PROJECT_ROOT / "evaluation_dataset_ffill"
FORECAST_INPUT_DIR = PROJECT_ROOT / "forecast_cache_input_settlement_v2"

# NPV capex offset, identical to complete_evaluation_window._complete_one:
#   wind_capacity_mw * 1000 + solar_capex + hydro_capex
_NPV_CAPEX_OFFSET = 1500.0 * 1000.0 + 600000.0 + 1200000.0

# H2 physical template is index-aligned to this anchor (row 0 == 2025-07-01 00:00).
_H2_ANCHOR = pd.Timestamp("2025-07-01 00:00:00")

# Column schema of the frozen first-window eval files (order matters).
_EVAL_COLUMNS = [
    "timestamp", "wind", "solar", "hydro", "load", "price",
    "scenario", "revenue", "battery_energy", "npv", "risk", "profit_label",
]

# Per-zone output/plumbing spec: (main, settlement, forecast_input, liquidity, price_area).
ZONES = {
    "DK1": {
        "main": "unseendata_2025q4.csv",
        "settlement": "unseendata_2025q4_settlement_real_v2.csv",
        "forecast_input": "unseendata_2025q4.csv",
        "liquidity": "unseendata_2025q4_liquidity_volume_real_v1.csv",
        "scenario_ref": "unseendata.csv",
    },
    "DK2": {
        "main": "unseendata_v2_2025q4.csv",
        "settlement": "unseendata_v2_2025q4_settlement_real_v2.csv",
        "forecast_input": "unseendata_v2_2025q4.csv",
        "liquidity": "unseendata_v2_2025q4_liquidity_volume_real_v1.csv",
        "scenario_ref": "unseendata_v2.csv",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scenario_label(ref_path: Path, fallback: str) -> str:
    try:
        ref = pd.read_csv(ref_path, usecols=["scenario"])
        val = str(ref["scenario"].iloc[-1])
        if val and val.lower() != "nan":
            return val
    except Exception:
        pass
    return fallback


def _fetch_price_chunk(
    start: pd.Timestamp, end: pd.Timestamp, area: str, *, retries: int, base_pause: float, max_pause: float
) -> pd.Series:
    """Fetch one [start, end) hourly price chunk with 429/empty/network retries."""
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M"),
        "end": end.strftime("%Y-%m-%dT%H:%M"),
        "filter": json.dumps({"PriceArea": [area]}),
        "columns": "HourDK,SpotPriceDKK",
        "sort": "HourDK ASC",
        "timezone": "dk",
        "limit": 0,  # no pagination cap -> return every record in the range
    }
    last_err: object = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(API_URL, params=params, timeout=90)
        except requests.RequestException as exc:  # transient network failure
            last_err = exc
            time.sleep(min(base_pause * attempt, max_pause))
            continue
        if response.status_code == 429:  # rate limited -> respect Retry-After
            retry_after = float(response.headers.get("Retry-After", 0) or 0)
            wait = max(retry_after, min(base_pause * attempt, max_pause))
            print(f"      [price] {area} {start.date()}: 429, waiting {wait:.0f}s "
                  f"(attempt {attempt}/{retries})")
            time.sleep(wait)
            continue
        response.raise_for_status()
        records = response.json().get("records", [])
        if not records:  # 200-but-empty is a soft rate-limit artifact -> back off
            last_err = RuntimeError("empty records")
            time.sleep(min(base_pause * attempt, max_pause))
            continue
        frame = pd.DataFrame(records)
        ts = pd.to_datetime(frame["HourDK"], errors="raise")
        prices = pd.to_numeric(frame["SpotPriceDKK"], errors="raise")
        return pd.Series(prices.to_numpy(dtype=float), index=ts).sort_index()
    raise RuntimeError(f"Elspot fetch failed for {area} {start}..{end} after {retries} attempts ({last_err})")


def _fetch_hourly_prices_resilient(
    start: pd.Timestamp, end: pd.Timestamp, area: str, *, retries: int = 20,
    base_pause: float = 5.0, max_pause: float = 90.0,
) -> pd.Series:
    """Official hourly Elspot prices over [start, end], fetched month-by-month.

    Same dataset/columns/timezone as the first-window protocol; month chunking
    keeps each request small and rate-limit friendly, and every chunk is retried
    on 429 / empty / network errors.
    """
    lo = start.floor("h")
    hi = end.floor("h") + pd.Timedelta(hours=1)  # exclusive hourly end bound
    edges = sorted(set([lo, hi]) | {m for m in pd.date_range(lo, hi, freq="MS") if lo < m < hi})
    parts = [
        _fetch_price_chunk(edges[i], edges[i + 1], area, retries=retries,
                           base_pause=base_pause, max_pause=max_pause)
        for i in range(len(edges) - 1)
    ]
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep="first")]


def _build_main_frame(
    *,
    area: str,
    timestamps: pd.DatetimeIndex,
    wind_series: pd.Series,
    template: pd.DataFrame,
    scenario_label: str,
    api_retries: int,
    api_pause: float,
) -> tuple[pd.DataFrame, dict]:
    """Assemble one zone's Q4 main eval frame using the first-window recipe."""
    # --- wind: reindex the ERA5-derived 10-min power series onto the Q4 grid.
    wind = wind_series.reindex(timestamps)
    trailing_nan = int(wind.isna().sum())
    if trailing_nan:
        # 2025.nc is hourly and ends 2025-12-31 23:00, so the final 10-min rows
        # (23:10-23:50) have no interpolation anchor. Hold the last hourly value
        # constant across its sub-steps -- identical convention to the hourly
        # price ffill used throughout this protocol.
        wind = wind.ffill()
        first_gap = timestamps[wind_series.reindex(timestamps).isna().to_numpy()][0]
        if trailing_nan > 6 or first_gap < pd.Timestamp("2025-12-31 23:00:00"):
            raise RuntimeError(
                f"{area}: unexpected wind gap ({trailing_nan} rows from {first_gap}); "
                "only a small year-end tail was expected."
            )
    wind_vals = wind.to_numpy(dtype=float)
    if not np.all(np.isfinite(wind_vals)):
        raise RuntimeError(f"{area}: wind still has non-finite values after year-end ffill")

    # --- solar/hydro/load: 2025-H2 template, index-aligned to 2025-07-01.
    idx = ((timestamps - _H2_ANCHOR) / pd.Timedelta(minutes=10)).astype(int)
    if int(idx.min()) < 0 or int(idx.max()) >= len(template):
        raise RuntimeError(
            f"{area}: H2 template (len={len(template)}) does not cover the Q4 window "
            f"(needs rows {int(idx.min())}..{int(idx.max())})"
        )
    rows = template.iloc[np.asarray(idx, dtype=int)]

    # --- price: official hourly Elspot for the zone, ffilled onto the 10-min grid.
    hourly = _fetch_hourly_prices_resilient(
        timestamps[0], timestamps[-1], area, retries=api_retries, base_pause=api_pause,
    )
    price = hourly.reindex(timestamps.floor("h"), method="ffill").to_numpy(dtype=float)
    if not np.all(np.isfinite(price)):
        raise RuntimeError(f"{area}: official price coverage is incomplete for the Q4 window")

    frame = pd.DataFrame({
        "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "wind": wind_vals,
        "solar": rows["solar"].to_numpy(dtype=float),
        "hydro": rows["hydro"].to_numpy(dtype=float),
        "load": rows["load"].to_numpy(dtype=float),
        "price": price,
        "scenario": scenario_label,
    })
    frame["revenue"] = (frame["wind"] + frame["solar"] + frame["hydro"]) * frame["price"]
    frame["battery_energy"] = _simulate_battery(frame)
    npv = float(frame["revenue"].sum() - _NPV_CAPEX_OFFSET)
    mean_rev = float(frame["revenue"].mean())
    frame["npv"] = npv
    frame["risk"] = float(frame["revenue"].std() / mean_rev) if abs(mean_rev) > 1e-8 else 0.0
    frame["profit_label"] = int(npv > 0.0)
    frame = frame.loc[:, _EVAL_COLUMNS]

    stats = {
        "area": area,
        "rows": int(len(frame)),
        "start": str(timestamps[0]),
        "end": str(timestamps[-1]),
        "wind_yearend_ffill_rows": trailing_nan,
        "price_min": float(np.min(price)),
        "price_max": float(np.max(price)),
        "npv": npv,
    }
    return frame, stats


def _run_step(label: str, argv: list[str], *, dry_run: bool) -> None:
    print("\n" + "=" * 88)
    print(f"[Q4] {label}")
    print(f"{sys.executable} {' '.join(argv)}")
    print("=" * 88)
    if dry_run:
        print("[Q4] (dry-run) skipped")
        return
    proc = subprocess.run([sys.executable, *argv], cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {proc.returncode})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window_start", default="2025-10-01 00:00")
    parser.add_argument("--window_end", default="2025-12-31 23:50")
    parser.add_argument("--freq", default="10min")
    parser.add_argument("--wind_netcdf", default="dataset generation/wind speed data/2025.nc")
    parser.add_argument("--h2_template", default="training_dataset_ffill/scenario_001.csv")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing Q4 output files (Stage 1 and downstream builders).")
    parser.add_argument("--skip_main", action="store_true",
                        help="Skip Stage 1 (assume the Q4 main eval files already exist).")
    parser.add_argument("--skip_settlement", action="store_true", help="Skip Stage 2 (settlement).")
    parser.add_argument("--skip_liquidity", action="store_true", help="Skip Stage 3 (liquidity).")
    parser.add_argument("--with_forecast_cache", action="store_true",
                        help="Stage 4: precompute the FoCAL hourly-forecast cache for the Q4 "
                             "window using the already-trained bank (no retraining).")
    parser.add_argument("--settlement_clip_min", type=float, default=-111750.0)
    parser.add_argument("--settlement_clip_max", type=float, default=111750.0)
    parser.add_argument("--api_pause_seconds", type=float, default=5.0)
    parser.add_argument("--api_retries", type=int, default=12)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the full plan (Stage 1 build + all subprocess commands) "
                             "without any network calls or file writes.")
    args = parser.parse_args()

    netcdf = PROJECT_ROOT / args.wind_netcdf
    template_path = PROJECT_ROOT / args.h2_template
    for required in (netcdf, template_path):
        if not required.exists():
            raise FileNotFoundError(f"Required source not found: {required}")

    timestamps = pd.date_range(args.window_start, args.window_end, freq=args.freq)
    print(f"[Q4] Second evaluation window: {timestamps[0]} -> {timestamps[-1]} "
          f"({len(timestamps):,} rows @ {args.freq}), zones: {', '.join(ZONES)}")
    print(f"[Q4] Wind: {args.wind_netcdf}   H2 template: {args.h2_template}")
    print(f"[Q4] Output dir: {EVAL_DIR.relative_to(PROJECT_ROOT).as_posix()}")

    manifest: dict[str, object] = {
        "script": Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix(),
        "window_start": str(timestamps[0]),
        "window_end": str(timestamps[-1]),
        "freq": args.freq,
        "wind_source": args.wind_netcdf,
        "physical_template": args.h2_template,
        "price_source": "Energi Data Service Elspotprices, timezone=dk",
        "zones": {},
    }

    # ---- Stage 1: main eval CSVs (both zones). -----------------------------
    if not args.skip_main:
        wind_series = None if args.dry_run else _wind_power_series(netcdf)
        template = None if args.dry_run else pd.read_csv(template_path)
        for area, spec in ZONES.items():
            dest = EVAL_DIR / spec["main"]
            print("\n" + "-" * 88)
            print(f"[Q4] Stage 1 main eval -> {dest.relative_to(PROJECT_ROOT).as_posix()}  ({area})")
            if args.dry_run:
                print("[Q4] (dry-run) would fetch prices + assemble frame + write CSV")
                continue
            if dest.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {dest}; pass --overwrite")
            scenario_label = _scenario_label(EVAL_DIR / spec["scenario_ref"], f"scenario_2025_Q4_OOS_{area}")
            frame, stats = _build_main_frame(
                area=area, timestamps=timestamps, wind_series=wind_series,
                template=template, scenario_label=scenario_label,
                api_retries=args.api_retries, api_pause=args.api_pause_seconds,
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            frame.to_csv(tmp, index=False)
            tmp.replace(dest)
            stats["path"] = dest.relative_to(PROJECT_ROOT).as_posix()
            stats["sha256"] = _sha256(dest)
            manifest["zones"][area] = {"main": stats}
            print(f"[Q4] wrote {stats['path']} rows={stats['rows']:,} "
                  f"price[{stats['price_min']:.1f},{stats['price_max']:.1f}] "
                  f"yearend_ffill={stats['wind_yearend_ffill_rows']}")
            time.sleep(1.0)  # be gentle between the two zone price fetches
    else:
        print("\n[Q4] Stage 1 skipped (--skip_main)")

    def rel(name: str) -> str:
        return (EVAL_DIR / name).relative_to(PROJECT_ROOT).as_posix()

    def frel(name: str) -> str:
        return (FORECAST_INPUT_DIR / name).relative_to(PROJECT_ROOT).as_posix()

    ow = ["--overwrite"] if args.overwrite else []

    # ---- Stage 2: real strict-settlement product (both zones). -------------
    if not args.skip_settlement:
        _run_step(
            "Stage 2  strict settlement (eval-only, both zones)",
            [
                "scripts/build_real_settlement_protocol_v2.py",
                "--skip_training_settlement", "--skip_forecast_training",
                "--eval_original_template", rel(ZONES["DK1"]["main"]),
                "--eval_original_settlement_dest", rel(ZONES["DK1"]["settlement"]),
                "--eval_original_forecast_input_dest", frel(ZONES["DK1"]["forecast_input"]),
                "--eval_original_price_area", "DK1",
                "--eval_v2_template", rel(ZONES["DK2"]["main"]),
                "--eval_v2_settlement_dest", rel(ZONES["DK2"]["settlement"]),
                "--eval_v2_forecast_input_dest", frel(ZONES["DK2"]["forecast_input"]),
                "--eval_v2_price_area", "DK2",
                "--settlement_clip_min", str(args.settlement_clip_min),
                "--settlement_clip_max", str(args.settlement_clip_max),
                "--api_pause_seconds", str(args.api_pause_seconds),
                "--api_retries", str(args.api_retries),
                *ow,
            ],
            dry_run=args.dry_run,
        )
    else:
        print("\n[Q4] Stage 2 skipped (--skip_settlement)")

    # ---- Stage 3: real balancing-activation liquidity product (both zones). -
    if not args.skip_liquidity:
        _run_step(
            "Stage 3  balancing-activation liquidity (eval-only, both zones)",
            [
                "scripts/build_real_liquidity_volume_protocol_v1.py",
                "--skip_training",
                "--dataset", "auto", "--volume_mode", "activation_sum",
                "--eval_original_template", rel(ZONES["DK1"]["main"]),
                "--eval_original_dest", rel(ZONES["DK1"]["liquidity"]),
                "--eval_original_price_area", "DK1",
                "--eval_v2_template", rel(ZONES["DK2"]["main"]),
                "--eval_v2_dest", rel(ZONES["DK2"]["liquidity"]),
                "--eval_v2_price_area", "DK2",
                "--api_pause_seconds", str(args.api_pause_seconds),
                "--api_retries", str(args.api_retries),
                *ow,
            ],
            dry_run=args.dry_run,
        )
    else:
        print("\n[Q4] Stage 3 skipped (--skip_liquidity)")

    # ---- Stage 4 (optional): FoCAL forecast cache for the Q4 window. --------
    if args.with_forecast_cache:
        for area in ("DK1", "DK2"):
            _run_step(
                f"Stage 4  forecast cache ({area}, existing bank, no retrain)",
                [
                    "scripts/precompute_hourly_forecast_cache_v2.py",
                    "--forecast_base_dir", "forecast_models_settlement_hourly_v2",
                    "--forecast_cache_dir", "forecast_cache_settlement_hourly_v2",
                    "--episode_data_dir", "forecast_cache_input_settlement_v2",
                    "--eval_data", frel(ZONES[area]["forecast_input"]),
                    "--skip_training_episodes",
                    "--overwrite_cache",
                ],
                dry_run=args.dry_run,
            )

    # ---- Manifest. ---------------------------------------------------------
    if not args.dry_run and not args.skip_main:
        out = PROJECT_ROOT / "second_window_2025q4_manifest.json"
        out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\n[Q4] Manifest: {out.relative_to(PROJECT_ROOT).as_posix()}")

    print("\n[Q4] Done. Second-window files (per zone) under "
          f"{EVAL_DIR.relative_to(PROJECT_ROOT).as_posix()}/ with the *_2025q4* infix.")
    if args.dry_run:
        print("[Q4] (dry-run) no files written and no network calls were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

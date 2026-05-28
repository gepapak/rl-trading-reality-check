"""Build ffill-only eval data and precompute the reserved episode-20 forecast cache.

This script wraps the Archive forecast-engine workflow for the forecast-edge
robustness prompt. It writes only to an isolated output root by default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_FORECAST_ENGINE = PROJECT_ROOT / "Archive" / "forecast_engine.py"


def _as_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _hourly_ffill_price(df: pd.DataFrame, *, timestamp_col: str, price_col: str) -> pd.Series:
    timestamps = pd.to_datetime(df[timestamp_col], errors="coerce")
    if timestamps.isna().any():
        bad = int(timestamps.isna().sum())
        raise ValueError(f"{timestamp_col} contains {bad} unparsable timestamps")

    tmp = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": pd.to_numeric(df[price_col], errors="coerce"),
        }
    )
    if tmp["price"].isna().any():
        bad = int(tmp["price"].isna().sum())
        raise ValueError(f"{price_col} contains {bad} non-numeric prices")

    tmp["hour"] = tmp["timestamp"].dt.floor("h")
    minute00 = tmp[tmp["timestamp"].dt.minute.eq(0)].drop_duplicates("hour", keep="first")
    hourly = minute00.set_index("hour")["price"].sort_index()

    missing_hours = sorted(set(tmp["hour"]) - set(hourly.index))
    if missing_hours:
        # Fallback for malformed inputs without exact minute-00 rows.
        hourly = tmp.sort_values("timestamp").drop_duplicates("hour", keep="first").set_index("hour")["price"].sort_index()

    out = tmp["hour"].map(hourly).astype(float)
    if out.isna().any():
        out = out.ffill().bfill()
    return out


def _simulate_battery_energy(df: pd.DataFrame) -> np.ndarray:
    battery_capacity_mwh = 10.0
    battery_power_limit_mw = 5.0
    battery_efficiency = 0.9
    battery_energy = 0.0
    out: List[float] = []
    avg_price = float(pd.to_numeric(df["price"], errors="coerce").mean())

    for _, row in df.iterrows():
        surplus = max(0.0, float(row["wind"] + row["solar"] + row["hydro"] - row["load"]))
        charge_energy = min(battery_power_limit_mw, surplus) * battery_efficiency
        if float(row["price"]) < avg_price:
            battery_energy += charge_energy
        else:
            discharge = min(battery_power_limit_mw, battery_energy)
            battery_energy -= discharge
        battery_energy = max(0.0, min(battery_capacity_mwh, battery_energy))
        out.append(float(battery_energy))
    return np.asarray(out, dtype=float)


def build_ffill_dataset(
    *,
    source_csv: Path,
    output_csv: Path,
    overwrite: bool = False,
    recompute_accounting: bool = True,
) -> Dict[str, Any]:
    if not source_csv.is_file():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")
    if output_csv.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {output_csv}")

    df = pd.read_csv(source_csv)
    if "timestamp" not in df.columns or "price" not in df.columns:
        raise ValueError(f"{source_csv} must contain timestamp and price columns")

    old_price = pd.to_numeric(df["price"], errors="coerce").astype(float)
    df["price"] = _hourly_ffill_price(df, timestamp_col="timestamp", price_col="price")

    if recompute_accounting and all(c in df.columns for c in ("wind", "solar", "hydro")):
        df["revenue"] = (df["wind"].astype(float) + df["solar"].astype(float) + df["hydro"].astype(float)) * df["price"].astype(float)
    if recompute_accounting and all(c in df.columns for c in ("wind", "solar", "hydro", "load")):
        df["battery_energy"] = _simulate_battery_energy(df)
    if recompute_accounting and "revenue" in df.columns:
        revenue = pd.to_numeric(df["revenue"], errors="coerce").astype(float)
        npv = float(revenue.sum() - (1500.0 * 1000.0 + 600000.0 + 1200000.0))
        risk = float(revenue.std() / max(float(revenue.mean()), 1e-8))
        df["npv"] = npv
        df["risk"] = risk
        df["profit_label"] = int(npv > 0.0)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    changed = (old_price.to_numpy(dtype=float) != df["price"].to_numpy(dtype=float))
    return {
        "source_csv": str(source_csv),
        "output_csv": str(output_csv),
        "rows": int(len(df)),
        "start": str(pd.to_datetime(df["timestamp"]).min()),
        "end": str(pd.to_datetime(df["timestamp"]).max()),
        "price_changed_rows": int(np.sum(changed)),
        "price_changed_frac": float(np.mean(changed)),
        "old_price_mean": float(old_price.mean()),
        "new_price_mean": float(pd.to_numeric(df["price"], errors="coerce").mean()),
    }


def _direction_hit(pred: np.ndarray, realized: np.ndarray) -> Dict[str, Any]:
    pred_s = np.sign(np.asarray(pred, dtype=float))
    real_s = np.sign(np.asarray(realized, dtype=float))
    mask = np.isfinite(pred_s) & np.isfinite(real_s) & (pred_s != 0.0) & (real_s != 0.0)
    if int(mask.sum()) == 0:
        return {"n": 0, "hit": None}
    return {"n": int(mask.sum()), "hit": float((pred_s[mask] == real_s[mask]).mean())}


def write_slope_diagnostics(*, csv_path: Path, output_csv: Path, horizon_steps: int = 6) -> None:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    price = pd.to_numeric(df["price"], errors="coerce").to_numpy(dtype=float)
    n = max(0, len(price) - int(horizon_steps))
    future = price[int(horizon_steps): int(horizon_steps) + n] - price[:n]
    last10 = np.r_[0.0, np.diff(price)][:n]
    minute = df["timestamp"].dt.minute.to_numpy()[:n]
    rows: List[Dict[str, Any]] = []
    for label, mask in (
        ("all", np.ones(n, dtype=bool)),
        ("minute00", minute == 0),
        ("non00", minute != 0),
        ("decision_mod6", (np.arange(n) % int(horizon_steps)) == 0),
    ):
        stats = _direction_hit(last10[mask], future[mask])
        rows.append(
            {
                "dataset": str(csv_path),
                "slice": label,
                "horizon_steps": int(horizon_steps),
                "last10_nonzero_frac": float(np.mean(last10[mask] != 0.0)) if int(mask.sum()) else 0.0,
                **stats,
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def run_command(cmd: List[str], *, cwd: Path, dry_run: bool = False) -> Dict[str, Any]:
    printable = " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd)
    if dry_run:
        print(f"[DRY-RUN] {printable}")
        return {"command": printable, "returncode": None, "dry_run": True}
    print(f"[RUN] {printable}")
    completed = subprocess.run(cmd, cwd=str(cwd), check=True)
    return {"command": printable, "returncode": int(completed.returncode), "dry_run": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ffill-only 2025 eval dataset and precompute reserved episode-20 forecast cache."
    )
    parser.add_argument("--source_eval_data", default="evaluation_dataset/unseendata.csv")
    parser.add_argument("--source_forecast_scenario20", default="forecast_training_dataset/forecast_scenario_20.csv")
    parser.add_argument("--output_root", default="robustness_ffill")
    parser.add_argument("--output_eval_data", default=None)
    parser.add_argument("--output_forecast_scenario20", default=None)
    parser.add_argument("--forecast_base_dir", default="forecast_models")
    parser.add_argument("--output_forecast_base_dir", default=None)
    parser.add_argument("--output_forecast_cache_dir", default=None)
    parser.add_argument("--archive_forecast_engine", default=str(ARCHIVE_FORECAST_ENGINE))
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--retrain_eval_forecaster", action="store_true")
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--skip_precompute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = _as_path(args.output_root)
    output_eval_data = _as_path(args.output_eval_data) if args.output_eval_data else output_root / "evaluation_dataset" / "unseendata.csv"
    output_scenario20 = (
        _as_path(args.output_forecast_scenario20)
        if args.output_forecast_scenario20
        else output_root / "forecast_training_dataset" / "forecast_scenario_20.csv"
    )
    output_cache_dir = _as_path(args.output_forecast_cache_dir) if args.output_forecast_cache_dir else output_root / "forecast_cache"
    forecast_base_dir = _as_path(args.forecast_base_dir)
    output_forecast_base_dir = (
        _as_path(args.output_forecast_base_dir)
        if args.output_forecast_base_dir
        else output_root / "forecast_models"
    )
    engine = _as_path(args.archive_forecast_engine)
    if not engine.is_file():
        raise FileNotFoundError(f"Archive forecast engine not found: {engine}")

    output_root.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "archive_forecast_engine": str(engine),
        "steps": [],
    }

    eval_info = build_ffill_dataset(
        source_csv=_as_path(args.source_eval_data),
        output_csv=output_eval_data,
        overwrite=bool(args.overwrite),
    )
    metadata["eval_dataset"] = eval_info
    write_slope_diagnostics(
        csv_path=output_eval_data,
        output_csv=output_root / "diagnostics" / "ffill_eval_slope_diagnostics.csv",
    )

    active_forecast_base_dir = forecast_base_dir
    if bool(args.retrain_eval_forecaster):
        scenario_info = build_ffill_dataset(
            source_csv=_as_path(args.source_forecast_scenario20),
            output_csv=output_scenario20,
            overwrite=bool(args.overwrite),
        )
        metadata["forecast_scenario20"] = scenario_info
        write_slope_diagnostics(
            csv_path=output_scenario20,
            output_csv=output_root / "diagnostics" / "ffill_forecast_scenario20_slope_diagnostics.csv",
        )
        train_cmd = [
            sys.executable,
            str(engine),
            "train-eval",
            "--scenario_path",
            str(output_scenario20),
            "--forecast_base_dir",
            str(output_forecast_base_dir),
        ]
        if bool(args.force_retrain):
            train_cmd.append("--force_retrain")
        metadata["steps"].append(run_command(train_cmd, cwd=PROJECT_ROOT, dry_run=bool(args.dry_run)))
        active_forecast_base_dir = output_forecast_base_dir

    if not bool(args.skip_precompute):
        precompute_cmd = [
            sys.executable,
            str(engine),
            "precompute-eval",
            "--eval_data",
            str(output_eval_data),
            "--forecast_base_dir",
            str(active_forecast_base_dir),
            "--forecast_cache_dir",
            str(output_cache_dir),
            "--batch_size",
            str(int(args.batch_size)),
        ]
        metadata["steps"].append(run_command(precompute_cmd, cwd=PROJECT_ROOT, dry_run=bool(args.dry_run)))

    metadata["outputs"] = {
        "output_root": str(output_root),
        "output_eval_data": str(output_eval_data),
        "forecast_base_dir_used_for_cache": str(active_forecast_base_dir),
        "output_forecast_cache_dir": str(output_cache_dir),
        "expected_eval_cache_dir": str(
            output_cache_dir
            / "forecast_cache_eval_episode20_2025"
            / "forecast_cache_eval_episode20_2025-full"
        ),
        "diagnostics_dir": str(output_root / "diagnostics"),
    }
    metadata_path = output_root / "ffill_build_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"  Eval data: {output_eval_data}")
    print(f"  Forecast cache root: {output_cache_dir}")
    print(f"  Expected cache dir for evaluation.py: {metadata['outputs']['expected_eval_cache_dir']}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

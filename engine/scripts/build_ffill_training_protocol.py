"""Build forward-filled training protocol folders without deleting originals.

Outputs preserve the existing file naming/layout:
- training_dataset_ffill/scenario_000.csv .. scenario_019.csv
- rolling_past_history_dataset_ffill/history_000.csv .. history_019.csv
- forecast_training_dataset_ffill/forecast_scenario_00.csv .. forecast_scenario_20.csv

Only the price path and dependent accounting columns are changed. Physical
generation/load columns are left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _as_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _hourly_ffill_price(df: pd.DataFrame) -> pd.Series:
    if "timestamp" not in df.columns or "price" not in df.columns:
        raise ValueError("CSV must contain timestamp and price columns")
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("timestamp column contains unparsable values")
    price = pd.to_numeric(df["price"], errors="coerce")
    if price.isna().any():
        raise ValueError("price column contains non-numeric values")

    tmp = pd.DataFrame({"timestamp": timestamps, "price": price.astype(float)})
    tmp["hour"] = tmp["timestamp"].dt.floor("h")
    minute00 = tmp[tmp["timestamp"].dt.minute.eq(0)].drop_duplicates("hour", keep="first")
    hourly = minute00.set_index("hour")["price"].sort_index()
    if len(hourly) == 0:
        hourly = tmp.sort_values("timestamp").drop_duplicates("hour", keep="first").set_index("hour")["price"].sort_index()
    out = tmp["hour"].map(hourly).astype(float)
    return out.ffill().bfill()


def _simulate_battery_energy(df: pd.DataFrame) -> np.ndarray:
    battery_capacity_mwh = 10.0
    battery_power_limit_mw = 5.0
    battery_efficiency = 0.9
    battery_energy = 0.0
    out: List[float] = []
    wind = pd.to_numeric(df["wind"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    solar = pd.to_numeric(df["solar"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    hydro = pd.to_numeric(df["hydro"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    load = pd.to_numeric(df["load"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    price = pd.to_numeric(df["price"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    avg_price = float(np.mean(price))

    for idx in range(len(price)):
        surplus = max(0.0, float(wind[idx] + solar[idx] + hydro[idx] - load[idx]))
        charge_energy = min(battery_power_limit_mw, surplus) * battery_efficiency
        if float(price[idx]) < avg_price:
            battery_energy += charge_energy
        else:
            discharge = min(battery_power_limit_mw, battery_energy)
            battery_energy -= discharge
        battery_energy = max(0.0, min(battery_capacity_mwh, battery_energy))
        out.append(float(battery_energy))
    return np.asarray(out, dtype=float)


def transform_csv(source: Path, dest: Path, overwrite: bool = False) -> Dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {dest}")

    df = pd.read_csv(source)
    old_price = pd.to_numeric(df["price"], errors="coerce").astype(float)
    df["price"] = _hourly_ffill_price(df)

    if all(col in df.columns for col in ("wind", "solar", "hydro")):
        df["revenue"] = (df["wind"].astype(float) + df["solar"].astype(float) + df["hydro"].astype(float)) * df["price"].astype(float)
    if all(col in df.columns for col in ("wind", "solar", "hydro", "load")):
        df["battery_energy"] = _simulate_battery_energy(df)
    if "revenue" in df.columns:
        revenue = pd.to_numeric(df["revenue"], errors="coerce").astype(float)
        npv = float(revenue.sum() - (1500.0 * 1000.0 + 600000.0 + 1200000.0))
        mean_revenue = float(revenue.mean())
        risk = float(revenue.std() / mean_revenue) if abs(mean_revenue) > 1e-8 else 0.0
        df["npv"] = npv
        df["risk"] = risk
        df["profit_label"] = int(npv > 0.0)

    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    old = old_price.to_numpy(dtype=float)
    new = pd.to_numeric(df["price"], errors="coerce").to_numpy(dtype=float)
    timestamps = pd.to_datetime(df["timestamp"], errors="raise")
    return {
        "source": _project_relative(source),
        "dest": _project_relative(dest),
        "source_sha256": _sha256(source),
        "dest_sha256": _sha256(dest),
        "rows": int(len(df)),
        "start": str(timestamps.iloc[0]),
        "end": str(timestamps.iloc[-1]),
        "old_nonzero_diff_frac": float(np.mean(np.diff(old, prepend=old[0]) != 0.0)) if len(old) else 0.0,
        "new_nonzero_diff_frac": float(np.mean(np.diff(new, prepend=new[0]) != 0.0)) if len(new) else 0.0,
        "price_changed_frac": float(np.mean(old != new)) if len(old) else 0.0,
    }


def transform_folder(source_dir: Path, dest_dir: Path, pattern: str, overwrite: bool) -> List[Dict[str, object]]:
    source_paths = sorted(
        path for path in source_dir.glob(pattern)
        if path.is_file() and not path.name.endswith("_manifest.csv")
    )
    if not source_paths:
        raise FileNotFoundError(f"No files matching {pattern} in {source_dir}")
    rows: List[Dict[str, object]] = []
    for source in source_paths:
        rows.append(transform_csv(source, dest_dir / source.name, overwrite=overwrite))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create forward-filled training protocol folders.")
    parser.add_argument("--training_source", default="training_dataset")
    parser.add_argument("--history_source", default="rolling_past_history_dataset")
    parser.add_argument("--forecast_training_source", default="forecast_training_dataset")
    parser.add_argument("--training_dest", default="training_dataset_ffill")
    parser.add_argument("--history_dest", default="rolling_past_history_dataset_ffill")
    parser.add_argument("--forecast_training_dest", default="forecast_training_dataset_ffill")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    training_rows = transform_folder(
        _as_path(args.training_source),
        _as_path(args.training_dest),
        "scenario_*.csv",
        args.overwrite,
    )
    history_rows = transform_folder(
        _as_path(args.history_source),
        _as_path(args.history_dest),
        "history_*.csv",
        args.overwrite,
    )
    forecast_rows = transform_folder(
        _as_path(args.forecast_training_source),
        _as_path(args.forecast_training_dest),
        "forecast_scenario_*.csv",
        args.overwrite,
    )

    # Evaluation starts after scenario_019 (2024-H2). Its causal bootstrap is
    # therefore history_020, an exact copy of the already-forward-filled final
    # training scenario. Do not transform the price a second time.
    training_dest = _as_path(args.training_dest)
    history_dest = _as_path(args.history_dest)
    final_scenarios = sorted(training_dest.glob("scenario_*.csv"))
    if not final_scenarios:
        raise FileNotFoundError(f"No transformed training scenarios in {training_dest}")
    final_scenario = final_scenarios[-1]
    final_idx = int(final_scenario.stem.split("_")[-1])
    eval_history = history_dest / f"history_{final_idx + 1:03d}.csv"
    if eval_history.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {eval_history}")
    shutil.copy2(final_scenario, eval_history)
    eval_df = pd.read_csv(eval_history, usecols=["timestamp"])
    eval_ts = pd.to_datetime(eval_df["timestamp"], errors="raise")
    history_rows.append({
        "source": _project_relative(final_scenario),
        "dest": _project_relative(eval_history),
        "source_sha256": _sha256(final_scenario),
        "dest_sha256": _sha256(eval_history),
        "rows": int(len(eval_df)),
        "start": str(eval_ts.iloc[0]),
        "end": str(eval_ts.iloc[-1]),
        "source_type": "final_training_episode_eval_handoff",
        "old_nonzero_diff_frac": None,
        "new_nonzero_diff_frac": None,
        "price_changed_frac": 0.0,
    })

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "hourly price held constant on the 10-minute environment grid",
        "training": training_rows,
        "history": history_rows,
        "forecast_training": forecast_rows,
    }

    manifest_records = []
    for row in history_rows:
        history_name = Path(str(row["dest"])).name
        episode = int(Path(history_name).stem.split("_")[-1])
        manifest_records.append({
            "marl_episode": f"{episode:03d}",
            "history_file": history_name,
            "source_file": row["source"],
            "source_type": row.get("source_type", "causal_history"),
            "rows": row["rows"],
            "start": row["start"],
            "end": row["end"],
            "source_sha256": row["source_sha256"],
            "output_sha256": row["dest_sha256"],
        })
    pd.DataFrame(manifest_records).sort_values("marl_episode").to_csv(
        history_dest / "history_manifest.csv",
        index=False,
    )

    out_path = PROJECT_ROOT / "ffill_training_protocol_metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {out_path}")
    print(f"Wrote ffill training data: {_as_path(args.training_dest)}")
    print(f"Wrote ffill histories: {_as_path(args.history_dest)}")
    print(f"Wrote ffill forecast training data: {_as_path(args.forecast_training_dest)}")


if __name__ == "__main__":
    main()

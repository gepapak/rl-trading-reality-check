"""Train a standalone ANN short-horizon forecast bank into a custom episode folder.

This script wraps the archived forecast-engine implementation without changing it.
It trains into a temporary isolated base directory, then moves the resulting
episode_<episode_num> folder to forecast_models/episode_<episode_label>.

Default use case:
  python scripts/train_forecast_episode_standalone.py ^
    --scenario_path forecast_training_dataset_2/forecast_scenario_20_2.csv ^
    --episode_num 20 ^
    --episode_label 20_2 ^
    --forecast_base_dir forecast_models
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = PROJECT_ROOT / "Archive"
DEFAULT_SCENARIO = PROJECT_ROOT / "forecast_training_dataset_2" / "forecast_scenario_20_2.csv"


def _as_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _import_archive_forecast_engine():
    archive = str(ARCHIVE_DIR)
    root = str(PROJECT_ROOT)
    if archive not in sys.path:
        sys.path.insert(0, archive)
    if root not in sys.path:
        sys.path.insert(1, root)
    import forecast_engine  # type: ignore

    return forecast_engine


def _safe_destination(base_dir: Path, episode_label: str) -> Path:
    label = str(episode_label).strip()
    if not label:
        raise ValueError("--episode_label must be non-empty")
    if any(ch in label for ch in "\\/:*?\"<>|"):
        raise ValueError(f"--episode_label contains invalid path characters: {label!r}")
    dest = (base_dir / f"episode_{label}").resolve()
    base = base_dir.resolve()
    if base not in dest.parents:
        raise ValueError(f"Refusing to write outside forecast_base_dir: {dest}")
    return dest


def train(args: argparse.Namespace) -> Path:
    scenario_path = _as_path(args.scenario_path)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario CSV not found: {scenario_path}")

    forecast_base_dir = _as_path(args.forecast_base_dir)
    forecast_base_dir.mkdir(parents=True, exist_ok=True)
    dest = _safe_destination(forecast_base_dir, str(args.episode_label))
    if dest.exists() and not bool(args.force_retrain):
        raise FileExistsError(f"Destination exists; pass --force_retrain to replace: {dest}")

    engine = _import_archive_forecast_engine()
    with tempfile.TemporaryDirectory(prefix="forecast_episode_train_", dir=str(PROJECT_ROOT)) as tmp:
        tmp_base = Path(tmp) / "forecast_models_tmp"
        result = engine.train_episode_forecasts_for_episode(
            episode_num=int(args.episode_num),
            episode_data_path=str(scenario_path),
            forecast_base_dir=str(tmp_base),
            force_retrain=True,
        )
        src = tmp_base / f"episode_{int(args.episode_num)}"
        if not src.exists():
            raise FileNotFoundError(f"Archive trainer did not produce expected folder: {src}")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "source_scenario": str(scenario_path),
        "episode_num_for_training_code": int(args.episode_num),
        "episode_label": str(args.episode_label),
        "output_dir": str(dest),
        "archive_forecast_engine": str(ARCHIVE_DIR / "forecast_engine.py"),
        "result": result,
    }
    meta_path = dest / "standalone_training_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote forecast model folder: {dest}")
    print(f"Wrote metadata: {meta_path}")
    return dest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a standalone forecast bank into forecast_models/episode_<label>.")
    parser.add_argument("--scenario_path", type=str, default=str(DEFAULT_SCENARIO))
    parser.add_argument("--episode_num", type=int, default=20, help="Integer episode number used by archived trainer.")
    parser.add_argument("--episode_label", type=str, default="20_2", help="Output label, default creates episode_20_2.")
    parser.add_argument("--forecast_base_dir", type=str, default="forecast_models")
    parser.add_argument("--force_retrain", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

"""Precompute a standalone evaluation forecast cache from a custom episode model.

Default use case:
  python scripts/build_eval_forecast_cache_standalone.py ^
    --eval_data robustness_ffill_retrained_dk2/evaluation_dataset/unseendata_2.csv ^
    --episode_label 20_2 ^
    --forecast_base_dir forecast_models ^
    --output_cache_dir forecast_cache/forecast_cache_eval_episode20_2025_2

The resulting cache folder can be passed directly to evaluation.py after the
direct-cache resolver patch in evaluation.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = PROJECT_ROOT / "Archive"
DEFAULT_EVAL_DATA = PROJECT_ROOT / "robustness_ffill_retrained_dk2" / "evaluation_dataset" / "unseendata_2.csv"
DEFAULT_OUTPUT_CACHE = PROJECT_ROOT / "forecast_cache" / "forecast_cache_eval_episode20_2025_2"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _as_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _import_archive_components():
    archive = str(ARCHIVE_DIR)
    root = str(PROJECT_ROOT)
    if archive not in sys.path:
        sys.path.insert(0, archive)
    if root not in sys.path:
        sys.path.insert(1, root)
    from config import EnhancedConfig  # type: ignore
    from generator import MultiHorizonForecastGenerator  # type: ignore

    return MultiHorizonForecastGenerator, EnhancedConfig


def _episode_dir(base_dir: Path, episode_label: str) -> Path:
    label = str(episode_label).strip()
    if not label:
        raise ValueError("--episode_label must be non-empty")
    if any(ch in label for ch in "\\/:*?\"<>|"):
        raise ValueError(f"--episode_label contains invalid path characters: {label!r}")
    out = (base_dir / f"episode_{label}").resolve()
    if base_dir.resolve() not in out.parents:
        raise ValueError(f"Episode folder escapes forecast_base_dir: {out}")
    return out


def build_cache(args: argparse.Namespace) -> Path:
    eval_data = _as_path(args.eval_data)
    if not eval_data.exists():
        raise FileNotFoundError(f"Evaluation data not found: {eval_data}")

    forecast_base = _as_path(args.forecast_base_dir)
    episode_dir = _episode_dir(forecast_base, str(args.episode_label))
    if not episode_dir.exists():
        raise FileNotFoundError(f"Forecast model folder not found: {episode_dir}")

    output_cache_dir = _as_path(args.output_cache_dir)
    output_cache_dir.mkdir(parents=True, exist_ok=True)

    forecaster_cls, config_cls = _import_archive_components()
    cfg = config_cls()
    forecaster = forecaster_cls(
        model_dir=str(episode_dir / "models"),
        scaler_dir=str(episode_dir / "scalers"),
        metadata_dir=str(episode_dir / "metadata"),
        look_back=int(getattr(cfg, "forecast_look_back", 24)),
        expert_refresh_stride=int(getattr(cfg, "investment_freq", 6)),
        verbose=bool(args.verbose),
        fallback_mode=False,
        config=cfg,
    )
    if not forecaster.is_complete_stack():
        raise RuntimeError(f"Forecast model stack is incomplete or not loadable: {episode_dir}")

    df = pd.read_csv(eval_data)
    forecaster.precompute_offline(
        df=df,
        timestamp_col="timestamp",
        batch_size=max(1, int(args.batch_size)),
        cache_dir=str(output_cache_dir),
    )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "eval_data": str(eval_data),
        "rows": int(len(df)),
        "episode_label": str(args.episode_label),
        "forecast_model_dir": str(episode_dir),
        "output_cache_dir": str(output_cache_dir),
    }
    meta_path = output_cache_dir / "standalone_cache_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote forecast cache: {output_cache_dir}")
    print(f"Wrote metadata: {meta_path}")
    return output_cache_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a standalone eval forecast cache from a custom episode model.")
    parser.add_argument("--eval_data", type=str, default=str(DEFAULT_EVAL_DATA))
    parser.add_argument("--episode_label", type=str, default="20_2")
    parser.add_argument("--forecast_base_dir", type=str, default="forecast_models")
    parser.add_argument("--output_cache_dir", type=str, default=str(DEFAULT_OUTPUT_CACHE))
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    build_cache(parse_args())


if __name__ == "__main__":
    main()

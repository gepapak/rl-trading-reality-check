"""Train v2 hourly-grid forecast models.

This wraps forecast_engine with an explicit hourly forecast configuration:
24 hourly look-back rows and a 1-hour short horizon.  The existing
forecast_engine CLI should not be used directly for hourly v2 because its
default short horizon is six 10-minute steps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EnhancedConfig
from forecast_engine import train_episode_forecasts_batch


def _as_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _hourly_v2_config(look_back_hours: int, horizon_hours: int) -> EnhancedConfig:
    cfg = EnhancedConfig()
    cfg.forecast_look_back = int(look_back_hours)
    cfg.forecast_horizons = dict(getattr(cfg, "forecast_horizons", {}) or {})
    cfg.forecast_horizons["short"] = int(horizon_hours)
    cfg.forecast_targets = ["price"]
    # Precompute for the hourly cache should generate one forecast per hourly
    # origin.  This is harmless during training and keeps model validation
    # helpers aligned if they are called by forecast_engine.
    cfg.investment_freq = 1
    return cfg


def _parse_episodes(args: argparse.Namespace) -> list[int]:
    if args.all:
        return list(range(21))
    if args.episodes:
        return [int(ep) for ep in args.episodes]
    raise ValueError("Specify --all or --episodes")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hourly v2 ANN forecast banks.")
    parser.add_argument("--episodes", type=int, nargs="+", default=None)
    parser.add_argument("--all", action="store_true", help="Train rolling episodes 0..20, including reserved eval episode 20")
    parser.add_argument("--forecast_training_dataset_dir", default="forecast_training_dataset_settlement_hourly_v2")
    parser.add_argument("--forecast_base_dir", default="forecast_models_settlement_hourly_v2")
    parser.add_argument("--look_back_hours", type=int, default=24)
    parser.add_argument("--horizon_hours", type=int, default=1)
    parser.add_argument(
        "--training_loss",
        choices=["statistical", "decision_focused_v1"],
        default="statistical",
        help=(
            "'statistical' reproduces the frozen v2 banks (MSE + BCE). "
            "'decision_focused_v1' trains the price head on the profit-weighted "
            "soft-position utility and the direction head with |spread|-weighted "
            "BCE; use a separate --forecast_base_dir so v2 banks stay frozen."
        ),
    )
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    episodes = _parse_episodes(args)
    train_dir = _as_path(args.forecast_training_dataset_dir)
    model_dir = _as_path(args.forecast_base_dir)

    training_loss = str(args.training_loss)
    if training_loss == "decision_focused_v1" and args.forecast_base_dir == "forecast_models_settlement_hourly_v2":
        raise ValueError(
            "decision_focused_v1 must not overwrite the frozen statistical v2 banks; "
            "pass a separate --forecast_base_dir so model banks cannot be mixed."
        )
    import os as _os
    _os.environ["FORECAST_TRAINING_LOSS"] = training_loss

    missing = [
        train_dir / f"forecast_scenario_{int(ep):02d}.csv"
        for ep in episodes
        if not (train_dir / f"forecast_scenario_{int(ep):02d}.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing hourly v2 forecast training file(s):\n"
            + "\n".join(f"  {p}" for p in missing[:10])
            + ("\n  ..." if len(missing) > 10 else "")
        )

    print("Hourly forecast model v2 configuration")
    print(f"  forecast_training_dataset_dir: {train_dir}")
    print(f"  forecast_base_dir: {model_dir}")
    print(f"  episodes: {episodes}")
    print(f"  look_back_hours: {int(args.look_back_hours)}")
    print(f"  horizon_hours: {int(args.horizon_hours)}")
    print(f"  training_loss: {training_loss}")
    print(f"  force_retrain: {bool(args.force_retrain)}")

    if args.dry_run:
        print("\nDry run complete. No models were trained.")
        return

    cfg = _hourly_v2_config(
        look_back_hours=int(args.look_back_hours),
        horizon_hours=int(args.horizon_hours),
    )
    results = train_episode_forecasts_batch(
        episodes=episodes,
        forecast_training_dataset_dir=str(train_dir),
        forecast_base_dir=str(model_dir),
        force_retrain=bool(args.force_retrain),
        config=cfg,
    )
    successful = sum(int(r.get("successful", 0)) for r in results)
    failed = sum(int(r.get("failed", 0)) for r in results)
    print("\nHourly forecast model v2 training complete.")
    print(f"Total successful expert fits: {successful}")
    print(f"Total failed expert fits: {failed}")
    print(f"Models written under: {model_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])

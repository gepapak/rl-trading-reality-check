#!/usr/bin/env python3
"""Bootstrap investor-only daily paths with time dependence preserved.

The investor-only path removes cumulative battery cash from the reported
trading sleeve. Deterministic controllers use one market path and a circular
moving-block bootstrap over daily returns. Learned methods additionally sample
the training seed on each bootstrap draw.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _seed(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _latest_per_seed(paths: list[Path]) -> dict[int, Path]:
    latest: dict[int, Path] = {}
    for path in sorted(paths, key=lambda item: item.stat().st_mtime):
        seed = _seed(path)
        if seed is not None:
            latest[seed] = path
    return latest


def _investor_daily_returns(
    path: Path,
    *,
    initial_sleeve_dkk: float,
) -> np.ndarray:
    header = set(pd.read_csv(path, nrows=0).columns)
    required = {"trading_cash_dkk", "financial_mtm_dkk"}
    missing = required - header
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    columns = sorted(
        required
        | (
            {
                "distribution_adjusted_trading_sleeve_dkk",
                "battery_cash_delta",
            }
            & header
        )
    )
    df = pd.read_csv(path, usecols=columns)
    cash = pd.to_numeric(df["trading_cash_dkk"], errors="coerce").fillna(0.0).to_numpy()
    mtm = pd.to_numeric(df["financial_mtm_dkk"], errors="coerce").fillna(0.0).to_numpy()
    if "distribution_adjusted_trading_sleeve_dkk" in df.columns:
        sleeve = (
            pd.to_numeric(
                df["distribution_adjusted_trading_sleeve_dkk"],
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy()
        )
    else:
        sleeve = cash + mtm
    if "battery_cash_delta" in df.columns:
        battery = (
            pd.to_numeric(df["battery_cash_delta"], errors="coerce")
            .fillna(0.0)
            .to_numpy()
        )
        sleeve = sleeve - np.cumsum(battery)
    finite = np.isfinite(sleeve)
    sleeve = np.asarray(sleeve[finite], dtype=np.float64)
    if sleeve.size < 145:
        raise ValueError(f"{path} has too few valid rows for daily metrics")
    complete_days = int(sleeve.size // 144)
    if complete_days < 1:
        raise ValueError(f"{path} has no complete 144-step day")
    daily = np.concatenate(
        (
            np.asarray([float(initial_sleeve_dkk)], dtype=np.float64),
            sleeve[143 : complete_days * 144 : 144],
        )
    )
    previous = daily[:-1]
    current = daily[1:]
    valid = np.isfinite(previous) & np.isfinite(current) & (np.abs(previous) > 1e-12)
    return np.asarray((current[valid] - previous[valid]) / previous[valid], dtype=np.float64)


def _acf(values: np.ndarray, lag: int) -> float:
    if values.size <= lag + 1:
        return float("nan")
    a = values[:-lag] - np.mean(values[:-lag])
    b = values[lag:] - np.mean(values[lag:])
    denominator = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / denominator) if denominator > 0.0 else float("nan")


def _hac_vif(values: np.ndarray, max_lag: int = 7) -> float:
    if values.size < 3:
        return 1.0
    total = 1.0
    lag_limit = min(int(max_lag), values.size - 2)
    for lag in range(1, lag_limit + 1):
        rho = _acf(values, lag)
        if math.isfinite(rho):
            total += 2.0 * (1.0 - lag / (lag_limit + 1.0)) * rho
    return float(max(total, 1e-12))


def _statistics(returns: np.ndarray) -> dict[str, float]:
    path = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    peak = np.maximum.accumulate(path)
    drawdown = float(np.max(np.where(peak > 0.0, (peak - path) / peak, 0.0)))
    volatility = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    risk_free_daily = (1.0 + 0.02) ** (1.0 / 365.25) - 1.0
    sharpe = (
        float(
            np.mean(returns - risk_free_daily)
            / volatility
            * math.sqrt(365.25)
            / math.sqrt(_hac_vif(returns, 7))
        )
        if volatility > 0.0
        else 0.0
    )
    return {
        "return_pct": float((path[-1] - 1.0) * 100.0),
        "daily_hac7_sharpe": sharpe,
        "max_drawdown_pct": drawdown * 100.0,
        "annualized_volatility": volatility * math.sqrt(365.25),
    }


def _circular_block_sample(
    values: np.ndarray,
    *,
    block_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(values.size)
    block = min(max(int(block_days), 1), n)
    chunks: list[np.ndarray] = []
    while sum(chunk.size for chunk in chunks) < n:
        start = int(rng.integers(0, n))
        indices = (start + np.arange(block, dtype=int)) % n
        chunks.append(values[indices])
    return np.concatenate(chunks)[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--env_logs_glob", nargs="+", required=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--deterministic_tolerance", type=float, default=1e-12)
    parser.add_argument("--block_days", type=int, default=7)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output_dir", default="results/investor_path_bootstrap")
    parser.add_argument(
        "--initial_trading_sleeve_dkk",
        type=float,
        default=None,
        help="Defaults to EnhancedConfig.init_budget * financial_allocation.",
    )
    args = parser.parse_args()

    matched: list[Path] = []
    for pattern in args.env_logs_glob:
        resolved = pattern if os.path.isabs(pattern) else str(PROJECT_ROOT / pattern)
        matched.extend(Path(item) for item in glob.glob(resolved, recursive=True))
    by_seed = _latest_per_seed([path for path in matched if path.is_file()])
    if not by_seed:
        raise FileNotFoundError(f"No seed-tagged env logs matched {args.env_logs_glob}")

    if args.initial_trading_sleeve_dkk is None:
        from config import EnhancedConfig

        cfg = EnhancedConfig()
        initial_sleeve_dkk = float(cfg.init_budget) * float(cfg.financial_allocation)
    else:
        initial_sleeve_dkk = float(args.initial_trading_sleeve_dkk)
    series = {
        seed: _investor_daily_returns(
            path,
            initial_sleeve_dkk=initial_sleeve_dkk,
        )
        for seed, path in by_seed.items()
    }
    if args.deterministic and len(series) > 1:
        lengths = {values.size for values in series.values()}
        if len(lengths) != 1:
            raise RuntimeError("Deterministic replicas have different daily-path lengths")
        matrix = np.vstack(list(series.values()))
        replica_spread = float(np.max(np.ptp(matrix, axis=0)))
        if replica_spread > float(args.deterministic_tolerance):
            raise RuntimeError(
                "Deterministic investor replicas differ: "
                f"max daily-return spread={replica_spread:.6g} exceeds "
                f"tolerance={float(args.deterministic_tolerance):.6g}"
            )
    else:
        replica_spread = 0.0

    rng = np.random.default_rng(int(args.seed))
    seeds = np.asarray(sorted(series), dtype=int)
    draws: dict[str, list[float]] = {
        key: [] for key in _statistics(next(iter(series.values())))
    }
    for _ in range(max(int(args.resamples), 1)):
        selected = int(seeds[0]) if args.deterministic else int(rng.choice(seeds))
        sampled = _circular_block_sample(
            series[selected],
            block_days=int(args.block_days),
            rng=rng,
        )
        stats = _statistics(sampled)
        for key, value in stats.items():
            draws[key].append(float(value))

    point_by_seed = {str(seed): _statistics(values) for seed, values in series.items()}
    intervals: dict[str, dict[str, float]] = {}
    for key, values in draws.items():
        array = np.asarray(values, dtype=np.float64)
        intervals[key] = {
            "bootstrap_mean": float(np.mean(array)),
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
        }

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"investor_path_bootstrap_{args.label}.json"
    payload: dict[str, Any] = {
        "label": args.label,
        "deterministic": bool(args.deterministic),
        "uncertainty_unit": (
            "daily_market_path_blocks"
            if args.deterministic
            else "training_seed_plus_daily_market_path_blocks"
        ),
        "bootstrap_method": "circular_moving_block",
        "path_convention": "true_initial_plus_complete_day_ends_v1",
        "initial_trading_sleeve_dkk": initial_sleeve_dkk,
        "block_days": int(args.block_days),
        "resamples": int(args.resamples),
        "replica_max_daily_return_spread": replica_spread,
        "files": {str(seed): str(path) for seed, path in by_seed.items()},
        "point_estimates_by_seed": point_by_seed,
        "bootstrap_intervals": intervals,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

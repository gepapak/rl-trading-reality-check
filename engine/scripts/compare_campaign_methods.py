#!/usr/bin/env python3
"""Create paired training-seed comparisons from detailed campaign metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COMPARISONS = [
    ("feasible_action_full", "mappo_marl"),
    ("forecast_observation_only", "mappo_marl"),
    ("cfm_zero_trust", "mappo_marl"),
    ("cfm_zero_trust_forecast_obs", "cfm_zero_trust"),
    ("feasible_action_full", "cfm_zero_trust_forecast_obs"),
    ("feasible_action_full", "cfm_zero_trust"),
    ("feasible_action_full", "forecast_observation_only"),
    ("feasible_action_full", "unconditioned_capacity"),
    ("feasible_action_full", "feasible_action_no_paired_reward"),
]

DEFAULT_METRICS = [
    "investor_contract_return_pct",
    "investor_contract_daily_hac7_sharpe_ratio",
    "investor_contract_max_drawdown_pct",
    "investor_contract_annualized_volatility",
]


def _two_sided_sign_p(wins: int, losses: int) -> float:
    n = int(wins + losses)
    if n <= 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2.0**n)
    return float(min(1.0, 2.0 * probability))


def _bootstrap_delta(
    values: np.ndarray,
    *,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(int(resamples), values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics_csv",
        default="results/detailed_metrics/all_detailed_metrics.csv",
    )
    parser.add_argument(
        "--comparisons",
        nargs="+",
        help="Pairs formatted method_a:method_b. Defaults to paper comparisons.",
    )
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260724)
    parser.add_argument("--output_dir", default="results/detailed_metrics")
    args = parser.parse_args()

    csv_path = Path(args.metrics_csv)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    frame = pd.read_csv(csv_path)
    required = {"arm", "region", "seed"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    comparisons = DEFAULT_COMPARISONS
    if args.comparisons:
        comparisons = []
        for value in args.comparisons:
            if ":" not in value:
                raise ValueError(
                    f"Comparison must be method_a:method_b, got {value!r}"
                )
            comparisons.append(tuple(value.split(":", 1)))

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.bootstrap_seed))
    for method_a, method_b in comparisons:
        regions = sorted(frame["region"].dropna().astype(str).unique())
        for region in regions:
            a = frame[
                (frame["arm"] == method_a) & (frame["region"] == region)
            ].copy()
            b = frame[
                (frame["arm"] == method_b) & (frame["region"] == region)
            ].copy()
            if a.empty or b.empty:
                continue
            a["seed"] = pd.to_numeric(a["seed"], errors="coerce")
            b["seed"] = pd.to_numeric(b["seed"], errors="coerce")
            for metric in args.metrics:
                if metric not in frame.columns:
                    continue
                left = a[["seed", metric]].rename(columns={metric: "a"})
                right = b[["seed", metric]].rename(columns={metric: "b"})
                paired = left.merge(right, on="seed", how="inner")
                paired["a"] = pd.to_numeric(paired["a"], errors="coerce")
                paired["b"] = pd.to_numeric(paired["b"], errors="coerce")
                paired = paired.replace([np.inf, -np.inf], np.nan).dropna()
                if paired.empty:
                    continue
                delta = (paired["a"] - paired["b"]).to_numpy(dtype=np.float64)
                lower_is_better = metric.endswith(
                    "drawdown_pct"
                ) or metric.endswith("annualized_volatility")
                oriented = -delta if lower_is_better else delta
                wins = int(np.sum(oriented > 0.0))
                losses = int(np.sum(oriented < 0.0))
                ties = int(np.sum(oriented == 0.0))
                lo, hi = _bootstrap_delta(
                    delta,
                    resamples=int(args.resamples),
                    rng=rng,
                )
                rows.append(
                    {
                        "method_a": method_a,
                        "method_b": method_b,
                        "region": region,
                        "metric": metric,
                        "n_paired_seeds": int(delta.size),
                        "seeds": ",".join(
                            str(int(seed)) for seed in paired["seed"]
                        ),
                        "method_a_mean": float(np.mean(paired["a"])),
                        "method_b_mean": float(np.mean(paired["b"])),
                        "paired_delta_a_minus_b_mean": float(np.mean(delta)),
                        "paired_delta_ci95_low": lo,
                        "paired_delta_ci95_high": hi,
                        "wins_a": wins,
                        "losses_a": losses,
                        "ties": ties,
                        "two_sided_sign_test_p": _two_sided_sign_p(wins, losses),
                        "preferred_direction": (
                            "lower_is_better"
                            if lower_is_better
                            else "higher_is_better"
                        ),
                        "uncertainty_unit": "paired_training_seed",
                    }
                )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output = output_dir / "paired_method_comparisons.csv"
    json_output = output_dir / "paired_method_comparisons.json"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {csv_output}")
    print(f"Wrote {json_output}")
    print(f"Paired comparison rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build metric-family comparison tables for FoCAL paper evaluation."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional


SUITE_FILES = {
    "tiers_tier1_baseline": {
        "label": "Tier1 baseline",
        "v1_summary": "batch_tier_phase_runs/tiers_tier1_baseline/tier1_only_seed_suite_summary.csv",
    },
    "tiers_tier1_forecast_utilization": {
        "label": "FoCAL / Tier1+forecast",
        "v1_summary": "batch_tier_phase_runs/tiers_tier1_forecast_utilization/tier1_forecast_utilization_only_seed_suite_summary.csv",
    },
    "abl_forecast_prior_only": {
        "label": "Ablation: forecast prior only",
        "v1_summary": "batch_tier_phase_runs/abl_forecast_prior_only/tier1_forecast_utilization_only_seed_suite_summary.csv",
    },
    "abl_no_prior_path_control": {
        "label": "Ablation: no prior path control",
        "v1_summary": "batch_tier_phase_runs/abl_no_prior_path_control/tier1_forecast_utilization_only_seed_suite_summary.csv",
    },
    "abl_weak_conformal_gate": {
        "label": "Ablation: weak conformal gate",
        "v1_summary": "batch_tier_phase_runs/abl_weak_conformal_gate/tier1_forecast_utilization_only_seed_suite_summary.csv",
    },
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def values(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    vals = []
    for row in rows:
        val = _safe_float(row.get(key))
        if val is not None:
            vals.append(val)
    return vals


def summarize(vals: List[float], scale: float = 1.0) -> Dict[str, Any]:
    scaled = [v * scale for v in vals]
    if not scaled:
        return {"n": 0, "mean": "", "std": ""}
    return {
        "n": len(scaled),
        "mean": mean(scaled),
        "std": pstdev(scaled) if len(scaled) > 1 else 0.0,
    }


def add_family_row(
    out: List[Dict[str, Any]],
    *,
    suite: str,
    label: str,
    family: str,
    source: str,
    rows: List[Dict[str, Any]],
    return_key: str,
    sharpe_key: str,
    drawdown_key: str,
    distributions_key: Optional[str] = None,
) -> None:
    ret = summarize(values(rows, return_key), scale=100.0)
    sharpe = summarize(values(rows, sharpe_key), scale=1.0)
    dd = summarize(values(rows, drawdown_key), scale=100.0)
    dist = summarize(values(rows, distributions_key), scale=1.0) if distributions_key else {"n": 0, "mean": "", "std": ""}
    out.append(
        {
            "suite": suite,
            "variant": label,
            "metric_family": family,
            "source": source,
            "n_return": ret["n"],
            "return_mean_pct": ret["mean"],
            "return_std_pct": ret["std"],
            "n_sharpe": sharpe["n"],
            "sharpe_mean": sharpe["mean"],
            "sharpe_std": sharpe["std"],
            "n_max_drawdown": dd["n"],
            "max_drawdown_mean_pct": dd["mean"],
            "max_drawdown_std_pct": dd["std"],
            "n_distributions": dist["n"],
            "total_distributions_mean_usd": dist["mean"],
            "total_distributions_std_usd": dist["std"],
        }
    )


def load_leakage_rows(paths: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
    mapped: Dict[str, List[Dict[str, Any]]] = {}
    suite_alias = {
        "tier1_baseline": "tiers_tier1_baseline",
        "tier1_forecast_utilization": "tiers_tier1_forecast_utilization",
        "abl_forecast_prior_only": "abl_forecast_prior_only",
        "abl_no_prior_path_control": "abl_no_prior_path_control",
        "abl_weak_conformal_gate": "abl_weak_conformal_gate",
    }
    for path in paths:
        for row in read_csv(path):
            raw_suite = str(row.get("suite", "") or "")
            suite = suite_alias.get(raw_suite, raw_suite)
            if suite:
                mapped.setdefault(suite, []).append(row)
    return mapped


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    val = _safe_float(value)
    if val is None:
        return "n/a"
    return f"{val:.{digits}f}"


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# FoCAL Metric-Family Comparison\n\n")
        f.write("All returns/drawdowns are percentages. Distribution values are USD.\n\n")
        f.write("| Variant | Family | N | Return mean +/- std | Sharpe mean +/- std | Max DD mean +/- std | Distributions mean +/- std |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            n = row.get("n_return", 0)
            f.write(
                f"| {row['variant']} | {row['metric_family']} | {n} | "
                f"{_fmt(row.get('return_mean_pct'))} +/- {_fmt(row.get('return_std_pct'))} | "
                f"{_fmt(row.get('sharpe_mean'))} +/- {_fmt(row.get('sharpe_std'))} | "
                f"{_fmt(row.get('max_drawdown_mean_pct'))} +/- {_fmt(row.get('max_drawdown_std_pct'))} | "
                f"{_fmt(row.get('total_distributions_mean_usd'), 1)} +/- {_fmt(row.get('total_distributions_std_usd'), 1)} |\n"
            )
        f.write("\nNote: trading-sleeve rows are only as complete as the available leakage/sleeve diagnostic CSVs.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FoCAL eval_v2 metric-family comparison tables.")
    parser.add_argument("--output_root", default="eval_v2")
    parser.add_argument(
        "--leakage_csv",
        nargs="*",
        default=[
            "batch_tier_phase_runs/leakage_sleeve_diagnostics_completed/completed_leakage_sleeve_seed7_plus_seed42_baseline.csv",
            "batch_tier_phase_runs/leakage_sleeve_diagnostics_seed42/leakage_sleeve_diagnostics_seed7_42.csv",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    leakage = load_leakage_rows([Path(p) for p in args.leakage_csv])
    rows: List[Dict[str, Any]] = []

    for suite, spec in SUITE_FILES.items():
        label = spec["label"]
        v1_rows = read_csv(Path(spec["v1_summary"]))
        no_sweeper_rows = read_csv(out_root / "no_sweeper" / suite / f"{suite}_no_sweeper_seed_summary.csv")
        add_family_row(
            rows,
            suite=suite,
            label=label,
            family="distribution_adjusted_wealth",
            source="v1_seed_suite_summary",
            rows=v1_rows,
            return_key="total_return",
            sharpe_key="sharpe_ratio",
            drawdown_key="max_drawdown",
            distributions_key="total_distributions_usd",
        )
        add_family_row(
            rows,
            suite=suite,
            label=label,
            family="reported_nav",
            source="v1_seed_suite_summary",
            rows=v1_rows,
            return_key="reported_nav_total_return",
            sharpe_key="reported_nav_sharpe_ratio",
            drawdown_key="reported_nav_max_drawdown",
            distributions_key="total_distributions_usd",
        )
        add_family_row(
            rows,
            suite=suite,
            label=label,
            family="no_sweeper_stress_eval",
            source="eval_v2_no_sweeper",
            rows=no_sweeper_rows,
            return_key="reported_nav_total_return",
            sharpe_key="reported_nav_sharpe_ratio",
            drawdown_key="reported_nav_max_drawdown",
            distributions_key="total_distributions_usd",
        )
        sleeve_rows = leakage.get(suite, [])
        add_family_row(
            rows,
            suite=suite,
            label=label,
            family="trading_sleeve_partial",
            source="leakage_sleeve_diagnostics",
            rows=sleeve_rows,
            return_key="trading_return_pct",
            sharpe_key="sleeve_trading_sharpe_ratio",
            drawdown_key="sleeve_trading_max_drawdown_pct",
            distributions_key="total_distributions_usd",
        )

    write_csv(out_root / "metric_families.csv", rows)
    write_markdown(out_root / "COMPARISON_metric_families.md", rows)
    print(f"Wrote {out_root / 'metric_families.csv'}")
    print(f"Wrote {out_root / 'COMPARISON_metric_families.md'}")


if __name__ == "__main__":
    main()

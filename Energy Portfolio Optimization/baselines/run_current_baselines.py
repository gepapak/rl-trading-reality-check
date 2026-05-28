from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BASELINES_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINES_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
    except Exception:
        pass
    return str(value)


def run_command(cmd: list[str], cwd: Path) -> None:
    print("")
    print("=" * 72)
    print(" ".join(str(x) for x in cmd))
    print("=" * 72)
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def load_summary(output_dir: Path) -> Dict[str, Any]:
    for name in ("summary_metrics.json", "summary.json", "evaluation_results.json"):
        path = output_dir / name
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("run_dir", str(output_dir))
            return data
    return {
        "status": "missing_summary",
        "error": f"No summary_metrics.json, summary.json, or evaluation_results.json in {output_dir}",
        "run_dir": str(output_dir),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _first_float(data: Dict[str, Any], keys: Iterable[str], default: float = 0.0) -> float:
    for key in keys:
        if key in data:
            return _safe_float(data.get(key), default)
    return default


def normalize_row(name: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    status = str(summary.get("status", "completed"))
    method = str(summary.get("method", name))
    role = "diagnostic" if name == "IEEE_Compliance_Diagnostic" else "financial_strategy"

    final_primary = _first_float(
        summary,
        ("final_portfolio_value", "distribution_adjusted_final_value_usd", "final_value_usd"),
    )
    initial_primary = _first_float(
        summary,
        ("initial_portfolio_value", "initial_value_usd"),
        800_000_000.0,
    )
    reported_final = _first_float(
        summary,
        ("reported_nav_final_portfolio_value", "final_value_usd", "final_portfolio_value"),
    )
    reported_return = _first_float(
        summary,
        ("reported_nav_total_return", "raw_nav_total_return", "total_return"),
    )
    valid_flag = summary.get("valid_for_publication_comparison", True)
    ranked = (
        status in {"completed", "completed_with_warnings"}
        and role != "diagnostic"
        and bool(valid_flag)
    )

    return {
        "baseline": name,
        "role": role,
        "ranked_financial_comparison": ranked,
        "status": status,
        "method": method,
        "final_portfolio_value_usd": final_primary,
        "initial_portfolio_value_usd": initial_primary,
        "total_return_pct": 100.0 * _safe_float(summary.get("total_return")),
        "annual_return_pct": 100.0 * _safe_float(summary.get("annual_return")),
        "sharpe_ratio": _safe_float(summary.get("sharpe_ratio")),
        "max_drawdown_pct": 100.0 * _safe_float(summary.get("max_drawdown")),
        "volatility": _safe_float(summary.get("volatility")),
        "total_distributions_usd": _safe_float(summary.get("total_distributions_usd")),
        "reported_nav_final_usd": reported_final,
        "reported_nav_return_pct": 100.0 * reported_return,
        "reported_nav_sharpe": _safe_float(summary.get("reported_nav_sharpe_ratio")),
        "models_loaded": int(_safe_float(summary.get("models_loaded"), 0.0)),
        "current_codebase_environment": bool(summary.get("current_codebase_environment", True)),
        "run_dir": str(summary.get("run_dir", "")),
        "skip_reason": str(summary.get("skip_reason", "")),
    }


def write_baseline_report(
    *,
    output_root: Path,
    results: Dict[str, Dict[str, Any]],
    data_path: Path,
    timesteps: int,
    seed: int,
) -> Tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = [normalize_row(name, summary) for name, summary in results.items()]

    csv_path = output_root / f"baseline_comparison_{timestamp}.csv"
    fieldnames = list(rows[0].keys()) if rows else ["baseline", "status"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_root / f"baseline_comparison_{timestamp}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("## Baseline Evaluation Report\n\n")
        f.write(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Evaluation data: `{data_path}`\n")
        f.write(f"- Timesteps requested: `{timesteps}`\n")
        f.write(f"- Seed: `{seed}`\n")
        f.write(f"- Evaluation contract: `tier1_2026_distribution_adjusted`\n")
        f.write(f"- Output CSV: `{csv_path.name}`\n\n")

        f.write("### Summary Table\n\n")
        f.write("| Baseline | Role | Ranked | Status | Final Wealth USD | Return % | Sharpe | Max DD % | Raw NAV Return % |\n")
        f.write("|---|---|---:|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['baseline']} | {row['role']} | {str(row['ranked_financial_comparison'])} | "
                f"{row['status']} | {row['final_portfolio_value_usd']:.2f} | "
                f"{row['total_return_pct']:.4f} | {row['sharpe_ratio']:.4f} | "
                f"{row['max_drawdown_pct']:.4f} | {row['reported_nav_return_pct']:.4f} |\n"
            )

        f.write("\nPrimary financial metrics use distribution-adjusted investor wealth, matching the current MARL evaluation contract.\n")
        f.write("Raw NAV is kept as a separate reported diagnostic because shareholder distributions reduce accounting NAV.\n")
        f.write("IEEE_Compliance_Diagnostic is a standards-compliance diagnostic and is not ranked as a financial strategy.\n\n")

        f.write("### Output Directories\n\n")
        for row in rows:
            f.write(f"- {row['baseline']}: `{row['run_dir']}`\n")
            if row["skip_reason"]:
                f.write(f"  - skip reason: {row['skip_reason']}\n")

    json_path = output_root / f"baseline_evaluation_{timestamp}.json"
    payload = {
        "evaluation_type": "baseline_comparison",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "eval_data_path": str(data_path),
        "timesteps_requested": int(timesteps),
        "seed": int(seed),
        "evaluation_contract": "tier1_2026_distribution_adjusted",
        "report_csv": str(csv_path),
        "report_md": str(md_path),
        "baseline_results": results,
        "comparison_rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    latest_path = output_root / "baseline_evaluation_latest.json"
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    return csv_path, md_path, json_path


def write_baseline_report(
    *,
    output_root: Path,
    results: Dict[str, Dict[str, Any]],
    data_path: Path,
    timesteps: int,
    seed: int,
) -> Path:
    """Write the publication baseline result as one aggregate JSON file only."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = [normalize_row(name, summary) for name, summary in results.items()]

    json_path = output_root / f"baseline_evaluation_{timestamp}.json"
    payload = {
        "evaluation_type": "baseline_comparison",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "eval_data_path": str(data_path),
        "timesteps_requested": int(timesteps),
        "seed": int(seed),
        "evaluation_contract": "tier1_2026_distribution_adjusted",
        "artifact_policy": "single_aggregate_json",
        "baseline_results": results,
        "comparison_rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    return json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run current publication baselines.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=PROJECT_ROOT / "evaluation_dataset" / "unseendata.csv",
        help="Evaluation CSV. Default: evaluation_dataset/unseendata.csv",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "baseline_results" / "current_2025",
        help="Directory where baseline outputs are written.",
    )
    parser.add_argument("--timesteps", type=int, default=39305)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--traditional_method",
        default="markowitz_mean_variance",
        choices=["equal_weight", "min_variance", "max_sharpe", "markowitz_mean_variance", "risk_parity"],
    )
    parser.add_argument(
        "--include_ieee",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip_ieee",
        action="store_true",
        help="Skip the IEEE compliance diagnostic tool.",
    )
    args = parser.parse_args()

    data_path = args.data_path.resolve()
    output_root = args.output_root.resolve()
    if not data_path.is_file():
        raise SystemExit(f"Evaluation data not found: {data_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.include_ieee:
        from evaluation import run_traditional_baselines

        results = run_traditional_baselines(
            str(data_path),
            timesteps=int(args.timesteps),
            output_dir=str(output_root),
            seed=int(args.seed),
        )
        json_path = write_baseline_report(
            output_root=output_root,
            results=results,
            data_path=data_path,
            timesteps=args.timesteps,
            seed=args.seed,
        )
        print("")
        print(f"Baseline run complete: {output_root}")
        print(f"Baseline JSON: {json_path}")
        return 0

    py = sys.executable
    runs: List[Tuple[str, List[str], Path]] = [
        (
            "Baseline1_TraditionalPortfolio",
            [
                py,
                str(BASELINES_ROOT / "Baseline1_TraditionalPortfolio" / "run_traditional_baseline.py"),
                "--data_path",
                str(data_path),
                "--output_dir",
                str(output_root / "Baseline1_TraditionalPortfolio"),
                "--timesteps",
                str(args.timesteps),
                "--method",
                args.traditional_method,
                "--seed",
                str(args.seed),
            ],
            output_root / "Baseline1_TraditionalPortfolio",
        ),
        (
            "Baseline2_RuleBasedHeuristic",
            [
                py,
                str(BASELINES_ROOT / "Baseline2_RuleBasedHeuristic" / "run_rule_based_baseline.py"),
                "--data_path",
                str(data_path),
                "--output_dir",
                str(output_root / "Baseline2_RuleBasedHeuristic"),
                "--timesteps",
                str(args.timesteps),
                "--seed",
                str(args.seed),
            ],
            output_root / "Baseline2_RuleBasedHeuristic",
        ),
        (
            "Baseline3_BuyAndHold",
            [
                py,
                str(BASELINES_ROOT / "Baseline3_BuyAndHold" / "run_buy_and_hold_baseline.py"),
                "--data_path",
                str(data_path),
                "--output_dir",
                str(output_root / "Baseline3_BuyAndHold"),
                "--timesteps",
                str(args.timesteps),
            ],
            output_root / "Baseline3_BuyAndHold",
        ),
    ]

    if not args.skip_ieee:
        runs.append(
            (
                "IEEE_Compliance_Diagnostic",
                [
                    py,
                    str(BASELINES_ROOT / "IEEE Compliance Tool" / "run_ieee_baseline.py"),
                    "--data_path",
                    str(data_path),
                    "--output_dir",
                    str(output_root / "IEEE_Compliance_Diagnostic"),
                    "--timesteps",
                    str(args.timesteps),
                    "--seed",
                    str(args.seed),
                ],
                output_root / "IEEE_Compliance_Diagnostic",
            )
        )

    results: Dict[str, Dict[str, Any]] = {}
    for name, cmd, run_dir in runs:
        run_command(cmd, PROJECT_ROOT)
        results[name] = load_summary(run_dir)

    json_path = write_baseline_report(
        output_root=output_root,
        results=results,
        data_path=data_path,
        timesteps=args.timesteps,
        seed=args.seed,
    )

    print("")
    print(f"Baseline run complete: {output_root}")
    print(f"Baseline JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

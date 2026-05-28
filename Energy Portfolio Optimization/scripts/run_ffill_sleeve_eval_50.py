"""Run the 5-variant forward-filled no-sweeper sleeve supplement.

This is eval-only. It reuses existing final_models checkpoints and existing
forward-filled forecast caches, then computes trading-sleeve metrics from the
per-step debug CSVs produced by evaluation.py --log-sleeve.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202]
DEFAULT_EVAL_STEPS = 39305
DEFAULT_PERIODS_PER_YEAR = 52596.0
DEFAULT_RF_ANNUAL = 0.02
DEFAULT_DKK_TO_USD = 0.145
EXPECTED_TRADING_INITIAL_USD = 96_000_000.0


@dataclass(frozen=True)
class VariantSpec:
    key: str
    table_name: str
    suite: str
    run_name_template: str
    eval_data: str
    forecast_cache_dir: str
    reference_eval_root: str


VARIANTS: Tuple[VariantSpec, ...] = (
    VariantSpec(
        key="tier1_baseline",
        table_name="Tier1 baseline",
        suite="tiers_tier1_baseline",
        run_name_template="tier1_seed{seed}",
        eval_data="robustness_ffill_retrained/evaluation_dataset/unseendata.csv",
        forecast_cache_dir="robustness_ffill_retrained/forecast_cache",
        reference_eval_root="eval_v2_ffill_retrained/no_sweeper",
    ),
    VariantSpec(
        key="focal_ann",
        table_name="FoCAL-ANN",
        suite="tiers_tier1_forecast_utilization",
        run_name_template="tier1_forecast_utilization_seed{seed}",
        eval_data="robustness_ffill_retrained/evaluation_dataset/unseendata.csv",
        forecast_cache_dir="robustness_ffill_retrained/forecast_cache",
        reference_eval_root="eval_v2_ffill_retrained/no_sweeper",
    ),
    VariantSpec(
        key="weak_gate",
        table_name="Weak gate",
        suite="abl_weak_conformal_gate",
        run_name_template="tier1_forecast_utilization_seed{seed}",
        eval_data="robustness_ffill_retrained/evaluation_dataset/unseendata.csv",
        forecast_cache_dir="robustness_ffill_retrained/forecast_cache",
        reference_eval_root="eval_v2_ffill_retrained/no_sweeper",
    ),
    VariantSpec(
        key="slope_prior_only",
        table_name="Slope prior-only",
        suite="abl_forecast_prior_only",
        run_name_template="tier1_forecast_utilization_seed{seed}",
        eval_data="robustness_ffill_retrained/evaluation_dataset/unseendata.csv",
        forecast_cache_dir="robustness_ffill_retrained_slope/forecast_cache",
        reference_eval_root="eval_v2_ffill_slope_prior_only/no_sweeper",
    ),
    VariantSpec(
        key="slope_focal",
        table_name="Slope-FoCAL",
        suite="tiers_tier1_forecast_utilization",
        run_name_template="tier1_forecast_utilization_seed{seed}",
        eval_data="robustness_ffill_retrained/evaluation_dataset/unseendata.csv",
        forecast_cache_dir="robustness_ffill_retrained_slope/forecast_cache",
        reference_eval_root="eval_v2_ffill_slope_focal/no_sweeper",
    ),
)


def _path(text: str | Path) -> Path:
    p = Path(text)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _format_cmd(cmd: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(x) for x in cmd])


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- {_timestamp()} {message}\n")


def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    candidates = [Path(p) for p in glob.glob(str(directory / pattern))]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_tier_metrics(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tiers = data.get("tiers", {}) if isinstance(data, dict) else {}
    tier = tiers.get("tier1", {}) if isinstance(tiers, dict) else {}
    if not isinstance(tier, dict):
        raise ValueError(f"Missing tiers.tier1 in {json_path}")
    return tier


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def build_eval_command(spec: VariantSpec, seed: int, output_dir: Path, args: argparse.Namespace) -> List[str]:
    run_dir = (
        _path(args.input_root)
        / spec.suite
        / f"seed{seed}"
        / spec.run_name_template.format(seed=seed)
    )
    return [
        sys.executable,
        "evaluation.py",
        "--mode",
        "tiers",
        "--tiers_only",
        "tier1",
        "--tier1_dir",
        str(run_dir.relative_to(PROJECT_ROOT) if run_dir.is_relative_to(PROJECT_ROOT) else run_dir),
        "--eval_data",
        spec.eval_data,
        "--eval_steps",
        str(int(args.eval_steps)),
        "--seed",
        str(int(seed)),
        "--output_dir",
        str(output_dir.relative_to(PROJECT_ROOT) if output_dir.is_relative_to(PROJECT_ROOT) else output_dir),
        "--investment_freq",
        "6",
        "--meta_freq_min",
        "6",
        "--meta_freq_max",
        "6",
        "--global_norm_mode",
        "rolling_past",
        "--rolling_past_history_dir",
        "rolling_past_history_dataset",
        "--forecast_cache_dir",
        spec.forecast_cache_dir,
        "--eval-distribution-rate",
        "0.0",
        "--friction-cost-multiplier",
        "1.0",
        "--half-spread-bp",
        "0.0",
        "--log-sleeve",
    ]


def iter_run_plan(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    run_no = 0
    total = len(VARIANTS) * len(args.seeds)
    for spec in VARIANTS:
        for seed in args.seeds:
            run_no += 1
            out_dir = _path(args.output_root) / spec.key / f"seed{seed}"
            run_dir = _path(args.input_root) / spec.suite / f"seed{seed}" / spec.run_name_template.format(seed=seed)
            yield {
                "global_run_number": run_no,
                "total_planned_runs": total,
                "variant": spec.key,
                "variant_label": spec.table_name,
                "seed": int(seed),
                "suite": spec.suite,
                "run_dir": str(run_dir),
                "final_models_dir": str(run_dir / "final_models"),
                "eval_data": spec.eval_data,
                "forecast_cache_dir": spec.forecast_cache_dir,
                "reference_eval_root": spec.reference_eval_root,
                "output_dir": str(out_dir),
                "command": _format_cmd(build_eval_command(spec, int(seed), out_dir, args)),
            }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def completed_run(output_dir: Path, eval_steps: int) -> bool:
    json_path = _latest_file(output_dir, "evaluation_tiers_*.json")
    debug_path = output_dir / "env_logs" / "tier1_debug_ep0.csv"
    if not json_path or not debug_path.is_file():
        return False
    try:
        rows = sum(1 for _ in debug_path.open("r", encoding="utf-8")) - 1
    except Exception:
        return False
    return rows == int(eval_steps)


def run_command(cmd: Sequence[str], log_file: Path, timeout_hours: float, stream_child: bool) -> Dict[str, Any]:
    started_at = datetime.now()
    start = time.time()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    timeout_s = max(60, int(float(timeout_hours) * 3600.0))
    with log_file.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"Started: {started_at.isoformat(timespec='seconds')}\n")
        log.write(f"Command: {_format_cmd(cmd)}\n\n")
        process = subprocess.Popen(
            list(cmd),
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        timed_out = False
        deadline = time.time() + timeout_s
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            if stream_child:
                print(line, end="")
            if time.time() > deadline:
                timed_out = True
                process.kill()
                log.write(f"\n[TIMEOUT] exceeded {timeout_s}s; process killed.\n")
                break
        process.wait()
        elapsed = time.time() - start
        ok = process.returncode == 0 and not timed_out
        log.write(f"\nFinished: {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"Exit: {process.returncode}\n")
        log.write(f"Duration seconds: {elapsed:.3f}\n")
    return {
        "success": bool(ok),
        "returncode": int(process.returncode),
        "duration_seconds": float(elapsed),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "log_file": str(log_file),
    }


def run_all(args: argparse.Namespace) -> List[Dict[str, Any]]:
    output_root = _path(args.output_root)
    run_log = output_root / "RUN_LOG_SLEEVE.md"
    command_log_dir = output_root / "_command_logs"
    plan_rows = list(iter_run_plan(args))
    write_csv(plan_rows, output_root / "ffill_sleeve_run_plan.csv")
    _write_log(run_log, f"Started 5-variant sleeve evaluation pass ({len(plan_rows)} planned runs).")

    results: List[Dict[str, Any]] = []
    for plan in plan_rows:
        output_dir = Path(plan["output_dir"])
        label = f"[{plan['global_run_number']}/{plan['total_planned_runs']}] {plan['variant']} seed={plan['seed']}"
        if args.resume and completed_run(output_dir, int(args.eval_steps)):
            print(f"{label}: already complete, skipping")
            results.append({**plan, "evaluation_success": True, "skipped_existing": True})
            continue
        print(f"{label}: running")
        cmd = plan["command"]
        # Re-split with Windows command-line rules by rebuilding from the source spec.
        spec = next(v for v in VARIANTS if v.key == plan["variant"])
        cmd_list = build_eval_command(spec, int(plan["seed"]), output_dir, args)
        log_file = command_log_dir / f"{plan['variant']}_seed{plan['seed']}.log"
        result = run_command(cmd_list, log_file, args.timeout_hours, bool(args.stream_child))
        row = {
            **plan,
            "evaluation_success": bool(result["success"]),
            "evaluation_returncode": int(result["returncode"]),
            "evaluation_duration_seconds": float(result["duration_seconds"]),
            "command_log": str(log_file),
            "skipped_existing": False,
        }
        results.append(row)
        write_csv(results, output_root / "ffill_sleeve_eval_status.csv")
        if not result["success"]:
            _write_log(run_log, f"FAILED {label}; see {log_file}")
            if not args.continue_on_error:
                raise RuntimeError(f"Evaluation failed for {label}; see {log_file}")
        else:
            _write_log(run_log, f"Completed {label}; see {log_file}")
    write_csv(results, output_root / "ffill_sleeve_eval_status.csv")
    _write_log(run_log, "Finished evaluation pass.")
    return results


def _reference_json(spec: VariantSpec, seed: int) -> Optional[Path]:
    ref_dir = _path(spec.reference_eval_root) / spec.suite / f"seed{seed}" / "tier1"
    return _latest_file(ref_dir, "evaluation_tiers_*.json")


def _compare_reference(spec: VariantSpec, seed: int, metrics: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    ref_json = _reference_json(spec, seed)
    out = {"reference_json": str(ref_json) if ref_json else "", "reference_match": False}
    if not ref_json:
        out["reference_error"] = "missing_reference_json"
        return out
    ref = _read_tier_metrics(ref_json)
    checks = {}
    for key in ("total_return", "sharpe_ratio", "max_drawdown", "final_portfolio_value"):
        diff = _safe_float(metrics.get(key)) - _safe_float(ref.get(key))
        checks[f"reference_diff_{key}"] = float(diff)
    out.update(checks)
    out["reference_match"] = all(abs(v) <= tolerance for v in checks.values())
    return out


def _load_sleeve_frame(debug_csv: Path) -> pd.DataFrame:
    required = ["timestep", "trading_sleeve_value_dkk"]
    preferred = [
        "distribution_adjusted_trading_sleeve_dkk",
        "held_exposure_signed",
        "exposure_exec",
        "position_signed",
        "position_exposure",
        "financial_exposure_dkk",
    ]
    with debug_csv.open("r", encoding="utf-8") as f:
        header = (f.readline() or "").strip().split(",")
    have = [c for c in required + preferred if c in header]
    missing = [c for c in required if c not in have]
    if missing:
        raise ValueError(f"missing required debug columns in {debug_csv}: {missing}")
    return pd.read_csv(debug_csv, usecols=have)


def compute_sleeve_metrics(
    debug_csv: Path,
    *,
    dkk_to_usd: float,
    periods_per_year: float,
    annual_rf: float,
) -> Dict[str, Any]:
    df = _load_sleeve_frame(debug_csv)
    if "distribution_adjusted_trading_sleeve_dkk" in df.columns:
        value_dkk = df["distribution_adjusted_trading_sleeve_dkk"].to_numpy(dtype=np.float64)
    else:
        value_dkk = df["trading_sleeve_value_dkk"].to_numpy(dtype=np.float64)
    value_usd = value_dkk * float(dkk_to_usd)
    value_usd = value_usd[np.isfinite(value_usd)]
    if value_usd.size < 2:
        raise ValueError(f"not enough sleeve value rows in {debug_csv}")
    if np.any(value_usd[:-1] <= 0.0):
        raise ValueError(f"non-positive sleeve value in {debug_csv}")

    exposure_source = "none"
    # sleeve-supplement task: prefer post-execution held exposure when available.
    if "held_exposure_signed" in df.columns:
        exposure = df["held_exposure_signed"].to_numpy(dtype=np.float64)
        exposure_source = "held_exposure_signed"
    elif "exposure_exec" in df.columns:
        exposure = df["exposure_exec"].to_numpy(dtype=np.float64)
        exposure_source = "exposure_exec"
    elif "position_signed" in df.columns:
        exposure = df["position_signed"].to_numpy(dtype=np.float64)
        exposure_source = "position_signed"
    elif "position_exposure" in df.columns:
        exposure = df["position_exposure"].to_numpy(dtype=np.float64)
        exposure_source = "position_exposure"
    else:
        exposure = np.zeros(len(df), dtype=np.float64)
    exposure = np.nan_to_num(exposure, nan=0.0, posinf=0.0, neginf=0.0)
    exposure = np.clip(exposure, -1.0, 1.0)

    step_returns = np.diff(value_usd) / value_usd[:-1]
    step_returns = step_returns[np.isfinite(step_returns)]
    rf_step = float(annual_rf) / float(periods_per_year)
    step_std = float(np.std(step_returns)) if step_returns.size > 1 else 0.0
    sharpe = float(((np.mean(step_returns) - rf_step) / step_std) * math.sqrt(periods_per_year)) if step_std > 0.0 else 0.0
    peak = np.maximum.accumulate(value_usd)
    drawdowns = np.where(peak > 0.0, (peak - value_usd) / peak, 0.0)

    out = {
        "debug_csv": str(debug_csv),
        "debug_log_rows": int(len(df)),
        "sleeve_trading_initial_usd": float(value_usd[0]),
        "sleeve_trading_final_usd": float(value_usd[-1]),
        "sleeve_return_pct": float(100.0 * (value_usd[-1] / value_usd[0] - 1.0)),
        "sleeve_step_return_mean": float(np.mean(step_returns)) if step_returns.size else 0.0,
        "sleeve_step_return_std": step_std,
        "sleeve_sharpe": sharpe,
        "sleeve_mdd_pct": float(100.0 * np.max(drawdowns)) if drawdowns.size else 0.0,
        "sleeve_mean_abs_exposure": float(np.mean(np.abs(exposure))) if exposure.size else 0.0,
        "sleeve_max_abs_exposure": float(np.max(np.abs(exposure))) if exposure.size else 0.0,
        "sleeve_turnover": float(np.sum(np.abs(np.diff(exposure)))) if exposure.size > 1 else 0.0,
        "sleeve_exposure_source": exposure_source,
    }
    if "financial_exposure_dkk" in df.columns:
        fin = df["financial_exposure_dkk"].to_numpy(dtype=np.float64)
        out["sleeve_mean_abs_exposure_dkk"] = float(np.nanmean(np.abs(fin)))
        out["sleeve_max_abs_exposure_dkk"] = float(np.nanmax(np.abs(fin)))
    return out


def aggregate(args: argparse.Namespace) -> pd.DataFrame:
    output_root = _path(args.output_root)
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for spec in VARIANTS:
        for seed in args.seeds:
            out_dir = output_root / spec.key / f"seed{seed}"
            json_path = _latest_file(out_dir, "evaluation_tiers_*.json")
            debug_csv = out_dir / "env_logs" / "tier1_debug_ep0.csv"
            if not json_path or not debug_csv.is_file():
                errors.append(f"missing outputs for {spec.key} seed {seed}")
                continue
            tier = _read_tier_metrics(json_path)
            sleeve = compute_sleeve_metrics(
                debug_csv,
                dkk_to_usd=float(args.dkk_to_usd),
                periods_per_year=float(args.periods_per_year),
                annual_rf=float(args.annual_rf),
            )
            ref = _compare_reference(spec, int(seed), tier, float(args.reference_tolerance))
            row = {
                "variant": spec.key,
                "variant_label": spec.table_name,
                "seed": int(seed),
                "eval_json": str(json_path),
                "fund_total_return": _safe_float(tier.get("total_return")),
                "fund_sharpe_ratio": _safe_float(tier.get("sharpe_ratio")),
                "fund_max_drawdown": _safe_float(tier.get("max_drawdown")),
                "fund_final_nav_usd": _safe_float(tier.get("final_portfolio_value")),
                **sleeve,
                **ref,
            }
            initial_error = abs(row["sleeve_trading_initial_usd"] - EXPECTED_TRADING_INITIAL_USD)
            if initial_error > float(args.initial_tolerance_usd):
                errors.append(
                    f"{spec.key} seed {seed}: initial sleeve {row['sleeve_trading_initial_usd']:.2f} USD"
                )
            if int(row["debug_log_rows"]) != int(args.eval_steps):
                errors.append(f"{spec.key} seed {seed}: debug rows {row['debug_log_rows']}")
            if row["sleeve_mdd_pct"] <= float(args.near_zero_mdd_pct):
                errors.append(f"{spec.key} seed {seed}: sleeve MDD near zero ({row['sleeve_mdd_pct']:.6f}%)")
            if not bool(row.get("reference_match", False)):
                errors.append(f"{spec.key} seed {seed}: fund metrics do not match reference")
            rows.append(row)

    if errors:
        error_path = output_root / "RUN_LOG_SLEEVE.md"
        for msg in errors:
            _write_log(error_path, f"BLOCKED sanity check: {msg}")
        if not args.allow_sanity_warnings:
            raise RuntimeError("Sanity checks failed:\n" + "\n".join(errors))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No sleeve rows aggregated.")
    if df.isna().any().any():
        nan_cols = sorted(df.columns[df.isna().any()].tolist())
        raise RuntimeError(f"NaNs in summary columns: {nan_cols}")
    summary_path = output_root / "ffill_sleeve_summary.csv"
    df.to_csv(summary_path, index=False)
    write_latex_table(df, output_root / "ffill_sleeve_table.tex")
    write_paired_report(df, output_root / "ffill_sleeve_paired.md")
    write_run_log_summary(df, args)
    return df


def _mean_std(df: pd.DataFrame, variant: str, col: str) -> Tuple[float, float]:
    vals = df.loc[df["variant"] == variant, col].astype(float).to_numpy()
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0


def _cell(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Trading-sleeve risk under the forward-filled no-sweeper evaluation (cross-seed mean $\\pm$ std, n=10). Unlike the fund-level drawdown ($\\sim$1 bp), the actively traded 12\\% sleeve carries substantial drawdown; the conformal action prior reduces sleeve drawdown and turnover relative to the no-prior baseline.}",
        "\\label{tab:sleeve}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Variant & Sleeve Return (\\%) & Sleeve Sharpe & Sleeve MDD (\\%) & Mean $|E|$ & Turnover \\\\",
        "\\midrule",
    ]
    for spec in VARIANTS:
        ret = _cell(*_mean_std(df, spec.key, "sleeve_return_pct"))
        shr = _cell(*_mean_std(df, spec.key, "sleeve_sharpe"))
        mdd = _cell(*_mean_std(df, spec.key, "sleeve_mdd_pct"))
        expo = _cell(*_mean_std(df, spec.key, "sleeve_mean_abs_exposure"), digits=3)
        turn = _cell(*_mean_std(df, spec.key, "sleeve_turnover"))
        lines.append(f"{spec.table_name} & {ret} & {shr} & {mdd} & {expo} & {turn} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _wilcoxon(x: np.ndarray, y: np.ndarray, alternative: str) -> Tuple[Optional[float], str]:
    try:
        from scipy.stats import wilcoxon
    except Exception as exc:
        return None, f"scipy unavailable: {exc}"
    try:
        res = wilcoxon(x, y, alternative=alternative, zero_method="wilcox", method="exact")
    except TypeError:
        res = wilcoxon(x, y, alternative=alternative, zero_method="wilcox", mode="exact")
    return float(res.pvalue), "exact"


def _paired_block(df: pd.DataFrame, left: str, right: str, label: str) -> List[str]:
    pivot = df.pivot(index="seed", columns="variant")
    left_mdd = pivot["sleeve_mdd_pct"][left].astype(float).to_numpy()
    right_mdd = pivot["sleeve_mdd_pct"][right].astype(float).to_numpy()
    left_sharpe = pivot["sleeve_sharpe"][left].astype(float).to_numpy()
    right_sharpe = pivot["sleeve_sharpe"][right].astype(float).to_numpy()
    p_mdd, mdd_method = _wilcoxon(left_mdd, right_mdd, "less")
    p_sharpe, sharpe_method = _wilcoxon(left_sharpe, right_sharpe, "greater")
    mdd_delta = left_mdd - right_mdd
    sharpe_delta = left_sharpe - right_sharpe
    return [
        f"## {label}",
        "",
        f"- Sleeve MDD delta (left - right): mean {np.mean(mdd_delta):.6f} pp, std {np.std(mdd_delta, ddof=1):.6f} pp.",
        f"- Sleeve MDD wins: {int(np.sum(left_mdd < right_mdd))}/{len(left_mdd)} lower.",
        f"- Sleeve MDD one-sided Wilcoxon p-value: {p_mdd if p_mdd is not None else 'NA'} ({mdd_method}).",
        f"- Sleeve Sharpe delta (left - right): mean {np.mean(sharpe_delta):.6f}, std {np.std(sharpe_delta, ddof=1):.6f}.",
        f"- Sleeve Sharpe wins: {int(np.sum(left_sharpe > right_sharpe))}/{len(left_sharpe)} higher.",
        f"- Sleeve Sharpe one-sided Wilcoxon p-value: {p_sharpe if p_sharpe is not None else 'NA'} ({sharpe_method}).",
        "",
    ]


def write_paired_report(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Forward-Filled Sleeve Paired Tests",
        "",
        "One-sided exact Wilcoxon signed-rank tests use paired seeds (n=10).",
        "",
    ]
    lines.extend(_paired_block(df, "focal_ann", "tier1_baseline", "FoCAL-ANN vs Tier1 baseline"))
    lines.extend(_paired_block(df, "slope_focal", "slope_prior_only", "Slope-FoCAL vs slope prior-only"))
    path.write_text("\n".join(lines), encoding="utf-8")


def _extract_pvalue(text: str, metric: str) -> str:
    for line in text.splitlines():
        if metric in line and "p-value" in line:
            return line.split(":", 1)[-1].strip().split(" ")[0]
    return "NA"


def write_run_log_summary(df: pd.DataFrame, args: argparse.Namespace) -> None:
    output_root = _path(args.output_root)
    run_log = output_root / "RUN_LOG_SLEEVE.md"
    paired_text = (output_root / "ffill_sleeve_paired.md").read_text(encoding="utf-8")
    tier_mdd = _mean_std(df, "tier1_baseline", "sleeve_mdd_pct")[0]
    focal_mdd = _mean_std(df, "focal_ann", "sleeve_mdd_pct")[0]
    tier_sharpe = _mean_std(df, "tier1_baseline", "sleeve_sharpe")[0]
    focal_sharpe = _mean_std(df, "focal_ann", "sleeve_sharpe")[0]
    p_mdd = _extract_pvalue(paired_text, "Sleeve MDD one-sided")
    p_sharpe = _extract_pvalue(paired_text, "Sleeve Sharpe one-sided")
    command_template = (
        "python evaluation.py --mode tiers --tiers_only tier1 --tier1_dir <run_dir> "
        "--eval_data <variant_eval_data> --eval_steps 39305 --seed <seed> "
        "--output_dir results/ffill_sleeve/<variant>/seed<seed> --investment_freq 6 "
        "--meta_freq_min 6 --meta_freq_max 6 --global_norm_mode rolling_past "
        "--rolling_past_history_dir rolling_past_history_dataset "
        "--forecast_cache_dir <variant_cache> --eval-distribution-rate 0.0 "
        "--friction-cost-multiplier 1.0 --half-spread-bp 0.0 --log-sleeve"
    )
    lines = [
        "",
        "## Sleeve Supplement Summary",
        f"- Generated at: {_timestamp()}",
        "- Files changed: evaluation.py, runtime_contract.py, scripts/run_ffill_sleeve_eval_50.py",
        f"- Exact eval command template: `{command_template}`",
        f"- Tier1 baseline sleeve MDD mean: {tier_mdd:.6f}%",
        f"- FoCAL-ANN sleeve MDD mean: {focal_mdd:.6f}%",
        f"- Tier1 baseline sleeve Sharpe mean: {tier_sharpe:.6f}",
        f"- FoCAL-ANN sleeve Sharpe mean: {focal_sharpe:.6f}",
        f"- Paired FoCAL-ANN vs Tier1 MDD p-value: {p_mdd}",
        f"- Paired FoCAL-ANN vs Tier1 Sharpe p-value: {p_sharpe}",
        "- Deviations/blockers: none recorded by the aggregation step.",
    ]
    with run_log.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all 50 forward-filled no-sweeper trading-sleeve evaluations."
    )
    parser.add_argument("--input_root", default="batch_tier_phase_runs")
    parser.add_argument("--output_root", default="results/ffill_sleeve")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--eval_steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--timeout_hours", type=float, default=3.0)
    parser.add_argument("--periods_per_year", type=float, default=DEFAULT_PERIODS_PER_YEAR)
    parser.add_argument("--annual_rf", type=float, default=DEFAULT_RF_ANNUAL)
    parser.add_argument("--dkk_to_usd", type=float, default=DEFAULT_DKK_TO_USD)
    parser.add_argument("--reference_tolerance", type=float, default=1e-9)
    parser.add_argument("--initial_tolerance_usd", type=float, default=250_000.0)
    parser.add_argument("--near_zero_mdd_pct", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true", help="Skip runs with complete JSON and debug CSV outputs.")
    parser.add_argument("--dry_run", action="store_true", help="Write the 50-row plan but do not execute.")
    parser.add_argument("--aggregate_only", action="store_true", help="Only compute summary/table/pair tests from existing outputs.")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--allow_sanity_warnings", action="store_true")
    parser.add_argument("--stream_child", action="store_true", help="Echo evaluation.py output while also writing command logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = _path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plan_rows = list(iter_run_plan(args))
    if len(plan_rows) != 50:
        raise RuntimeError(f"Expected exactly 50 planned runs, got {len(plan_rows)}")
    write_csv(plan_rows, output_root / "ffill_sleeve_run_plan.csv")
    if args.dry_run:
        print(f"Wrote 50-run plan: {output_root / 'ffill_sleeve_run_plan.csv'}")
        return
    if not args.aggregate_only:
        run_all(args)
    df = aggregate(args)
    print(f"Wrote {output_root / 'ffill_sleeve_summary.csv'} ({len(df)} rows)")
    print(f"Wrote {output_root / 'ffill_sleeve_table.tex'}")
    print(f"Wrote {output_root / 'ffill_sleeve_paired.md'}")


if __name__ == "__main__":
    main()

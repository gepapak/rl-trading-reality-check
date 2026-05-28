"""Run the forward-filled no-sweeper market-impact sweep.

This is eval-only. It reuses existing final_models checkpoints and writes only
under eval_v2_ffill_retrained/no_sweeper/impact_sweep and results/ffill_impact.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
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
DEFAULT_EVAL_DATA = "robustness_ffill_retrained/evaluation_dataset/unseendata.csv"
DEFAULT_FORECAST_CACHE = "robustness_ffill_retrained/forecast_cache"
DEFAULT_DKK_TO_USD = 0.145
DEFAULT_OUTPUT_ROOT = "eval_v2_ffill_retrained/no_sweeper/impact_sweep"
DEFAULT_REPORT_ROOT = "results/ffill_impact"
VOLUME_IMPACT_REFS = {"volume", "market_volume", "day_ahead_volume", "liquidity_volume"}


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    suite: str
    run_name_template: str


@dataclass(frozen=True)
class RegimeSpec:
    key: str
    label: str
    impact_coef_bp: float
    impact_exponent: float = 0.5
    impact_ref_notional: str = "sleeve"


VARIANTS: Tuple[VariantSpec, ...] = (
    VariantSpec(
        key="tier1_baseline",
        label="Tier1 baseline",
        suite="tiers_tier1_baseline",
        run_name_template="tier1_seed{seed}",
    ),
    VariantSpec(
        key="focal_ann",
        label="FoCAL-ANN",
        suite="tiers_tier1_forecast_utilization",
        run_name_template="tier1_forecast_utilization_seed{seed}",
    ),
    VariantSpec(
        key="weak_gate",
        label="Weak gate",
        suite="abl_weak_conformal_gate",
        run_name_template="tier1_forecast_utilization_seed{seed}",
    ),
)


REGIMES: Tuple[RegimeSpec, ...] = (
    RegimeSpec(key="off", label="Off", impact_coef_bp=0.0),
    RegimeSpec(key="moderate", label="Moderate", impact_coef_bp=20.0),
    RegimeSpec(key="severe", label="Severe", impact_coef_bp=50.0),
)


def _path(text: str | Path) -> Path:
    p = Path(text)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _format_cmd(cmd: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(x) for x in cmd])


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- {_timestamp()} {message}\n")


def _impact_ref_notional_for_run(regime: RegimeSpec, args: argparse.Namespace) -> str:
    return str(getattr(args, "impact_ref_notional", None) or regime.impact_ref_notional)


def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    files = [Path(p) for p in glob.glob(str(directory / pattern))]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _read_tier_metrics(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object: {json_path}")
    tiers = payload.get("tiers", {})
    tier = tiers.get("tier1", {}) if isinstance(tiers, dict) else {}
    if not isinstance(tier, dict):
        raise ValueError(f"Missing tiers.tier1 in {json_path}")
    return tier


def build_eval_command(
    spec: VariantSpec,
    regime: RegimeSpec,
    seed: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> List[str]:
    run_dir = _path(args.input_root) / spec.suite / f"seed{seed}" / spec.run_name_template.format(seed=seed)
    impact_ref_notional = _impact_ref_notional_for_run(regime, args)
    cmd = [
        sys.executable,
        "evaluation.py",
        "--mode",
        "tiers",
        "--tiers_only",
        "tier1",
        "--tier1_dir",
        _display_path(run_dir),
        "--eval_data",
        str(args.eval_data),
        "--eval_steps",
        str(int(args.eval_steps)),
        "--seed",
        str(int(seed)),
        "--output_dir",
        _display_path(output_dir),
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
        str(args.forecast_cache_dir),
        "--eval-distribution-rate",
        "0.0",
        "--friction-cost-multiplier",
        "1.0",
        "--half-spread-bp",
        "0.0",
        "--impact-coef-bp",
        str(float(regime.impact_coef_bp)),
        "--impact-exponent",
        str(float(regime.impact_exponent)),
        "--impact-ref-notional",
        impact_ref_notional,
        "--log-sleeve",
    ]
    if str(impact_ref_notional).strip().lower() in VOLUME_IMPACT_REFS:
        # market-impact task
        if not str(getattr(args, "impact_volume_data", "") or "").strip():
            raise ValueError("--impact-volume-data is required for --impact-ref-notional volume")
        cmd.extend(["--impact-volume-data", str(args.impact_volume_data)])
        if str(getattr(args, "impact_volume_column", "") or "").strip():
            cmd.extend(["--impact-volume-column", str(args.impact_volume_column)])
        if str(getattr(args, "impact_volume_unit", "") or "").strip():
            cmd.extend(["--impact-volume-unit", str(args.impact_volume_unit)])
        if str(getattr(args, "impact_volume_timestamp_column", "") or "").strip():
            cmd.extend(["--impact-volume-timestamp-column", str(args.impact_volume_timestamp_column)])
        if getattr(args, "impact_volume_max_staleness_min", None) is not None:
            cmd.extend(["--impact-volume-max-staleness-min", str(float(args.impact_volume_max_staleness_min))])
        if getattr(args, "impact_volume_price_floor_dkk_per_mwh", None) is not None:
            cmd.extend(
                [
                    "--impact-volume-price-floor-dkk-per-mwh",
                    str(float(args.impact_volume_price_floor_dkk_per_mwh)),
                ]
            )
    return cmd


def iter_run_plan(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    run_no = 0
    total = len(VARIANTS) * len(REGIMES) * len(args.seeds)
    for spec in VARIANTS:
        for regime in REGIMES:
            for seed in args.seeds:
                impact_ref_notional = _impact_ref_notional_for_run(regime, args)
                run_no += 1
                run_dir = _path(args.input_root) / spec.suite / f"seed{seed}" / spec.run_name_template.format(seed=seed)
                out_dir = _path(args.output_root) / spec.key / regime.key / f"seed{seed}" / "tier1"
                yield {
                    "global_run_number": run_no,
                    "total_planned_runs": total,
                    "variant": spec.key,
                    "variant_label": spec.label,
                    "suite": spec.suite,
                    "regime": regime.key,
                    "regime_label": regime.label,
                    "impact_coef_bp": float(regime.impact_coef_bp),
                    "impact_exponent": float(regime.impact_exponent),
                    "impact_ref_notional": impact_ref_notional,
                    "impact_volume_data": str(getattr(args, "impact_volume_data", "") or ""),
                    "impact_volume_column": str(getattr(args, "impact_volume_column", "") or ""),
                    "impact_volume_unit": str(getattr(args, "impact_volume_unit", "") or ""),
                    "impact_volume_price_floor_dkk_per_mwh": float(
                        getattr(args, "impact_volume_price_floor_dkk_per_mwh", 50.0)
                    ),
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "final_models_dir": str(run_dir / "final_models"),
                    "output_dir": str(out_dir),
                    "command": _format_cmd(build_eval_command(spec, regime, int(seed), out_dir, args)),
                }


def write_csv(rows: List[Dict[str, Any]], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
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
        tier = _read_tier_metrics(json_path)
        rows = sum(1 for _ in debug_path.open("r", encoding="utf-8")) - 1
    except Exception:
        return False
    return (
        str(tier.get("status", "")).lower() == "completed"
        and int(_safe_float(tier.get("evaluation_steps"), -1)) == int(eval_steps)
        and int(_safe_float(tier.get("models_loaded"), -1)) == 4
        and rows == int(eval_steps)
    )


def run_command(cmd: Sequence[str], log_file: Path, timeout_hours: float, stream_child: bool) -> Dict[str, Any]:
    started_at = datetime.now()
    start = time.time()
    timeout_s = max(60, int(float(timeout_hours) * 3600.0))
    log_file.parent.mkdir(parents=True, exist_ok=True)
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
    report_root = _path(args.report_root)
    command_log_dir = report_root / "_command_logs"
    run_log = report_root / "RUN_LOG_IMPACT.md"
    rows: List[Dict[str, Any]] = []
    plan_rows = list(iter_run_plan(args))
    write_csv(plan_rows, report_root / "impact_sweep_run_plan.csv")
    _write_log(run_log, f"Started impact sweep ({len(plan_rows)} planned eval runs).")

    for plan in plan_rows:
        output_dir = Path(plan["output_dir"])
        label = (
            f"[{plan['global_run_number']}/{plan['total_planned_runs']}] "
            f"{plan['variant']} {plan['regime']} seed={plan['seed']}"
        )
        if not Path(plan["final_models_dir"]).is_dir():
            msg = f"Missing final_models for {label}: {plan['final_models_dir']}"
            _write_log(run_log, f"BLOCKED {msg}")
            if not args.continue_on_error:
                raise FileNotFoundError(msg)
            rows.append({**plan, "evaluation_success": False, "evaluation_error": "missing_final_models"})
            continue
        if args.resume and completed_run(output_dir, int(args.eval_steps)):
            print(f"{label}: already complete, skipping")
            rows.append({**plan, "evaluation_success": True, "skipped_existing": True})
            continue

        print(f"{label}: running")
        spec = next(v for v in VARIANTS if v.key == plan["variant"])
        regime = next(r for r in REGIMES if r.key == plan["regime"])
        cmd = build_eval_command(spec, regime, int(plan["seed"]), output_dir, args)
        log_file = command_log_dir / f"{plan['variant']}_{plan['regime']}_seed{plan['seed']}.log"
        result = run_command(cmd, log_file, args.timeout_hours, bool(args.stream_child))
        row = {
            **plan,
            "evaluation_success": bool(result["success"]),
            "evaluation_returncode": int(result["returncode"]),
            "evaluation_duration_seconds": float(result["duration_seconds"]),
            "command_log": str(log_file),
            "skipped_existing": False,
        }
        rows.append(row)
        write_csv(rows, report_root / "impact_eval_status.csv")
        if not result["success"]:
            _write_log(run_log, f"FAILED {label}; see {log_file}")
            if not args.continue_on_error:
                raise RuntimeError(f"Evaluation failed for {label}; see {log_file}")
        else:
            _write_log(run_log, f"Completed {label}; see {log_file}")

    write_csv(rows, report_root / "impact_eval_status.csv")
    _write_log(run_log, "Finished impact sweep eval pass.")
    return rows


def _debug_exposure_series(debug_csv: Path) -> np.ndarray:
    with debug_csv.open("r", encoding="utf-8") as f:
        header = (f.readline() or "").strip().split(",")
    for col in ("held_exposure_signed", "exposure_exec", "position_signed", "position_exposure"):
        if col in header:
            arr = pd.read_csv(debug_csv, usecols=[col])[col].to_numpy(dtype=np.float64)
            return np.nan_to_num(np.clip(arr, -1.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0)
    return np.zeros(sum(1 for _ in debug_csv.open("r", encoding="utf-8")) - 1, dtype=np.float64)


def _sleeve_turnover(debug_csv: Path) -> Tuple[float, int]:
    exposure = _debug_exposure_series(debug_csv)
    turnover = float(np.sum(np.abs(np.diff(exposure)))) if exposure.size > 1 else 0.0
    return turnover, int(exposure.size)


def _reference_json(spec: VariantSpec, seed: int) -> Optional[Path]:
    ref_dir = _path("eval_v2_ffill_retrained/no_sweeper") / spec.suite / f"seed{seed}" / "tier1"
    return _latest_file(ref_dir, "evaluation_tiers_*.json")


def _reference_diffs(spec: VariantSpec, seed: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
    ref_json = _reference_json(spec, seed)
    out: Dict[str, Any] = {"reference_json": str(ref_json) if ref_json else "", "reference_match": False}
    if not ref_json:
        out["reference_error"] = "missing_reference_json"
        return out
    ref = _read_tier_metrics(ref_json)
    diffs = {}
    for key in ("total_return", "sharpe_ratio", "max_drawdown", "final_portfolio_value"):
        diffs[f"reference_diff_{key}"] = _safe_float(metrics.get(key)) - _safe_float(ref.get(key))
    out.update(diffs)
    out["reference_match"] = all(abs(v) <= 1e-9 for v in diffs.values())
    return out


def aggregate(args: argparse.Namespace) -> pd.DataFrame:
    report_root = _path(args.report_root)
    run_log = report_root / "RUN_LOG_IMPACT.md"
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for spec in VARIANTS:
        for regime in REGIMES:
            for seed in args.seeds:
                out_dir = _path(args.output_root) / spec.key / regime.key / f"seed{seed}" / "tier1"
                json_path = _latest_file(out_dir, "evaluation_tiers_*.json")
                debug_csv = out_dir / "env_logs" / "tier1_debug_ep0.csv"
                if not json_path or not debug_csv.is_file():
                    errors.append(f"{spec.key} {regime.key} seed {seed}: missing JSON or debug CSV")
                    continue
                tier = _read_tier_metrics(json_path)
                turnover, debug_rows = _sleeve_turnover(debug_csv)
                row: Dict[str, Any] = {
                    "variant": spec.key,
                    "variant_label": spec.label,
                    "suite": spec.suite,
                    "regime": regime.key,
                    "regime_label": regime.label,
                    "impact_coef_bp": float(regime.impact_coef_bp),
                    "impact_exponent": float(regime.impact_exponent),
                    "impact_ref_notional": str(tier.get("impact_ref_notional", _impact_ref_notional_for_run(regime, args))),
                    "impact_volume_data": str(tier.get("impact_volume_data_path", "not_applicable") or "not_applicable"),
                    "impact_volume_column": str(tier.get("impact_volume_column", "not_applicable") or "not_applicable"),
                    "impact_volume_unit": str(tier.get("impact_volume_unit", "not_applicable") or "not_applicable"),
                    "impact_volume_price_floor_dkk_per_mwh": _safe_float(
                        tier.get("impact_volume_price_floor_dkk_per_mwh")
                    ),
                    "impact_liquidity_source": str(tier.get("impact_liquidity_source", "sleeve") or "sleeve"),
                    "last_market_impact_ref_notional_dkk": _safe_float(tier.get("last_market_impact_ref_notional_dkk")),
                    "last_market_impact_participation": _safe_float(tier.get("last_market_impact_participation")),
                    "last_market_impact_bp": _safe_float(tier.get("last_market_impact_bp")),
                    "seed": int(seed),
                    "eval_json": str(json_path),
                    "debug_csv": str(debug_csv),
                    "debug_log_rows": int(debug_rows),
                    "total_return": _safe_float(tier.get("total_return")),
                    "return_pct": 100.0 * _safe_float(tier.get("total_return")),
                    "sharpe_ratio": _safe_float(tier.get("sharpe_ratio")),
                    "max_drawdown": _safe_float(tier.get("max_drawdown")),
                    "mdd_bp": 10000.0 * _safe_float(tier.get("max_drawdown")),
                    "volatility": _safe_float(tier.get("volatility")),
                    "final_portfolio_value": _safe_float(tier.get("final_portfolio_value")),
                    "final_nav_m": _safe_float(tier.get("final_portfolio_value")) / 1_000_000.0,
                    "total_impact_cost_usd": _safe_float(tier.get("total_market_impact_cost_usd")),
                    "total_impact_cost_m": _safe_float(tier.get("total_market_impact_cost_usd")) / 1_000_000.0,
                    "sleeve_turnover": float(turnover),
                    "evaluation_steps": int(_safe_float(tier.get("evaluation_steps"), -1)),
                    "models_loaded": int(_safe_float(tier.get("models_loaded"), -1)),
                    "status": str(tier.get("status", "")),
                    "reference_json": "not_applicable",
                    "reference_match": False,
                    "reference_diff_total_return": 0.0,
                    "reference_diff_sharpe_ratio": 0.0,
                    "reference_diff_max_drawdown": 0.0,
                    "reference_diff_final_portfolio_value": 0.0,
                }
                if regime.key == "off":
                    row.update(_reference_diffs(spec, int(seed), tier))
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        errors.append("No impact sweep rows aggregated")
    else:
        errors.extend(_sanity_errors(df, args))

    if errors:
        for msg in errors:
            _write_log(run_log, f"BLOCKED sanity check: {msg}")
        if not args.allow_sanity_warnings:
            raise RuntimeError("Impact sweep sanity checks failed:\n" + "\n".join(errors))

    summary_path = report_root / "impact_sweep_summary.csv"
    report_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)
    write_latex_table(df, report_root / "impact_table.tex")
    write_paired_report(df, report_root / "impact_paired.md")
    write_gap_figure(df, report_root / "impact_gap_figure")
    write_run_log_summary(df, args)
    return df


def _sanity_errors(df: pd.DataFrame, args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    expected_rows = len(VARIANTS) * len(REGIMES) * len(args.seeds)
    if len(df) != expected_rows:
        errors.append(f"expected {expected_rows} rows, got {len(df)}")
    if df.isna().any().any():
        errors.append(f"NaNs in columns: {sorted(df.columns[df.isna().any()].tolist())}")
    for _, row in df.iterrows():
        label = f"{row['variant']} {row['regime']} seed {int(row['seed'])}"
        if int(row["evaluation_steps"]) != int(args.eval_steps):
            errors.append(f"{label}: evaluation_steps={row['evaluation_steps']}")
        if int(row["models_loaded"]) != 4:
            errors.append(f"{label}: models_loaded={row['models_loaded']}")
        if str(row["status"]).lower() != "completed":
            errors.append(f"{label}: status={row['status']}")
        if int(row["debug_log_rows"]) != int(args.eval_steps):
            errors.append(f"{label}: debug_log_rows={row['debug_log_rows']}")
    off = df[df["regime"] == "off"]
    for _, row in off.iterrows():
        if not bool(row.get("reference_match", False)):
            errors.append(f"{row['variant']} off seed {int(row['seed'])}: impact-off reference mismatch")
    for variant in df["variant"].unique():
        for seed in sorted(df["seed"].unique()):
            sub = df[(df["variant"] == variant) & (df["seed"] == seed)].set_index("regime")
            if not all(k in sub.index for k in ("off", "moderate", "severe")):
                continue
            returns = [float(sub.loc[k, "total_return"]) for k in ("off", "moderate", "severe")]
            navs = [float(sub.loc[k, "final_portfolio_value"]) for k in ("off", "moderate", "severe")]
            if not (returns[0] + 1e-12 >= returns[1] >= returns[2] - 1e-12):
                errors.append(f"{variant} seed {int(seed)}: return is not monotone decreasing with impact")
            if not (navs[0] + 1e-6 >= navs[1] >= navs[2] - 1e-6):
                errors.append(f"{variant} seed {int(seed)}: final NAV is not monotone decreasing with impact")
    for regime in ("off", "moderate", "severe"):
        means = df[df["regime"] == regime].groupby("variant")["total_impact_cost_usd"].mean()
        if all(k in means.index for k in ("weak_gate", "tier1_baseline", "focal_ann")):
            if not (means["weak_gate"] + 1e-9 >= means["tier1_baseline"] >= means["focal_ann"] - 1e-9):
                errors.append(
                    f"{regime}: impact cost ordering violated "
                    f"(weak={means['weak_gate']}, tier1={means['tier1_baseline']}, focal={means['focal_ann']})"
                )
    for regime in ("off", "moderate", "severe"):
        sub = df[df["regime"] == regime]
        focal = sub[sub["variant"] == "focal_ann"]["sharpe_ratio"].mean()
        tier1 = sub[sub["variant"] == "tier1_baseline"]["sharpe_ratio"].mean()
        if not (float(focal) > float(tier1)):
            errors.append(f"{regime}: FoCAL-ANN mean Sharpe gap is not positive")
    return errors


def _mean_std(df: pd.DataFrame, variant: str, regime: str, col: str) -> Tuple[float, float]:
    vals = df[(df["variant"] == variant) & (df["regime"] == regime)][col].astype(float).to_numpy()
    if vals.size == 0:
        raise ValueError(f"No data for {variant} {regime} {col}")
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0


def _cell(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Forward-filled no-sweeper performance under square-root market-impact costs (cross-seed mean $\\pm$ std, n=10).}",
        "\\label{tab:impact}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Variant & Impact & Return (\\%) & Sharpe & MDD (bp) & Final NAV (\\$M) & Impact cost (\\$M) \\\\",
        "\\midrule",
    ]
    for spec in VARIANTS:
        for regime in REGIMES:
            ret = _cell(*_mean_std(df, spec.key, regime.key, "return_pct"))
            shr = _cell(*_mean_std(df, spec.key, regime.key, "sharpe_ratio"))
            mdd = _cell(*_mean_std(df, spec.key, regime.key, "mdd_bp"))
            nav = _cell(*_mean_std(df, spec.key, regime.key, "final_nav_m"))
            impact = _cell(*_mean_std(df, spec.key, regime.key, "total_impact_cost_m"))
            lines.append(f"{spec.label} & {regime.label} & {ret} & {shr} & {mdd} & {nav} & {impact} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _wilcoxon(left: np.ndarray, right: np.ndarray, alternative: str) -> Tuple[Optional[float], str]:
    try:
        from scipy.stats import wilcoxon
    except Exception as exc:
        return None, f"scipy unavailable: {exc}"
    try:
        res = wilcoxon(left, right, alternative=alternative, zero_method="wilcox", method="exact")
    except TypeError:
        res = wilcoxon(left, right, alternative=alternative, zero_method="wilcox", mode="exact")
    return float(res.pvalue), "exact"


def _paired_arrays(df: pd.DataFrame, regime: str, col: str) -> Tuple[np.ndarray, np.ndarray]:
    sub = df[(df["regime"] == regime) & (df["variant"].isin(["focal_ann", "tier1_baseline"]))]
    pivot = sub.pivot(index="seed", columns="variant", values=col)
    return pivot["focal_ann"].astype(float).to_numpy(), pivot["tier1_baseline"].astype(float).to_numpy()


def write_paired_report(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Market-Impact Paired Tests",
        "",
        "One-sided exact Wilcoxon signed-rank tests use paired seeds (n=10).",
        "",
    ]
    for regime in ("off", "moderate", "severe"):
        focal_sharpe, tier_sharpe = _paired_arrays(df, regime, "sharpe_ratio")
        focal_nav, tier_nav = _paired_arrays(df, regime, "final_portfolio_value")
        p_sharpe, sharpe_method = _wilcoxon(focal_sharpe, tier_sharpe, "greater")
        p_nav, nav_method = _wilcoxon(focal_nav, tier_nav, "greater")
        sharpe_delta = focal_sharpe - tier_sharpe
        nav_delta_m = (focal_nav - tier_nav) / 1_000_000.0
        lines.extend(
            [
                f"## {regime}",
                "",
                f"- Sharpe delta (FoCAL-ANN - Tier1): mean {np.mean(sharpe_delta):.6f}, std {np.std(sharpe_delta, ddof=1):.6f}.",
                f"- Sharpe wins: {int(np.sum(focal_sharpe > tier_sharpe))}/{len(focal_sharpe)} higher.",
                f"- Sharpe one-sided Wilcoxon p-value: {p_sharpe if p_sharpe is not None else 'NA'} ({sharpe_method}).",
                f"- Final NAV delta (FoCAL-ANN - Tier1): mean {np.mean(nav_delta_m):.6f} $M, std {np.std(nav_delta_m, ddof=1):.6f} $M.",
                f"- Final NAV wins: {int(np.sum(focal_nav > tier_nav))}/{len(focal_nav)} higher.",
                f"- Final NAV one-sided Wilcoxon p-value: {p_nav if p_nav is not None else 'NA'} ({nav_method}).",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 12345) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if values.size == 0:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(int(n_boot), values.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def write_gap_figure(df: pd.DataFrame, path_stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(REGIMES), dtype=float)
    labels = [r.label for r in REGIMES]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.0), sharex=True)

    for variant, label in (("tier1_baseline", "Tier1"), ("focal_ann", "FoCAL-ANN")):
        means = []
        stds = []
        for regime in REGIMES:
            vals = df[(df["variant"] == variant) & (df["regime"] == regime.key)]["sharpe_ratio"].astype(float).to_numpy()
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)))
        means_arr = np.asarray(means)
        stds_arr = np.asarray(stds)
        axes[0].plot(x, means_arr, marker="o", label=label)
        axes[0].fill_between(x, means_arr - stds_arr, means_arr + stds_arr, alpha=0.18)

    gap_means = []
    ci_low = []
    ci_high = []
    for idx, regime in enumerate(REGIMES):
        focal, tier = _paired_arrays(df, regime.key, "sharpe_ratio")
        gap = focal - tier
        gap_means.append(float(np.mean(gap)))
        lo, hi = _bootstrap_ci(gap, seed=12345 + idx)
        ci_low.append(lo)
        ci_high.append(hi)
    gap_means_arr = np.asarray(gap_means)
    axes[1].plot(x, gap_means_arr, marker="o", color="black")
    axes[1].fill_between(x, np.asarray(ci_low), np.asarray(ci_high), color="gray", alpha=0.25)
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")

    axes[0].set_ylabel("Sharpe ratio")
    axes[1].set_ylabel("FoCAL - Tier1 Sharpe")
    axes[1].set_xlabel("Impact regime")
    axes[0].legend(frameon=False)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _extract_pvalue(text: str, section: str, metric: str) -> str:
    active = False
    for line in text.splitlines():
        if line.startswith("## "):
            active = section in line
        if active and metric in line and "p-value" in line:
            return line.split(":", 1)[-1].strip().split(" ")[0]
    return "NA"


def write_run_log_summary(df: pd.DataFrame, args: argparse.Namespace) -> None:
    report_root = _path(args.report_root)
    run_log = report_root / "RUN_LOG_IMPACT.md"
    paired_text = (report_root / "impact_paired.md").read_text(encoding="utf-8")
    lines = [
        "",
        "## Market-Impact Sweep Summary",
        f"- Generated at: {_timestamp()}",
        "- Files changed: config.py, environment.py, evaluation.py, scripts/run_impact_sweep_eval.py",
        "- Impact model: temporary cost C=(kappa/1e4)*(q/Q_ref)^alpha*q charged against trading cash at the existing fill-cost site.",
        f"- Regimes: off kappa=0 bp, moderate kappa=20 bp, severe kappa=50 bp; alpha=0.5 square-root law; Q_ref='{str(getattr(args, 'impact_ref_notional', '') or 'sleeve')}'.",
        "- Rationale: kappa is bp of slippage at full-sleeve participation; 20/50 bp bracket moderate-to-severe temporary impact while preserving the original default-friction run as the off control.",
    ]
    if str(getattr(args, "impact_ref_notional", "") or "").strip().lower() in VOLUME_IMPACT_REFS:
        lines.append(
            "- Volume-calibrated reference: hourly market volume is aligned to the 10-minute evaluation grid and converted to DKK notional as volume_mwh * max(abs(price_dkk_per_mwh), price_floor)."
        )
    for regime in ("off", "moderate", "severe"):
        sub = df[df["regime"] == regime]
        focal_sharpe = sub[sub["variant"] == "focal_ann"]["sharpe_ratio"].mean()
        tier_sharpe = sub[sub["variant"] == "tier1_baseline"]["sharpe_ratio"].mean()
        focal_nav = sub[sub["variant"] == "focal_ann"]["final_portfolio_value"].mean()
        tier_nav = sub[sub["variant"] == "tier1_baseline"]["final_portfolio_value"].mean()
        costs = sub.groupby("variant")["total_impact_cost_usd"].mean().to_dict()
        lines.extend(
            [
                f"- {regime}: FoCAL-Tier1 Sharpe gap = {float(focal_sharpe - tier_sharpe):.6f}; "
                f"Final NAV gap = {(float(focal_nav - tier_nav) / 1_000_000.0):.6f} $M.",
                f"- {regime}: Sharpe p-value = {_extract_pvalue(paired_text, regime, 'Sharpe one-sided')}; "
                f"Final NAV p-value = {_extract_pvalue(paired_text, regime, 'Final NAV one-sided')}.",
                f"- {regime}: mean impact cost USD = {costs}.",
            ]
        )
    lines.append("- Deviations/blockers: none recorded by the aggregation step.")
    with run_log.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run order-size-dependent market-impact eval sweep.")
    parser.add_argument("--input_root", default="batch_tier_phase_runs")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report_root", default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--eval_data", default=DEFAULT_EVAL_DATA)
    parser.add_argument("--forecast_cache_dir", default=DEFAULT_FORECAST_CACHE)
    parser.add_argument("--eval_steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument(
        "--impact_ref_notional",
        "--impact-ref-notional",
        dest="impact_ref_notional",
        default=None,
        help="Impact reference notional: omit for regime default ('sleeve'), pass 'volume' for volume-calibrated impact.",
    )
    parser.add_argument(
        "--impact_volume_data",
        "--impact-volume-data",
        dest="impact_volume_data",
        default="",
        help="Timestamped market-volume CSV for --impact-ref-notional volume.",
    )
    parser.add_argument(
        "--impact_volume_column",
        "--impact-volume-column",
        dest="impact_volume_column",
        default="",
        help="Volume column name, or comma-separated columns to sum. Blank autodetects a single volume column.",
    )
    parser.add_argument(
        "--impact_volume_unit",
        "--impact-volume-unit",
        dest="impact_volume_unit",
        choices=["mwh", "dkk"],
        default="mwh",
        help="Unit of impact volume data.",
    )
    parser.add_argument(
        "--impact_volume_timestamp_column",
        "--impact-volume-timestamp-column",
        dest="impact_volume_timestamp_column",
        default="timestamp",
        help="Timestamp column in --impact-volume-data.",
    )
    parser.add_argument(
        "--impact_volume_max_staleness_min",
        "--impact-volume-max-staleness-min",
        dest="impact_volume_max_staleness_min",
        type=float,
        default=90.0,
        help="Maximum minutes to forward-fill hourly market volume onto the 10-minute eval grid.",
    )
    parser.add_argument(
        "--impact_volume_price_floor_dkk_per_mwh",
        "--impact-volume-price-floor-dkk-per-mwh",
        dest="impact_volume_price_floor_dkk_per_mwh",
        type=float,
        default=50.0,
        help="Price floor used when converting MWh volume to DKK notional.",
    )
    parser.add_argument("--timeout_hours", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true", help="Skip complete eval outputs.")
    parser.add_argument("--dry_run", action="store_true", help="Write the 90-run plan but do not execute.")
    parser.add_argument("--aggregate_only", action="store_true", help="Only aggregate existing outputs.")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--allow_sanity_warnings", action="store_true")
    parser.add_argument("--stream_child", action="store_true", help="Echo evaluation.py output while logging it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    impact_ref = str(getattr(args, "impact_ref_notional", "") or "").strip().lower()
    if impact_ref in VOLUME_IMPACT_REFS:
        # market-impact task
        if not str(getattr(args, "impact_volume_data", "") or "").strip():
            raise ValueError("--impact-volume-data is required when --impact-ref-notional volume is used")
        if args.output_root == DEFAULT_OUTPUT_ROOT:
            args.output_root = "eval_v2_ffill_retrained/no_sweeper/impact_volume_sweep"
        if args.report_root == DEFAULT_REPORT_ROOT:
            args.report_root = "results/ffill_impact_volume"
    report_root = _path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    plan_rows = list(iter_run_plan(args))
    if len(plan_rows) != 90:
        raise RuntimeError(f"Expected exactly 90 impact eval runs, got {len(plan_rows)}")
    write_csv(plan_rows, report_root / "impact_sweep_run_plan.csv")
    if args.dry_run:
        print(f"Wrote 90-run impact plan: {report_root / 'impact_sweep_run_plan.csv'}")
        return
    if not args.aggregate_only:
        run_all(args)
    df = aggregate(args)
    print(f"Wrote {report_root / 'impact_sweep_summary.csv'} ({len(df)} rows)")
    print(f"Wrote {report_root / 'impact_table.tex'}")
    print(f"Wrote {report_root / 'impact_paired.md'}")
    print(f"Wrote {report_root / 'impact_gap_figure.pdf'}")


if __name__ == "__main__":
    main()

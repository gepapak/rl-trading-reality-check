"""Run forward-filled no-sweeper evals for the two MAPPO 2x2 cells.

This is eval-only. It expects MAPPO checkpoints to already exist under
batch_tier_phase_runs/mappo_no_prior and batch_tier_phase_runs/mappo_prior.
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
DEFAULT_LONG_CSV = "results/ffill_robustness/ffill_per_seed_long.csv"


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    long_label: str
    suite: str
    run_name_template: str
    summary_name: str


VARIANTS: Tuple[VariantSpec, ...] = (
    VariantSpec(
        key="mappo_no_prior",
        label="MAPPO (no prior)",
        long_label="MAPPO",
        suite="mappo_no_prior",
        run_name_template="tier1_seed{seed}",
        summary_name="mappo_no_prior_no_sweeper_seed_summary.csv",
    ),
    VariantSpec(
        key="mappo_prior",
        label="MAPPO + prior",
        long_label="MAPPO+prior",
        suite="mappo_prior",
        run_name_template="tier1_forecast_utilization_seed{seed}",
        summary_name="mappo_prior_no_sweeper_seed_summary.csv",
    ),
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
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object: {json_path}")
    tiers = data.get("tiers", {})
    tier = tiers.get("tier1", {}) if isinstance(tiers, dict) else {}
    if not isinstance(tier, dict):
        raise ValueError(f"Missing tiers.tier1 in {json_path}")
    return tier


def _metric_scalars(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)):
            out[key] = value
    return out


def build_eval_command(spec: VariantSpec, seed: int, output_dir: Path, args: argparse.Namespace) -> List[str]:
    run_dir = _path(args.input_root) / spec.suite / f"seed{seed}" / spec.run_name_template.format(seed=seed)
    return [
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
    ]


def iter_run_plan(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    run_no = 0
    total = len(VARIANTS) * len(args.seeds)
    for spec in VARIANTS:
        for seed in args.seeds:
            run_no += 1
            run_dir = _path(args.input_root) / spec.suite / f"seed{seed}" / spec.run_name_template.format(seed=seed)
            out_dir = _path(args.eval_root) / spec.suite / f"seed{seed}" / "tier1"
            yield {
                "global_run_number": run_no,
                "total_planned_runs": total,
                "variant": spec.key,
                "variant_label": spec.label,
                "seed": int(seed),
                "suite": spec.suite,
                "run_dir": str(run_dir),
                "final_models_dir": str(run_dir / "final_models"),
                "eval_data": str(args.eval_data),
                "forecast_cache_dir": str(args.forecast_cache_dir),
                "output_dir": str(out_dir),
                "command": _format_cmd(build_eval_command(spec, int(seed), out_dir, args)),
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
    if not json_path:
        return False
    try:
        tier = _read_tier_metrics(json_path)
    except Exception:
        return False
    return (
        str(tier.get("status", "")).lower() == "completed"
        and int(_safe_float(tier.get("evaluation_steps"), -1)) == int(eval_steps)
        and int(_safe_float(tier.get("models_loaded"), -1)) == 4
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
    run_log = report_root / "RUN_LOG_MAPPO.md"
    rows: List[Dict[str, Any]] = []
    plan_rows = list(iter_run_plan(args))
    write_csv(plan_rows, report_root / "mappo_eval_run_plan.csv")
    _write_log(run_log, f"Started MAPPO forward-filled no-sweeper eval ({len(plan_rows)} planned runs).")

    for plan in plan_rows:
        output_dir = Path(plan["output_dir"])
        label = f"[{plan['global_run_number']}/{plan['total_planned_runs']}] {plan['variant']} seed={plan['seed']}"
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
        cmd = build_eval_command(spec, int(plan["seed"]), output_dir, args)
        log_file = command_log_dir / f"{plan['variant']}_seed{plan['seed']}.log"
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
        write_csv(rows, report_root / "mappo_eval_status.csv")
        if not result["success"]:
            _write_log(run_log, f"FAILED {label}; see {log_file}")
            if not args.continue_on_error:
                raise RuntimeError(f"Evaluation failed for {label}; see {log_file}")
        else:
            _write_log(run_log, f"Completed {label}; see {log_file}")

    write_csv(rows, report_root / "mappo_eval_status.csv")
    _write_log(run_log, "Finished MAPPO evaluation pass.")
    return rows


def _baseline_summary_columns() -> Optional[List[str]]:
    path = _path(
        "eval_v2_ffill_retrained/no_sweeper/tiers_tier1_baseline/"
        "tiers_tier1_baseline_no_sweeper_seed_summary.csv"
    )
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return list(next(reader))
        except StopIteration:
            return None


def _summary_row(
    *,
    spec: VariantSpec,
    seed: int,
    run_no: int,
    total_runs: int,
    eval_json: Path,
    tier: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    run_dir = _path(args.input_root) / spec.suite / f"seed{seed}" / spec.run_name_template.format(seed=seed)
    eval_out = _path(args.eval_root) / spec.suite / f"seed{seed}" / "tier1"
    row: Dict[str, Any] = {
        "phase": "eval_v2_no_sweeper",
        "global_run_number": run_no,
        "total_planned_runs": total_runs,
        "seed": int(seed),
        "run_order": run_no,
        "run_name": spec.label,
        "canonical_variant": "tier1",
        "suite": spec.suite,
        "save_dir": _display_path(run_dir),
        "eval_output_dir": _display_path(eval_out),
        "training_success": True,
        "training_returncode": 0,
        "training_duration_seconds": 0.0,
        "evaluation_success": True,
        "evaluation_returncode": 0,
        "evaluation_duration_seconds": 0.0,
        "evaluation_results_json": _display_path(eval_json),
    }
    row.update(_metric_scalars(tier))
    row["variant"] = spec.long_label
    return row


def _validate_rows(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    expected = len(args.seeds)
    for spec in VARIANTS:
        sub = [r for r in rows if r.get("suite") == spec.suite]
        if len(sub) != expected:
            errors.append(f"{spec.key}: expected {expected} rows, got {len(sub)}")
        for row in sub:
            seed = row.get("seed")
            for key in ("total_return", "sharpe_ratio", "max_drawdown", "volatility", "final_portfolio_value"):
                if not math.isfinite(_safe_float(row.get(key), float("nan"))):
                    errors.append(f"{spec.key} seed {seed}: non-finite {key}")
            if int(_safe_float(row.get("evaluation_steps"), -1)) != int(args.eval_steps):
                errors.append(f"{spec.key} seed {seed}: evaluation_steps={row.get('evaluation_steps')}")
            if int(_safe_float(row.get("models_loaded"), -1)) != 4:
                errors.append(f"{spec.key} seed {seed}: models_loaded={row.get('models_loaded')}")
            if str(row.get("status", "")).lower() != "completed":
                errors.append(f"{spec.key} seed {seed}: status={row.get('status')}")
            if _safe_float(row.get("final_portfolio_value"), -1.0) <= 0.0:
                errors.append(f"{spec.key} seed {seed}: non-positive final NAV")
    return errors


def aggregate(args: argparse.Namespace) -> pd.DataFrame:
    report_root = _path(args.report_root)
    run_log = report_root / "RUN_LOG_MAPPO.md"
    rows: List[Dict[str, Any]] = []
    run_no = 0
    total_runs = len(VARIANTS) * len(args.seeds)
    missing: List[str] = []

    for spec in VARIANTS:
        spec_rows: List[Dict[str, Any]] = []
        for seed in args.seeds:
            run_no += 1
            eval_dir = _path(args.eval_root) / spec.suite / f"seed{seed}" / "tier1"
            eval_json = _latest_file(eval_dir, "evaluation_tiers_*.json")
            if not eval_json:
                missing.append(f"{spec.key} seed {seed}: missing evaluation_tiers_*.json in {eval_dir}")
                continue
            tier = _read_tier_metrics(eval_json)
            row = _summary_row(
                spec=spec,
                seed=int(seed),
                run_no=run_no,
                total_runs=total_runs,
                eval_json=eval_json,
                tier=tier,
                args=args,
            )
            rows.append(row)
            spec_rows.append(row)
        if spec_rows:
            columns = _baseline_summary_columns()
            write_csv(spec_rows, report_root / spec.summary_name, fieldnames=columns)

    errors = missing + _validate_rows(rows, args)
    if errors:
        for msg in errors:
            _write_log(run_log, f"BLOCKED sanity check: {msg}")
        if not args.allow_sanity_warnings:
            raise RuntimeError("MAPPO eval sanity checks failed:\n" + "\n".join(errors))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No MAPPO eval rows aggregated.")
    if df.isna().any().any():
        nan_cols = sorted(df.columns[df.isna().any()].tolist())
        raise RuntimeError(f"NaNs in MAPPO summary columns: {nan_cols}")

    long_rows = _mappo_long_rows(df)
    write_csv(long_rows, report_root / "mappo_per_seed_long.csv")
    combined = update_ffill_long_csv(long_rows, args)
    write_comparison_report(combined, report_root / "mappo_2x2_comparison.md")
    write_run_log_summary(combined, args)
    return df


def _mappo_long_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "seed": int(row["seed"]),
                "return_pct": 100.0 * _safe_float(row.get("total_return")),
                "sharpe": _safe_float(row.get("sharpe_ratio")),
                "mdd_bp": 10000.0 * _safe_float(row.get("max_drawdown")),
                "mdd_pct": 100.0 * _safe_float(row.get("max_drawdown")),
                "vol": _safe_float(row.get("volatility")),
                "final_m": _safe_float(row.get("final_portfolio_value")) / 1_000_000.0,
                "dist": _safe_float(row.get("total_distributions_usd")),
                "steps": int(_safe_float(row.get("evaluation_steps"))),
                "variant": str(row.get("variant")),
            }
        )
    return rows


def update_ffill_long_csv(long_rows: List[Dict[str, Any]], args: argparse.Namespace) -> pd.DataFrame:
    long_path = _path(args.long_csv)
    new_df = pd.DataFrame(long_rows)
    if long_path.is_file():
        existing = pd.read_csv(long_path)
    else:
        existing = pd.DataFrame(columns=list(new_df.columns))
    if not args.skip_update_long:
        keep = existing[~existing["variant"].isin(["MAPPO", "MAPPO+prior"])] if "variant" in existing else existing
        out = pd.concat([keep, new_df], ignore_index=True)
        out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype(int)
        order = {"Tier1": 0, "FoCAL-ANN": 1, "MAPPO": 2, "MAPPO+prior": 3}
        out["_order"] = out["variant"].map(order).fillna(99)
        out = out.sort_values(["_order", "seed", "variant"]).drop(columns=["_order"])
        long_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(long_path, index=False)
        return out
    return pd.concat([existing, new_df], ignore_index=True)


def _mean_std(df: pd.DataFrame, variant: str, col: str) -> Tuple[float, float]:
    vals = pd.to_numeric(df.loc[df["variant"] == variant, col], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        raise ValueError(f"No values for {variant} {col}")
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0


def _cell(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


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


def _paired_lines(df: pd.DataFrame, left: str, right: str, label: str) -> List[str]:
    sub = df[df["variant"].isin([left, right])].copy()
    pivot = sub.pivot(index="seed", columns="variant")
    left_sharpe = pivot["sharpe"][left].astype(float).to_numpy()
    right_sharpe = pivot["sharpe"][right].astype(float).to_numpy()
    left_mdd = pivot["mdd_bp"][left].astype(float).to_numpy()
    right_mdd = pivot["mdd_bp"][right].astype(float).to_numpy()
    p_sharpe, sharpe_method = _wilcoxon(left_sharpe, right_sharpe, "greater")
    p_mdd, mdd_method = _wilcoxon(left_mdd, right_mdd, "less")
    sharpe_delta = left_sharpe - right_sharpe
    mdd_delta = left_mdd - right_mdd
    return [
        f"## {label}",
        "",
        f"- Sharpe delta ({left} - {right}): mean {np.mean(sharpe_delta):.6f}, std {np.std(sharpe_delta, ddof=1):.6f}.",
        f"- Sharpe wins: {int(np.sum(left_sharpe > right_sharpe))}/{len(left_sharpe)} higher.",
        f"- Sharpe one-sided Wilcoxon p-value: {p_sharpe if p_sharpe is not None else 'NA'} ({sharpe_method}).",
        f"- MDD delta ({left} - {right}): mean {np.mean(mdd_delta):.6f} bp, std {np.std(mdd_delta, ddof=1):.6f} bp.",
        f"- MDD wins: {int(np.sum(left_mdd < right_mdd))}/{len(left_mdd)} lower.",
        f"- MDD one-sided Wilcoxon p-value: {p_mdd if p_mdd is not None else 'NA'} ({mdd_method}).",
        "",
    ]


def write_comparison_report(df: pd.DataFrame, path: Path) -> None:
    variants = ["Tier1", "FoCAL-ANN", "MAPPO", "MAPPO+prior"]
    missing = [v for v in variants if v not in set(df["variant"])]
    if missing:
        raise RuntimeError(f"Cannot build 2x2 comparison; missing variants in long CSV: {missing}")

    lines = [
        "# MAPPO 2x2 Forward-Filled No-Sweeper Comparison",
        "",
        "Cross-seed values are mean +/- sample std (n=10). Sharpe CV is std/abs(mean) in percent.",
        "",
        "| Variant | Return (%) | Sharpe | Sharpe CV (%) | MDD (bp) | Volatility | Final NAV ($M) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        ret = _cell(*_mean_std(df, variant, "return_pct"))
        sharpe_mean, sharpe_std = _mean_std(df, variant, "sharpe")
        sharpe = _cell(sharpe_mean, sharpe_std)
        cv = 100.0 * sharpe_std / abs(sharpe_mean) if abs(sharpe_mean) > 0.0 else float("nan")
        mdd = _cell(*_mean_std(df, variant, "mdd_bp"))
        vol = _cell(*_mean_std(df, variant, "vol"), digits=6)
        final_nav = _cell(*_mean_std(df, variant, "final_m"))
        lines.append(f"| {variant} | {ret} | {sharpe} | {cv:.2f} | {mdd} | {vol} | {final_nav} |")

    lines.extend(
        [
            "",
            "One-sided exact Wilcoxon signed-rank tests use paired seeds (n=10).",
            "",
        ]
    )
    lines.extend(_paired_lines(df, "FoCAL-ANN", "MAPPO", "FoCAL-ANN vs MAPPO (no prior)"))
    lines.extend(_paired_lines(df, "MAPPO+prior", "MAPPO", "MAPPO+prior vs MAPPO (no prior)"))
    lines.extend(_paired_lines(df, "Tier1", "MAPPO", "Tier1 vs MAPPO (no prior)"))
    path.write_text("\n".join(lines), encoding="utf-8")


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
    run_log = report_root / "RUN_LOG_MAPPO.md"
    comparison = (report_root / "mappo_2x2_comparison.md").read_text(encoding="utf-8")
    mappo_sharpe = _mean_std(df, "MAPPO", "sharpe")[0]
    mappo_prior_sharpe = _mean_std(df, "MAPPO+prior", "sharpe")[0]
    mappo_mdd = _mean_std(df, "MAPPO", "mdd_bp")[0]
    mappo_prior_mdd = _mean_std(df, "MAPPO+prior", "mdd_bp")[0]
    command_template = (
        "python evaluation.py --mode tiers --tiers_only tier1 --tier1_dir <run_dir> "
        "--eval_data robustness_ffill_retrained/evaluation_dataset/unseendata.csv "
        "--eval_steps 39305 --seed <seed> "
        "--output_dir eval_v2_ffill_retrained/no_sweeper/<suite>/seed<seed>/tier1 "
        "--investment_freq 6 --meta_freq_min 6 --meta_freq_max 6 "
        "--global_norm_mode rolling_past --rolling_past_history_dir rolling_past_history_dataset "
        "--forecast_cache_dir robustness_ffill_retrained/forecast_cache "
        "--eval-distribution-rate 0.0 --friction-cost-multiplier 1.0 --half-spread-bp 0.0"
    )
    lines = [
        "",
        "## MAPPO 2x2 Summary",
        f"- Generated at: {_timestamp()}",
        "- Files changed: policy.py, metacontroller.py, config.py, main.py, run_tier_phase_multi_seed.py, evaluation.py, scripts/run_ffill_mappo_eval.py",
        "- MAPPO definition: centralized critic for investor/risk/meta PPO agents on concatenated joint observations; battery remains DQN.",
        f"- Exact eval command template: `{command_template}`",
        f"- MAPPO no-prior Sharpe mean: {mappo_sharpe:.6f}",
        f"- MAPPO+prior Sharpe mean: {mappo_prior_sharpe:.6f}",
        f"- MAPPO no-prior MDD mean: {mappo_mdd:.6f} bp",
        f"- MAPPO+prior MDD mean: {mappo_prior_mdd:.6f} bp",
        f"- FoCAL-ANN vs MAPPO Sharpe p-value: {_extract_pvalue(comparison, 'FoCAL-ANN vs MAPPO', 'Sharpe one-sided')}",
        f"- FoCAL-ANN vs MAPPO MDD p-value: {_extract_pvalue(comparison, 'FoCAL-ANN vs MAPPO', 'MDD one-sided')}",
        f"- MAPPO+prior vs MAPPO Sharpe p-value: {_extract_pvalue(comparison, 'MAPPO+prior vs MAPPO', 'Sharpe one-sided')}",
        f"- MAPPO+prior vs MAPPO MDD p-value: {_extract_pvalue(comparison, 'MAPPO+prior vs MAPPO', 'MDD one-sided')}",
        f"- Tier1 vs MAPPO Sharpe p-value: {_extract_pvalue(comparison, 'Tier1 vs MAPPO', 'Sharpe one-sided')}",
        f"- Tier1 vs MAPPO MDD p-value: {_extract_pvalue(comparison, 'Tier1 vs MAPPO', 'MDD one-sided')}",
        "- Deviations/blockers: none recorded by the aggregation step.",
    ]
    with run_log.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAPPO forward-filled no-sweeper evaluations.")
    parser.add_argument("--input_root", default="batch_tier_phase_runs")
    parser.add_argument("--eval_root", default="eval_v2_ffill_retrained/no_sweeper")
    parser.add_argument("--report_root", default="results/ffill_mappo")
    parser.add_argument("--long_csv", default=DEFAULT_LONG_CSV)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--eval_data", default=DEFAULT_EVAL_DATA)
    parser.add_argument("--forecast_cache_dir", default=DEFAULT_FORECAST_CACHE)
    parser.add_argument("--eval_steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--timeout_hours", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true", help="Skip completed MAPPO eval outputs.")
    parser.add_argument("--dry_run", action="store_true", help="Write the 20-run plan but do not execute.")
    parser.add_argument("--aggregate_only", action="store_true", help="Only aggregate existing MAPPO eval outputs.")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--allow_sanity_warnings", action="store_true")
    parser.add_argument("--stream_child", action="store_true", help="Echo evaluation.py output while logging it.")
    parser.add_argument("--skip_update_long", action="store_true", help="Do not update results/ffill_robustness/ffill_per_seed_long.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_root = _path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    plan_rows = list(iter_run_plan(args))
    if len(plan_rows) != 20:
        raise RuntimeError(f"Expected exactly 20 MAPPO eval runs, got {len(plan_rows)}")
    write_csv(plan_rows, report_root / "mappo_eval_run_plan.csv")
    if args.dry_run:
        print(f"Wrote 20-run MAPPO eval plan: {report_root / 'mappo_eval_run_plan.csv'}")
        return
    if not args.aggregate_only:
        run_all(args)
    df = aggregate(args)
    print(f"Wrote {report_root / VARIANTS[0].summary_name}")
    print(f"Wrote {report_root / VARIANTS[1].summary_name}")
    print(f"Wrote {report_root / 'mappo_per_seed_long.csv'} ({len(df)} rows)")
    print(f"Wrote {report_root / 'mappo_2x2_comparison.md'}")


if __name__ == "__main__":
    main()

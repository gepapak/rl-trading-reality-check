"""Eval-only no-sweeper pass for existing FoCAL/Tier1 runs.

This script reuses final_models under batch_tier_phase_runs and writes a
separate eval_v2 tree. It never trains or overwrites v1 artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_tier_phase_multi_seed import (  # noqa: E402
    extract_eval_metrics,
    find_latest_file,
    format_cmd,
    resolve_eval_steps,
    write_seed_summary_csv,
)


SUITES = {
    "tiers_tier1_baseline": {
        "run_name": "tier1_seed{seed}",
        "display": "Tier1 baseline",
        "canonical_variant": "tier1",
    },
    "tiers_tier1_forecast_utilization": {
        "run_name": "tier1_forecast_utilization_seed{seed}",
        "display": "Tier1 + forecast prior",
        "canonical_variant": "tier1_forecast_utilization",
    },
    "abl_forecast_prior_only": {
        "run_name": "tier1_forecast_utilization_seed{seed}",
        "display": "Ablation: forecast prior only",
        "canonical_variant": "tier1_forecast_utilization",
    },
    "abl_no_prior_path_control": {
        "run_name": "tier1_forecast_utilization_seed{seed}",
        "display": "Ablation: no prior path control",
        "canonical_variant": "tier1_forecast_utilization",
    },
    "abl_weak_conformal_gate": {
        "run_name": "tier1_forecast_utilization_seed{seed}",
        "display": "Ablation: weak conformal gate",
        "canonical_variant": "tier1_forecast_utilization",
    },
}


DEFAULT_SEEDS = [7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202]


def run_command(cmd: List[str], label: str, timeout_hours: float) -> Dict[str, Any]:
    started_at = datetime.now()
    start = time.time()
    print(f"\n{'=' * 100}")
    print(f"Starting: {label}")
    print(f"Command: {format_cmd(cmd)}")
    print(f"Started at: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 100}\n")

    timeout_s = max(60, int(float(timeout_hours) * 3600.0))
    process = subprocess.Popen(
        cmd,
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
    for line in process.stdout:
        print(line, end="")
        if time.time() > deadline:
            timed_out = True
            process.kill()
            print(f"\n[TIMEOUT] {label} exceeded {timeout_s}s; process killed.")
            break
    process.wait()
    elapsed = time.time() - start
    ok = process.returncode == 0 and not timed_out
    print(f"\n{'=' * 100}")
    print(f"{'SUCCESS' if ok else 'FAILED'}: {label} (exit={process.returncode})")
    print(f"Duration: {int(elapsed // 60)}m {int(elapsed % 60)}s")
    print(f"{'=' * 100}\n")
    return {
        "success": bool(ok),
        "returncode": int(process.returncode),
        "duration_seconds": float(elapsed),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now().isoformat(),
    }


def read_protocol(suite_dir: Path) -> Dict[str, Any]:
    protocol_path = suite_dir / "phase_protocol.json"
    if not protocol_path.is_file():
        return {}
    with open(protocol_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def write_run_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"- {datetime.now().isoformat(timespec='seconds')} {message}\n")


def evaluate_one(
    *,
    suite: str,
    seed: int,
    args: argparse.Namespace,
    run_number: int,
    total_runs: int,
    run_log: Path,
) -> Dict[str, Any]:
    spec = SUITES[suite]
    run_name = spec["run_name"].format(seed=seed)
    run_dir = Path(args.input_root) / suite / f"seed{seed}" / run_name
    final_models_dir = run_dir / "final_models"
    if not final_models_dir.is_dir():
        raise FileNotFoundError(f"Missing final_models: {final_models_dir}")

    eval_out = Path(args.output_root) / "no_sweeper" / suite / f"seed{seed}" / "tier1"
    eval_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "evaluation.py",
        "--mode",
        "tiers",
        "--tiers_only",
        "tier1",
        "--tier1_dir",
        str(run_dir),
        "--eval_data",
        str(args.eval_data),
        "--eval_steps",
        str(args.eval_steps),
        "--seed",
        str(seed),
        "--output_dir",
        str(eval_out),
        "--investment_freq",
        str(args.investment_freq),
        "--meta_freq_min",
        str(args.meta_freq_min),
        "--meta_freq_max",
        str(args.meta_freq_max),
        "--global_norm_mode",
        str(args.global_norm_mode),
        "--rolling_past_history_dir",
        str(args.rolling_past_history_dir),
        "--forecast_cache_dir",
        str(args.forecast_cache_dir),
        "--eval-distribution-rate",
        str(float(args.eval_distribution_rate)),
    ]

    label = f"[{run_number}/{total_runs}] No-sweeper eval {suite} seed={seed}"
    result = run_command(cmd, label, timeout_hours=float(args.timeout_hours))
    row: Dict[str, Any] = {
        "phase": "eval_v2_no_sweeper",
        "global_run_number": run_number,
        "total_planned_runs": total_runs,
        "seed": seed,
        "run_order": run_number,
        "run_name": spec["display"],
        "canonical_variant": spec["canonical_variant"],
        "suite": suite,
        "save_dir": str(run_dir),
        "eval_output_dir": str(eval_out),
        "training_success": True,
        "training_returncode": 0,
        "training_duration_seconds": 0.0,
        "evaluation_success": bool(result.get("success", False)),
        "evaluation_returncode": int(result.get("returncode", -1)),
        "evaluation_duration_seconds": float(result.get("duration_seconds", 0.0)),
    }
    if row["evaluation_success"]:
        eval_json_path = find_latest_file(str(eval_out), "evaluation_tiers_*.json")
        if eval_json_path:
            row["evaluation_results_json"] = eval_json_path
            row.update(extract_eval_metrics(eval_json_path, spec["canonical_variant"]))
            distributions = float(row.get("total_distributions_usd", 0.0) or 0.0)
            if abs(distributions) > float(args.distribution_tolerance_usd):
                row["distribution_check_warning"] = (
                    f"Expected zero distributions under no-sweeper eval, got {distributions}"
                )
                write_run_log(run_log, f"WARNING {suite} seed {seed}: {row['distribution_check_warning']}")
    return row


def write_run_plan(args: argparse.Namespace, suites: List[str], seeds: List[int], path: Path) -> None:
    rows = []
    run_no = 0
    total = len(suites) * len(seeds)
    for suite in suites:
        spec = SUITES[suite]
        for seed in seeds:
            run_no += 1
            run_dir = Path(args.input_root) / suite / f"seed{seed}" / spec["run_name"].format(seed=seed)
            eval_out = Path(args.output_root) / "no_sweeper" / suite / f"seed{seed}" / "tier1"
            rows.append(
                {
                    "global_run_number": run_no,
                    "total_planned_runs": total,
                    "seed": seed,
                    "suite": suite,
                    "run_name": spec["display"],
                    "canonical_variant": spec["canonical_variant"],
                    "save_dir": str(run_dir),
                    "eval_output_dir": str(eval_out),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eval_v2 no-sweeper evaluations without retraining.")
    parser.add_argument("--input_root", default="batch_tier_phase_runs")
    parser.add_argument("--output_root", default="eval_v2")
    parser.add_argument("--suites", nargs="+", default=list(SUITES.keys()), choices=list(SUITES.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--eval_data", default="evaluation_dataset/unseendata.csv")
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--forecast_cache_dir", default="forecast_cache")
    parser.add_argument("--global_norm_mode", default="rolling_past", choices=["rolling_past", "global"])
    parser.add_argument("--rolling_past_history_dir", default="rolling_past_history_dataset")
    parser.add_argument("--investment_freq", type=int, default=6)
    parser.add_argument("--meta_freq_min", type=int, default=6)
    parser.add_argument("--meta_freq_max", type=int, default=6)
    parser.add_argument("--eval_distribution_rate", "--eval-distribution-rate", dest="eval_distribution_rate", type=float, default=0.0)
    parser.add_argument("--timeout_hours", type=float, default=3.0)
    parser.add_argument("--continue_on_error", action="store_true", default=False)
    parser.add_argument("--distribution_tolerance_usd", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_steps is None:
        args.eval_steps = resolve_eval_steps(str(args.eval_data), None)

    output_root = Path(args.output_root)
    run_log = output_root / "RUN_LOG.md"
    output_root.mkdir(parents=True, exist_ok=True)
    write_run_log(run_log, "Started eval_v2 no-sweeper evaluation pass.")

    for suite in args.suites:
        protocol = read_protocol(Path(args.input_root) / suite)
        if protocol:
            write_run_log(
                run_log,
                f"Read protocol for {suite}: phase={protocol.get('phase')} hash={protocol.get('runtime_contract_hash')}",
            )

    write_run_plan(
        args,
        list(args.suites),
        list(args.seeds),
        output_root / "no_sweeper_run_plan.csv",
    )

    total_runs = len(args.suites) * len(args.seeds)
    run_no = 0
    all_ok = True
    for suite in args.suites:
        rows: List[Dict[str, Any]] = []
        summary_path = output_root / "no_sweeper" / suite / f"{suite}_no_sweeper_seed_summary.csv"
        for seed in args.seeds:
            run_no += 1
            try:
                row = evaluate_one(
                    suite=suite,
                    seed=int(seed),
                    args=args,
                    run_number=run_no,
                    total_runs=total_runs,
                    run_log=run_log,
                )
                rows.append(row)
                write_seed_summary_csv(rows, str(summary_path))
                if not bool(row.get("evaluation_success", False)):
                    all_ok = False
                    write_run_log(run_log, f"FAILED {suite} seed {seed}: evaluation returned failure.")
                    if not args.continue_on_error:
                        raise RuntimeError(f"Evaluation failed for {suite} seed {seed}")
            except Exception as exc:
                all_ok = False
                write_run_log(run_log, f"BLOCKED {suite} seed {seed}: {type(exc).__name__}: {exc}")
                if not args.continue_on_error:
                    raise
        write_seed_summary_csv(rows, str(summary_path))
        write_run_log(run_log, f"Wrote {summary_path}")

    write_run_log(run_log, "Finished eval_v2 no-sweeper evaluation pass.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

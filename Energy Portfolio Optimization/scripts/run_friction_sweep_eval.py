"""Eval-only friction sweep for existing Tier1 and FoCAL final_models.

This runner reuses trained models and writes a separate eval_v2/friction_sweep
tree. It does not train and does not overwrite batch_tier_phase_runs artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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
}


REGIMES = {
    "default": {"friction_cost_multiplier": 1.0, "half_spread_bp": 0.0},
    "medium": {"friction_cost_multiplier": 2.0, "half_spread_bp": 5.0},
    "high": {"friction_cost_multiplier": 3.0, "half_spread_bp": 10.0},
}


DEFAULT_SEEDS = [7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202]


def write_run_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"- {datetime.now().isoformat(timespec='seconds')} {message}\n")


def run_command(cmd: List[str], label: str, timeout_hours: float | None = None) -> Dict[str, Any]:
    started_at = datetime.now()
    start = time.time()
    print(f"\n{'=' * 100}")
    print(f"Starting: {label}")
    print(f"Command: {format_cmd(cmd)}")
    print(f"Started at: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 100}\n")

    timeout_s = None
    if timeout_hours is not None and float(timeout_hours) > 0.0:
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
    deadline = None if timeout_s is None else time.time() + timeout_s
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        if deadline is not None and time.time() > deadline:
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


def write_run_plan(
    *,
    args: argparse.Namespace,
    suites: Iterable[str],
    regimes: Iterable[str],
    seeds: Iterable[int],
    path: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    run_no = 0
    suites = list(suites)
    regimes = list(regimes)
    seeds = list(seeds)
    total = len(suites) * len(regimes) * len(seeds)
    for suite in suites:
        spec = SUITES[suite]
        for regime in regimes:
            params = REGIMES[regime]
            for seed in seeds:
                run_no += 1
                run_dir = Path(args.input_root) / suite / f"seed{seed}" / spec["run_name"].format(seed=seed)
                eval_out = Path(args.output_root) / suite / regime / f"seed{seed}" / "tier1"
                rows.append(
                    {
                        "global_run_number": run_no,
                        "total_planned_runs": total,
                        "seed": int(seed),
                        "suite": suite,
                        "regime": regime,
                        "run_name": spec["display"],
                        "canonical_variant": spec["canonical_variant"],
                        "friction_cost_multiplier": params["friction_cost_multiplier"],
                        "half_spread_bp": params["half_spread_bp"],
                        "eval_distribution_rate": args.eval_distribution_rate,
                        "save_dir": str(run_dir),
                        "eval_output_dir": str(eval_out),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def evaluate_one(
    *,
    suite: str,
    regime: str,
    seed: int,
    args: argparse.Namespace,
    run_number: int,
    total_runs: int,
) -> Dict[str, Any]:
    spec = SUITES[suite]
    params = REGIMES[regime]
    run_name = spec["run_name"].format(seed=seed)
    run_dir = Path(args.input_root) / suite / f"seed{seed}" / run_name
    final_models_dir = run_dir / "final_models"
    if not final_models_dir.is_dir():
        raise FileNotFoundError(f"Missing final_models: {final_models_dir}")

    eval_out = Path(args.output_root) / suite / regime / f"seed{seed}" / "tier1"
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
        "--friction-cost-multiplier",
        str(float(params["friction_cost_multiplier"])),
        "--half-spread-bp",
        str(float(params["half_spread_bp"])),
    ]

    label = f"[{run_number}/{total_runs}] Friction eval {suite} seed={seed} regime={regime}"
    result = run_command(cmd, label, timeout_hours=args.timeout_hours)
    row: Dict[str, Any] = {
        "phase": "eval_v2_friction_sweep",
        "global_run_number": run_number,
        "total_planned_runs": total_runs,
        "seed": int(seed),
        "run_order": run_number,
        "run_name": spec["display"],
        "canonical_variant": spec["canonical_variant"],
        "suite": suite,
        "friction_regime": regime,
        "friction_cost_multiplier": float(params["friction_cost_multiplier"]),
        "half_spread_bp": float(params["half_spread_bp"]),
        "eval_distribution_rate": float(args.eval_distribution_rate),
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
    return row


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def log_monotonicity_warnings(rows: List[Dict[str, Any]], run_log: Path) -> None:
    by_key: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        if not row.get("evaluation_success"):
            continue
        key = (str(row.get("suite", "")), int(row.get("seed", 0)))
        by_key.setdefault(key, {})[str(row.get("friction_regime", ""))] = row

    for (suite, seed), regime_rows in by_key.items():
        if not all(name in regime_rows for name in ("default", "medium", "high")):
            continue
        returns = {
            name: _as_float(regime_rows[name].get("total_return"))
            for name in ("default", "medium", "high")
        }
        if any(v is None for v in returns.values()):
            continue
        if not (returns["default"] + 1e-12 >= returns["medium"] >= returns["high"] - 1e-12):
            write_run_log(
                run_log,
                (
                    f"WARNING non-monotonic total_return for {suite} seed {seed}: "
                    f"default={returns['default']}, medium={returns['medium']}, high={returns['high']}"
                ),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eval-only friction sweep for existing Tier1/FoCAL models.")
    parser.add_argument("--input_root", default="batch_tier_phase_runs")
    parser.add_argument("--output_root", default="eval_v2/friction_sweep")
    parser.add_argument("--suites", nargs="+", default=list(SUITES.keys()), choices=list(SUITES.keys()))
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES.keys()), choices=list(REGIMES.keys()))
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
    parser.add_argument("--timeout_hours", type=float, default=None, help="Optional positive timeout per eval job. Omit for no timeout.")
    parser.add_argument("--continue_on_error", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_steps is None:
        args.eval_steps = resolve_eval_steps(str(args.eval_data), None)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_log = output_root / "RUN_LOG.md"
    write_run_log(
        run_log,
        (
            "Started friction sweep eval-only pass "
            f"(eval_distribution_rate={args.eval_distribution_rate})."
        ),
    )

    for suite in args.suites:
        protocol = read_protocol(Path(args.input_root) / suite)
        if protocol:
            write_run_log(
                run_log,
                f"Read protocol for {suite}: phase={protocol.get('phase')} hash={protocol.get('runtime_contract_hash')}",
            )

    write_run_plan(
        args=args,
        suites=list(args.suites),
        regimes=list(args.regimes),
        seeds=list(args.seeds),
        path=output_root / "friction_sweep_run_plan.csv",
    )

    total_runs = len(args.suites) * len(args.regimes) * len(args.seeds)
    run_no = 0
    all_rows: List[Dict[str, Any]] = []
    all_ok = True
    for suite in args.suites:
        for regime in args.regimes:
            rows: List[Dict[str, Any]] = []
            summary_path = output_root / suite / f"{suite}_{regime}_seed_summary.csv"
            for seed in args.seeds:
                run_no += 1
                try:
                    row = evaluate_one(
                        suite=suite,
                        regime=regime,
                        seed=int(seed),
                        args=args,
                        run_number=run_no,
                        total_runs=total_runs,
                    )
                    rows.append(row)
                    all_rows.append(row)
                    write_seed_summary_csv(rows, str(summary_path))
                    write_seed_summary_csv(all_rows, str(output_root / "friction_sweep_summary.csv"))
                    if not bool(row.get("evaluation_success", False)):
                        all_ok = False
                        write_run_log(run_log, f"FAILED {suite} {regime} seed {seed}: evaluation returned failure.")
                        if not args.continue_on_error:
                            raise RuntimeError(f"Evaluation failed for {suite} {regime} seed {seed}")
                except Exception as exc:
                    all_ok = False
                    write_run_log(run_log, f"BLOCKED {suite} {regime} seed {seed}: {type(exc).__name__}: {exc}")
                    if not args.continue_on_error:
                        raise
            write_seed_summary_csv(rows, str(summary_path))
            write_run_log(run_log, f"Wrote {summary_path}")

    write_seed_summary_csv(all_rows, str(output_root / "friction_sweep_summary.csv"))
    log_monotonicity_warnings(all_rows, run_log)
    write_run_log(run_log, "Finished friction sweep eval-only pass.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

"""Master runner for the "one tail, four illusions" evidence campaign. Everything lives inside Paper1.

Phases (run in order with --phase all, or individually):
  setup      build/verify the audit engine (code copy of Paper1/Prototype5 + data junctions + audit patch)
  validate   strict re-evaluation of frozen checkpoints; must reproduce frozen sleeve returns (gate)
  evals      learned agents (MARL, feasible-action, anchor) under shortcut protocols L1..L7
  retrain    MARL retrained under the price-taker shortcut (3 seeds) + its evaluations under L1 and L0
  aggregate  investor-only metrics, path/ruin checks, metric illusions, pre-registered prediction verdicts

Resume-safe: every job writes its own output folder; completed jobs (JSON with status=completed) are skipped.
Usage:  python run_scarcity_tail_campaign.py --phase all --workers 5
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

AUD = Path(__file__).resolve().parent
PAPER1 = AUD.parent
P5 = PAPER1 / "Prototype5"
ENG = AUD / "engine"
RUNS = AUD / "campaign_runs"
LOGS = RUNS / "logs"
INTERP = {"original": AUD / "data" / "unseendata_interpolated.csv", "v2": AUD / "data" / "unseendata_v2_interpolated.csv"}
EVAL_DATA = {"original": "evaluation_dataset_ffill/unseendata.csv", "v2": "evaluation_dataset_ffill/unseendata_v2.csv"}
ALL_SEEDS = [7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202]
SUB_SEEDS = [7, 42, 123]
RETRAIN_SEEDS = [7, 42, 123]
KEY_VARIANTS = ["L1_price_taker", "L2_no_solvency", "L3_price_taker_no_solvency"]
SIDE_VARIANTS = ["L4_load_proxy_liquidity", "L5_percent_payoff", "L6_interpolated_prices", "L7_thesis_like"]
NO_SOLVENCY_CFG = {"enable_trading_sleeve_margin": False}
L7_PARTS = {"interp", "percent_mtm", "price_taker", "no_solvency", "legacy_fee", "sweeper"}
THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "PYTHONUTF8": "1",
              "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "campaign.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def replace_arg(args: list[str], flag: str, value: str | None) -> list[str]:
    """Replace (or append) a flag's value; value=None removes a boolean flag."""
    out, i, done = [], 0, False
    while i < len(args):
        if args[i] == flag:
            if value is not None:
                out += [flag, value]
                i += 2
            else:
                i += 1
            done = True
        else:
            out.append(args[i])
            i += 1
    if not done and value is not None:
        out += [flag, value]
    return out


# ----------------------------------------------------------------------------------------------- job building
def base_eval_cmd(agent: str, seed: int, region: str, model_dir: Path) -> list[str]:
    marl = _load("marl_engine", ENG / "marl.py")
    saved = sys.argv
    sys.argv = ["marl.py"]
    try:
        margs = marl.parse_args()
    finally:
        sys.argv = saved
    cmd = [sys.executable, "evaluation.py", "--mode", "tiers", "--tiers_only", "tier1", "--tier1_dir", str(model_dir),
           "--seed", str(seed), "--eval_data", EVAL_DATA[region]]
    cmd += list(marl.EVAL_PROTOCOL_ARGS)
    cmd += marl._settlement_eval_args(margs, region)
    cmd += marl._liquidity_eval_args(margs, region)
    if agent in ("anchor", "feasible"):
        abl = _load("abl_engine", ENG / "Ablations" / "mechanism_ablations.py")
        arm = "focal_anchor" if agent == "anchor" else "feasible_action_full"
        cmd += ["--enable_forecast_utilization"] + abl.prior_args_for_arm(arm)
    return cmd


def apply_variant(cmd: list[str], variant: str, region: str) -> tuple[list[str], dict]:
    cfg: dict = {}
    parts = set()
    if variant == "L0_strict":
        return cmd, cfg
    if variant in ("L1_price_taker", "L3_price_taker_no_solvency", "L7_thesis_like"):
        parts.add("price_taker")
    if variant in ("L2_no_solvency", "L3_price_taker_no_solvency", "L7_thesis_like"):
        parts.add("no_solvency")
    if variant == "L4_load_proxy_liquidity":
        parts.add("load_proxy")
    if variant == "L5_percent_payoff":
        parts.add("percent_payoff")
    if variant in ("L6_interpolated_prices", "L7_thesis_like"):
        parts.add("interp")
    if variant == "L7_thesis_like":
        parts |= {"percent_mtm", "legacy_fee", "sweeper"}
    if variant.startswith("L7_minus_"):  # PREREGISTRATION.md addendum 3: leave-one-out of L7
        drop = variant[len("L7_minus_"):]
        assert drop in L7_PARTS, variant
        parts |= L7_PARTS - {drop}
    if variant == "L8_pctmtm_nosolv":
        parts |= {"percent_mtm", "no_solvency"}
    if "price_taker" in parts:
        cmd = replace_arg(cmd, "--liquidity_participation_cap_fraction", "0.0")
    if "no_solvency" in parts:
        cmd = replace_arg(cmd, "--mtm_loss_exit_threshold_pct", "1000.0")
        cfg.update(NO_SOLVENCY_CFG)
    if "load_proxy" in parts:
        cmd = replace_arg(cmd, "--impact_ref_notional", "sleeve")
        cmd = replace_arg(cmd, "--liquidity_volume_source", "load")
    if "percent_payoff" in parts:
        cmd = replace_arg(cmd, "--mtm_horizon_payoff_denominator_mode", "reference_price")
    if "interp" in parts:
        cmd = replace_arg(cmd, "--eval_data", str(INTERP[region]))
    if "percent_mtm" in parts:
        cmd = replace_arg(cmd, "--mtm_return_model", "percent_capped")
        cmd = replace_arg(cmd, "--disable_mtm_return_cap", None)
    if "legacy_fee" in parts:
        cmd = replace_arg(cmd, "--market_fee_model", "legacy_notional_fixed")
    if "sweeper" in parts:
        cmd = replace_arg(cmd, "--eval-distribution-rate", "0.1")
        cmd = replace_arg(cmd, "--distribution_rate", "0.1")
    return cmd, cfg


def model_dir_for(agent: str, seed: int, retrained: bool = False) -> Path:
    if retrained:
        return RUNS / "marl_price_taker_trained" / f"seed{seed}" / f"tier1_seed{seed}"
    if agent == "marl":
        return P5 / "batch_tier_phase_runs" / "prototype5_mappo_marl_final_v1" / f"seed{seed}" / f"tier1_seed{seed}"
    arm = "focal_anchor" if agent == "anchor" else "feasible_action_full"
    return (P5 / "Ablations" / "batch_tier_phase_runs" / "prototype5_mechanism_ablations_final_v1" / arm
            / f"seed{seed}" / f"tier1_forecast_utilization_seed{seed}")


def eval_jobs(phase: str) -> list[dict]:
    jobs = []

    def add(variant, agent, seeds, retrained=False, tag=None):
        for s in seeds:
            md = model_dir_for(agent, s, retrained)
            for region in ("original", "v2"):
                name = f"{variant}__{tag or agent}__seed{s}__{region}"
                jobs.append(dict(name=name, variant=variant, agent=tag or agent, seed=s, region=region, model_dir=md,
                                 base_agent=agent))

    if phase == "validate":
        add("L0_strict", "marl", [7]); add("L0_strict", "feasible", [7]); add("L0_strict", "anchor", [7])
    elif phase == "evals":
        for v in KEY_VARIANTS:
            add(v, "marl", ALL_SEEDS); add(v, "feasible", ALL_SEEDS); add(v, "anchor", [7])
        for v in SIDE_VARIANTS:
            add(v, "marl", SUB_SEEDS); add(v, "feasible", SUB_SEEDS); add(v, "anchor", [7])
    elif phase == "l7_ext":  # PREREGISTRATION.md addendum 2: L7 for the remaining MARL seeds
        add("L7_thesis_like", "marl", [s for s in ALL_SEEDS if s not in SUB_SEEDS])
    elif phase == "l7_loo":  # PREREGISTRATION.md addendum 3
        for v in [f"L7_minus_{c}" for c in sorted(L7_PARTS)] + ["L8_pctmtm_nosolv"]:
            add(v, "marl", SUB_SEEDS); add(v, "anchor", [7])
    elif phase == "retrain_evals":
        add("L1_price_taker", "marl", RETRAIN_SEEDS, retrained=True, tag="marl_trained_price_taker")
        add("L0_strict", "marl", RETRAIN_SEEDS, retrained=True, tag="marl_trained_price_taker")
    return jobs


def job_out(job: dict) -> Path:
    return RUNS / "evals" / job["name"]


def job_done(job: dict) -> bool:
    for f in glob.glob(str(job_out(job) / "**" / "evaluation_tiers_*.json"), recursive=True):
        try:
            if json.load(open(f, encoding="utf-8"))["tiers"]["tier1"].get("status") == "completed":
                return True
        except Exception:
            pass
    return False


def run_eval_job(job: dict) -> tuple[str, int, int]:
    if job_done(job):
        return job["name"], 0, 0
    if not (job["model_dir"] / "final_models").is_dir():
        log(f"[MISSING] {job['name']}: no final_models at {job['model_dir']}")
        return job["name"], 2, 0
    cmd = base_eval_cmd(job["base_agent"], job["seed"], job["region"], job["model_dir"])
    cmd, cfg = apply_variant(cmd, job["variant"], job["region"])
    out = job_out(job)
    cmd = replace_arg(cmd, "--output_dir", str(out))
    env = dict(os.environ, **THREAD_ENV, SIM_AUDIT_VARIANT=job["variant"])
    if job["variant"] != "L0_strict" or job["agent"].startswith("marl_trained"):
        env["SIM_AUDIT_ALLOW_PROTOCOL_MISMATCH"] = "1"
    if cfg:
        env["SIM_AUDIT_CFG_OVERRIDES"] = json.dumps(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_job.json").write_text(json.dumps({**{k: str(v) for k, v in job.items()}, "cmd": cmd, "cfg_overrides": cfg}, indent=1))
    t = time.time()
    with (LOGS / f"{job['name']}.log").open("w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.run(cmd, cwd=str(ENG), env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    return job["name"], rc, int(time.time() - t)


def run_jobs(jobs: list[dict], workers: int) -> list[tuple]:
    todo = [j for j in jobs if not job_done(j)]
    log(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already complete, running {len(todo)} with {workers} workers")
    res = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for name, rc, sec in ex.map(run_eval_job, todo):
            log(f"  {name}: exit={rc} ({sec}s)")
            res.append((name, rc, sec))
    return res


# ----------------------------------------------------------------------------------------------- phases
def phase_setup() -> None:
    rc = subprocess.run([sys.executable, str(AUD / "setup_audit_engine.py")]).returncode
    if rc:
        raise SystemExit("setup failed")
    for p in INTERP.values():
        if not p.is_file():
            subprocess.run([sys.executable, str(AUD / "make_interpolated_eval_data.py")], check=True)
    log("setup ok")


def frozen_json(job: dict) -> Path | None:
    if job["base_agent"] == "marl":
        pat = job["model_dir"].parent / ("evaluations_2025" if job["region"] == "original" else "evaluations_2025_v2") / "tier1" / "evaluation_tiers_*.json"
    else:
        pat = job["model_dir"].parent / ("evaluations_2025" if job["region"] == "original" else "evaluations_2025_v2") / "tier1_forecast_utilization" / "evaluation_tiers_*.json"
    fs = sorted(glob.glob(str(pat)))
    return Path(fs[-1]) if fs else None


def sleeve_return(json_path: Path) -> float:
    t = json.load(open(json_path, encoding="utf-8"))["tiers"]["tier1"]
    return float(t["sleeve_metrics"]["sleeve_trading_return_pct"])


def phase_validate(workers: int) -> None:
    jobs = eval_jobs("validate")
    run_jobs(jobs, workers)
    ok = True
    for j in jobs:
        new = sorted(glob.glob(str(job_out(j) / "**" / "evaluation_tiers_*.json"), recursive=True))
        ref = frozen_json(j)
        if not new or ref is None:
            log(f"[VALIDATE] {j['name']}: missing output or reference"); ok = False; continue
        a, b = sleeve_return(Path(new[-1])), sleeve_return(ref)
        good = abs(a - b) <= 1e-6
        ok &= good
        log(f"[VALIDATE] {j['name']}: audit={a:.8f} frozen={b:.8f} -> {'OK' if good else 'MISMATCH'}")
    (RUNS / "VALIDATION_PASSED" if ok else RUNS / "VALIDATION_FAILED").write_text(datetime.now().isoformat())
    if not ok:
        raise SystemExit("validation gate FAILED: shortcut results must not be used")
    log("validation gate PASSED")


def phase_evals(workers: int) -> None:
    if not (RUNS / "VALIDATION_PASSED").exists():
        raise SystemExit("run --phase validate first (gate)")
    run_jobs(eval_jobs("evals"), workers)


def phase_retrain(parallel: int) -> None:
    if not (RUNS / "VALIDATION_PASSED").exists():
        raise SystemExit("run --phase validate first (gate)")
    suite = RUNS / "marl_price_taker_trained"

    def train(seed: int):
        if (suite / f"seed{seed}" / f"tier1_seed{seed}" / "final_models").is_dir():
            return seed, 0
        cmd = [sys.executable, "marl.py", "--seeds", str(seed), "--suite_dir", str(suite),
               "--result_root", str(RUNS / "marl_price_taker_trained_results"),
               "--liquidity_participation_cap_fraction", "0.0", "--skip_training_if_complete"]
        with (LOGS / f"retrain_price_taker_seed{seed}.log").open("w", encoding="utf-8", errors="replace") as f:
            rc = subprocess.run(cmd, cwd=str(ENG), env=dict(os.environ, **THREAD_ENV), stdout=f, stderr=subprocess.STDOUT).returncode
        return seed, rc

    log(f"retraining MARL under price-taker shortcut, seeds {RETRAIN_SEEDS}, {parallel} in parallel")
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for seed, rc in ex.map(train, RETRAIN_SEEDS):
            log(f"  retrain seed {seed}: exit={rc}")
    run_jobs(eval_jobs("retrain_evals"), parallel)


def phase_aggregate() -> None:
    rc = subprocess.run([sys.executable, str(AUD / "aggregate_scarcity_tail_campaign.py")]).returncode
    if rc:
        raise SystemExit("aggregate failed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["setup", "validate", "evals", "retrain", "l7_ext", "l7_loo", "aggregate", "all"], default="all")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--retrain_parallel", type=int, default=3)
    ap.add_argument("--dry_run", action="store_true", help="print job list and first commands only")
    a = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        for ph in ("validate", "evals", "retrain_evals"):
            js = eval_jobs(ph)
            print(f"{ph}: {len(js)} jobs")
            j = js[0]
            cmd, cfg = apply_variant(base_eval_cmd(j["base_agent"], j["seed"], j["region"], j["model_dir"]), j["variant"], j["region"])
            print("  e.g.", j["name"], "cfg", cfg)
            print("  ", subprocess.list2cmdline(cmd)[:600], "...")
        return 0
    phases = ["setup", "validate", "evals", "retrain", "aggregate"] if a.phase == "all" else [a.phase]
    for ph in phases:
        log(f"=== phase {ph} ===")
        {"setup": phase_setup, "validate": lambda: phase_validate(a.workers), "evals": lambda: phase_evals(a.workers),
         "retrain": lambda: phase_retrain(a.retrain_parallel), "aggregate": phase_aggregate,
         "l7_ext": lambda: run_jobs(eval_jobs("l7_ext"), a.workers),
         "l7_loo": lambda: run_jobs(eval_jobs("l7_loo"), a.workers)}[ph]()
    log("campaign finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

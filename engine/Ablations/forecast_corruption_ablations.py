#!/usr/bin/env python3
"""Evaluate one trained forecast-aware arm with corrupted forecasts.

The default target is the headline deterministic FoCAL anchor. This is the
causally relevant placebo test: zero, shuffled, and sign-flipped forecast
inputs are applied to the same controller whose forecast value is claimed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mechanism_ablations as mech


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
PARENT_ROOT = PROJECT_ROOT
SEEDS = mech.SEEDS
MODES = ["zero_edge", "shuffle", "sign_flip"]
REGIONS = ["original", "v2"]

EVAL_PROTOCOL_ARGS = [
    "--investment_freq",
    "6",
    "--meta_freq_min",
    "6",
    "--meta_freq_max",
    "6",
    "--global_norm_mode",
    "rolling_past",
    "--rolling_past_history_dir",
    "rolling_past_history_dataset_ffill",
    "--eval-distribution-rate",
    "0.0",
    "--mtm_return_model",
    "horizon_settlement",
    "--mtm_reference_price_dkk_per_mwh",
    "500",
    "--mtm_settlement_horizon_steps",
    "6",
    "--mtm_entry_price_mode",
    "current_price",
    "--mtm_horizon_payoff_denominator_mode",
    "mwh_volume",
    "--disable_mtm_return_cap",
    "--investor_notional_sizing_base",
    "initial_trading_sleeve",
    "--max_position_size",
    "0.1",
    "--capital_allocation_fraction",
    "0.6",
    "--mtm_loss_exit_threshold_pct",
    "0.15",
    "--friction_cost_multiplier",
    "1.0",
    *mech.MARKET_FEE_ARGS,
    "--no_trade_threshold",
    "0.01",
    "--no_trade_threshold_reference",
    "executable_capacity",
    "--half_spread_bp",
    "5.0",
    "--impact_coef_bp",
    "20.0",
    "--impact_exponent",
    "0.5",
    "--impact_ref_notional",
    "volume",
    "--liquidity_participation_cap_fraction",
    "0.25",
    "--liquidity_volume_source",
    "impact_volume",
    "--liquidity_volume_multiplier",
    "1.0",
    "--liquidity_min_volume_mwh",
    "1.0",
    "--liquidity_tail_impact_threshold_dkk_per_mwh",
    "5000.0",
    "--liquidity_tail_impact_multiplier",
    "3.0",
    "--liquidity_tail_impact_power",
    "1.0",
    "--liquidity_tail_impact_max_multiplier",
    "10.0",
    "--enable_collateral_cash_drag",
    "--collateral_notional_margin_fraction",
    "0.02",
    "--collateral_stress_loss_fraction",
    "0.10",
    "--collateral_stress_price_dkk_per_mwh",
    "25000.0",
    "--collateral_funding_rate_annual",
    "0.05",
    "--collateral_tradeable_haircut",
    "1.0",
    "--distribution_rate",
    "0.0",
    "--meta_controller_rule_based",
    "--risk_controller_rule_based",
    "--enable_forecast_utilization",
    "--log-sleeve",
    "--sleeve-sharpe-mode",
    "daily_hac_7",
]


def _force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run(cmd: list[str], label: str, log_path: Path, *, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(x) for x in cmd)
    print("\n" + "=" * 100)
    print(f"[RUN] {label}")
    print(printable)
    print(f"Log: {log_path}")
    print("=" * 100)
    if dry_run:
        return
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n[{label}]\n{printable}\n")
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=_subprocess_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}. See {log_path}")


def _eval_data(region: str) -> Path:
    if region == "v2":
        return PARENT_ROOT / "evaluation_dataset_ffill" / "unseendata_v2.csv"
    return PARENT_ROOT / "evaluation_dataset_ffill" / "unseendata.csv"


def _settlement_data(region: str) -> Path:
    if region == "v2":
        return PARENT_ROOT / "evaluation_dataset_ffill" / "unseendata_v2_settlement_real_v2.csv"
    return PARENT_ROOT / "evaluation_dataset_ffill" / "unseendata_settlement_real_v2.csv"


def _liquidity_volume_data(region: str) -> Path:
    if region == "v2":
        return PARENT_ROOT / "evaluation_dataset_ffill" / "unseendata_v2_liquidity_volume_real_v1.csv"
    return PARENT_ROOT / "evaluation_dataset_ffill" / "unseendata_liquidity_volume_real_v1.csv"


def _source_eval_cache(region: str) -> Path:
    base = PARENT_ROOT / "forecast_cache_settlement_hourly_v2" / "forecast_cache_eval_episode20_2025"
    if region == "v2":
        return base / "forecast_cache_eval_episode20_2025-unseendata_v2"
    return base / "forecast_cache_eval_episode20_2025-full"


def _dest_eval_cache(cache_root: Path, mode: str, region: str) -> Path:
    return cache_root / mode / region


def _assert_shared_inputs() -> None:
    required = []
    for region in REGIONS:
        required.extend([
            _eval_data(region),
            _settlement_data(region),
            _liquidity_volume_data(region),
            _source_eval_cache(region),
        ])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing shared Prototype files:\n" + "\n".join(missing))


def _assert_main_checkpoints(
    suite_root: Path,
    seeds: list[int],
    *,
    arm: str,
) -> None:
    missing = []
    for seed in seeds:
        final_models = (
            suite_root
            / f"seed{seed}"
            / f"tier1_forecast_utilization_seed{seed}"
            / "final_models"
        )
        if not final_models.is_dir():
            missing.append(str(final_models))
    if missing:
        raise FileNotFoundError(
            f"Missing {arm} checkpoints. Train them first:\n"
            f"  python Ablations/mechanism_ablations.py --arms {arm} --seeds "
            + " ".join(str(s) for s in seeds)
            + "\nMissing:\n"
            + "\n".join(missing)
        )


def _build_cache_args(args: argparse.Namespace, mode: str, region: str, dest: Path) -> list[str]:
    cmd = [
        args.python,
        "make_forecast_cache_ablation.py",
        "--source",
        str(_source_eval_cache(region)),
        "--dest",
        str(dest),
        "--mode",
        mode,
        "--seed",
        str(int(args.cache_seed)),
        "--price_data",
        str(_eval_data(region)),
        "--entry_price_mode",
        "current_price",
        "--steps_per_day",
        "144",
    ]
    return cmd


def _eval_cmd(
    args: argparse.Namespace,
    *,
    seed: int,
    mode: str,
    region: str,
    cache_dir: Path,
    suite_root: Path,
    output_root: Path,
) -> list[str]:
    model_dir = (
        suite_root
        / f"seed{seed}"
        / f"tier1_forecast_utilization_seed{seed}"
    )
    output_dir = output_root / mode / region / f"seed{seed}"
    cmd = [
        args.python,
        "evaluation.py",
        "--mode",
        "tiers",
        "--tiers_only",
        "tier1",
        "--tier1_dir",
        str(model_dir),
        "--seed",
        str(int(seed)),
        "--output_dir",
        str(output_dir),
        "--eval_data",
        str(_eval_data(region)),
        "--forecast_cache_dir",
        str(cache_dir),
    ]
    cmd += EVAL_PROTOCOL_ARGS
    cmd += mech.prior_args_for_arm(args.arm)
    cmd += [
        "--mtm_settlement_price_mode",
        "external_series",
        "--mtm_external_settlement_price_data_path",
        str(_settlement_data(region)),
        "--mtm_external_settlement_price_column",
        "settlement_price",
        "--mtm_external_settlement_timestamp_column",
        "timestamp",
        "--mtm_external_settlement_min_price_dkk_per_mwh",
        str(float(args.settlement_clip_min_dkk)),
        "--mtm_external_settlement_max_price_dkk_per_mwh",
        str(float(args.settlement_clip_max_dkk)),
        "--impact_volume_data",
        str(_liquidity_volume_data(region)),
        "--impact_volume_column",
        "market_volume_mwh",
        "--impact_volume_unit",
        "mwh",
        "--impact_volume_timestamp_column",
        "timestamp",
        "--impact_volume_max_staleness_min",
        "90.0",
        "--impact_volume_price_floor_dkk_per_mwh",
        "50.0",
    ]
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--regions", nargs="+", default=REGIONS, choices=REGIONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument(
        "--arm",
        choices=sorted(mech.ARM_SPECS),
        default="focal_anchor",
        help="Trained arm whose forecast inputs are replaced at evaluation time.",
    )
    parser.add_argument(
        "--suite_root",
        default=None,
        help=(
            "Directory containing seed*/tier1_forecast_utilization_seed*. "
            "Defaults to the selected arm under the mechanism suite."
        ),
    )
    parser.add_argument(
        "--result_root",
        default=None,
        help=(
            "Output root. Defaults to the headline path for focal_anchor and "
            "an arm-suffixed path for any other target."
        ),
    )
    parser.add_argument("--cache_root", default="Ablations/forecast_cache_corruptions_v1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cache_seed", type=int, default=42)
    parser.add_argument("--rebuild_caches", action="store_true")
    parser.add_argument("--skip_cache_build", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--settlement_clip_min_dkk", type=float, default=mech.SETTLEMENT_CLIP_MIN_DKK_DEFAULT)
    parser.add_argument("--settlement_clip_max_dkk", type=float, default=mech.SETTLEMENT_CLIP_MAX_DKK_DEFAULT)
    return parser.parse_args()


def main() -> int:
    _force_utf8_stdio()
    args = parse_args()
    suite_root = (
        PROJECT_ROOT / args.suite_root
        if args.suite_root
        else PROJECT_ROOT
        / "Ablations"
        / "batch_tier_phase_runs"
        / "prototype5_mechanism_ablations_final_v1"
        / str(args.arm)
    )
    default_result_root = (
        "Ablations/results/prototype5_forecast_corruption_final_v1"
        if args.arm == "focal_anchor"
        else f"Ablations/results/prototype5_forecast_corruption_{args.arm}_final_v1"
    )
    result_root = PROJECT_ROOT / (args.result_root or default_result_root)
    cache_root = PROJECT_ROOT / args.cache_root
    logs_dir = result_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        _assert_shared_inputs()
        _assert_main_checkpoints(
            suite_root,
            [int(s) for s in args.seeds],
            arm=str(args.arm),
        )

    print("Ablation workspace:", SCRIPT_ROOT)
    print("Shared Prototype5 root:", PROJECT_ROOT)
    print("Modes:", args.modes)
    print("Regions:", args.regions)
    print("Seeds:", args.seeds)
    print("Target arm:", args.arm)
    print("Checkpoint suite:", suite_root)

    for mode in args.modes:
        for region in args.regions:
            dest = _dest_eval_cache(cache_root, mode, region)
            if args.rebuild_caches and dest.exists():
                resolved_dest = dest.resolve()
                resolved_root = cache_root.resolve()
                if resolved_dest == resolved_root or resolved_root not in resolved_dest.parents:
                    raise RuntimeError(f"Refusing to delete unexpected cache path: {resolved_dest}")
                if not args.dry_run:
                    shutil.rmtree(resolved_dest)
            if not args.skip_cache_build:
                if dest.exists() and not args.rebuild_caches:
                    print(f"[SKIP] cache exists: {dest}")
                else:
                    _run(
                        _build_cache_args(args, mode, region, dest),
                        f"build {mode} cache for {region}",
                        logs_dir / f"build_{mode}_{region}.log",
                        dry_run=bool(args.dry_run),
                    )

            for seed in args.seeds:
                _run(
                    _eval_cmd(
                        args,
                        seed=int(seed),
                        mode=mode,
                        region=region,
                        cache_dir=dest,
                        suite_root=suite_root,
                        output_root=result_root,
                    ),
                    f"evaluate {args.arm} seed {seed} with {mode} cache on {region}",
                    logs_dir / f"{mode}_{region}_seed{seed}.log",
                    dry_run=bool(args.dry_run),
                )

    print("\nDone. Corruption results under:", result_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

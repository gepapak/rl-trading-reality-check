#!/usr/bin/env python3
"""Run the Prototype5 final-campaign mechanism arms and ablations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
PARENT_ROOT = PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))
from protocol_preflight import assert_final_data_contract

SEEDS = [7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202]

SETTLEMENT_CLIP_MIN_DKK_DEFAULT = -111750.0
SETTLEMENT_CLIP_MAX_DKK_DEFAULT = 111750.0
MARKET_FEE_ARGS = [
    "--market_fee_model", "nord_pool_intraday_2026",
    "--transaction_fee_dkk_per_mwh", "0.9238",
    "--annual_market_access_fee_dkk", "160175.0",
    "--market_access_fee_allocation_fraction", "1.0",
    "--market_fee_source_id", "nord_pool_nordic_baltic_2026_standard_participant",
]


def _rel(path: str) -> str:
    return path


COMMON_PROTOCOL_ARGS = [
    "--episode_data_dir", _rel("training_dataset_ffill"),
    "--global_norm_mode", "rolling_past",
    "--rolling_past_history_dir", _rel("rolling_past_history_dataset_ffill"),
    "--eval_data", _rel("evaluation_dataset_ffill/unseendata.csv"),
    "--lr", "0.0003",
    "--ent_coef", "0.03",
    "--algo", "mappo",
    "--eval-distribution-rate", "0.0",
    "--mtm_return_model", "horizon_settlement",
    "--mtm_reference_price_dkk_per_mwh", "500",
    "--mtm_settlement_horizon_steps", "6",
    "--mtm_entry_price_mode", "current_price",
    "--mtm_horizon_payoff_denominator_mode", "mwh_volume",
    "--disable_mtm_return_cap",
    "--investor_notional_sizing_base", "initial_trading_sleeve",
    "--max_position_size", "0.1",
    "--capital_allocation_fraction", "0.6",
    "--mtm_loss_exit_threshold_pct", "0.15",
    "--friction_cost_multiplier", "1.0",
    *MARKET_FEE_ARGS,
    "--no_trade_threshold", "0.01",
    "--no_trade_threshold_reference", "executable_capacity",
    "--half_spread_bp", "5.0",
    "--impact_coef_bp", "20.0",
    "--impact_exponent", "0.5",
    "--impact_ref_notional", "volume",
    "--liquidity_participation_cap_fraction", "0.25",
    "--liquidity_volume_source", "impact_volume",
    "--liquidity_volume_multiplier", "1.0",
    "--liquidity_min_volume_mwh", "1.0",
    "--liquidity_tail_impact_threshold_dkk_per_mwh", "5000.0",
    "--liquidity_tail_impact_multiplier", "3.0",
    "--liquidity_tail_impact_power", "1.0",
    "--liquidity_tail_impact_max_multiplier", "10.0",
    "--enable_collateral_cash_drag",
    "--collateral_notional_margin_fraction", "0.02",
    "--collateral_stress_loss_fraction", "0.10",
    "--collateral_stress_price_dkk_per_mwh", "25000.0",
    "--collateral_funding_rate_annual", "0.05",
    "--collateral_tradeable_haircut", "1.0",
    "--distribution_rate", "0.0",
    "--meta_controller_rule_based",
    "--risk_controller_rule_based",
    "--sleeve_sharpe_mode", "daily_hac_7",
]


BASE_V3_PRIOR_ARGS = [
    "--forecast_prior_control_mode", "distributional",
    "--forecast_prior_window", "500",
    "--forecast_prior_min_samples", "50",
    "--forecast_prior_hit_lcb_z", "1.64",
    "--forecast_prior_default_residual", "500.0",
    "--forecast_prior_directional_floor", "0.50",
    "--forecast_prior_error_hurdle", "0.50",
    "--forecast_prior_use_direction_confidence", "true",
    "--forecast_prior_direction_confidence_power", "1.0",
    "--forecast_prior_calibrate_on_decision_grid", "true",
    "--forecast_prior_max_abs_exposure", "0.60",
    "--forecast_prior_distributional_window", "2000",
    "--forecast_prior_distributional_min_samples", "50",
    "--forecast_prior_distributional_tail_quantile", "0.99",
    "--forecast_prior_distributional_disaster_quantile", "0.999",
    "--forecast_prior_distributional_loss_budget_fraction", "0.05",
    "--forecast_prior_distributional_disaster_loss_budget_fraction", "0.20",
    "--forecast_prior_distributional_conditional_tail_floor", "2000.0",
    "--forecast_prior_distributional_disaster_tail_floor", "25000.0",
    "--forecast_prior_distributional_default_tail_return", "25000.0",
    "--forecast_prior_distributional_return_clip", "111750.0",
    "--forecast_prior_distributional_edge_scale", "250.0",
    "--forecast_prior_distributional_edge_hurdle", "0.0",
    "--forecast_prior_distributional_confidence_power", "1.0",
    "--forecast_prior_distributional_bucket_mode", "conditional",
    "--forecast_prior_distributional_use_confidence_weight", "true",
    "--forecast_prior_distributional_fixed_cap_abs", "0.10",
    "--forecast_prior_distributional_cold_start_exposure", "0.0",
    "--forecast_prior_observation_only_mappo", "false",
    "--forecast_prior_feasible_action_mappo", "false",
    "--forecast_prior_mirror_mappo", "false",
    "--forecast_mirror_prior_concentration", "8.0",
    "--forecast_mirror_trust_max", "1.0",
    "--forecast_mirror_trust_hidden_dim", "32",
    "--forecast_mirror_trust_initial", "0.10",
    "--forecast_mirror_trust_regularization", "0.01",
    "--forecast_mirror_trust_min_samples", "32",
    "--forecast_mirror_trust_batch_size", "256",
    "--forecast_mirror_trust_gradient_clip", "5.0",
    "--forecast_mirror_counterfactual_learning", "true",
    "--forecast_mirror_mask_forecast_observation", "false",
    "--forecast_mirror_counterfactual_grid_size", "41",
    "--forecast_mirror_counterfactual_payoff_scale", "250.0",
    "--forecast_mirror_counterfactual_loss_aversion", "0.25",
    "--forecast_prior_feasible_action_condition_capacity_on_evidence", "true",
    "--forecast_prior_feasible_action_evidence_power", "1.0",
    "--forecast_prior_feasible_action_window", "500",
    "--forecast_prior_feasible_action_min_samples", "50",
    "--forecast_prior_feasible_action_cvar_quantile", "0.90",
    "--forecast_prior_feasible_action_shortfall_budget", "0.05",
    "--forecast_prior_feasible_action_dual_lr", "0.02",
    "--forecast_prior_feasible_action_dual_max", "5.0",
    "--forecast_prior_feasible_action_reward_weight", "0.25",
    "--forecast_prior_feasible_action_payoff_scale", "250.0",
    "--forecast_prior_feasible_action_score_clip", "2.0",
    "--forecast_prior_vol_half_life_steps", "288",
    "--forecast_prior_vol_target", "0.15",
    "--forecast_prior_horizon_steps", "6",
    "--forecast_prior_denom_floor", "50.0",
    "--forecast_prior_payoff_target_mode", "auto",
    "--forecast_prior_target_alignment_version", "same_delivery_causal_v3",
]


def _replace_arg(args: list[str], flag: str, value: str) -> list[str]:
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(args):
        if args[i] == flag:
            out.extend([flag, value])
            i += 2
            replaced = True
        else:
            out.append(args[i])
            i += 1
    if not replaced:
        out.extend([flag, value])
    return out


def protocol_args_for_arm(arm: str) -> list[str]:
    """Return the shared protocol with only declared mechanism overrides."""
    args = list(COMMON_PROTOCOL_ARGS)
    return args


def prior_args_for_arm(arm: str) -> list[str]:
    args = list(BASE_V3_PRIOR_ARGS)
    if arm == "global_evidence":
        args = _replace_arg(args, "--forecast_prior_distributional_bucket_mode", "global")
    elif arm == "fixed_direction_cap":
        args = _replace_arg(args, "--forecast_prior_distributional_bucket_mode", "directional_fixed_cap")
        args = _replace_arg(args, "--forecast_prior_distributional_use_confidence_weight", "false")
        args = _replace_arg(args, "--forecast_prior_distributional_fixed_cap_abs", "0.10")
    elif arm == "no_confidence_weight":
        args = _replace_arg(args, "--forecast_prior_distributional_use_confidence_weight", "false")
    elif arm == "anchor_gated":
        # Deterministic conviction gate on the pure anchor: abstain when the two
        # forecast heads disagree. Parameter-free; no learned components.
        args = _replace_arg(args, "--forecast_prior_agreement_gate", "true")
    elif arm == "feasible_action_full":
        args = _replace_arg(args, "--forecast_prior_feasible_action_mappo", "true")
    elif arm == "forecast_observation_only":
        args = _replace_arg(args, "--forecast_prior_observation_only_mappo", "true")
    elif arm == "unconditioned_capacity":
        args = _replace_arg(args, "--forecast_prior_feasible_action_mappo", "true")
        args = _replace_arg(
            args,
            "--forecast_prior_feasible_action_condition_capacity_on_evidence",
            "false",
        )
    elif arm == "feasible_action_no_paired_reward":
        args = _replace_arg(args, "--forecast_prior_feasible_action_mappo", "true")
        args = _replace_arg(args, "--forecast_prior_feasible_action_reward_weight", "0.0")
        args = _replace_arg(args, "--forecast_prior_feasible_action_dual_max", "0.0")
    elif arm in {"cfm_zero_trust", "cfm_zero_trust_forecast_obs"}:
        args = _replace_arg(args, "--forecast_prior_mirror_mappo", "true")
        args = _replace_arg(
            args,
            "--forecast_prior_feasible_action_condition_capacity_on_evidence",
            "false",
        )
        args = _replace_arg(args, "--forecast_mirror_trust_max", "0.0")
        args = _replace_arg(
            args, "--forecast_mirror_counterfactual_learning", "false"
        )
        if arm == "cfm_zero_trust":
            args = _replace_arg(
                args, "--forecast_mirror_mask_forecast_observation", "true"
            )
    elif arm != "focal_anchor":
        raise ValueError(f"Unknown arm: {arm}")
    return args


ARM_SPECS = {
    "cfm_zero_trust": (
        "Risk-limited signed MAPPO with zero mirror exponent and masked forecast inputs"
    ),
    "cfm_zero_trust_forecast_obs": (
        "Risk-limited signed MAPPO with forecast observations but zero mirror exponent"
    ),
    "feasible_action_full": (
        "Evidence-conditioned feasible-action MAPPO with matured matched settlement utility"
    ),
    "forecast_observation_only": (
        "MAPPO with the identical 20D forecast observations but ordinary delta action and reward"
    ),
    "unconditioned_capacity": (
        "Feasible-action MAPPO with the former full tail-capacity amplification geometry"
    ),
    "focal_anchor": "Deterministic FoCAL anchor under the identical protocol",
    "anchor_gated": (
        "Deterministic FoCAL anchor with parameter-free conviction gate: abstain "
        "when the regression-sign and margin-sign forecast heads disagree"
    ),
    "feasible_action_no_paired_reward": (
        "Same feasible action geometry without paired advantage or shortfall penalty"
    ),
    "global_evidence": "FoCAL anchor with pooled rather than conditional evidence",
    "fixed_direction_cap": "Forecast direction with a fixed cap and disaster guard",
    "no_confidence_weight": "Conditional FoCAL anchor without confidence weighting",
}


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


def _run(cmd: list[str], label: str, log_path: Path, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = subprocess.list2cmdline(cmd)
    print("")
    print("=" * 100)
    print(f"[RUN] {label}")
    print(line)
    print(f"Log: {log_path}")
    print("=" * 100)
    if dry_run:
        return
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] START {label}\n{line}\n")
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for out_line in process.stdout:
            print(out_line, end="", flush=True)
            log.write(out_line)
            log.flush()
        rc = process.wait()
        log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] END {label} exit={rc}\n")
    if rc != 0:
        raise RuntimeError(f"{label} failed with exit code {rc}. See {log_path}")


def _settlement_train_args(args: argparse.Namespace) -> list[str]:
    return [
        "--mtm_settlement_price_mode", "external_series",
        "--mtm_external_settlement_price_data_path", _rel("settlement_price_dataset_real_v2/scenario_{episode:03d}.csv"),
        "--eval_mtm_external_settlement_price_data_path", _rel("evaluation_dataset_ffill/unseendata_settlement_real_v2.csv"),
        "--mtm_external_settlement_price_column", "settlement_price",
        "--mtm_external_settlement_timestamp_column", "timestamp",
        "--mtm_external_settlement_min_price_dkk_per_mwh", str(float(args.settlement_clip_min_dkk)),
        "--mtm_external_settlement_max_price_dkk_per_mwh", str(float(args.settlement_clip_max_dkk)),
    ]


def _settlement_eval_args(args: argparse.Namespace, region: str) -> list[str]:
    settlement_path = (
        _rel("evaluation_dataset_ffill/unseendata_v2_settlement_real_v2.csv")
        if region == "v2"
        else _rel("evaluation_dataset_ffill/unseendata_settlement_real_v2.csv")
    )
    return [
        "--mtm_settlement_price_mode", "external_series",
        "--mtm_external_settlement_price_data_path", settlement_path,
        "--mtm_external_settlement_price_column", "settlement_price",
        "--mtm_external_settlement_timestamp_column", "timestamp",
        "--mtm_external_settlement_min_price_dkk_per_mwh", str(float(args.settlement_clip_min_dkk)),
        "--mtm_external_settlement_max_price_dkk_per_mwh", str(float(args.settlement_clip_max_dkk)),
    ]


def _liquidity_train_args() -> list[str]:
    return [
        "--impact_volume_data", _rel("liquidity_volume_dataset_real_v1/scenario_{episode:03d}.csv"),
        "--eval_impact_volume_data", _rel("evaluation_dataset_ffill/unseendata_liquidity_volume_real_v1.csv"),
        "--impact_volume_column", "market_volume_mwh",
        "--impact_volume_unit", "mwh",
        "--impact_volume_timestamp_column", "timestamp",
        "--impact_volume_max_staleness_min", "90.0",
        "--impact_volume_price_floor_dkk_per_mwh", "50.0",
    ]


def _liquidity_eval_args(region: str) -> list[str]:
    volume_path = (
        _rel("evaluation_dataset_ffill/unseendata_v2_liquidity_volume_real_v1.csv")
        if region == "v2"
        else _rel("evaluation_dataset_ffill/unseendata_liquidity_volume_real_v1.csv")
    )
    return [
        "--impact_volume_data", volume_path,
        "--impact_volume_column", "market_volume_mwh",
        "--impact_volume_unit", "mwh",
        "--impact_volume_timestamp_column", "timestamp",
        "--impact_volume_max_staleness_min", "90.0",
        "--impact_volume_price_floor_dkk_per_mwh", "50.0",
    ]


def _check_shared_inputs() -> None:
    required = [
        PARENT_ROOT / "training_dataset_ffill",
        PARENT_ROOT / "rolling_past_history_dataset_ffill",
        PARENT_ROOT / "forecast_cache_settlement_hourly_v2",
        PARENT_ROOT / "evaluation_dataset_ffill" / "unseendata.csv",
        PARENT_ROOT / "evaluation_dataset_ffill" / "unseendata_v2.csv",
        PARENT_ROOT / "evaluation_dataset_ffill" / "unseendata_settlement_real_v2.csv",
        PARENT_ROOT / "evaluation_dataset_ffill" / "unseendata_v2_settlement_real_v2.csv",
        PARENT_ROOT / "settlement_price_dataset_real_v2",
        PARENT_ROOT / "liquidity_volume_dataset_real_v1" / "scenario_000.csv",
        PARENT_ROOT / "evaluation_dataset_ffill" / "unseendata_liquidity_volume_real_v1.csv",
        PARENT_ROOT / "evaluation_dataset_ffill" / "unseendata_v2_liquidity_volume_real_v1.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing shared Prototype inputs:\n" + "\n".join(missing))


def _seed_model_dir(suite_dir: Path, arm: str, seed: int) -> Path:
    return suite_dir / arm / f"seed{seed}" / f"tier1_forecast_utilization_seed{seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["focal_anchor"],
        help=(
            "Mechanism arms to run. The safe standalone default is the paper's "
            "deterministic focal_anchor; the campaign harnesses always pass an "
            "explicit arm list."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7],
        help=(
            "Training seeds. One seed is the correct default for deterministic "
            "arms; learned campaign arms receive the full seed list explicitly."
        ),
    )
    parser.add_argument(
        "--suite_root",
        default="Ablations/batch_tier_phase_runs/prototype5_mechanism_ablations_final_v1",
    )
    parser.add_argument(
        "--result_root",
        default="Ablations/results/prototype5_mechanism_ablations_final_v1",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--forecast_cache_dir",
        default="forecast_cache_settlement_hourly_v2",
        help=(
            "Forecast cache root for training and both evaluations. Any override "
            "must be metadata-compatible and use distinct suite/result roots so "
            "different forecast banks cannot be mixed."
        ),
    )
    parser.add_argument("--skip_training_if_complete", action="store_true")
    parser.add_argument(
        "--debug_logs",
        action="store_true",
        help=(
            "Save heavy training debug CSVs. Disabled by default; evaluation sleeve logs "
            "are always written for detailed metrics."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--settlement_clip_min_dkk", type=float, default=SETTLEMENT_CLIP_MIN_DKK_DEFAULT)
    parser.add_argument("--settlement_clip_max_dkk", type=float, default=SETTLEMENT_CLIP_MAX_DKK_DEFAULT)
    parser.add_argument(
        "--liquidity_participation_cap_fraction",
        type=float,
        default=None,
        help="Override the arm's market-volume participation cap in both training and evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    _force_utf8_stdio()
    args = parse_args()
    if (
        args.liquidity_participation_cap_fraction is not None
        and not 0.0 <= float(args.liquidity_participation_cap_fraction) <= 1.0
    ):
        raise ValueError("--liquidity_participation_cap_fraction must be between 0 and 1")
    assert_final_data_contract(PARENT_ROOT)
    unknown = [a for a in args.arms if a not in ARM_SPECS and a != "all"]
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}. Available: {sorted(ARM_SPECS)}")
    arms = list(ARM_SPECS) if "all" in args.arms else list(args.arms)
    if not args.dry_run:
        _check_shared_inputs()

    suite_root = PROJECT_ROOT / args.suite_root
    result_root = PROJECT_ROOT / args.result_root
    logs_dir = result_root / "logs"
    suite_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("Ablation workspace:", SCRIPT_ROOT)
    print("Shared Prototype5 root:", PROJECT_ROOT)
    print("Arms:", arms)
    print("Seeds:", args.seeds)

    for arm in arms:
        arm_suite = suite_root / arm
        arm_protocol_args = protocol_args_for_arm(arm)
        if args.liquidity_participation_cap_fraction is not None:
            arm_protocol_args = _replace_arg(
                arm_protocol_args,
                "--liquidity_participation_cap_fraction",
                str(float(args.liquidity_participation_cap_fraction)),
            )
        arm_prior_args = prior_args_for_arm(arm)
        for seed in args.seeds:
            model_dir = _seed_model_dir(suite_root, arm, int(seed))
            final_models = model_dir / "final_models"
            if args.skip_training_if_complete and final_models.is_dir():
                print(f"[SKIP] {arm} seed {seed}: final_models exists")
            else:
                train_cmd = [
                    args.python,
                    "run_tier_phase_multi_seed.py",
                    "--seeds", str(seed),
                    "--phase", "tier1_forecast_utilization_only",
                    "--suite_dir", str(arm_suite),
                    "--forecast_cache_dir", _rel(str(args.forecast_cache_dir)),
                    "--skip_eval",
                ]
                train_cmd += arm_protocol_args
                if args.debug_logs:
                    train_cmd += ["--enable_episode_csv_logs", "--debug", "--log-sleeve"]
                train_cmd += _settlement_train_args(args)
                train_cmd += _liquidity_train_args()
                train_cmd += arm_prior_args
                _run(
                    train_cmd,
                    f"{arm} train seed {seed}: {ARM_SPECS[arm]}",
                    logs_dir / f"{arm}_seed{seed}_train.log",
                    dry_run=bool(args.dry_run),
                )

            eval_common = [
                args.python,
                "evaluation.py",
                "--mode", "tiers",
                "--tiers_only", "tier1",
                "--tier1_dir", str(model_dir),
                "--seed", str(seed),
                "--forecast_cache_dir", _rel(str(args.forecast_cache_dir)),
                "--investment_freq", "6",
                "--meta_freq_min", "6",
                "--meta_freq_max", "6",
                "--global_norm_mode", "rolling_past",
                "--rolling_past_history_dir", _rel("rolling_past_history_dataset_ffill"),
                "--eval-distribution-rate", "0.0",
                "--mtm_return_model", "horizon_settlement",
                "--mtm_reference_price_dkk_per_mwh", "500",
                "--mtm_settlement_horizon_steps", "6",
                "--mtm_entry_price_mode", "current_price",
                "--mtm_horizon_payoff_denominator_mode", "mwh_volume",
                "--disable_mtm_return_cap",
                "--investor_notional_sizing_base", "initial_trading_sleeve",
                "--max_position_size", "0.1",
                "--capital_allocation_fraction", "0.6",
                "--mtm_loss_exit_threshold_pct", "0.15",
                "--friction_cost_multiplier", "1.0",
                *MARKET_FEE_ARGS,
                "--no_trade_threshold", "0.01",
                "--no_trade_threshold_reference", "executable_capacity",
                "--half_spread_bp", "5.0",
                "--impact_coef_bp", "20.0",
                "--impact_exponent", "0.5",
                "--impact_ref_notional", "volume",
                "--liquidity_participation_cap_fraction", "0.25",
                "--liquidity_volume_source", "impact_volume",
                "--liquidity_volume_multiplier", "1.0",
                "--liquidity_min_volume_mwh", "1.0",
                "--liquidity_tail_impact_threshold_dkk_per_mwh", "5000.0",
                "--liquidity_tail_impact_multiplier", "3.0",
                "--liquidity_tail_impact_power", "1.0",
                "--liquidity_tail_impact_max_multiplier", "10.0",
                "--enable_collateral_cash_drag",
                "--collateral_notional_margin_fraction", "0.02",
                "--collateral_stress_loss_fraction", "0.10",
                "--collateral_stress_price_dkk_per_mwh", "25000.0",
                "--collateral_funding_rate_annual", "0.05",
                "--collateral_tradeable_haircut", "1.0",
                "--distribution_rate", "0.0",
                "--meta_controller_rule_based",
                "--risk_controller_rule_based",
                "--enable_forecast_utilization",
                "--sleeve-sharpe-mode", "daily_hac_7",
                "--log-sleeve",
            ]
            eval_common = _replace_arg(
                eval_common,
                "--liquidity_participation_cap_fraction",
                str(float(args.liquidity_participation_cap_fraction))
                if args.liquidity_participation_cap_fraction is not None
                else "0.25",
            )
            eval_original_cmd = list(eval_common)
            eval_original_cmd += [
                "--output_dir", str(model_dir.parent / "evaluations_2025" / "tier1_forecast_utilization"),
                "--eval_data", _rel("evaluation_dataset_ffill/unseendata.csv"),
            ]
            eval_original_cmd += arm_prior_args
            eval_original_cmd += _settlement_eval_args(args, "original")
            eval_original_cmd += _liquidity_eval_args("original")
            _run(
                eval_original_cmd,
                f"{arm} original eval seed {seed}",
                logs_dir / f"{arm}_seed{seed}_eval_original.log",
                dry_run=bool(args.dry_run),
            )

            eval_v2_cmd = list(eval_common)
            eval_v2_cmd += [
                "--output_dir", str(model_dir.parent / "evaluations_2025_v2" / "tier1_forecast_utilization"),
                "--eval_data", _rel("evaluation_dataset_ffill/unseendata_v2.csv"),
            ]
            eval_v2_cmd += arm_prior_args
            eval_v2_cmd += _settlement_eval_args(args, "v2")
            eval_v2_cmd += _liquidity_eval_args("v2")
            _run(
                eval_v2_cmd,
                f"{arm} v2 eval seed {seed}",
                logs_dir / f"{arm}_seed{seed}_eval_v2.log",
                dry_run=bool(args.dry_run),
            )

    print("\nDone. Ablation suites under:", suite_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

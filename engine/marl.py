#!/usr/bin/env python3
"""Run MAPPO MARL 10-seed training and evaluate each seed on both price regions."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from protocol_preflight import assert_final_data_contract


PROJECT_ROOT = Path(__file__).resolve().parent
SEEDS = [7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202]

SUITE_DIR = "batch_tier_phase_runs/prototype5_mappo_marl_final_v1"
RESULT_ROOT = "results/prototype5_mappo_marl_final_v1"

# Harmonized EU balancing-energy technical price bounds (+/-15,000 EUR/MWh at
# ~7.45 DKK/EUR). Wider than every observed raw settlement print, so the bound
# acts as a technical guard rather than a distribution-shaping clip.
SETTLEMENT_CLIP_MIN_DKK_DEFAULT = -111750.0
SETTLEMENT_CLIP_MAX_DKK_DEFAULT = 111750.0


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


_force_utf8_stdio()


MARKET_FEE_ARGS = [
    "--market_fee_model", "nord_pool_intraday_2026",
    "--transaction_fee_dkk_per_mwh", "0.9238",
    "--annual_market_access_fee_dkk", "160175.0",
    "--market_access_fee_allocation_fraction", "1.0",
    "--market_fee_source_id", "nord_pool_nordic_baltic_2026_standard_participant",
]


COMMON_PROTOCOL_ARGS = [
    "--episode_data_dir", "training_dataset_ffill",
    "--global_norm_mode", "rolling_past",
    "--rolling_past_history_dir", "rolling_past_history_dataset_ffill",
    "--eval_data", "evaluation_dataset_ffill/unseendata.csv",
    "--lr", "0.0003",
    "--ent_coef", "0.03",
    "--algo", "mappo",
    "--forecast_cache_dir", "forecast_cache_settlement_hourly_v2",
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
    # Energinet's balancing-product minimum bid is 1 MW. For this one-hour
    # contract, 1 MWh is a numerical/market-lot floor that does not override
    # the observed activation-volume series in most hours.
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


EVAL_PROTOCOL_ARGS = [
    "--investment_freq", "6",
    "--meta_freq_min", "6",
    "--meta_freq_max", "6",
    "--global_norm_mode", "rolling_past",
    "--rolling_past_history_dir", "rolling_past_history_dataset_ffill",
    "--forecast_cache_dir", "forecast_cache_settlement_hourly_v2",
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
    "--log-sleeve",
    "--sleeve-sharpe-mode", "daily_hac_7",
]


def _settlement_train_args(args: argparse.Namespace) -> list[str]:
    if bool(getattr(args, "energy_index_settlement", False)):
        return []
    if bool(getattr(args, "basis_risk", False)):
        return [
            "--mtm_settlement_price_mode", "cross_zone_basis",
            "--mtm_basis_price_data_path", str(args.basis_training_template),
            "--eval_mtm_basis_price_data_path", str(args.basis_eval_original),
            "--mtm_basis_scale", str(float(args.basis_scale)),
            "--mtm_basis_centering_mode", str(args.basis_centering_mode),
            "--mtm_basis_centering_window_steps", str(int(args.basis_centering_window_steps)),
        ]
    return [
        "--mtm_settlement_price_mode", "external_series",
        "--mtm_external_settlement_price_data_path", str(args.settlement_training_template),
        "--eval_mtm_external_settlement_price_data_path", str(args.settlement_eval_original),
        "--mtm_external_settlement_price_column", str(args.settlement_price_column),
        "--mtm_external_settlement_timestamp_column", str(args.settlement_timestamp_column),
        "--mtm_external_settlement_min_price_dkk_per_mwh", str(float(args.settlement_clip_min_dkk)),
        "--mtm_external_settlement_max_price_dkk_per_mwh", str(float(args.settlement_clip_max_dkk)),
    ]


def _settlement_eval_args(args: argparse.Namespace, region: str) -> list[str]:
    if bool(getattr(args, "energy_index_settlement", False)):
        return []
    if bool(getattr(args, "basis_risk", False)):
        basis_path = args.basis_eval_v2 if region == "v2" else args.basis_eval_original
        return [
            "--mtm_settlement_price_mode", "cross_zone_basis",
            "--mtm_basis_price_data_path", str(basis_path),
            "--mtm_basis_scale", str(float(args.basis_scale)),
            "--mtm_basis_centering_mode", str(args.basis_centering_mode),
            "--mtm_basis_centering_window_steps", str(int(args.basis_centering_window_steps)),
        ]
    settlement_path = args.settlement_eval_v2 if region == "v2" else args.settlement_eval_original
    return [
        "--mtm_settlement_price_mode", "external_series",
        "--mtm_external_settlement_price_data_path", str(settlement_path),
        "--mtm_external_settlement_price_column", str(args.settlement_price_column),
        "--mtm_external_settlement_timestamp_column", str(args.settlement_timestamp_column),
        "--mtm_external_settlement_min_price_dkk_per_mwh", str(float(args.settlement_clip_min_dkk)),
        "--mtm_external_settlement_max_price_dkk_per_mwh", str(float(args.settlement_clip_max_dkk)),
    ]


def _liquidity_train_args(args: argparse.Namespace) -> list[str]:
    if bool(getattr(args, "load_proxy_liquidity", False)):
        return ["--impact_ref_notional", "sleeve", "--liquidity_volume_source", "load"]
    return [
        "--impact_volume_data", str(args.liquidity_volume_training_template),
        "--eval_impact_volume_data", str(args.liquidity_volume_eval_original),
        "--impact_volume_column", str(args.liquidity_volume_column),
        "--impact_volume_unit", str(args.liquidity_volume_unit),
        "--impact_volume_timestamp_column", str(args.liquidity_volume_timestamp_column),
        "--impact_volume_max_staleness_min", str(float(args.liquidity_volume_max_staleness_min)),
        "--impact_volume_price_floor_dkk_per_mwh", str(float(args.impact_volume_price_floor_dkk_per_mwh)),
    ]


def _liquidity_eval_args(args: argparse.Namespace, region: str) -> list[str]:
    if bool(getattr(args, "load_proxy_liquidity", False)):
        return ["--impact_ref_notional", "sleeve", "--liquidity_volume_source", "load"]
    volume_path = args.liquidity_volume_eval_v2 if region == "v2" else args.liquidity_volume_eval_original
    return [
        "--impact_volume_data", str(volume_path),
        "--impact_volume_column", str(args.liquidity_volume_column),
        "--impact_volume_unit", str(args.liquidity_volume_unit),
        "--impact_volume_timestamp_column", str(args.liquidity_volume_timestamp_column),
        "--impact_volume_max_staleness_min", str(float(args.liquidity_volume_max_staleness_min)),
        "--impact_volume_price_floor_dkk_per_mwh", str(float(args.impact_volume_price_floor_dkk_per_mwh)),
    ]


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


def _assert_settlement_clip_consistency(args: argparse.Namespace) -> None:
    """Refuse to run when settlement files were built with a tighter clip than configured."""
    if bool(getattr(args, "energy_index_settlement", False)) or bool(getattr(args, "basis_risk", False)):
        return
    import pandas as pd

    clip_min = float(args.settlement_clip_min_dkk)
    clip_max = float(args.settlement_clip_max_dkk)
    candidates = [
        PROJECT_ROOT / str(args.settlement_eval_original),
        PROJECT_ROOT / str(args.settlement_eval_v2),
        PROJECT_ROOT / str(args.settlement_training_template).replace("{episode:03d}", "000"),
    ]
    stale = []
    for path in candidates:
        if not path.is_file():
            continue
        prices = pd.to_numeric(
            pd.read_csv(path, usecols=[str(args.settlement_price_column)])[str(args.settlement_price_column)],
            errors="coerce",
        ).dropna()
        if prices.empty:
            continue
        lo, hi = float(prices.min()), float(prices.max())
        at_legacy_bounds = abs(lo + 1000.0) < 1e-6 or abs(hi - 10000.0) < 1e-6
        if at_legacy_bounds and (clip_min < -1000.0 - 1e-6 or clip_max > 10000.0 + 1e-6):
            stale.append(f"  {path} (min={lo:.1f}, max={hi:.1f})")
    if stale:
        raise RuntimeError(
            "Settlement files below still carry the legacy [-1000, 10000] DKK clip but the "
            f"protocol now uses [{clip_min:.0f}, {clip_max:.0f}]. Rebuild them first:\n"
            + "\n".join(stale)
            + "\n\n  python scripts/build_real_settlement_protocol_v2.py "
            + f"--settlement_clip_min {clip_min:.0f} --settlement_clip_max {clip_max:.0f} --overwrite\n"
            "Then retrain the forecast models and rebuild the hourly forecast cache "
            "(scripts/train_hourly_forecast_models_v2.py, scripts/precompute_hourly_forecast_cache_v2.py) "
            "because the forecast targets share the settlement clip."
        )


def _assert_liquidity_volume_files(args: argparse.Namespace) -> None:
    if bool(getattr(args, "load_proxy_liquidity", False)):
        return
    candidates = [
        PROJECT_ROOT / str(args.liquidity_volume_eval_original),
        PROJECT_ROOT / str(args.liquidity_volume_eval_v2),
        PROJECT_ROOT / str(args.liquidity_volume_training_template).replace("{episode:03d}", "000"),
    ]
    missing = [str(path) for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Real liquidity-volume files are missing:\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\n\nBuild them first with:\n"
            "  python scripts/build_real_liquidity_volume_protocol_v1.py --overwrite"
        )


def _seed_model_dir(suite_dir: Path, seed: int) -> Path:
    return suite_dir / f"seed{seed}" / f"tier1_seed{seed}"


def _replace_arg(args: list[str], flag: str, value: str) -> list[str]:
    out = list(args)
    try:
        index = out.index(flag)
    except ValueError:
        out.extend([flag, value])
    else:
        out[index + 1] = value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--suite_dir", default=SUITE_DIR)
    parser.add_argument("--result_root", default=RESULT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip_training_if_complete", action="store_true")
    parser.add_argument(
        "--energy_index_settlement",
        action="store_true",
        help="Disable external real-settlement files and settle against the base energy index.",
    )
    parser.add_argument(
        "--settlement_training_template",
        default="settlement_price_dataset_real_v2/scenario_{episode:03d}.csv",
        help="Episode-aware real settlement file template for training.",
    )
    parser.add_argument(
        "--settlement_eval_original",
        default="evaluation_dataset_ffill/unseendata_settlement_real_v2.csv",
        help="Real settlement file for evaluation_dataset_ffill/unseendata.csv.",
    )
    parser.add_argument(
        "--settlement_eval_v2",
        default="evaluation_dataset_ffill/unseendata_v2_settlement_real_v2.csv",
        help="Real settlement file for evaluation_dataset_ffill/unseendata_v2.csv.",
    )
    parser.add_argument("--settlement_price_column", default="settlement_price")
    parser.add_argument("--settlement_timestamp_column", default="timestamp")
    parser.add_argument(
        "--settlement_clip_min_dkk",
        type=float,
        default=SETTLEMENT_CLIP_MIN_DKK_DEFAULT,
        help="Lower technical bound for the external settlement series (DKK/MWh).",
    )
    parser.add_argument(
        "--settlement_clip_max_dkk",
        type=float,
        default=SETTLEMENT_CLIP_MAX_DKK_DEFAULT,
        help="Upper technical bound for the external settlement series (DKK/MWh).",
    )
    parser.add_argument(
        "--basis_risk",
        action="store_true",
        help="Use cross-zone basis settlement instead of the default external real-settlement protocol.",
    )
    parser.add_argument(
        "--basis_training_template",
        default="basis_price_dataset_dk2_ffill/scenario_{episode:03d}.csv",
        help="Episode-aware peer price file template for training.",
    )
    parser.add_argument(
        "--basis_eval_original",
        default="evaluation_dataset_ffill/unseendata_v2.csv",
        help="Peer price file used when evaluating the original DK1 evaluation dataset.",
    )
    parser.add_argument(
        "--basis_eval_v2",
        default="evaluation_dataset_ffill/unseendata.csv",
        help="Peer price file used when evaluating the DK2 evaluation dataset.",
    )
    parser.add_argument("--basis_scale", type=float, default=0.50)
    parser.add_argument(
        "--basis_centering_mode",
        choices=["none", "rolling_median", "expanding_median"],
        default="rolling_median",
    )
    parser.add_argument("--basis_centering_window_steps", type=int, default=4320)
    parser.add_argument("--load_proxy_liquidity", action="store_true")
    parser.add_argument(
        "--liquidity_volume_training_template",
        default="liquidity_volume_dataset_real_v1/scenario_{episode:03d}.csv",
    )
    parser.add_argument(
        "--liquidity_volume_eval_original",
        default="evaluation_dataset_ffill/unseendata_liquidity_volume_real_v1.csv",
    )
    parser.add_argument(
        "--liquidity_volume_eval_v2",
        default="evaluation_dataset_ffill/unseendata_v2_liquidity_volume_real_v1.csv",
    )
    parser.add_argument("--liquidity_volume_column", default="market_volume_mwh")
    parser.add_argument("--liquidity_volume_unit", choices=["mwh", "dkk"], default="mwh")
    parser.add_argument("--liquidity_volume_timestamp_column", default="timestamp")
    parser.add_argument("--liquidity_volume_max_staleness_min", type=float, default=90.0)
    parser.add_argument("--impact_volume_price_floor_dkk_per_mwh", type=float, default=50.0)
    parser.add_argument(
        "--liquidity_participation_cap_fraction",
        type=float,
        default=0.25,
        help="Maximum share of observed market volume executable per decision (0 disables the cap).",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= float(args.liquidity_participation_cap_fraction) <= 1.0:
        raise ValueError("--liquidity_participation_cap_fraction must be between 0 and 1")
    assert_final_data_contract(PROJECT_ROOT)
    suite_dir = PROJECT_ROOT / args.suite_dir
    result_root = PROJECT_ROOT / args.result_root
    logs_dir = result_root / "logs"
    if not args.dry_run:
        _assert_settlement_clip_consistency(args)
        _assert_liquidity_volume_files(args)

    cap_value = str(float(args.liquidity_participation_cap_fraction))
    train_protocol_args = _replace_arg(
        COMMON_PROTOCOL_ARGS,
        "--liquidity_participation_cap_fraction",
        cap_value,
    )
    eval_protocol_args = _replace_arg(
        EVAL_PROTOCOL_ARGS,
        "--liquidity_participation_cap_fraction",
        cap_value,
    )

    for seed in args.seeds:
        model_dir = _seed_model_dir(suite_dir, int(seed))
        final_models = model_dir / "final_models"
        if args.skip_training_if_complete and final_models.is_dir():
            print(f"[SKIP] seed {seed} training already has final_models: {final_models}")
        else:
            train_cmd = [
                args.python,
                "run_tier_phase_multi_seed.py",
                "--seeds", str(seed),
                "--phase", "tier1_only",
                "--suite_dir", args.suite_dir,
                "--skip_eval",
            ] + train_protocol_args + _settlement_train_args(args) + _liquidity_train_args(args)
            _run(train_cmd, f"MAPPO MARL train seed {seed}", logs_dir / f"marl_seed{seed}_train.log", args.dry_run)

        if not args.dry_run and not final_models.is_dir():
            raise FileNotFoundError(f"Missing trained final_models for seed {seed}: {final_models}")

        eval_original_out = suite_dir / f"seed{seed}" / "evaluations_2025" / "tier1"
        eval_original_cmd = [
            args.python,
            "evaluation.py",
            "--mode", "tiers",
            "--tiers_only", "tier1",
            "--tier1_dir", str(model_dir),
            "--seed", str(seed),
            "--output_dir", str(eval_original_out),
            "--eval_data", "evaluation_dataset_ffill/unseendata.csv",
        ] + eval_protocol_args + _settlement_eval_args(args, "original") + _liquidity_eval_args(args, "original")
        _run(
            eval_original_cmd,
            f"MAPPO MARL original eval seed {seed}",
            logs_dir / f"marl_seed{seed}_eval_original.log",
            args.dry_run,
        )

        eval_v2_out = suite_dir / f"seed{seed}" / "evaluations_2025_v2" / "tier1"
        eval_v2_cmd = [
            args.python,
            "evaluation.py",
            "--mode", "tiers",
            "--tiers_only", "tier1",
            "--tier1_dir", str(model_dir),
            "--seed", str(seed),
            "--output_dir", str(eval_v2_out),
            "--eval_data", "evaluation_dataset_ffill/unseendata_v2.csv",
        ] + eval_protocol_args + _settlement_eval_args(args, "v2") + _liquidity_eval_args(args, "v2")
        _run(eval_v2_cmd, f"MAPPO MARL v2 eval seed {seed}", logs_dir / f"marl_seed{seed}_eval_v2.log", args.dry_run)

    print("\nDone. MARL suite:", suite_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

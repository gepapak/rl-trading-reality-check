#!/usr/bin/env python3
"""
Run Tier-1 baseline training and evaluation across multiple seeds.

For each seed: train via main.py, then evaluate via evaluation.py --mode tiers.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from runtime_contract import (
    build_runtime_contract,
    controller_contract_settings,
    engine_file_hashes,
    execution_contract_settings,
    forecast_prior_contract_settings,
    mtm_contract_settings,
    runtime_contract_hash,
    sizing_contract_settings,
)
from forecast_prior_cli import (
    add_forecast_prior_override_args,
    collect_forecast_prior_overrides,
    forecast_prior_override_cli_args,
)


DEFAULT_ROLLING_PAST_HISTORY_DIR = "rolling_past_history_dataset"


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


def _effective_rolling_past_history_dir(args) -> str:
    explicit = str(getattr(args, "rolling_past_history_dir", "") or "").strip()
    if str(getattr(args, "global_norm_mode", "")).strip().lower() == "rolling_past":
        return explicit or DEFAULT_ROLLING_PAST_HISTORY_DIR
    return explicit


VARIANT_TRAIN_SPECS = {
    "tier1": {
        "name": "Tier 1 hybrid RL baseline",
        "slug": "tier1",
        "extra": [],
        "eval_extra": [],
        "reuse_train_from": None,
    },
    "tier1_forecast_utilization": {
        "name": "Tier 1 + evidence-calibrated ANN forecast-cache utilization",
        "slug": "tier1_forecast_utilization",
        "extra": ["--enable_forecast_utilization"],
        "eval_extra": ["--enable_forecast_utilization"],
        "reuse_train_from": None,
    },
}

PHASE_VARIANTS = {
    "tier1_only": ["tier1"],
    "tier1_forecast_utilization_only": ["tier1_forecast_utilization"],
    "tier1_forecast_utilization_pair": ["tier1", "tier1_forecast_utilization"],
}

EVAL_DIR_ARG_BY_VARIANT = {
    "tier1": "--tier1_dir",
    "tier1_forecast_utilization": "--tier1_dir",
}

EVAL_TIERS_ONLY_BY_VARIANT = {
    "tier1": "tier1",
    "tier1_forecast_utilization": "tier1",
}

# Variants that require their own evaluation forecast-cache directory.
# Empty string ⇒ use the value of args.forecast_cache_dir (default).
EVAL_FORECAST_CACHE_DIR_OVERRIDE = {
    "tier1": "",
    "tier1_forecast_utilization": "",
}


def format_cmd(cmd):
    return subprocess.list2cmdline(cmd)


def _run_timeout_seconds() -> int:
    try:
        hours = float(os.environ.get("TIER_RUN_TIMEOUT_HOURS", "12"))
        return max(60, int(hours * 3600))
    except Exception:
        return 12 * 3600


def run_command(cmd, name: str) -> dict:
    started_at = datetime.now()
    start = time.time()
    print(f"\n{'='*100}")
    print(f"Starting: {name}")
    print(f"Command: {format_cmd(cmd)}")
    print(f"Started at: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")

    try:
        timeout_s = _run_timeout_seconds()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_subprocess_env(),
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        timed_out = False
        deadline = time.time() + timeout_s
        for line in process.stdout:
            print(line, end="", flush=True)
            if time.time() > deadline:
                process.kill()
                timed_out = True
                print(f"\n[TIMEOUT] {name} exceeded {timeout_s}s — process killed.")
                break
        process.wait()
        elapsed = time.time() - start
        ok = process.returncode == 0 and not timed_out
        finished_at = datetime.now()
        print(f"\n{'='*100}")
        print(f"{'SUCCESS' if ok else 'FAILED'}: {name} (exit={process.returncode})")
        print(f"Duration: {int(elapsed//3600)}h {int((elapsed%3600)//60)}m {int(elapsed%60)}s")
        print(f"{'='*100}\n")
        return {
            "success": ok,
            "returncode": int(process.returncode),
            "duration_seconds": elapsed,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"\nError: {e}")
        return {
            "success": False,
            "returncode": -1,
            "duration_seconds": elapsed,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now().isoformat(),
        }


def resolve_eval_steps(eval_data_path: str, eval_steps: int = None) -> int:
    if eval_steps is not None and eval_steps > 0:
        return int(eval_steps)
    try:
        import pandas as pd
        df = pd.read_csv(eval_data_path, usecols=[0])
        return max(1, len(df) - 1)
    except Exception:
        return 1000


def find_latest_file(directory: str, pattern: str) -> str:
    import glob
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def extract_eval_metrics(json_path: str, canonical_variant: str = None) -> dict:
    out = {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return out

        source = data
        if canonical_variant and isinstance(data.get("tiers"), dict):
            tier_entry = data.get("tiers", {}).get(canonical_variant, {})
            if not tier_entry and str(canonical_variant).startswith("tier1"):
                tier_entry = data.get("tiers", {}).get("tier1", {})
            if isinstance(tier_entry, dict):
                source = tier_entry

        if isinstance(source, dict):
            for k, v in source.items():
                if isinstance(v, (int, float, str, bool)):
                    out[k] = v
            sleeve = source.get("sleeve_metrics", {})
            if isinstance(sleeve, dict):
                for k, v in sleeve.items():
                    if isinstance(v, (int, float, str, bool)):
                        out[k] = v

        for k in ("tier_report_scope", "tier_report_csv", "tier_report_md"):
            v = data.get(k)
            if isinstance(v, (int, float, str, bool)):
                out[k] = v
        return out
    except Exception:
        return out


def write_seed_summary_csv(rows: list, csv_path: str):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_run_plan_csv(rows: list, csv_path: str):
    if not rows:
        return
    fieldnames = [
        "global_run_number",
        "total_planned_runs",
        "seed",
        "seed_run_order",
        "run_name",
        "canonical_variant",
        "save_dir",
        "eval_output_dir",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_existing_summary_csv(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        return []
    rows = []
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(dict(row))
    except Exception:
        return []
    return rows


def upsert_summary_row(rows: list, row: dict) -> list:
    key = str(row.get("global_run_number", ""))
    if not key:
        rows.append(row)
        return rows
    for i, existing in enumerate(rows):
        if str(existing.get("global_run_number", "")) == key:
            merged = dict(existing)
            merged.update(row)
            rows[i] = merged
            return rows
    rows.append(row)
    return rows


def as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y")


def _parse_seeds(seed_tokens):
    seeds = []
    for token in seed_tokens:
        for part in str(token).replace(",", " ").split():
            try:
                seeds.append(int(part))
            except ValueError:
                pass
    seen = set()
    return [s for s in seeds if not (s in seen or seen.add(s))]


def _build_suite_dir(output_root: str, phase: str, run_tag: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"tiers_{phase}_{stamp}"
    if run_tag:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(run_tag).strip())
        if safe:
            base = f"{base}_{safe}"
    path = os.path.join(output_root, base)
    os.makedirs(path, exist_ok=True)
    return path


def write_phase_protocol_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_existing_protocol(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed Tier-1 training and evaluation.")
    parser.add_argument("--seeds", nargs="+", required=True, help="Seeds, e.g. --seeds 7 42 123 789 2025")
    parser.add_argument("--phase", type=str, default="tier1_only",
                        choices=list(PHASE_VARIANTS.keys()),
                        help="Phase to run.")
    parser.add_argument("--episode_data_dir", type=str, default="training_dataset")
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--end_episode", type=int, default=19)
    parser.add_argument("--global_norm_mode", type=str, default="rolling_past", choices=["rolling_past", "global"])
    parser.add_argument("--rolling_past_history_dir", type=str, default="")
    parser.add_argument("--investment_freq", type=int, default=6)
    parser.add_argument("--meta_freq_min", type=int, default=6)
    parser.add_argument("--meta_freq_max", type=int, default=6)
    parser.add_argument("--cooling_period", type=int, default=0)
    parser.add_argument("--forecast_cache_dir", type=str, default="forecast_cache")
    parser.add_argument(
        "--mtm_return_model",
        "--mtm-return-model",
        dest="mtm_return_model",
        type=str,
        default=None,
        choices=[
            "percent",
            "percent_capped",
            "notional_price_diff",
            "horizon_settlement",
            "horizon_settlement_continuous",
        ],
        help=(
            "Financial MTM payoff model. Use horizon_settlement for strict "
            "fixed-horizon settlement against a causal entry benchmark."
        ),
    )
    parser.add_argument(
        "--mtm_reference_price_dkk_per_mwh",
        "--mtm-reference-price-dkk-per-mwh",
        dest="mtm_reference_price_dkk_per_mwh",
        type=float,
        default=None,
        help="Reference price used by notional_price_diff MTM.",
    )
    parser.add_argument("--mtm_settlement_horizon_steps", "--mtm-settlement-horizon-steps", dest="mtm_settlement_horizon_steps", type=int, default=None)
    parser.add_argument(
        "--mtm_entry_price_mode",
        "--mtm-entry-price-mode",
        dest="mtm_entry_price_mode",
        type=str,
        default=None,
        choices=["same_hour_prev_day", "rolling_same_hour_median", "current_price"],
    )
    parser.add_argument(
        "--mtm_horizon_payoff_denominator_mode",
        "--mtm-horizon-payoff-denominator-mode",
        dest="mtm_horizon_payoff_denominator_mode",
        type=str,
        default=None,
        choices=["reference_price", "entry_price_floor", "mwh_volume", "mwh"],
        help=(
            "Denominator for horizon_settlement payoff. reference_price preserves "
            "price_diff/reference; entry_price_floor uses price_diff/max(abs(entry_price), reference); "
            "mwh_volume stores explicit MWh volume and pays volume*(settlement-entry)."
        ),
    )
    parser.add_argument(
        "--mtm_settlement_price_mode",
        "--mtm-settlement-price-mode",
        dest="mtm_settlement_price_mode",
        type=str,
        default=None,
        choices=[
            "energy_index",
            "none",
            "base",
            "cross_zone_basis",
            "basis_adjusted",
            "external_series",
            "external",
            "realized",
            "real_settlement",
        ],
        help="Settlement price process for horizon_settlement MTM.",
    )
    parser.add_argument("--mtm_basis_price_data_path", "--mtm-basis-price-data-path", dest="mtm_basis_price_data_path", type=str, default=None)
    parser.add_argument(
        "--eval_mtm_basis_price_data_path",
        "--eval-mtm-basis-price-data-path",
        dest="eval_mtm_basis_price_data_path",
        type=str,
        default=None,
        help=(
            "Evaluation-only basis price CSV/path. Use this when training uses an episode "
            "template such as training_dataset_peer_ffill/scenario_{episode:03d}.csv."
        ),
    )
    parser.add_argument("--mtm_basis_price_column", "--mtm-basis-price-column", dest="mtm_basis_price_column", type=str, default=None)
    parser.add_argument("--mtm_basis_timestamp_column", "--mtm-basis-timestamp-column", dest="mtm_basis_timestamp_column", type=str, default=None)
    parser.add_argument("--mtm_basis_scale", "--mtm-basis-scale", dest="mtm_basis_scale", type=float, default=None)
    parser.add_argument(
        "--mtm_basis_centering_mode",
        "--mtm-basis-centering-mode",
        dest="mtm_basis_centering_mode",
        type=str,
        default=None,
        choices=["none", "rolling_median", "expanding_median"],
    )
    parser.add_argument("--mtm_basis_centering_window_steps", "--mtm-basis-centering-window-steps", dest="mtm_basis_centering_window_steps", type=int, default=None)
    parser.add_argument("--mtm_external_settlement_price_data_path", "--mtm-external-settlement-price-data-path", dest="mtm_external_settlement_price_data_path", type=str, default=None)
    parser.add_argument(
        "--eval_mtm_external_settlement_price_data_path",
        "--eval-mtm-external-settlement-price-data-path",
        dest="eval_mtm_external_settlement_price_data_path",
        type=str,
        default=None,
        help="Evaluation-only external settlement CSV/path for external_series mode.",
    )
    parser.add_argument("--mtm_external_settlement_price_column", "--mtm-external-settlement-price-column", dest="mtm_external_settlement_price_column", type=str, default=None)
    parser.add_argument("--mtm_external_settlement_timestamp_column", "--mtm-external-settlement-timestamp-column", dest="mtm_external_settlement_timestamp_column", type=str, default=None)
    parser.add_argument("--mtm_external_settlement_min_price_dkk_per_mwh", "--mtm-external-settlement-min-price-dkk-per-mwh", dest="mtm_external_settlement_min_price_dkk_per_mwh", type=float, default=None)
    parser.add_argument("--mtm_external_settlement_max_price_dkk_per_mwh", "--mtm-external-settlement-max-price-dkk-per-mwh", dest="mtm_external_settlement_max_price_dkk_per_mwh", type=float, default=None)
    parser.add_argument(
        "--disable_mtm_return_cap",
        "--disable-mtm-return-cap",
        dest="disable_mtm_return_cap",
        action="store_true",
        help="Disable MTM return clipping.",
    )
    parser.add_argument("--mtm_price_return_cap_min", "--mtm-price-return-cap-min", dest="mtm_price_return_cap_min", type=float, default=None)
    parser.add_argument("--mtm_price_return_cap_max", "--mtm-price-return-cap-max", dest="mtm_price_return_cap_max", type=float, default=None)
    parser.add_argument(
        "--investor_notional_sizing_base",
        "--investor-notional-sizing-base",
        dest="investor_notional_sizing_base",
        type=str,
        default=None,
        choices=["initial_trading_sleeve", "live_trading_cash", "initial_fund_nav"],
        help="Base used to convert normalized investor exposure into DKK notional.",
    )
    parser.add_argument("--max_position_size", "--max-position-size", dest="max_position_size", type=float, default=None)
    parser.add_argument("--capital_allocation_fraction", "--capital-allocation-fraction", dest="capital_allocation_fraction", type=float, default=None)
    parser.add_argument("--mtm_loss_exit_threshold_pct", "--mtm-loss-exit-threshold-pct", dest="mtm_loss_exit_threshold_pct", type=float, default=None)
    parser.add_argument("--friction_cost_multiplier", "--friction-cost-multiplier", dest="friction_cost_multiplier", type=float, default=None)
    parser.add_argument(
        "--market_fee_model",
        "--market-fee-model",
        dest="market_fee_model",
        choices=["legacy_notional_fixed", "nord_pool_intraday_2026"],
        default=None,
    )
    parser.add_argument(
        "--transaction_fee_dkk_per_mwh",
        "--transaction-fee-dkk-per-mwh",
        dest="transaction_fee_dkk_per_mwh",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--annual_market_access_fee_dkk",
        "--annual-market-access-fee-dkk",
        dest="annual_market_access_fee_dkk",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--market_access_fee_allocation_fraction",
        "--market-access-fee-allocation-fraction",
        dest="market_access_fee_allocation_fraction",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--market_fee_source_id",
        "--market-fee-source-id",
        dest="market_fee_source_id",
        type=str,
        default=None,
    )
    parser.add_argument("--no_trade_threshold", "--no-trade-threshold", dest="no_trade_threshold", type=float, default=None)
    parser.add_argument(
        "--no_trade_threshold_reference",
        "--no-trade-threshold-reference",
        dest="no_trade_threshold_reference",
        choices=["executable_capacity", "max_position"],
        default=None,
    )
    parser.add_argument("--half_spread_bp", "--half-spread-bp", dest="half_spread_bp", type=float, default=None)
    parser.add_argument("--impact_coef_bp", "--impact-coef-bp", dest="impact_coef_bp", type=float, default=None)
    parser.add_argument("--impact_exponent", "--impact-exponent", dest="impact_exponent", type=float, default=None)
    parser.add_argument("--impact_ref_notional", "--impact-ref-notional", dest="impact_ref_notional", type=str, default=None)
    parser.add_argument("--impact_volume_data", "--impact-volume-data", dest="impact_volume_data", type=str, default=None)
    parser.add_argument(
        "--eval_impact_volume_data",
        "--eval-impact-volume-data",
        dest="eval_impact_volume_data",
        type=str,
        default=None,
        help="Evaluation-only liquidity/impact volume CSV/path for volume-based impact/liquidity.",
    )
    parser.add_argument("--impact_volume_column", "--impact-volume-column", dest="impact_volume_column", type=str, default=None)
    parser.add_argument("--impact_volume_unit", "--impact-volume-unit", dest="impact_volume_unit", choices=["mwh", "dkk"], default=None)
    parser.add_argument("--impact_volume_timestamp_column", "--impact-volume-timestamp-column", dest="impact_volume_timestamp_column", type=str, default=None)
    parser.add_argument("--impact_volume_max_staleness_min", "--impact-volume-max-staleness-min", dest="impact_volume_max_staleness_min", type=float, default=None)
    parser.add_argument("--impact_volume_price_floor_dkk_per_mwh", "--impact-volume-price-floor-dkk-per-mwh", dest="impact_volume_price_floor_dkk_per_mwh", type=float, default=None)
    parser.add_argument("--liquidity_participation_cap_fraction", "--liquidity-participation-cap-fraction", dest="liquidity_participation_cap_fraction", type=float, default=None)
    parser.add_argument("--liquidity_volume_source", "--liquidity-volume-source", dest="liquidity_volume_source", choices=["load", "generation", "max_load_generation", "impact_volume"], default=None)
    parser.add_argument("--liquidity_volume_multiplier", "--liquidity-volume-multiplier", dest="liquidity_volume_multiplier", type=float, default=None)
    parser.add_argument("--liquidity_min_volume_mwh", "--liquidity-min-volume-mwh", dest="liquidity_min_volume_mwh", type=float, default=None)
    parser.add_argument("--liquidity_tail_impact_threshold_dkk_per_mwh", "--liquidity-tail-impact-threshold-dkk-per-mwh", dest="liquidity_tail_impact_threshold_dkk_per_mwh", type=float, default=None)
    parser.add_argument("--liquidity_tail_impact_multiplier", "--liquidity-tail-impact-multiplier", dest="liquidity_tail_impact_multiplier", type=float, default=None)
    parser.add_argument("--liquidity_tail_impact_power", "--liquidity-tail-impact-power", dest="liquidity_tail_impact_power", type=float, default=None)
    parser.add_argument("--liquidity_tail_impact_max_multiplier", "--liquidity-tail-impact-max-multiplier", dest="liquidity_tail_impact_max_multiplier", type=float, default=None)
    parser.add_argument("--enable_collateral_cash_drag", "--enable-collateral-cash-drag", dest="enable_collateral_cash_drag", action="store_true", default=None)
    parser.add_argument("--disable_collateral_cash_drag", "--disable-collateral-cash-drag", dest="disable_collateral_cash_drag", action="store_true", default=False)
    parser.add_argument("--collateral_notional_margin_fraction", "--collateral-notional-margin-fraction", dest="collateral_notional_margin_fraction", type=float, default=None)
    parser.add_argument("--collateral_stress_loss_fraction", "--collateral-stress-loss-fraction", dest="collateral_stress_loss_fraction", type=float, default=None)
    parser.add_argument("--collateral_stress_price_dkk_per_mwh", "--collateral-stress-price-dkk-per-mwh", dest="collateral_stress_price_dkk_per_mwh", type=float, default=None)
    parser.add_argument("--collateral_funding_rate_annual", "--collateral-funding-rate-annual", dest="collateral_funding_rate_annual", type=float, default=None)
    parser.add_argument("--collateral_tradeable_haircut", "--collateral-tradeable-haircut", dest="collateral_tradeable_haircut", type=float, default=None)
    parser.add_argument("--distribution_rate", "--distribution-rate", dest="distribution_rate", type=float, default=None)
    parser.add_argument("--algo", type=str, default="ippo", choices=["ippo", "mappo"])
    parser.add_argument(
        "--risk_controller_rule_based",
        "--risk-controller-rule-based",
        dest="risk_controller_rule_based",
        action="store_true",
        default=False,
        help="Train/evaluate with the deterministic rule-based risk controller.",
    )
    parser.add_argument(
        "--meta_controller_rule_based",
        "--meta-controller-rule-based",
        dest="meta_controller_rule_based",
        action="store_true",
        default=False,
        help="Train/evaluate with the deterministic rule-based meta allocator.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent_coef", type=float, default=0.03)
    parser.add_argument("--ppo_log_std_init", type=float, default=None)
    parser.add_argument("--enable_ppo_use_sde", action="store_true")
    parser.add_argument(
        "--enable_episode_csv_logs",
        action="store_true",
        help="Forward to main.py to save per-episode debug CSVs and captured training output.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Forward to main.py to enable debug mode in the training environment.",
    )
    parser.add_argument(
        "--log-sleeve",
        "--log_sleeve",
        dest="log_sleeve",
        action="store_true",
        help="Forward to evaluation.py to save per-step sleeve logs and sleeve supplement metrics.",
    )
    parser.add_argument(
        "--sleeve_sharpe_mode",
        "--sleeve-sharpe-mode",
        dest="sleeve_sharpe_mode",
        choices=["daily", "daily_hac_7", "hac_144"],
        default="daily_hac_7",
        help="Primary trading-sleeve Sharpe reported by evaluation.",
    )
    parser.add_argument("--eval_data", type=str, default="evaluation_dataset/unseendata.csv")
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Train only and skip the built-in post-training eval. Use a separate eval script afterward.",
    )
    parser.add_argument(
        "--eval_distribution_rate",
        "--eval-distribution-rate",
        dest="eval_distribution_rate",
        type=float,
        default=None,
        help="Evaluation-only cash-sweeper distribution rate override; omitted preserves v1 behavior.",
    )
    parser.add_argument("--output_root", type=str, default="batch_tier_phase_runs")
    parser.add_argument("--suite_dir", type=str, default="",
                        help="Existing suite directory to resume into. If empty, a new one is created.")
    parser.add_argument("--start_run_number", type=int, default=1,
                        help="Global run number to start from (1-based) for resume after OOM.")
    parser.add_argument("--end_run_number", type=int, default=None,
                        help="Optional global run number to stop at (inclusive).")
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--continue_on_error", action="store_true")
    add_forecast_prior_override_args(parser)

    args = parser.parse_args()
    if str(args.global_norm_mode).strip().lower() != "rolling_past":
        print("Error: this suite is pinned to rolling_past for consistency. Use --global_norm_mode rolling_past.")
        sys.exit(1)
    seeds = _parse_seeds(args.seeds)
    if not seeds:
        print("Error: No valid seeds provided.")
        sys.exit(1)

    phase_variants = PHASE_VARIANTS[args.phase]
    py = sys.executable
    eval_steps = resolve_eval_steps(args.eval_data, args.eval_steps)

    if str(args.suite_dir).strip():
        suite_dir = str(args.suite_dir).strip()
        os.makedirs(suite_dir, exist_ok=True)
    else:
        suite_dir = _build_suite_dir(args.output_root, args.phase, args.run_tag)

    summary_csv = os.path.join(suite_dir, f"{args.phase}_seed_suite_summary.csv")
    run_plan_csv = os.path.join(suite_dir, f"{args.phase}_run_plan.csv")

    common_args = [
        "--episode_training",
        "--episode_data_dir", args.episode_data_dir,
        "--start_episode", str(args.start_episode),
        "--end_episode", str(args.end_episode),
        "--global_norm_mode", str(args.global_norm_mode),
        "--investment_freq", str(args.investment_freq),
        "--meta_freq_min", str(args.meta_freq_min),
        "--meta_freq_max", str(args.meta_freq_max),
        "--cooling_period", str(args.cooling_period),
        "--algo", str(args.algo),
        "--lr", str(args.lr),
        "--ent_coef", str(args.ent_coef),
    ]
    if args.ppo_log_std_init is not None:
        common_args.extend(["--ppo_log_std_init", str(args.ppo_log_std_init)])
    eff_roll = _effective_rolling_past_history_dir(args)
    if eff_roll:
        common_args.extend(["--rolling_past_history_dir", eff_roll])
    if bool(args.enable_ppo_use_sde):
        common_args.append("--ppo_use_sde")
    if bool(args.enable_episode_csv_logs):
        common_args.append("--enable_episode_csv_logs")
    if bool(args.debug):
        common_args.append("--debug")
    controller_args = []
    if bool(getattr(args, "risk_controller_rule_based", False)):
        controller_args.append("--risk_controller_rule_based")
    if bool(getattr(args, "meta_controller_rule_based", False)):
        controller_args.append("--meta_controller_rule_based")
    common_args.extend(controller_args)

    forecast_args = [
        "--forecast_cache_dir", args.forecast_cache_dir,
    ]
    sizing_args = []
    if args.investor_notional_sizing_base is not None:
        sizing_args.extend(["--investor_notional_sizing_base", str(args.investor_notional_sizing_base)])
    execution_args = []
    if args.max_position_size is not None:
        execution_args.extend(["--max_position_size", str(float(args.max_position_size))])
    if args.capital_allocation_fraction is not None:
        execution_args.extend(["--capital_allocation_fraction", str(float(args.capital_allocation_fraction))])
    if args.mtm_loss_exit_threshold_pct is not None:
        execution_args.extend(["--mtm_loss_exit_threshold_pct", str(float(args.mtm_loss_exit_threshold_pct))])
    if args.friction_cost_multiplier is not None:
        execution_args.extend(["--friction_cost_multiplier", str(float(args.friction_cost_multiplier))])
    if args.market_fee_model is not None:
        execution_args.extend(["--market_fee_model", str(args.market_fee_model)])
    if args.transaction_fee_dkk_per_mwh is not None:
        execution_args.extend([
            "--transaction_fee_dkk_per_mwh",
            str(float(args.transaction_fee_dkk_per_mwh)),
        ])
    if args.annual_market_access_fee_dkk is not None:
        execution_args.extend([
            "--annual_market_access_fee_dkk",
            str(float(args.annual_market_access_fee_dkk)),
        ])
    if args.market_access_fee_allocation_fraction is not None:
        execution_args.extend([
            "--market_access_fee_allocation_fraction",
            str(float(args.market_access_fee_allocation_fraction)),
        ])
    if args.market_fee_source_id is not None:
        execution_args.extend(["--market_fee_source_id", str(args.market_fee_source_id)])
    if args.no_trade_threshold is not None:
        execution_args.extend(["--no_trade_threshold", str(float(args.no_trade_threshold))])
    if args.no_trade_threshold_reference is not None:
        execution_args.extend(["--no_trade_threshold_reference", str(args.no_trade_threshold_reference)])
    if args.half_spread_bp is not None:
        execution_args.extend(["--half_spread_bp", str(float(args.half_spread_bp))])
    if args.impact_coef_bp is not None:
        execution_args.extend(["--impact_coef_bp", str(float(args.impact_coef_bp))])
    if args.impact_exponent is not None:
        execution_args.extend(["--impact_exponent", str(float(args.impact_exponent))])
    if args.impact_ref_notional is not None:
        execution_args.extend(["--impact_ref_notional", str(args.impact_ref_notional)])
    if args.impact_volume_data is not None:
        execution_args.extend(["--impact_volume_data", str(args.impact_volume_data)])
    if args.impact_volume_column is not None:
        execution_args.extend(["--impact_volume_column", str(args.impact_volume_column)])
    if args.impact_volume_unit is not None:
        execution_args.extend(["--impact_volume_unit", str(args.impact_volume_unit)])
    if args.impact_volume_timestamp_column is not None:
        execution_args.extend(["--impact_volume_timestamp_column", str(args.impact_volume_timestamp_column)])
    if args.impact_volume_max_staleness_min is not None:
        execution_args.extend(["--impact_volume_max_staleness_min", str(float(args.impact_volume_max_staleness_min))])
    if args.impact_volume_price_floor_dkk_per_mwh is not None:
        execution_args.extend(["--impact_volume_price_floor_dkk_per_mwh", str(float(args.impact_volume_price_floor_dkk_per_mwh))])
    for _name in (
        "liquidity_participation_cap_fraction",
        "liquidity_volume_multiplier",
        "liquidity_min_volume_mwh",
        "liquidity_tail_impact_threshold_dkk_per_mwh",
        "liquidity_tail_impact_multiplier",
        "liquidity_tail_impact_power",
        "liquidity_tail_impact_max_multiplier",
        "collateral_notional_margin_fraction",
        "collateral_stress_loss_fraction",
        "collateral_stress_price_dkk_per_mwh",
        "collateral_funding_rate_annual",
        "collateral_tradeable_haircut",
    ):
        if getattr(args, _name, None) is not None:
            execution_args.extend([f"--{_name}", str(float(getattr(args, _name)))])
    if args.liquidity_volume_source is not None:
        execution_args.extend(["--liquidity_volume_source", str(args.liquidity_volume_source)])
    if args.enable_collateral_cash_drag is not None:
        if bool(args.enable_collateral_cash_drag):
            execution_args.append("--enable_collateral_cash_drag")
    if bool(getattr(args, "disable_collateral_cash_drag", False)):
        execution_args.append("--disable_collateral_cash_drag")
    if args.distribution_rate is not None:
        execution_args.extend(["--distribution_rate", str(float(args.distribution_rate))])
    mtm_args = []
    if args.mtm_return_model is not None:
        mtm_args.extend(["--mtm_return_model", str(args.mtm_return_model)])
    if args.mtm_reference_price_dkk_per_mwh is not None:
        mtm_args.extend(["--mtm_reference_price_dkk_per_mwh", str(float(args.mtm_reference_price_dkk_per_mwh))])
    if args.mtm_settlement_horizon_steps is not None:
        mtm_args.extend(["--mtm_settlement_horizon_steps", str(int(args.mtm_settlement_horizon_steps))])
    if args.mtm_entry_price_mode is not None:
        mtm_args.extend(["--mtm_entry_price_mode", str(args.mtm_entry_price_mode)])
    if args.mtm_horizon_payoff_denominator_mode is not None:
        mtm_args.extend(["--mtm_horizon_payoff_denominator_mode", str(args.mtm_horizon_payoff_denominator_mode)])
    if args.mtm_settlement_price_mode is not None:
        mtm_args.extend(["--mtm_settlement_price_mode", str(args.mtm_settlement_price_mode)])
    if args.mtm_basis_price_data_path is not None:
        mtm_args.extend(["--mtm_basis_price_data_path", str(args.mtm_basis_price_data_path)])
    if args.mtm_basis_price_column is not None:
        mtm_args.extend(["--mtm_basis_price_column", str(args.mtm_basis_price_column)])
    if args.mtm_basis_timestamp_column is not None:
        mtm_args.extend(["--mtm_basis_timestamp_column", str(args.mtm_basis_timestamp_column)])
    if args.mtm_basis_scale is not None:
        mtm_args.extend(["--mtm_basis_scale", str(float(args.mtm_basis_scale))])
    if args.mtm_basis_centering_mode is not None:
        mtm_args.extend(["--mtm_basis_centering_mode", str(args.mtm_basis_centering_mode)])
    if args.mtm_basis_centering_window_steps is not None:
        mtm_args.extend(["--mtm_basis_centering_window_steps", str(int(args.mtm_basis_centering_window_steps))])
    if args.mtm_external_settlement_price_data_path is not None:
        mtm_args.extend([
            "--mtm_external_settlement_price_data_path",
            str(args.mtm_external_settlement_price_data_path),
        ])
    if args.mtm_external_settlement_price_column is not None:
        mtm_args.extend([
            "--mtm_external_settlement_price_column",
            str(args.mtm_external_settlement_price_column),
        ])
    if args.mtm_external_settlement_timestamp_column is not None:
        mtm_args.extend([
            "--mtm_external_settlement_timestamp_column",
            str(args.mtm_external_settlement_timestamp_column),
        ])
    if args.mtm_external_settlement_min_price_dkk_per_mwh is not None:
        mtm_args.extend([
            "--mtm_external_settlement_min_price_dkk_per_mwh",
            str(float(args.mtm_external_settlement_min_price_dkk_per_mwh)),
        ])
    if args.mtm_external_settlement_max_price_dkk_per_mwh is not None:
        mtm_args.extend([
            "--mtm_external_settlement_max_price_dkk_per_mwh",
            str(float(args.mtm_external_settlement_max_price_dkk_per_mwh)),
        ])
    if bool(args.disable_mtm_return_cap):
        mtm_args.append("--disable_mtm_return_cap")
    if args.mtm_price_return_cap_min is not None:
        mtm_args.extend(["--mtm_price_return_cap_min", str(float(args.mtm_price_return_cap_min))])
    if args.mtm_price_return_cap_max is not None:
        mtm_args.extend(["--mtm_price_return_cap_max", str(float(args.mtm_price_return_cap_max))])
    mtm_settings = mtm_contract_settings(args)
    sizing_settings = sizing_contract_settings(args)
    execution_settings = execution_contract_settings(args)
    controller_settings = controller_contract_settings(args)
    forecast_prior_overrides = collect_forecast_prior_overrides(args)
    forecast_prior_args = forecast_prior_override_cli_args(args)

    base_contract = build_runtime_contract(
        global_norm_mode=str(args.global_norm_mode),
        rolling_past_history_dir=eff_roll,
        investment_freq=int(args.investment_freq),
        meta_freq_min=int(args.meta_freq_min),
        meta_freq_max=int(args.meta_freq_max),
        enable_forecast_utilization=False,
        mtm_settings=mtm_settings,
        sizing_settings=sizing_settings,
        execution_settings=execution_settings,
        controller_settings=controller_settings,
    )
    base_contract_hash = runtime_contract_hash(base_contract)
    forecast_contract = build_runtime_contract(
        global_norm_mode=str(args.global_norm_mode),
        rolling_past_history_dir=eff_roll,
        investment_freq=int(args.investment_freq),
        meta_freq_min=int(args.meta_freq_min),
        meta_freq_max=int(args.meta_freq_max),
        enable_forecast_utilization=True,
        forecast_prior_settings=forecast_prior_contract_settings(forecast_prior_overrides),
        mtm_settings=mtm_settings,
        sizing_settings=sizing_settings,
        execution_settings=execution_settings,
        controller_settings=controller_settings,
    )
    runtime_contracts_by_variant = {
        "tier1": base_contract,
        "tier1_forecast_utilization": forecast_contract,
    }
    runtime_contract_hashes_by_variant = {
        key: runtime_contract_hash(value)
        for key, value in runtime_contracts_by_variant.items()
    }
    active_runtime_contract_hashes = {
        key: runtime_contract_hashes_by_variant[key]
        for key in phase_variants
    }

    protocol_path = os.path.join(suite_dir, "phase_protocol.json")

    if str(args.suite_dir).strip():
        existing_protocol = _load_existing_protocol(protocol_path)
        if existing_protocol:
            compatibility_checks = {
                "phase": args.phase,
                "global_norm_mode": str(args.global_norm_mode),
                "algo": str(args.algo),
                "investment_freq": int(args.investment_freq),
                "meta_freq_min": int(args.meta_freq_min),
                "meta_freq_max": int(args.meta_freq_max),
                "ppo_use_sde": bool(args.enable_ppo_use_sde),
                "ppo_log_std_init": None if args.ppo_log_std_init is None else float(args.ppo_log_std_init),
                "sleeve_sharpe_mode": str(args.sleeve_sharpe_mode),
                "runtime_contract_hash": base_contract_hash,
                "variant_runtime_contract_hashes": active_runtime_contract_hashes,
                "forecast_args": forecast_args,
                "forecast_prior_overrides": forecast_prior_overrides,
                "mtm_args": mtm_args,
                "mtm_settings": mtm_settings,
                "sizing_args": sizing_args,
                "sizing_settings": sizing_settings,
                "execution_args": execution_args,
                "execution_settings": execution_settings,
                "controller_args": controller_args,
                "controller_settings": controller_settings,
            }
            for key, expected in compatibility_checks.items():
                actual = existing_protocol.get(key)
                if key == "algo" and actual is None:
                    actual = "ippo"
                if actual != expected:
                    print(
                        f"Error: suite_dir protocol mismatch for '{key}': "
                        f"existing={actual!r}, requested={expected!r}"
                    )
                    sys.exit(1)
            if str(args.global_norm_mode).strip().lower() == "rolling_past":
                exp_roll = _effective_rolling_past_history_dir(args)
                act_roll = str(existing_protocol.get("rolling_past_history_dir", "") or "").strip()
                norm_roll = lambda x: x or DEFAULT_ROLLING_PAST_HISTORY_DIR
                if norm_roll(act_roll) != norm_roll(exp_roll):
                    print(
                        "Error: suite_dir protocol mismatch for 'rolling_past_history_dir' "
                        f"(normalized): existing={act_roll!r}, requested={exp_roll!r}"
                    )
                    sys.exit(1)

    write_phase_protocol_json(
        protocol_path,
        {
            "phase": args.phase,
            "phase_variants": list(phase_variants),
            "global_norm_mode": str(args.global_norm_mode),
            "algo": str(args.algo),
            "rolling_past_history_dir": _effective_rolling_past_history_dir(args),
            "runtime_contract": base_contract,
            "runtime_contract_hash": base_contract_hash,
            "engine_file_hashes": engine_file_hashes(),
            "variant_runtime_contracts": {
                key: runtime_contracts_by_variant[key]
                for key in phase_variants
            },
            "variant_runtime_contract_hashes": active_runtime_contract_hashes,
            "investment_freq": int(args.investment_freq),
            "meta_freq_min": int(args.meta_freq_min),
            "meta_freq_max": int(args.meta_freq_max),
            "ppo_use_sde": bool(args.enable_ppo_use_sde),
            "ppo_log_std_init": None if args.ppo_log_std_init is None else float(args.ppo_log_std_init),
            "skip_eval": bool(args.skip_eval),
            "sleeve_sharpe_mode": str(args.sleeve_sharpe_mode),
            "shared_training_args": common_args,
            "forecast_args": forecast_args,
            "forecast_prior_overrides": forecast_prior_overrides,
            "mtm_args": mtm_args,
            "mtm_settings": mtm_settings,
            "sizing_args": sizing_args,
            "sizing_settings": sizing_settings,
            "execution_args": execution_args,
            "execution_settings": execution_settings,
            "controller_args": controller_args,
            "controller_settings": controller_settings,
        },
    )

    # Build full run plan with deterministic global numbering.
    planned_runs = []
    global_run = 0
    for seed in seeds:
        seed_dir = os.path.join(suite_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        # Pre-resolve save_dirs by canonical variant within this seed so that
        # variants with reuse_train_from=<other> point at <other>'s save_dir.
        save_dir_by_canonical = {}
        for canonical in phase_variants:
            spec = VARIANT_TRAIN_SPECS[canonical]
            reuse_from = spec.get("reuse_train_from")
            if reuse_from:
                if reuse_from not in save_dir_by_canonical:
                    raise ValueError(
                        f"Variant '{canonical}' has reuse_train_from='{reuse_from}' "
                        f"but '{reuse_from}' has not been planned earlier in phase "
                        f"'{args.phase}'. Reorder PHASE_VARIANTS so the source "
                        f"variant comes first."
                    )
                save_dir_by_canonical[canonical] = save_dir_by_canonical[reuse_from]
            else:
                save_dir_by_canonical[canonical] = os.path.join(
                    seed_dir, f"{spec['slug']}_seed{seed}"
                )
        for idx, canonical in enumerate(phase_variants, 1):
            spec = VARIANT_TRAIN_SPECS[canonical]
            save_dir = save_dir_by_canonical[canonical]
            eval_out = os.path.join(seed_dir, "evaluations_2025", canonical)
            global_run += 1
            planned_runs.append({
                "global_run_number": global_run,
                "total_planned_runs": 0,
                "seed": seed,
                "seed_run_order": idx,
                "run_name": spec["name"],
                "canonical_variant": canonical,
                "save_dir": save_dir,
                "eval_output_dir": eval_out,
            })
    total_runs = len(planned_runs)
    for r in planned_runs:
        r["total_planned_runs"] = total_runs

    if args.start_run_number < 1:
        print("Error: --start_run_number must be >= 1")
        sys.exit(1)
    if args.start_run_number > total_runs:
        print(f"Error: --start_run_number {args.start_run_number} exceeds total planned runs {total_runs}")
        sys.exit(1)
    end_run = total_runs if args.end_run_number is None else int(args.end_run_number)
    if end_run < args.start_run_number:
        print("Error: --end_run_number must be >= --start_run_number")
        sys.exit(1)
    end_run = min(end_run, total_runs)

    write_run_plan_csv(planned_runs, run_plan_csv)

    print(f"\nPhase: {args.phase}")
    print(f"Seeds: {seeds}")
    print(f"Suite dir: {suite_dir}")
    print(f"Protocol: {protocol_path}")
    print(f"Run plan CSV: {run_plan_csv}")
    print(f"Summary CSV: {summary_csv}\n")
    print(f"Run range: {args.start_run_number}..{end_run} / {total_runs}\n")

    summary_rows = read_existing_summary_csv(summary_csv)
    stop = False
    for plan in planned_runs:
        if stop:
            break
        run_no = int(plan["global_run_number"])
        if run_no < args.start_run_number or run_no > end_run:
            continue

        seed = int(plan["seed"])
        idx = int(plan["seed_run_order"])
        canonical = str(plan["canonical_variant"])
        spec = VARIANT_TRAIN_SPECS[canonical]
        save_dir = str(plan["save_dir"])
        eval_out = str(plan["eval_output_dir"])

        label = f"[Run {run_no}/{total_runs}] {spec['name']} (seed={seed}, order={idx})"
        variant_extra = list(spec.get("extra", []) or [])
        reuse_from = spec.get("reuse_train_from")
        if reuse_from:
            print(f"\n[REUSE_TRAIN] {label}: reusing trained checkpoint from variant "
                  f"'{reuse_from}' at {save_dir}")
            if not os.path.isdir(os.path.join(save_dir, "final_models")):
                train_result = {
                    "success": False,
                    "returncode": -2,
                    "duration_seconds": 0,
                    "started_at": datetime.now().isoformat(),
                    "finished_at": datetime.now().isoformat(),
                }
                print(f"[REUSE_TRAIN][ERROR] expected final_models dir not found at "
                      f"{save_dir}; the source variant must have completed training.")
            else:
                train_result = {
                    "success": True,
                    "returncode": 0,
                    "duration_seconds": 0,
                    "started_at": datetime.now().isoformat(),
                    "finished_at": datetime.now().isoformat(),
                    "reused_from": reuse_from,
                }
        else:
            train_cmd = (
                [py, "main.py"]
                + common_args
                + ["--seed", str(seed), "--save_dir", save_dir]
                + forecast_args
                + mtm_args
                + sizing_args
                + execution_args
                + variant_extra
            )
            if canonical == "tier1_forecast_utilization":
                train_cmd += forecast_prior_args
            train_result = run_command(train_cmd, label)

        row = {
            "phase": args.phase,
            "global_run_number": run_no,
            "total_planned_runs": total_runs,
            "seed": seed,
            "run_order": idx,
            "run_name": spec["name"],
            "canonical_variant": canonical,
            "save_dir": save_dir,
            "eval_output_dir": eval_out,
            "training_success": train_result.get("success", False),
            "training_returncode": train_result.get("returncode", -1),
            "training_duration_seconds": train_result.get("duration_seconds", 0),
            "evaluation_success": False,
            "evaluation_skipped": bool(args.skip_eval),
        }

        if train_result.get("success") and args.skip_eval:
            row["evaluation_success"] = True
        elif train_result.get("success"):
            eval_dir_arg = EVAL_DIR_ARG_BY_VARIANT[canonical]
            os.makedirs(eval_out, exist_ok=True)
            tiers_only_val = EVAL_TIERS_ONLY_BY_VARIANT[canonical]
            # Variant-specific eval forecast-cache override (FoCAL needs the
            # episode-20 unseen-data cache; baseline ignores the cache entirely).
            eval_forecast_cache_dir = (
                EVAL_FORECAST_CACHE_DIR_OVERRIDE.get(canonical, "")
                or args.forecast_cache_dir
            )
            eval_cmd = [
                py, "evaluation.py", "--mode", "tiers",
                "--tiers_only", tiers_only_val,
                eval_dir_arg, save_dir,
                "--eval_data", args.eval_data,
                "--eval_steps", str(eval_steps),
                "--seed", str(seed),
                "--output_dir", eval_out,
                "--investment_freq", str(args.investment_freq),
                "--meta_freq_min", str(args.meta_freq_min),
                "--meta_freq_max", str(args.meta_freq_max),
                "--global_norm_mode", str(args.global_norm_mode),
                "--forecast_cache_dir", eval_forecast_cache_dir,
            ]
            eval_cmd.extend(mtm_args)
            if args.eval_mtm_basis_price_data_path is not None:
                eval_cmd.extend(["--mtm_basis_price_data_path", str(args.eval_mtm_basis_price_data_path)])
            if args.eval_mtm_external_settlement_price_data_path is not None:
                eval_cmd.extend([
                    "--mtm_external_settlement_price_data_path",
                    str(args.eval_mtm_external_settlement_price_data_path),
                ])
            eval_cmd.extend(sizing_args)
            eval_cmd.extend(execution_args)
            if args.eval_impact_volume_data is not None:
                eval_cmd.extend(["--impact_volume_data", str(args.eval_impact_volume_data)])
            elif args.impact_volume_data is not None and "{episode" in str(args.impact_volume_data):
                raise RuntimeError(
                    "--impact_volume_data contains an episode template; provide "
                    "--eval_impact_volume_data for built-in evaluation, or use --skip_eval."
                )
            eval_cmd.extend(controller_args)
            # Variant-specific eval extras (e.g., extra eval-only flags).
            eval_cmd.extend(list(spec.get("eval_extra", []) or []))
            if canonical == "tier1_forecast_utilization":
                eval_cmd.extend(forecast_prior_args)
            if bool(getattr(args, "log_sleeve", False)):
                eval_cmd.append("--log-sleeve")
                eval_cmd.extend(["--sleeve-sharpe-mode", str(args.sleeve_sharpe_mode)])
            # Evaluation-only paper stress toggle; absent means v1 behavior.
            if args.eval_distribution_rate is not None:
                eval_cmd.extend(["--eval-distribution-rate", str(float(args.eval_distribution_rate))])
            if eff_roll:
                eval_cmd.extend(["--rolling_past_history_dir", eff_roll])
            eval_label = f"[Run {run_no}/{total_runs}] Eval {canonical} (seed={seed})"
            eval_result = run_command(eval_cmd, eval_label)
            row["evaluation_success"] = eval_result.get("success", False)
            if row["evaluation_success"]:
                eval_json_path = find_latest_file(eval_out, "evaluation_tiers_*.json")
                if eval_json_path:
                    row["evaluation_results_json"] = eval_json_path
                    row.update(extract_eval_metrics(eval_json_path, canonical))

        summary_rows = upsert_summary_row(summary_rows, row)
        write_seed_summary_csv(summary_rows, summary_csv)

        if (not train_result.get("success") or not row.get("evaluation_success", False)) and not args.continue_on_error:
            stop = True
            break

    print(f"\nDone. Summary: {summary_csv}")
    overall_ok = all(as_bool(r.get("training_success")) and as_bool(r.get("evaluation_success")) for r in summary_rows)
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()

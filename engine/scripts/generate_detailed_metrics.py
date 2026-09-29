#!/usr/bin/env python3
"""Generate detailed final-paper metrics across Prototype result folders.

The script is intentionally read-only with respect to experiment outputs. It:

* scans plain MAPPO, the headline forecast anchor, learned integration methods,
  mechanism ablations, forecast-corruption evaluations, final baselines, and
  temporal baselines;
* extracts fund-level and trading-sleeve metrics from JSON/CSV results;
* recomputes trading-sleeve metrics from evaluation env logs when logs exist;
* flags runs where sleeve metrics cannot be verified because no env log exists;
* writes a PowerShell script with re-evaluation commands for missing sleeve logs.

Run from Prototype:

    python scripts/generate_detailed_metrics.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATIONS_ROOT = PROJECT_ROOT / "Ablations"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Ablations.mechanism_ablations import (
    prior_args_for_arm as campaign_prior_args_for_arm,
)

SEEDS = ["7", "42", "123", "2025", "3007", "5001", "8102", "9005", "10001", "11202"]

SETTLEMENT_CLIP_MIN_DKK = -111750.0
SETTLEMENT_CLIP_MAX_DKK = 111750.0
MARKET_FEE_ARGS = [
    "--market_fee_model", "nord_pool_intraday_2026",
    "--transaction_fee_dkk_per_mwh", "0.9238",
    "--annual_market_access_fee_dkk", "160175.0",
    "--market_access_fee_allocation_fraction", "1.0",
    "--market_fee_source_id", "nord_pool_nordic_baltic_2026_standard_participant",
]


COMMON_EVAL_ARGS = [
    "--mode",
    "tiers",
    "--tiers_only",
    "tier1",
    "--forecast_cache_dir",
    "forecast_cache_settlement_hourly_v2",
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
    *MARKET_FEE_ARGS,
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
    "--sleeve-sharpe-mode",
    "daily_hac_7",
    "--log-sleeve",
]


def prior_args_for_arm(arm: str) -> List[str]:
    """Use the campaign runner's canonical mechanism definition for reruns."""
    return list(campaign_prior_args_for_arm(arm))


def _settlement_eval_args(region: str) -> List[str]:
    settlement_path = (
        "evaluation_dataset_ffill/unseendata_v2_settlement_real_v2.csv"
        if region == "v2"
        else "evaluation_dataset_ffill/unseendata_settlement_real_v2.csv"
    )
    return [
        "--mtm_settlement_price_mode",
        "external_series",
        "--mtm_external_settlement_price_data_path",
        settlement_path,
        "--mtm_external_settlement_price_column",
        "settlement_price",
        "--mtm_external_settlement_timestamp_column",
        "timestamp",
        "--mtm_external_settlement_min_price_dkk_per_mwh",
        str(SETTLEMENT_CLIP_MIN_DKK),
        "--mtm_external_settlement_max_price_dkk_per_mwh",
        str(SETTLEMENT_CLIP_MAX_DKK),
    ]


def _eval_data_arg(region: str) -> str:
    return "evaluation_dataset_ffill/unseendata_v2.csv" if region == "v2" else "evaluation_dataset_ffill/unseendata.csv"


def _eval_dir_name(region: str) -> str:
    return "evaluations_2025_v2" if region == "v2" else "evaluations_2025"


def _liquidity_eval_args(region: str) -> List[str]:
    volume_path = (
        "evaluation_dataset_ffill/unseendata_v2_liquidity_volume_real_v1.csv"
        if region == "v2"
        else "evaluation_dataset_ffill/unseendata_liquidity_volume_real_v1.csv"
    )
    return [
        "--impact_volume_data",
        volume_path,
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


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _json_return_pct(tier: Dict[str, Any]) -> Optional[float]:
    if "total_return" in tier:
        value = _safe_float(tier.get("total_return"))
        return None if value is None else value * 100.0
    return _safe_float(tier.get("total_return_pct"))


def _json_drawdown_pct(tier: Dict[str, Any]) -> Optional[float]:
    if "max_drawdown" in tier:
        value = _safe_float(tier.get("max_drawdown"))
        return None if value is None else value * 100.0
    return _safe_float(tier.get("max_drawdown_pct"))


def _series_returns(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.asarray([], dtype=np.float64)
    prev = arr[:-1]
    cur = arr[1:]
    ok = np.isfinite(prev) & np.isfinite(cur) & (np.abs(prev) > 1e-12)
    return ((cur[ok] - prev[ok]) / prev[ok]).astype(np.float64)


def _subsample_path(values: Sequence[float], step: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr
    stride = max(int(step), 1)
    idx = np.arange(0, arr.size, stride, dtype=int)
    if idx.size < 2 and arr.size >= 2:
        idx = np.asarray([0, arr.size - 1], dtype=int)
    return arr[idx]


def _complete_period_path(
    post_step_values: Sequence[float],
    *,
    initial_value: float,
    steps_per_period: int,
) -> np.ndarray:
    """Return true initial equity followed by complete period-end equity."""
    arr = np.asarray(post_step_values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    stride = max(int(steps_per_period), 1)
    complete = int(arr.size // stride)
    if complete <= 0:
        return np.asarray([float(initial_value)], dtype=np.float64)
    period_ends = arr[stride - 1 : complete * stride : stride]
    return np.concatenate(
        (
            np.asarray([float(initial_value)], dtype=np.float64),
            period_ends,
        )
    )


def _annual_rate_to_step_rate(annual_rate: float, periods_per_year: float) -> float:
    annual = float(annual_rate)
    periods = float(max(periods_per_year, 1.0))
    if not np.isfinite(annual) or not np.isfinite(periods) or annual <= -1.0:
        return 0.0
    return float((1.0 + annual) ** (1.0 / periods) - 1.0)


def _acf(values: np.ndarray, lag: int) -> float:
    lag = int(lag)
    if values.size <= lag + 1:
        return float("nan")
    a = values[:-lag] - np.mean(values[:-lag])
    b = values[lag:] - np.mean(values[lag:])
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def _hac_variance_inflation(returns: np.ndarray, max_lag: int) -> float:
    if returns.size < 3 or int(max_lag) <= 0:
        return 1.0
    q = min(int(max_lag), returns.size - 2)
    total = 1.0
    for lag in range(1, q + 1):
        rho = _acf(returns, lag)
        if not math.isfinite(rho):
            continue
        weight = 1.0 - (lag / (q + 1.0))
        total += 2.0 * weight * rho
    return float(max(total, 1e-12))


def _path_sharpe(values: Sequence[float], periods_per_year: float, annual_risk_free_rate: float = 0.02) -> Dict[str, float]:
    returns = _series_returns(values)
    if returns.size < 2:
        return {
            "n_returns": float(returns.size),
            "mean_return": 0.0,
            "step_volatility": 0.0,
            "annualized_volatility": 0.0,
            "annualized_sharpe": 0.0,
        }
    periods = float(max(periods_per_year, 1.0))
    step_vol = float(np.std(returns, ddof=1))
    step_rf = _annual_rate_to_step_rate(float(annual_risk_free_rate), periods)
    excess_mean = float(np.mean(returns - step_rf))
    sharpe = float((excess_mean / step_vol) * math.sqrt(periods)) if step_vol > 0.0 else 0.0
    return {
        "n_returns": float(returns.size),
        "mean_return": float(np.mean(returns)),
        "step_volatility": step_vol,
        "annualized_volatility": float(step_vol * math.sqrt(periods)),
        "annualized_sharpe": sharpe,
    }


def _path_performance_metrics(
    values: Sequence[float],
    *,
    prefix: str,
    margin_active_any: bool = False,
    annual_risk_free_rate: float = 0.02,
    initial_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute the paper risk metrics for one economically isolated equity path."""
    path = np.asarray(values, dtype=np.float64).reshape(-1)
    path = path[np.isfinite(path)]
    if path.size == 0:
        return {f"{prefix}_metrics_error": "empty_path"}

    initial = float(path[0] if initial_value is None else initial_value)
    full_path = (
        path
        if initial_value is None
        else np.concatenate((np.asarray([initial], dtype=np.float64), path))
    )
    daily_path = (
        _subsample_path(path, 144)
        if initial_value is None
        else _complete_period_path(
            path,
            initial_value=initial,
            steps_per_period=144,
        )
    )
    daily_returns = _series_returns(daily_path)
    daily_stats = _path_sharpe(
        daily_path,
        periods_per_year=365.25,
        annual_risk_free_rate=annual_risk_free_rate,
    )
    daily_zero_rf_stats = _path_sharpe(
        daily_path,
        periods_per_year=365.25,
        annual_risk_free_rate=0.0,
    )
    hac_lag = min(7, max(int(daily_returns.size) - 2, 0))
    hac_vif = _hac_variance_inflation(daily_returns, hac_lag)
    hac_sharpe = (
        float(daily_stats["annualized_sharpe"] / math.sqrt(hac_vif))
        if hac_vif > 0.0
        else 0.0
    )
    hac_zero_rf = (
        float(daily_zero_rf_stats["annualized_sharpe"] / math.sqrt(hac_vif))
        if hac_vif > 0.0
        else 0.0
    )

    peak = np.maximum.accumulate(full_path)
    drawdowns = np.where(peak > 0.0, (peak - full_path) / peak, 0.0)
    max_drawdown = float(np.max(drawdowns)) if drawdowns.size else 0.0
    nonpositive_equity = bool(np.any(full_path <= 0.0))
    ruined = bool(margin_active_any or nonpositive_equity or max_drawdown >= 0.99)

    out: Dict[str, Any] = {
        f"{prefix}_initial_usd": initial,
        f"{prefix}_final_usd": float(path[-1]),
        f"{prefix}_gain_usd": float(path[-1] - initial),
        f"{prefix}_return_pct": (
            float((path[-1] / initial - 1.0) * 100.0)
            if abs(initial) > 1e-12
            else None
        ),
        f"{prefix}_daily_sharpe_ratio": (
            None if ruined else float(daily_stats["annualized_sharpe"])
        ),
        f"{prefix}_daily_hac7_sharpe_ratio": None if ruined else hac_sharpe,
        f"{prefix}_daily_hac7_zero_rf_sharpe_ratio": None if ruined else hac_zero_rf,
        f"{prefix}_daily_hac7_vif": float(hac_vif),
        f"{prefix}_annualized_volatility": float(
            daily_stats["annualized_volatility"] * math.sqrt(hac_vif)
        ),
        f"{prefix}_max_drawdown_pct": float(max_drawdown * 100.0),
        f"{prefix}_daily_n_returns": float(daily_returns.size),
        f"{prefix}_eval_window_days": float(path.size / 144.0),
        f"{prefix}_path_convention": "true_initial_plus_complete_day_ends_v1",
        f"{prefix}_ruined": ruined,
        f"{prefix}_ruin_reason": ";".join(
            reason
            for condition, reason in [
                (margin_active_any, "maintenance_margin_triggered"),
                (nonpositive_equity, "nonpositive_equity"),
                (max_drawdown >= 0.99, "drawdown_at_least_99pct"),
            ]
            if condition
        ),
    }
    return out


def _read_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
        return (handle.readline() or "").strip().split(",")


def _to_numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in df.columns:
        return np.full(len(df), float(default), dtype=np.float64)
    return pd.to_numeric(df[column], errors="coerce").fillna(float(default)).to_numpy(dtype=np.float64)


def compute_sleeve_metrics_from_env_log(csv_path: Path, dkk_to_usd_rate: float = 0.145) -> Dict[str, Any]:
    required = {
        "fund_nav_dkk",
        "trading_cash_dkk",
        "physical_book_value_dkk",
        "accumulated_operational_revenue_dkk",
        "financial_mtm_dkk",
    }
    optional = {
        "timestamp",
        "financial_exposure_dkk",
        "decision_step",
        "total_distributions_dkk",
        "distribution_adjusted_nav_dkk",
        "distribution_adjusted_trading_sleeve_dkk",
        "cumulative_trading_costs_dkk",
        "cumulative_impact_costs_dkk",
        "cumulative_volume_transaction_fees_dkk",
        "cumulative_market_access_fees_dkk",
        "cumulative_collateral_funding_costs_dkk",
        "battery_cash_delta",
        "collateral_required_dkk",
        "collateral_funding_cost_dkk",
        "collateral_open_notional_dkk",
        "collateral_open_volume_mwh",
        "liquidity_market_volume_mwh",
        "liquidity_causal_market_volume_mwh",
        "liquidity_volume_cap_mwh",
        "liquidity_requested_volume_mwh",
        "liquidity_executed_volume_mwh",
        "liquidity_participation",
        "liquidity_scale",
        "liquidity_tail_impact_multiplier",
        "liquidity_tail_spread_dkk_per_mwh",
        "horizon_settlement_pnl_dkk",
        "horizon_settlement_count",
        "forecast_prior_active",
        "forecast_prior_exposure",
        "forecast_only_prior_exposure",
        "forecast_beta_exposure",
        "forecast_beta_strength",
        "forecast_beta_active",
        "forecast_beta_payoff_lcb",
        "forecast_prior_skill",
        "forecast_prior_hit_lcb",
        "forecast_prior_hit_rate",
        "forecast_prior_direction_confidence",
        "forecast_prior_confidence_weight",
        "forecast_adjustment",
        "forecast_alignment",
        "forecast_tail_cap_abs",
        "forecast_tail_return",
        "forecast_tail_count",
        "distributional_edge_strength",
        "distributional_conditional_tail",
        "distributional_disaster_tail",
        "distributional_conditional_cap_abs",
        "distributional_disaster_cap_abs",
        "distributional_bucket_count",
        "distributional_global_count",
        "distributional_using_bucket",
        "distributional_bucket_mode_global",
        "distributional_bucket_mode_directional_fixed_cap",
        "trading_sleeve_margin_active",
        "trading_sleeve_margin_step",
        "trading_sleeve_margin_gap_dkk",
    }

    header = set(_read_header(csv_path))
    missing = sorted(required - header)
    if missing:
        return {"sleeve_metrics_error": "missing_required_columns:" + ",".join(missing)}

    usecols = sorted(required | (optional & header))
    df = pd.read_csv(csv_path, usecols=usecols)
    if df.empty:
        return {"sleeve_metrics_error": "empty_env_log"}

    nav_reported = _to_numeric(df, "fund_nav_dkk") * dkk_to_usd_rate
    nav = (
        _to_numeric(df, "distribution_adjusted_nav_dkk") * dkk_to_usd_rate
        if "distribution_adjusted_nav_dkk" in df.columns
        else nav_reported
    )
    trading_reported = (_to_numeric(df, "trading_cash_dkk") + _to_numeric(df, "financial_mtm_dkk")) * dkk_to_usd_rate
    trading = (
        _to_numeric(df, "distribution_adjusted_trading_sleeve_dkk") * dkk_to_usd_rate
        if "distribution_adjusted_trading_sleeve_dkk" in df.columns
        else trading_reported
    )
    operating = (
        _to_numeric(df, "physical_book_value_dkk") + _to_numeric(df, "accumulated_operational_revenue_dkk")
    ) * dkk_to_usd_rate
    battery_cash_delta_dkk = (
        _to_numeric(df, "battery_cash_delta")
        if "battery_cash_delta" in df.columns
        else np.zeros(len(df), dtype=np.float64)
    )
    cumulative_battery_cash_usd = np.cumsum(battery_cash_delta_dkk) * dkk_to_usd_rate
    investor_contract = trading - cumulative_battery_cash_usd
    from config import EnhancedConfig

    cfg = EnhancedConfig()
    initial_fund_usd = float(cfg.init_budget) * dkk_to_usd_rate
    initial_sleeve_usd = (
        float(cfg.init_budget)
        * float(cfg.financial_allocation)
        * dkk_to_usd_rate
    )
    initial_operating_usd = initial_fund_usd - initial_sleeve_usd
    trading_with_initial = np.concatenate(
        (np.asarray([initial_sleeve_usd], dtype=np.float64), trading)
    )

    metrics: Dict[str, Any] = {
        "sleeve_path_convention": "true_initial_plus_complete_day_ends_v1",
        "sleeve_total_initial_usd": initial_fund_usd,
        "sleeve_total_final_usd": float(nav[-1]),
        "sleeve_total_gain_usd": float(nav[-1] - initial_fund_usd),
        "sleeve_reported_nav_initial_usd": float(nav_reported[0]),
        "sleeve_reported_nav_final_usd": float(nav_reported[-1]),
        "sleeve_trading_initial_usd": initial_sleeve_usd,
        "sleeve_trading_final_usd": float(trading[-1]),
        "sleeve_trading_gain_usd": float(trading[-1] - initial_sleeve_usd),
        "sleeve_reported_trading_initial_usd": float(trading_reported[0]),
        "sleeve_reported_trading_final_usd": float(trading_reported[-1]),
        "sleeve_operating_initial_usd": initial_operating_usd,
        "sleeve_operating_final_usd": float(operating[-1]),
        "sleeve_operating_gain_usd": float(operating[-1] - initial_operating_usd),
        "battery_cumulative_cash_delta_usd": float(cumulative_battery_cash_usd[-1]),
    }
    total_gain = float(nav[-1] - initial_fund_usd)
    trading_gain = float(trading[-1] - initial_sleeve_usd)
    if abs(total_gain) > 1e-12:
        metrics["sleeve_trading_gain_share"] = float(trading_gain / total_gain)
    if abs(initial_sleeve_usd) > 1e-12:
        metrics["sleeve_trading_return_pct"] = float(
            (trading[-1] / initial_sleeve_usd - 1.0) * 100.0
        )
    if abs(initial_operating_usd) > 1e-12:
        metrics["sleeve_operating_return_pct"] = float(
            (operating[-1] / initial_operating_usd - 1.0) * 100.0
        )

    daily_path = _complete_period_path(
        trading,
        initial_value=initial_sleeve_usd,
        steps_per_period=144,
    )
    daily_returns = _series_returns(daily_path)
    daily_stats = _path_sharpe(daily_path, periods_per_year=365.25)
    daily_zero_rf_stats = _path_sharpe(
        daily_path,
        periods_per_year=365.25,
        annual_risk_free_rate=0.0,
    )
    daily_hac_lag = min(7, max(int(daily_returns.size) - 2, 0))
    daily_hac_vif = _hac_variance_inflation(daily_returns, daily_hac_lag)
    daily_hac = (
        float(daily_stats["annualized_sharpe"] / math.sqrt(daily_hac_vif))
        if daily_hac_vif > 0.0
        else 0.0
    )
    daily_hac_zero_rf = (
        float(daily_zero_rf_stats["annualized_sharpe"] / math.sqrt(daily_hac_vif))
        if daily_hac_vif > 0.0
        else 0.0
    )
    weekly_stats = _path_sharpe(_subsample_path(daily_path, 7), periods_per_year=365.25 / 7.0)
    monthly_stats = _path_sharpe(_subsample_path(daily_path, 30), periods_per_year=365.25 / 30.0)
    peak = np.maximum.accumulate(trading_with_initial)
    drawdowns = np.where(
        peak > 0.0,
        (peak - trading_with_initial) / peak,
        0.0,
    )
    drawdown = float(np.max(drawdowns)) if drawdowns.size else 0.0
    margin_active_any = False
    if "trading_sleeve_margin_active" in df.columns:
        margin_values = _to_numeric(df, "trading_sleeve_margin_active")
        margin_active_any = bool(np.any(margin_values > 0.5))
    nonpositive_equity = bool(np.any(trading_with_initial <= 0.0))
    ruined = bool(margin_active_any or nonpositive_equity or drawdown >= 0.99)
    ruin_reasons: List[str] = []
    if margin_active_any:
        ruin_reasons.append("maintenance_margin_triggered")
    if nonpositive_equity:
        ruin_reasons.append("nonpositive_sleeve_equity")
    if drawdown >= 0.99:
        ruin_reasons.append("drawdown_at_least_99pct")
    daily_sharpe = float(daily_stats["annualized_sharpe"])
    daily_zero_rf_sharpe = float(daily_zero_rf_stats["annualized_sharpe"])
    weekly_sharpe = float(weekly_stats["annualized_sharpe"])
    monthly_sharpe = float(monthly_stats["annualized_sharpe"])
    metrics.update(
        {
            "sleeve_trading_sharpe_primary_mode": "daily_hac_7",
            "sleeve_trading_ruined": ruined,
            "sleeve_trading_ruin_reason": ";".join(ruin_reasons),
            "sleeve_trading_annual_risk_free_rate": 0.02,
            "sleeve_trading_daily_sharpe_ratio_raw": daily_sharpe,
            "sleeve_trading_daily_hac7_sharpe_ratio_raw": float(daily_hac),
            "sleeve_trading_daily_zero_rf_sharpe_ratio_raw": daily_zero_rf_sharpe,
            "sleeve_trading_daily_hac7_zero_rf_sharpe_ratio_raw": float(daily_hac_zero_rf),
            "sleeve_trading_daily_sharpe_ratio": None if ruined else daily_sharpe,
            "sleeve_trading_daily_hac7_sharpe_ratio": None if ruined else float(daily_hac),
            "sleeve_trading_daily_zero_rf_sharpe_ratio": None if ruined else daily_zero_rf_sharpe,
            "sleeve_trading_daily_hac7_zero_rf_sharpe_ratio": None if ruined else float(daily_hac_zero_rf),
            "sleeve_trading_daily_hac7_vif": float(daily_hac_vif),
            "sleeve_trading_daily_hac7_nonannual_365": float(daily_hac / math.sqrt(365.25)),
            "sleeve_trading_daily_hac7_annualized_252": float(daily_hac / math.sqrt(365.25) * math.sqrt(252.0)),
            "sleeve_trading_7d_sharpe_ratio_raw": weekly_sharpe,
            "sleeve_trading_7d_sharpe_ratio": None if ruined else weekly_sharpe,
            "sleeve_trading_7d_n_returns": float(weekly_stats["n_returns"]),
            "sleeve_trading_30d_sharpe_ratio_raw": monthly_sharpe,
            "sleeve_trading_30d_sharpe_ratio": None if ruined else monthly_sharpe,
            "sleeve_trading_30d_n_returns": float(monthly_stats["n_returns"]),
            "sleeve_trading_volatility": float(daily_stats["annualized_volatility"] * math.sqrt(daily_hac_vif)),
            "sleeve_trading_step_volatility": float(daily_stats["step_volatility"] * math.sqrt(daily_hac_vif)),
            "sleeve_trading_sharpe_ratio_raw": float(daily_hac),
            "sleeve_trading_sharpe_ratio": None if ruined else float(daily_hac),
            "sleeve_trading_max_drawdown_pct": drawdown * 100.0,
            "sleeve_trading_daily_n_returns": float(daily_returns.size),
            "sleeve_eval_window_days": float(len(trading) / 144.0),
        }
    )
    metrics.update(
        _path_performance_metrics(
            investor_contract,
            prefix="investor_contract",
            margin_active_any=margin_active_any,
            annual_risk_free_rate=0.02,
            initial_value=initial_sleeve_usd,
        )
    )
    eval_window_days = float(len(trading) / 144.0)
    if eval_window_days > 0.0 and initial_sleeve_usd > 0.0 and trading[-1] > 0.0:
        metrics["sleeve_trading_annualized_geometric_return_pct"] = float(
            (
                (trading[-1] / initial_sleeve_usd)
                ** (365.25 / eval_window_days)
                - 1.0
            )
            * 100.0
        )
    if len(daily_path) > 1 and daily_path[0] != 0.0:
        metrics["sleeve_trading_complete_daily_window_return_pct"] = float(
            (daily_path[-1] / daily_path[0] - 1.0) * 100.0
        )
    if len(daily_path) and daily_path[-1] != 0.0:
        metrics["sleeve_trading_final_partial_day_return_pct"] = float(
            (trading[-1] / daily_path[-1] - 1.0) * 100.0
        )

    if "financial_exposure_dkk" in df.columns:
        expo = _to_numeric(df, "financial_exposure_dkk")
        decision = _to_numeric(df, "decision_step") > 0.5 if "decision_step" in df.columns else np.ones(len(df), dtype=bool)
        expo_dec = expo[decision] if decision.any() else expo
        metrics["sleeve_mean_abs_exposure_dkk"] = float(np.mean(np.abs(expo_dec))) if expo_dec.size else 0.0
        metrics["sleeve_max_abs_exposure_dkk"] = float(np.max(np.abs(expo_dec))) if expo_dec.size else 0.0
        gross_turnover = float(np.sum(np.abs(np.diff(expo)))) if expo.size > 1 else 0.0
        initial_sleeve_dkk = (
            float(initial_sleeve_usd / dkk_to_usd_rate)
            if dkk_to_usd_rate
            else 0.0
        )
        metrics["sleeve_exposure_gross_turnover_dkk"] = gross_turnover
        if initial_sleeve_dkk > 0.0:
            days = max(float(len(expo)) / 144.0, 1e-9)
            metrics["sleeve_exposure_turnover_x_sleeve"] = float(gross_turnover / initial_sleeve_dkk)
            metrics["sleeve_exposure_turnover_x_sleeve_per_day"] = float(gross_turnover / initial_sleeve_dkk / days)

    if "cumulative_trading_costs_dkk" in df.columns:
        costs = _to_numeric(df, "cumulative_trading_costs_dkk")
        total_costs_usd = float(costs[-1] * dkk_to_usd_rate)
        metrics["sleeve_total_trading_costs_usd"] = total_costs_usd
        if "cumulative_impact_costs_dkk" in df.columns:
            impact = _to_numeric(df, "cumulative_impact_costs_dkk")
            metrics["sleeve_total_impact_costs_usd"] = float(impact[-1] * dkk_to_usd_rate)
        if "cumulative_volume_transaction_fees_dkk" in df.columns:
            volume_fees = _to_numeric(df, "cumulative_volume_transaction_fees_dkk")
            metrics["sleeve_total_volume_transaction_fees_usd"] = float(
                volume_fees[-1] * dkk_to_usd_rate
            )
        if "cumulative_market_access_fees_dkk" in df.columns:
            access_fees = _to_numeric(df, "cumulative_market_access_fees_dkk")
            metrics["sleeve_total_market_access_fees_usd"] = float(
                access_fees[-1] * dkk_to_usd_rate
            )
        if "cumulative_collateral_funding_costs_dkk" in df.columns:
            collateral_costs = _to_numeric(df, "cumulative_collateral_funding_costs_dkk")
            metrics["sleeve_total_collateral_funding_costs_usd"] = float(
                collateral_costs[-1] * dkk_to_usd_rate
            )
        gross_gain_usd = float(trading_gain + total_costs_usd)
        metrics["sleeve_gross_trading_gain_usd"] = gross_gain_usd
        if abs(gross_gain_usd) > 1e-9:
            metrics["sleeve_friction_share_of_gross_gain"] = float(total_costs_usd / abs(gross_gain_usd))

    if "liquidity_participation" in df.columns:
        participation = _to_numeric(df, "liquidity_participation")
        metrics["sleeve_liquidity_max_participation"] = float(np.max(participation)) if participation.size else 0.0
        metrics["sleeve_liquidity_mean_participation"] = float(np.mean(participation)) if participation.size else 0.0
    if "liquidity_scale" in df.columns:
        scale = _to_numeric(df, "liquidity_scale", default=1.0)
        metrics["sleeve_liquidity_bind_share"] = float(np.mean(scale < 0.999999)) if scale.size else 0.0
        metrics["sleeve_liquidity_min_scale"] = float(np.min(scale)) if scale.size else 1.0
    if "liquidity_tail_impact_multiplier" in df.columns:
        tail_mult = _to_numeric(df, "liquidity_tail_impact_multiplier", default=1.0)
        metrics["sleeve_liquidity_tail_impact_max_multiplier"] = float(np.max(tail_mult)) if tail_mult.size else 1.0
        metrics["sleeve_liquidity_tail_impact_mean_multiplier"] = float(np.mean(tail_mult)) if tail_mult.size else 1.0
    if "collateral_required_dkk" in df.columns:
        collateral_required = _to_numeric(df, "collateral_required_dkk")
        metrics["sleeve_collateral_required_max_dkk"] = (
            float(np.max(collateral_required)) if collateral_required.size else 0.0
        )
        metrics["sleeve_collateral_required_mean_dkk"] = (
            float(np.mean(collateral_required)) if collateral_required.size else 0.0
        )

    if "horizon_settlement_pnl_dkk" in df.columns and "horizon_settlement_count" in df.columns:
        pnl = _to_numeric(df, "horizon_settlement_pnl_dkk")
        cnt = _to_numeric(df, "horizon_settlement_count")
        events = pnl[cnt > 0.5]
        metrics["sleeve_settlement_event_count"] = float(events.size)
        if events.size:
            metrics["sleeve_settlement_pnl_mean_dkk"] = float(np.mean(events))
            metrics["sleeve_settlement_pnl_std_dkk"] = float(np.std(events, ddof=0))
            metrics["sleeve_settlement_pnl_hit_rate"] = float(np.mean(events > 0.0))
            metrics["sleeve_settlement_pnl_p1_dkk"] = float(np.percentile(events, 1))
            metrics["sleeve_settlement_pnl_p5_dkk"] = float(np.percentile(events, 5))
            metrics["sleeve_settlement_pnl_p95_dkk"] = float(np.percentile(events, 95))
            metrics["sleeve_settlement_pnl_p99_dkk"] = float(np.percentile(events, 99))
            losses = events[events < 0.0]
            gains = events[events > 0.0]
            metrics["sleeve_settlement_total_loss_dkk"] = float(-np.sum(losses))
            metrics["sleeve_settlement_total_gain_dkk"] = float(np.sum(gains))
            metrics["sleeve_settlement_average_loss_dkk"] = (
                float(np.mean(losses)) if losses.size else 0.0
            )
            metrics["sleeve_settlement_average_gain_dkk"] = (
                float(np.mean(gains)) if gains.size else 0.0
            )
            metrics["sleeve_settlement_max_loss_dkk"] = (
                float(np.min(losses)) if losses.size else 0.0
            )
            metrics["sleeve_settlement_max_gain_dkk"] = (
                float(np.max(gains)) if gains.size else 0.0
            )
            for pct, label in [(1.0, "1"), (5.0, "5")]:
                cutoff = float(np.percentile(events, pct))
                tail = events[events <= cutoff]
                metrics[f"sleeve_settlement_expected_shortfall_{label}pct_dkk"] = (
                    float(np.mean(tail)) if tail.size else 0.0
                )
            tail_count = max(1, int(math.ceil(events.size * 0.01)))
            if losses.size:
                worst_losses = np.sort(losses)[: min(tail_count, losses.size)]
                total_loss = float(-np.sum(losses))
                metrics["sleeve_settlement_worst1pct_loss_share"] = (
                    float(-np.sum(worst_losses) / total_loss) if total_loss > 0.0 else 0.0
                )
            if gains.size:
                best_gains = np.sort(gains)[-min(tail_count, gains.size):]
                total_gain = float(np.sum(gains))
                metrics["sleeve_settlement_best1pct_gain_share"] = (
                    float(np.sum(best_gains) / total_gain) if total_gain > 0.0 else 0.0
                )

    decision_mask = _to_numeric(df, "decision_step") > 0.5 if "decision_step" in df.columns else None
    if decision_mask is not None:
        metrics["focal_diagnostic_decision_rows"] = float(np.sum(decision_mask))
    for col, out_name, reducer in [
        ("forecast_prior_active", "focal_active_share", lambda x: np.mean(x > 0.5)),
        ("forecast_prior_exposure", "focal_mean_abs_prior_exposure", lambda x: np.mean(np.abs(x))),
        ("forecast_only_prior_exposure", "focal_forecast_only_mean_abs_exposure", lambda x: np.mean(np.abs(x))),
        ("forecast_beta_exposure", "focal_beta_mean_abs_exposure", lambda x: np.mean(np.abs(x))),
        ("forecast_beta_strength", "focal_beta_mean_strength", np.mean),
        ("forecast_beta_active", "focal_beta_active_share", lambda x: np.mean(x > 0.5)),
        ("forecast_beta_payoff_lcb", "focal_beta_mean_payoff_lcb", np.mean),
        ("forecast_prior_skill", "focal_mean_skill", np.mean),
        ("forecast_prior_hit_lcb", "focal_mean_hit_lcb", np.mean),
        ("forecast_prior_hit_rate", "focal_mean_hit_rate", np.mean),
        ("forecast_prior_direction_confidence", "focal_mean_direction_confidence", np.mean),
        ("forecast_prior_confidence_weight", "focal_mean_confidence_weight", np.mean),
        ("forecast_adjustment", "focal_mean_abs_adjustment", lambda x: np.mean(np.abs(x))),
        ("forecast_tail_cap_abs", "focal_tail_cap_abs_mean", np.mean),
        ("forecast_tail_return", "focal_tail_return_mean", np.mean),
        ("distributional_edge_strength", "distributional_edge_strength_mean", np.mean),
        ("distributional_conditional_tail", "distributional_conditional_tail_mean", np.mean),
        ("distributional_disaster_tail", "distributional_disaster_tail_mean", np.mean),
        ("distributional_conditional_cap_abs", "distributional_conditional_cap_abs_mean", np.mean),
        ("distributional_disaster_cap_abs", "distributional_disaster_cap_abs_mean", np.mean),
    ]:
        if col in df.columns:
            arr = _to_numeric(df, col)
            metrics[out_name] = float(reducer(arr)) if arr.size else 0.0
            if decision_mask is not None and decision_mask.any():
                metrics[out_name + "_decision"] = float(reducer(arr[decision_mask]))

    if "forecast_alignment" in df.columns:
        arr = _to_numeric(df, "forecast_alignment")
        metrics["focal_agreement_share"] = float(np.mean(arr > 0.0)) if arr.size else 0.0
        metrics["focal_conflict_share"] = float(np.mean(arr < 0.0)) if arr.size else 0.0
        if decision_mask is not None and decision_mask.any():
            arr_dec = arr[decision_mask]
            metrics["focal_agreement_share_decision"] = float(np.mean(arr_dec > 0.0))
            metrics["focal_conflict_share_decision"] = float(np.mean(arr_dec < 0.0))

    if "trading_sleeve_margin_active" in df.columns:
        active = _to_numeric(df, "trading_sleeve_margin_active")
        metrics["sleeve_margin_active_any"] = bool(np.any(active > 0.5))
        metrics["sleeve_margin_active_share"] = float(np.mean(active > 0.5)) if active.size else 0.0
    if "trading_sleeve_margin_step" in df.columns:
        steps = _to_numeric(df, "trading_sleeve_margin_step", -1.0)
        valid = steps[steps >= 0.0]
        metrics["sleeve_margin_first_step"] = float(np.min(valid)) if valid.size else -1.0
    if "trading_sleeve_margin_gap_dkk" in df.columns:
        gap = _to_numeric(df, "trading_sleeve_margin_gap_dkk")
        metrics["sleeve_margin_min_gap_dkk"] = float(np.min(gap)) if gap.size else 0.0

    ruin = bool(
        metrics.get("sleeve_margin_active_any", False)
        or metrics.get("sleeve_trading_max_drawdown_pct", 0.0) >= 99.0
    )
    low_events = float(metrics.get("sleeve_settlement_event_count", 0.0) or 0.0) < 50.0
    metrics["sleeve_ruin_guard_flag"] = ruin
    metrics["sleeve_low_event_guard_flag"] = low_events
    metrics["sleeve_trading_sharpe_ruin_guarded"] = (
        None if ruin or low_events else metrics.get("sleeve_trading_sharpe_ratio")
    )
    return metrics


def _latest_json(path: Path) -> Optional[Path]:
    files = sorted(path.glob("evaluation_tiers_*.json"))
    return files[-1] if files else None


def _tier_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    tiers = payload.get("tiers")
    if isinstance(tiers, dict):
        tier = tiers.get("tier1")
        if isinstance(tier, dict):
            return tier
        for value in tiers.values():
            if isinstance(value, dict):
                return value
    return payload


def _find_env_log(result_json: Path, tier: Dict[str, Any]) -> Optional[Path]:
    candidates: List[Path] = []
    raw = tier.get("env_debug_log")
    if raw:
        candidates.append(Path(str(raw)))
    raw_dir = tier.get("env_log_dir")
    if raw_dir:
        candidates.extend([Path(str(raw_dir)) / "tier1_debug_ep0.csv", Path(str(raw_dir)) / "tier1_portfolio_ep0.csv"])
    candidates.extend(
        [
            result_json.parent / "env_logs" / "tier1_debug_ep0.csv",
            result_json.parent / "env_logs" / "tier1_portfolio_ep0.csv",
        ]
    )
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.is_file():
            return candidate
    return None


def _flatten_metrics(prefix: str, metrics: Dict[str, Any], row: Dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, (str, bool)) or value is None:
            row[f"{prefix}{key}"] = value
        else:
            fval = _safe_float(value)
            if fval is not None:
                row[f"{prefix}{key}"] = fval


def _row_from_json(
    *,
    category: str,
    arm: str,
    region: str,
    seed: str,
    result_json: Path,
    allow_recompute: bool = True,
) -> Dict[str, Any]:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    tier = _tier_from_payload(payload)
    sleeve_json = tier.get("sleeve_metrics") if isinstance(tier.get("sleeve_metrics"), dict) else {}
    env_log = _find_env_log(result_json, tier)
    sleeve_recomputed: Dict[str, Any] = {}
    if allow_recompute and env_log is not None:
        try:
            sleeve_recomputed = compute_sleeve_metrics_from_env_log(env_log)
        except Exception as exc:
            sleeve_recomputed = {"sleeve_metrics_error": f"recompute_failed:{exc}"}

    sleeve = sleeve_recomputed if sleeve_recomputed and "sleeve_metrics_error" not in sleeve_recomputed else sleeve_json
    sleeve_source = "env_log_recomputed" if sleeve is sleeve_recomputed and sleeve_recomputed else "json"
    if not sleeve:
        sleeve_source = "missing"

    row: Dict[str, Any] = {
        "category": category,
        "arm": arm,
        "region": region,
        "seed": seed,
        "status": tier.get("status", ""),
        "result_json": str(result_json),
        "env_log": str(env_log) if env_log else "",
        "has_json_sleeve_metrics": bool(sleeve_json),
        "has_env_log": bool(env_log),
        "sleeve_metrics_source": sleeve_source,
        "needs_sleeve_rerun": bool(not sleeve_json and env_log is None and category == "mechanism_ablation"),
        "fund_return_pct": _json_return_pct(tier),
        "fund_sharpe": _safe_float(tier.get("sharpe_ratio")),
        "fund_drawdown_pct": _json_drawdown_pct(tier),
        "reported_nav_sharpe": _safe_float(tier.get("reported_nav_sharpe_ratio")),
        "eval_runtime_contract_hash": tier.get("eval_runtime_contract_hash", ""),
        "train_runtime_contract_hash": tier.get("train_runtime_contract_hash", ""),
        "eval_forecast_overlay": tier.get("eval_forecast_overlay", ""),
        "mtm_return_model": tier.get("mtm_return_model", ""),
        "mtm_settlement_price_mode": tier.get("mtm_settlement_price_mode", ""),
    }
    _flatten_metrics("", sleeve, row)
    if sleeve_json and sleeve_recomputed and "sleeve_metrics_error" not in sleeve_recomputed:
        row["sleeve_json_vs_recomputed_return_delta"] = (
            _safe_float(sleeve_recomputed.get("sleeve_trading_return_pct")) or 0.0
        ) - (_safe_float(sleeve_json.get("sleeve_trading_return_pct")) or 0.0)
        row["sleeve_json_vs_recomputed_sharpe_delta"] = (
            _safe_float(sleeve_recomputed.get("sleeve_trading_sharpe_ratio")) or 0.0
        ) - (_safe_float(sleeve_json.get("sleeve_trading_sharpe_ratio")) or 0.0)
    if "sleeve_ruin_guard_flag" not in row:
        dd = _safe_float(row.get("sleeve_trading_max_drawdown_pct"))
        events = _safe_float(row.get("sleeve_settlement_event_count"))
        row["sleeve_ruin_guard_flag"] = bool(
            row.get("sleeve_margin_active_any", False)
            or (dd is not None and dd >= 99.0)
        )
        row["sleeve_low_event_guard_flag"] = bool(events is not None and events < 50.0)
        row["sleeve_trading_sharpe_ruin_guarded"] = (
            None
            if row["sleeve_ruin_guard_flag"] or row["sleeve_low_event_guard_flag"]
            else row.get("sleeve_trading_sharpe_ratio")
        )
    return row


def _scan_main_rows() -> List[Dict[str, Any]]:
    specs = [
        ("main", "mappo_marl", PROJECT_ROOT / "batch_tier_phase_runs" / "prototype5_mappo_marl_final_v1", "tier1"),
    ]
    rows: List[Dict[str, Any]] = []
    for category, arm, base, tier_dir in specs:
        for region, eval_dir in [("original", "evaluations_2025"), ("v2", "evaluations_2025_v2")]:
            for seed in SEEDS:
                result_json = _latest_json(base / f"seed{seed}" / eval_dir / tier_dir)
                if result_json is not None:
                    rows.append(_row_from_json(category=category, arm=arm, region=region, seed=seed, result_json=result_json))
    return rows


def _scan_mechanism_rows() -> List[Dict[str, Any]]:
    base = ABLATIONS_ROOT / "batch_tier_phase_runs" / "prototype5_mechanism_ablations_final_v1"
    rows: List[Dict[str, Any]] = []
    if not base.is_dir():
        return rows
    for arm_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for region, eval_dir in [("original", "evaluations_2025"), ("v2", "evaluations_2025_v2")]:
            for seed in SEEDS:
                result_json = _latest_json(arm_dir / f"seed{seed}" / eval_dir / "tier1_forecast_utilization")
                if result_json is not None:
                    rows.append(
                        _row_from_json(
                            category="mechanism_ablation",
                            arm=arm_dir.name,
                            region=region,
                            seed=seed,
                            result_json=result_json,
                        )
                    )
    return rows


def _scan_corruption_rows() -> List[Dict[str, Any]]:
    bases = [
        (
            ABLATIONS_ROOT / "results" / "prototype5_forecast_corruption_final_v1",
            ["original", "v2"],
            "forecast_corruption",
        ),
    ]
    rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for base, regions, category in bases:
        if not base.is_dir():
            continue
        for mode in ["zero_edge", "shuffle", "sign_flip"]:
            for region_dir in regions:
                region = "v2" if region_dir == "unseendata_v2" else region_dir
                for seed in SEEDS:
                    key = (category, mode, region, seed)
                    if key in seen:
                        continue
                    result_json = _latest_json(base / mode / region_dir / f"seed{seed}")
                    if result_json is not None:
                        seen.add(key)
                        rows.append(
                            _row_from_json(
                                category=category,
                                arm=mode,
                                region=region,
                                seed=seed,
                                result_json=result_json,
                            )
                        )
    return rows


def _scan_final_baselines() -> List[Dict[str, Any]]:
    path = (
        PROJECT_ROOT
        / "baseline_results"
        / "prototype5_final_paper_baselines_v1"
        / "final_paper_baseline_summary_latest.csv"
    )
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: Dict[str, Any] = {
                "category": "final_baseline",
                "arm": raw.get("baseline", ""),
                "method": raw.get("method", ""),
                "region": "v2" if raw.get("region") == "unseendata_v2" else raw.get("region", ""),
                "seed": "",
                "status": raw.get("status", ""),
                "result_json": "",
                "env_log": "",
                "has_json_sleeve_metrics": True,
                "has_env_log": False,
                "sleeve_metrics_source": "baseline_csv",
                "needs_sleeve_rerun": False,
                "fund_return_pct": _safe_float(raw.get("total_return_pct")),
                "fund_sharpe": _safe_float(raw.get("total_nav_sharpe_ratio")),
                "fund_drawdown_pct": _safe_float(raw.get("max_drawdown_pct")),
                "sleeve_trading_return_pct": _safe_float(raw.get("sleeve_trading_return_pct")),
                "sleeve_trading_sharpe_ratio": _safe_float(raw.get("sleeve_trading_sharpe_ratio")),
                "sleeve_trading_ruined": str(raw.get("sleeve_trading_ruined", "")).strip().lower()
                in {"1", "true", "yes"},
                "sleeve_trading_max_drawdown_pct": _safe_float(raw.get("sleeve_trading_max_drawdown_pct")),
                # Deterministic baselines do not operate the battery agent, so
                # their trading-sleeve path is already the investor-contract path.
                "investor_contract_return_pct": _safe_float(raw.get("sleeve_trading_return_pct")),
                "investor_contract_daily_hac7_sharpe_ratio": _safe_float(
                    raw.get("sleeve_trading_sharpe_ratio")
                ),
                "investor_contract_max_drawdown_pct": _safe_float(
                    raw.get("sleeve_trading_max_drawdown_pct")
                ),
                "investor_contract_ruined": str(
                    raw.get("sleeve_trading_ruined", "")
                ).strip().lower()
                in {"1", "true", "yes"},
                "sleeve_mean_abs_exposure_dkk": _safe_float(raw.get("sleeve_mean_abs_exposure_dkk")),
                "sleeve_max_abs_exposure_dkk": _safe_float(raw.get("sleeve_max_abs_exposure_dkk")),
                "sleeve_total_trading_costs_usd": _safe_float(raw.get("total_transaction_costs_usd")),
                "sleeve_total_volume_transaction_fees_usd": _safe_float(
                    raw.get("total_volume_transaction_fees_usd")
                ),
                "sleeve_total_market_access_fees_usd": _safe_float(
                    raw.get("total_market_access_fees_usd")
                ),
                "sleeve_total_impact_costs_usd": _safe_float(raw.get("total_market_impact_cost_usd")),
                "sleeve_total_collateral_funding_costs_usd": _safe_float(
                    raw.get("total_collateral_funding_cost_usd")
                ),
            }
            dd = _safe_float(row.get("sleeve_trading_max_drawdown_pct"))
            row["sleeve_ruin_guard_flag"] = bool(
                row.get("sleeve_trading_ruined", False)
                or (dd is not None and dd >= 99.0)
            )
            row["sleeve_low_event_guard_flag"] = False
            row["sleeve_trading_sharpe_ruin_guarded"] = (
                None if row["sleeve_ruin_guard_flag"] else row["sleeve_trading_sharpe_ratio"]
            )
            rows.append(row)
    return rows


def _scan_temporal_baselines() -> List[Dict[str, Any]]:
    base = PROJECT_ROOT / "baseline_results" / "prototype5_temporal_baselines_v1"
    rows: List[Dict[str, Any]] = []
    if not base.is_dir():
        return rows
    for matched in ["marl_matched", "focal_matched"]:
        for region_dir in ["original", "unseendata_v2"]:
            path = base / matched / region_dir / "temporal_price_baseline_summary.csv"
            if not path.is_file():
                continue
            with path.open("r", newline="", encoding="utf-8") as handle:
                for raw in csv.DictReader(handle):
                    rule = raw.get("strategy") or raw.get("baseline") or raw.get("method") or raw.get("rule") or "temporal_rule"
                    row: Dict[str, Any] = {
                        "category": "temporal_baseline",
                        "arm": f"{matched}:{rule}",
                        "region": "v2" if region_dir == "unseendata_v2" else region_dir,
                        "seed": "",
                        "status": "completed",
                        "result_json": "",
                        "env_log": "",
                        "has_json_sleeve_metrics": True,
                        "has_env_log": False,
                        "sleeve_metrics_source": "temporal_csv",
                        "needs_sleeve_rerun": False,
                        "sleeve_trading_return_pct": _safe_float(
                            raw.get("return_pct") or raw.get("total_return_pct") or raw.get("sleeve_return_pct")
                        ),
                        "sleeve_trading_sharpe_ratio": _safe_float(
                            raw.get("daily_hac7_sharpe") or raw.get("sleeve_trading_sharpe_ratio") or raw.get("sharpe_ratio")
                        ),
                        "sleeve_trading_daily_hac7_nonannual_365": _safe_float(raw.get("daily_hac7_nonannual_365")),
                        "sleeve_trading_daily_hac7_annualized_252": _safe_float(raw.get("daily_hac7_annualized_252")),
                        "sleeve_trading_max_drawdown_pct": _safe_float(
                            raw.get("max_drawdown_pct") or raw.get("drawdown_pct") or raw.get("sleeve_max_drawdown_pct")
                        ),
                        # Temporal rules do not operate the battery agent.
                        "investor_contract_return_pct": _safe_float(
                            raw.get("return_pct") or raw.get("total_return_pct") or raw.get("sleeve_return_pct")
                        ),
                        "investor_contract_daily_hac7_sharpe_ratio": _safe_float(
                            raw.get("daily_hac7_sharpe")
                            or raw.get("sleeve_trading_sharpe_ratio")
                            or raw.get("sharpe_ratio")
                        ),
                        "investor_contract_max_drawdown_pct": _safe_float(
                            raw.get("max_drawdown_pct")
                            or raw.get("drawdown_pct")
                            or raw.get("sleeve_max_drawdown_pct")
                        ),
                    }
                    dd = _safe_float(row.get("sleeve_trading_max_drawdown_pct"))
                    row["sleeve_ruin_guard_flag"] = bool(dd is not None and dd >= 99.0)
                    row["investor_contract_ruined"] = row["sleeve_ruin_guard_flag"]
                    row["sleeve_low_event_guard_flag"] = False
                    row["sleeve_trading_sharpe_ruin_guarded"] = (
                        None if row["sleeve_ruin_guard_flag"] else row["sleeve_trading_sharpe_ratio"]
                    )
                    rows.append(row)
    return rows


def _numeric_values(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        value = _safe_float(row.get(key))
        if value is not None:
            out.append(value)
    return out


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed_text: str,
    resamples: int = 10000,
) -> Tuple[float, float]:
    seed_bytes = hashlib.sha256(seed_text.encode("utf-8")).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
    indices = rng.integers(
        0,
        values.size,
        size=(int(resamples), values.size),
    )
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("category", "")), str(row.get("arm", "")), str(row.get("region", "")))].append(row)
    metrics = [
        "fund_return_pct",
        "fund_sharpe",
        "fund_drawdown_pct",
        "sleeve_trading_return_pct",
        "sleeve_trading_sharpe_ratio",
        "sleeve_trading_sharpe_ruin_guarded",
        "sleeve_trading_daily_hac7_sharpe_ratio",
        "sleeve_trading_daily_hac7_nonannual_365",
        "sleeve_trading_daily_hac7_annualized_252",
        "sleeve_trading_30d_sharpe_ratio",
        "sleeve_trading_max_drawdown_pct",
        "investor_contract_return_pct",
        "investor_contract_daily_hac7_sharpe_ratio",
        "investor_contract_daily_hac7_zero_rf_sharpe_ratio",
        "investor_contract_annualized_volatility",
        "investor_contract_max_drawdown_pct",
        "sleeve_settlement_pnl_hit_rate",
        "sleeve_settlement_expected_shortfall_1pct_dkk",
        "sleeve_settlement_expected_shortfall_5pct_dkk",
        "sleeve_settlement_worst1pct_loss_share",
        "sleeve_settlement_best1pct_gain_share",
        "sleeve_mean_abs_exposure_dkk",
        "sleeve_exposure_turnover_x_sleeve_per_day",
        "sleeve_friction_share_of_gross_gain",
        "sleeve_total_volume_transaction_fees_usd",
        "sleeve_total_market_access_fees_usd",
        "sleeve_total_impact_costs_usd",
        "sleeve_total_collateral_funding_costs_usd",
        "sleeve_liquidity_bind_share",
        "sleeve_liquidity_max_participation",
        "sleeve_liquidity_tail_impact_mean_multiplier",
        "sleeve_collateral_required_mean_dkk",
    ]
    summary_rows: List[Dict[str, Any]] = []
    for (category, arm, region), group_rows in sorted(groups.items()):
        out: Dict[str, Any] = {
            "category": category,
            "arm": arm,
            "region": region,
            "n_rows": len(group_rows),
            "n_with_sleeve_metrics": sum(1 for row in group_rows if row.get("sleeve_metrics_source") != "missing"),
            "n_needs_sleeve_rerun": sum(1 for row in group_rows if row.get("needs_sleeve_rerun")),
            "n_ruin_guarded": sum(1 for row in group_rows if row.get("sleeve_ruin_guard_flag")),
        }
        seed_ids = {
            str(row.get("seed", "")).strip()
            for row in group_rows
            if str(row.get("seed", "")).strip()
        }
        out["n_training_seeds"] = len(seed_ids)
        out["uncertainty_unit"] = (
            "training_seed"
            if len(seed_ids) > 1
            else "descriptive_single_path_or_deterministic"
        )
        for metric in metrics:
            vals = _numeric_values(group_rows, metric)
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            out[f"{metric}_mean"] = float(np.mean(arr))
            out[f"{metric}_median"] = float(np.median(arr))
            out[f"{metric}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            out[f"{metric}_min"] = float(np.min(arr))
            out[f"{metric}_max"] = float(np.max(arr))
            if len(seed_ids) > 1 and arr.size > 1:
                lo, hi = _bootstrap_mean_ci(
                    arr,
                    seed_text=f"{category}|{arm}|{region}|{metric}",
                )
                out[f"{metric}_seed_bootstrap_ci95_low"] = lo
                out[f"{metric}_seed_bootstrap_ci95_high"] = hi
        summary_rows.append(out)
    return summary_rows


def _fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    priority = [
        "category",
        "arm",
        "method",
        "region",
        "seed",
        "status",
        "fund_return_pct",
        "fund_sharpe",
        "fund_drawdown_pct",
        "sleeve_trading_return_pct",
        "sleeve_trading_sharpe_ratio",
        "sleeve_trading_sharpe_ruin_guarded",
        "sleeve_trading_daily_hac7_nonannual_365",
        "sleeve_trading_daily_hac7_annualized_252",
        "sleeve_trading_30d_sharpe_ratio",
        "sleeve_trading_max_drawdown_pct",
        "investor_contract_return_pct",
        "investor_contract_daily_hac7_sharpe_ratio",
        "investor_contract_daily_hac7_zero_rf_sharpe_ratio",
        "investor_contract_annualized_volatility",
        "investor_contract_max_drawdown_pct",
        "investor_contract_ruined",
        "sleeve_settlement_pnl_hit_rate",
        "sleeve_settlement_expected_shortfall_1pct_dkk",
        "sleeve_settlement_expected_shortfall_5pct_dkk",
        "sleeve_settlement_worst1pct_loss_share",
        "sleeve_settlement_best1pct_gain_share",
        "sleeve_mean_abs_exposure_dkk",
        "sleeve_max_abs_exposure_dkk",
        "sleeve_exposure_turnover_x_sleeve_per_day",
        "sleeve_friction_share_of_gross_gain",
        "sleeve_total_trading_costs_usd",
        "sleeve_total_volume_transaction_fees_usd",
        "sleeve_total_market_access_fees_usd",
        "sleeve_total_impact_costs_usd",
        "sleeve_total_collateral_funding_costs_usd",
        "sleeve_liquidity_bind_share",
        "sleeve_liquidity_max_participation",
        "sleeve_liquidity_mean_participation",
        "sleeve_liquidity_tail_impact_mean_multiplier",
        "sleeve_collateral_required_mean_dkk",
        "sleeve_ruin_guard_flag",
        "sleeve_low_event_guard_flag",
        "sleeve_metrics_source",
        "has_json_sleeve_metrics",
        "has_env_log",
        "needs_sleeve_rerun",
        "result_json",
        "env_log",
    ]
    keys = {key for row in rows for key in row.keys()}
    return [key for key in priority if key in keys] + sorted(keys - set(priority))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _rerun_command_for_missing(row: Dict[str, Any], python_exe: str) -> Optional[str]:
    if row.get("category") != "mechanism_ablation" or not row.get("needs_sleeve_rerun"):
        return None
    arm = str(row.get("arm"))
    seed = str(row.get("seed"))
    region = str(row.get("region"))
    model_dir = (
        Path("Ablations")
        / "batch_tier_phase_runs"
        / "prototype5_mechanism_ablations_final_v1"
        / arm
        / f"seed{seed}"
        / f"tier1_forecast_utilization_seed{seed}"
    )
    output_dir = model_dir.parent / _eval_dir_name(region) / "tier1_forecast_utilization"
    cmd = [
        python_exe,
        "evaluation.py",
        *COMMON_EVAL_ARGS,
        "--tier1_dir",
        str(model_dir),
        "--seed",
        seed,
        "--output_dir",
        str(output_dir),
        "--eval_data",
        _eval_data_arg(region),
        *prior_args_for_arm(arm),
        *_settlement_eval_args(region),
        *_liquidity_eval_args(region),
    ]
    return subprocess.list2cmdline(cmd)


def _write_rerun_script(path: Path, rows: List[Dict[str, Any]], python_exe: str) -> int:
    commands = []
    for row in rows:
        cmd = _rerun_command_for_missing(row, python_exe)
        if cmd:
            commands.append((row, cmd))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("$ErrorActionPreference = 'Stop'\n")
        handle.write("$env:PYTHONUTF8 = '1'\n")
        handle.write("$env:PYTHONIOENCODING = 'utf-8'\n")
        handle.write(f"Push-Location -LiteralPath {subprocess.list2cmdline([str(PROJECT_ROOT)])}\n")
        handle.write("try {\n")
        for row, cmd in commands:
            handle.write(
                f"  Write-Host \"[RUN] {row.get('arm')} seed {row.get('seed')} {row.get('region')} sleeve-log eval\"\n"
            )
            handle.write(f"  {cmd}\n")
        handle.write("} finally {\n")
        handle.write("  Pop-Location\n")
        handle.write("}\n")
    return len(commands)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="Prototype root directory.")
    parser.add_argument("--output-dir", default="results/detailed_metrics")
    parser.add_argument("--python", default="python")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global PROJECT_ROOT, ABLATIONS_ROOT
    PROJECT_ROOT = Path(args.root).resolve()
    ABLATIONS_ROOT = PROJECT_ROOT / "Ablations"
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    rows.extend(_scan_main_rows())
    rows.extend(_scan_mechanism_rows())
    rows.extend(_scan_corruption_rows())
    rows.extend(_scan_final_baselines())
    rows.extend(_scan_temporal_baselines())

    summary_rows = _summarize(rows)
    missing_rows = [row for row in rows if row.get("needs_sleeve_rerun")]

    all_csv = output_dir / "all_detailed_metrics.csv"
    summary_csv = output_dir / "summary_by_group.csv"
    missing_csv = output_dir / "missing_sleeve_metrics.csv"
    json_path = output_dir / "all_detailed_metrics.json"
    rerun_script = output_dir / "rerun_missing_sleeve_evals.ps1"

    _write_csv(all_csv, rows)
    _write_csv(summary_csv, summary_rows)
    _write_csv(missing_csv, missing_rows)
    _write_json(
        json_path,
        {
            "project_root": str(PROJECT_ROOT),
            "n_rows": len(rows),
            "n_summary_rows": len(summary_rows),
            "n_missing_sleeve_rows": len(missing_rows),
            "rows": rows,
            "summary_rows": summary_rows,
        },
    )
    n_commands = _write_rerun_script(rerun_script, rows, str(args.python))

    print(f"Wrote: {all_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {missing_csv}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {rerun_script} ({n_commands} commands)")
    print("")
    print(f"Rows: {len(rows)}")
    print(f"Groups: {len(summary_rows)}")
    print(f"Rows missing sleeve metrics/logs: {len(missing_rows)}")
    if missing_rows:
        print("Run this if you need sleeve verification for missing mechanism ablations:")
        print(f"  powershell -ExecutionPolicy Bypass -File \"{rerun_script}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

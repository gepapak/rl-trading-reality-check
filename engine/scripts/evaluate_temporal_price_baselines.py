#!/usr/bin/env python3
"""Evaluate simple temporal price-direction baselines under the Prototype3 protocol.

These baselines deliberately use only past prices:

- ``prev_hour_momentum``: sign of the previous 1-hour price move.
- ``same_hour_yesterday``: sign of yesterday's same-hour 1-hour move.
- ``combined_agreement``: trades only when both rules agree.

The financial accounting mirrors the frozen same-delivery protocol:
current-price entry, delayed publication of the same-delivery settlement,
explicit MWh-volume payoff, transaction/spread/impact costs, fixed initial
sleeve sizing, and open contracts carried at cost until settlement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market_fee_protocol import base_execution_fee_components, market_access_fee_for_step


RULES = ("prev_hour_momentum", "same_hour_yesterday", "combined_agreement")


def _force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_timestamps(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    try:
        return parsed.dt.tz_convert(None)
    except Exception:
        return parsed


def _settlement_prices(data: pd.DataFrame, base_prices: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    basis_path_raw = str(getattr(args, "settlement_price_data", "") or "").strip()
    if not basis_path_raw:
        return np.asarray(base_prices, dtype=np.float64)
    basis_path = _path(basis_path_raw)
    basis_df = pd.read_csv(basis_path, low_memory=False)
    price_col = str(getattr(args, "settlement_price_column", "price") or "price")
    ts_col = str(getattr(args, "settlement_timestamp_column", "timestamp") or "timestamp")
    if price_col not in basis_df.columns:
        raise ValueError(f"Settlement price column '{price_col}' not found in {basis_path}")
    if ts_col not in basis_df.columns or "timestamp" not in data.columns:
        peer = pd.to_numeric(basis_df[price_col], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
        if len(peer) < len(base_prices):
            peer = np.pad(peer, (0, len(base_prices) - len(peer)), mode="edge")
        peer = peer[: len(base_prices)]
    else:
        source = pd.DataFrame(
            {
                "timestamp": _parse_timestamps(basis_df[ts_col]),
                "peer_price": pd.to_numeric(basis_df[price_col], errors="coerce"),
            }
        ).dropna(subset=["timestamp", "peer_price"])
        target = pd.DataFrame(
            {
                "timestamp": _parse_timestamps(data["timestamp"]),
                "_row": np.arange(len(data), dtype=np.int64),
            }
        ).dropna(subset=["timestamp"])
        source = source.sort_values("timestamp").groupby("timestamp", as_index=False)["peer_price"].last()
        target = target.sort_values("timestamp")
        merged = pd.merge(target, source, on="timestamp", how="left").sort_values("_row")
        coverage = float(merged["peer_price"].notna().mean())
        if coverage < 0.80:
            raise ValueError(f"Settlement price timestamp coverage too low: {coverage:.1%}")
        peer = merged["peer_price"].ffill().bfill().to_numpy(dtype=np.float64)
    settlement_mode = (
        str(getattr(args, "settlement_price_mode", "cross_zone_basis") or "cross_zone_basis")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if settlement_mode in {"external", "external_series", "realized", "real_settlement"}:
        clip_min = float(getattr(args, "settlement_clip_min_dkk", -111750.0))
        clip_max = float(getattr(args, "settlement_clip_max_dkk", 111750.0))
        return np.clip(peer, clip_min, clip_max)

    peer = np.clip(peer, 10.0, 2000.0)
    base = np.asarray(base_prices, dtype=np.float64)
    raw_basis = pd.Series(peer - base)
    mode = str(getattr(args, "basis_centering_mode", "rolling_median") or "rolling_median").strip().lower()
    if mode == "none":
        center = pd.Series(np.zeros(len(raw_basis), dtype=np.float64))
    elif mode == "expanding_median":
        center = raw_basis.expanding(min_periods=1).median().shift(1)
        center.iloc[0] = raw_basis.iloc[0]
    else:
        window = int(max(1, getattr(args, "basis_centering_window_steps", 4320)))
        center = raw_basis.rolling(window=window, min_periods=1).median().shift(1)
        center.iloc[0] = raw_basis.iloc[0]
    component = float(max(getattr(args, "basis_scale", 0.0), 0.0)) * (raw_basis - center.ffill().bfill()).fillna(0.0)
    return np.clip(base + component.to_numpy(dtype=np.float64), 10.0, 2000.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _annual_rate_to_step_rate(rate: float, periods: float) -> float:
    periods = float(max(periods, 1.0))
    return float((1.0 + float(rate)) ** (1.0 / periods) - 1.0)


def _series_returns(values: Iterable[float]) -> np.ndarray:
    vals = np.asarray(list(values), dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.asarray([], dtype=np.float64)
    prev = vals[:-1]
    curr = vals[1:]
    mask = np.abs(prev) > 1e-12
    out = np.zeros_like(curr, dtype=np.float64)
    out[mask] = curr[mask] / prev[mask] - 1.0
    return out[np.isfinite(out)]


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
        weight = 1.0 - lag / (q + 1.0)
        total += 2.0 * weight * rho
    return float(max(total, 1e-12))


def _subsample_path(values: np.ndarray, step: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return vals
    idx = np.arange(0, vals.size, int(max(step, 1)), dtype=int)
    if idx.size < 2 and vals.size >= 2:
        idx = np.asarray([0, vals.size - 1], dtype=int)
    return vals[idx]


def _daily_sharpe_stats(
    equity: np.ndarray,
    *,
    initial_sleeve: float,
    annual_risk_free_rate: float,
) -> Dict[str, float]:
    values = np.asarray(equity, dtype=np.float64).reshape(-1)
    complete_days = int(values.size // 144)
    daily_path = np.concatenate(
        (
            np.asarray([float(initial_sleeve)], dtype=np.float64),
            values[143 : complete_days * 144 : 144],
        )
    )
    returns = _series_returns(daily_path)
    if returns.size < 2:
        return {
            "daily_sharpe": 0.0,
            "daily_hac7_sharpe": 0.0,
            "daily_hac7_vif": 1.0,
            "daily_hac7_annualized_252": 0.0,
            "daily_hac7_nonannual_365": 0.0,
            "daily_n_returns": float(returns.size),
            "daily_mean_return": 0.0,
            "daily_volatility": 0.0,
        }
    std = float(np.std(returns, ddof=1))
    if std <= 1e-12 or not math.isfinite(std):
        return {
            "daily_sharpe": 0.0,
            "daily_hac7_sharpe": 0.0,
            "daily_hac7_vif": 1.0,
            "daily_hac7_annualized_252": 0.0,
            "daily_hac7_nonannual_365": 0.0,
            "daily_n_returns": float(returns.size),
            "daily_mean_return": float(np.mean(returns)),
            "daily_volatility": 0.0,
        }
    rf = _annual_rate_to_step_rate(float(annual_risk_free_rate), 365.25)
    daily_sharpe = float(np.mean(returns - rf) / std * math.sqrt(365.25))
    vif = _hac_variance_inflation(returns, min(7, max(returns.size - 2, 0)))
    daily_hac = float(daily_sharpe / math.sqrt(vif)) if vif > 0.0 else 0.0
    daily_hac_nonannual = float(daily_hac / math.sqrt(365.25))
    return {
        "daily_sharpe": daily_sharpe,
        "daily_hac7_sharpe": daily_hac,
        "daily_hac7_vif": float(vif),
        "daily_hac7_annualized_252": float(daily_hac_nonannual * math.sqrt(252.0)),
        "daily_hac7_nonannual_365": daily_hac_nonannual,
        "daily_n_returns": float(returns.size),
        "daily_mean_return": float(np.mean(returns)),
        "daily_volatility": std,
    }


def _daily_returns_frame(
    timestamps: pd.Series,
    equity: np.ndarray,
    *,
    initial_sleeve: float,
) -> pd.DataFrame:
    complete_days = int(len(equity) // 144)
    idx = np.arange(143, complete_days * 144, 144, dtype=int)
    daily_equity = np.concatenate(
        (
            np.asarray([float(initial_sleeve)], dtype=np.float64),
            np.asarray(equity, dtype=np.float64)[idx],
        )
    )
    end_ts = pd.to_datetime(timestamps.iloc[idx], errors="coerce") if len(timestamps) else pd.Series(idx)
    daily_returns = _series_returns(daily_equity)
    return pd.DataFrame(
        {
            "timestamp": [str(x) for x in end_ts],
            "equity_dkk": daily_equity[1:],
            "daily_return": daily_returns,
        }
    )


def _max_drawdown(
    equity: np.ndarray,
    *,
    initial_sleeve: Optional[float] = None,
) -> float:
    arr = np.asarray(equity, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    if initial_sleeve is not None:
        arr = np.concatenate(
            (
                np.asarray([float(initial_sleeve)], dtype=np.float64),
                arr,
            )
        )
    peak = np.maximum.accumulate(arr)
    dd = np.where(peak > 0.0, (peak - arr) / peak, 0.0)
    return float(np.max(dd)) if dd.size else 0.0


def _market_volume_mwh(data: pd.DataFrame, t: int, args: argparse.Namespace) -> float:
    min_volume = float(max(getattr(args, "liquidity_min_volume_mwh", 100.0), 1e-9))
    multiplier = float(max(getattr(args, "liquidity_volume_multiplier", 1.0), 1e-9))
    source = (
        str(getattr(args, "liquidity_volume_source", "load") or "load")
        .strip()
        .lower()
        .replace("-", "_")
    )
    idx = int(max(0, min(int(t), len(data) - 1)))
    horizon_hours = float(max(float(args.time_step_hours) * float(max(args.horizon_steps, 1)), float(args.time_step_hours)))

    def col(name: str) -> float:
        if name not in data.columns:
            return 0.0
        value = _safe_float(data[name].iloc[idx], 0.0)
        return float(max(value, 0.0))

    generation = col("wind") + col("solar") + col("hydro")
    if source == "impact_volume":
        value = col("_impact_market_volume_mwh")
    elif source == "generation":
        value = generation * horizon_hours
    elif source == "max_load_generation":
        value = max(col("load"), generation, 0.0) * horizon_hours
    else:
        value = col("load") * horizon_hours
        if value <= 0.0:
            value = generation * horizon_hours
    return float(max(value * multiplier, min_volume))


def _attach_impact_volume(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    source = (
        str(getattr(args, "liquidity_volume_source", "load") or "load")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if source != "impact_volume":
        return data
    path_raw = str(getattr(args, "impact_volume_data", "") or "").strip()
    if not path_raw:
        raise ValueError("liquidity_volume_source='impact_volume' requires --impact_volume_data")
    path = Path(path_raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    volume_df = pd.read_csv(path, low_memory=False)
    ts_col = str(getattr(args, "impact_volume_timestamp_column", "timestamp") or "timestamp")
    vol_col = str(getattr(args, "impact_volume_column", "market_volume_mwh") or "market_volume_mwh")
    if ts_col not in volume_df.columns or vol_col not in volume_df.columns:
        raise ValueError(f"Impact volume file {path} must contain {ts_col!r} and {vol_col!r}")
    aligned = pd.merge_asof(
        pd.DataFrame(
            {
                "_row": np.arange(len(data), dtype=np.int64),
                "timestamp": _parse_timestamps(data["timestamp"]),
            }
        ).sort_values("timestamp"),
        pd.DataFrame(
            {
                "timestamp": _parse_timestamps(volume_df[ts_col]),
                "_impact_market_volume_mwh": pd.to_numeric(volume_df[vol_col], errors="coerce"),
            }
        ).dropna(subset=["timestamp", "_impact_market_volume_mwh"]).sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(minutes=float(getattr(args, "impact_volume_max_staleness_min", 90.0))),
    ).sort_values("_row")
    if aligned["_impact_market_volume_mwh"].isna().any():
        missing = int(aligned["_impact_market_volume_mwh"].isna().sum())
        raise ValueError(f"Impact volume data missing/stale for {missing} temporal baseline rows: {path}")
    out = data.copy()
    out["_impact_market_volume_mwh"] = aligned["_impact_market_volume_mwh"].to_numpy(dtype=float)
    return out


def _cap_notional_by_liquidity(notional: float, data: pd.DataFrame, t: int, args: argparse.Namespace) -> tuple[float, Dict[str, float]]:
    cap_fraction = float(max(getattr(args, "liquidity_participation_cap_fraction", 0.0), 0.0))
    requested_volume = abs(float(notional)) / max(float(args.payoff_reference_price), 1e-9)
    observed_step = int(t) - int(max(getattr(args, "horizon_steps", 6), 1)) - 1
    if cap_fraction <= 0.0:
        market_volume = 0.0
    elif observed_step >= 0:
        market_volume = _market_volume_mwh(data, observed_step, args)
    else:
        min_volume = float(max(getattr(args, "liquidity_min_volume_mwh", 100.0), 1e-9))
        multiplier = float(max(getattr(args, "liquidity_volume_multiplier", 1.0), 1e-9))
        market_volume = min_volume * multiplier
    cap_volume = market_volume * cap_fraction if cap_fraction > 0.0 else 0.0
    if cap_fraction > 0.0 and requested_volume > cap_volume > 0.0:
        scale = float(np.clip(cap_volume / max(requested_volume, 1e-9), 0.0, 1.0))
    else:
        scale = 1.0
    executed_volume = requested_volume * scale
    return float(notional) * scale, {
        "liquidity_market_volume_mwh": float(market_volume),
        "liquidity_volume_cap_mwh": float(cap_volume),
        "liquidity_requested_volume_mwh": float(requested_volume),
        "liquidity_executed_volume_mwh": float(executed_volume),
        "liquidity_participation": float(executed_volume / max(market_volume, 1e-9)) if market_volume > 0.0 else 0.0,
        "liquidity_scale": float(scale),
    }


def _tail_impact_multiplier(entry_price: float, settlement_price: float, args: argparse.Namespace) -> tuple[float, float]:
    base = float(max(getattr(args, "liquidity_tail_impact_multiplier", 1.0), 0.0))
    spread = abs(float(settlement_price) - float(entry_price))
    if base <= 1.0:
        return 1.0, float(spread)
    threshold = float(max(getattr(args, "liquidity_tail_impact_threshold_dkk_per_mwh", 5000.0), 1e-9))
    power = float(max(getattr(args, "liquidity_tail_impact_power", 1.0), 0.0))
    max_mult = float(max(getattr(args, "liquidity_tail_impact_max_multiplier", base), 1.0))
    stress = max(0.0, spread / threshold - 1.0)
    return float(np.clip(1.0 + (base - 1.0) * (stress ** power), 1.0, max_mult)), float(spread)


def _open_contract_totals(contracts: Iterable[Dict[str, Any]]) -> tuple[float, float]:
    open_notional = 0.0
    open_volume = 0.0
    for contract in contracts:
        notional = abs(_safe_float(contract.get("notional_dkk"), _safe_float(contract.get("notional"), 0.0)))
        open_notional += notional
        open_volume += abs(_safe_float(contract.get("volume_mwh"), 0.0))
    return float(open_notional), float(open_volume)


def _collateral_required(contracts: Iterable[Dict[str, Any]], args: argparse.Namespace) -> tuple[float, float, float]:
    if not bool(getattr(args, "enable_collateral_cash_drag", False)):
        return 0.0, 0.0, 0.0
    open_notional, open_volume = _open_contract_totals(contracts)
    notional_margin = float(max(getattr(args, "collateral_notional_margin_fraction", 0.02), 0.0))
    stress_fraction = float(max(getattr(args, "collateral_stress_loss_fraction", 0.10), 0.0))
    stress_price = float(max(getattr(args, "collateral_stress_price_dkk_per_mwh", 25000.0), 0.0))
    required = float(max(open_notional * notional_margin, open_volume * stress_price * stress_fraction, 0.0))
    return required, float(open_notional), float(open_volume)


def _collateral_funding_cost(required: float, args: argparse.Namespace) -> float:
    rate = float(max(getattr(args, "collateral_funding_rate_annual", 0.05), 0.0))
    return float(max(required, 0.0) * rate * float(args.time_step_hours) / (365.25 * 24.0))


def _roll_cost(
    notional_abs: float,
    budget: float,
    args: argparse.Namespace,
    *,
    traded_volume_mwh: float = 0.0,
    market_volume_mwh: float = 0.0,
    tail_impact_multiplier: float = 1.0,
) -> float:
    notional_abs = float(max(notional_abs, 0.0))
    if notional_abs <= 100.0:
        return 0.0
    fee_components = base_execution_fee_components(
        model=args.market_fee_model,
        abs_notional_dkk=notional_abs,
        abs_volume_mwh=abs(float(traded_volume_mwh)),
        friction_multiplier=float(args.friction_cost_multiplier),
        transaction_cost_bps=float(args.transaction_cost_bps),
        transaction_fixed_cost_dkk=float(args.transaction_fixed_cost),
        transaction_fee_dkk_per_mwh=float(args.transaction_fee_dkk_per_mwh),
    )
    spread_cost = notional_abs * float(args.half_spread_bp) / 10000.0
    impact_cost = 0.0
    if float(args.impact_coef_bp) > 0.0:
        if float(market_volume_mwh) > 0.0:
            ref_notional = max(
                float(market_volume_mwh) * float(args.payoff_reference_price), 1.0
            )
        else:
            ref_notional = max(float(budget), 1.0)
        participation = notional_abs / ref_notional
        impact_bp = (
            float(args.impact_coef_bp)
            * float(max(tail_impact_multiplier, 1.0))
            * (participation ** float(args.impact_exponent))
        )
        impact_cost = notional_abs * impact_bp / 10000.0
    return float(fee_components["base_execution_fee_dkk"] + spread_cost + impact_cost)


def _no_trade_threshold_dkk(
    max_notional_dkk: float,
    liquidity_cap_mwh: float,
    args: argparse.Namespace,
) -> float:
    configured = float(max(getattr(args, "no_trade_threshold", 0.01), 0.0))
    if configured <= 0.0:
        return 0.0
    if configured > 1.0:
        return configured
    reference = float(max(max_notional_dkk, 1.0))
    if float(liquidity_cap_mwh) > 0.0:
        executable_notional = float(liquidity_cap_mwh) * float(args.payoff_reference_price)
        reference = min(reference, max(executable_notional, 1.0))
    return float(configured * reference)


def _payoff_denominator(entry_price: float, reference_price: float) -> float:
    entry_abs = abs(float(entry_price)) if math.isfinite(float(entry_price)) else 0.0
    return float(max(entry_abs, float(reference_price)))


def _payoff_mode(args: argparse.Namespace) -> str:
    return str(getattr(args, "payoff_mode", "mwh_volume") or "mwh_volume").strip().lower().replace("-", "_")


def _contract_pnl_and_unit_payoff(
    contract: Dict[str, Any],
    settlement_price: float,
    payoff_reference_price: float,
    payoff_mode: str,
) -> tuple[float, float]:
    entry = float(contract["entry_price"])
    spread = float(settlement_price) - entry
    if payoff_mode in {"mwh_volume", "mwh"}:
        volume_mwh = float(
            contract.get(
                "volume_mwh",
                float(contract["notional_dkk"]) / max(float(payoff_reference_price), 1e-9),
            )
        )
        return float(volume_mwh * spread), float(spread)
    denom = _payoff_denominator(entry, float(payoff_reference_price))
    realized_return = float(spread / denom)
    return float(contract["notional_dkk"]) * realized_return, realized_return


def _rule_direction(rule: str, prices: np.ndarray, t: int, horizon: int, steps_per_day: int) -> float:
    if rule == "prev_hour_momentum":
        if t - horizon < 0:
            return 0.0
        move = float(prices[t] - prices[t - horizon])
        return float(np.sign(move))

    if rule == "same_hour_yesterday":
        start = t - steps_per_day
        end = start + horizon
        if start < 0 or end >= t or end >= len(prices):
            return 0.0
        move = float(prices[end] - prices[start])
        return float(np.sign(move))

    if rule == "combined_agreement":
        a = _rule_direction("prev_hour_momentum", prices, t, horizon, steps_per_day)
        b = _rule_direction("same_hour_yesterday", prices, t, horizon, steps_per_day)
        if a != 0.0 and a == b:
            return float(a)
        return 0.0

    raise ValueError(f"unknown temporal rule: {rule}")


def evaluate_rule(data: pd.DataFrame, rule: str, args: argparse.Namespace) -> Dict[str, Any]:
    timestamps = (
        pd.to_datetime(data["timestamp"], errors="coerce")
        if "timestamp" in data.columns
        else pd.Series(range(len(data)))
    )
    prices = pd.to_numeric(data["price"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    if len(prices) < int(args.horizon_steps) + 2:
        raise ValueError("evaluation data is too short for the requested horizon")
    settlement_prices = _settlement_prices(data, prices, args)

    horizon = int(args.horizon_steps)
    decision_freq = int(args.decision_freq)
    steps_per_day = int(round(24.0 / max(float(args.time_step_hours), 1e-9)))
    steps_per_day = max(1, steps_per_day)
    initial_sleeve = float(args.initial_fund_dkk) * float(args.financial_allocation)
    max_notional = (
        initial_sleeve
        * float(args.capital_allocation_fraction)
        * float(args.max_position_size)
    )
    target_norm = float(max(0.0, args.rule_target_norm))
    target_notional = min(max_notional, target_norm * max_notional)

    cash = float(initial_sleeve)
    contracts: List[Dict[str, Any]] = []
    equity: List[float] = []
    trade_rows: List[Dict[str, Any]] = []
    settled_rows: List[Dict[str, Any]] = []
    cumulative_pnl = 0.0
    cumulative_cost = 0.0
    cumulative_market_access_fee = 0.0
    cumulative_collateral_funding_cost = 0.0
    liquidity_bind_count = 0
    max_liquidity_participation = 0.0
    max_collateral_required = 0.0
    notional_abs_values: List[float] = []
    margin_active = False
    margin_step = -1
    margin_threshold = initial_sleeve * float(args.maintenance_margin_fraction)

    settlement_floor = float(getattr(args, "settlement_clip_min_dkk", -111750.0))
    settlement_ceiling = float(getattr(args, "settlement_clip_max_dkk", 111750.0))
    collateral_required = 0.0
    collateral_open_notional = 0.0
    collateral_open_volume = 0.0
    for t in range(len(prices)):
        access_fee = market_access_fee_for_step(
            annual_fee_dkk=float(args.annual_market_access_fee_dkk),
            allocation_fraction=float(args.market_access_fee_allocation_fraction),
            time_step_hours=float(args.time_step_hours),
        )
        cash = max(0.0, cash - access_fee)
        cumulative_cost += float(access_fee)
        cumulative_market_access_fee += float(access_fee)
        current_price = float(np.clip(prices[t], -1000.0, 1e9))
        # The decision at t precedes publication of delivery t-horizon. Risk
        # inputs therefore stop at the latest strictly completed delivery.
        observed_step = int(t) - int(horizon) - 1
        if observed_step >= 0:
            observed_entry_price = float(np.clip(prices[observed_step], -1000.0, 1e9))
            observed_settlement_price = float(
                np.clip(settlement_prices[observed_step], settlement_floor, settlement_ceiling)
            )
        else:
            observed_entry_price = current_price
            observed_settlement_price = current_price

        if not margin_active and t % decision_freq == 0 and t + horizon < len(prices):
            direction = _rule_direction(rule, prices, t, horizon, steps_per_day)
            if direction != 0.0 and target_notional > 100.0:
                tradeable_after_collateral = max_notional
                if bool(getattr(args, "enable_collateral_cash_drag", False)):
                    haircut = float(np.clip(getattr(args, "collateral_tradeable_haircut", 1.0), 0.0, 1.0))
                    tradeable_after_collateral = max(0.0, max_notional - haircut * collateral_required)
                notional = float(direction * min(target_notional, tradeable_after_collateral))
                notional, liquidity_diag = _cap_notional_by_liquidity(notional, data, t, args)
                if float(liquidity_diag.get("liquidity_scale", 1.0)) < 0.999999:
                    liquidity_bind_count += 1
                max_liquidity_participation = max(
                    max_liquidity_participation,
                    float(liquidity_diag.get("liquidity_participation", 0.0)),
                )
                threshold_dkk = _no_trade_threshold_dkk(
                    max_notional,
                    float(liquidity_diag.get("liquidity_volume_cap_mwh", 0.0)),
                    args,
                )
                if abs(notional) > max(100.0, threshold_dkk):
                    entry = float(prices[t])
                    tail_multiplier, tail_spread = _tail_impact_multiplier(
                        observed_entry_price, observed_settlement_price, args
                    )
                    cost = _roll_cost(
                        abs(notional),
                        cash,
                        args,
                        traded_volume_mwh=abs(
                            float(notional) / max(float(args.payoff_reference_price), 1e-9)
                        ),
                        market_volume_mwh=float(
                            liquidity_diag.get("liquidity_market_volume_mwh", 0.0)
                        ),
                        tail_impact_multiplier=tail_multiplier,
                    )
                    cash = max(0.0, cash - cost)
                    cumulative_cost += cost
                    notional_abs_values.append(abs(notional))
                    contracts.append(
                        {
                            "open_step": int(t),
                            "delivery_step": int(t),
                            "settle_step": int(t + horizon),
                            "entry_price": current_price,
                            "notional_dkk": notional,
                            "volume_mwh": float(notional / max(float(args.payoff_reference_price), 1e-9)),
                        }
                    )
                    trade_rows.append(
                        {
                            "timestep": int(t),
                            "timestamp": str(timestamps.iloc[t]) if hasattr(timestamps, "iloc") else str(t),
                            "rule": rule,
                            "direction": float(direction),
                            "entry_price": current_price,
                            "notional_dkk": notional,
                            "volume_mwh": float(notional / max(float(args.payoff_reference_price), 1e-9)),
                            "liquidity_participation": float(liquidity_diag.get("liquidity_participation", 0.0)),
                            "liquidity_scale": float(liquidity_diag.get("liquidity_scale", 1.0)),
                            "liquidity_tail_impact_multiplier": float(tail_multiplier),
                            "liquidity_tail_spread_dkk_per_mwh": float(tail_spread),
                            "roll_cost_dkk": float(cost),
                        }
                    )
        # Match the learned-environment event clock: execute the action first,
        # then publish and book contracts whose settlement delay expires at t.
        remaining: List[Dict[str, Any]] = []
        for contract in contracts:
            if int(contract["settle_step"]) > t:
                remaining.append(contract)
                continue
            delivery_step = int(contract.get("delivery_step", contract["open_step"]))
            contract_settlement_price = float(
                np.clip(settlement_prices[delivery_step], settlement_floor, settlement_ceiling)
            )
            entry = float(contract["entry_price"])
            pnl, realized_return = _contract_pnl_and_unit_payoff(
                contract,
                contract_settlement_price,
                float(args.payoff_reference_price),
                _payoff_mode(args),
            )
            cash = max(0.0, cash + pnl)
            cumulative_pnl += pnl
            settled_rows.append(
                {
                    "open_step": int(contract["open_step"]),
                    "delivery_step": delivery_step,
                    "settle_step": int(t),
                    "entry_price": entry,
                    "settlement_price": contract_settlement_price,
                    "notional_dkk": float(contract["notional_dkk"]),
                    "realized_return": float(realized_return),
                    "pnl_dkk": float(pnl),
                    "hit": bool(
                        np.sign(float(contract["notional_dkk"])) == np.sign(realized_return)
                        and realized_return != 0.0
                    ),
                }
            )
        contracts = remaining
        collateral_required, collateral_open_notional, collateral_open_volume = _collateral_required(contracts, args)
        max_collateral_required = max(max_collateral_required, collateral_required)
        funding_cost = _collateral_funding_cost(collateral_required, args)
        if funding_cost > 0.0:
            cash = max(0.0, cash - funding_cost)
            cumulative_collateral_funding_cost += funding_cost

        # Same-delivery contracts are carried at cost; only matured settlement
        # PnL enters the sleeve path.
        sleeve_value = float(cash)
        if (not margin_active) and sleeve_value <= margin_threshold:
            margin_active = True
            margin_step = int(t)
            cash = max(0.0, float(sleeve_value))
            contracts = []
            sleeve_value = float(cash)

        equity.append(float(sleeve_value))

    equity_arr = np.asarray(equity, dtype=np.float64)
    sharpe_stats = _daily_sharpe_stats(
        equity_arr,
        initial_sleeve=initial_sleeve,
        annual_risk_free_rate=float(args.annual_risk_free_rate),
    )
    settled = pd.DataFrame(settled_rows)
    hit_rate = float(settled["hit"].mean()) if not settled.empty else 0.0
    settled_return = settled["realized_return"].to_numpy(dtype=np.float64) if not settled.empty else np.asarray([])
    notional_abs = np.asarray(notional_abs_values, dtype=np.float64)
    result = {
        "rule": rule,
        "status": "completed",
        "rows": int(len(data)),
        "decision_freq": decision_freq,
        "horizon_steps": horizon,
        "initial_trading_sleeve_dkk": float(initial_sleeve),
        "max_protocol_notional_dkk": float(max_notional),
        "rule_target_norm": float(target_norm),
        "target_notional_dkk": float(target_notional),
        "final_trading_sleeve_dkk": float(equity_arr[-1]) if equity_arr.size else float(initial_sleeve),
        "total_return": float(equity_arr[-1] / max(initial_sleeve, 1e-12) - 1.0) if equity_arr.size else 0.0,
        "return_pct": float((equity_arr[-1] / max(initial_sleeve, 1e-12) - 1.0) * 100.0) if equity_arr.size else 0.0,
        "daily_sharpe_raw": float(sharpe_stats["daily_sharpe"]),
        "daily_hac7_sharpe_raw": float(sharpe_stats["daily_hac7_sharpe"]),
        "daily_sharpe": None if margin_active else float(sharpe_stats["daily_sharpe"]),
        "daily_hac7_sharpe": None if margin_active else float(sharpe_stats["daily_hac7_sharpe"]),
        "daily_hac7_vif": float(sharpe_stats["daily_hac7_vif"]),
        "daily_hac7_annualized_252": None if margin_active else float(sharpe_stats["daily_hac7_annualized_252"]),
        "daily_hac7_nonannual_365": None if margin_active else float(sharpe_stats["daily_hac7_nonannual_365"]),
        "daily_n_returns": float(sharpe_stats["daily_n_returns"]),
        "daily_mean_return": float(sharpe_stats["daily_mean_return"]),
        "daily_volatility": float(sharpe_stats["daily_volatility"]),
        "max_drawdown": _max_drawdown(
            equity_arr,
            initial_sleeve=initial_sleeve,
        ),
        "max_drawdown_pct": 100.0 * _max_drawdown(
            equity_arr,
            initial_sleeve=initial_sleeve,
        ),
        "path_convention": "true_initial_plus_complete_day_ends_v1",
        "cumulative_pnl_dkk": float(cumulative_pnl),
        "cumulative_cost_dkk": float(cumulative_cost),
        "cumulative_market_access_fee_dkk": float(cumulative_market_access_fee),
        "market_fee_model": str(args.market_fee_model),
        "transaction_fee_dkk_per_mwh": float(args.transaction_fee_dkk_per_mwh),
        "annual_market_access_fee_dkk": float(args.annual_market_access_fee_dkk),
        "market_access_fee_allocation_fraction": float(args.market_access_fee_allocation_fraction),
        "market_fee_source_id": str(args.market_fee_source_id),
        "cumulative_collateral_funding_cost_dkk": float(cumulative_collateral_funding_cost),
        "open_contracts_left": int(len(contracts)),
        "trades": int(len(trade_rows)),
        "settled_contracts": int(len(settled_rows)),
        "margin_active": bool(margin_active),
        "margin_step": int(margin_step),
        "maintenance_margin_fraction": float(args.maintenance_margin_fraction),
        "maintenance_margin_threshold_dkk": float(margin_threshold),
        "directional_hit_rate": hit_rate,
        "mean_settled_return": float(np.mean(settled_return)) if settled_return.size else 0.0,
        "std_settled_return": float(np.std(settled_return, ddof=1)) if settled_return.size > 1 else 0.0,
        "mean_abs_notional_dkk": float(np.mean(notional_abs)) if notional_abs.size else 0.0,
        "max_abs_notional_dkk": float(np.max(notional_abs)) if notional_abs.size else 0.0,
        "liquidity_bind_count": int(liquidity_bind_count),
        "max_liquidity_participation": float(max_liquidity_participation),
        "max_collateral_required_dkk": float(max_collateral_required),
        "final_collateral_open_notional_dkk": float(collateral_open_notional if "collateral_open_notional" in locals() else 0.0),
        "final_collateral_open_volume_mwh": float(collateral_open_volume if "collateral_open_volume" in locals() else 0.0),
        "settlement_price_data": str(getattr(args, "settlement_price_data", "") or ""),
        "basis_scale": float(getattr(args, "basis_scale", 0.0)),
        "rule_description": {
            "prev_hour_momentum": "Long if the previous 1-hour price move was positive; short if negative.",
            "same_hour_yesterday": "Long if yesterday's same-hour 1-hour move was positive; short if negative.",
            "combined_agreement": "Trade only when previous-hour momentum and same-hour-yesterday direction agree.",
        }[rule],
    }
    result["_trade_rows"] = trade_rows
    result["_settled_rows"] = settled_rows
    result["_equity"] = equity_arr
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_data", default="evaluation_dataset_ffill/unseendata.csv")
    parser.add_argument("--output_dir", default="baseline_results/temporal_price_baselines")
    parser.add_argument("--rules", nargs="+", default=list(RULES), choices=list(RULES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decision_freq", type=int, default=6)
    parser.add_argument("--horizon_steps", type=int, default=6)
    parser.add_argument("--time_step_hours", type=float, default=10.0 / 60.0)
    parser.add_argument("--initial_fund_dkk", type=float, default=None)
    parser.add_argument("--financial_allocation", type=float, default=None)
    parser.add_argument("--capital_allocation_fraction", type=float, default=0.60)
    parser.add_argument("--max_position_size", type=float, default=0.10)
    parser.add_argument(
        "--rule_target_norm",
        type=float,
        default=0.125,
        help=(
            "Fraction of the protocol max notional used by the temporal rule. "
            "Default 0.125 gives about 5M DKK notional under Prototype3, matching "
            "the learned policies' mean exposure scale."
        ),
    )
    parser.add_argument("--payoff_reference_price", type=float, default=500.0)
    parser.add_argument(
        "--payoff_mode",
        choices=["entry_price_floor", "mwh_volume"],
        default="mwh_volume",
        help="entry_price_floor uses notional*price_return; mwh_volume uses explicit MWh volume times price spread.",
    )
    parser.add_argument("--annual_risk_free_rate", type=float, default=0.02)
    parser.add_argument("--transaction_cost_bps", type=float, default=None)
    parser.add_argument("--transaction_fixed_cost", type=float, default=None)
    parser.add_argument("--friction_cost_multiplier", type=float, default=1.0)
    parser.add_argument(
        "--market_fee_model",
        choices=["legacy_notional_fixed", "nord_pool_intraday_2026"],
        default="legacy_notional_fixed",
    )
    parser.add_argument("--transaction_fee_dkk_per_mwh", type=float, default=0.0)
    parser.add_argument("--annual_market_access_fee_dkk", type=float, default=0.0)
    parser.add_argument("--market_access_fee_allocation_fraction", type=float, default=1.0)
    parser.add_argument("--market_fee_source_id", default="legacy_unsourced_fixed_fee")
    parser.add_argument("--half_spread_bp", type=float, default=5.0)
    parser.add_argument("--impact_coef_bp", type=float, default=20.0)
    parser.add_argument("--impact_exponent", type=float, default=0.5)
    parser.add_argument("--no_trade_threshold", type=float, default=0.01)
    parser.add_argument("--liquidity_participation_cap_fraction", type=float, default=0.0)
    parser.add_argument("--liquidity_volume_source", choices=["load", "generation", "max_load_generation", "impact_volume"], default="load")
    parser.add_argument("--impact_volume_data", default="")
    parser.add_argument("--impact_volume_column", default="market_volume_mwh")
    parser.add_argument("--impact_volume_unit", choices=["mwh"], default="mwh")
    parser.add_argument("--impact_volume_timestamp_column", default="timestamp")
    parser.add_argument("--impact_volume_max_staleness_min", type=float, default=90.0)
    parser.add_argument("--impact_volume_price_floor_dkk_per_mwh", type=float, default=50.0)
    parser.add_argument("--liquidity_volume_multiplier", type=float, default=1.0)
    parser.add_argument("--liquidity_min_volume_mwh", type=float, default=1.0)
    parser.add_argument("--liquidity_tail_impact_threshold_dkk_per_mwh", type=float, default=5000.0)
    parser.add_argument("--liquidity_tail_impact_multiplier", type=float, default=1.0)
    parser.add_argument("--liquidity_tail_impact_power", type=float, default=1.0)
    parser.add_argument("--liquidity_tail_impact_max_multiplier", type=float, default=10.0)
    parser.add_argument("--enable_collateral_cash_drag", action="store_true")
    parser.add_argument("--collateral_notional_margin_fraction", type=float, default=0.02)
    parser.add_argument("--collateral_stress_loss_fraction", type=float, default=0.10)
    parser.add_argument("--collateral_stress_price_dkk_per_mwh", type=float, default=25000.0)
    parser.add_argument("--collateral_funding_rate_annual", type=float, default=0.05)
    parser.add_argument("--collateral_tradeable_haircut", type=float, default=1.0)
    parser.add_argument("--maintenance_margin_fraction", type=float, default=0.05)
    parser.add_argument("--settlement_price_data", default="")
    parser.add_argument(
        "--settlement_price_mode",
        choices=["cross_zone_basis", "external_series", "external", "realized", "real_settlement"],
        default="cross_zone_basis",
    )
    parser.add_argument("--settlement_price_column", default="price")
    parser.add_argument("--settlement_timestamp_column", default="timestamp")
    parser.add_argument(
        "--settlement_clip_min_dkk",
        type=float,
        default=-111750.0,
        help="Lower technical bound applied to the external settlement series (DKK/MWh).",
    )
    parser.add_argument(
        "--settlement_clip_max_dkk",
        type=float,
        default=111750.0,
        help="Upper technical bound applied to the external settlement series (DKK/MWh).",
    )
    parser.add_argument("--basis_scale", type=float, default=0.50)
    parser.add_argument("--basis_centering_mode", choices=["none", "rolling_median", "expanding_median"], default="rolling_median")
    parser.add_argument("--basis_centering_window_steps", type=int, default=4320)
    return parser.parse_args()


def main() -> int:
    _force_utf8_stdio()
    args = parse_args()
    np.random.seed(int(args.seed))

    from config import EnhancedConfig

    cfg = EnhancedConfig()
    if args.initial_fund_dkk is None:
        args.initial_fund_dkk = _safe_float(getattr(cfg, "init_budget", 0.0), 0.0)
    if args.financial_allocation is None:
        args.financial_allocation = _safe_float(getattr(cfg, "financial_allocation", 0.12), 0.12)
    if args.transaction_cost_bps is None:
        args.transaction_cost_bps = _safe_float(getattr(cfg, "transaction_cost_bps", 0.5), 0.5)
    if args.transaction_fixed_cost is None:
        args.transaction_fixed_cost = _safe_float(getattr(cfg, "transaction_fixed_cost", 0.0), 0.0)

    eval_path = _path(args.eval_data)
    out_dir = _path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(eval_path, low_memory=False)
    data = _attach_impact_volume(data, args)
    if "price" not in data.columns:
        raise SystemExit(f"missing required column 'price' in {eval_path}")

    payload: Dict[str, Any] = {
        "evaluation_type": "temporal_price_direction_baselines",
        "eval_data": str(eval_path),
        "seed": int(args.seed),
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in {"rules"}
        },
        "rules": {},
    }
    summary_rows: List[Dict[str, Any]] = []
    timestamps = pd.to_datetime(data["timestamp"], errors="coerce") if "timestamp" in data.columns else pd.Series(range(len(data)))

    for rule in args.rules:
        metrics = evaluate_rule(data, rule, args)
        trade_rows = metrics.pop("_trade_rows")
        settled_rows = metrics.pop("_settled_rows")
        equity = metrics.pop("_equity")
        payload["rules"][rule] = metrics

        pd.DataFrame(trade_rows).to_csv(out_dir / f"trades_{rule}.csv", index=False)
        pd.DataFrame(settled_rows).to_csv(out_dir / f"settlements_{rule}.csv", index=False)
        _daily_returns_frame(
            timestamps,
            equity,
            initial_sleeve=float(metrics["initial_trading_sleeve_dkk"]),
        ).to_csv(out_dir / f"daily_returns_{rule}.csv", index=False)
        pd.DataFrame(
            {
                "timestep": np.arange(len(equity), dtype=int),
                "timestamp": data["timestamp"] if "timestamp" in data.columns else np.arange(len(equity), dtype=int),
                "trading_sleeve_value_dkk": equity,
            }
        ).to_csv(out_dir / f"equity_path_{rule}.csv", index=False)
        summary_rows.append({k: v for k, v in metrics.items() if not isinstance(v, (dict, list))})

    json_path = out_dir / "temporal_price_baseline_results.json"
    csv_path = out_dir / "temporal_price_baseline_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

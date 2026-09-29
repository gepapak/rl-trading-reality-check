"""Deterministic current-price entry rule baseline.

This is a focused payoff-entry ablation. It does not use the trained RL models.
The rule observes the current price relative to the same-hour previous-day price,
opens a signed hourly energy-price exposure, and settles it after a fixed horizon
using current-price entry by default. The previous-day entry mode is retained
only as an explicit diagnostic because it is not tradable after observing the
current price.
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


def _same_hour_previous_day_price(prices: np.ndarray, idx: int, steps_per_day: int) -> float:
    idx = int(np.clip(idx, 0, len(prices) - 1))
    current = float(np.clip(prices[idx], -1000.0, 1e9))
    prev_idx = idx - steps_per_day
    if prev_idx >= 0:
        value = float(np.clip(prices[prev_idx], -1000.0, 1e9))
        if np.isfinite(value):
            return value

    values = []
    cursor = prev_idx
    while cursor >= 0 and len(values) < 30:
        value = float(np.clip(prices[cursor], -1000.0, 1e9))
        if np.isfinite(value):
            values.append(value)
        cursor -= steps_per_day
    if values:
        return float(np.median(np.asarray(values, dtype=np.float64)))
    return current


def _entry_price(prices: np.ndarray, idx: int, mode: str, steps_per_day: int) -> float:
    current = float(np.clip(prices[int(np.clip(idx, 0, len(prices) - 1))], -1000.0, 1e9))
    mode = str(mode or "current_price").strip().lower()
    if mode == "current_price":
        return current
    if mode == "same_hour_prev_day":
        return _same_hour_previous_day_price(prices, idx, steps_per_day)
    raise ValueError(f"unsupported entry mode: {mode}")


def _acf(values: np.ndarray, lag: int) -> float:
    if values.size <= int(lag) + 1:
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


def _daily_sharpe_stats(
    timestamps: pd.Series,
    equity: np.ndarray,
    *,
    initial_sleeve: float,
    annual_risk_free_rate: float,
) -> Dict[str, float]:
    if len(equity) < 2:
        return {"daily_sharpe": 0.0, "daily_hac7_sharpe": 0.0, "daily_hac7_vif": 1.0, "daily_n_returns": 0.0}
    values = np.asarray(equity, dtype=np.float64).reshape(-1)
    complete_days = int(values.size // 144)
    if complete_days <= 0:
        return {"daily_sharpe": 0.0, "daily_hac7_sharpe": 0.0, "daily_hac7_vif": 1.0, "daily_n_returns": 0.0}
    daily = np.concatenate(
        (
            np.asarray([float(initial_sleeve)], dtype=np.float64),
            values[143 : complete_days * 144 : 144],
        )
    )
    returns = pd.Series(daily).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2:
        return {"daily_sharpe": 0.0, "daily_hac7_sharpe": 0.0, "daily_hac7_vif": 1.0, "daily_n_returns": float(len(returns))}
    std = float(returns.std(ddof=1))
    if std <= 1e-12 or not math.isfinite(std):
        return {"daily_sharpe": 0.0, "daily_hac7_sharpe": 0.0, "daily_hac7_vif": 1.0, "daily_n_returns": float(len(returns))}
    annual_rf = float(annual_risk_free_rate)
    daily_rf = (
        float((1.0 + annual_rf) ** (1.0 / 365.25) - 1.0)
        if math.isfinite(annual_rf) and annual_rf > -1.0
        else 0.0
    )
    daily_sharpe = float((returns.mean() - daily_rf) / std * math.sqrt(365.25))
    arr = returns.to_numpy(dtype=np.float64)
    vif = _hac_variance_inflation(arr, min(7, max(arr.size - 2, 0)))
    return {
        "daily_sharpe": daily_sharpe,
        "daily_hac7_sharpe": float(daily_sharpe / math.sqrt(vif)) if vif > 0.0 else 0.0,
        "daily_hac7_vif": float(vif),
        "daily_n_returns": float(len(returns)),
    }


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
    dd = 1.0 - arr / np.maximum(peak, 1e-12)
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
        return float(max(_safe_float(data[name].iloc[idx], 0.0), 0.0))

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
        raise ValueError(f"Impact volume data missing/stale for {missing} rule baseline rows: {path}")
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
        open_notional += abs(_safe_float(contract.get("notional"), 0.0))
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


def evaluate_mode(data: pd.DataFrame, entry_mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    timestamps = pd.to_datetime(data["timestamp"], errors="coerce") if "timestamp" in data.columns else pd.Series(range(len(data)))
    prices = pd.to_numeric(data["price"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    if len(prices) < int(args.horizon_steps) + 2:
        raise ValueError("evaluation data is too short for the requested horizon")
    settlement_prices = _settlement_prices(data, prices, args)

    steps_per_day = int(round(24.0 / max(float(args.time_step_hours), 1e-9)))
    steps_per_day = max(1, steps_per_day)
    initial_sleeve = float(args.initial_fund_dkk) * float(args.financial_allocation)
    max_notional = (
        initial_sleeve
        * float(args.capital_allocation_fraction)
        * float(args.max_position_size)
    )
    budget = float(initial_sleeve)
    contracts: List[Dict[str, Any]] = []
    equity = []
    trades = []
    settled_returns = []
    hit_count = 0
    active_count = 0
    cumulative_pnl = 0.0
    cumulative_cost = 0.0
    cumulative_market_access_fee = 0.0
    cumulative_collateral_funding_cost = 0.0
    liquidity_bind_count = 0
    max_liquidity_participation = 0.0
    max_collateral_required = 0.0
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
        budget = max(0.0, budget - access_fee)
        cumulative_cost += float(access_fee)
        cumulative_market_access_fee += float(access_fee)
        current = float(np.clip(prices[t], -1000.0, 1e9))
        observed_step = int(t) - int(max(args.horizon_steps, 1)) - 1
        if observed_step >= 0:
            observed_entry_price = float(np.clip(prices[observed_step], -1000.0, 1e9))
            observed_settlement_price = float(
                np.clip(settlement_prices[observed_step], settlement_floor, settlement_ceiling)
            )
        else:
            observed_entry_price = current
            observed_settlement_price = current

        if (
            not margin_active
            and t % int(args.decision_freq) == 0
            and t + int(args.horizon_steps) < len(prices)
        ):
            prev_day = _same_hour_previous_day_price(prices, t, steps_per_day)
            signal = (current - prev_day) / max(abs(prev_day), float(args.signal_denom_floor))
            if abs(signal) < float(args.no_trade_band):
                target_norm = 0.0
            else:
                target_norm = float(np.clip(float(args.rule_gain) * signal, -float(args.rule_exposure_cap), float(args.rule_exposure_cap)))
            notional = target_norm * max_notional
            if abs(notional) > 100.0:
                entry = _entry_price(prices, t, entry_mode, steps_per_day)
                if bool(getattr(args, "enable_collateral_cash_drag", False)):
                    haircut = float(np.clip(getattr(args, "collateral_tradeable_haircut", 1.0), 0.0, 1.0))
                    tradeable_after_collateral = max(0.0, max_notional - haircut * collateral_required)
                    notional = float(np.sign(notional) * min(abs(notional), tradeable_after_collateral))
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
                    tail_multiplier, tail_spread = _tail_impact_multiplier(
                        observed_entry_price, observed_settlement_price, args
                    )
                    cost = _roll_cost(
                        abs(notional),
                        budget,
                        args,
                        traded_volume_mwh=abs(
                            float(notional) / max(float(args.payoff_reference_price), 1e-9)
                        ),
                        market_volume_mwh=float(
                            liquidity_diag.get("liquidity_market_volume_mwh", 0.0)
                        ),
                        tail_impact_multiplier=tail_multiplier,
                    )
                    budget = max(0.0, budget - cost)
                    cumulative_cost += cost
                    contracts.append(
                        {
                            "open_step": int(t),
                            "delivery_step": int(t),
                            "settle_step": int(t + int(args.horizon_steps)),
                            "entry_price": float(entry),
                            "notional": float(notional),
                            "volume_mwh": float(notional / max(float(args.payoff_reference_price), 1e-9)),
                            "signal": float(signal),
                            "liquidity_participation": float(liquidity_diag.get("liquidity_participation", 0.0)),
                            "liquidity_scale": float(liquidity_diag.get("liquidity_scale", 1.0)),
                            "liquidity_tail_impact_multiplier": float(tail_multiplier),
                            "liquidity_tail_spread_dkk_per_mwh": float(tail_spread),
                        }
                    )
                    trades.append(
                        {
                            "timestep": int(t),
                            "timestamp": str(timestamps.iloc[t]) if hasattr(timestamps, "iloc") else str(t),
                            "entry_mode": entry_mode,
                            "current_price": float(current),
                            "same_hour_prev_day_price": float(prev_day),
                            "entry_price": float(entry),
                            "signal": float(signal),
                            "notional_dkk": float(notional),
                            "volume_mwh": float(notional / max(float(args.payoff_reference_price), 1e-9)),
                            "roll_cost_dkk": float(cost),
                        }
                    )
        remaining = []
        for contract in contracts:
            if int(contract["settle_step"]) > t:
                remaining.append(contract)
                continue
            delivery_step = int(contract.get("delivery_step", contract["open_step"]))
            contract_settlement_price = float(
                np.clip(settlement_prices[delivery_step], settlement_floor, settlement_ceiling)
            )
            entry = float(contract["entry_price"])
            spread = contract_settlement_price - entry
            payoff_mode = str(args.payoff_mode).strip().lower().replace("-", "_")
            if payoff_mode in {"mwh_volume", "mwh"}:
                volume_mwh = float(contract.get("volume_mwh", float(contract["notional"]) / max(float(args.payoff_reference_price), 1e-9)))
                realized_return = float(spread)
                pnl = float(volume_mwh * spread)
            else:
                denom = max(abs(entry), float(args.payoff_reference_price))
                realized_return = (contract_settlement_price - entry) / denom
                pnl = float(contract["notional"]) * realized_return
            budget = max(0.0, budget + pnl)
            cumulative_pnl += pnl
            settled_returns.append(float(realized_return))
            if abs(float(contract["notional"])) > 100.0:
                active_count += 1
                if np.sign(float(contract["notional"])) == np.sign(realized_return) and realized_return != 0.0:
                    hit_count += 1
        contracts = remaining
        collateral_required, collateral_open_notional, collateral_open_volume = _collateral_required(contracts, args)
        max_collateral_required = max(max_collateral_required, collateral_required)
        funding_cost = _collateral_funding_cost(collateral_required, args)
        if funding_cost > 0.0:
            budget = max(0.0, budget - funding_cost)
            cumulative_collateral_funding_cost += funding_cost

        sleeve_value = float(budget)
        if (not margin_active) and sleeve_value <= margin_threshold:
            margin_active = True
            margin_step = int(t)
            budget = max(0.0, float(sleeve_value))
            contracts = []
            sleeve_value = float(budget)
        equity.append(float(sleeve_value))

    equity_arr = np.asarray(equity, dtype=np.float64)
    total_return = float(equity_arr[-1] / max(initial_sleeve, 1e-12) - 1.0)
    sharpe_stats = _daily_sharpe_stats(
        timestamps,
        equity_arr,
        initial_sleeve=initial_sleeve,
        annual_risk_free_rate=float(args.annual_risk_free_rate),
    )
    return {
        "entry_mode": entry_mode,
        "status": "completed",
        "rows": int(len(data)),
        "decision_freq": int(args.decision_freq),
        "horizon_steps": int(args.horizon_steps),
        "initial_trading_sleeve_dkk": float(initial_sleeve),
        "max_notional_dkk": float(max_notional),
        "final_trading_sleeve_dkk": float(equity_arr[-1]),
        "total_return": total_return,
        "return_pct": 100.0 * total_return,
        "daily_sharpe_raw": float(sharpe_stats["daily_sharpe"]),
        "daily_hac7_sharpe_raw": float(sharpe_stats["daily_hac7_sharpe"]),
        "daily_sharpe": None if margin_active else float(sharpe_stats["daily_sharpe"]),
        "daily_hac7_sharpe": None if margin_active else float(sharpe_stats["daily_hac7_sharpe"]),
        "daily_hac7_vif": float(sharpe_stats["daily_hac7_vif"]),
        "daily_n_returns": float(sharpe_stats["daily_n_returns"]),
        "max_drawdown": _max_drawdown(
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
        "trades": int(len(trades)),
        "settled_contracts": int(len(settled_returns)),
        "margin_active": bool(margin_active),
        "margin_step": int(margin_step),
        "maintenance_margin_fraction": float(args.maintenance_margin_fraction),
        "maintenance_margin_threshold_dkk": float(margin_threshold),
        "directional_hit_rate": float(hit_count / active_count) if active_count else 0.0,
        "mean_settled_return": float(np.mean(settled_returns)) if settled_returns else 0.0,
        "std_settled_return": float(np.std(settled_returns, ddof=1)) if len(settled_returns) > 1 else 0.0,
        "liquidity_bind_count": int(liquidity_bind_count),
        "max_liquidity_participation": float(max_liquidity_participation),
        "max_collateral_required_dkk": float(max_collateral_required),
        "final_collateral_open_notional_dkk": float(collateral_open_notional if "collateral_open_notional" in locals() else 0.0),
        "final_collateral_open_volume_mwh": float(collateral_open_volume if "collateral_open_volume" in locals() else 0.0),
        "settlement_price_data": str(getattr(args, "settlement_price_data", "") or ""),
        "basis_scale": float(getattr(args, "basis_scale", 0.0)),
        "rule_description": (
            "At each hourly decision, target signed exposure is proportional to "
            "current price minus same-hour previous-day price. Contracts settle "
            "after the fixed horizon using the selected entry-price benchmark."
        ),
        "trades_detail": trades,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_data", default="evaluation_dataset_ffill/unseendata_v2.csv")
    parser.add_argument("--output_dir", default="results/rule_price_entry_baseline_unseendata_v2")
    parser.add_argument("--entry_modes", nargs="+", default=["current_price"], choices=["current_price", "same_hour_prev_day"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decision_freq", type=int, default=6)
    parser.add_argument("--horizon_steps", type=int, default=6)
    parser.add_argument("--time_step_hours", type=float, default=10.0 / 60.0)
    parser.add_argument("--annual_risk_free_rate", type=float, default=0.02)
    parser.add_argument("--initial_fund_dkk", type=float, default=None)
    parser.add_argument("--financial_allocation", type=float, default=None)
    parser.add_argument("--capital_allocation_fraction", type=float, default=0.60)
    parser.add_argument("--max_position_size", type=float, default=0.10)
    parser.add_argument("--rule_exposure_cap", type=float, default=0.60)
    parser.add_argument("--rule_gain", type=float, default=3.0)
    parser.add_argument("--no_trade_band", type=float, default=0.002)
    parser.add_argument("--signal_denom_floor", type=float, default=50.0)
    parser.add_argument("--payoff_reference_price", type=float, default=500.0)
    parser.add_argument(
        "--payoff_mode",
        choices=["entry_price_floor", "mwh_volume"],
        default="mwh_volume",
        help="entry_price_floor uses notional*price_return; mwh_volume uses explicit MWh volume times price spread.",
    )
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

    results = {
        "evaluation_type": "rule_price_entry_baseline",
        "eval_data": str(eval_path),
        "seed": int(args.seed),
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in {"entry_modes"}
        },
        "entry_modes": {},
    }
    summary_rows = []
    for mode in args.entry_modes:
        metrics = evaluate_mode(data, mode, args)
        trade_details = metrics.pop("trades_detail")
        results["entry_modes"][mode] = metrics
        pd.DataFrame(trade_details).to_csv(out_dir / f"trades_{mode}.csv", index=False)
        summary_rows.append({k: v for k, v in metrics.items() if not isinstance(v, (dict, list))})

    json_path = out_dir / "rule_price_entry_baseline_results.json"
    csv_path = out_dir / "rule_price_entry_baseline_summary.csv"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

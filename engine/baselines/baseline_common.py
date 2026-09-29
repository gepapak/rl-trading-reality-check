"""Shared baseline utilities aligned with the current Tier1 evaluation contract."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from market_fee_protocol import base_execution_fee_components, market_access_fee_for_step


BASELINES_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINES_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class BaselineEconomicConfig:
    initial_budget_usd: float
    dkk_to_usd_rate: float
    annual_risk_free_rate: float
    distribution_rate: float
    target_cash_ratio: float
    min_distribution_threshold_ratio: float
    time_step_hours: float
    owned_wind_capacity_mw: float
    owned_solar_capacity_mw: float
    owned_hydro_capacity_mw: float
    owned_battery_capacity_mwh: float
    wind_capex_per_mw_usd: float
    solar_capex_per_mw_usd: float
    hydro_capex_per_mw_usd: float
    battery_capex_per_mwh_usd: float

    @property
    def initial_budget_dkk(self) -> float:
        return self.initial_budget_usd / self.dkk_to_usd_rate

    @property
    def wind_capex_per_mw_dkk(self) -> float:
        return self.wind_capex_per_mw_usd / self.dkk_to_usd_rate

    @property
    def solar_capex_per_mw_dkk(self) -> float:
        return self.solar_capex_per_mw_usd / self.dkk_to_usd_rate

    @property
    def hydro_capex_per_mw_dkk(self) -> float:
        return self.hydro_capex_per_mw_usd / self.dkk_to_usd_rate

    @property
    def battery_capex_per_mwh_dkk(self) -> float:
        return self.battery_capex_per_mwh_usd / self.dkk_to_usd_rate


def load_baseline_economic_config() -> BaselineEconomicConfig:
    """Read the economic constants from the current project config."""
    try:
        from config import EnhancedConfig

        cfg = EnhancedConfig()
        return BaselineEconomicConfig(
            initial_budget_usd=float(getattr(cfg, "init_budget_usd", 800_000_000.0)),
            dkk_to_usd_rate=float(getattr(cfg, "dkk_to_usd_rate", 0.145)),
            annual_risk_free_rate=float(getattr(cfg, "risk_free_rate", 0.02)),
            distribution_rate=float(getattr(cfg, "distribution_rate", 0.10)),
            target_cash_ratio=float(getattr(cfg, "target_cash_ratio", 0.15)),
            min_distribution_threshold_ratio=float(
                getattr(cfg, "min_distribution_threshold_ratio", 0.01)
            ),
            time_step_hours=float(getattr(cfg, "time_step_hours", 10.0 / 60.0)),
            owned_wind_capacity_mw=float(getattr(cfg, "owned_wind_capacity_mw", 270.0)),
            owned_solar_capacity_mw=float(getattr(cfg, "owned_solar_capacity_mw", 100.0)),
            owned_hydro_capacity_mw=float(getattr(cfg, "owned_hydro_capacity_mw", 40.0)),
            owned_battery_capacity_mwh=float(getattr(cfg, "owned_battery_capacity_mwh", 10.0)),
            wind_capex_per_mw_usd=float(getattr(cfg, "wind_capex_per_mw", 2_000_000.0)),
            solar_capex_per_mw_usd=float(getattr(cfg, "solar_capex_per_mw", 1_000_000.0)),
            hydro_capex_per_mw_usd=float(getattr(cfg, "hydro_capex_per_mw", 1_500_000.0)),
            battery_capex_per_mwh_usd=float(
                getattr(cfg, "battery_capex_per_mwh", 400_000.0)
            ),
        )
    except Exception:
        return BaselineEconomicConfig(
            initial_budget_usd=800_000_000.0,
            dkk_to_usd_rate=0.145,
            annual_risk_free_rate=0.02,
            distribution_rate=0.10,
            target_cash_ratio=0.15,
            min_distribution_threshold_ratio=0.01,
            time_step_hours=10.0 / 60.0,
            owned_wind_capacity_mw=270.0,
            owned_solar_capacity_mw=100.0,
            owned_hydro_capacity_mw=40.0,
            owned_battery_capacity_mwh=10.0,
            wind_capex_per_mw_usd=2_000_000.0,
            solar_capex_per_mw_usd=1_000_000.0,
            hydro_capex_per_mw_usd=1_500_000.0,
            battery_capex_per_mwh_usd=400_000.0,
        )


def _parse_timestamps(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    try:
        return parsed.dt.tz_convert(None)
    except Exception:
        return parsed


def detect_timebase_hours(data: pd.DataFrame, default_hours: float = 10.0 / 60.0) -> float:
    """Infer hours per row from a timestamp column, falling back to 10 minutes."""
    if "timestamp" not in data.columns:
        return float(default_hours)
    try:
        parsed = pd.to_datetime(data["timestamp"], errors="coerce").dropna().sort_values()
        if parsed.size < 2:
            return float(default_hours)
        deltas = parsed.diff().dt.total_seconds().to_numpy(dtype=np.float64)
        deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
        if deltas.size == 0:
            return float(default_hours)
        return float(np.median(deltas) / 3600.0)
    except Exception:
        return float(default_hours)


def periods_per_year_from_hours(timebase_hours: float) -> float:
    return float((365.25 * 24.0) / max(float(timebase_hours), 1e-12))


def infer_periods_per_year(timestamps, default_periods_per_year: float = 52560.0) -> float:
    try:
        if timestamps is None:
            return float(default_periods_per_year)
        ts = pd.Series(timestamps).dropna()
        if ts.empty:
            return float(default_periods_per_year)
        parsed = pd.to_datetime(ts, errors="coerce").dropna().sort_values()
        if parsed.size < 2:
            return float(default_periods_per_year)
        deltas = parsed.diff().dt.total_seconds().to_numpy(dtype=np.float64)
        deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
        if deltas.size == 0:
            return float(default_periods_per_year)
        seconds_per_year = 365.25 * 24.0 * 60.0 * 60.0
        return float(np.clip(seconds_per_year / float(np.median(deltas)), 1.0, 525960.0))
    except Exception:
        return float(default_periods_per_year)


def annual_rate_to_step_rate(annual_rate: float, periods_per_year: float) -> float:
    annual = float(annual_rate)
    periods = float(max(periods_per_year, 1.0))
    if not np.isfinite(annual) or annual <= -1.0:
        return 0.0
    return float((1.0 + annual) ** (1.0 / periods) - 1.0)


def distribute_excess_cash(cash_value: float, nav_value: float, cfg: BaselineEconomicConfig) -> Tuple[float, float]:
    """Apply the same shareholder cash-distribution rule used by Tier1."""
    cash = float(cash_value)
    nav = float(max(nav_value, 0.0))
    target_cash = nav * float(cfg.target_cash_ratio)
    excess_cash = cash - target_cash
    threshold = nav * float(cfg.min_distribution_threshold_ratio)
    if excess_cash <= threshold:
        return cash, 0.0
    distribution = excess_cash * float(cfg.distribution_rate)
    return cash - distribution, float(max(distribution, 0.0))


def _finite_values(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def compute_performance_metrics(
    nav_values: Iterable[float],
    *,
    timestamps=None,
    periods_per_year: Optional[float] = None,
    annual_risk_free_rate: float = 0.02,
    total_distributions: float = 0.0,
    distribution_adjusted_values: Optional[Iterable[float]] = None,
    value_suffix: str = "usd",
) -> Dict[str, Any]:
    """Compute metrics with distribution-adjusted wealth as the primary series."""
    reported = _finite_values(nav_values)
    if distribution_adjusted_values is None:
        primary = reported
    else:
        primary = _finite_values(distribution_adjusted_values)
    if primary.size == 0:
        return {"error": "No finite portfolio values"}

    periods = (
        float(periods_per_year)
        if periods_per_year is not None and np.isfinite(periods_per_year) and periods_per_year > 0.0
        else infer_periods_per_year(timestamps)
    )
    returns = np.diff(primary) / np.clip(primary[:-1], 1e-12, None)
    returns = returns[np.isfinite(returns)]
    step_vol = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    ann_vol = float(step_vol * math.sqrt(periods))
    step_rf = annual_rate_to_step_rate(annual_risk_free_rate, periods)
    excess = returns - step_rf
    sharpe = float((np.mean(excess) / step_vol) * math.sqrt(periods)) if step_vol > 0.0 else 0.0
    peak = np.maximum.accumulate(primary)
    drawdown = np.where(peak > 0.0, (peak - primary) / peak, 0.0)

    out: Dict[str, Any] = {
        "total_return": float(primary[-1] / primary[0] - 1.0) if primary.size > 1 else 0.0,
        "annual_return": (
            float((primary[-1] / primary[0]) ** (periods / max(primary.size - 1, 1)) - 1.0)
            if primary.size > 1 and primary[0] > 0.0 and primary[-1] > 0.0
            else 0.0
        ),
        "volatility": ann_vol,
        "step_volatility": step_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(np.max(drawdown)) if drawdown.size else 0.0,
        "initial_portfolio_value": float(primary[0]),
        "final_portfolio_value": float(primary[-1]),
        "periods_per_year": float(periods),
        "annual_risk_free_rate": float(annual_risk_free_rate),
        "per_step_risk_free_rate": float(step_rf),
        "distribution_adjusted_evaluation": True,
        f"total_distributions_{value_suffix}": float(max(total_distributions, 0.0)),
    }

    if reported.size:
        reported_returns = np.diff(reported) / np.clip(reported[:-1], 1e-12, None)
        reported_returns = reported_returns[np.isfinite(reported_returns)]
        reported_step_vol = float(np.std(reported_returns, ddof=1)) if reported_returns.size > 1 else 0.0
        reported_peak = np.maximum.accumulate(reported)
        reported_dd = np.where(reported_peak > 0.0, (reported_peak - reported) / reported_peak, 0.0)
        reported_excess = reported_returns - step_rf
        out.update(
            {
                "reported_nav_total_return": float(reported[-1] / reported[0] - 1.0)
                if reported.size > 1
                else 0.0,
                "reported_nav_annual_return": (
                    float((reported[-1] / reported[0]) ** (periods / max(reported.size - 1, 1)) - 1.0)
                    if reported.size > 1 and reported[0] > 0.0 and reported[-1] > 0.0
                    else 0.0
                ),
                "reported_nav_volatility": float(reported_step_vol * math.sqrt(periods)),
                "reported_nav_step_volatility": reported_step_vol,
                "reported_nav_sharpe_ratio": float(
                    (np.mean(reported_excess) / reported_step_vol) * math.sqrt(periods)
                )
                if reported_step_vol > 0.0
                else 0.0,
                "reported_nav_max_drawdown": float(np.max(reported_dd)) if reported_dd.size else 0.0,
                "reported_nav_initial_portfolio_value": float(reported[0]),
                "reported_nav_final_portfolio_value": float(reported[-1]),
            }
        )
    return out


def _subsample_path(values: np.ndarray, step: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return arr
    stride = max(int(step), 1)
    return arr[::stride]


def _series_returns(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return np.asarray([], dtype=np.float64)
    returns = np.diff(arr) / np.clip(arr[:-1], 1e-12, None)
    return returns[np.isfinite(returns)]


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


def _path_sharpe(
    values: np.ndarray,
    *,
    periods_per_year: float,
    annual_risk_free_rate: float,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return {
            "annualized_sharpe": 0.0,
            "annualized_volatility": 0.0,
            "step_volatility": 0.0,
        }
    returns = _series_returns(arr)
    if returns.size <= 1:
        return {
            "annualized_sharpe": 0.0,
            "annualized_volatility": 0.0,
            "step_volatility": 0.0,
        }
    periods = float(max(periods_per_year, 1.0))
    step_vol = float(np.std(returns, ddof=1))
    step_rf = annual_rate_to_step_rate(float(annual_risk_free_rate), periods)
    excess = returns - step_rf
    sharpe = float((np.mean(excess) / step_vol) * math.sqrt(periods)) if step_vol > 0.0 else 0.0
    return {
        "annualized_sharpe": sharpe,
        "annualized_volatility": float(step_vol * math.sqrt(periods)),
        "step_volatility": step_vol,
    }


def compute_trading_sleeve_metrics(
    records: Iterable[Dict[str, Any]],
    *,
    dkk_to_usd_rate: float,
    annual_risk_free_rate: float = 0.02,
    primary_sharpe_mode: str = "daily_hac_7",
    initial_trading_sleeve_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Paper-facing sleeve decomposition for baseline ledgers.

    This mirrors the evaluation report convention: the primary trading-sleeve
    Sharpe uses non-overlapping daily returns with an optional Newey-West/HAC
    adjustment for serial correlation.
    """
    rows = list(records)
    if not rows:
        return {}
    trading = np.asarray(
        [
            float(
                r.get(
                    "distribution_adjusted_trading_sleeve_usd",
                    r.get("trading_sleeve_usd", 0.0),
                )
            )
            for r in rows
        ],
        dtype=np.float64,
    )
    operating = np.asarray(
        [
            (
                float(r.get("physical_book_value_dkk", 0.0))
                + float(r.get("accumulated_operational_revenue_dkk", 0.0))
            )
            * float(dkk_to_usd_rate)
            for r in rows
        ],
        dtype=np.float64,
    )
    nav = np.asarray(
        [float(r.get("distribution_adjusted_value_usd", r.get("portfolio_value_usd", 0.0))) for r in rows],
        dtype=np.float64,
    )
    exposure = np.asarray(
        [abs(float(r.get("current_exposure_dkk", 0.0))) for r in rows],
        dtype=np.float64,
    )
    decision_mask = np.asarray(
        [bool(r.get("decision_step", False)) for r in rows],
        dtype=bool,
    )

    if trading.size == 0 or not np.isfinite(trading[0]) or abs(float(trading[0])) <= 1e-12:
        return {}

    initial_trading = (
        float(initial_trading_sleeve_usd)
        if initial_trading_sleeve_usd is not None
        else float(trading[0])
    )
    trading_path = np.concatenate(
        (
            np.asarray([initial_trading], dtype=np.float64),
            trading,
        )
    )
    total_gain = float(nav[-1] - nav[0]) if nav.size else 0.0
    trading_gain = float(trading[-1] - initial_trading)
    operating_gain = float(operating[-1] - operating[0]) if operating.size else 0.0
    complete_days = int(trading.size // 144)
    daily_path = np.concatenate(
        (
            np.asarray([initial_trading], dtype=np.float64),
            trading[143 : complete_days * 144 : 144],
        )
    )
    daily_returns = _series_returns(daily_path)
    daily_stats = _path_sharpe(
        daily_path,
        periods_per_year=365.25,
        annual_risk_free_rate=float(annual_risk_free_rate),
    )
    daily_hac_lag = min(7, max(int(daily_returns.size) - 2, 0))
    daily_hac_vif = _hac_variance_inflation(daily_returns, daily_hac_lag)
    daily_hac_sharpe = (
        float(daily_stats["annualized_sharpe"] / math.sqrt(daily_hac_vif))
        if daily_hac_vif > 0.0 else 0.0
    )
    mode = str(primary_sharpe_mode or "daily_hac_7").strip().lower().replace("-", "_")
    if mode in {"daily_hac", "daily_hac7", "daily_newey_west", "daily_newey_west_7"}:
        mode = "daily_hac_7"
    if mode not in {"daily", "daily_hac_7"}:
        mode = "daily_hac_7"
    if mode == "daily_hac_7":
        primary_sharpe = daily_hac_sharpe
        primary_vol = float(daily_stats["annualized_volatility"] * math.sqrt(daily_hac_vif))
        primary_step_vol = float(daily_stats["step_volatility"] * math.sqrt(daily_hac_vif))
    else:
        primary_sharpe = float(daily_stats["annualized_sharpe"])
        primary_vol = float(daily_stats["annualized_volatility"])
        primary_step_vol = float(daily_stats["step_volatility"])
    peak = np.maximum.accumulate(trading_path)
    dd = np.where(peak > 0.0, (peak - trading_path) / peak, 0.0)
    max_dd = float(np.max(dd)) if dd.size else 0.0
    margin_active_any = bool(
        any(bool(row.get("trading_sleeve_margin_active", False)) for row in rows)
    )
    nonpositive_equity = bool(np.any(trading_path <= 0.0))
    ruined = bool(margin_active_any or nonpositive_equity or max_dd >= 0.99)
    ruin_reasons = []
    if margin_active_any:
        ruin_reasons.append("maintenance_margin_triggered")
    if nonpositive_equity:
        ruin_reasons.append("nonpositive_sleeve_equity")
    if max_dd >= 0.99:
        ruin_reasons.append("drawdown_at_least_99pct")
    decision_exposure = exposure[decision_mask] if exposure.size and decision_mask.any() else exposure
    return {
        "sleeve_total_initial_usd": float(nav[0]) if nav.size else 0.0,
        "sleeve_total_final_usd": float(nav[-1]) if nav.size else 0.0,
        "sleeve_total_gain_usd": total_gain,
        "sleeve_trading_initial_usd": initial_trading,
        "sleeve_trading_final_usd": float(trading[-1]),
        "sleeve_trading_gain_usd": trading_gain,
        "sleeve_operating_initial_usd": float(operating[0]) if operating.size else 0.0,
        "sleeve_operating_final_usd": float(operating[-1]) if operating.size else 0.0,
        "sleeve_operating_gain_usd": operating_gain,
        "sleeve_trading_gain_share": float(trading_gain / total_gain) if abs(total_gain) > 1e-12 else 0.0,
        "sleeve_trading_return_pct": float((trading[-1] / initial_trading - 1.0) * 100.0),
        "sleeve_trading_path_convention": "true_initial_plus_complete_day_ends_v1",
        "sleeve_operating_return_pct": (
            float((operating[-1] / operating[0] - 1.0) * 100.0)
            if operating.size and abs(float(operating[0])) > 1e-12
            else 0.0
        ),
        "sleeve_trading_sharpe_primary_mode": mode,
        "sleeve_trading_ruined": ruined,
        "sleeve_trading_ruin_reason": ";".join(ruin_reasons),
        "sleeve_trading_daily_sharpe_ratio_raw": float(daily_stats["annualized_sharpe"]),
        "sleeve_trading_daily_hac7_sharpe_ratio_raw": float(daily_hac_sharpe),
        "sleeve_trading_daily_sharpe_ratio": None if ruined else float(daily_stats["annualized_sharpe"]),
        "sleeve_trading_daily_hac7_sharpe_ratio": None if ruined else float(daily_hac_sharpe),
        "sleeve_trading_daily_hac7_vif": float(daily_hac_vif),
        "sleeve_trading_daily_n_returns": float(daily_returns.size),
        "sleeve_trading_volatility": float(primary_vol),
        "sleeve_trading_step_volatility": float(primary_step_vol),
        "sleeve_trading_sharpe_ratio_raw": float(primary_sharpe),
        "sleeve_trading_sharpe_ratio": None if ruined else float(primary_sharpe),
        "sleeve_trading_max_drawdown_pct": max_dd * 100.0,
        "sleeve_mean_abs_exposure_dkk": float(np.mean(decision_exposure)) if decision_exposure.size else 0.0,
        "sleeve_max_abs_exposure_dkk": float(np.max(decision_exposure)) if decision_exposure.size else 0.0,
    }


def add_common_summary_fields(
    summary: Dict[str, Any],
    *,
    method: str,
    status: str = "completed",
) -> Dict[str, Any]:
    summary["method"] = method
    summary["status"] = status
    summary["evaluation_contract"] = "prototype5_final_campaign_v1"
    summary["baseline_source_folder"] = "baselines"
    return summary


class HybridFundLedger:
    """Current-codebase hybrid fund accounting used by publication baselines.

    This mirrors the Tier1 environment accounting contract:
    - fixed physical infrastructure is deployed from the 88% physical sleeve;
    - strategy decisions affect only the 12% trading sleeve;
    - operating revenue, financial MTM, transaction costs, depreciation, battery
      cash flow, and shareholder distributions are tracked separately.
    """

    def __init__(self, data: pd.DataFrame, *, seed: int = 42, timebase_hours: Optional[float] = None):
        from config import EnhancedConfig

        self.config = EnhancedConfig()
        self.config.seed = int(seed)
        self.config.enable_forecast_utilization = False
        self.config.enable_forecast_utilisation = False

        self.econ_cfg = load_baseline_economic_config()
        self.data = data.reset_index(drop=True).copy()
        self.timebase_hours = float(timebase_hours or detect_timebase_hours(self.data, self.econ_cfg.time_step_hours))
        self.periods_per_year = periods_per_year_from_hours(self.timebase_hours)
        self.dkk_to_usd_rate = float(getattr(self.config, "dkk_to_usd_rate", self.econ_cfg.dkk_to_usd_rate))

        self.price = self.data.get("price", pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self.settlement_price = self._build_settlement_price_series()
        self.settlement_basis_component = self.settlement_price - self.price
        self.settlement_price_mode = (
            str(getattr(self.config, "mtm_settlement_price_mode", "energy_index") or "energy_index")
            .strip()
            .lower()
            .replace("-", "_")
        )
        self.wind = self.data.get("wind", pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self.solar = self.data.get("solar", pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        self.hydro = self.data.get("hydro", pd.Series(0.0, index=self.data.index)).astype(float).to_numpy()
        default_load = self.wind + self.solar + self.hydro
        self.load = self.data.get("load", pd.Series(default_load, index=self.data.index)).astype(float).to_numpy()
        self._impact_ref_notional_by_step = None
        self._impact_volume_mwh_by_step = None
        self._init_market_impact_liquidity_reference()

        self.asset_capex = self.config.get_asset_capex(currency="DKK")
        self.initial_budget_dkk = float(getattr(self.config, "init_budget", self.econ_cfg.initial_budget_dkk))
        self.initial_budget_usd = self.initial_budget_dkk * self.dkk_to_usd_rate
        self.trading_allocation_budget = self.initial_budget_dkk * float(getattr(self.config, "financial_allocation", 0.12))
        self.budget = float(self.trading_allocation_budget)
        self.capital_allocation_fraction = float(getattr(self.config, "capital_allocation_fraction", 0.60))
        self.max_position_size = float(getattr(self.config, "max_position_size", 0.35))
        self.investment_freq = int(max(1, getattr(self.config, "investment_freq", 6)))

        self.physical_assets = {
            "wind_capacity_mw": float(getattr(self.config, "owned_wind_capacity_mw", self.econ_cfg.owned_wind_capacity_mw)),
            "solar_capacity_mw": float(getattr(self.config, "owned_solar_capacity_mw", self.econ_cfg.owned_solar_capacity_mw)),
            "hydro_capacity_mw": float(getattr(self.config, "owned_hydro_capacity_mw", self.econ_cfg.owned_hydro_capacity_mw)),
            "battery_capacity_mwh": float(getattr(self.config, "owned_battery_capacity_mwh", self.econ_cfg.owned_battery_capacity_mwh)),
        }
        self.physical_book_initial = self._physical_book_value(current_timestep=0)

        self.financial_positions = {
            "wind_instrument_value": 0.0,
            "solar_instrument_value": 0.0,
            "hydro_instrument_value": 0.0,
        }
        self.financial_mtm_positions = {
            "wind_instrument_value": 0.0,
            "solar_instrument_value": 0.0,
            "hydro_instrument_value": 0.0,
        }

        init_soc = float(np.clip(
            getattr(self.config, "battery_initial_soc", 0.5),
            getattr(self.config, "batt_soc_min", 0.1),
            getattr(self.config, "batt_soc_max", 0.9),
        ))
        self.battery_energy_mwh = init_soc * self.physical_assets["battery_capacity_mwh"]
        self.battery_discharge_power = 0.0

        self.accumulated_operational_revenue = 0.0
        self.total_distributions = 0.0
        self.cumulative_generation_revenue = 0.0
        self.cumulative_battery_revenue = 0.0
        self.cumulative_mtm_pnl = 0.0
        self.cumulative_transaction_costs = 0.0
        self.cumulative_market_impact_costs = 0.0
        self.cumulative_volume_transaction_fees = 0.0
        self.cumulative_market_access_fees = 0.0
        self._last_volume_transaction_fee = 0.0
        self._last_market_access_fee = 0.0
        self._market_access_fee_last_charged_step = -1
        self.last_transaction_cost = 0.0
        self.last_mtm_pnl = 0.0
        self.last_battery_cash_delta = 0.0
        self.last_generation_revenue = 0.0
        self.last_distribution = 0.0
        self._last_market_impact_cost = 0.0
        self._last_market_impact_ref_notional = 0.0
        self._last_market_impact_participation = 0.0
        self._last_market_impact_bp = 0.0
        self._last_liquidity_market_volume_mwh = 0.0
        self._last_liquidity_causal_market_volume_mwh = 0.0
        self._last_liquidity_volume_cap_mwh = 0.0
        self._last_liquidity_requested_volume_mwh = 0.0
        self._last_liquidity_executed_volume_mwh = 0.0
        self._last_liquidity_participation = 0.0
        self._last_liquidity_scale = 1.0
        self._last_liquidity_tail_impact_multiplier = 1.0
        self._last_liquidity_tail_spread_dkk_per_mwh = 0.0
        self._last_collateral_required_dkk = 0.0
        self._last_collateral_funding_cost_dkk = 0.0
        self._last_collateral_open_notional_dkk = 0.0
        self._last_collateral_open_volume_mwh = 0.0
        self.cumulative_collateral_funding_costs = 0.0
        self._last_horizon_settlement_count = 0
        self._last_horizon_open_count = 0
        self._last_horizon_open_notional = 0.0
        self._last_horizon_roll_cost = 0.0
        self._last_horizon_settlement_pnl = 0.0
        self._last_horizon_entry_price = 0.0
        self._last_horizon_settlement_price = 0.0
        self._last_horizon_settlement_basis = 0.0
        self._last_horizon_mark_price = 0.0
        self._last_horizon_mark_basis = 0.0
        self._last_horizon_payoff_denominator = 0.0
        self._last_horizon_settlement_notional = 0.0
        self._last_horizon_settlement_volume_mwh = 0.0
        self._last_horizon_unrealized_pnl = 0.0
        self._last_horizon_unrealized_notional = 0.0
        self._last_horizon_unrealized_volume_mwh = 0.0
        self._last_horizon_open_volume_mwh = 0.0
        self._last_horizon_mtm_incremental_pnl = 0.0
        self._horizon_contracts = []
        self._last_mtm_exit_count = 0
        self._mtm_loss_exit_count = 0
        self._mtm_loss_exit_first_step = -1
        self._trading_sleeve_margin_active = False
        self._trading_sleeve_margin_step = -1
        self._trading_sleeve_margin_count = 0
        self._trading_sleeve_margin_phase = ""
        self._last_trading_sleeve_value_dkk = float(self.budget)
        self._last_trading_sleeve_margin_threshold_dkk = 0.0
        self._last_trading_sleeve_margin_gap_dkk = float(self.budget)

        self.nav_values_usd = []
        self.adjusted_nav_values_usd = []
        self.records = []

    def _settlement_mode(self) -> str:
        mode = (
            str(getattr(self.config, "mtm_settlement_price_mode", "energy_index") or "energy_index")
            .strip()
            .lower()
            .replace("-", "_")
        )
        aliases = {
            "none": "energy_index",
            "base": "energy_index",
            "basis": "cross_zone_basis",
            "basis_adjusted": "cross_zone_basis",
            "external": "external_series",
            "realized": "external_series",
            "real_settlement": "external_series",
            "settlement": "external_series",
        }
        return aliases.get(mode, mode)

    def _load_peer_price_series(self) -> Optional[np.ndarray]:
        path_raw = str(getattr(self.config, "mtm_basis_price_data_path", "") or "").strip()
        if not path_raw:
            return None
        basis_path = Path(path_raw)
        if not basis_path.is_absolute():
            basis_path = PROJECT_ROOT / basis_path
        if not basis_path.is_file():
            raise FileNotFoundError(f"mtm_basis_price_data_path not found for baseline ledger: {basis_path}")

        price_col = str(getattr(self.config, "mtm_basis_price_column", "price") or "price")
        ts_col = str(getattr(self.config, "mtm_basis_timestamp_column", "timestamp") or "timestamp")
        peer_df = pd.read_csv(basis_path)
        if price_col not in peer_df.columns:
            raise ValueError(f"Basis price column '{price_col}' not found in {basis_path}")
        peer_prices = pd.to_numeric(peer_df[price_col], errors="coerce")

        if ts_col in peer_df.columns and "timestamp" in self.data.columns:
            source = pd.DataFrame(
                {
                    "timestamp": _parse_timestamps(peer_df[ts_col]),
                    "peer_price": peer_prices,
                }
            ).dropna(subset=["timestamp", "peer_price"])
            target = pd.DataFrame(
                {
                    "timestamp": _parse_timestamps(self.data["timestamp"]),
                    "_row": np.arange(len(self.data), dtype=np.int64),
                }
            ).dropna(subset=["timestamp"])
            source = source.sort_values("timestamp").groupby("timestamp", as_index=False)["peer_price"].last()
            target = target.sort_values("timestamp")
            merged = pd.merge(target, source, on="timestamp", how="left").sort_values("_row")
            coverage = float(merged["peer_price"].notna().mean())
            if coverage < 0.80:
                raise ValueError(
                    f"Basis price timestamp coverage is too low ({coverage:.1%}) for {basis_path}"
                )
            peer = merged["peer_price"].ffill().bfill().to_numpy(dtype=np.float64)
        else:
            peer = peer_prices.ffill().bfill().to_numpy(dtype=np.float64)
            if len(peer) < len(self.price):
                peer = np.pad(peer, (0, len(self.price) - len(peer)), mode="edge")
            peer = peer[: len(self.price)]

        if len(peer) != len(self.price):
            raise ValueError(f"Basis price length mismatch: got {len(peer)}, expected {len(self.price)}")
        return np.clip(peer, 10.0, 2000.0)

    def _load_external_settlement_series(self) -> Optional[np.ndarray]:
        path_raw = str(getattr(self.config, "mtm_external_settlement_price_data_path", "") or "").strip()
        if not path_raw:
            return None
        path_raw = path_raw.format(episode=0, episode_num=0) if "{episode" in path_raw or "{episode_num" in path_raw else path_raw
        settlement_path = Path(path_raw)
        if not settlement_path.is_absolute():
            settlement_path = PROJECT_ROOT / settlement_path
        if not settlement_path.is_file():
            raise FileNotFoundError(
                f"mtm_external_settlement_price_data_path not found for baseline ledger: {settlement_path}"
            )

        price_col = str(
            getattr(self.config, "mtm_external_settlement_price_column", "settlement_price")
            or "settlement_price"
        )
        ts_col = str(
            getattr(self.config, "mtm_external_settlement_timestamp_column", "timestamp")
            or "timestamp"
        )
        source_df = pd.read_csv(settlement_path)
        if price_col not in source_df.columns:
            raise ValueError(f"External settlement price column '{price_col}' not found in {settlement_path}")
        source_prices = pd.to_numeric(source_df[price_col], errors="coerce")

        if ts_col in source_df.columns and "timestamp" in self.data.columns:
            source = pd.DataFrame(
                {
                    "timestamp": _parse_timestamps(source_df[ts_col]),
                    "settlement_price": source_prices,
                }
            ).dropna(subset=["timestamp", "settlement_price"])
            target = pd.DataFrame(
                {
                    "timestamp": _parse_timestamps(self.data["timestamp"]),
                    "_row": np.arange(len(self.data), dtype=np.int64),
                }
            ).dropna(subset=["timestamp"])
            source = source.sort_values("timestamp").groupby("timestamp", as_index=False)["settlement_price"].last()
            target = target.sort_values("timestamp")
            merged = pd.merge(target, source, on="timestamp", how="left").sort_values("_row")
            coverage = float(merged["settlement_price"].notna().mean())
            if coverage < 1.0:
                missing = int(merged["settlement_price"].isna().sum())
                raise ValueError(
                    f"External settlement timestamps are incomplete for {settlement_path}: "
                    f"coverage={coverage:.6%}, missing_rows={missing}"
                )
            settlement = merged["settlement_price"].to_numpy(dtype=np.float64)
        else:
            settlement = source_prices.to_numpy(dtype=np.float64)

        if len(settlement) != len(self.price):
            raise ValueError(
                f"External settlement length mismatch: got {len(settlement)}, expected {len(self.price)}"
            )
        if not np.all(np.isfinite(settlement)):
            raise ValueError(f"External settlement series contains non-finite values: {settlement_path}")
        min_price = float(getattr(self.config, "mtm_external_settlement_min_price_dkk_per_mwh", -111750.0))
        max_price = float(getattr(self.config, "mtm_external_settlement_max_price_dkk_per_mwh", 111750.0))
        return np.clip(settlement, min_price, max_price)

    def _build_settlement_price_series(self) -> np.ndarray:
        base = np.asarray(self.price, dtype=np.float64)
        mode = self._settlement_mode()
        if mode == "external_series":
            external = self._load_external_settlement_series()
            return base.copy() if external is None else np.asarray(external, dtype=np.float64)
        if mode != "cross_zone_basis":
            return base.copy()
        peer = self._load_peer_price_series()
        if peer is None:
            return base.copy()

        raw_basis = pd.Series(np.asarray(peer, dtype=np.float64) - base)
        mode = (
            str(getattr(self.config, "mtm_basis_centering_mode", "rolling_median") or "rolling_median")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if mode == "none":
            center = pd.Series(np.zeros(len(raw_basis), dtype=np.float64))
        elif mode == "expanding_median":
            center = raw_basis.expanding(min_periods=1).median().shift(1)
            center.iloc[0] = raw_basis.iloc[0]
        else:
            window = int(max(1, getattr(self.config, "mtm_basis_centering_window_steps", 4320)))
            center = raw_basis.rolling(window=window, min_periods=1).median().shift(1)
            center.iloc[0] = raw_basis.iloc[0]

        scale = float(max(getattr(self.config, "mtm_basis_scale", 0.0), 0.0))
        component = scale * (raw_basis - center.ffill().bfill()).fillna(0.0)
        return np.clip(base + component.to_numpy(dtype=np.float64), 10.0, 2000.0)

    def _settlement_price_at(self, timestep: int) -> float:
        if len(self.settlement_price) == 0:
            return 0.0
        idx = int(max(0, min(int(timestep), len(self.settlement_price) - 1)))
        floor = float(
            getattr(self.config, "mtm_external_settlement_min_price_dkk_per_mwh", -1000.0)
        )
        return float(np.clip(self.settlement_price[idx], floor, 1e9))

    def _settlement_basis_at(self, timestep: int) -> float:
        if len(self.settlement_basis_component) == 0:
            return 0.0
        idx = int(max(0, min(int(timestep), len(self.settlement_basis_component) - 1)))
        value = float(self.settlement_basis_component[idx])
        return value if np.isfinite(value) else 0.0

    @property
    def battery_soc(self) -> float:
        cap = max(self.physical_assets.get("battery_capacity_mwh", 0.0), 1e-12)
        return float(np.clip(self.battery_energy_mwh / cap, 0.0, 1.0))

    @property
    def current_total_exposure_dkk(self) -> float:
        return float(sum(self.financial_positions.values()))

    @property
    def current_abs_exposure_dkk(self) -> float:
        return float(sum(abs(v) for v in self.financial_positions.values()))

    def _investor_sizing_base_dkk(self) -> float:
        mode = str(
            getattr(self.config, "investor_notional_sizing_base", "initial_trading_sleeve")
            or "initial_trading_sleeve"
        ).strip().lower().replace("-", "_")
        if mode in {"live_budget", "legacy"}:
            mode = "live_trading_cash"
        elif mode in {"initial_sleeve", "fixed_sleeve", "fixed_trading_sleeve"}:
            mode = "initial_trading_sleeve"
        elif mode in {"initial_nav", "initial_fund"}:
            mode = "initial_fund_nav"

        if mode == "live_trading_cash":
            base = float(self.budget)
        elif mode == "initial_fund_nav":
            base = float(self.initial_budget_dkk)
        else:
            base = float(self.trading_allocation_budget)
        return float(max(base, 0.0))

    def _tradeable_capital_dkk(self) -> float:
        cap = float(np.clip(self.capital_allocation_fraction, 0.0, 1.0))
        tradeable = float(self._investor_sizing_base_dkk() * cap)
        if bool(getattr(self.config, "enable_collateral_cash_drag", False)):
            haircut = float(np.clip(getattr(self.config, "collateral_tradeable_haircut", 1.0), 0.0, 1.0))
            reserved = float(max(getattr(self, "_last_collateral_required_dkk", 0.0), self._collateral_required_dkk()))
            tradeable = float(max(tradeable - haircut * reserved, 0.0))
        return float(max(tradeable, 0.0))

    def _is_investor_decision_step(self, timestep: int) -> bool:
        t = int(timestep)
        freq = int(max(getattr(self.config, "investment_freq", self.investment_freq) or 1, 1))
        return bool(t > 0 and t % freq == 0)

    def _trading_sleeve_margin_threshold_dkk(self) -> float:
        fraction = float(
            np.clip(
                getattr(self.config, "trading_sleeve_maintenance_margin_fraction", 0.05),
                0.0,
                1.0,
            )
        )
        return float(self.trading_allocation_budget * fraction)

    def _current_trading_sleeve_value_dkk(self) -> float:
        mtm = float(sum(self.financial_mtm_positions.values()))
        value = float(self.budget + mtm)
        return value if np.isfinite(value) else 0.0

    def _liquidate_trading_sleeve_for_margin(self, timestep: int, *, phase: str) -> None:
        sleeve_value = self._current_trading_sleeve_value_dkk()
        realized_cash = float(max(sleeve_value, 0.0))
        self.budget = realized_cash
        for key in self.financial_positions:
            self.financial_positions[key] = 0.0
        for key in self.financial_mtm_positions:
            self.financial_mtm_positions[key] = 0.0
        self._horizon_contracts = []
        self._last_horizon_unrealized_pnl = 0.0
        self._last_horizon_unrealized_notional = 0.0
        self._last_horizon_unrealized_volume_mwh = 0.0
        self._last_horizon_open_notional = 0.0
        self._last_horizon_open_volume_mwh = 0.0
        self._last_trading_sleeve_value_dkk = realized_cash

    def _check_trading_sleeve_margin(self, timestep: int, *, phase: str) -> bool:
        if not bool(getattr(self.config, "enable_trading_sleeve_margin", True)):
            self._last_trading_sleeve_value_dkk = self._current_trading_sleeve_value_dkk()
            self._last_trading_sleeve_margin_threshold_dkk = 0.0
            self._last_trading_sleeve_margin_gap_dkk = self._last_trading_sleeve_value_dkk
            return False

        threshold = self._trading_sleeve_margin_threshold_dkk()
        sleeve_value = self._current_trading_sleeve_value_dkk()
        gap = float(sleeve_value - threshold)
        self._last_trading_sleeve_value_dkk = float(sleeve_value)
        self._last_trading_sleeve_margin_threshold_dkk = float(threshold)
        self._last_trading_sleeve_margin_gap_dkk = float(gap)
        if bool(self._trading_sleeve_margin_active):
            return True
        if sleeve_value > threshold:
            return False

        self._trading_sleeve_margin_active = True
        self._trading_sleeve_margin_step = int(timestep)
        self._trading_sleeve_margin_count += 1
        self._trading_sleeve_margin_phase = str(phase)
        if bool(getattr(self.config, "trading_sleeve_margin_liquidate_positions", True)):
            self._liquidate_trading_sleeve_for_margin(int(timestep), phase=phase)
            self._last_trading_sleeve_margin_gap_dkk = float(self.budget - threshold)
        return True

    def _physical_book_value(self, current_timestep: int) -> float:
        years_elapsed = float(current_timestep) / (365.25 * 24.0 * 6.0)
        dep_rate = float(getattr(self.config, "annual_depreciation_rate", 0.02))
        max_dep = float(getattr(self.config, "max_depreciation_ratio", 0.75))
        dep = min(years_elapsed * dep_rate, max_dep)
        return float(
            self.physical_assets["wind_capacity_mw"] * self.asset_capex["wind_mw"] * (1.0 - dep)
            + self.physical_assets["solar_capacity_mw"] * self.asset_capex["solar_mw"] * (1.0 - dep)
            + self.physical_assets["hydro_capacity_mw"] * self.asset_capex["hydro_mw"] * (1.0 - dep)
            + self.physical_assets["battery_capacity_mwh"] * self.asset_capex["battery_mwh"] * (1.0 - dep)
        )

    def fund_nav_dkk(self, current_timestep: int) -> float:
        from financial_engine import FinancialEngine

        return float(FinancialEngine.calculate_fund_nav(
            budget=self.budget,
            physical_assets=self.physical_assets,
            asset_capex=self.asset_capex,
            financial_mtm_values=self.financial_mtm_positions,
            accumulated_operational_revenue=self.accumulated_operational_revenue,
            current_timestep=int(current_timestep),
            config=self.config,
        ))

    def _generation_revenue(self, timestep: int) -> float:
        from financial_engine import FinancialEngine

        price = float(np.clip(self.price[timestep], -1000.0, 1e9))
        return float(FinancialEngine.calculate_generation_revenue(
            timestep=int(timestep),
            price=price,
            wind_data=self.wind,
            solar_data=self.solar,
            hydro_data=self.hydro,
            wind_scale=1.0,
            solar_scale=1.0,
            hydro_scale=1.0,
            physical_assets=self.physical_assets,
            asset_capex=self.asset_capex,
            config=self.config,
            electricity_markup=float(getattr(self.config, "electricity_markup", 1.0)),
            currency_conversion=float(getattr(self.config, "currency_conversion", 1.0)),
        ))

    def _is_horizon_settlement_mtm(self) -> bool:
        return (
            str(getattr(self.config, "mtm_return_model", "percent_capped") or "percent_capped")
            .strip()
            .lower()
            in {"horizon_settlement", "horizon_settlement_continuous"}
        )

    def _is_continuous_horizon_mtm(self) -> bool:
        return (
            str(getattr(self.config, "mtm_return_model", "percent_capped") or "percent_capped")
            .strip()
            .lower()
            == "horizon_settlement_continuous"
        )

    def _horizon_entry_price(self, timestep: int) -> float:
        if len(self.price) == 0:
            return 0.0
        idx = int(max(0, min(int(timestep), len(self.price) - 1)))
        current_price = float(np.clip(self.price[idx], -1000.0, 1e9))
        mode = (
            str(getattr(self.config, "mtm_entry_price_mode", "current_price") or "current_price")
            .strip()
            .lower()
        )
        steps_per_day = int(round(24.0 / max(float(self.timebase_hours), 1e-9)))
        steps_per_day = max(1, steps_per_day)

        if mode == "current_price":
            return current_price

        if mode == "same_hour_prev_day":
            prev_day_idx = idx - steps_per_day
            if prev_day_idx >= 0:
                value = float(np.clip(self.price[prev_day_idx], -1000.0, 1e9))
                if np.isfinite(value):
                    return value

        same_hour_values = []
        cursor = idx - steps_per_day
        while cursor >= 0 and len(same_hour_values) < 30:
            value = float(np.clip(self.price[cursor], -1000.0, 1e9))
            if np.isfinite(value):
                same_hour_values.append(value)
            cursor -= steps_per_day
        if same_hour_values:
            return float(np.median(np.asarray(same_hour_values, dtype=np.float64)))
        return current_price

    def _market_impact_reference_notional(self, timestep: int) -> float:
        mode_raw = getattr(self.config, "impact_ref_notional", "sleeve")
        mode = str(mode_raw).strip().lower()
        if mode == "sleeve":
            return float(max(self.budget, 1.0))
        if mode in {"volume", "market_volume", "day_ahead_volume", "liquidity_volume"}:
            ref = getattr(self, "_impact_ref_notional_by_step", None)
            if ref is None:
                raise RuntimeError("Volume-calibrated baseline impact reference was not initialized")
            idx = int(max(0, min(int(timestep), len(ref) - 1)))
            return float(max(ref[idx], 1.0))
        try:
            return float(max(float(mode_raw), 1.0))
        except Exception:
            return float(max(self.budget, 1.0))

    def _market_liquidity_volume_mwh(self, timestep: int) -> float:
        min_volume = float(max(getattr(self.config, "liquidity_min_volume_mwh", 1.0), 1e-9))
        multiplier = float(max(getattr(self.config, "liquidity_volume_multiplier", 1.0), 1e-9))
        source = (
            str(getattr(self.config, "liquidity_volume_source", "load") or "load")
            .strip()
            .lower()
            .replace("-", "_")
        )
        idx = int(max(0, min(int(timestep), len(self.price) - 1))) if len(self.price) else 0
        horizon_hours = float(
            max(
                float(getattr(self.config, "time_step_hours", self.timebase_hours))
                * float(max(getattr(self.config, "mtm_settlement_horizon_steps", 6), 1)),
                float(getattr(self.config, "time_step_hours", self.timebase_hours)),
            )
        )
        if source == "impact_volume":
            volumes = getattr(self, "_impact_volume_mwh_by_step", None)
            if volumes is None:
                raise RuntimeError("liquidity_volume_source='impact_volume' requires MWh impact-volume data")
            idx_v = int(max(0, min(idx, len(volumes) - 1)))
            value = float(volumes[idx_v])
        elif source == "generation":
            value = float(max(self.wind[idx] + self.solar[idx] + self.hydro[idx], 0.0) * horizon_hours)
        elif source == "max_load_generation":
            generation = float(max(self.wind[idx] + self.solar[idx] + self.hydro[idx], 0.0))
            value = float(max(float(self.load[idx]), generation, 0.0) * horizon_hours)
        else:
            value = float(max(float(self.load[idx]), 0.0) * horizon_hours)
        if not np.isfinite(value) or value <= 0.0:
            value = min_volume
        return float(max(value * multiplier, min_volume))

    def _latest_strictly_observed_delivery_step(self, timestep: int) -> int:
        horizon = int(max(getattr(self.config, "mtm_settlement_horizon_steps", 6), 1))
        return int(timestep) - horizon - 1

    def _causal_market_liquidity_volume_mwh(self, timestep: int) -> float:
        observed_step = self._latest_strictly_observed_delivery_step(int(timestep))
        if observed_step < 0:
            min_volume = float(max(getattr(self.config, "liquidity_min_volume_mwh", 1.0), 1e-9))
            multiplier = float(max(getattr(self.config, "liquidity_volume_multiplier", 1.0), 1e-9))
            return float(min_volume * multiplier)
        return float(self._market_liquidity_volume_mwh(observed_step))

    @staticmethod
    def _impact_volume_columns(df: pd.DataFrame, configured: str) -> list[str]:
        configured = str(configured or "").strip()
        if configured:
            cols = [c.strip() for c in configured.split(",") if c.strip()]
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"impact volume column(s) not found: {missing}")
            return cols
        lower_to_col = {str(c).strip().lower(): c for c in df.columns}
        for name in (
            "market_volume_mwh",
            "volume_mwh",
            "day_ahead_volume_mwh",
            "traded_volume_mwh",
            "auction_volume_mwh",
            "matched_volume_mwh",
            "volume",
        ):
            if name in lower_to_col:
                return [lower_to_col[name]]
        candidates = [
            c
            for c in df.columns
            if ("volume" in str(c).lower() or "mwh" in str(c).lower())
            and pd.to_numeric(df[c], errors="coerce").notna().any()
        ]
        if len(candidates) == 1:
            return candidates
        if not candidates:
            raise ValueError("Could not autodetect an impact-volume column")
        raise ValueError(
            "Multiple possible impact-volume columns found; configure impact_volume_column: "
            + ", ".join(map(str, candidates))
        )

    def _init_market_impact_liquidity_reference(self) -> None:
        mode = str(getattr(self.config, "impact_ref_notional", "sleeve") or "sleeve").strip().lower()
        liquidity_source = (
            str(getattr(self.config, "liquidity_volume_source", "load") or "load")
            .strip()
            .lower()
            .replace("-", "_")
        )
        uses_volume = mode in {"volume", "market_volume", "day_ahead_volume", "liquidity_volume"}
        needs_mwh = liquidity_source == "impact_volume"
        if not uses_volume and not needs_mwh:
            return

        path_raw = str(getattr(self.config, "impact_volume_data_path", "") or "").strip()
        if not path_raw:
            raise ValueError("Real-volume liquidity requires impact_volume_data_path")
        path = Path(path_raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Impact/liquidity volume data not found: {path}")

        volume_df = pd.read_csv(path, low_memory=False)
        ts_col = str(getattr(self.config, "impact_volume_timestamp_column", "timestamp") or "timestamp")
        if ts_col not in volume_df.columns:
            raise ValueError(f"Impact volume timestamp column '{ts_col}' not found in {path}")
        if "timestamp" not in self.data.columns:
            raise ValueError("Baseline data must include a timestamp column for real-volume liquidity")

        volume_cols = self._impact_volume_columns(volume_df, str(getattr(self.config, "impact_volume_column", "")))
        volume_values = volume_df[volume_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        source = pd.DataFrame(
            {
                "timestamp": _parse_timestamps(volume_df[ts_col]),
                "volume_value": volume_values,
            }
        ).dropna(subset=["timestamp", "volume_value"])
        source = source[source["volume_value"] >= 0.0].sort_values("timestamp")
        if source.empty:
            raise ValueError(f"Impact volume data has no usable rows: {path}")
        source = source.groupby("timestamp", as_index=False)["volume_value"].sum()

        target = pd.DataFrame(
            {
                "_row": np.arange(len(self.data), dtype=np.int64),
                "timestamp": _parse_timestamps(self.data["timestamp"]),
            }
        ).dropna(subset=["timestamp"]).sort_values("timestamp")
        if len(target) != len(self.data):
            raise ValueError("Baseline timestamps could not all be parsed for real-volume liquidity")

        tolerance = pd.Timedelta(
            minutes=float(getattr(self.config, "impact_volume_max_staleness_minutes", 90.0) or 90.0)
        )
        aligned = pd.merge_asof(
            target,
            source,
            on="timestamp",
            direction="backward",
            tolerance=tolerance,
        ).sort_values("_row")
        if aligned["volume_value"].isna().any():
            missing = int(aligned["volume_value"].isna().sum())
            raise ValueError(f"Impact volume data missing/stale for {missing} baseline rows: {path}")

        unit = str(getattr(self.config, "impact_volume_unit", "mwh") or "mwh").strip().lower()
        values = aligned["volume_value"].to_numpy(dtype=np.float64)
        if unit == "mwh":
            min_volume = float(max(getattr(self.config, "liquidity_min_volume_mwh", 1.0), 1e-9))
            if self._is_mwh_volume_payoff():
                conversion_price = float(
                    max(getattr(self.config, "mtm_reference_price_dkk_per_mwh", 500.0), 1e-6)
                )
                price_scale = np.full(len(values), conversion_price, dtype=np.float64)
            else:
                price_floor = float(
                    getattr(self.config, "impact_volume_price_floor_dkk_per_mwh", 50.0) or 50.0
                )
                price_scale = np.maximum(
                    np.abs(np.asarray(self.price, dtype=np.float64)), price_floor
                )
            self._impact_ref_notional_by_step = np.maximum(values, min_volume) * price_scale
            self._impact_volume_mwh_by_step = values
        elif unit == "dkk":
            self._impact_ref_notional_by_step = values
            self._impact_volume_mwh_by_step = None
            if needs_mwh:
                raise ValueError("liquidity_volume_source='impact_volume' requires impact_volume_unit='mwh'")
        else:
            raise ValueError("impact_volume_unit must be 'mwh' or 'dkk'")

    def _liquidity_tail_impact_multiplier(self, timestep: int) -> float:
        base = float(max(getattr(self.config, "liquidity_tail_impact_multiplier", 1.0), 0.0))
        if base <= 1.0:
            self._last_liquidity_tail_impact_multiplier = 1.0
            self._last_liquidity_tail_spread_dkk_per_mwh = 0.0
            return 1.0
        threshold = float(max(getattr(self.config, "liquidity_tail_impact_threshold_dkk_per_mwh", 5000.0), 1e-9))
        power = float(max(getattr(self.config, "liquidity_tail_impact_power", 1.0), 0.0))
        max_mult = float(max(getattr(self.config, "liquidity_tail_impact_max_multiplier", base), 1.0))
        observed_step = self._latest_strictly_observed_delivery_step(int(timestep))
        try:
            spread = (
                abs(
                    float(self._settlement_price_at(observed_step))
                    - float(self._horizon_entry_price(observed_step))
                )
                if observed_step >= 0
                else 0.0
            )
        except Exception:
            spread = 0.0
        stress = max(0.0, float(spread) / threshold - 1.0)
        multiplier = float(np.clip(1.0 + (base - 1.0) * (stress ** power), 1.0, max_mult))
        self._last_liquidity_tail_impact_multiplier = multiplier
        self._last_liquidity_tail_spread_dkk_per_mwh = float(spread)
        return multiplier

    def _cap_exposures_by_liquidity(
        self,
        timestep: int,
        entry_price: float,
        exposures: Dict[str, float],
    ) -> Dict[str, float]:
        clean = {
            key: float(value)
            for key, value in (exposures or {}).items()
            if key in {"wind_instrument_value", "solar_instrument_value", "hydro_instrument_value"}
            and np.isfinite(float(value))
        }
        requested_volume = 0.0
        if self._is_mwh_volume_payoff():
            requested_volume = float(
                sum(abs(self._horizon_contract_volume_mwh(entry_price, value)) for value in clean.values())
            )
        cap_fraction = float(max(getattr(self.config, "liquidity_participation_cap_fraction", 0.0), 0.0))
        market_volume = float(self._market_liquidity_volume_mwh(int(timestep))) if cap_fraction > 0.0 else 0.0
        causal_market_volume = (
            float(self._causal_market_liquidity_volume_mwh(int(timestep)))
            if cap_fraction > 0.0
            else 0.0
        )
        cap_volume = (
            float(min(market_volume, causal_market_volume) * cap_fraction)
            if cap_fraction > 0.0
            else 0.0
        )
        if cap_fraction > 0.0 and requested_volume > cap_volume > 0.0:
            scale = float(np.clip(cap_volume / max(requested_volume, 1e-9), 0.0, 1.0))
        else:
            scale = 1.0
        executed_volume = float(requested_volume * scale)
        participation = float(executed_volume / max(market_volume, 1e-9)) if market_volume > 0.0 else 0.0
        self._last_liquidity_market_volume_mwh = float(market_volume)
        self._last_liquidity_causal_market_volume_mwh = float(causal_market_volume)
        self._last_liquidity_volume_cap_mwh = float(cap_volume)
        self._last_liquidity_requested_volume_mwh = float(requested_volume)
        self._last_liquidity_executed_volume_mwh = float(executed_volume)
        self._last_liquidity_participation = float(participation)
        self._last_liquidity_scale = float(scale)
        return {key: float(value) * scale for key, value in clean.items()}

    def _execution_no_trade_threshold_dkk(self, max_position_notional_dkk: float) -> float:
        """Return a no-trade threshold scaled to feasible execution capacity."""
        configured = float(max(getattr(self.config, "no_trade_threshold", 0.0), 0.0))
        if configured <= 0.0:
            threshold = 0.0
            reference = 0.0
        elif configured > 1.0:
            threshold = configured
            reference = configured
        else:
            reference = float(max(max_position_notional_dkk, 1.0))
            reference_mode = str(
                getattr(self.config, "no_trade_threshold_reference", "executable_capacity")
                or "executable_capacity"
            ).strip().lower().replace("-", "_")
            cap_mwh = float(max(getattr(self, "_last_liquidity_volume_cap_mwh", 0.0), 0.0))
            if (
                reference_mode == "executable_capacity"
                and self._is_mwh_volume_payoff()
                and cap_mwh > 0.0
            ):
                conversion_price = float(
                    max(getattr(self.config, "mtm_reference_price_dkk_per_mwh", 500.0), 1e-6)
                )
                reference = min(reference, max(cap_mwh * conversion_price, 1.0))
            threshold = configured * reference
        self._last_no_trade_threshold_reference_dkk = float(reference)
        self._last_no_trade_threshold_dkk = float(threshold)
        return float(threshold)

    def _open_horizon_contract_totals(self) -> Tuple[float, float]:
        open_notional = 0.0
        open_volume_mwh = 0.0
        for contract in list(self._horizon_contracts or []):
            entry_price = float(contract.get("entry_price", 0.0))
            volumes = contract.get("volumes_mwh", {}) or {}
            for key, exposure in (contract.get("exposures", {}) or {}).items():
                try:
                    exposure_f = float(exposure)
                except Exception:
                    continue
                if not np.isfinite(exposure_f):
                    continue
                open_notional += abs(exposure_f)
                if self._is_mwh_volume_payoff():
                    open_volume_mwh += abs(
                        float(volumes.get(key, self._horizon_contract_volume_mwh(entry_price, exposure_f)))
                    )
        return float(open_notional), float(open_volume_mwh)

    def _collateral_required_dkk(self) -> float:
        if not bool(getattr(self.config, "enable_collateral_cash_drag", False)):
            self._last_collateral_required_dkk = 0.0
            self._last_collateral_open_notional_dkk = 0.0
            self._last_collateral_open_volume_mwh = 0.0
            return 0.0
        open_notional, open_volume_mwh = self._open_horizon_contract_totals()
        notional_margin = float(max(getattr(self.config, "collateral_notional_margin_fraction", 0.02), 0.0))
        stress_fraction = float(max(getattr(self.config, "collateral_stress_loss_fraction", 0.10), 0.0))
        stress_price = float(max(getattr(self.config, "collateral_stress_price_dkk_per_mwh", 25000.0), 0.0))
        required = float(max(open_notional * notional_margin, open_volume_mwh * stress_price * stress_fraction, 0.0))
        self._last_collateral_required_dkk = required
        self._last_collateral_open_notional_dkk = float(open_notional)
        self._last_collateral_open_volume_mwh = float(open_volume_mwh)
        return required

    def _apply_collateral_cash_drag(self) -> float:
        required = float(self._collateral_required_dkk())
        if required <= 0.0:
            self._last_collateral_funding_cost_dkk = 0.0
            return 0.0
        rate = float(max(getattr(self.config, "collateral_funding_rate_annual", 0.05), 0.0))
        cost = float(required * rate * self.timebase_hours / (365.25 * 24.0))
        if not np.isfinite(cost) or cost <= 0.0:
            self._last_collateral_funding_cost_dkk = 0.0
            return 0.0
        self.budget = max(0.0, self.budget - cost)
        self._last_collateral_funding_cost_dkk = cost
        self.cumulative_collateral_funding_costs += cost
        return cost

    def _charge_horizon_roll_cost(
        self,
        total_notional: float,
        timestep: int,
        total_volume_mwh: Optional[float] = None,
    ) -> float:
        total_notional = float(max(total_notional, 0.0))
        if total_notional <= 100.0:
            self._last_horizon_roll_cost = 0.0
            return 0.0

        friction_multiplier = float(getattr(self.config, "friction_cost_multiplier", 1.0) or 1.0)
        half_spread_bp = float(getattr(self.config, "half_spread_bp", 0.0) or 0.0)
        volume_mwh = (
            float(max(total_volume_mwh, 0.0))
            if total_volume_mwh is not None
            else float(total_notional / max(getattr(self.config, "mtm_reference_price_dkk_per_mwh", 500.0), 1e-9))
        )
        fee_components = base_execution_fee_components(
            model=getattr(self.config, "market_fee_model", "legacy_notional_fixed"),
            abs_notional_dkk=total_notional,
            abs_volume_mwh=volume_mwh,
            friction_multiplier=friction_multiplier,
            transaction_cost_bps=float(getattr(self.config, "transaction_cost_bps", 0.5)),
            transaction_fixed_cost_dkk=float(getattr(self.config, "transaction_fixed_cost", 0.0)),
            transaction_fee_dkk_per_mwh=float(
                getattr(self.config, "transaction_fee_dkk_per_mwh", 0.0)
            ),
        )
        spread_cost = total_notional * half_spread_bp / 10000.0

        impact_cost = 0.0
        impact_coef_bp = float(getattr(self.config, "impact_coef_bp", 0.0) or 0.0)
        if impact_coef_bp > 0.0:
            impact_exponent = float(getattr(self.config, "impact_exponent", 0.5) or 0.5)
            impact_ref_notional = self._market_impact_reference_notional(timestep)
            participation = total_notional / max(float(impact_ref_notional), 1.0)
            tail_multiplier = float(self._liquidity_tail_impact_multiplier(int(timestep)))
            impact_bp = impact_coef_bp * tail_multiplier * (participation ** impact_exponent)
            impact_cost = total_notional * impact_bp / 10000.0
            self._last_market_impact_ref_notional = float(impact_ref_notional)
            self._last_market_impact_participation = float(participation)
            self._last_market_impact_bp = float(impact_bp)
        else:
            self._last_market_impact_ref_notional = 0.0
            self._last_market_impact_participation = 0.0
            self._last_market_impact_bp = 0.0

        roll_cost = float(
            fee_components["base_execution_fee_dkk"] + spread_cost + impact_cost
        )
        self.budget = max(0.0, self.budget - roll_cost)
        self.last_transaction_cost += roll_cost
        self.cumulative_transaction_costs += roll_cost
        volume_fee = float(fee_components["volume_transaction_fee_dkk"])
        self._last_volume_transaction_fee = volume_fee
        self.cumulative_volume_transaction_fees += volume_fee
        self._last_horizon_roll_cost = roll_cost
        self._last_market_impact_cost += float(impact_cost)
        self.cumulative_market_impact_costs += float(impact_cost)
        return roll_cost

    def _apply_market_access_fee(self, timestep: int) -> float:
        step = int(timestep)
        if int(getattr(self, "_market_access_fee_last_charged_step", -1)) == step:
            return 0.0
        fee = market_access_fee_for_step(
            annual_fee_dkk=float(getattr(self.config, "annual_market_access_fee_dkk", 0.0)),
            allocation_fraction=float(
                getattr(self.config, "market_access_fee_allocation_fraction", 1.0)
            ),
            time_step_hours=float(self.timebase_hours),
        )
        self._market_access_fee_last_charged_step = step
        self._last_market_access_fee = float(fee)
        if fee <= 0.0:
            return 0.0
        self.budget = max(0.0, self.budget - fee)
        self.cumulative_market_access_fees += float(fee)
        self.cumulative_transaction_costs += float(fee)
        return float(fee)

    def _horizon_payoff_denominator(self, entry_price: float) -> float:
        reference_price = float(max(getattr(self.config, "mtm_reference_price_dkk_per_mwh", 500.0), 1e-6))
        mode = (
            str(getattr(self.config, "mtm_horizon_payoff_denominator_mode", "reference_price") or "reference_price")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if mode == "entry_price_floor":
            entry_abs = abs(float(entry_price)) if np.isfinite(float(entry_price)) else 0.0
            return float(max(entry_abs, reference_price))
        return reference_price

    def _is_mwh_volume_payoff(self) -> bool:
        mode = (
            str(getattr(self.config, "mtm_horizon_payoff_denominator_mode", "reference_price") or "reference_price")
            .strip()
            .lower()
            .replace("-", "_")
        )
        return mode in {"mwh_volume", "mwh"}

    def _horizon_contract_volume_mwh(self, entry_price: float, exposure_dkk: float) -> float:
        reference_price = float(max(getattr(self.config, "mtm_reference_price_dkk_per_mwh", 500.0), 1e-6))
        return float(float(exposure_dkk) / max(reference_price, 1e-9))

    def _horizon_unit_payoff(self, entry_price: float, settlement_price: float) -> float:
        spread = float(settlement_price) - float(entry_price)
        if self._is_mwh_volume_payoff():
            return float(spread)
        payoff_denominator = float(self._horizon_payoff_denominator(entry_price))
        return float(spread / max(payoff_denominator, 1e-12))

    def _open_horizon_settlement_contract(self, timestep: int, exposures: Dict[str, float]) -> None:
        if not self._is_horizon_settlement_mtm():
            return
        if bool(self._trading_sleeve_margin_active) and bool(
            getattr(self.config, "trading_sleeve_margin_disable_trading", True)
        ):
            return
        horizon = int(max(1, getattr(self.config, "mtm_settlement_horizon_steps", 6) or 6))
        settle_step = int(timestep) + horizon
        if settle_step >= len(self.price):
            return

        clean_exposures = {
            key: float(value)
            for key, value in (exposures or {}).items()
            if key in {"wind_instrument_value", "solar_instrument_value", "hydro_instrument_value"}
            and np.isfinite(float(value))
        }
        total_notional = float(sum(abs(v) for v in clean_exposures.values()))
        if total_notional <= 100.0:
            return

        entry_price = float(self._horizon_entry_price(int(timestep)))
        volumes_mwh = {
            key: self._horizon_contract_volume_mwh(entry_price, value)
            for key, value in clean_exposures.items()
        } if self._is_mwh_volume_payoff() else {}
        total_volume_mwh = float(sum(abs(v) for v in volumes_mwh.values()))
        self._horizon_contracts.append(
            {
                "open_step": int(timestep),
                "delivery_step": int(timestep),
                "settle_step": int(settle_step),
                "entry_price": entry_price,
                "exposures": clean_exposures,
                "volumes_mwh": volumes_mwh,
            }
        )
        self._last_horizon_open_count += 1
        self._last_horizon_open_notional = float(total_notional)
        self._last_horizon_entry_price = entry_price
        self._charge_horizon_roll_cost(total_notional, int(timestep), total_volume_mwh)

    def _settle_horizon_contracts_for_step(self, timestep: int) -> Tuple[float, Dict[str, float]]:
        per_asset = {
            "wind_instrument_value": 0.0,
            "solar_instrument_value": 0.0,
            "hydro_instrument_value": 0.0,
        }
        if not self._is_horizon_settlement_mtm() or len(self.price) == 0:
            return 0.0, per_asset

        remaining = []
        total_pnl = 0.0
        settled_count = 0
        settled_abs_notional = 0.0
        settled_abs_volume_mwh = 0.0
        denominator_weighted_sum = 0.0
        settlement_price_weighted_sum = 0.0
        settlement_basis_weighted_sum = 0.0
        last_settlement_price = 0.0
        last_settlement_basis = 0.0

        for contract in list(self._horizon_contracts or []):
            if int(contract.get("settle_step", 10**18)) > int(timestep):
                remaining.append(contract)
                continue
            delivery_step = int(contract.get("delivery_step", contract.get("open_step", timestep)))
            settlement_price = float(self._settlement_price_at(delivery_step))
            settlement_basis = float(self._settlement_basis_at(delivery_step))
            entry_price = float(contract.get("entry_price", settlement_price))
            payoff_denominator = float(self._horizon_payoff_denominator(entry_price))
            unit_payoff = float(self._horizon_unit_payoff(entry_price, settlement_price))
            exposures = contract.get("exposures", {}) or {}
            volumes_mwh = contract.get("volumes_mwh", {}) or {}
            contract_abs_notional = float(
                sum(abs(float(v)) for v in exposures.values() if np.isfinite(float(v)))
            )
            if contract_abs_notional > 0.0:
                settled_abs_notional += contract_abs_notional
                denominator_weighted_sum += payoff_denominator * contract_abs_notional
                settlement_price_weighted_sum += settlement_price * contract_abs_notional
                settlement_basis_weighted_sum += settlement_basis * contract_abs_notional
            last_settlement_price = settlement_price
            last_settlement_basis = settlement_basis
            for key, exposure in exposures.items():
                if key not in per_asset:
                    continue
                if self._is_mwh_volume_payoff():
                    volume_mwh = float(volumes_mwh.get(key, self._horizon_contract_volume_mwh(entry_price, exposure)))
                    pnl = float(volume_mwh * unit_payoff)
                    settled_abs_volume_mwh += abs(volume_mwh)
                else:
                    pnl = float(exposure) * unit_payoff
                per_asset[key] += pnl
                total_pnl += pnl
            settled_count += 1

        self._horizon_contracts = remaining
        self._last_horizon_settlement_count = int(settled_count)
        self._last_horizon_settlement_pnl = float(total_pnl)
        self._last_horizon_settlement_price = (
            float(settlement_price_weighted_sum / settled_abs_notional)
            if settled_count > 0 and settled_abs_notional > 0.0
            else (last_settlement_price if settled_count > 0 else 0.0)
        )
        self._last_horizon_settlement_basis = (
            float(settlement_basis_weighted_sum / settled_abs_notional)
            if settled_count > 0 and settled_abs_notional > 0.0
            else (last_settlement_basis if settled_count > 0 else 0.0)
        )
        self._last_horizon_settlement_notional = float(settled_abs_notional) if settled_count > 0 else 0.0
        self._last_horizon_settlement_volume_mwh = float(settled_abs_volume_mwh) if settled_count > 0 else 0.0
        self._last_horizon_payoff_denominator = (
            float(denominator_weighted_sum / settled_abs_notional)
            if settled_count > 0 and settled_abs_notional > 0.0
            else 0.0
        )
        if settled_count > 0:
            self.budget = max(0.0, self.budget + float(total_pnl))
        return float(total_pnl), per_asset

    def _mark_horizon_contracts_to_market_for_step(self, timestep: int) -> Tuple[float, Dict[str, float]]:
        per_asset = {
            "wind_instrument_value": 0.0,
            "solar_instrument_value": 0.0,
            "hydro_instrument_value": 0.0,
        }
        self._last_horizon_unrealized_pnl = 0.0
        self._last_horizon_unrealized_notional = 0.0
        self._last_horizon_unrealized_volume_mwh = 0.0
        self._last_horizon_mark_price = 0.0
        self._last_horizon_mark_basis = 0.0
        return 0.0, per_asset

    def _add_mtm_for_step(self, timestep: int) -> float:
        from financial_engine import FinancialEngine

        if self._is_horizon_settlement_mtm():
            if self._is_continuous_horizon_mtm():
                previous_unrealized = {
                    k: float(self.financial_mtm_positions.get(k, 0.0))
                    for k in ("wind_instrument_value", "solar_instrument_value", "hydro_instrument_value")
                }
                previous_unrealized_total = float(sum(previous_unrealized.values()))
                settled_pnl, settled_per_asset = self._settle_horizon_contracts_for_step(int(timestep))
                unrealized_pnl, unrealized_per_asset = self._mark_horizon_contracts_to_market_for_step(int(timestep))
                mtm_pnl = float(settled_pnl + unrealized_pnl - previous_unrealized_total)
                for key in self.financial_mtm_positions:
                    self.financial_mtm_positions[key] = float(unrealized_per_asset.get(key, 0.0))
                self._last_horizon_mtm_incremental_pnl = float(mtm_pnl)
            else:
                mtm_pnl, _per_asset = self._settle_horizon_contracts_for_step(int(timestep))
            self.last_mtm_pnl = float(mtm_pnl)
            self.cumulative_mtm_pnl += float(mtm_pnl)
            return float(mtm_pnl)

        current_price = float(np.clip(self.price[timestep], -1000.0, 1e9))
        price_returns = FinancialEngine.calculate_price_returns(
            timestep=int(timestep),
            current_price=current_price,
            price_history=self.price,
        )
        raw_price_return = float(price_returns["price_return"])
        price_change = float(price_returns.get("price_change_dkk_per_mwh", 0.0))
        mtm_model = str(getattr(self.config, "mtm_return_model", "percent_capped") or "percent_capped").strip().lower()
        if mtm_model == "notional_price_diff":
            reference = float(max(getattr(self.config, "mtm_reference_price_dkk_per_mwh", 500.0), 1e-6))
            price_return = float(price_change / reference)
        else:
            price_return = raw_price_return
        mtm_pnl, per_asset = FinancialEngine.calculate_mtm_pnl_from_exposure(
            financial_exposures=self.financial_positions,
            price_return=price_return,
            config=self.config,
        )
        for key in self.financial_mtm_positions:
            self.financial_mtm_positions[key] = float(self.financial_mtm_positions.get(key, 0.0) + per_asset.get(key, 0.0))
        self.last_mtm_pnl = float(mtm_pnl)
        self.cumulative_mtm_pnl += float(mtm_pnl)
        return float(mtm_pnl)

    def _risk_budget_weights(self) -> np.ndarray:
        allocation = getattr(self.config, "risk_budget_allocation", None)
        if isinstance(allocation, dict):
            weights = np.array([
                allocation.get("wind", 0.40),
                allocation.get("solar", 0.35),
                allocation.get("hydro", 0.25),
            ], dtype=np.float64)
        else:
            weights = np.array([0.40, 0.35, 0.25], dtype=np.float64)
        denom = float(np.sum(np.abs(weights)))
        if denom <= 0.0:
            weights = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            denom = 3.0
        return weights / denom

    def _settle_mtm_on_resize(self, old_positions: Dict[str, float], target_positions: Dict[str, float]) -> None:
        for asset in ("wind", "solar", "hydro"):
            key = f"{asset}_instrument_value"
            old_exp = float(old_positions.get(key, 0.0))
            new_exp = float(target_positions.get(key, 0.0))
            old_mtm = float(self.financial_mtm_positions.get(key, 0.0))
            if abs(old_exp) <= 100.0:
                continue

            closed_fraction = 0.0
            if old_exp * new_exp < 0.0:
                closed_fraction = 1.0
            elif abs(new_exp) < abs(old_exp) and old_exp * new_exp >= 0.0:
                closed_fraction = (abs(old_exp) - abs(new_exp)) / max(abs(old_exp), 1e-9)
            if closed_fraction <= 0.0:
                continue

            realized = old_mtm * float(np.clip(closed_fraction, 0.0, 1.0))
            self.budget += realized
            self.financial_mtm_positions[key] = float(old_mtm - realized)

    def rebalance_exposure(self, target_exposure: float, timestep: int) -> float:
        """Set aggregate trading exposure in [-1, 1] of the current max position."""
        if int(timestep) == 0:
            self.last_transaction_cost = 0.0
            return 0.0
        if bool(self._trading_sleeve_margin_active) and bool(
            getattr(self.config, "trading_sleeve_margin_disable_trading", True)
        ):
            self.last_transaction_cost = 0.0
            self._last_market_impact_cost = 0.0
            self._last_market_impact_ref_notional = 0.0
            self._last_market_impact_participation = 0.0
            self._last_market_impact_bp = 0.0
            return 0.0
        if not self._is_investor_decision_step(int(timestep)):
            self.last_transaction_cost = 0.0
            self._last_market_impact_cost = 0.0
            self._last_market_impact_ref_notional = 0.0
            self._last_market_impact_participation = 0.0
            self._last_market_impact_bp = 0.0
            return 0.0

        target_exposure = float(np.clip(target_exposure, -1.0, 1.0))
        max_pos_size = float(max(self._tradeable_capital_dkk() * self.max_position_size, 1.0))
        weights = self._risk_budget_weights()
        target_positions = {
            "wind_instrument_value": float(target_exposure * weights[0] * max_pos_size),
            "solar_instrument_value": float(target_exposure * weights[1] * max_pos_size),
            "hydro_instrument_value": float(target_exposure * weights[2] * max_pos_size),
        }
        old_positions = dict(self.financial_positions)

        mtm_loss_threshold_pct = float(
            max(getattr(self.config, "mtm_loss_exit_threshold_pct", 0.03), 0.0)
        )
        mtm_exits_forced = []
        for asset in ("wind", "solar", "hydro"):
            key = f"{asset}_instrument_value"
            current_pos = float(old_positions.get(key, 0.0))
            if abs(current_pos) <= 100.0:
                continue
            mtm_value = float(self.financial_mtm_positions.get(key, 0.0))
            mtm_loss_pct = (-mtm_value / max(abs(current_pos), 1.0)) if mtm_value < 0.0 else 0.0
            if mtm_loss_pct > mtm_loss_threshold_pct:
                target_positions[key] = 0.0
                mtm_exits_forced.append(asset)

        self._last_mtm_exit_count = int(len(mtm_exits_forced))
        if mtm_exits_forced:
            self._mtm_loss_exit_count += int(len(mtm_exits_forced))
            if int(self._mtm_loss_exit_first_step) < 0:
                self._mtm_loss_exit_first_step = int(timestep)

        if self._is_horizon_settlement_mtm():
            entry_price = float(self._horizon_entry_price(int(timestep)))
            target_positions = self._cap_exposures_by_liquidity(int(timestep), entry_price, target_positions)

        total_traded_notional = float(sum(abs(target_positions[k] - old_positions.get(k, 0.0)) for k in target_positions))
        threshold = self._execution_no_trade_threshold_dkk(max_pos_size)

        self.last_transaction_cost = 0.0
        self._last_market_impact_cost = 0.0
        self._last_market_impact_ref_notional = 0.0
        self._last_market_impact_participation = 0.0
        self._last_market_impact_bp = 0.0

        if self._is_horizon_settlement_mtm():
            if total_traded_notional > threshold:
                self.financial_positions.update(target_positions)
                active_positions = target_positions
                returned_notional = total_traded_notional
            else:
                entry_price = float(self._horizon_entry_price(int(timestep)))
                active_positions = self._cap_exposures_by_liquidity(
                    int(timestep), entry_price, old_positions
                )
                if any(
                    abs(float(active_positions.get(k, 0.0)) - float(old_positions.get(k, 0.0))) > 1e-9
                    for k in active_positions
                ):
                    self.financial_positions.update(active_positions)
                returned_notional = 0.0
            self._open_horizon_settlement_contract(int(timestep), active_positions)
            return float(returned_notional)

        self._settle_mtm_on_resize(old_positions, target_positions)

        if total_traded_notional <= threshold:
            self.last_transaction_cost = 0.0
            return 0.0

        friction_multiplier = float(getattr(self.config, "friction_cost_multiplier", 1.0) or 1.0)
        half_spread_bp = float(getattr(self.config, "half_spread_bp", 0.0) or 0.0)
        fee_components = base_execution_fee_components(
            model=getattr(self.config, "market_fee_model", "legacy_notional_fixed"),
            abs_notional_dkk=total_traded_notional,
            abs_volume_mwh=float(
                total_traded_notional
                / max(getattr(self.config, "mtm_reference_price_dkk_per_mwh", 500.0), 1e-9)
            ),
            friction_multiplier=friction_multiplier,
            transaction_cost_bps=float(getattr(self.config, "transaction_cost_bps", 0.5)),
            transaction_fixed_cost_dkk=float(getattr(self.config, "transaction_fixed_cost", 0.0)),
            transaction_fee_dkk_per_mwh=float(
                getattr(self.config, "transaction_fee_dkk_per_mwh", 0.0)
            ),
        )
        spread_cost = total_traded_notional * half_spread_bp / 10000.0
        impact_cost = 0.0
        impact_coef_bp = float(getattr(self.config, "impact_coef_bp", 0.0) or 0.0)
        if impact_coef_bp > 0.0:
            impact_exponent = float(getattr(self.config, "impact_exponent", 0.5) or 0.5)
            impact_ref_notional = self._market_impact_reference_notional(int(timestep))
            participation = total_traded_notional / max(float(impact_ref_notional), 1.0)
            tail_multiplier = float(self._liquidity_tail_impact_multiplier(int(timestep)))
            impact_bp = impact_coef_bp * tail_multiplier * (participation ** impact_exponent)
            impact_cost = total_traded_notional * impact_bp / 10000.0
            self._last_market_impact_ref_notional = float(impact_ref_notional)
            self._last_market_impact_participation = float(participation)
            self._last_market_impact_bp = float(impact_bp)
        txn_cost = float(fee_components["base_execution_fee_dkk"] + spread_cost + impact_cost)
        self.budget = max(0.0, self.budget - txn_cost)
        self.cumulative_transaction_costs += float(txn_cost)
        volume_fee = float(fee_components["volume_transaction_fee_dkk"])
        self._last_volume_transaction_fee = volume_fee
        self.cumulative_volume_transaction_fees += volume_fee
        self.cumulative_market_impact_costs += float(impact_cost)
        self.last_transaction_cost = float(txn_cost)
        self._last_market_impact_cost = float(impact_cost)
        self.financial_positions.update(target_positions)
        return total_traded_notional

    def execute_battery_action(self, action: str | int | float, timestep: int) -> Tuple[float, str]:
        from trading_engine import TradingEngine

        if isinstance(action, str):
            normalized = action.strip().lower()
            if normalized == "charge":
                bat_action = 0
            elif normalized == "discharge":
                bat_action = len(getattr(self.config, "battery_discrete_action_levels", [-1.0, -0.5, 0.0, 0.5, 1.0])) - 1
            else:
                bat_action = len(getattr(self.config, "battery_discrete_action_levels", [-1.0, -0.5, 0.0, 0.5, 1.0])) // 2
        else:
            bat_action = action
            levels = getattr(self.config, "battery_discrete_action_levels", [-1.0, -0.5, 0.0, 0.5, 1.0])
            idx = int(np.clip(int(round(float(np.asarray(action).reshape(-1)[0]))), 0, len(levels) - 1))
            val = float(levels[idx])
            normalized = "charge" if val < 0.0 else "discharge" if val > 0.0 else "idle"

        price = float(np.clip(
            self.price[timestep],
            getattr(self.config, "minimum_price_filter", -1000.0),
            getattr(self.config, "maximum_price_cap", 1e9),
        ))
        cash_delta, state = TradingEngine.execute_battery_operations(
            bat_action=np.array([bat_action]),
            timestep=int(timestep),
            battery_capacity_mwh=self.physical_assets["battery_capacity_mwh"],
            battery_energy=self.battery_energy_mwh,
            price=price,
            batt_power_c_rate=float(getattr(self.config, "batt_power_c_rate", 0.5)),
            batt_eta_charge=float(getattr(self.config, "batt_eta_charge", 0.90)),
            batt_eta_discharge=float(getattr(self.config, "batt_eta_discharge", 0.90)),
            batt_soc_min=float(getattr(self.config, "batt_soc_min", 0.1)),
            batt_soc_max=float(getattr(self.config, "batt_soc_max", 0.9)),
            batt_degradation_cost=float(getattr(self.config, "battery_degradation_cost_mwh", 0.0)),
            battery_dispatch_policy_fn=None,
            config=self.config,
            use_heuristic_dispatch=False,
        )
        self.battery_energy_mwh = float(state.get("battery_energy", self.battery_energy_mwh))
        self.battery_discharge_power = float(state.get("battery_discharge_power", 0.0))
        self.last_battery_cash_delta = float(cash_delta)
        self.cumulative_battery_revenue += float(cash_delta)
        return float(cash_delta), normalized

    def step(
        self,
        timestep: int,
        *,
        target_exposure: Optional[float] = None,
        battery_action: str | int | float = "idle",
    ) -> Dict[str, Any]:
        t = int(timestep)
        self._apply_market_access_fee(t)
        traded_notional = 0.0
        if self._is_horizon_settlement_mtm():
            # Match the MARL event clock: the target for t is selected before
            # maturity-t settlement is observed or booked.
            if target_exposure is not None:
                traded_notional = self.rebalance_exposure(float(target_exposure), t)
            self._add_mtm_for_step(t)
            self._apply_collateral_cash_drag()
            self._check_trading_sleeve_margin(t, phase="post_mtm")
        else:
            self._add_mtm_for_step(t)
            self._apply_collateral_cash_drag()
            self._check_trading_sleeve_margin(t, phase="post_mtm")
            if target_exposure is not None:
                traded_notional = self.rebalance_exposure(float(target_exposure), t)
        battery_cash_delta, battery_action_label = self.execute_battery_action(battery_action, t)

        generation_revenue = self._generation_revenue(t)
        battery_opex = float(getattr(self.config, "battery_opex_rate", 0.0002)) * self.physical_assets["battery_capacity_mwh"]
        operational_revenue = float(generation_revenue - battery_opex)

        self.budget = max(0.0, self.budget + battery_cash_delta)
        self._check_trading_sleeve_margin(t, phase="pre_nav")
        nav_before_current_ops = self.fund_nav_dkk(t)

        from financial_engine import FinancialEngine

        self.budget, distribution_amount = FinancialEngine.distribute_excess_cash(
            budget=self.budget,
            current_fund_nav=nav_before_current_ops,
            init_budget=self.initial_budget_dkk,
            config=self.config,
        )
        self.total_distributions += float(distribution_amount)
        self.last_distribution = float(distribution_amount)

        self.accumulated_operational_revenue += operational_revenue
        self.cumulative_generation_revenue += float(generation_revenue)
        self.last_generation_revenue = float(generation_revenue)

        nav_dkk = self.fund_nav_dkk(t)
        nav_usd = nav_dkk * self.dkk_to_usd_rate
        adjusted_usd = nav_usd + self.total_distributions * self.dkk_to_usd_rate
        physical_book = self._physical_book_value(t)
        financial_mtm = float(sum(self.financial_mtm_positions.values()))
        trading_sleeve = self.budget + financial_mtm
        distribution_adjusted_trading_sleeve = trading_sleeve + self.total_distributions
        is_decision_step = self._is_investor_decision_step(t)

        self.nav_values_usd.append(nav_usd)
        self.adjusted_nav_values_usd.append(adjusted_usd)
        record = {
            "timestep": t,
            "portfolio_value": nav_dkk,
            "portfolio_value_usd": nav_usd,
            "distribution_adjusted_value": nav_dkk + self.total_distributions,
            "distribution_adjusted_value_usd": adjusted_usd,
            "cash": self.budget,
            "cash_usd": self.budget * self.dkk_to_usd_rate,
            "trading_cash_dkk": self.budget,
            "trading_cash_usd": self.budget * self.dkk_to_usd_rate,
            "physical_book_value_dkk": physical_book,
            "physical_book_value_usd": physical_book * self.dkk_to_usd_rate,
            "capacity_value": physical_book,
            "capacity_value_usd": physical_book * self.dkk_to_usd_rate,
            "financial_mtm_dkk": financial_mtm,
            "financial_mtm_usd": financial_mtm * self.dkk_to_usd_rate,
            "trading_sleeve_dkk": trading_sleeve,
            "trading_sleeve_usd": trading_sleeve * self.dkk_to_usd_rate,
            "distribution_adjusted_trading_sleeve_dkk": distribution_adjusted_trading_sleeve,
            "distribution_adjusted_trading_sleeve_usd": distribution_adjusted_trading_sleeve * self.dkk_to_usd_rate,
            "wind_capacity": self.physical_assets["wind_capacity_mw"],
            "solar_capacity": self.physical_assets["solar_capacity_mw"],
            "hydro_capacity": self.physical_assets["hydro_capacity_mw"],
            "battery_capacity_mwh": self.physical_assets["battery_capacity_mwh"],
            "battery_soc": self.battery_soc,
            "battery_action": battery_action_label,
            "battery_revenue": battery_cash_delta,
            "operational_revenue": operational_revenue,
            "generation_revenue": generation_revenue,
            "cash_return": 0.0,
            "mtm_pnl": self.last_mtm_pnl,
            "transaction_cost": self.last_transaction_cost,
            "cumulative_transaction_costs_dkk": float(self.cumulative_transaction_costs),
            "cumulative_volume_transaction_fees_dkk": float(
                self.cumulative_volume_transaction_fees
            ),
            "cumulative_market_access_fees_dkk": float(self.cumulative_market_access_fees),
            "market_access_fee_step_dkk": float(self._last_market_access_fee),
            "total_traded_notional": traded_notional,
            "target_exposure": float(target_exposure) if target_exposure is not None else np.nan,
            "current_exposure_dkk": self.current_total_exposure_dkk,
            "current_abs_exposure_dkk": self.current_abs_exposure_dkk,
            "financial_exposure_dkk": self.current_abs_exposure_dkk,
            "decision_step": bool(is_decision_step),
            "accumulated_operational_revenue_dkk": self.accumulated_operational_revenue,
            "distribution_amount": float(distribution_amount),
            "total_distributions": self.total_distributions,
            "total_distributions_dkk": self.total_distributions,
            "horizon_settlement_count": int(getattr(self, "_last_horizon_settlement_count", 0)),
            "horizon_settlement_pnl": float(getattr(self, "_last_horizon_settlement_pnl", 0.0)),
            "horizon_settlement_notional": float(getattr(self, "_last_horizon_settlement_notional", 0.0)),
            "horizon_settlement_volume_mwh": float(getattr(self, "_last_horizon_settlement_volume_mwh", 0.0)),
            "horizon_open_volume_mwh": float(getattr(self, "_last_horizon_open_volume_mwh", 0.0)),
            "horizon_unrealized_volume_mwh": float(getattr(self, "_last_horizon_unrealized_volume_mwh", 0.0)),
            "horizon_entry_price": float(getattr(self, "_last_horizon_entry_price", 0.0)),
            "horizon_settlement_price": float(getattr(self, "_last_horizon_settlement_price", 0.0)),
            "horizon_settlement_basis": float(getattr(self, "_last_horizon_settlement_basis", 0.0)),
            "horizon_mark_price": float(getattr(self, "_last_horizon_mark_price", 0.0)),
            "horizon_mark_basis": float(getattr(self, "_last_horizon_mark_basis", 0.0)),
            "horizon_payoff_denominator": float(getattr(self, "_last_horizon_payoff_denominator", 0.0)),
            "mtm_settlement_price_mode": str(getattr(self.config, "mtm_settlement_price_mode", "energy_index")),
            "horizon_roll_cost": float(getattr(self, "_last_horizon_roll_cost", 0.0)),
            "market_impact_cost": float(getattr(self, "_last_market_impact_cost", 0.0)),
            "market_impact_ref_notional_dkk": float(getattr(self, "_last_market_impact_ref_notional", 0.0)),
            "market_impact_participation": float(getattr(self, "_last_market_impact_participation", 0.0)),
            "market_impact_bp": float(getattr(self, "_last_market_impact_bp", 0.0)),
            "liquidity_market_volume_mwh": float(getattr(self, "_last_liquidity_market_volume_mwh", 0.0)),
            "liquidity_causal_market_volume_mwh": float(
                getattr(self, "_last_liquidity_causal_market_volume_mwh", 0.0)
            ),
            "liquidity_volume_cap_mwh": float(getattr(self, "_last_liquidity_volume_cap_mwh", 0.0)),
            "liquidity_requested_volume_mwh": float(getattr(self, "_last_liquidity_requested_volume_mwh", 0.0)),
            "liquidity_executed_volume_mwh": float(getattr(self, "_last_liquidity_executed_volume_mwh", 0.0)),
            "liquidity_participation": float(getattr(self, "_last_liquidity_participation", 0.0)),
            "liquidity_scale": float(getattr(self, "_last_liquidity_scale", 1.0)),
            "liquidity_tail_impact_multiplier": float(getattr(self, "_last_liquidity_tail_impact_multiplier", 1.0)),
            "liquidity_tail_spread_dkk_per_mwh": float(getattr(self, "_last_liquidity_tail_spread_dkk_per_mwh", 0.0)),
            "collateral_required_dkk": float(getattr(self, "_last_collateral_required_dkk", 0.0)),
            "collateral_funding_cost_dkk": float(getattr(self, "_last_collateral_funding_cost_dkk", 0.0)),
            "cumulative_collateral_funding_costs_dkk": float(getattr(self, "cumulative_collateral_funding_costs", 0.0)),
            "collateral_open_notional_dkk": float(getattr(self, "_last_collateral_open_notional_dkk", 0.0)),
            "collateral_open_volume_mwh": float(getattr(self, "_last_collateral_open_volume_mwh", 0.0)),
            "mtm_loss_exit_count": int(getattr(self, "_last_mtm_exit_count", 0)),
            "trading_sleeve_margin_active": bool(self._trading_sleeve_margin_active),
            "trading_sleeve_margin_step": int(self._trading_sleeve_margin_step),
            "trading_sleeve_margin_threshold_dkk": float(self._last_trading_sleeve_margin_threshold_dkk),
            "trading_sleeve_margin_gap_dkk": float(self._last_trading_sleeve_margin_gap_dkk),
            "price": float(self.price[t]),
            "wind_gen": float(self.wind[t]),
            "solar_gen": float(self.solar[t]),
            "hydro_gen": float(self.hydro[t]),
        }
        self.records.append(record)
        return record

    def performance_metrics(self) -> Dict[str, Any]:
        metrics = compute_performance_metrics(
            self.nav_values_usd,
            timestamps=self.data.get("timestamp"),
            annual_risk_free_rate=float(getattr(self.config, "risk_free_rate", self.econ_cfg.annual_risk_free_rate)),
            total_distributions=self.total_distributions * self.dkk_to_usd_rate,
            distribution_adjusted_values=self.adjusted_nav_values_usd,
            value_suffix="usd",
        )
        final_nav = float(self.nav_values_usd[-1]) if self.nav_values_usd else self.initial_budget_usd
        final_adjusted = float(self.adjusted_nav_values_usd[-1]) if self.adjusted_nav_values_usd else final_nav
        metrics.update({
            "final_value_usd": final_nav,
            "initial_value_usd": self.initial_budget_usd,
            "distribution_adjusted_final_value_usd": final_adjusted,
            "final_portfolio_value": final_adjusted,
            "initial_portfolio_value": self.initial_budget_usd,
            "total_distributions_usd": float(self.total_distributions * self.dkk_to_usd_rate),
            "hybrid_benchmark_contract": True,
            "physical_sleeve_initial_usd": float(self.physical_book_initial * self.dkk_to_usd_rate),
            "trading_sleeve_initial_usd": float(self.trading_allocation_budget * self.dkk_to_usd_rate),
            "physical_allocation": float(getattr(self.config, "physical_allocation", 0.88)),
            "financial_allocation": float(getattr(self.config, "financial_allocation", 0.12)),
            "final_trading_cash_usd": float(self.budget * self.dkk_to_usd_rate),
            "final_financial_mtm_usd": float(sum(self.financial_mtm_positions.values()) * self.dkk_to_usd_rate),
            "total_generation_revenue_usd": float(self.cumulative_generation_revenue * self.dkk_to_usd_rate),
            "total_operational_revenue_usd": float(self.accumulated_operational_revenue * self.dkk_to_usd_rate),
            "total_battery_revenue_usd": float(self.cumulative_battery_revenue * self.dkk_to_usd_rate),
            "total_mtm_pnl_usd": float(self.cumulative_mtm_pnl * self.dkk_to_usd_rate),
            "total_transaction_costs_usd": float(self.cumulative_transaction_costs * self.dkk_to_usd_rate),
            "total_volume_transaction_fees_usd": float(
                self.cumulative_volume_transaction_fees * self.dkk_to_usd_rate
            ),
            "total_market_access_fees_usd": float(
                self.cumulative_market_access_fees * self.dkk_to_usd_rate
            ),
            "total_market_impact_cost_usd": float(self.cumulative_market_impact_costs * self.dkk_to_usd_rate),
            "total_collateral_funding_cost_usd": float(self.cumulative_collateral_funding_costs * self.dkk_to_usd_rate),
            "mtm_return_model": str(getattr(self.config, "mtm_return_model", "")),
            "mtm_reference_price_dkk_per_mwh": float(getattr(self.config, "mtm_reference_price_dkk_per_mwh", 500.0)),
            "mtm_settlement_horizon_steps": int(getattr(self.config, "mtm_settlement_horizon_steps", 0) or 0),
            "mtm_entry_price_mode": str(getattr(self.config, "mtm_entry_price_mode", "")),
            "mtm_horizon_payoff_denominator_mode": str(getattr(self.config, "mtm_horizon_payoff_denominator_mode", "")),
            "mtm_settlement_price_mode": str(getattr(self.config, "mtm_settlement_price_mode", "energy_index")),
            "mtm_basis_price_data_path": str(getattr(self.config, "mtm_basis_price_data_path", "")),
            "mtm_basis_scale": float(getattr(self.config, "mtm_basis_scale", 0.0)),
            "mtm_basis_centering_mode": str(getattr(self.config, "mtm_basis_centering_mode", "")),
            "mtm_basis_component_std_dkk_per_mwh": float(np.nanstd(self.settlement_basis_component))
            if len(self.settlement_basis_component)
            else 0.0,
            "investor_notional_sizing_base": str(getattr(self.config, "investor_notional_sizing_base", "")),
            "max_position_size": float(getattr(self.config, "max_position_size", self.max_position_size)),
            "capital_allocation_fraction": float(getattr(self.config, "capital_allocation_fraction", self.capital_allocation_fraction)),
            "friction_cost_multiplier": float(getattr(self.config, "friction_cost_multiplier", 1.0) or 1.0),
            "market_fee_model": str(getattr(self.config, "market_fee_model", "legacy_notional_fixed")),
            "transaction_fee_dkk_per_mwh": float(
                getattr(self.config, "transaction_fee_dkk_per_mwh", 0.0)
            ),
            "annual_market_access_fee_dkk": float(
                getattr(self.config, "annual_market_access_fee_dkk", 0.0)
            ),
            "market_access_fee_allocation_fraction": float(
                getattr(self.config, "market_access_fee_allocation_fraction", 1.0)
            ),
            "market_fee_source_id": str(
                getattr(self.config, "market_fee_source_id", "legacy_unsourced_fixed_fee")
            ),
            "half_spread_bp": float(getattr(self.config, "half_spread_bp", 0.0) or 0.0),
            "impact_coef_bp": float(getattr(self.config, "impact_coef_bp", 0.0) or 0.0),
            "impact_exponent": float(getattr(self.config, "impact_exponent", 0.5) or 0.5),
            "impact_ref_notional": str(getattr(self.config, "impact_ref_notional", "")),
            "liquidity_participation_cap_fraction": float(getattr(self.config, "liquidity_participation_cap_fraction", 0.0)),
            "liquidity_volume_source": str(getattr(self.config, "liquidity_volume_source", "")),
            "liquidity_volume_multiplier": float(getattr(self.config, "liquidity_volume_multiplier", 1.0)),
            "liquidity_min_volume_mwh": float(getattr(self.config, "liquidity_min_volume_mwh", 1.0)),
            "liquidity_tail_impact_threshold_dkk_per_mwh": float(getattr(self.config, "liquidity_tail_impact_threshold_dkk_per_mwh", 5000.0)),
            "liquidity_tail_impact_multiplier": float(getattr(self.config, "liquidity_tail_impact_multiplier", 1.0)),
            "liquidity_tail_impact_power": float(getattr(self.config, "liquidity_tail_impact_power", 1.0)),
            "liquidity_tail_impact_max_multiplier": float(getattr(self.config, "liquidity_tail_impact_max_multiplier", 10.0)),
            "enable_collateral_cash_drag": bool(getattr(self.config, "enable_collateral_cash_drag", False)),
            "collateral_notional_margin_fraction": float(getattr(self.config, "collateral_notional_margin_fraction", 0.02)),
            "collateral_stress_loss_fraction": float(getattr(self.config, "collateral_stress_loss_fraction", 0.10)),
            "collateral_stress_price_dkk_per_mwh": float(getattr(self.config, "collateral_stress_price_dkk_per_mwh", 25000.0)),
            "collateral_funding_rate_annual": float(getattr(self.config, "collateral_funding_rate_annual", 0.05)),
            "collateral_tradeable_haircut": float(getattr(self.config, "collateral_tradeable_haircut", 1.0)),
            "investment_freq": int(getattr(self.config, "investment_freq", self.investment_freq) or self.investment_freq),
            "wind_capacity_mw": float(self.physical_assets["wind_capacity_mw"]),
            "solar_capacity_mw": float(self.physical_assets["solar_capacity_mw"]),
            "hydro_capacity_mw": float(self.physical_assets["hydro_capacity_mw"]),
            "battery_capacity_mwh": float(self.physical_assets["battery_capacity_mwh"]),
            "final_battery_soc": float(self.battery_soc),
            "trading_sleeve_margin_active_any": bool(self._trading_sleeve_margin_active),
            "trading_sleeve_margin_first_step": int(self._trading_sleeve_margin_step),
            "trading_sleeve_margin_count": int(self._trading_sleeve_margin_count),
            "trading_sleeve_margin_threshold_dkk": float(self._last_trading_sleeve_margin_threshold_dkk),
            "trading_sleeve_margin_gap_dkk": float(self._last_trading_sleeve_margin_gap_dkk),
            "mtm_loss_exit_total_count": int(getattr(self, "_mtm_loss_exit_count", 0)),
            "mtm_loss_exit_first_step": int(getattr(self, "_mtm_loss_exit_first_step", -1)),
        })
        metrics.update(compute_trading_sleeve_metrics(
            self.records,
            dkk_to_usd_rate=self.dkk_to_usd_rate,
            annual_risk_free_rate=float(getattr(self.config, "risk_free_rate", self.econ_cfg.annual_risk_free_rate)),
            primary_sharpe_mode=str(getattr(self.config, "sleeve_sharpe_mode", "daily_hac_7")),
            initial_trading_sleeve_usd=float(self.trading_allocation_budget * self.dkk_to_usd_rate),
        ))
        return metrics

"""Market-access and execution-fee protocol shared by all paper methods.

The primary protocol is a transparent execution-cost proxy based on Nord Pool's
2026 Nordic/Baltic fee schedule for a standard participant in the intraday
continuous market. It is not claimed to reproduce every bilateral BRP or
balancing-settlement tariff. Spread, market impact, collateral, and funding
costs are modelled separately by the environment.

Official source (effective 1 January 2026):
https://www.nordpoolgroup.com/4a7ad8/globalassets/trading-and-services/
nord-pool-fee-schedule-2026-nordic-and-baltic-market.pdf
"""

from __future__ import annotations

from typing import Any, Dict


LEGACY_FEE_MODEL = "legacy_notional_fixed"
NORD_POOL_INTRADAY_2026_FEE_MODEL = "nord_pool_intraday_2026"
SUPPORTED_MARKET_FEE_MODELS = {
    LEGACY_FEE_MODEL,
    NORD_POOL_INTRADAY_2026_FEE_MODEL,
}

NORD_POOL_FEE_SOURCE_ID = "nord_pool_nordic_baltic_2026_standard_participant"
NORD_POOL_FEE_SOURCE_URL = (
    "https://www.nordpoolgroup.com/4a7ad8/globalassets/trading-and-services/"
    "nord-pool-fee-schedule-2026-nordic-and-baltic-market.pdf"
)
NORD_POOL_FEE_EFFECTIVE_DATE = "2026-01-01"

# Fixed in the experiment contract so all seeds and comparators use identical
# DKK costs instead of a time-varying foreign-exchange series.
NORD_POOL_PROTOCOL_EUR_DKK = 7.45
NORD_POOL_INTRADAY_COMBINED_FEE_EUR_PER_MWH = 0.124
NORD_POOL_STANDARD_ACCESS_FEE_EUR_PER_YEAR = 21_500.0
NORD_POOL_INTRADAY_COMBINED_FEE_DKK_PER_MWH = (
    NORD_POOL_INTRADAY_COMBINED_FEE_EUR_PER_MWH * NORD_POOL_PROTOCOL_EUR_DKK
)
NORD_POOL_STANDARD_ACCESS_FEE_DKK_PER_YEAR = (
    NORD_POOL_STANDARD_ACCESS_FEE_EUR_PER_YEAR * NORD_POOL_PROTOCOL_EUR_DKK
)

HOURS_PER_YEAR = 365.25 * 24.0


def normalize_market_fee_model(value: Any) -> str:
    model = str(value or LEGACY_FEE_MODEL).strip().lower().replace("-", "_")
    aliases = {
        "legacy": LEGACY_FEE_MODEL,
        "fixed": LEGACY_FEE_MODEL,
        "notional_fixed": LEGACY_FEE_MODEL,
        "nord_pool": NORD_POOL_INTRADAY_2026_FEE_MODEL,
        "nordpool": NORD_POOL_INTRADAY_2026_FEE_MODEL,
        "nord_pool_intraday": NORD_POOL_INTRADAY_2026_FEE_MODEL,
    }
    return aliases.get(model, model)


def base_execution_fee_components(
    *,
    model: Any,
    abs_notional_dkk: float,
    abs_volume_mwh: float,
    friction_multiplier: float,
    transaction_cost_bps: float,
    transaction_fixed_cost_dkk: float,
    transaction_fee_dkk_per_mwh: float,
) -> Dict[str, float]:
    """Return action-dependent base fees before spread and market impact."""
    normalized = normalize_market_fee_model(model)
    if normalized not in SUPPORTED_MARKET_FEE_MODELS:
        raise ValueError(
            f"Unsupported market_fee_model={model!r}; expected one of "
            f"{sorted(SUPPORTED_MARKET_FEE_MODELS)}"
        )

    notional = max(float(abs_notional_dkk), 0.0)
    volume = max(float(abs_volume_mwh), 0.0)
    multiplier = max(float(friction_multiplier), 0.0)

    if normalized == NORD_POOL_INTRADAY_2026_FEE_MODEL:
        volume_fee = volume * max(float(transaction_fee_dkk_per_mwh), 0.0) * multiplier
        legacy_notional_fee = 0.0
        legacy_fixed_fee = 0.0
    else:
        volume_fee = 0.0
        legacy_notional_fee = (
            notional * max(float(transaction_cost_bps), 0.0) / 10_000.0 * multiplier
        )
        legacy_fixed_fee = max(float(transaction_fixed_cost_dkk), 0.0) * multiplier

    base_fee = volume_fee + legacy_notional_fee + legacy_fixed_fee
    return {
        "base_execution_fee_dkk": float(base_fee),
        "volume_transaction_fee_dkk": float(volume_fee),
        "legacy_notional_fee_dkk": float(legacy_notional_fee),
        "legacy_fixed_fee_dkk": float(legacy_fixed_fee),
    }


def market_access_fee_for_step(
    *,
    annual_fee_dkk: float,
    allocation_fraction: float,
    time_step_hours: float,
) -> float:
    """Pro-rate annual market access to one simulation step."""
    annual = max(float(annual_fee_dkk), 0.0)
    allocation = min(max(float(allocation_fraction), 0.0), 1.0)
    hours = max(float(time_step_hours), 0.0)
    return float(annual * allocation * hours / HOURS_PER_YEAR)

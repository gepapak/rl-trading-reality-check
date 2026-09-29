#!/usr/bin/env python3
"""Exercise the strict MWh liquidity, impact, and execution path without training."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EnhancedConfig
from environment import RenewableMultiAgentEnv
from evaluation import _compute_sleeve_metrics_from_env_log
from logger import RewardLogger
from market_fee_protocol import (
    NORD_POOL_FEE_SOURCE_ID,
    NORD_POOL_INTRADAY_COMBINED_FEE_DKK_PER_MWH,
    NORD_POOL_STANDARD_ACCESS_FEE_DKK_PER_YEAR,
    market_access_fee_for_step,
)
from runtime_contract import execution_contract_settings


def _config() -> EnhancedConfig:
    cfg = EnhancedConfig()
    cfg.seed = 7
    cfg.current_episode_num = 0
    cfg.enable_forecast_utilization = False
    cfg.investment_freq = 6
    cfg.mtm_return_model = "horizon_settlement"
    cfg.mtm_reference_price_dkk_per_mwh = 500.0
    cfg.mtm_settlement_horizon_steps = 6
    cfg.mtm_entry_price_mode = "current_price"
    cfg.mtm_horizon_payoff_denominator_mode = "mwh_volume"
    cfg.mtm_settlement_price_mode = "external_series"
    cfg.mtm_external_settlement_price_data_path = str(
        ROOT / "settlement_price_dataset_real_v2" / "scenario_000.csv"
    )
    cfg.mtm_external_settlement_price_column = "settlement_price"
    cfg.mtm_external_settlement_timestamp_column = "timestamp"
    cfg.mtm_external_settlement_min_price_dkk_per_mwh = -111750.0
    cfg.mtm_external_settlement_max_price_dkk_per_mwh = 111750.0
    cfg.impact_ref_notional = "volume"
    cfg.impact_coef_bp = 20.0
    cfg.impact_exponent = 0.5
    cfg.impact_volume_data_path = str(
        ROOT / "liquidity_volume_dataset_real_v1" / "scenario_000.csv"
    )
    cfg.impact_volume_column = "market_volume_mwh"
    cfg.impact_volume_unit = "mwh"
    cfg.impact_volume_timestamp_column = "timestamp"
    cfg.impact_volume_max_staleness_minutes = 90.0
    cfg.liquidity_volume_source = "impact_volume"
    cfg.liquidity_participation_cap_fraction = 0.25
    cfg.liquidity_volume_multiplier = 1.0
    cfg.liquidity_min_volume_mwh = 1.0
    cfg.investor_notional_sizing_base = "initial_trading_sleeve"
    cfg.max_position_size = 0.10
    cfg.capital_allocation_fraction = 0.60
    cfg.no_trade_threshold = 0.01
    cfg.no_trade_threshold_reference = "executable_capacity"
    cfg.market_fee_model = "nord_pool_intraday_2026"
    cfg.transaction_fee_dkk_per_mwh = NORD_POOL_INTRADAY_COMBINED_FEE_DKK_PER_MWH
    cfg.annual_market_access_fee_dkk = NORD_POOL_STANDARD_ACCESS_FEE_DKK_PER_YEAR
    cfg.market_access_fee_allocation_fraction = 1.0
    cfg.market_fee_source_id = NORD_POOL_FEE_SOURCE_ID
    return cfg


def main() -> int:
    data = pd.read_csv(
        ROOT / "training_dataset_ffill" / "scenario_000.csv",
        low_memory=False,
    ).head(30)
    cfg = _config()
    env = RenewableMultiAgentEnv(data, investment_freq=6, config=cfg)
    env.reset(seed=7)

    traded = float(
        env._execute_investor_trades(
            np.asarray([1.0], dtype=np.float32),
            timestep=6,
        )
    )
    contracts = list(env._horizon_contracts)
    threshold = float(getattr(env, "_last_no_trade_threshold_dkk", 0.0))
    cap_mwh = float(getattr(env, "_last_liquidity_volume_cap_mwh", 0.0))
    market_mwh = float(getattr(env, "_last_liquidity_market_volume_mwh", 0.0))
    causal_market_mwh = float(
        getattr(env, "_last_liquidity_causal_market_volume_mwh", 0.0)
    )

    if traded <= threshold or not contracts:
        raise AssertionError(
            "A feasible liquidity-capped investor action did not open a settlement contract: "
            f"traded={traded:.6f}, threshold={threshold:.6f}, contracts={len(contracts)}"
        )
    if cap_mwh <= 0.0 or market_mwh <= 0.0:
        raise AssertionError("Real-liquidity diagnostics were not populated")

    last = contracts[-1]
    contract_mwh = float(sum(abs(v) for v in last["volumes_mwh"].values()))
    contract_notional = float(sum(abs(v) for v in last["exposures"].values()))
    mwh_participation = contract_mwh / market_mwh
    notional_participation = contract_notional / env._market_impact_reference_notional(6)
    if not np.isclose(mwh_participation, notional_participation, atol=1e-12, rtol=0.0):
        raise AssertionError(
            "Impact participation is not dimensionally aligned with MWh participation: "
            f"mwh={mwh_participation:.12f}, notional={notional_participation:.12f}"
        )
    if mwh_participation > cfg.liquidity_participation_cap_fraction + 1e-12:
        raise AssertionError("Executed volume exceeded the hard participation cap")
    expected_volume_fee = contract_mwh * cfg.transaction_fee_dkk_per_mwh
    actual_volume_fee = float(getattr(env, "_last_volume_transaction_fee", 0.0))
    if not np.isclose(actual_volume_fee, expected_volume_fee, atol=1e-9, rtol=0.0):
        raise AssertionError(
            "Executed contract did not use the MWh-scaled market fee: "
            f"actual={actual_volume_fee:.12f}, expected={expected_volume_fee:.12f}"
        )
    expected_access_fee = market_access_fee_for_step(
        annual_fee_dkk=cfg.annual_market_access_fee_dkk,
        allocation_fraction=cfg.market_access_fee_allocation_fraction,
        time_step_hours=cfg.time_step_hours,
    )
    charged_access_fee = float(env._apply_market_access_fee(7))
    duplicate_access_fee = float(env._apply_market_access_fee(7))
    if not np.isclose(charged_access_fee, expected_access_fee, atol=1e-12, rtol=0.0):
        raise AssertionError("Annual access fee was not pro-rated correctly in the environment")
    if duplicate_access_fee != 0.0:
        raise AssertionError("Annual access fee was charged twice for one environment step")

    contract = execution_contract_settings(cfg)
    if contract.get("no_trade_threshold") != 0.01:
        raise AssertionError("no_trade_threshold is missing from the runtime contract")
    if contract.get("no_trade_threshold_reference") != "executable_capacity":
        raise AssertionError("no_trade_threshold_reference is missing from the runtime contract")
    if contract.get("market_fee_model") != "nord_pool_intraday_2026":
        raise AssertionError("market_fee_model is missing from the runtime contract")

    cfg.no_trade_threshold = 1000.0
    raw_threshold = env._execution_no_trade_threshold_dkk(39_724_137.0, cap_mwh)
    if raw_threshold != 1000.0:
        raise AssertionError("Raw-DKK no-trade threshold compatibility is broken")

    with tempfile.TemporaryDirectory(prefix="prototype3_liquidity_log_") as tmp:
        reward_logger = RewardLogger(log_dir=tmp, tier_name="smoke", enabled=True)
        reward_logger.start_episode(0)
        reward_logger.log_step(
            timestep=0,
            liquidity_market_volume_mwh=market_mwh,
            liquidity_causal_market_volume_mwh=causal_market_mwh,
            liquidity_volume_cap_mwh=cap_mwh,
            liquidity_participation=mwh_participation,
            market_impact_ref_notional_dkk=env._market_impact_reference_notional(6),
            market_impact_participation=notional_participation,
            no_trade_threshold_dkk=threshold,
            no_trade_threshold_reference_dkk=float(
                getattr(env, "_last_no_trade_threshold_reference_dkk", 0.0)
            ),
            cumulative_volume_transaction_fees_dkk=actual_volume_fee,
            cumulative_market_access_fees_dkk=float(env.cumulative_market_access_fees),
            market_access_fee_step_dkk=charged_access_fee,
        )
        reward_logger.close()
        logged = pd.read_csv(Path(tmp) / "smoke_debug_ep0.csv")
        required = {
            "liquidity_market_volume_mwh",
            "liquidity_causal_market_volume_mwh",
            "liquidity_volume_cap_mwh",
            "liquidity_participation",
            "market_impact_ref_notional_dkk",
            "market_impact_participation",
            "no_trade_threshold_dkk",
            "no_trade_threshold_reference_dkk",
            "cumulative_volume_transaction_fees_dkk",
            "cumulative_market_access_fees_dkk",
            "market_access_fee_step_dkk",
        }
        missing = sorted(required.difference(logged.columns))
        if missing:
            raise AssertionError(f"Detailed financial diagnostics were not serialized: {missing}")
        if not np.isclose(
            float(logged.loc[0, "liquidity_participation"]),
            mwh_participation,
        ):
            raise AssertionError("Serialized liquidity participation is incorrect")
        detailed = _compute_sleeve_metrics_from_env_log(
            str(Path(tmp) / "smoke_debug_ep0.csv"),
            primary_sharpe_mode="daily_hac_7",
        )
        if "sleeve_market_impact_max_participation" not in detailed:
            raise AssertionError("Evaluation did not aggregate market-impact participation")
        if "sleeve_no_trade_threshold_max_dkk" not in detailed:
            raise AssertionError("Evaluation did not aggregate no-trade threshold diagnostics")
        if "sleeve_total_volume_transaction_fees_usd" not in detailed:
            raise AssertionError("Evaluation did not aggregate MWh transaction fees")
        if "sleeve_total_market_access_fees_usd" not in detailed:
            raise AssertionError("Evaluation did not aggregate annual market-access fees")

    print(
        "PASS: traded={:.2f} DKK threshold={:.2f} DKK cap={:.3f} MWh "
        "participation={:.3%} contracts={}".format(
            traded,
            threshold,
            cap_mwh,
            mwh_participation,
            len(contracts),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

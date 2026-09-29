"""Regression checks for the frozen same-delivery causal v3 data protocol."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASELINES_ROOT = ROOT / "baselines"
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from environment import RenewableMultiAgentEnv
from baseline_common import HybridFundLedger
from forecast_price_experts import PriceShortExpertBank, _create_horizon_windows
from policy import CentralizedCriticRolloutBuffer
from scripts.build_real_settlement_protocol_v2 import _align_settlement_to_template


def _minimal_environment() -> RenewableMultiAgentEnv:
    env = object.__new__(RenewableMultiAgentEnv)
    env.config = SimpleNamespace(
        mtm_return_model="horizon_settlement",
        mtm_settlement_horizon_steps=6,
        mtm_horizon_payoff_denominator_mode="mwh_volume",
        mtm_reference_price_dkk_per_mwh=500.0,
        mtm_entry_price_mode="current_price",
        mtm_external_settlement_min_price_dkk_per_mwh=-111750.0,
        mtm_external_settlement_max_price_dkk_per_mwh=111750.0,
        transaction_cost_bps=0.0,
        transaction_fixed_cost=0.0,
        friction_cost_multiplier=0.0,
        half_spread_bp=0.0,
        impact_coef_bp=0.0,
        impact_exponent=0.5,
        impact_ref_notional="volume",
        liquidity_volume_source="impact_volume",
        liquidity_volume_multiplier=1.0,
        liquidity_min_volume_mwh=1.0,
        liquidity_tail_impact_multiplier=3.0,
        liquidity_tail_impact_threshold_dkk_per_mwh=100.0,
        liquidity_tail_impact_power=1.0,
        liquidity_tail_impact_max_multiplier=10.0,
        time_step_hours=10.0 / 60.0,
    )
    env._price_raw = np.arange(12, dtype=np.float64) * 10.0 + 10.0
    env._horizon_settlement_price_raw = np.arange(12, dtype=np.float64) * 100.0 + 100.0
    env._horizon_settlement_basis_component = (
        env._horizon_settlement_price_raw - env._price_raw
    )
    env._impact_volume_mwh_by_step = np.arange(12, dtype=np.float64) + 10.0
    env.budget = 1_000_000.0
    env.cumulative_transaction_costs = 0.0
    env.cumulative_market_impact_costs = 0.0
    env._last_investor_transaction_cost = 0.0
    env._last_market_impact_cost = 0.0
    env._last_market_impact_ref_notional = 0.0
    env._last_market_impact_participation = 0.0
    env._last_market_impact_bp = 0.0
    env._reset_horizon_settlement_contracts()
    return env


def _check_forecast_information_set() -> None:
    bank = object.__new__(PriceShortExpertBank)
    bank.look_back = 3
    bank.artifacts = {
        "ann": SimpleNamespace(metadata={"training_price_mean": -7.0})
    }
    values = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    windows = bank._build_series_windows(values)
    expected = np.asarray(
        [
            [-7.0, -7.0, -7.0],
            [-7.0, -7.0, 10.0],
            [-7.0, 10.0, 20.0],
            [10.0, 20.0, 30.0],
        ],
        dtype=np.float32,
    )
    if not np.array_equal(windows, expected):
        raise AssertionError(f"Forecast windows are not strictly causal:\n{windows}")

    settlement = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    entry = np.asarray([101.0, 102.0, 103.0, 104.0], dtype=np.float32)
    _, targets, anchors = _create_horizon_windows(
        settlement,
        look_back=2,
        horizon_steps=1,
        reference_series=entry,
    )
    if not np.array_equal(targets, np.asarray([3.0, 4.0], dtype=np.float32)):
        raise AssertionError("Same-delivery settlement targets are wrong")
    if not np.array_equal(anchors, np.asarray([103.0, 104.0], dtype=np.float32)):
        raise AssertionError("Direction labels are not anchored to the delivery entry price")


def _check_delivery_clock_and_liquidity() -> None:
    env = _minimal_environment()
    exposure = {"wind_instrument_value": 5_000.0}
    env._open_horizon_settlement_contract(2, exposure)
    if len(env._horizon_contracts) != 1:
        raise AssertionError("Synthetic contract was not opened")
    contract = env._horizon_contracts[0]
    if contract.get("delivery_step") != 2 or contract.get("settle_step") != 8:
        raise AssertionError(f"Contract clock is wrong: {contract}")

    premature_pnl, _ = env._settle_horizon_contracts_for_step(7)
    if premature_pnl != 0.0 or len(env._horizon_contracts) != 1:
        raise AssertionError("Contract settled before its maturity clock")
    mark_pnl, _ = env._mark_horizon_contracts_to_market_for_step(7)
    if mark_pnl != 0.0:
        raise AssertionError("Final settlement leaked into open-contract MTM")

    before = float(env.budget)
    pnl, _ = env._settle_horizon_contracts_for_step(8)
    expected_pnl = (5_000.0 / 500.0) * (300.0 - 30.0)
    wrong_delayed_pnl = (5_000.0 / 500.0) * (900.0 - 30.0)
    if not np.isclose(pnl, expected_pnl):
        raise AssertionError(f"Expected same-delivery PnL {expected_pnl}, got {pnl}")
    if np.isclose(pnl, wrong_delayed_pnl):
        raise AssertionError("PnL still uses the maturity-step settlement price")
    if not np.isclose(env.budget - before, expected_pnl):
        raise AssertionError("Settled PnL was not booked exactly once")

    # At action step 9, step 2 is the latest strictly observed delivery.
    causal_volume = env._causal_market_liquidity_volume_mwh(9)
    if not np.isclose(causal_volume, 12.0):
        raise AssertionError(f"Causal liquidity used the wrong delivery: {causal_volume}")
    tail_multiplier = env._liquidity_tail_impact_multiplier(9)
    expected_multiplier = 1.0 + 2.0 * ((abs(300.0 - 30.0) / 100.0) - 1.0)
    if not np.isclose(tail_multiplier, expected_multiplier):
        raise AssertionError("Tail-impact state is not based on the latest observed delivery")


def _check_strict_settlement_alignment() -> None:
    env = _minimal_environment()
    timestamps = pd.date_range("2025-01-01", periods=4, freq="10min")
    env.data = pd.DataFrame({"timestamp": timestamps.astype(str)})
    with tempfile.TemporaryDirectory(prefix="same_delivery_alignment_") as tmp:
        path = Path(tmp) / "settlement.csv"
        pd.DataFrame(
            {
                "timestamp": timestamps[:3].astype(str),
                "settlement_price": [100.0, 101.0, 102.0],
            }
        ).to_csv(path, index=False)
        env.config.mtm_external_settlement_price_data_path = str(path)
        env.config.mtm_external_settlement_price_column = "settlement_price"
        env.config.mtm_external_settlement_timestamp_column = "timestamp"
        try:
            env._load_aligned_external_settlement_series(np.ones(4, dtype=np.float64))
        except ValueError as exc:
            if "incomplete" not in str(exc).lower():
                raise
        else:
            raise AssertionError("Incomplete settlement coverage was silently filled")

    template = pd.DataFrame({"timestamp": timestamps.astype(str)})
    source = pd.DataFrame(
        {
            "timestamp": [timestamps[0]],
            "settlement_price": [100.0],
            "settlement_source": ["synthetic"],
            "settlement_price_area": ["DK1"],
            "settlement_price_rule": ["smoke"],
        }
    )
    aligned = _align_settlement_to_template(
        template,
        Path("synthetic.csv"),
        source,
        tolerance_minutes=75.0,
    )
    if float(aligned.attrs.get("alignment_coverage", 0.0)) != 1.0:
        raise AssertionError("Settlement builder did not record complete alignment")
    try:
        _align_settlement_to_template(
            template,
            Path("synthetic.csv"),
            source,
            tolerance_minutes=5.0,
        )
    except RuntimeError as exc:
        if "no forward/backward filling" not in str(exc).lower():
            raise
    else:
        raise AssertionError("Settlement builder accepted incomplete source coverage")


def _check_baseline_delivery_clock() -> None:
    ledger = object.__new__(HybridFundLedger)
    ledger.config = SimpleNamespace(
        mtm_return_model="horizon_settlement",
        mtm_settlement_horizon_steps=6,
        mtm_horizon_payoff_denominator_mode="mwh_volume",
        mtm_reference_price_dkk_per_mwh=500.0,
    )
    ledger.price = np.arange(12, dtype=np.float64) * 10.0 + 10.0
    ledger.settlement_price = np.arange(12, dtype=np.float64) * 100.0 + 100.0
    ledger.settlement_basis_component = ledger.settlement_price - ledger.price
    ledger.budget = 1_000_000.0
    ledger._horizon_contracts = [
        {
            "open_step": 2,
            "delivery_step": 2,
            "settle_step": 8,
            "entry_price": 30.0,
            "exposures": {"wind_instrument_value": 5_000.0},
            "volumes_mwh": {"wind_instrument_value": 10.0},
        }
    ]
    pnl, _ = ledger._settle_horizon_contracts_for_step(8)
    expected = 10.0 * (300.0 - 30.0)
    delayed = 10.0 * (900.0 - 30.0)
    if not np.isclose(pnl, expected) or np.isclose(pnl, delayed):
        raise AssertionError(
            f"Baseline ledger is not same-delivery aligned: pnl={pnl}, expected={expected}"
        )


def _check_actor_mask_buffer() -> None:
    try:
        from gymnasium import spaces
    except ImportError:
        from gym import spaces

    buffer = CentralizedCriticRolloutBuffer(
        buffer_size=6,
        observation_space=spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        device="cpu",
        gae_lambda=0.95,
        gamma=0.99,
        n_envs=1,
        central_obs_dim=3,
    )
    for pos, value in enumerate([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        buffer.add_action_mask(pos, value)
    if not np.array_equal(buffer.action_masks[:, 0], np.asarray([1, 0, 0, 0, 0, 0])):
        raise AssertionError("MAPPO actor decision mask was not stored correctly")


def main() -> int:
    _check_forecast_information_set()
    _check_delivery_clock_and_liquidity()
    _check_strict_settlement_alignment()
    _check_baseline_delivery_clock()
    _check_actor_mask_buffer()
    print("PASS: same-delivery causal v3 settlement, forecasts, liquidity, alignment, and actor mask")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

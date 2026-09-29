"""Unit tests for the paper market-fee contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market_fee_protocol import (
    HOURS_PER_YEAR,
    NORD_POOL_INTRADAY_2026_FEE_MODEL,
    NORD_POOL_INTRADAY_COMBINED_FEE_DKK_PER_MWH,
    NORD_POOL_STANDARD_ACCESS_FEE_DKK_PER_YEAR,
    base_execution_fee_components,
    market_access_fee_for_step,
)


class MarketFeeProtocolTests(unittest.TestCase):
    def test_nord_pool_fee_scales_with_mwh_not_notional(self) -> None:
        low_notional = base_execution_fee_components(
            model=NORD_POOL_INTRADAY_2026_FEE_MODEL,
            abs_notional_dkk=500.0,
            abs_volume_mwh=10.0,
            friction_multiplier=1.0,
            transaction_cost_bps=99.0,
            transaction_fixed_cost_dkk=9999.0,
            transaction_fee_dkk_per_mwh=NORD_POOL_INTRADAY_COMBINED_FEE_DKK_PER_MWH,
        )
        high_notional = base_execution_fee_components(
            model=NORD_POOL_INTRADAY_2026_FEE_MODEL,
            abs_notional_dkk=5_000_000.0,
            abs_volume_mwh=10.0,
            friction_multiplier=1.0,
            transaction_cost_bps=99.0,
            transaction_fixed_cost_dkk=9999.0,
            transaction_fee_dkk_per_mwh=NORD_POOL_INTRADAY_COMBINED_FEE_DKK_PER_MWH,
        )
        expected = 10.0 * NORD_POOL_INTRADAY_COMBINED_FEE_DKK_PER_MWH
        self.assertAlmostEqual(low_notional["base_execution_fee_dkk"], expected, places=12)
        self.assertAlmostEqual(high_notional["base_execution_fee_dkk"], expected, places=12)
        self.assertEqual(low_notional["legacy_fixed_fee_dkk"], 0.0)
        self.assertEqual(low_notional["legacy_notional_fee_dkk"], 0.0)

    def test_friction_multiplier_is_an_explicit_stress_multiplier(self) -> None:
        components = base_execution_fee_components(
            model=NORD_POOL_INTRADAY_2026_FEE_MODEL,
            abs_notional_dkk=5_000.0,
            abs_volume_mwh=10.0,
            friction_multiplier=2.0,
            transaction_cost_bps=0.5,
            transaction_fixed_cost_dkk=172.0,
            transaction_fee_dkk_per_mwh=NORD_POOL_INTRADAY_COMBINED_FEE_DKK_PER_MWH,
        )
        self.assertAlmostEqual(
            components["base_execution_fee_dkk"],
            20.0 * NORD_POOL_INTRADAY_COMBINED_FEE_DKK_PER_MWH,
            places=12,
        )

    def test_legacy_model_is_backward_compatible(self) -> None:
        components = base_execution_fee_components(
            model="legacy_notional_fixed",
            abs_notional_dkk=1_000_000.0,
            abs_volume_mwh=10.0,
            friction_multiplier=2.0,
            transaction_cost_bps=0.5,
            transaction_fixed_cost_dkk=172.0,
            transaction_fee_dkk_per_mwh=999.0,
        )
        self.assertAlmostEqual(components["legacy_notional_fee_dkk"], 100.0)
        self.assertAlmostEqual(components["legacy_fixed_fee_dkk"], 344.0)
        self.assertAlmostEqual(components["base_execution_fee_dkk"], 444.0)
        self.assertEqual(components["volume_transaction_fee_dkk"], 0.0)

    def test_step_access_fees_sum_to_annual_fee(self) -> None:
        step_hours = 10.0 / 60.0
        per_step = market_access_fee_for_step(
            annual_fee_dkk=NORD_POOL_STANDARD_ACCESS_FEE_DKK_PER_YEAR,
            allocation_fraction=1.0,
            time_step_hours=step_hours,
        )
        steps_per_year = HOURS_PER_YEAR / step_hours
        self.assertAlmostEqual(
            per_step * steps_per_year,
            NORD_POOL_STANDARD_ACCESS_FEE_DKK_PER_YEAR,
            places=7,
        )

    def test_access_allocation_fraction_is_proportional(self) -> None:
        full = market_access_fee_for_step(
            annual_fee_dkk=160_175.0,
            allocation_fraction=1.0,
            time_step_hours=1.0,
        )
        partial = market_access_fee_for_step(
            annual_fee_dkk=160_175.0,
            allocation_fraction=0.25,
            time_step_hours=1.0,
        )
        self.assertAlmostEqual(partial, 0.25 * full, places=12)


if __name__ == "__main__":
    unittest.main()

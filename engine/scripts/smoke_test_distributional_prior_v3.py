#!/usr/bin/env python3
"""Synthetic smoke test for the FoCAL v3 distributional prior.

The test is deliberately small and deterministic. It checks the mechanism
without training MAPPO:

1. a positive high-confidence forecast bucket earns positive exposure;
2. a negative-edge bucket is shut down;
3. the disaster tail budget prevents exposure from exceeding its cap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distributional_forecast_prior import DistributionalTailBudgetSizer


def main() -> int:
    rng = np.random.default_rng(7)
    sizer = DistributionalTailBudgetSizer(
        window_size=2000,
        min_samples=40,
        tail_quantile=0.99,
        disaster_quantile=0.999,
        conditional_loss_budget_fraction=0.05,
        disaster_loss_budget_fraction=0.20,
        conditional_tail_floor=2000.0,
        disaster_tail_floor=25000.0,
        default_tail_return=25000.0,
        return_clip=111750.0,
        edge_scale=250.0,
        confidence_power=1.0,
        max_abs_exposure=0.60,
    )

    # Good positive forecast regime in active Prototype3 units: realized
    # settlement payoff is a DKK/MWh spread, not a percentage return.
    for _ in range(500):
        payoff = rng.normal(450.0, 350.0)
        if rng.random() < 0.02:
            payoff -= rng.uniform(2000.0, 5000.0)
        sizer.update(
            forecast_sign=1.0,
            confidence=0.86,
            predicted_payoff=300.0,
            realized_payoff=payoff,
        )

    # Bad positive forecast regime: same forecast direction but poor confidence
    # bucket and negative signed edge.
    for _ in range(500):
        payoff = rng.normal(-250.0, 250.0)
        if rng.random() < 0.02:
            payoff -= rng.uniform(2000.0, 5000.0)
        sizer.update(
            forecast_sign=1.0,
            confidence=0.58,
            predicted_payoff=300.0,
            realized_payoff=payoff,
        )

    # Pooled disaster evidence. This should constrain, but not automatically
    # kill, an otherwise good conditional bucket. Use a separate magnitude
    # bucket so it contributes to the pooled disaster tail without poisoning
    # the good conditional bucket.
    for _ in range(20):
        sizer.update(
            forecast_sign=1.0,
            confidence=0.86,
            predicted_payoff=6000.0,
            realized_payoff=-30000.0,
        )

    # In mwh_volume mode the environment passes the maximum position quantity
    # in MWh to the sizer. The legacy parameter name still says notional.
    common = dict(
        max_position_notional_dkk=19_200.0,
        sleeve_value_dkk=96_000_000.0,
        margin_surplus_dkk=92_000_000.0,
    )
    good = sizer.size(
        forecast_sign=1.0,
        confidence=0.86,
        predicted_payoff=300.0,
        **common,
    )
    bad = sizer.size(
        forecast_sign=1.0,
        confidence=0.58,
        predicted_payoff=300.0,
        **common,
    )

    print("FoCAL v3 distributional smoke test")
    print(f"  good exposure      : {good['prior_exposure']:.4f}")
    print(f"  good edge_strength : {good['edge_strength']:.4f}")
    print(f"  good cond/dis tail : {good['conditional_tail_return']:.4f} / {good['disaster_tail_return']:.4f}")
    print(f"  good cap cond/dis  : {good['conditional_cap_abs']:.4f} / {good['disaster_cap_abs']:.4f}")
    print(f"  bad exposure       : {bad['prior_exposure']:.4f}")
    print(f"  bad edge_strength  : {bad['edge_strength']:.4f}")

    assert good["prior_exposure"] > 0.01, "good conditional bucket should open positive exposure"
    assert bad["prior_exposure"] <= 1e-9, "negative forecast-aligned edge should be shut down"
    assert abs(good["prior_exposure"]) <= good["tail_cap_abs"] + 1e-12, "good exposure exceeds tail cap"
    assert good["tail_cap_abs"] <= good["disaster_cap_abs"] + 1e-12 or good["disaster_cap_abs"] <= good["conditional_cap_abs"] + 1e-12
    print("  status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

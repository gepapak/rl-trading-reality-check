"""Forecast-conditioned feasible-action control for the investor MAPPO policy.

The forecast controller supplies a signed FoCAL anchor and the execution engine
supplies a non-negative, evidence-conditioned feasible MWh capacity. The learned
action is a one-dimensional coordinate inside that feasible interval:

* ``-1``: abstain;
* ``0``: execute the FoCAL anchor;
* ``+1``: use all remaining feasible capacity in the forecast direction.

This geometry prevents a projection layer from mapping many unrelated raw
policy actions to the same executed position. The paired settlement scorer uses
only strictly matured, one-step outcomes under a matched action origin. It is a
practical reward-shaping mechanism, not a full counterfactual fund simulation,
and does not claim a formal safety or convergence guarantee.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict

import numpy as np


@dataclass(frozen=True)
class FeasibleAction:
    action: float
    anchor_quantity_mwh: float
    capacity_mwh: float
    target_quantity_mwh: float
    anchor_fraction: float
    target_fraction: float
    adjustment_mwh: float


def map_feasible_action(
    *,
    action: float,
    anchor_quantity_mwh: float,
    capacity_mwh: float,
) -> FeasibleAction:
    """Map a normalized policy action to a forecast-aligned feasible quantity."""
    values = (action, anchor_quantity_mwh, capacity_mwh)
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("Feasible-action inputs must be finite")

    z = float(np.clip(action, -1.0, 1.0))
    capacity = float(max(capacity_mwh, 0.0))
    anchor = float(np.clip(anchor_quantity_mwh, -capacity, capacity))
    anchor_abs = float(abs(anchor))

    if capacity <= 1e-12 or anchor_abs <= 1e-12:
        target = 0.0
    elif z <= 0.0:
        target_abs = float(anchor_abs * (1.0 + z))
        target = float(np.sign(anchor) * target_abs)
    else:
        target_abs = float(anchor_abs + z * (capacity - anchor_abs))
        target = float(np.sign(anchor) * target_abs)

    anchor_fraction = float(anchor_abs / capacity) if capacity > 1e-12 else 0.0
    target_fraction = float(abs(target) / capacity) if capacity > 1e-12 else 0.0
    return FeasibleAction(
        action=z,
        anchor_quantity_mwh=anchor,
        capacity_mwh=capacity,
        target_quantity_mwh=float(np.clip(target, -capacity, capacity)),
        anchor_fraction=float(np.clip(anchor_fraction, 0.0, 1.0)),
        target_fraction=float(np.clip(target_fraction, 0.0, 1.0)),
        adjustment_mwh=float(target - anchor),
    )


@dataclass(frozen=True)
class PairedSettlementOutcome:
    policy_utility_dkk: float
    anchor_utility_dkk: float
    advantage_dkk: float
    normalized_advantage: float
    shortfall: float
    cvar_shortfall: float
    dual_lambda: float
    reward: float


class ForecastRelativeUtilityScorer:
    """Risk-aware matched settlement scorer for policy versus FoCAL anchor."""

    def __init__(
        self,
        *,
        window_size: int = 500,
        min_samples: int = 50,
        cvar_quantile: float = 0.90,
        shortfall_budget: float = 0.05,
        dual_learning_rate: float = 0.02,
        dual_max: float = 5.0,
        reward_weight: float = 0.25,
        payoff_scale_dkk_per_mwh: float = 250.0,
        score_clip: float = 2.0,
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.min_samples = max(1, int(min_samples))
        self.cvar_quantile = float(np.clip(cvar_quantile, 0.50, 0.9999))
        self.shortfall_budget = float(max(shortfall_budget, 0.0))
        self.dual_learning_rate = float(max(dual_learning_rate, 0.0))
        self.dual_max = float(max(dual_max, 0.0))
        self.reward_weight = float(max(reward_weight, 0.0))
        self.payoff_scale = float(max(payoff_scale_dkk_per_mwh, 1e-9))
        self.score_clip = float(max(score_clip, 1e-9))
        self.shortfalls: Deque[float] = deque(maxlen=self.window_size)
        self.advantages: Deque[float] = deque(maxlen=self.window_size)
        self.dual_lambda = 0.0
        self.last_outcome: PairedSettlementOutcome | None = None

    def _cvar(self) -> float:
        values = np.asarray(self.shortfalls, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < self.min_samples:
            return 0.0

        # Exact empirical expected shortfall over the worst (1-alpha) mass.
        # Selecting ``values >= VaR`` is biased when many observations tie at
        # the quantile (especially the common point mass at zero shortfall).
        # Fractional boundary weight keeps the requested tail probability exact
        # for every finite sample size.
        descending = np.sort(values)[::-1]
        tail_mass = float((1.0 - self.cvar_quantile) * descending.size)
        if tail_mass <= 1e-12:
            return float(descending[0])
        whole = int(np.floor(tail_mass))
        fraction = float(tail_mass - whole)
        weighted_sum = float(np.sum(descending[:whole])) if whole else 0.0
        if fraction > 1e-12 and whole < descending.size:
            weighted_sum += fraction * float(descending[whole])
        return float(weighted_sum / tail_mass)

    def update(
        self,
        *,
        policy_quantity_mwh: float,
        anchor_quantity_mwh: float,
        executable_capacity_mwh: float,
        realized_unit_payoff_dkk_per_mwh: float,
        policy_execution_cost_dkk: float,
        anchor_execution_cost_dkk: float,
        policy_funding_cost_dkk: float = 0.0,
        anchor_funding_cost_dkk: float = 0.0,
    ) -> PairedSettlementOutcome:
        values = (
            policy_quantity_mwh,
            anchor_quantity_mwh,
            executable_capacity_mwh,
            realized_unit_payoff_dkk_per_mwh,
            policy_execution_cost_dkk,
            anchor_execution_cost_dkk,
            policy_funding_cost_dkk,
            anchor_funding_cost_dkk,
        )
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("Paired settlement inputs must be finite")

        payoff = float(realized_unit_payoff_dkk_per_mwh)
        policy_utility = float(policy_quantity_mwh) * payoff - max(
            float(policy_execution_cost_dkk), 0.0
        ) - max(float(policy_funding_cost_dkk), 0.0)
        anchor_utility = float(anchor_quantity_mwh) * payoff - max(
            float(anchor_execution_cost_dkk), 0.0
        ) - max(float(anchor_funding_cost_dkk), 0.0)
        advantage = float(policy_utility - anchor_utility)
        normalizer = max(abs(float(executable_capacity_mwh)) * self.payoff_scale, 1.0)
        normalized = float(np.clip(advantage / normalizer, -self.score_clip, self.score_clip))
        shortfall = float(max(-normalized, 0.0))

        self.advantages.append(normalized)
        self.shortfalls.append(shortfall)
        cvar = self._cvar()
        if len(self.shortfalls) >= self.min_samples:
            self.dual_lambda = float(
                np.clip(
                    self.dual_lambda
                    + self.dual_learning_rate * (cvar - self.shortfall_budget),
                    0.0,
                    self.dual_max,
                )
            )

        excess_shortfall = float(max(shortfall - self.shortfall_budget, 0.0))
        reward = float(
            self.reward_weight * normalized - self.dual_lambda * excess_shortfall
        )
        outcome = PairedSettlementOutcome(
            policy_utility_dkk=policy_utility,
            anchor_utility_dkk=anchor_utility,
            advantage_dkk=advantage,
            normalized_advantage=normalized,
            shortfall=shortfall,
            cvar_shortfall=cvar,
            dual_lambda=self.dual_lambda,
            reward=reward,
        )
        self.last_outcome = outcome
        return outcome

    def diagnostics(self) -> Dict[str, float]:
        advantages = np.asarray(self.advantages, dtype=np.float64)
        shortfalls = np.asarray(self.shortfalls, dtype=np.float64)
        last = self.last_outcome
        return {
            "count": float(len(self.advantages)),
            "advantage_mean": float(np.mean(advantages)) if advantages.size else 0.0,
            "shortfall_mean": float(np.mean(shortfalls)) if shortfalls.size else 0.0,
            "cvar_shortfall": float(self._cvar()),
            "dual_lambda": float(self.dual_lambda),
            "last_advantage_dkk": float(last.advantage_dkk) if last else 0.0,
            "last_normalized_advantage": float(last.normalized_advantage) if last else 0.0,
            "last_reward": float(last.reward) if last else 0.0,
        }


__all__ = [
    "FeasibleAction",
    "ForecastRelativeUtilityScorer",
    "PairedSettlementOutcome",
    "map_feasible_action",
]

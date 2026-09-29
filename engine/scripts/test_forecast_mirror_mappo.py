#!/usr/bin/env python3
"""Deterministic synthetic checks for Counterfactual Forecast-Mirror MAPPO."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecast_mirror_mappo import beta_grid_weights, counterfactual_trust_gradient
from distributional_forecast_prior import DistributionalTailBudgetSizer
from policy import CentralizedCriticForecastMirrorBetaActorCriticPolicy


def _constant_schedule(_: float) -> float:
    return 3e-4


def _policy(*, trust_max: float = 1.0, trust_initial: float = 0.10):
    torch.manual_seed(7)
    return CentralizedCriticForecastMirrorBetaActorCriticPolicy(
        observation_space=spaces.Box(-1.0, 1.0, shape=(22,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        lr_schedule=_constant_schedule,
        net_arch=[32, 16],
        central_obs_dim=8,
        central_net_arch=[16, 8],
        forecast_mirror_prior_concentration=8.0,
        forecast_mirror_trust_max=trust_max,
        forecast_mirror_trust_initial=trust_initial,
        forecast_mirror_trust_min_samples=4,
        forecast_mirror_trust_batch_size=16,
    )


def _obs(prior_coordinate: float, evidence: float) -> torch.Tensor:
    value = torch.zeros((1, 22), dtype=torch.float32)
    value[:, 20] = prior_coordinate
    value[:, 21] = evidence
    return value


class ForecastMirrorTests(unittest.TestCase):
    def test_classifier_opposition_cannot_increase_forecast_sizing(self):
        sizer = DistributionalTailBudgetSizer(
            min_samples=1,
            bucket_mode="global",
            conditional_tail_floor=100.0,
            disaster_tail_floor=500.0,
            default_tail_return=500.0,
            edge_scale=100.0,
        )
        sizer.update(
            forecast_sign=1.0,
            confidence=0.9,
            predicted_payoff=250.0,
            realized_payoff=300.0,
        )
        common = dict(
            forecast_sign=1.0,
            predicted_payoff=250.0,
            max_position_notional_dkk=100.0,
            sleeve_value_dkk=100_000.0,
            margin_surplus_dkk=100_000.0,
        )
        aligned = sizer.size(confidence=0.9, **common)
        opposed = sizer.size(confidence=0.1, **common)

        self.assertGreater(aligned["confidence_weight"], 0.0)
        self.assertGreater(aligned["prior_exposure"], 0.0)
        self.assertEqual(opposed["confidence_weight"], 0.0)
        self.assertEqual(opposed["prior_exposure"], 0.0)

    def test_pooled_tail_support_is_forecast_independent(self):
        sizer = DistributionalTailBudgetSizer(
            min_samples=1,
            conditional_tail_floor=100.0,
            disaster_tail_floor=500.0,
            default_tail_return=500.0,
            edge_scale=100.0,
        )
        for sign, payoff in ((1.0, 200.0), (-1.0, -400.0), (1.0, 300.0)):
            sizer.update(
                forecast_sign=sign,
                confidence=0.9,
                predicted_payoff=250.0,
                realized_payoff=payoff,
            )
        long = sizer.size(
            forecast_sign=1.0,
            confidence=0.9,
            predicted_payoff=250.0,
            max_position_notional_dkk=10_000.0,
            sleeve_value_dkk=100_000.0,
            margin_surplus_dkk=100_000.0,
        )
        short = sizer.size(
            forecast_sign=-1.0,
            confidence=0.6,
            predicted_payoff=-50.0,
            max_position_notional_dkk=10_000.0,
            sleeve_value_dkk=100_000.0,
            margin_surplus_dkk=100_000.0,
        )
        zero_sign = sizer.size(
            forecast_sign=0.0,
            confidence=0.5,
            predicted_payoff=0.0,
            max_position_notional_dkk=10_000.0,
            sleeve_value_dkk=100_000.0,
            margin_surplus_dkk=100_000.0,
        )
        self.assertEqual(long["global_cap_abs"], short["global_cap_abs"])
        self.assertEqual(long["global_tail_return"], short["global_tail_return"])
        self.assertEqual(long["global_cap_abs"], zero_sign["global_cap_abs"])
        self.assertGreater(zero_sign["global_cap_abs"], 0.0)

    def test_zero_evidence_is_exact_plain_mappo(self):
        policy = _policy()
        obs = _obs(0.9, 0.0)
        with torch.no_grad():
            policy.get_distribution(obs)
        diag = policy.get_mirror_diagnostics()
        np.testing.assert_allclose(diag["final_alpha"], diag["base_alpha"], atol=0.0)
        np.testing.assert_allclose(diag["final_beta"], diag["base_beta"], atol=0.0)

    def test_zero_trust_is_exact_plain_mappo(self):
        policy = _policy(trust_max=0.0)
        obs = _obs(-0.9, 1.0)
        with torch.no_grad():
            policy.get_distribution(obs)
        diag = policy.get_mirror_diagnostics()
        np.testing.assert_allclose(diag["final_alpha"], diag["base_alpha"], atol=0.0)
        np.testing.assert_allclose(diag["final_beta"], diag["base_beta"], atol=0.0)

    def test_forecast_shifts_density_but_keeps_signed_support(self):
        policy = _policy(trust_initial=0.5)
        with torch.no_grad():
            positive = policy.get_distribution(_obs(0.9, 1.0)).distribution.mean.item()
            negative = policy.get_distribution(_obs(-0.9, 1.0)).distribution.mean.item()
        self.assertGreater(positive, negative)
        samples = policy.get_distribution(_obs(0.9, 1.0)).distribution.sample((2000,))
        signed = (2.0 * samples) - 1.0
        self.assertTrue(bool((signed < 0.0).any()))
        self.assertTrue(bool((signed > 0.0).any()))

    def test_ppo_log_probability_recomputation_is_exact(self):
        policy = _policy(trust_initial=0.3)
        obs = _obs(0.7, 0.8)
        central = torch.zeros((1, 8), dtype=torch.float32)
        with torch.no_grad():
            dist = policy.get_distribution(obs)
            action = dist.get_actions(deterministic=False)
            expected = dist.log_prob(action)
            _, observed, _ = policy.evaluate_actions_centralized(obs, action, central)
        torch.testing.assert_close(expected, observed, rtol=0.0, atol=1e-7)

    def test_counterfactual_gradient_has_correct_direction(self):
        grid = (np.arange(101, dtype=np.float64) + 0.5) / 101.0
        signed_action = 2.0 * grid - 1.0
        positive = counterfactual_trust_gradient(
            unit_grid=grid,
            scores=signed_action,
            final_alpha=3.0,
            final_beta=2.0,
            prior_alpha=8.0,
            prior_beta=2.0,
            evidence=1.0,
        )
        negative = counterfactual_trust_gradient(
            unit_grid=grid,
            scores=-signed_action,
            final_alpha=3.0,
            final_beta=2.0,
            prior_alpha=8.0,
            prior_beta=2.0,
            evidence=1.0,
        )
        self.assertGreater(positive.trust_gradient, 0.0)
        self.assertLess(negative.trust_gradient, 0.0)

    def test_counterfactual_math_rejects_invalid_distribution_inputs(self):
        grid = (np.arange(11, dtype=np.float64) + 0.5) / 11.0
        with self.assertRaises(ValueError):
            beta_grid_weights(float("nan"), 2.0, grid)
        with self.assertRaises(ValueError):
            beta_grid_weights(2.0, 2.0, np.asarray([], dtype=np.float64))
        with self.assertRaises(ValueError):
            counterfactual_trust_gradient(
                unit_grid=grid,
                scores=np.zeros_like(grid),
                final_alpha=2.0,
                final_beta=2.0,
                prior_alpha=2.0,
                prior_beta=2.0,
                evidence=float("nan"),
            )

    def test_matured_gradient_updates_trust(self):
        policy = _policy(trust_initial=0.25)
        obs_np = _obs(0.8, 1.0).numpy().reshape(-1)
        before = float(policy._trust(_obs(0.8, 1.0)).item())
        policy.add_mirror_trust_samples(
            [{"observation": obs_np, "trust_gradient": 0.5} for _ in range(4)]
        )
        result = policy.train_mirror_trust()
        after = float(policy._trust(_obs(0.8, 1.0)).item())
        self.assertEqual(result["updated"], 1.0)
        self.assertGreater(after, before)

    def test_policy_checkpoint_round_trip_preserves_mirror_state(self):
        policy = _policy(trust_initial=0.30)
        obs = _obs(0.7, 0.8)
        with tempfile.TemporaryDirectory(prefix="cfm_policy_") as temp_dir:
            checkpoint = str(Path(temp_dir) / "policy.pt")
            policy.save(checkpoint)
            restored = CentralizedCriticForecastMirrorBetaActorCriticPolicy.load(
                checkpoint
            )
        with torch.no_grad():
            expected = policy.get_distribution(obs).distribution.mean
            observed = restored.get_distribution(obs).distribution.mean
        torch.testing.assert_close(expected, observed, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

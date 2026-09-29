"""Synthetic and short real-data smoke tests for feasible-action FoCAL-MAPPO."""

from __future__ import annotations

import sys
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import EnhancedConfig
from environment import RenewableMultiAgentEnv, StabilizedObservationManager
from feasible_action_mappo import ForecastRelativeUtilityScorer, map_feasible_action
from logger import RewardLogger
from metacontroller import MultiESGAgent, _flat_obs_dim
from runtime_contract import forecast_prior_contract_settings


def _test_mapping() -> None:
    expected = {
        -1.0: 0.0,
        -0.5: 10.0,
        0.0: 20.0,
        0.5: 35.0,
        1.0: 50.0,
    }
    positive = []
    for action, target in expected.items():
        mapped = map_feasible_action(
            action=action,
            anchor_quantity_mwh=20.0,
            capacity_mwh=50.0,
        )
        assert np.isclose(mapped.target_quantity_mwh, target)
        positive.append(mapped.target_quantity_mwh)
    assert np.all(np.diff(positive) > 0.0)

    negative = [
        map_feasible_action(
            action=action,
            anchor_quantity_mwh=-20.0,
            capacity_mwh=50.0,
        ).target_quantity_mwh
        for action in expected
    ]
    assert np.all(np.diff(negative) < 0.0)
    assert np.isclose(negative[0], 0.0)
    assert np.isclose(negative[-1], -50.0)


def _test_paired_scorer() -> None:
    scorer = ForecastRelativeUtilityScorer(
        min_samples=1,
        cvar_quantile=0.90,
        shortfall_budget=0.05,
        dual_learning_rate=0.10,
        reward_weight=0.25,
        payoff_scale_dkk_per_mwh=100.0,
    )
    win = scorer.update(
        policy_quantity_mwh=2.0,
        anchor_quantity_mwh=1.0,
        executable_capacity_mwh=2.0,
        realized_unit_payoff_dkk_per_mwh=100.0,
        policy_execution_cost_dkk=10.0,
        anchor_execution_cost_dkk=5.0,
    )
    assert np.isclose(win.advantage_dkk, 95.0)
    assert win.normalized_advantage > 0.0 and win.reward > 0.0
    loss = scorer.update(
        policy_quantity_mwh=0.0,
        anchor_quantity_mwh=2.0,
        executable_capacity_mwh=2.0,
        realized_unit_payoff_dkk_per_mwh=100.0,
        policy_execution_cost_dkk=0.0,
        anchor_execution_cost_dkk=5.0,
    )
    assert loss.shortfall > 0.0
    assert loss.dual_lambda > 0.0
    assert loss.reward < 0.0

    funded = ForecastRelativeUtilityScorer(
        min_samples=1,
        payoff_scale_dkk_per_mwh=100.0,
    ).update(
        policy_quantity_mwh=2.0,
        anchor_quantity_mwh=1.0,
        executable_capacity_mwh=2.0,
        realized_unit_payoff_dkk_per_mwh=100.0,
        policy_execution_cost_dkk=10.0,
        anchor_execution_cost_dkk=5.0,
        policy_funding_cost_dkk=10.0,
        anchor_funding_cost_dkk=2.0,
    )
    assert np.isclose(funded.advantage_dkk, 87.0)


def _test_exact_expected_shortfall() -> None:
    scorer = ForecastRelativeUtilityScorer(
        min_samples=1,
        cvar_quantile=0.90,
    )
    scorer.shortfalls.extend([0.0] * 95 + [2.0] * 5)
    assert np.isclose(scorer._cvar(), 1.0), (
        "Expected shortfall must average exactly the worst 10% mass, including "
        "fractional quantile-boundary weight"
    )


def _test_strict_maturity() -> None:
    env = RenewableMultiAgentEnv.__new__(RenewableMultiAgentEnv)
    env.config = SimpleNamespace(
        forecast_prior_feasible_action_window=20,
        forecast_prior_feasible_action_min_samples=1,
        forecast_prior_feasible_action_cvar_quantile=0.90,
        forecast_prior_feasible_action_shortfall_budget=0.05,
        forecast_prior_feasible_action_dual_lr=0.02,
        forecast_prior_feasible_action_dual_max=5.0,
        forecast_prior_feasible_action_reward_weight=0.25,
        forecast_prior_feasible_action_payoff_scale=100.0,
        forecast_prior_feasible_action_score_clip=2.0,
    )
    env._feasible_action_scorer = None
    env._feasible_action_last_update_t = -1
    env._feasible_action_reward_step = 0.0
    env._feasible_action_last_diag = {}
    env._feasible_action_pending = [
        {
            "step": 0,
            "due_step": 6,
            "policy_quantity_mwh": 2.0,
            "anchor_quantity_mwh": 1.0,
            "executable_capacity_mwh": 2.0,
            "policy_execution_cost_dkk": 0.0,
            "anchor_execution_cost_dkk": 0.0,
        }
    ]
    env._forecast_unit_payoff_for_origin = lambda _origin, _due: 100.0
    env._update_feasible_action_evidence(6)
    assert len(env._feasible_action_pending) == 1
    assert env._feasible_action_reward_step == 0.0
    env._update_feasible_action_evidence(7)
    assert len(env._feasible_action_pending) == 0
    assert env._feasible_action_reward_step > 0.0


def _configured_env(
    rows: int = 48,
    *,
    feasible_action: bool = True,
    observation_only: bool = False,
) -> tuple[RenewableMultiAgentEnv, EnhancedConfig]:
    data = pd.read_csv(
        ROOT / "training_dataset_ffill" / "scenario_000.csv",
        low_memory=False,
    ).head(rows)
    cfg = EnhancedConfig()
    cfg.seed = 7
    cfg.current_episode_num = 0
    cfg.algo = "mappo"
    cfg.enable_forecast_utilization = True
    cfg.forecast_cache_dir = str(ROOT / "forecast_cache_settlement_hourly_v2")
    cfg.forecast_prior_feasible_action_mappo = bool(feasible_action)
    cfg.forecast_prior_observation_only_mappo = bool(observation_only)
    cfg.forecast_prior_feasible_action_min_samples = 1
    cfg.forecast_prior_distributional_min_samples = 1
    cfg.forecast_prior_distributional_cold_start_exposure = 0.05
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
    cfg.impact_volume_data_path = str(
        ROOT / "liquidity_volume_dataset_real_v1" / "scenario_000.csv"
    )
    cfg.impact_volume_column = "market_volume_mwh"
    cfg.impact_volume_unit = "mwh"
    cfg.impact_volume_timestamp_column = "timestamp"
    cfg.liquidity_volume_source = "impact_volume"
    cfg.liquidity_participation_cap_fraction = 0.25
    cfg.investor_notional_sizing_base = "initial_trading_sleeve"
    cfg.max_position_size = 0.10
    cfg.capital_allocation_fraction = 0.60
    cfg.no_trade_threshold = 0.01
    cfg.no_trade_threshold_reference = "executable_capacity"
    cfg.risk_controller_rule_based = True
    cfg.meta_controller_rule_based = True
    cfg.update_every = 32
    cfg.n_steps = 32
    cfg.batch_size = 16
    cfg.n_epochs = 1
    cfg.agent_policies = [dict(item) for item in cfg.agent_policies]
    cfg.agent_policies[2]["mode"] = "RULE"
    cfg.agent_policies[3]["mode"] = "RULE"
    cfg.enable_episode_csv_logs = False
    return RenewableMultiAgentEnv(data, investment_freq=6, config=cfg), cfg


def _test_contract_and_policy_stack() -> None:
    off = SimpleNamespace(config=SimpleNamespace(
        forecast_prior_feasible_action_mappo=False,
        forecast_prior_observation_only_mappo=False,
    ))
    on = SimpleNamespace(config=SimpleNamespace(
        forecast_prior_feasible_action_mappo=True,
        forecast_prior_observation_only_mappo=False,
    ))
    obs_only = SimpleNamespace(config=SimpleNamespace(
        forecast_prior_feasible_action_mappo=False,
        forecast_prior_observation_only_mappo=True,
    ))
    assert StabilizedObservationManager(off).obs_space("investor_0").shape == (12,)
    assert StabilizedObservationManager(on).obs_space("investor_0").shape == (20,)
    assert StabilizedObservationManager(obs_only).obs_space("investor_0").shape == (20,)

    cfg = EnhancedConfig()
    assert "forecast_prior_feasible_action_mappo" not in forecast_prior_contract_settings(cfg)
    cfg.enable_forecast_utilization = True
    cfg.forecast_prior_feasible_action_mappo = True
    contract = forecast_prior_contract_settings(cfg)
    assert contract["forecast_prior_feasible_action_mappo"] is True
    assert contract["forecast_prior_feasible_action_condition_capacity_on_evidence"] is True
    assert contract["forecast_prior_feasible_action_min_samples"] == 50

    cfg.forecast_prior_feasible_action_mappo = False
    cfg.forecast_prior_observation_only_mappo = True
    observation_contract = forecast_prior_contract_settings(cfg)
    assert observation_contract["forecast_prior_observation_only_mappo"] is True
    assert observation_contract[
        "forecast_prior_feasible_action_condition_capacity_on_evidence"
    ] is True
    logger_parameters = inspect.signature(RewardLogger.log_step).parameters
    for field in (
        "distributional_confidence_weight",
        "feasible_action_hard_capacity_mwh",
        "feasible_action_evidence_authority",
        "feasible_action_capacity_conditioned_on_evidence",
        "feasible_action_policy_funding_cost_dkk",
        "feasible_action_anchor_funding_cost_dkk",
    ):
        assert field in logger_parameters

    env, cfg = _configured_env(rows=48)
    observations, _ = env.reset(seed=7)
    agent = MultiESGAgent(cfg, env, "cpu", training=True, debug=False)
    assert env.observation_space("investor_0").shape == (20,)
    assert agent._central_obs_dim == 56
    investor = next(
        policy
        for policy in agent.policies
        if getattr(policy, "agent_name", "") == "investor_0"
    )
    assert _flat_obs_dim(getattr(investor, "observation_space", None)) == 20
    action, _ = investor.predict(observations["investor_0"], deterministic=True)
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    assert action.size == 1 and np.all(np.isfinite(action))

    before = [parameter.detach().cpu().clone() for parameter in investor.policy.parameters()]
    agent.learn(total_timesteps=32, overall_target=32)
    after = [parameter.detach().cpu() for parameter in investor.policy.parameters()]
    largest_update = max(
        float((new - old).abs().max().item()) for old, new in zip(before, after)
    )
    assert largest_update > 0.0, "MAPPO investor weights did not update"


def _test_observation_only_identification_arm() -> None:
    env, _ = _configured_env(
        rows=48,
        feasible_action=False,
        observation_only=True,
    )
    observations, _ = env.reset(seed=7)
    assert observations["investor_0"].shape == (20,)
    tradeable = float(env._get_investor_tradeable_capital_dkk())
    direct = env._map_investor_control_to_exposure(0.40, tradeable_capital=tradeable)
    mapped = env._map_investor_action_to_exposure(
        0.40,
        tradeable_capital=tradeable,
        timestep=0,
    )
    assert np.allclose(np.asarray(mapped[:4]), np.asarray(direct))
    assert mapped[4]["forecast_control_mode"] == "forecast_observation_only_mappo"


def _test_evidence_conditioned_capacity() -> None:
    env, cfg = _configured_env(rows=48)
    env.reset(seed=7)
    tradeable = float(env._get_investor_tradeable_capital_dkk())
    reference = env._feasible_action_geometry(
        step=0,
        anchor_target=0.0,
        distributional_diag={
            "forecast_tail_cap_abs": 0.50,
            "distributional_edge_strength": 1.0,
            "distributional_confidence_weight": 1.0,
        },
        current_exposure=0.0,
        tradeable_capital=tradeable,
    )
    hard_fraction = float(
        reference["hard_capacity_mwh"] / max(reference["max_position_mwh"], 1e-12)
    )
    weak = env._feasible_action_geometry(
        step=0,
        anchor_target=0.10 * hard_fraction,
        distributional_diag={
            "forecast_tail_cap_abs": 0.50,
            "distributional_edge_strength": 0.01,
            "distributional_confidence_weight": 0.10,
        },
        current_exposure=0.0,
        tradeable_capital=tradeable,
    )
    strong = env._feasible_action_geometry(
        step=0,
        anchor_target=0.50 * hard_fraction,
        distributional_diag={
            "forecast_tail_cap_abs": 0.50,
            "distributional_edge_strength": 1.0,
            "distributional_confidence_weight": 1.0,
        },
        current_exposure=0.0,
        tradeable_capital=tradeable,
    )
    assert abs(weak["anchor_candidate_mwh"]) <= weak["capacity_mwh"] + 1e-12
    assert weak["capacity_mwh"] < weak["hard_capacity_mwh"]
    assert np.isclose(strong["capacity_mwh"], strong["hard_capacity_mwh"])

    cfg.forecast_prior_feasible_action_condition_capacity_on_evidence = False
    unrestricted = env._feasible_action_geometry(
        step=0,
        anchor_target=0.10 * hard_fraction,
        distributional_diag={
            "forecast_tail_cap_abs": 0.50,
            "distributional_edge_strength": 0.01,
            "distributional_confidence_weight": 0.10,
        },
        current_exposure=0.0,
        tradeable_capital=tradeable,
    )
    assert np.isclose(unrestricted["capacity_mwh"], unrestricted["hard_capacity_mwh"])


def _test_short_rollout() -> None:
    env, _ = _configured_env(rows=48)
    observations, _ = env.reset(seed=7)
    assert observations["investor_0"].shape == (20,)
    for _ in range(30):
        observations, rewards, _, _, _ = env.step(
            {
                "investor_0": np.asarray([0.50], dtype=np.float32),
                "battery_operator_0": 2,
                "risk_controller_0": np.asarray([1.0], dtype=np.float32),
                "meta_controller_0": np.asarray([0.0], dtype=np.float32),
            }
        )
        assert observations["investor_0"].shape == (20,)
        assert np.isfinite(float(rewards["investor_0"]))
        exec_diag = (getattr(env, "_last_actions", {}) or {}).get(
            "investor_0_exec", {}
        )
        if float(exec_diag.get("feasible_action_enabled", 0.0)) > 0.5:
            capacity = float(exec_diag.get("feasible_action_capacity_mwh", 0.0))
            policy = float(exec_diag.get("feasible_action_policy_quantity_mwh", 0.0))
            assert abs(policy) <= capacity + 1e-8
    assert float((getattr(env, "_feasible_action_last_diag", {}) or {}).get("count", 0.0)) >= 1.0


def main() -> int:
    _test_mapping()
    _test_paired_scorer()
    _test_exact_expected_shortfall()
    _test_strict_maturity()
    _test_contract_and_policy_stack()
    _test_observation_only_identification_arm()
    _test_evidence_conditioned_capacity()
    _test_short_rollout()
    print(
        "[OK] feasible-action FoCAL-MAPPO: monotone mapping, exact expected "
        "shortfall, matched settlement utility, strict maturity, evidence-conditioned "
        "capacity, observation-only identification, real MAPPO update, and rollout"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

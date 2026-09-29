#!/usr/bin/env python3
"""Short real-data rollout smoke for Counterfactual Forecast-Mirror MAPPO."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import EnhancedConfig
from environment import RenewableMultiAgentEnv
from metacontroller import MultiESGAgent
from policy import (
    CentralizedCriticBetaActorCriticPolicy,
    CentralizedCriticForecastMirrorBetaActorCriticPolicy,
)
from runtime_contract import forecast_prior_contract_settings


def _configured_env(rows: int = 72, *, mirror: bool = True):
    data = pd.read_csv(
        ROOT / "training_dataset_ffill" / "scenario_000.csv",
        low_memory=False,
    ).head(rows)
    cfg = EnhancedConfig()
    cfg.seed = 7
    cfg.current_episode_num = 0
    cfg.algo = "mappo"
    cfg.enable_forecast_utilization = bool(mirror)
    cfg.forecast_cache_dir = str(ROOT / "forecast_cache_settlement_hourly_v2")
    cfg.forecast_prior_control_mode = "distributional"
    cfg.forecast_prior_mirror_mappo = bool(mirror)
    cfg.forecast_prior_feasible_action_mappo = False
    cfg.forecast_prior_observation_only_mappo = False
    cfg.forecast_prior_feasible_action_condition_capacity_on_evidence = False
    cfg.forecast_prior_distributional_min_samples = 1
    cfg.forecast_prior_distributional_cold_start_exposure = 0.05
    cfg.forecast_mirror_trust_min_samples = 1
    cfg.forecast_mirror_trust_batch_size = 32
    cfg.forecast_mirror_counterfactual_grid_size = 21
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
    cfg.impact_ref_notional = "volume"
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


def main() -> int:
    env, cfg = _configured_env()
    observations, _ = env.reset(seed=7)
    if observations["investor_0"].shape != (22,):
        raise AssertionError("Forecast-mirror investor observation is not 22D")
    contract = forecast_prior_contract_settings(cfg)
    if contract.get("forecast_prior_mirror_mappo") is not True:
        raise AssertionError("Mirror mode is absent from the runtime contract")
    agent = MultiESGAgent(cfg, env, "cpu", training=True, debug=False)
    investor = next(
        item for item in agent.policies if getattr(item, "agent_name", "") == "investor_0"
    )
    if not isinstance(
        investor.policy, CentralizedCriticForecastMirrorBetaActorCriticPolicy
    ):
        raise AssertionError("Investor was not built with the forecast-mirror MAPPO policy")
    before = [parameter.detach().cpu().clone() for parameter in investor.policy.parameters()]
    agent.learn(total_timesteps=32, overall_target=32)
    after = [parameter.detach().cpu() for parameter in investor.policy.parameters()]
    largest_update = max(
        float((new - old).abs().max().item()) for old, new in zip(before, after)
    )
    if largest_update <= 0.0:
        raise AssertionError("Forecast-mirror MAPPO parameters did not update")
    mirror = investor.policy.get_mirror_diagnostics()
    for key in ("base_alpha", "prior_alpha", "final_alpha", "evidence", "trust"):
        if key not in mirror or not np.isfinite(np.asarray(mirror[key], dtype=float)).all():
            raise AssertionError(f"Missing or non-finite mirror diagnostic: {key}")
    train_diag = dict(investor.policy._last_mirror_train_diagnostics)
    if train_diag.get("updated", 0.0) != 1.0:
        raise AssertionError(
            "The short rollout produced no matured counterfactual trust update: "
            f"{train_diag}"
        )
    cfg.forecast_mirror_mask_forecast_observation = True
    env._fill_obs()
    masked = np.asarray(env._obs_buf["investor_0"], dtype=float).copy()
    if not np.allclose(masked[12:17], 0.0) or not np.allclose(masked[20:22], 0.0):
        raise AssertionError("Forecast-free identification mask leaked forecast features")

    plain_env, plain_cfg = _configured_env(mirror=False)
    plain_obs, _ = plain_env.reset(seed=7)
    if plain_obs["investor_0"].shape != (12,):
        raise AssertionError("Plain MAPPO investor observation is not the original 12D state")
    plain_agent = MultiESGAgent(plain_cfg, plain_env, "cpu", training=True, debug=False)
    plain_investor = next(
        item
        for item in plain_agent.policies
        if getattr(item, "agent_name", "") == "investor_0"
    )
    if not isinstance(plain_investor.policy, CentralizedCriticBetaActorCriticPolicy):
        raise AssertionError("Plain MARL no longer uses the centralized Beta MAPPO actor")
    if isinstance(
        plain_investor.policy,
        CentralizedCriticForecastMirrorBetaActorCriticPolicy,
    ):
        raise AssertionError("Plain MARL was contaminated by the forecast-mirror policy")
    print(
        "[OK] CFM-MAPPO real-data smoke: 22D observation, centralized Beta actor, "
        "signed execution, matured trust update, PPO parameter update, and clean "
        "12D plain-MARL fallback"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

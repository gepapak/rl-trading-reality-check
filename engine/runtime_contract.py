"""Runtime contract helpers for train/eval consistency checks."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


FORECAST_PRIOR_CONTRACT_KEYS = (
    ("forecast_prior_control_mode", "distributional", str),
    ("forecast_prior_window", 500, int),
    ("forecast_prior_min_samples", 50, int),
    ("forecast_prior_hit_lcb_z", 1.64, float),
    ("forecast_prior_residual_quantile", 0.70, float),
    ("forecast_prior_default_residual", 500.0, float),
    ("forecast_prior_edge_gain", 3.0, float),
    ("forecast_prior_error_hurdle", 0.50, float),
    ("forecast_prior_skill_power", 1.0, float),
    ("forecast_prior_directional_floor", 0.50, float),
    ("forecast_prior_use_direction_confidence", True, _as_bool),
    ("forecast_prior_direction_confidence_power", 1.0, float),
    ("forecast_prior_calibrate_on_decision_grid", True, _as_bool),
    ("forecast_prior_max_abs_exposure", 0.60, float),
    ("forecast_prior_residual_scale", 0.10, float),
    ("forecast_prior_residual_to_prior_ratio", 0.35, float),
    ("forecast_prior_inactive_residual_scale", 0.0, float),
    ("forecast_prior_blend", 0.85, float),
    ("forecast_prior_policy_relative_gate", True, _as_bool),
    ("forecast_prior_policy_relative_min_samples", 50, int),
    ("forecast_prior_policy_relative_hit_margin", 0.02, float),
    ("forecast_prior_policy_relative_advantage_scale", 0.02, float),
    ("forecast_prior_residual_evidence_gate", True, _as_bool),
    ("forecast_prior_residual_evidence_floor", 0.0, float),
    ("forecast_prior_beta_enabled", False, _as_bool),
    ("forecast_prior_beta_min_samples", 50, int),
    ("forecast_prior_beta_lcb_scale", 0.005, float),
    ("forecast_prior_beta_max_abs_exposure", 0.60, float),
    ("forecast_prior_beta_weight", 1.0, float),
    ("forecast_prior_beta_return_clip", 0.25, float),
    ("forecast_prior_beta_allow_short", True, _as_bool),
    ("forecast_prior_tail_cap_enabled", True, _as_bool),
    ("forecast_prior_tail_min_samples", 50, int),
    ("forecast_prior_tail_quantile", 0.99, float),
    ("forecast_prior_tail_loss_budget_fraction", 0.05, float),
    ("forecast_prior_tail_stress_return_floor", 25000.0, float),
    ("forecast_prior_tail_default_return", 25000.0, float),
    ("forecast_prior_tail_return_clip", 111750.0, float),
    ("forecast_prior_distributional_window", 2000, int),
    ("forecast_prior_distributional_min_samples", 50, int),
    ("forecast_prior_distributional_tail_quantile", 0.99, float),
    ("forecast_prior_distributional_disaster_quantile", 0.999, float),
    ("forecast_prior_distributional_loss_budget_fraction", 0.05, float),
    ("forecast_prior_distributional_disaster_loss_budget_fraction", 0.20, float),
    ("forecast_prior_distributional_conditional_tail_floor", 2000.0, float),
    ("forecast_prior_distributional_disaster_tail_floor", 25000.0, float),
    ("forecast_prior_distributional_default_tail_return", 25000.0, float),
    ("forecast_prior_distributional_return_clip", 111750.0, float),
    ("forecast_prior_distributional_edge_scale", 250.0, float),
    ("forecast_prior_distributional_edge_hurdle", 0.0, float),
    ("forecast_prior_distributional_confidence_power", 1.0, float),
    ("forecast_prior_distributional_bucket_mode", "conditional", str),
    ("forecast_prior_distributional_use_confidence_weight", True, _as_bool),
    ("forecast_prior_distributional_fixed_cap_abs", 0.10, float),
    ("forecast_prior_distributional_cold_start_exposure", 0.0, float),
    ("forecast_prior_observation_only_mappo", False, _as_bool),
    ("forecast_prior_feasible_action_mappo", False, _as_bool),
    ("forecast_prior_mirror_mappo", False, _as_bool),
    ("forecast_mirror_prior_concentration", 8.0, float),
    ("forecast_mirror_trust_max", 1.0, float),
    ("forecast_mirror_trust_hidden_dim", 32, int),
    ("forecast_mirror_trust_initial", 0.10, float),
    ("forecast_mirror_trust_regularization", 0.01, float),
    ("forecast_mirror_trust_min_samples", 32, int),
    ("forecast_mirror_trust_batch_size", 256, int),
    ("forecast_mirror_trust_gradient_clip", 5.0, float),
    ("forecast_mirror_counterfactual_learning", True, _as_bool),
    ("forecast_mirror_mask_forecast_observation", False, _as_bool),
    ("forecast_mirror_counterfactual_grid_size", 41, int),
    ("forecast_mirror_counterfactual_payoff_scale", 250.0, float),
    ("forecast_mirror_counterfactual_loss_aversion", 0.25, float),
    ("forecast_mirror_anchor_advantage_reward", False, _as_bool),
    ("forecast_mirror_anchor_advantage_weight", 0.25, float),
    ("forecast_mirror_mask_base_actor_forecast", False, _as_bool),
    ("forecast_prior_feasible_action_condition_capacity_on_evidence", True, _as_bool),
    ("forecast_prior_feasible_action_evidence_power", 1.0, float),
    ("forecast_prior_feasible_action_window", 500, int),
    ("forecast_prior_feasible_action_min_samples", 50, int),
    ("forecast_prior_feasible_action_cvar_quantile", 0.90, float),
    ("forecast_prior_feasible_action_shortfall_budget", 0.05, float),
    ("forecast_prior_feasible_action_dual_lr", 0.02, float),
    ("forecast_prior_feasible_action_dual_max", 5.0, float),
    ("forecast_prior_feasible_action_reward_weight", 0.25, float),
    ("forecast_prior_feasible_action_payoff_scale", 250.0, float),
    ("forecast_prior_feasible_action_score_clip", 2.0, float),
    ("forecast_prior_vol_half_life_steps", 288, int),
    ("forecast_prior_vol_target", 0.15, float),
    ("forecast_prior_horizon_steps", 6, int),
    ("forecast_prior_denom_floor", 50.0, float),
    ("forecast_prior_payoff_target_mode", "auto", str),
    ("forecast_prior_target_alignment_version", "same_delivery_causal_v3", str),
    ("forecast_prior_agreement_gate", False, _as_bool),
)

_FEASIBLE_ACTION_CONTRACT_KEYS = {
    "forecast_prior_feasible_action_mappo",
    "forecast_prior_feasible_action_condition_capacity_on_evidence",
    "forecast_prior_feasible_action_evidence_power",
    "forecast_prior_feasible_action_window",
    "forecast_prior_feasible_action_min_samples",
    "forecast_prior_feasible_action_cvar_quantile",
    "forecast_prior_feasible_action_shortfall_budget",
    "forecast_prior_feasible_action_dual_lr",
    "forecast_prior_feasible_action_dual_max",
    "forecast_prior_feasible_action_reward_weight",
    "forecast_prior_feasible_action_payoff_scale",
    "forecast_prior_feasible_action_score_clip",
}

_MIRROR_CONTRACT_KEYS = {
    "forecast_prior_mirror_mappo",
    "forecast_mirror_prior_concentration",
    "forecast_mirror_trust_max",
    "forecast_mirror_trust_hidden_dim",
    "forecast_mirror_trust_initial",
    "forecast_mirror_trust_regularization",
    "forecast_mirror_trust_min_samples",
    "forecast_mirror_trust_batch_size",
    "forecast_mirror_trust_gradient_clip",
    "forecast_mirror_counterfactual_learning",
    "forecast_mirror_mask_forecast_observation",
    "forecast_mirror_counterfactual_grid_size",
    "forecast_mirror_counterfactual_payoff_scale",
    "forecast_mirror_counterfactual_loss_aversion",
    "forecast_mirror_anchor_advantage_reward",
    "forecast_mirror_anchor_advantage_weight",
    "forecast_mirror_mask_base_actor_forecast",
}


def _get_value(getter: Any, key: str, default: Any) -> Any:
    value = getter(key, None)
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    return value


def mtm_contract_settings(source: Any) -> Dict[str, Any]:
    """Extract financial MTM payoff settings that must match train/eval."""
    if isinstance(source, dict):
        get = source.get
        disable_cap = get("disable_mtm_return_cap", None)
    else:
        get = lambda key, default=None: getattr(source, key, default)
        disable_cap = getattr(source, "disable_mtm_return_cap", None)

    apply_cap = get("mtm_apply_price_return_cap", True)
    if disable_cap is not None:
        apply_cap = not _as_bool(disable_cap)

    cap_min = get("mtm_price_return_cap_min", -0.001)
    if cap_min is None:
        cap_min = -0.001
    cap_max = get("mtm_price_return_cap_max", 0.001)
    if cap_max is None:
        cap_max = 0.001

    nav_clip = get("financial_mtm_nav_clip_enabled", None)
    if nav_clip is None:
        nav_clip = apply_cap

    return {
        "mtm_return_model": str(get("mtm_return_model", "percent_capped") or "percent_capped").strip().lower(),
        "mtm_reference_price_dkk_per_mwh": float(_get_value(get, "mtm_reference_price_dkk_per_mwh", 500.0)),
        "mtm_settlement_horizon_steps": int(_get_value(get, "mtm_settlement_horizon_steps", 6)),
        "mtm_entry_price_mode": str(_get_value(get, "mtm_entry_price_mode", "current_price")).strip().lower(),
        "mtm_horizon_payoff_denominator_mode": str(
            _get_value(get, "mtm_horizon_payoff_denominator_mode", "reference_price")
        ).strip().lower().replace("-", "_"),
        "mtm_settlement_price_mode": str(
            _get_value(get, "mtm_settlement_price_mode", "energy_index")
        ).strip().lower().replace("-", "_"),
        "mtm_basis_price_column": str(_get_value(get, "mtm_basis_price_column", "price")),
        "mtm_basis_timestamp_column": str(_get_value(get, "mtm_basis_timestamp_column", "timestamp")),
        "mtm_basis_scale": float(_get_value(get, "mtm_basis_scale", 0.50)),
        "mtm_basis_centering_mode": str(
            _get_value(get, "mtm_basis_centering_mode", "rolling_median")
        ).strip().lower().replace("-", "_"),
        "mtm_basis_centering_window_steps": int(
            _get_value(get, "mtm_basis_centering_window_steps", 4320)
        ),
        "mtm_external_settlement_price_column": str(
            _get_value(get, "mtm_external_settlement_price_column", "settlement_price")
        ),
        "mtm_external_settlement_timestamp_column": str(
            _get_value(get, "mtm_external_settlement_timestamp_column", "timestamp")
        ),
        "mtm_external_settlement_min_price_dkk_per_mwh": float(
            _get_value(get, "mtm_external_settlement_min_price_dkk_per_mwh", -111750.0)
        ),
        "mtm_external_settlement_max_price_dkk_per_mwh": float(
            _get_value(get, "mtm_external_settlement_max_price_dkk_per_mwh", 111750.0)
        ),
        "mtm_apply_price_return_cap": _as_bool(apply_cap),
        "mtm_price_return_cap_min": float(cap_min),
        "mtm_price_return_cap_max": float(cap_max),
        "financial_mtm_nav_clip_enabled": _as_bool(nav_clip),
    }


def sizing_contract_settings(source: Any) -> Dict[str, Any]:
    """Extract investor notional sizing settings that must match train/eval."""
    if isinstance(source, dict):
        get = source.get
    else:
        get = lambda key, default=None: getattr(source, key, default)

    mode = str(get("investor_notional_sizing_base", "initial_trading_sleeve") or "initial_trading_sleeve")
    mode = mode.strip().lower().replace("-", "_")
    return {
        "investor_notional_sizing_base": mode,
    }


def execution_contract_settings(source: Any) -> Dict[str, Any]:
    """Extract execution/friction settings that materially affect training/eval."""
    if isinstance(source, dict):
        get = source.get
    else:
        get = lambda key, default=None: getattr(source, key, default)

    return {
        "friction_cost_multiplier": float(_get_value(get, "friction_cost_multiplier", 1.0)),
        "market_fee_model": str(
            _get_value(get, "market_fee_model", "legacy_notional_fixed")
        ).strip().lower().replace("-", "_"),
        "transaction_fee_dkk_per_mwh": float(
            _get_value(get, "transaction_fee_dkk_per_mwh", 0.0)
        ),
        "annual_market_access_fee_dkk": float(
            _get_value(get, "annual_market_access_fee_dkk", 0.0)
        ),
        "market_access_fee_allocation_fraction": float(
            _get_value(get, "market_access_fee_allocation_fraction", 1.0)
        ),
        "market_fee_source_id": str(
            _get_value(get, "market_fee_source_id", "legacy_unsourced_fixed_fee")
        ),
        "no_trade_threshold": float(_get_value(get, "no_trade_threshold", 0.01)),
        "no_trade_threshold_reference": str(
            _get_value(get, "no_trade_threshold_reference", "executable_capacity")
        ).strip().lower(),
        "half_spread_bp": float(_get_value(get, "half_spread_bp", 0.0)),
        "impact_coef_bp": float(_get_value(get, "impact_coef_bp", 0.0)),
        "impact_exponent": float(_get_value(get, "impact_exponent", 0.5)),
        "impact_ref_notional": str(_get_value(get, "impact_ref_notional", "sleeve")).strip().lower(),
        # The training file is an episode template while evaluation uses region-specific
        # files. The file path is therefore excluded, like the external settlement path;
        # the material interpretation settings below remain contract-tracked.
        "impact_volume_column": str(_get_value(get, "impact_volume_column", "")),
        "impact_volume_unit": str(_get_value(get, "impact_volume_unit", "mwh")).strip().lower(),
        "impact_volume_timestamp_column": str(_get_value(get, "impact_volume_timestamp_column", "timestamp")),
        "impact_volume_max_staleness_minutes": float(
            _get_value(get, "impact_volume_max_staleness_minutes", _get_value(get, "impact_volume_max_staleness_min", 90.0))
        ),
        "impact_volume_price_floor_dkk_per_mwh": float(
            _get_value(get, "impact_volume_price_floor_dkk_per_mwh", 50.0)
        ),
        "liquidity_participation_cap_fraction": float(
            _get_value(get, "liquidity_participation_cap_fraction", 0.0)
        ),
        "liquidity_volume_source": str(_get_value(get, "liquidity_volume_source", "load")).strip().lower(),
        "liquidity_volume_multiplier": float(_get_value(get, "liquidity_volume_multiplier", 1.0)),
        "liquidity_min_volume_mwh": float(_get_value(get, "liquidity_min_volume_mwh", 1.0)),
        "liquidity_tail_impact_threshold_dkk_per_mwh": float(
            _get_value(get, "liquidity_tail_impact_threshold_dkk_per_mwh", 5000.0)
        ),
        "liquidity_tail_impact_multiplier": float(
            _get_value(get, "liquidity_tail_impact_multiplier", 1.0)
        ),
        "liquidity_tail_impact_power": float(_get_value(get, "liquidity_tail_impact_power", 1.0)),
        "liquidity_tail_impact_max_multiplier": float(
            _get_value(get, "liquidity_tail_impact_max_multiplier", 10.0)
        ),
        "enable_collateral_cash_drag": _as_bool(_get_value(get, "enable_collateral_cash_drag", False)),
        "collateral_notional_margin_fraction": float(
            _get_value(get, "collateral_notional_margin_fraction", 0.02)
        ),
        "collateral_stress_loss_fraction": float(_get_value(get, "collateral_stress_loss_fraction", 0.10)),
        "collateral_stress_price_dkk_per_mwh": float(
            _get_value(get, "collateral_stress_price_dkk_per_mwh", 25000.0)
        ),
        "collateral_funding_rate_annual": float(_get_value(get, "collateral_funding_rate_annual", 0.05)),
        "collateral_tradeable_haircut": float(_get_value(get, "collateral_tradeable_haircut", 1.0)),
        "max_position_size": float(_get_value(get, "max_position_size", 0.35)),
        "capital_allocation_fraction": float(_get_value(get, "capital_allocation_fraction", 0.60)),
        "mtm_loss_exit_threshold_pct": float(_get_value(get, "mtm_loss_exit_threshold_pct", 0.06)),
        "distribution_rate": float(_get_value(get, "distribution_rate", 0.10)),
        "enable_trading_sleeve_margin": _as_bool(_get_value(get, "enable_trading_sleeve_margin", True)),
        "trading_sleeve_maintenance_margin_fraction": float(
            _get_value(get, "trading_sleeve_maintenance_margin_fraction", 0.05)
        ),
        "trading_sleeve_margin_liquidate_positions": _as_bool(
            _get_value(get, "trading_sleeve_margin_liquidate_positions", True)
        ),
        "trading_sleeve_margin_disable_trading": _as_bool(
            _get_value(get, "trading_sleeve_margin_disable_trading", True)
        ),
    }


def controller_contract_settings(source: Any) -> Dict[str, Any]:
    """Extract controller-mode settings that must match train/eval."""
    if isinstance(source, dict):
        get = source.get
    else:
        get = lambda key, default=None: getattr(source, key, default)

    return {
        "risk_controller_rule_based": _as_bool(_get_value(get, "risk_controller_rule_based", False)),
        "meta_controller_rule_based": _as_bool(_get_value(get, "meta_controller_rule_based", False)),
    }


def forecast_prior_contract_settings(source: Any) -> Dict[str, Any]:
    """Extract the forecast-prior settings that must match between train/eval."""
    out: Dict[str, Any] = {}
    feasible_action_enabled = _as_bool(
        source.get("forecast_prior_feasible_action_mappo", False)
        if isinstance(source, dict)
        else getattr(source, "forecast_prior_feasible_action_mappo", False)
    )
    observation_only_enabled = _as_bool(
        source.get("forecast_prior_observation_only_mappo", False)
        if isinstance(source, dict)
        else getattr(source, "forecast_prior_observation_only_mappo", False)
    )
    mirror_enabled = _as_bool(
        source.get("forecast_prior_mirror_mappo", False)
        if isinstance(source, dict)
        else getattr(source, "forecast_prior_mirror_mappo", False)
    )
    feasible_settings_relevant = bool(
        feasible_action_enabled or observation_only_enabled or mirror_enabled
    )
    for key, default, caster in FORECAST_PRIOR_CONTRACT_KEYS:
        # Feasible-action settings are hash-relevant as one atomic opt-in block.
        # The observation-only arm uses the same executable-set geometry in its
        # 20D state, so geometry settings remain contract-relevant there too.
        if key in _FEASIBLE_ACTION_CONTRACT_KEYS and not feasible_settings_relevant:
            continue
        if key in _MIRROR_CONTRACT_KEYS and not mirror_enabled:
            continue
        if key == "forecast_prior_horizon_steps":
            if isinstance(source, dict):
                if key in source:
                    value = source.get(key, default)
                else:
                    horizons = source.get("forecast_horizons", {}) or {}
                    value = horizons.get("short", source.get("short_horizon_steps", default)) if isinstance(horizons, dict) else default
            else:
                value = getattr(source, key, None)
                if value is None:
                    horizons = getattr(source, "forecast_horizons", {}) or {}
                    value = horizons.get("short", getattr(source, "short_horizon_steps", default)) if isinstance(horizons, dict) else default
        elif key == "forecast_prior_denom_floor":
            if isinstance(source, dict):
                value = source.get(key, source.get("minimum_price_filter", default))
            else:
                value = getattr(source, key, getattr(source, "minimum_price_filter", default))
        elif isinstance(source, dict):
            value = source.get(key, default)
        else:
            value = getattr(source, key, default)
        if value is None:
            value = default
        if key in {"forecast_prior_control_mode", "forecast_prior_payoff_target_mode"}:
            out[key] = str(value or default).strip().lower().replace("-", "_")
        elif key == "forecast_prior_target_alignment_version":
            out[key] = str(value or default).strip().lower()
        else:
            out[key] = caster(value)
    return out


def build_runtime_contract(
    *,
    global_norm_mode: str,
    rolling_past_history_dir: str,
    investment_freq: int,
    meta_freq_min: int,
    meta_freq_max: int,
    enable_forecast_utilization: bool = False,
    forecast_prior_settings: Optional[Dict[str, Any]] = None,
    mtm_settings: Optional[Dict[str, Any]] = None,
    sizing_settings: Optional[Dict[str, Any]] = None,
    execution_settings: Optional[Dict[str, Any]] = None,
    controller_settings: Optional[Dict[str, Any]] = None,
    log_sleeve: bool = False,
) -> Dict[str, Any]:
    """Produce the canonical Tier1 train/eval consistency contract."""
    contract = {
        "global_norm_mode": str(global_norm_mode or "rolling_past").strip().lower(),
        "rolling_past_history_dir": str(rolling_past_history_dir or "").strip(),
        "investment_freq": int(investment_freq),
        "meta_freq_min": int(meta_freq_min),
        "meta_freq_max": int(meta_freq_max),
        "enable_forecast_utilization": bool(enable_forecast_utilization),
    }
    if sizing_settings is not None:
        contract["sizing"] = dict(sizing_settings or {})
    if execution_settings is not None:
        contract["execution"] = dict(execution_settings or {})
    if controller_settings is not None:
        contract["controllers"] = dict(controller_settings or {})
    if bool(enable_forecast_utilization):
        contract["forecast_prior"] = dict(forecast_prior_settings or {})
    if mtm_settings is not None:
        contract["mtm"] = dict(mtm_settings or {})
    # sleeve-supplement task
    if bool(log_sleeve):
        contract["log_sleeve"] = True
    return contract


def runtime_contract_hash(contract: Dict[str, Any]) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


ENGINE_HASH_FILES = (
    "config.py",
    "feasible_action_mappo.py",
    "environment.py",
    "evaluation.py",
    "financial_engine.py",
    "distributional_forecast_prior.py",
    "forecast_prior.py",
    "forecast_price_experts.py",
    "generator.py",
    "forecast_mirror_mappo.py",
    "forecast_prior_cli.py",
    "logger.py",
    "main.py",
    "metacontroller.py",
    "observation_builder.py",
    "policy.py",
    "protocol_preflight.py",
    "run_tier_phase_multi_seed.py",
    "runtime_contract.py",
)


def engine_file_hashes(base_dir: Optional[str] = None) -> Dict[str, str]:
    """SHA-256 of the shared engine files, recorded as run metadata.

    Deliberately NOT part of the runtime contract hash (that would break
    re-evaluation after any cosmetic edit). Its purpose is post-hoc proof
    that Prototype5 main runs and Prototype5 ablation runs executed the same engine
    code, up to documented, contract-tracked mechanism differences.
    """
    root = base_dir or os.path.dirname(os.path.abspath(__file__))
    out: Dict[str, str] = {}
    for name in ENGINE_HASH_FILES:
        path = os.path.join(root, name)
        try:
            with open(path, "rb") as f:
                out[name] = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            out[name] = ""
    return out

#!/usr/bin/env python3
"""Run final paper-comparable deterministic baselines for Prototype5.

This runner evaluates baselines 1-3 with the same accounting protocol used by
the final MAPPO/MARL experiments:

- forward-filled evaluation data;
- causal fixed-horizon settlement (`horizon_settlement`);
- current observable entry price;
- explicit MWh-volume payoff (`volume_MWh * settlement spread_DKK/MWh`);
- fixed initial trading-sleeve sizing;
- final transaction, spread, and market-impact costs;
- daily Newey-West/HAC trading-sleeve Sharpe as the primary sleeve risk metric.

The capped-long price-sleeve comparator is included explicitly because the
Markowitz optimizer can collapse to a near-constant capped long price exposure
under this protocol. Reporting both avoids overstating the classical optimizer's
dynamic content.

The same-hour previous-day entry rule is retained only as a diagnostic
counterfactual because it allows a rule to observe the current price and then
enter at yesterday's same-hour price. It is not used as the reported tradable
baseline protocol.

The IEEE compliance tool is a diagnostic, not a financial baseline. SARL is not
included unless a current trained SARL checkpoint is explicitly supplied outside
this runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "baseline_results" / "market_fee_v4"
DEFAULT_TIMESTEPS = 39311
DEFAULT_SEED = 42

# Harmonized EU balancing-energy technical price bounds (±15,000 EUR/MWh at
# ~7.45 DKK/EUR). Wider than every observed raw settlement print in the
# datasets, so this is a technical guard, not a distribution-shaping clip.
SETTLEMENT_CLIP_MIN_DKK_DEFAULT = -111750.0
SETTLEMENT_CLIP_MAX_DKK_DEFAULT = 111750.0


def _force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


_force_utf8_stdio()


FINAL_PROTOCOL: Dict[str, Any] = {
    "protocol_name": "prototype5_final_campaign_v1",
    "price_fill": "ffill_only",
    "distribution_rate": 0.0,
    "eval_distribution_rate": 0.0,
    "mtm_return_model": "horizon_settlement",
    "mtm_reference_price_dkk_per_mwh": 500.0,
    "mtm_settlement_horizon_steps": 6,
    "mtm_entry_price_mode": "current_price",
    "mtm_horizon_payoff_denominator_mode": "mwh_volume",
    "mtm_settlement_price_mode": "external_series",
    "mtm_basis_price_data_path": "",
    "mtm_basis_scale": 0.50,
    "mtm_basis_centering_mode": "rolling_median",
    "mtm_basis_centering_window_steps": 4320,
    "mtm_external_settlement_price_data_path": "",
    "mtm_external_settlement_price_column": "settlement_price",
    "mtm_external_settlement_timestamp_column": "timestamp",
    "mtm_external_settlement_min_price_dkk_per_mwh": SETTLEMENT_CLIP_MIN_DKK_DEFAULT,
    "mtm_external_settlement_max_price_dkk_per_mwh": SETTLEMENT_CLIP_MAX_DKK_DEFAULT,
    "mtm_apply_price_return_cap": False,
    "investor_notional_sizing_base": "initial_trading_sleeve",
    "max_position_size": 0.10,
    "capital_allocation_fraction": 0.60,
    "mtm_loss_exit_threshold_pct": 0.15,
    "friction_cost_multiplier": 1.0,
    "market_fee_model": "nord_pool_intraday_2026",
    "transaction_fee_dkk_per_mwh": 0.9238,
    "annual_market_access_fee_dkk": 160175.0,
    "market_access_fee_allocation_fraction": 1.0,
    "market_fee_source_id": "nord_pool_nordic_baltic_2026_standard_participant",
    "no_trade_threshold": 0.01,
    "no_trade_threshold_reference": "executable_capacity",
    "half_spread_bp": 5.0,
    "impact_coef_bp": 20.0,
    "impact_exponent": 0.5,
    "impact_ref_notional": "volume",
    "impact_volume_data_path": "",
    "impact_volume_column": "market_volume_mwh",
    "impact_volume_unit": "mwh",
    "impact_volume_timestamp_column": "timestamp",
    "impact_volume_max_staleness_minutes": 90.0,
    "impact_volume_price_floor_dkk_per_mwh": 50.0,
    "liquidity_participation_cap_fraction": 0.25,
    "liquidity_volume_source": "impact_volume",
    "liquidity_volume_multiplier": 1.0,
    "liquidity_min_volume_mwh": 1.0,
    "liquidity_tail_impact_threshold_dkk_per_mwh": 5000.0,
    "liquidity_tail_impact_multiplier": 3.0,
    "liquidity_tail_impact_power": 1.0,
    "liquidity_tail_impact_max_multiplier": 10.0,
    "enable_collateral_cash_drag": True,
    "collateral_notional_margin_fraction": 0.02,
    "collateral_stress_loss_fraction": 0.10,
    "collateral_stress_price_dkk_per_mwh": 25000.0,
    "collateral_funding_rate_annual": 0.05,
    "collateral_tradeable_haircut": 1.0,
    "investment_freq": 6,
    "annual_risk_free_rate": 0.02,
    "sleeve_sharpe_mode": "daily_hac_7",
    "enable_trading_sleeve_margin": True,
    "trading_sleeve_maintenance_margin_fraction": 0.05,
    "trading_sleeve_margin_liquidate_positions": True,
    "trading_sleeve_margin_disable_trading": True,
}


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _optional_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _format_optional(value: Any, width: int = 14, precision: int = 3) -> str:
    out = _optional_float(value)
    if out is None:
        return f"{'N/A':>{width}}"
    return f"{out:>{width}.{precision}f}"


def _path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _available_regions(include_v2: bool = True) -> List[Dict[str, Any]]:
    regions = [
        {
            "region": "original",
            "label": "original_price_region",
            "eval_data": PROJECT_ROOT / "evaluation_dataset_ffill" / "unseendata.csv",
        }
    ]
    v2_path = PROJECT_ROOT / "evaluation_dataset_ffill" / "unseendata_v2.csv"
    if include_v2 and v2_path.is_file():
        regions.append(
            {
                "region": "unseendata_v2",
                "label": "second_price_region",
                "eval_data": v2_path,
            }
        )
    return regions


def install_final_protocol_config_patch() -> None:
    """Patch config.EnhancedConfig in this process before importing evaluation."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import config as config_module

    original_cls = config_module.EnhancedConfig

    class FinalPaperBaselineConfig(original_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.distribution_rate = float(FINAL_PROTOCOL["distribution_rate"])
            self.eval_distribution_rate = float(FINAL_PROTOCOL["eval_distribution_rate"])
            self.mtm_return_model = str(FINAL_PROTOCOL["mtm_return_model"])
            self.mtm_reference_price_dkk_per_mwh = float(
                FINAL_PROTOCOL["mtm_reference_price_dkk_per_mwh"]
            )
            self.mtm_settlement_horizon_steps = int(
                FINAL_PROTOCOL["mtm_settlement_horizon_steps"]
            )
            self.mtm_entry_price_mode = str(FINAL_PROTOCOL["mtm_entry_price_mode"])
            self.mtm_horizon_payoff_denominator_mode = str(
                FINAL_PROTOCOL["mtm_horizon_payoff_denominator_mode"]
            )
            self.mtm_settlement_price_mode = str(FINAL_PROTOCOL["mtm_settlement_price_mode"])
            self.mtm_basis_price_data_path = str(FINAL_PROTOCOL["mtm_basis_price_data_path"])
            self.mtm_basis_scale = float(FINAL_PROTOCOL["mtm_basis_scale"])
            self.mtm_basis_centering_mode = str(FINAL_PROTOCOL["mtm_basis_centering_mode"])
            self.mtm_basis_centering_window_steps = int(
                FINAL_PROTOCOL["mtm_basis_centering_window_steps"]
            )
            self.mtm_external_settlement_price_data_path = str(
                FINAL_PROTOCOL["mtm_external_settlement_price_data_path"]
            )
            self.mtm_external_settlement_price_column = str(
                FINAL_PROTOCOL["mtm_external_settlement_price_column"]
            )
            self.mtm_external_settlement_timestamp_column = str(
                FINAL_PROTOCOL["mtm_external_settlement_timestamp_column"]
            )
            self.mtm_external_settlement_min_price_dkk_per_mwh = float(
                FINAL_PROTOCOL["mtm_external_settlement_min_price_dkk_per_mwh"]
            )
            self.mtm_external_settlement_max_price_dkk_per_mwh = float(
                FINAL_PROTOCOL["mtm_external_settlement_max_price_dkk_per_mwh"]
            )
            self.mtm_apply_price_return_cap = bool(FINAL_PROTOCOL["mtm_apply_price_return_cap"])
            self.investor_notional_sizing_base = str(
                FINAL_PROTOCOL["investor_notional_sizing_base"]
            )
            self.max_position_size = float(FINAL_PROTOCOL["max_position_size"])
            self.capital_allocation_fraction = float(
                FINAL_PROTOCOL["capital_allocation_fraction"]
            )
            self.mtm_loss_exit_threshold_pct = float(
                FINAL_PROTOCOL["mtm_loss_exit_threshold_pct"]
            )
            self.friction_cost_multiplier = float(FINAL_PROTOCOL["friction_cost_multiplier"])
            self.market_fee_model = str(FINAL_PROTOCOL["market_fee_model"])
            self.transaction_fee_dkk_per_mwh = float(
                FINAL_PROTOCOL["transaction_fee_dkk_per_mwh"]
            )
            self.annual_market_access_fee_dkk = float(
                FINAL_PROTOCOL["annual_market_access_fee_dkk"]
            )
            self.market_access_fee_allocation_fraction = float(
                FINAL_PROTOCOL["market_access_fee_allocation_fraction"]
            )
            self.market_fee_source_id = str(FINAL_PROTOCOL["market_fee_source_id"])
            self.no_trade_threshold = float(FINAL_PROTOCOL["no_trade_threshold"])
            self.no_trade_threshold_reference = str(
                FINAL_PROTOCOL["no_trade_threshold_reference"]
            )
            self.half_spread_bp = float(FINAL_PROTOCOL["half_spread_bp"])
            self.impact_coef_bp = float(FINAL_PROTOCOL["impact_coef_bp"])
            self.impact_exponent = float(FINAL_PROTOCOL["impact_exponent"])
            self.impact_ref_notional = str(FINAL_PROTOCOL["impact_ref_notional"])
            self.impact_volume_data_path = str(FINAL_PROTOCOL["impact_volume_data_path"])
            self.impact_volume_column = str(FINAL_PROTOCOL["impact_volume_column"])
            self.impact_volume_unit = str(FINAL_PROTOCOL["impact_volume_unit"])
            self.impact_volume_timestamp_column = str(FINAL_PROTOCOL["impact_volume_timestamp_column"])
            self.impact_volume_max_staleness_minutes = float(
                FINAL_PROTOCOL["impact_volume_max_staleness_minutes"]
            )
            self.impact_volume_price_floor_dkk_per_mwh = float(
                FINAL_PROTOCOL["impact_volume_price_floor_dkk_per_mwh"]
            )
            self.liquidity_participation_cap_fraction = float(
                FINAL_PROTOCOL["liquidity_participation_cap_fraction"]
            )
            self.liquidity_volume_source = str(FINAL_PROTOCOL["liquidity_volume_source"])
            self.liquidity_volume_multiplier = float(FINAL_PROTOCOL["liquidity_volume_multiplier"])
            self.liquidity_min_volume_mwh = float(FINAL_PROTOCOL["liquidity_min_volume_mwh"])
            self.liquidity_tail_impact_threshold_dkk_per_mwh = float(
                FINAL_PROTOCOL["liquidity_tail_impact_threshold_dkk_per_mwh"]
            )
            self.liquidity_tail_impact_multiplier = float(
                FINAL_PROTOCOL["liquidity_tail_impact_multiplier"]
            )
            self.liquidity_tail_impact_power = float(FINAL_PROTOCOL["liquidity_tail_impact_power"])
            self.liquidity_tail_impact_max_multiplier = float(
                FINAL_PROTOCOL["liquidity_tail_impact_max_multiplier"]
            )
            self.enable_collateral_cash_drag = bool(FINAL_PROTOCOL["enable_collateral_cash_drag"])
            self.collateral_notional_margin_fraction = float(
                FINAL_PROTOCOL["collateral_notional_margin_fraction"]
            )
            self.collateral_stress_loss_fraction = float(
                FINAL_PROTOCOL["collateral_stress_loss_fraction"]
            )
            self.collateral_stress_price_dkk_per_mwh = float(
                FINAL_PROTOCOL["collateral_stress_price_dkk_per_mwh"]
            )
            self.collateral_funding_rate_annual = float(FINAL_PROTOCOL["collateral_funding_rate_annual"])
            self.collateral_tradeable_haircut = float(FINAL_PROTOCOL["collateral_tradeable_haircut"])
            self.investment_freq = int(FINAL_PROTOCOL["investment_freq"])
            self.sleeve_sharpe_mode = str(FINAL_PROTOCOL["sleeve_sharpe_mode"])
            self.enable_trading_sleeve_margin = bool(FINAL_PROTOCOL["enable_trading_sleeve_margin"])
            self.trading_sleeve_maintenance_margin_fraction = float(
                FINAL_PROTOCOL["trading_sleeve_maintenance_margin_fraction"]
            )
            self.trading_sleeve_margin_liquidate_positions = bool(
                FINAL_PROTOCOL["trading_sleeve_margin_liquidate_positions"]
            )
            self.trading_sleeve_margin_disable_trading = bool(
                FINAL_PROTOCOL["trading_sleeve_margin_disable_trading"]
            )
            self.final_paper_baseline_protocol = True

    config_module.EnhancedConfig = FinalPaperBaselineConfig


def _normalize_row(region: str, baseline_key: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "region": region,
        "baseline": baseline_key,
        "method": str(result.get("method", baseline_key)),
        "status": str(result.get("status", "completed")),
        "role": str(result.get("role", "")),
        "final_portfolio_value_usd": _safe_float(
            result.get("final_portfolio_value", result.get("distribution_adjusted_final_value_usd"))
        ),
        "initial_portfolio_value_usd": _safe_float(
            result.get("initial_portfolio_value", result.get("initial_value_usd")),
            800_000_000.0,
        ),
        "total_return_pct": 100.0 * _safe_float(result.get("total_return")),
        "annual_return_pct": 100.0 * _safe_float(result.get("annual_return")),
        "total_nav_sharpe_ratio": _safe_float(result.get("sharpe_ratio")),
        "max_drawdown_pct": 100.0 * _safe_float(result.get("max_drawdown")),
        "sleeve_trading_return_pct": _safe_float(result.get("sleeve_trading_return_pct")),
        "sleeve_trading_sharpe_ratio": _optional_float(result.get("sleeve_trading_sharpe_ratio")),
        "sleeve_trading_sharpe_ratio_raw": _optional_float(
            result.get("sleeve_trading_sharpe_ratio_raw")
        ),
        "sleeve_trading_ruined": bool(result.get("sleeve_trading_ruined", False)),
        "sleeve_trading_sharpe_mode": str(result.get("sleeve_trading_sharpe_primary_mode", "")),
        "sleeve_trading_max_drawdown_pct": _safe_float(
            result.get("sleeve_trading_max_drawdown_pct")
        ),
        "sleeve_mean_abs_exposure_dkk": _safe_float(result.get("sleeve_mean_abs_exposure_dkk")),
        "sleeve_max_abs_exposure_dkk": _safe_float(result.get("sleeve_max_abs_exposure_dkk")),
        "total_transaction_costs_usd": _safe_float(result.get("total_transaction_costs_usd")),
        "total_volume_transaction_fees_usd": _safe_float(
            result.get("total_volume_transaction_fees_usd")
        ),
        "total_market_access_fees_usd": _safe_float(
            result.get("total_market_access_fees_usd")
        ),
        "total_market_impact_cost_usd": _safe_float(result.get("total_market_impact_cost_usd")),
        "total_collateral_funding_cost_usd": _safe_float(result.get("total_collateral_funding_cost_usd")),
        "total_mtm_pnl_usd": _safe_float(result.get("total_mtm_pnl_usd")),
        "total_operational_revenue_usd": _safe_float(result.get("total_operational_revenue_usd")),
        "eval_data_path": str(result.get("eval_data_path", "")),
    }


def _read_rule_summary(region: str, output_dir: Path) -> List[Dict[str, Any]]:
    summary_path = output_dir / "rule_price_entry_baseline_summary.csv"
    if not summary_path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with summary_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "region": region,
                    "baseline": f"rule_price_entry_{row.get('entry_mode', '')}",
                    "method": "Rule current-price entry heuristic",
                    "status": row.get("status", "completed"),
                    "role": "deterministic_current_price_rule",
                    "final_portfolio_value_usd": "",
                    "initial_portfolio_value_usd": "",
                    "total_return_pct": _safe_float(row.get("return_pct")),
                    "annual_return_pct": "",
                    "total_nav_sharpe_ratio": "",
                    "max_drawdown_pct": 100.0 * _safe_float(row.get("max_drawdown")),
                    "sleeve_trading_return_pct": _safe_float(row.get("return_pct")),
                    "sleeve_trading_sharpe_ratio": _optional_float(
                        row.get("daily_hac7_sharpe", row.get("daily_sharpe"))
                    ),
                    "sleeve_trading_sharpe_ratio_raw": _optional_float(
                        row.get("daily_hac7_sharpe_raw", row.get("daily_hac7_sharpe"))
                    ),
                    "sleeve_trading_ruined": str(row.get("margin_active", "")).strip().lower()
                    in {"1", "true", "yes"},
                    "sleeve_trading_sharpe_mode": "daily_hac_7",
                    "sleeve_trading_max_drawdown_pct": 100.0 * _safe_float(row.get("max_drawdown")),
                    "sleeve_mean_abs_exposure_dkk": "",
                    "sleeve_max_abs_exposure_dkk": "",
                    "total_transaction_costs_usd": "",
                    "total_market_impact_cost_usd": "",
                    "total_collateral_funding_cost_usd": "",
                    "total_mtm_pnl_usd": "",
                    "total_operational_revenue_usd": "",
                    "eval_data_path": "",
                }
            )
    return rows


def _run_rule_price_entry(
    region: str,
    eval_data: Path,
    output_dir: Path,
    seed: int,
    *,
    include_previous_day_diagnostic: bool = False,
    settlement_price_data: str = "",
    basis_scale: float = 0.50,
    basis_centering_mode: str = "rolling_median",
    basis_centering_window_steps: int = 4320,
    settlement_clip_min_dkk: float = SETTLEMENT_CLIP_MIN_DKK_DEFAULT,
    settlement_clip_max_dkk: float = SETTLEMENT_CLIP_MAX_DKK_DEFAULT,
) -> None:
    entry_modes = ["current_price"]
    if include_previous_day_diagnostic:
        entry_modes.append("same_hour_prev_day")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_rule_price_entry_baseline.py"),
        "--eval_data",
        str(eval_data),
        "--output_dir",
        str(output_dir),
        "--entry_modes",
        *entry_modes,
        "--seed",
        str(seed),
        "--decision_freq",
        str(FINAL_PROTOCOL["investment_freq"]),
        "--horizon_steps",
        str(FINAL_PROTOCOL["mtm_settlement_horizon_steps"]),
        "--annual_risk_free_rate",
        str(FINAL_PROTOCOL["annual_risk_free_rate"]),
        "--capital_allocation_fraction",
        str(FINAL_PROTOCOL["capital_allocation_fraction"]),
        "--max_position_size",
        str(FINAL_PROTOCOL["max_position_size"]),
        "--rule_exposure_cap",
        "0.60",
        "--rule_gain",
        "3.0",
        "--no_trade_band",
        "0.002",
        "--signal_denom_floor",
        "50.0",
        "--payoff_reference_price",
        str(FINAL_PROTOCOL["mtm_reference_price_dkk_per_mwh"]),
        "--payoff_mode",
        "mwh_volume",
        "--friction_cost_multiplier",
        str(FINAL_PROTOCOL["friction_cost_multiplier"]),
        "--market_fee_model",
        str(FINAL_PROTOCOL["market_fee_model"]),
        "--transaction_fee_dkk_per_mwh",
        str(FINAL_PROTOCOL["transaction_fee_dkk_per_mwh"]),
        "--annual_market_access_fee_dkk",
        str(FINAL_PROTOCOL["annual_market_access_fee_dkk"]),
        "--market_access_fee_allocation_fraction",
        str(FINAL_PROTOCOL["market_access_fee_allocation_fraction"]),
        "--market_fee_source_id",
        str(FINAL_PROTOCOL["market_fee_source_id"]),
        "--no_trade_threshold",
        str(FINAL_PROTOCOL["no_trade_threshold"]),
        "--half_spread_bp",
        str(FINAL_PROTOCOL["half_spread_bp"]),
        "--impact_coef_bp",
        str(FINAL_PROTOCOL["impact_coef_bp"]),
        "--impact_exponent",
        str(FINAL_PROTOCOL["impact_exponent"]),
        "--liquidity_participation_cap_fraction",
        str(FINAL_PROTOCOL["liquidity_participation_cap_fraction"]),
        "--liquidity_volume_source",
        str(FINAL_PROTOCOL["liquidity_volume_source"]),
        "--impact_volume_data",
        str(FINAL_PROTOCOL["impact_volume_data_path"]),
        "--impact_volume_column",
        str(FINAL_PROTOCOL["impact_volume_column"]),
        "--impact_volume_unit",
        str(FINAL_PROTOCOL["impact_volume_unit"]),
        "--impact_volume_timestamp_column",
        str(FINAL_PROTOCOL["impact_volume_timestamp_column"]),
        "--impact_volume_max_staleness_min",
        str(FINAL_PROTOCOL["impact_volume_max_staleness_minutes"]),
        "--impact_volume_price_floor_dkk_per_mwh",
        str(FINAL_PROTOCOL["impact_volume_price_floor_dkk_per_mwh"]),
        "--liquidity_volume_multiplier",
        str(FINAL_PROTOCOL["liquidity_volume_multiplier"]),
        "--liquidity_min_volume_mwh",
        str(FINAL_PROTOCOL["liquidity_min_volume_mwh"]),
        "--liquidity_tail_impact_threshold_dkk_per_mwh",
        str(FINAL_PROTOCOL["liquidity_tail_impact_threshold_dkk_per_mwh"]),
        "--liquidity_tail_impact_multiplier",
        str(FINAL_PROTOCOL["liquidity_tail_impact_multiplier"]),
        "--liquidity_tail_impact_power",
        str(FINAL_PROTOCOL["liquidity_tail_impact_power"]),
        "--liquidity_tail_impact_max_multiplier",
        str(FINAL_PROTOCOL["liquidity_tail_impact_max_multiplier"]),
        "--enable_collateral_cash_drag",
        "--collateral_notional_margin_fraction",
        str(FINAL_PROTOCOL["collateral_notional_margin_fraction"]),
        "--collateral_stress_loss_fraction",
        str(FINAL_PROTOCOL["collateral_stress_loss_fraction"]),
        "--collateral_stress_price_dkk_per_mwh",
        str(FINAL_PROTOCOL["collateral_stress_price_dkk_per_mwh"]),
        "--collateral_funding_rate_annual",
        str(FINAL_PROTOCOL["collateral_funding_rate_annual"]),
        "--collateral_tradeable_haircut",
        str(FINAL_PROTOCOL["collateral_tradeable_haircut"]),
    ]
    if settlement_price_data:
        cmd.extend(
            [
                "--settlement_price_data",
                str(settlement_price_data),
                "--settlement_price_mode",
                str(FINAL_PROTOCOL["mtm_settlement_price_mode"]),
                "--settlement_price_column",
                str(FINAL_PROTOCOL["mtm_external_settlement_price_column"]),
                "--settlement_timestamp_column",
                str(FINAL_PROTOCOL["mtm_external_settlement_timestamp_column"]),
                "--settlement_clip_min_dkk",
                str(float(settlement_clip_min_dkk)),
                "--settlement_clip_max_dkk",
                str(float(settlement_clip_max_dkk)),
                "--basis_scale",
                str(float(basis_scale)),
                "--basis_centering_mode",
                str(basis_centering_mode),
                "--basis_centering_window_steps",
                str(int(basis_centering_window_steps)),
            ]
        )
    print("")
    print(f"[RUN] Rule current-price entry heuristic for {region}")
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=_subprocess_env(), check=True)


def write_outputs(
    *,
    output_root: Path,
    payload: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> Dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_root / f"final_paper_baseline_evaluation_{timestamp}.json"
    csv_path = output_root / f"final_paper_baseline_summary_{timestamp}.csv"
    latest_json = output_root / "final_paper_baseline_evaluation_latest.json"
    latest_csv = output_root / "final_paper_baseline_summary_latest.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    with latest_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    fieldnames = list(rows[0].keys()) if rows else ["region", "baseline", "status"]
    for path in (csv_path, latest_csv):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return {
        "json": json_path,
        "csv": csv_path,
        "latest_json": latest_json,
        "latest_csv": latest_csv,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--skip_v2",
        action="store_true",
        help="Only evaluate evaluation_dataset_ffill/unseendata.csv.",
    )
    parser.add_argument(
        "--skip_rule_price_entry",
        action="store_true",
        help="Skip the deterministic current-price rule baseline.",
    )
    parser.add_argument(
        "--include_previous_day_rule_diagnostic",
        action="store_true",
        help=(
            "Also run the non-tradable previous-day entry diagnostic. "
            "This is excluded by default because it can observe the current "
            "price and enter at yesterday's same-hour price."
        ),
    )
    parser.add_argument("--basis_risk", action="store_true")
    parser.add_argument("--energy_index_settlement", action="store_true")
    parser.add_argument("--settlement_eval_original", default="evaluation_dataset_ffill/unseendata_settlement_real_v2.csv")
    parser.add_argument("--settlement_eval_v2", default="evaluation_dataset_ffill/unseendata_v2_settlement_real_v2.csv")
    parser.add_argument("--settlement_price_column", default="settlement_price")
    parser.add_argument("--settlement_timestamp_column", default="timestamp")
    parser.add_argument("--liquidity_volume_eval_original", default="evaluation_dataset_ffill/unseendata_liquidity_volume_real_v1.csv")
    parser.add_argument("--liquidity_volume_eval_v2", default="evaluation_dataset_ffill/unseendata_v2_liquidity_volume_real_v1.csv")
    parser.add_argument("--liquidity_volume_column", default="market_volume_mwh")
    parser.add_argument("--liquidity_volume_unit", choices=["mwh", "dkk"], default="mwh")
    parser.add_argument("--liquidity_volume_timestamp_column", default="timestamp")
    parser.add_argument("--liquidity_volume_max_staleness_min", type=float, default=90.0)
    parser.add_argument("--impact_volume_price_floor_dkk_per_mwh", type=float, default=50.0)
    parser.add_argument(
        "--settlement_clip_min_dkk",
        type=float,
        default=SETTLEMENT_CLIP_MIN_DKK_DEFAULT,
        help="Lower technical bound for the external settlement series (DKK/MWh).",
    )
    parser.add_argument(
        "--settlement_clip_max_dkk",
        type=float,
        default=SETTLEMENT_CLIP_MAX_DKK_DEFAULT,
        help="Upper technical bound for the external settlement series (DKK/MWh).",
    )
    parser.add_argument("--basis_eval_original", default="evaluation_dataset_ffill/unseendata_v2.csv")
    parser.add_argument("--basis_eval_v2", default="evaluation_dataset_ffill/unseendata.csv")
    parser.add_argument("--basis_scale", type=float, default=0.50)
    parser.add_argument("--basis_centering_mode", choices=["none", "rolling_median", "expanding_median"], default="rolling_median")
    parser.add_argument("--basis_centering_window_steps", type=int, default=4320)
    args = parser.parse_args()

    FINAL_PROTOCOL["mtm_external_settlement_min_price_dkk_per_mwh"] = float(args.settlement_clip_min_dkk)
    FINAL_PROTOCOL["mtm_external_settlement_max_price_dkk_per_mwh"] = float(args.settlement_clip_max_dkk)
    FINAL_PROTOCOL["impact_volume_column"] = str(args.liquidity_volume_column)
    FINAL_PROTOCOL["impact_volume_unit"] = str(args.liquidity_volume_unit)
    FINAL_PROTOCOL["impact_volume_timestamp_column"] = str(args.liquidity_volume_timestamp_column)
    FINAL_PROTOCOL["impact_volume_max_staleness_minutes"] = float(args.liquidity_volume_max_staleness_min)
    FINAL_PROTOCOL["impact_volume_price_floor_dkk_per_mwh"] = float(args.impact_volume_price_floor_dkk_per_mwh)

    output_root = _path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    regions = _available_regions(include_v2=not args.skip_v2)
    for region in regions:
        if not region["eval_data"].is_file():
            raise SystemExit(f"Evaluation data not found: {region['eval_data']}")

    install_final_protocol_config_patch()

    from evaluation import run_traditional_baselines

    all_results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    print("Final paper baseline protocol")
    print(f"  project root: {PROJECT_ROOT}")
    print(f"  output root:  {output_root}")
    print(f"  timesteps:    {int(args.timesteps)}")
    print(f"  seed:         {int(args.seed)}")
    print(f"  protocol:     {FINAL_PROTOCOL['protocol_name']}")

    for region in regions:
        region_key = str(region["region"])
        eval_data = Path(region["eval_data"]).resolve()
        region_out = output_root / region_key
        region_out.mkdir(parents=True, exist_ok=True)
        if bool(args.basis_risk):
            basis_path = args.basis_eval_v2 if region_key == "unseendata_v2" else args.basis_eval_original
            FINAL_PROTOCOL["mtm_settlement_price_mode"] = "cross_zone_basis"
            FINAL_PROTOCOL["mtm_basis_price_data_path"] = str(basis_path)
            FINAL_PROTOCOL["mtm_basis_scale"] = float(args.basis_scale)
            FINAL_PROTOCOL["mtm_basis_centering_mode"] = str(args.basis_centering_mode)
            FINAL_PROTOCOL["mtm_basis_centering_window_steps"] = int(args.basis_centering_window_steps)
            FINAL_PROTOCOL["mtm_external_settlement_price_data_path"] = ""
            FINAL_PROTOCOL["mtm_external_settlement_price_column"] = "price"
            FINAL_PROTOCOL["mtm_external_settlement_timestamp_column"] = "timestamp"
        elif bool(args.energy_index_settlement):
            basis_path = ""
            FINAL_PROTOCOL["mtm_settlement_price_mode"] = "energy_index"
            FINAL_PROTOCOL["mtm_basis_price_data_path"] = ""
            FINAL_PROTOCOL["mtm_external_settlement_price_data_path"] = ""
        else:
            basis_path = args.settlement_eval_v2 if region_key == "unseendata_v2" else args.settlement_eval_original
            FINAL_PROTOCOL["mtm_settlement_price_mode"] = "external_series"
            FINAL_PROTOCOL["mtm_basis_price_data_path"] = ""
            FINAL_PROTOCOL["mtm_external_settlement_price_data_path"] = str(basis_path)
            FINAL_PROTOCOL["mtm_external_settlement_price_column"] = str(args.settlement_price_column)
            FINAL_PROTOCOL["mtm_external_settlement_timestamp_column"] = str(args.settlement_timestamp_column)
        liquidity_path = (
            args.liquidity_volume_eval_v2
            if region_key == "unseendata_v2"
            else args.liquidity_volume_eval_original
        )
        FINAL_PROTOCOL["impact_volume_data_path"] = str(liquidity_path)
        region_protocol = dict(FINAL_PROTOCOL)

        print("")
        print(f"[RUN] Baselines 1-4 for {region_key}: {eval_data}")
        results = run_traditional_baselines(
            str(eval_data),
            timesteps=int(args.timesteps),
            output_dir=str(region_out),
            seed=int(args.seed),
        )
        for baseline_key, result in results.items():
            if isinstance(result, dict):
                result["final_paper_protocol"] = dict(region_protocol)
                result["eval_data_path"] = str(eval_data)
        all_results[region_key] = {
            "eval_data_path": str(eval_data),
            "protocol": dict(region_protocol),
            "baseline_results": results,
        }
        rows.extend(_normalize_row(region_key, key, result) for key, result in results.items())

        if not args.skip_rule_price_entry:
            rule_dir_name = (
                "rule_current_price_with_previous_day_diagnostic"
                if args.include_previous_day_rule_diagnostic
                else "rule_current_price_entry"
            )
            rule_out = region_out / rule_dir_name
            _run_rule_price_entry(
                region_key,
                eval_data,
                rule_out,
                int(args.seed),
                include_previous_day_diagnostic=bool(args.include_previous_day_rule_diagnostic),
                settlement_price_data=str(basis_path),
                basis_scale=float(args.basis_scale),
                basis_centering_mode=str(args.basis_centering_mode),
                basis_centering_window_steps=int(args.basis_centering_window_steps),
                settlement_clip_min_dkk=float(args.settlement_clip_min_dkk),
                settlement_clip_max_dkk=float(args.settlement_clip_max_dkk),
            )
            rows.extend(_read_rule_summary(region_key, rule_out))

    payload = {
        "evaluation_type": "final_paper_baselines",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "output_root": str(output_root),
        "timesteps_requested": int(args.timesteps),
        "seed": int(args.seed),
        "protocol": dict(FINAL_PROTOCOL),
        "included_regions": [str(r["region"]) for r in regions],
        "baseline_scope": {
            "included": [
                "Baseline1 Traditional Portfolio",
                "Baseline2 Rule-Based Heuristic",
                "Baseline3 Hybrid Buy-and-Hold",
                "Baseline4 Capped Long Price Sleeve",
                "Rule current-price entry heuristic",
            ],
            "excluded_by_default": [
                "IEEE compliance diagnostic, because it is not a financial strategy baseline",
                "SARL, because a current trained SARL checkpoint must be supplied explicitly",
                "Previous-day entry rule diagnostic, because it is non-tradable after observing the current price",
            ],
        },
        "results": all_results,
        "comparison_rows": rows,
    }
    paths = write_outputs(output_root=output_root, payload=payload, rows=rows)

    print("")
    print("Final paper baseline summary")
    print("-" * 118)
    print(
        f"{'region':<15} {'baseline':<36} {'return %':>10} "
        f"{'sleeve ret %':>13} {'sleeve sharpe':>14} {'sleeve DD %':>12}"
    )
    for row in rows:
        print(
            f"{str(row.get('region', '')):<15} "
            f"{str(row.get('baseline', '')):<36} "
            f"{_safe_float(row.get('total_return_pct')):>10.3f} "
            f"{_safe_float(row.get('sleeve_trading_return_pct')):>13.3f} "
            f"{_format_optional(row.get('sleeve_trading_sharpe_ratio'), 14, 3)} "
            f"{_safe_float(row.get('sleeve_trading_max_drawdown_pct')):>12.3f}"
        )

    print("")
    print(f"Wrote JSON: {paths['json']}")
    print(f"Wrote CSV:  {paths['csv']}")
    print(f"Latest JSON: {paths['latest_json']}")
    print(f"Latest CSV:  {paths['latest_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# UNIFIED CONSTANTS (SINGLE SOURCE OF TRUTH)
# ============================================================================
# These constants ensure consistent configuration across all modes and code paths

BASE_FEATURE_DIM = 18  # Base observation feature dimension (policy obs space)

# Price normalization constants
PRICE_MEAN = 250.0          # Historical mean price (DKK/MWh)
PRICE_STD = 50.0            # Historical std dev (DKK/MWh)
PRICE_CLIP_SIGMA = 3.0      # Clip at Â±3 sigma for outlier handling

# Risk controller constants
RISK_LOOKBACK_WINDOW_DEFAULT = 144  # Default lookback window for risk calculations
RISK_LOOKBACK_WINDOW_MAX = 200      # Maximum lookback window (memory cap)

# Adaptive scaling constants
ADAPTIVE_SCALE_SATURATION_THRESHOLD = 0.95
ADAPTIVE_SCALE_MIN_OFFSET = 0.01            # Small offset from min/max boundaries
ADAPTIVE_SCALE_COMPRESSION_FACTOR = 5.0     # Compression factor for values near boundaries

# Regulatory risk constants
REGULATORY_RISK_BASE_STRESS_WEIGHT = 0.4    # Weight for market stress in regulatory risk
REGULATORY_RISK_FALLBACK = 0.20             # Fallback regulatory risk when no history
REGULATORY_RISK_SEASON_BASE = 0.05          # Base seasonal component
REGULATORY_RISK_SEASON_AMPLITUDE = 0.05     # Amplitude of seasonal variation
REGULATORY_RISK_FALLBACK_ERROR = 0.35       # Fallback value on calculation error

# Risk fallback values (on calculation errors)
RISK_FALLBACK_MARKET = 0.30
RISK_FALLBACK_OPERATIONAL = 0.20
RISK_FALLBACK_PORTFOLIO = 0.25
RISK_FALLBACK_LIQUIDITY = 0.15
RISK_FALLBACK_REGULATORY = 0.35
RISK_FALLBACK_OVERALL = 0.25

# Risk action defaults
RISK_ACTION_MULTIPLIER_DEFAULT = 1.0
RISK_ACTION_MAX_INVESTMENT_DEFAULT = 0.30
RISK_ACTION_CASH_RESERVE_DEFAULT = 0.10
RISK_ACTION_HEDGE_DEFAULT = 0.50
RISK_ACTION_REBALANCE_DEFAULT = 0.30
RISK_ACTION_TOLERANCE_DEFAULT = 0.70

# Environment constants
ENV_MARKET_STRESS_DEFAULT = 0.5
ENV_OVERALL_RISK_DEFAULT = 0.5
ENV_MARKET_RISK_DEFAULT = 0.5
ENV_POSITION_EXPOSURE_THRESHOLD = 0.001
ENV_EXPLORATION_BONUS_MULTIPLIER = 0.5      # Multiplier for exploration bonus

def normalize_price(price_raw: float) -> float:
    """
    UNIFIED PRICE NORMALIZATION (SINGLE SOURCE OF TRUTH)

    Converts raw price to [-1, 1] range using z-score normalization with clipping.
    This function is used everywhere in the codebase to ensure consistency.

    Uses z-score normalization with clipping:
    - Converts raw price to standard deviations from mean
    - Clips extreme values to prevent outliers
    - Divides by PRICE_CLIP_SIGMA to get [-1,1] range

    Args:
        price_raw: Raw price in DKK/MWh

    Returns:
        Normalized price in [-1.0, 1.0] range
    """
    import numpy as np
    z_score = (price_raw - PRICE_MEAN) / PRICE_STD
    z_clipped = np.clip(z_score, -PRICE_CLIP_SIGMA, PRICE_CLIP_SIGMA)
    return float(z_clipped / PRICE_CLIP_SIGMA)


class EnhancedConfig:
    """Enhanced configuration class with optimization support and centralized hardcoded values"""

    # Configuration version for validation
    CONFIG_VERSION = "2.2.0"  # Updated for unified normalization

    def __init__(self, optimized_params: Optional[Dict[str, Any]] = None):
        # (Removed) Fair-comparison flag: fairness is now the default behavior.
        # =============================================================================
        # FUND STRUCTURE AND ECONOMICS
        # =============================================================================

        # Currency conversion (DKK to USD) - SINGLE SOURCE OF TRUTH
        self.dkk_to_usd_rate = 0.145  # 1 USD = ~6.9 DKK (2024 rate)
        self.currency_conversion = 1.0  # Keep at 1.0 - all calculations in DKK, convert only for final reporting

        # Fund size and allocation
        # UPDATED: New fund size $800M with updated allocations
        self.init_budget_usd = 8e8  # $800M fund size in USD
        self.init_budget = self.init_budget_usd / self.dkk_to_usd_rate  # Convert to DKK (~4.83B DKK)

        # Calculate actual allocations based on new fund structure
        # Physical CAPEX: $704M (88%), Trading capital: $96M (12%)
        self.physical_allocation = 0.88  # 88% physical assets ($704M of $800M)
        self.financial_allocation = 0.12  # 12% financial instruments ($96M of $800M)

        # Asset ownership fractions (of total installed capacity) - REALISTIC with market-rate CAPEX
        self.wind_ownership_fraction = 0.18   # 18% of 1,500MW wind farm (270MW owned)
        self.solar_ownership_fraction = 0.10  # 10% of 1,000MW solar farm (100MW owned)
        self.hydro_ownership_fraction = 0.04  # 4% of 1,000MW hydro plant (40MW owned)

        # Total installed capacities (MW)
        self.total_wind_capacity_mw = 1500.0  # Total wind farm capacity
        self.total_solar_capacity_mw = 1000.0  # Total solar farm capacity
        self.total_hydro_capacity_mw = 1000.0  # Total hydro plant capacity

        # Derived owned capacities (calculated from fractions)
        self.owned_wind_capacity_mw = self.wind_ownership_fraction * self.total_wind_capacity_mw  # 270 MW
        self.owned_solar_capacity_mw = self.solar_ownership_fraction * self.total_solar_capacity_mw  # 100 MW
        self.owned_hydro_capacity_mw = self.hydro_ownership_fraction * self.total_hydro_capacity_mw  # 40 MW
        self.owned_battery_capacity_mwh = 10.0  # 10 MWh direct ownership

        # CAPEX values ($/MW or $/MWh) - REALISTIC MARKET RATES
        self.wind_capex_per_mw = 2000000.0   # $2.0M/MW (market rate)
        self.solar_capex_per_mw = 1000000.0  # $1.0M/MW (market rate)
        self.hydro_capex_per_mw = 1500000.0  # $1.5M/MW (market rate)
        self.battery_capex_per_mwh = 400000.0  # $400k/MWh (market rate)

        # Operating costs and fees - FIXED: Convert USD to DKK
        self.operating_cost_rate = 0.025  # 3% of revenue
        self.maintenance_cost_mwh = 3.5 / self.dkk_to_usd_rate  # $3.5/MWh â†’ ~24.1 DKK/MWh
        self.insurance_rate = 0.004  # 0.4% of asset value annually
        self.management_fee_rate = 0.01  # 1% of fund value annually (realistic for infrastructure)
        self.property_tax_rate = 0.005  # 0.5% of asset value annually
        self.debt_service_rate = 0.015  # 1.5% of asset value annually (realistic debt service)
        self.distribution_rate = 0.10  # 10% of excess cash distributed
        self.eval_distribution_rate = None  # Eval-only cash-sweeper override; None preserves training/live behavior.
        self.target_cash_ratio = 0.15  # FIXED: Increased to 15% to give agents more working capital
        self.min_distribution_threshold_ratio = 0.01  # FIXED: Increased to 1% to reduce distribution frequency
        self.administration_fee_rate = 0.0001  # 0.01% of fund value annually (basic admin)

        # Grid and transmission fees - FIXED: Convert USD to DKK
        self.grid_connection_fee_mwh = 0.5 / self.dkk_to_usd_rate  # $0.5/MWh â†’ ~3.5 DKK/MWh
        self.transmission_fee_mwh = 1.2 / self.dkk_to_usd_rate    # $1.2/MWh â†’ ~8.3 DKK/MWh

        # Battery costs and parameters - FIXED: Convert USD to DKK
        self.battery_degradation_cost_mwh = 1.0 / self.dkk_to_usd_rate  # $1/MWh â†’ ~6.9 DKK/MWh
        self.batt_soc_min = 0.1  # 10% minimum state of charge
        self.batt_soc_max = 0.9  # 90% maximum state of charge
        self.batt_eta_charge = 0.90     # charge efficiency (PyPSA: 90%)
        self.batt_eta_discharge = 0.90  # discharge efficiency (PyPSA: 90%)
        self.batt_power_c_rate = 0.5    # max power: 5MW for 10MWh = 0.5 C-rate (PyPSA specifications)
        self.battery_initial_soc = 0.50  # Literature-standard neutral ESS benchmark

        # Trading costs - FIXED: Convert USD to DKK
        self.transaction_fixed_cost = 25.0 / self.dkk_to_usd_rate  # $25/trade â†’ ~172 DKK/trade
        self.transaction_cost_bps = 0.5  # 0.5 basis points (institutional rates)
        self.friction_cost_multiplier = 1.0  # Eval friction sweep multiplier; 1.0 preserves existing costs.
        # Explicit market-fee contract. The default preserves historical runs;
        # paper wrappers select the sourced MWh-scaled Nord Pool proxy.
        self.market_fee_model = "legacy_notional_fixed"
        self.transaction_fee_dkk_per_mwh = 0.0
        self.annual_market_access_fee_dkk = 0.0
        self.market_access_fee_allocation_fraction = 1.0
        self.market_fee_source_id = "legacy_unsourced_fixed_fee"
        self.half_spread_bp = 0.0  # Eval friction sweep half-spread in basis points; 0.0 preserves v1 behavior.
        self.impact_coef_bp = 0.0  # Eval-only size-dependent market impact; 0.0 preserves existing behavior.
        self.impact_exponent = 0.5  # Eval-only square-root temporary impact exponent.
        self.impact_ref_notional = "sleeve"  # Eval-only impact Q_ref; "sleeve" uses current trading cash.
        self.impact_volume_data_path = ""  # Eval-only liquidity CSV for impact_ref_notional="volume".
        self.impact_volume_timestamp_column = "timestamp"  # market-impact task
        self.impact_volume_column = ""  # market-impact task; blank autodetects a single volume column.
        self.impact_volume_unit = "mwh"  # market-impact task; {"mwh", "dkk"}.
        self.impact_volume_max_staleness_minutes = 90.0  # market-impact task
        self.impact_volume_price_floor_dkk_per_mwh = 50.0  # market-impact task
        # Liquidity and collateral stress controls. Defaults are off for
        # backward compatibility; paper wrappers enable the stricter protocol.
        self.liquidity_participation_cap_fraction = 0.0
        self.liquidity_volume_source = "load"  # {"load", "generation", "max_load_generation", "impact_volume"}
        self.liquidity_volume_multiplier = 1.0
        self.liquidity_min_volume_mwh = 1.0
        self.liquidity_tail_impact_threshold_dkk_per_mwh = 5000.0
        self.liquidity_tail_impact_multiplier = 1.0
        self.liquidity_tail_impact_power = 1.0
        self.liquidity_tail_impact_max_multiplier = 10.0
        self.enable_collateral_cash_drag = False
        self.collateral_notional_margin_fraction = 0.02
        self.collateral_stress_loss_fraction = 0.10
        self.collateral_stress_price_dkk_per_mwh = 25000.0
        self.collateral_funding_rate_annual = 0.05
        self.collateral_tradeable_haircut = 1.0

        # Battery dispatch economic thresholds
        self.battery_hurdle_min_dkk = 5.0  # Minimum hurdle rate in DKK/MWh
        self.battery_price_sensitivity = 0.02  # Price-based hurdle factor
        self.battery_rt_loss_weight = 0.3  # Weight of round-trip loss in hurdle calculation
        self.battery_use_heuristic_dispatch = False  # Tier1 battery should be RL-driven by default
        self.battery_action_mode = "discrete"  # {"discrete", "target_soc", "power"} explicit power-level control
        self.battery_discrete_action_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.battery_action_threshold = 0.35  # Used only by continuous battery action modes
        self.battery_arbitrage_reward_weight = 1.00  # Direct realized arbitrage reward
        self.battery_arbitrage_reward_scale_dkk = 250.0
        self.battery_arbitrage_reward_clip = 5.0
        self.battery_timing_reward_weight = 0.20   # Bonus/penalty for price-aligned charge/discharge timing
        self.battery_floor_penalty_weight = 0.10   # Penalty per step for sitting at SOC floor
        self.battery_infeasible_penalty_weight = 0.10  # Penalty for infeasible discharge/charge attempts

        # Battery operational parameters - FIXED: Convert USD to DKK
        self.battery_opex_rate = 0.0002  # Battery operational cost rate per MWh capacity
        self.performance_fee_rate = 0.20  # 20% of profits above benchmark (institutional standard)
        self.trading_cost_rate = 0.001   # 0.1% of transaction value (trading costs)

        # Price and revenue parameters
        self.minimum_price_floor = 50.0  # Minimum price floor in DKK/MWh
        self.maximum_price_cap = 2000.0  # Maximum price cap in DKK/MWh for filtering
        self.minimum_price_filter = 10.0  # Minimum price for data filtering in DKK/MWh
        self.time_step_hours = 10.0 / 60.0  # 10-minute timesteps in hours

        # Financial parameters - INFRASTRUCTURE FUND FOCUS
        self.max_leverage = 1.05  # INFRASTRUCTURE: Minimal leverage (5% max) typical for conservative infrastructure funds
        self.electricity_markup = 1.0  # Fund sells at market price

        # ENHANCED: Operational revenue reallocation policy - DISABLED FOR CAPITAL PRESERVATION
        self.allow_operational_reallocation = False  # DISABLED: Prevent unrealistic cash accumulation
        self.max_reallocation_rate = 0.0  # DISABLED: No reallocation allowed
        self.trading_capital_emergency_threshold = 0.50  # Not used when disabled
        self.max_total_reallocation = 0.0  # DISABLED: No lifetime reallocation

        # Mark-to-market payoff model. "percent_capped" preserves the original
        # capped percentage-return simulator. "notional_price_diff" marks DKK
        # notional as an implicit MWh contract using a fixed reference price,
        # avoiding unstable percentage returns around zero/negative power prices.
        # "horizon_settlement" is the stricter paper-audit mode: it realizes a
        # fixed-horizon contract against a causal entry benchmark instead of
        # paying adjacent spot-price ticks. "horizon_settlement_continuous" is
        # retained only as a compatibility label. Without an independently
        # observed intrahour quote, the strict protocol carries open contracts
        # at zero unrealized PnL and controls them with collateral stress.
        self.mtm_return_model = "percent_capped"
        self.mtm_reference_price_dkk_per_mwh = 500.0
        self.mtm_settlement_horizon_steps = 6
        self.mtm_entry_price_mode = "current_price"
        # Settlement-price process used by horizon_settlement* contracts:
        # - energy_index: settle against the same energy price index as the entry
        #   series. This preserves the pre-basis baseline behavior.
        # - cross_zone_basis: settle against the entry energy index plus an
        #   empirical tracking-error component from another price area. This
        #   models energy-index-linked instruments with imperfect replication.
        # - external_series: settle against an explicit realized settlement
        #   series such as imbalance, intraday, or balancing price. This is the
        #   preferred paper protocol when real settlement prices are available.
        self.mtm_settlement_price_mode = "energy_index"
        self.mtm_basis_price_data_path = ""
        self.mtm_basis_price_column = "price"
        self.mtm_basis_timestamp_column = "timestamp"
        self.mtm_basis_scale = 0.50
        self.mtm_basis_centering_mode = "rolling_median"
        self.mtm_basis_centering_window_steps = 4320
        self.mtm_external_settlement_price_data_path = ""
        self.mtm_external_settlement_price_column = "settlement_price"
        self.mtm_external_settlement_timestamp_column = "timestamp"
        self.mtm_external_settlement_min_price_dkk_per_mwh = -111750.0
        self.mtm_external_settlement_max_price_dkk_per_mwh = 111750.0
        # Limited-liability trading sleeve. If the sleeve equity falls below a
        # maintenance threshold, open financial contracts are liquidated and the
        # investor sleeve is disabled for the rest of the episode.
        self.enable_trading_sleeve_margin = True
        self.trading_sleeve_maintenance_margin_fraction = 0.05
        self.trading_sleeve_margin_liquidate_positions = True
        self.trading_sleeve_margin_disable_trading = True
        # Horizon-settlement payoff denominator:
        # - reference_price preserves the historical protocol: price_diff / reference.
        # - entry_price_floor treats DKK notional as an energy contract whose MWh
        #   quantity is scaled by the observed entry price, with the reference as
        #   a denominator floor for low/negative prices.
        # - mwh_volume uses an explicit MWh contract: position_MWh =
        #   DKK_notional / reference_price, and PnL = MWh * (settlement - entry).
        self.mtm_horizon_payoff_denominator_mode = "reference_price"
        self.mtm_apply_price_return_cap = True

        # Mark-to-market volatility controls for the original percent model.
        self.mtm_price_return_cap_min = -0.001  # REALISTIC: -0.1% maximum loss per 10-min step (â‰ˆ-14% daily)
        self.mtm_price_return_cap_max = 0.001   # REALISTIC: +0.1% maximum gain per 10-min step (â‰ˆ+14% daily)

        # FAIR SPARSE REWARD FIXES: Multi-step returns and MTM inclusion
        # These apply to ALL tiers equally, making rewards less sparse without unfair advantages
        # REDUCED: Smaller window (2 instead of 3) to preserve reward signal strength
        # REDUCED: Lower MTM weight (0.2 instead of 0.3) to reduce noise
        self.pnl_reward_multi_step_window = 2  # Use rolling average of 2 timesteps (reduced from 3 to preserve signal)
        self.pnl_reward_include_mtm = True  # Include unrealized PnL delta in rewards (fair for all tiers)
        self.pnl_reward_mtm_weight = 0.2  # Weight for MTM component (20% MTM, 80% realized return) - reduced from 0.3 to reduce noise

        # MTM threshold for position updates
        self.mtm_update_threshold = 1e-9  # Threshold for applying MTM updates to positions

        # NAV bounds for renewable energy fund - INFRASTRUCTURE REALISTIC
        # The environment intentionally uses unconstrained NAV to allow realistic fund dynamics
        # Only prevents negative NAV; no artificial min/max bounds applied
        self.nav_min_ratio = 0.90  # INFRASTRUCTURE: Minimum 90% of initial (conservative downside protection) [UNUSED]
        self.nav_max_ratio = 1.10  # INFRASTRUCTURE: Maximum 110% of initial (5% return cap - very realistic for infrastructure) [UNUSED]



        # =============================================================================
        # ENVIRONMENT AND SIMULATION PARAMETERS
        # =============================================================================

        # Meta controller ranges
        #
        # Goal: high notional exposure WITHOUT forcing the investor policy to saturate at +/-1.
        # Keep capital allocation reliably high, but pin the live trade cadence to the
        # canonical short horizon so Tier1 trades on 1h decisions by default.
        self.meta_freq_min = 6   # 1h @ 10-min steps (short horizon)
        self.meta_freq_max = 6   # fixed short-horizon cadence
        self.meta_cap_min = 0.10  # allow meaningful de-risking when sleeve edge is weak
        self.meta_cap_max = 0.80  # cap capital so a single factor sleeve cannot dominate the fund
        self.meta_rule_update_only_on_investor_step = True
        self.meta_rule_signal_ema_alpha = 0.10
        self.meta_rule_target_smoothing_alpha = 0.20
        self.meta_rule_cap_deadband = 0.03
        self.sat_eps = 1e-3

        # Investment and operational parameters - FORECAST OPTIMIZATION
        # Startup/default investor cadence before meta-control takes over in live runs.
        self.investment_freq = 6   # 1 hour @ 10-min steps
        self.min_investment_freq = 4   # 40 min minimum @ 10-min steps
        self.max_investment_freq = 12  # 2 hour maximum @ 10-min steps
        # Strategic controller cadences — must be multiples of investment_freq for
        # clean alignment with investor decision points.
        self.risk_action_freq = 12   # 2 hours: risk sets cap before every 2 investor decisions
        self.meta_action_freq = 24   # 4 hours: meta adjusts allocation once per half-session block
        self.capital_allocation_fraction = 0.60  # neutral starting point; meta-controller clamps within [meta_cap_min, meta_cap_max]
        # Investor notional sizing base:
        # - initial_trading_sleeve: fixed 12% trading sleeve, prevents MTM profits from increasing allowed notional
        # - live_trading_cash: legacy behavior, sizes from current trading cash/NAV
        # - initial_fund_nav: fixed total initial fund NAV
        self.investor_notional_sizing_base = "initial_trading_sleeve"
        self.risk_multiplier = 1.0  # Raised from 0.5 for full-size baseline trades
        # Clean Tier-1 sizing contract: risk controller sets an absolute
        # exposure cap inside the trading sleeve, not a leverage multiplier.
        self.risk_exposure_cap_min = 0.25
        self.risk_exposure_cap_max = 1.00
        # Backward-compatible contract:
        # - values in [0, 1] are treated as a fraction of max position notional
        # - values > 1 are treated as a raw DKK threshold
        self.no_trade_threshold = 0.01
        # Under a hard liquidity cap, use the smaller of the portfolio position
        # limit and the executable market capacity. Otherwise a portfolio-scale
        # threshold can mechanically suppress every feasible market order.
        self.no_trade_threshold_reference = "executable_capacity"

        # Battery operational parameters (batt_eta_charge * batt_eta_discharge = round-trip efficiency)

        # =============================================================================
        # MEMORY AND PERFORMANCE PARAMETERS
        # =============================================================================

        # Paper-grade correctness: never silently pad/truncate observation vectors.
        # If an observation dimension mismatch occurs, abort with a clear error so the run is reproducible.
        self.strict_obs_validation = True

        # Memory limits (MB) - INCREASED FOR PERFORMANCE
        self.max_memory_mb = 6000.0  # Environment memory limit (INCREASED)
        self.metacontroller_memory_mb = 12000  # Meta controller memory limit (INCREASED)
        self.wrapper_memory_mb = 2000  # Wrapper memory limit (INCREASED)

        self.lru_cache_size = 3000  # Increased for better performance
        self.lru_memory_limit_mb = 100.0  # Increased for better performance

        # =============================================================================
        # RISK MANAGEMENT PARAMETERS
        # =============================================================================

        # Risk calculation lookback
        self.risk_lookback_window = 96  # OPTIMIZED: Reduced from 144 to 96 for more responsive risk management

        # OPERATIONAL EXCELLENCE: Capital preservation risk management
        self.max_drawdown_threshold = 0.10  # CAPITAL PRESERVATION: 10% maximum drawdown
        self.volatility_lookback = 60  # CAPITAL PRESERVATION: 60 days for very stable calculations
        self.max_position_size = 0.35  # larger per-asset sizing (primary lever for notional exposure)
        self.volatility_scaling = 0.8   # CAPITAL PRESERVATION: High scaling for minimal risk
        self.target_sharpe_ratio = 2.0  # CAPITAL PRESERVATION: High target for excellent risk-adjusted returns
        self.risk_free_rate = 0.02      # 2% risk-free rate (Danish government bonds)
        self.sleeve_sharpe_mode = "daily_hac_7"  # Paper-facing sleeve Sharpe: daily Newey-West/HAC adjusted
        self.max_asset_correlation = 0.3 # CAPITAL PRESERVATION: Low correlation for maximum diversification

        # ONE-FACTOR TRADING SLEEVE:
        # The investor trades one aggregate energy-price factor. The three sleeve
        # entries below are bookkeeping weights used to split that single factor
        # exposure for risk accounting and MTM attribution.
        self.risk_budget_allocation = {
            'wind': 0.40,      # 40% of risk budget to wind (highest volatility)
            'solar': 0.35,     # 35% of risk budget to solar (medium volatility)
            'hydro': 0.25,     # 25% of risk budget to hydro (lowest volatility)
        }

        # OPERATIONAL EXCELLENCE: Renewable energy fund focus parameters
        self.operational_revenue_target = 1200.0   # RENEWABLE ENERGY: Realistic 1200 DKK per step target (~6% annual return)
        self.hedging_effectiveness_target = 0.95  # CAPITAL PRESERVATION: 95% hedge effectiveness target
        self.max_portfolio_volatility = 0.06      # CAPITAL PRESERVATION: Maximum 6% annual volatility
        self.operational_revenue_weight = 0.95    # CAPITAL PRESERVATION: 95% focus on operations vs 5% trading

        # Risk weights (should sum to ~1.0)
        self.risk_weights = {
            'market': 0.25,
            'operational': 0.20,
            'portfolio': 0.25,
            'liquidity': 0.15,
            'regulatory': 0.15
        }

        # Risk thresholds - OPTIMIZED for better risk-return balance
        self.risk_thresholds = {
            'market_stress_high': 0.80,      # OPTIMIZED: Reduced from 0.85 for earlier risk response
            'market_stress_medium': 0.60,    # OPTIMIZED: Reduced from 0.65 for better sensitivity
            'volatility_high': 0.75,         # OPTIMIZED: Reduced from 0.80 for better volatility control
            'volatility_medium': 0.45,       # OPTIMIZED: Reduced from 0.50 for earlier intervention
            'portfolio_concentration_high': 0.70,  # OPTIMIZED: Reduced from 0.75 for better diversification
            'liquidity_stress_high': 0.75,   # OPTIMIZED: Reduced from 0.80 for better liquidity management
        }

        # Risk calculation component weights - PHASE 5.10 FIX: Parameterized risk model
        self.market_risk_volatility_weight = 0.6
        self.market_risk_momentum_weight = 0.4
        self.market_risk_momentum_factor = 0.5
        self.operational_risk_volatility_weight = 0.7
        self.operational_risk_intermittency_weight = 0.3
        self.portfolio_risk_concentration_weight = 0.6
        self.portfolio_risk_capital_weight = 0.4
        self.portfolio_risk_exposure_weight = 0.25  # Trading exposure (financial positions) in portfolio risk
        self.liquidity_risk_buffer_weight = 0.6
        self.liquidity_risk_cashflow_weight = 0.4
        # NEW: Infrastructure fund performance targets
        self.annual_return_target = 0.06   # INFRASTRUCTURE: 6% annual return target (conservative infrastructure)
        self.max_annual_volatility = 0.08  # INFRASTRUCTURE: 8% maximum annual volatility (very stable)
        self.min_sharpe_ratio = 1.5        # INFRASTRUCTURE: High risk-adjusted returns for conservative funds

        # Physical asset depreciation parameters
        self.annual_depreciation_rate = 0.02  # 2% annual straight-line depreciation
        self.max_depreciation_ratio = 0.75  # Maximum 75% depreciation over asset life

        # =============================================================================
        # FORECASTING AND PRICING PARAMETERS
        # =============================================================================

        # Price scaling and limits
        self.price_scale = 10.0  # Price normalization scale
        self.price_clip_min = -1000.0  # Minimum price for clipping
        self.price_clip_max = 1e9  # Maximum price for clipping

        # PRICE NORMALIZATION: Unified rule across all code paths
        # Actual implementation: price_n = clip(z_score, -3, 3) / 3 â†’ [-1, 1]
        # This ensures consistent feature scaling across environment and wrapper.
        # The actual pipeline uses: z_score clipped to Â±3Ïƒ, then divided by 3 to get [-1,1]
        self.price_z_score_clip = 3.0  # ACTUAL: Clip z-scores to Â±3Ïƒ before dividing by 3

        # FALLBACK STATS: For consistent two-step normalization when rolling stats unavailable
        self.price_fallback_mean = 250.0  # Typical DKK/MWh price for fallback z-score calculation
        self.price_fallback_std = 50.0    # Typical price volatility for fallback z-score calculation

        # CANONICAL HORIZONS: Short-only (immediate/medium/long removed)
        self.forecast_horizons = {
            "short": 6,          # 1 hour - 6 steps ahead
        }
        self.forecast_look_back = 24  # Canonical look-back for forecast training/loading
        self.forecast_targets = ["price"]

        # Raise MTM loss exit threshold to allow positions to develop before forced exits (from 3% -> 6%)
        self.mtm_loss_exit_threshold_pct = 0.06

        # CAPACITY-FACTOR TO MW CONVERSION SCALES: Configurable for different datasets
        # These values are used when converting capacity factors [0,1] to raw MW values
        # Default values derived from training scaler analysis - can be overridden
        self.mw_conversion_scales = {
            'wind': 1103,    # From training scaler mean: 1103.4 MW
            'solar': 100,    # From training scaler mean: 61.5 MW (min 100 for stability)
            'hydro': 534,    # From training scaler mean: 534.1 MW
            'load': 2999,    # From training scaler mean: 2999.8 MW
        }

        # Default forecast values (MW or DKK/MWh) - FIXED: Standardized on DKK
        self.default_forecasts = {
            "wind": 330.0,    # ~30% of capacity
            "solar": 20.0,    # ~20% of capacity
            "hydro": 267.0,   # ~50% of capacity
            "price": 345.0,   # DKK/MWh (consistent with environment)
            "load": 1800.0,   # ~60% of capacity
        }

        # =============================================================================
        # DEEP LEARNING PARAMETERS
        # =============================================================================

        # Portfolio optimization neural network
        self.portfolio_learning_rate = 0.002  # USER SPECIFIED: Higher learning rate for early profitability
        self.portfolio_num_assets = 5
        self.portfolio_dropout_rate_1 = 0.2
        self.portfolio_dropout_rate_2 = 0.3

        # Loss function weights
        self.portfolio_risk_loss_weight = 0.5
        self.portfolio_weight_sum_loss_weight = 10.0
        self.portfolio_negative_weight_penalty_weight = 5.0
        self.portfolio_concentration_penalty_weight = 0.1
        self.portfolio_transaction_cost_penalty = 0.001  # 0.1% penalty

        # =============================================================================
        # REWARD PARAMETERS
        # =============================================================================

        # Reward calculation parameters - CAPITAL PRESERVATION FOCUS
        self.base_reward_scale = 1.0  # CAPITAL PRESERVATION: Minimal reward scale for stability
        self.profit_reward_weight = 1.0  # CAPITAL PRESERVATION: Equal weight to profit (not dominant)
        self.risk_penalty_weight = 5.0  # CAPITAL PRESERVATION: Very strong risk penalty for safety

        # Reward normalization and clipping
        self.reward_clip_min = -10.0  # Minimum reward value
        self.reward_clip_max = 10.0   # Maximum reward value
        self.reward_normalization_factor = 1e6  # Factor to normalize large monetary values

        # Agent-specific reward weights
        self.investor_reward_weight = 1.0      # Investor agent reward scaling
        self.battery_reward_weight = 1.0       # Battery operator reward scaling
        self.risk_controller_reward_weight = 1.0  # Risk controller reward scaling
        self.meta_controller_reward_weight = 1.0   # Meta controller reward scaling

        # Forecast-cache directory consumed by the cache-only investor prior.
        self.forecast_cache_dir = "forecast_cache"

        # ===== Forecast-cache investor exposure prior =======================
        # Opt-in execution layer. When True, the ANN forecast cache is converted
        # into an evidence-calibrated, risk-aware target exposure and applied by the
        # FoCAL action-prior layer. Investor observations stay fixed at 12D.
        self.enable_forecast_utilization = False
        self.forecast_prior_window = 500
        self.forecast_prior_min_samples = 50
        self.forecast_prior_hit_lcb_z = 1.64
        self.forecast_prior_residual_quantile = 0.70
        self.forecast_prior_default_residual = 500.0
        self.forecast_prior_edge_gain = 3.0
        # Net-edge filter: the cached forecast must clear a fraction of its
        # own causal residual error before it can size exposure.
        self.forecast_prior_error_hurdle = 0.50
        self.forecast_prior_skill_power = 1.0
        # The paper protocol uses an evidence-gated action-prior layer. The
        # forecast prior is settlement-aligned and the learned investor policy
        # earns residual authority only after its counterfactual residual target
        # improves on the causal prior.
        self.forecast_prior_directional_floor = 0.50
        # Current forecast confidence is load-bearing: historical reliability
        # alone should not size exposure when the live directional probability
        # is near 50/50.
        self.forecast_prior_use_direction_confidence = True
        self.forecast_prior_direction_confidence_power = 1.0
        # Deterministic conviction gate (anchor_gated arm): abstain (force zero
        # exposure) when the ANN regression-sign and classifier-margin-sign
        # forecast heads DISAGREE. Parameter-free participation filter; improves
        # risk-adjusted return by declining low-conviction trades. Default off
        # reproduces the ungated anchor exactly.
        self.forecast_prior_agreement_gate = False
        # Align rolling hit/residual calibration with the investor decision
        # cadence. This avoids over-counting repeated forward-filled
        # ten-minute rows for an hourly day-ahead price signal.
        self.forecast_prior_calibrate_on_decision_grid = True
        # Prototype FoCAL v3: the cached ANN forecast supplies the direction,
        # while a causal conditional payoff distribution determines the target
        # exposure under sleeve tail budgets. The old residual mode is retained
        # only for legacy comparisons and is not the active Prototype protocol.
        self.forecast_prior_control_mode = "distributional"
        self.forecast_prior_max_abs_exposure = 0.60
        self.forecast_prior_residual_scale = 0.10
        self.forecast_prior_residual_to_prior_ratio = 0.35
        self.forecast_prior_inactive_residual_scale = 0.0
        self.forecast_prior_blend = 0.85
        self.forecast_prior_policy_relative_gate = True
        self.forecast_prior_policy_relative_min_samples = 50
        self.forecast_prior_policy_relative_hit_margin = 0.02
        self.forecast_prior_policy_relative_advantage_scale = 0.02
        self.forecast_prior_residual_evidence_gate = True
        self.forecast_prior_residual_evidence_floor = 0.0
        # Optional market-beta/carry prior. It is disabled in the main FoCAL
        # protocol because the paper contribution is the forecast-prior action
        # layer, not a generic settlement-carry rule. Keep it available only
        # for explicit ablations.
        self.forecast_prior_beta_enabled = False
        self.forecast_prior_beta_min_samples = 50
        self.forecast_prior_beta_lcb_scale = 0.005
        self.forecast_prior_beta_max_abs_exposure = 0.60
        self.forecast_prior_beta_weight = 1.0
        self.forecast_prior_beta_return_clip = 0.25
        self.forecast_prior_beta_allow_short = True
        # Survival cap for imbalance-linked payoff tails. The cap uses only
        # causal realized settlement payoffs plus a stress floor, then limits
        # the prior against that stress return.
        self.forecast_prior_tail_cap_enabled = True
        self.forecast_prior_tail_min_samples = 50
        self.forecast_prior_tail_quantile = 0.99
        self.forecast_prior_tail_loss_budget_fraction = 0.05
        self.forecast_prior_tail_stress_return_floor = 25000.0
        self.forecast_prior_tail_default_return = 25000.0
        self.forecast_prior_tail_return_clip = 111750.0
        # FoCAL v3 distributional tail-budget settings. Conditional risk is
        # estimated in forecast-feature buckets, while a wider pooled disaster
        # budget prevents the prior from becoming a static long-tail harvester.
        self.forecast_prior_distributional_window = 2000
        self.forecast_prior_distributional_min_samples = 50
        self.forecast_prior_distributional_tail_quantile = 0.99
        self.forecast_prior_distributional_disaster_quantile = 0.999
        self.forecast_prior_distributional_loss_budget_fraction = 0.05
        self.forecast_prior_distributional_disaster_loss_budget_fraction = 0.20
        self.forecast_prior_distributional_conditional_tail_floor = 2000.0
        self.forecast_prior_distributional_disaster_tail_floor = 25000.0
        self.forecast_prior_distributional_default_tail_return = 25000.0
        self.forecast_prior_distributional_return_clip = 111750.0
        self.forecast_prior_distributional_edge_scale = 250.0
        self.forecast_prior_distributional_edge_hurdle = 0.0
        self.forecast_prior_distributional_confidence_power = 1.0
        self.forecast_prior_distributional_bucket_mode = "conditional"
        self.forecast_prior_distributional_use_confidence_weight = True
        self.forecast_prior_distributional_fixed_cap_abs = 0.10
        self.forecast_prior_distributional_cold_start_exposure = 0.0
        # Active learned integration. MAPPO selects a normalized
        # coordinate inside the FoCAL-aligned executable MWh interval. The
        # investor receives a strictly matured matched settlement-utility signal
        # versus the contemporaneous deterministic FoCAL anchor.
        self.forecast_prior_observation_only_mappo = False
        self.forecast_prior_feasible_action_mappo = False
        # Counterfactual Forecast-Mirror MAPPO. The MAPPO actor
        # remains active over the full signed feasible interval. A causal
        # forecast expert mirrors its Beta density through learned trust that
        # is updated only after the corresponding settlement matures.
        self.forecast_prior_mirror_mappo = False
        self.forecast_mirror_prior_concentration = 8.0
        self.forecast_mirror_trust_max = 1.0
        self.forecast_mirror_trust_hidden_dim = 32
        self.forecast_mirror_trust_initial = 0.10
        self.forecast_mirror_trust_regularization = 0.01
        self.forecast_mirror_trust_min_samples = 32
        self.forecast_mirror_trust_batch_size = 256
        self.forecast_mirror_trust_gradient_clip = 5.0
        self.forecast_mirror_counterfactual_learning = True
        self.forecast_mirror_mask_forecast_observation = False
        self.forecast_mirror_counterfactual_grid_size = 41
        self.forecast_mirror_counterfactual_payoff_scale = 250.0
        self.forecast_mirror_counterfactual_loss_aversion = 0.25
        # CFM corrected-v5 mechanism knobs (default off = legacy CFM behaviour).
        # (1) matched anchor-advantage reward: trains the base actor on genuine
        #     incremental utility over the deterministic forecast anchor, using
        #     the identical matured settlement, fees, funding and constraints.
        self.forecast_mirror_anchor_advantage_reward = False
        self.forecast_mirror_anchor_advantage_weight = 0.25
        # (2) dual-view observation: mask forecast-derived dims from the base
        #     actor so the forecast enters only through the trust-gated mirror,
        #     removing base-actor/forecast double counting. Trust net keeps 22D.
        self.forecast_mirror_mask_base_actor_forecast = False
        self.forecast_prior_feasible_action_condition_capacity_on_evidence = True
        self.forecast_prior_feasible_action_evidence_power = 1.0
        self.forecast_prior_feasible_action_window = 500
        self.forecast_prior_feasible_action_min_samples = 50
        self.forecast_prior_feasible_action_cvar_quantile = 0.90
        self.forecast_prior_feasible_action_shortfall_budget = 0.05
        self.forecast_prior_feasible_action_dual_lr = 0.02
        self.forecast_prior_feasible_action_dual_max = 5.0
        self.forecast_prior_feasible_action_reward_weight = 0.25
        self.forecast_prior_feasible_action_payoff_scale = 250.0
        self.forecast_prior_feasible_action_score_clip = 2.0
        self.forecast_prior_vol_half_life_steps = 288
        self.forecast_prior_vol_target = 0.15
        self.forecast_prior_horizon_steps = 6
        # Keep the executable default identical to the frozen runtime contract.
        self.forecast_prior_denom_floor = 50.0
        # The prior's calibration target must match the financial payoff. In
        # auto mode, horizon_settlement MTM uses forecasted settlement price
        # versus the same causal entry benchmark as the environment.
        self.forecast_prior_payoff_target_mode = "auto"
        self.forecast_prior_target_alignment_version = "same_delivery_causal_v3"
        # Enabled prior runs must not silently degrade into baseline because of
        # a missing or malformed ANN cache.
        self.forecast_prior_fail_fast = True

        # Per-episode debug/health CSV logging is expensive during multi-seed
        # training. Keep it disabled by default for paper runs; checkpoints and
        # evaluation JSON are still written by their normal code paths.
        self.enable_episode_csv_logs = False

        # =============================================================================
        # REINFORCEMENT LEARNING PARAMETERS
        # =============================================================================

        # Training defaults
        # update_every = PPO rollout buffer collection length (also sets self.n_steps in metacontroller).
        # rollout_cap  = hard ceiling on steps collected per rollout call in _collect_rollouts_enhanced.
        # With update_every=256, rollout_cap=256, batch_size=64: 256/64=4 minibatches per epoch × 10 epochs
        # = 40 gradient steps per rollout — matches IPPO best-practice for non-stationary MARL.
        self.update_every = 256
        self.rollout_cap  = 256
        self.lr = 1.5e-3
        self.ent_coef = 0.030  # promotes exploration; investor especially benefits from this
        self.verbose = 1
        self.seed = 42
        self.multithreading = False
        self.algo = "ippo"
        self.mappo_central_net_arch = [256, 128, 64]

        # Per-agent reward normalization (Welford online mean/std, applied before buffer insertion).
        # Decouples reward scale from hyperparameter tuning; each agent's reward stream is
        # normalized independently so different reward magnitudes don't bias value estimates.
        self.reward_normalization_enabled = True
        self.reward_normalization_clip    = 10.0   # final clip after normalization
        self.reward_normalization_warmup  = 100    # steps before normalization activates

        # PPO-specific parameters
        # batch_size=64: with rollout cap at 256 steps, each of n_epochs=10 passes
        # gets 4 minibatches (256/64=4). This matches IPPO literature — minibatch
        # diversity matters more than large batch size in non-stationary MARL.
        self.batch_size = 64
        self.gamma = 0.995  # long horizon for fund-level credit assignment
        self.gae_lambda = 0.98  # high lambda: low bias, acceptable variance with long rollouts
        self.clip_range = 0.15  # conservative clip: limits policy drift in non-stationary env
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.n_epochs = 10  # 10 passes × 4 minibatches = 40 gradient steps per rollout
        self.n_steps = 1024  # SB3 policy buffer size (actual rollout cap capped to 256 in metacontroller)
        self.meta_lr = 5e-5
        self.risk_lr = 3e-5
        self.meta_n_epochs = 4
        self.risk_n_epochs = 3
        self.meta_clip_range = 0.10
        self.risk_clip_range = 0.08
        self.meta_max_grad_norm = 0.30
        self.risk_max_grad_norm = 0.25
        self.meta_ent_coef = 0.010
        self.risk_ent_coef = 0.008
        self.meta_target_kl = 0.020
        self.risk_target_kl = 0.015
        self.ppo_nan_max_rollbacks_per_learn = 2

        # Optional SB3 PPO exploration controls. Keep the default path vanilla:
        # standard Gaussian policy, no gSDE, default policy log_std_init.
        self.ppo_use_sde = False
        self.ppo_sde_sample_freq = 1
        self.ppo_log_std_init = None
        self.dqn_buffer_size = 50000
        self.dqn_learning_starts = 1000
        self.dqn_batch_size = 128
        self.dqn_train_freq = 4
        self.dqn_gradient_steps = 1
        self.dqn_target_update_interval = 500
        self.dqn_exploration_fraction = 0.40       # fraction of steps over which epsilon decays
        self.dqn_exploration_initial_eps = 1.0
        self.dqn_exploration_final_eps = 0.10      # keep 10% randomness throughout (was 0.05)
        # Investor-only bounded-support policy:
        # use a Beta policy for bounded continuous exposure instead of a Gaussian.
        self.investor_use_beta_policy = True
        self.investor_beta_epsilon = 1e-6
        # Debug: force-disable gSDE regardless of CLI flags
        self.force_disable_sde = False

        # Minimal, thesis-defensible regularizer to discourage always-at-bounds allocations.
        # Interpretable as a soft leverage/turnover proxy; applied with warmup in the env reward.
        self.investor_action_l2_penalty = 0.0
        self.investor_action_penalty_warmup_steps = 200

        # Investor observation: price momentum scale (return / scale maps to [-1,1])
        # 0.08 = 8% return over investment horizon maps to ±1
        self.investor_price_momentum_scale = 0.08
        self.investor_recent_return_lookback = 12
        self.investor_realized_vol_scale = 0.05

        # Direct investor trading-PnL reward.
        # Tier-1 investor should pursue trading profits, while still remaining
        # coordinated with the broader MARL system. Use a modest direct MTM term
        # to keep the sleeve profit-aware without making it purely myopic.
        # REMOVED: investor_mtm_delta (redundant with investor_trading_profit_delta)
        self.investor_mtm_reward_weight = 0.0
        self.investor_mtm_reward_scale = 5000.0
        self.investor_mtm_reward_clip = 1.0

        # Tier-1 clean reward contract:
        # pay the investor once for sleeve performance via normalized return plus
        # local path quality, then subtract explicit path-risk and policy-health
        # penalties. Keep the direct profit term off by default to avoid
        # double-paying the same trade.
        self.investor_clean_reward_contract = True
        self.investor_trading_profit_weight = 0.0
        self.investor_trading_profit_scale = 3000.0
        self.investor_trading_profit_clip = 1.5

        # Hedging alignment: OFF. Investor focuses on profitable trading, not hedging ops.
        self.investor_hedging_reward_weight = 0.0

        # Shared base-reward mix: OFF for investor/battery (agent-specific rewards only).
        self.investor_base_reward_weight = 0.0
        self.battery_base_reward_weight = 0.0
        self.risk_base_reward_weight = 0.0
        self.meta_base_reward_weight = 0.0
        # Default OFF: do not add a rolling average of recent trading gains on top
        # of NAV growth inside the shared reward. That term made profitable short
        # books sticky across episodes and was a core driver of Tier-1 one-sign drift.
        self.shared_trading_score_weight = 0.0

        # Investor local trading-sleeve objective.
        # Keep Tier-1 simple: reward realized trading profit on allocated
        # capital, add only a small quality stabilizer, and keep the downside
        # contract limited to cost, mild drawdown, and one PPO-health brake.
        self.investor_trading_history_lookback = 48
        self.investor_trading_return_weight = 0.22
        self.investor_trading_return_scale = 0.003
        self.investor_trading_return_clip = 1.5
        # Clean investor contract: health belongs in PPO optimization, not in a
        # second path-quality bonus inside the environment reward.
        self.investor_trading_quality_weight = 0.0
        self.investor_trading_quality_clip = 2.0
        self.investor_trading_vol_floor = 5e-4
        self.investor_trading_drawdown_weight = 0.06
        self.investor_trading_drawdown_scale = 0.05
        self.investor_trading_cost_weight = 0.05
        self.investor_trading_cost_scale = 0.002
        self.investor_tracker_return_clip = 0.50
        # Active inventory risk charge: a single, principled brake on carrying
        # large directional inventory, especially in volatile regimes.
        self.investor_active_risk_weight = 0.18
        self.investor_active_risk_free_band = 0.40
        self.investor_active_risk_vol_mult = 1.00

        # Meta-controller dense local objective.
        # Meta is a NAV-first allocator for the investor sleeve. It should respond
        # mainly to sleeve edge and risk, while the fund-level shared reward
        # remains the dominant objective. Battery-specific shaping is disabled.
        self.meta_local_investor_weight = 0.20
        self.meta_local_battery_weight = 0.00
        self.meta_local_risk_weight = 0.10
        self.meta_local_signal_clip = 2.0
        self.meta_capital_alignment_weight = 0.10
        self.risk_controller_rule_based = False
        self.meta_controller_rule_based = False

        # Global training step counter (persists across episode environments in episode training).
        # Needed because episode training creates a NEW env each episode, so env-local counters reset.
        self.training_global_step = 0

        # Investor action is exposure-only (single scalar). Allocation is fixed internally.

        # Action squashing (anti-clip artifact). If True, continuous actions that exceed [-1,1]
        # are smoothly squashed with tanh before clipping. If False, we rely on clipping alone.
        # Default OFF: with meta-controlled exposure, we generally prefer not to soften actions unless needed.
        # Turn ON for an ablation if you see hard-clip induced saturation again.
        self.enable_action_tanh_squash = True

        # Exposure-scalar remapping.
        # The investor controls a single signed exposure-adjustment scalar by default.
        # This is healthier than an absolute target controller because "hold current
        # risk" maps to a near-zero action instead of parking the policy mean on an
        # internal PPO wall. The mapped signal still passes through a signed-power
        # transform so interior actions remain meaningful.
        self.investor_exposure_action_mode = "delta"   # {"delta", "absolute"}
        self.investor_delta_exposure_scale = 0.20      # smaller step size keeps vanilla PPO from jumping to the live cap too quickly
        self.investor_exposure_power = 1.0
        # Tier-1 investor now controls direct signed trading exposure. No
        # structural hedge anchor is applied in the active trading path.
        self.investor_use_risk_budget_weights = True

        # Investor penalties default OFF.
        # Tier-1 should stand on reward alignment and protocol stability, not on
        # penalty terms that artificially hold the policy away from collapse.
        self.investor_action_boundary_penalty = 0.0
        self.investor_action_boundary_threshold = 0.95

        # Exposure hinge penalty default OFF.
        self.investor_action_exposure_penalty = 0.0
        self.investor_action_exposure_threshold = 0.90  # allow high exposure; only penalize near-max

        # Episode-boundary collapse breaker (targets: exposure stuck at ~1.0 for long stretches).
        # This ramps up ONLY when exposure stays above a high threshold for consecutive steps.
        # It is designed to "unstick" PPO in ep1+ without affecting normal, dynamic regimes.
        self.investor_exposure_stuck_threshold = 0.98
        self.investor_exposure_stuck_steps = 500
        self.investor_exposure_stuck_penalty = 0.0
        # Investor policy-mean saturation monitor:
        # track when the deterministic policy mean sits too close to the action
        # boundary for long stretches. This is diagnostics-only metadata.
        self.investor_mean_clip_hit_window = 128
        self.investor_mean_clip_hit_warmup_steps = 2000
        self.investor_mean_clip_hit_threshold = 0.35
        self.investor_mean_clip_hit_margin = 0.96
        # Deterministic-policy anti-collapse:
        # diagnostics-only in the default Tier-1 baseline.
        self.investor_mean_collapse_window = 128
        self.investor_mean_collapse_warmup_steps = 2000
        self.investor_mean_collapse_abs_mean_threshold = 0.22
        self.investor_mean_collapse_sign_threshold = 0.78

        # =============================================================================
        # GLOBAL NORMALIZATION (ANTI EPISODE-BOUNDARY DISTRIBUTION SHIFT)
        # =============================================================================
        # Episode training currently normalizes price via rolling mean/std computed inside each 6-month episode.
        # That makes the observation distribution non-stationary across episodes (ep0 vs ep1), which can trigger
        # policy saturation/collapse even when weights are carried over correctly.
        #
        # When enabled, the environment uses global normalization statistics (computed once from the full
        # training_dataset) for price z-scoring, and fixed p95 scales for load/wind/solar/hydro. This makes
        # the observation distribution consistent across episodes and improves continual learning stability.
        self.use_global_normalization = True
        self.global_price_mean = 485.22962686711367
        self.global_price_std = 589.9661249957974
        # Fixed p95 scales derived from full training_dataset (scenario_*.csv, seed 789 set)
        self.global_wind_scale = 1500.0
        self.global_solar_scale = 654.4126
        self.global_hydro_scale = 867.15735
        self.global_load_scale = 4088.5974563779437
        # rolling_past mode now uses persistent online statistics carried across episodes.
        # It is strictly causal: no full-training priors are injected into the online state.
        self.rolling_past_price_std_floor = 75.0
        self.rolling_past_scale_ema_alpha = 0.10
        self.rolling_past_history_enable = True
        self.rolling_past_history_dir = "rolling_past_history_dataset"
        self.rolling_past_history_tail_days = 365
        self.rolling_past_history_rows_per_day = 144
        self.rolling_past_price_state = None
        self.rolling_past_wind_scale = None
        self.rolling_past_solar_scale = None
        self.rolling_past_hydro_scale = None
        self.rolling_past_load_scale = None

        # Network architecture
        self.net_arch = [256, 128, 64]  # Deeper network for better pattern recognition
        self.activation_fn = "relu"  # Changed from tanh for better gradient flow

        # Agent policies
        self.agent_policies = [
            {"mode": "PPO"},  # investor_0
            {"mode": "DQN"},  # battery_operator_0
            {"mode": "PPO"},  # risk_controller_0
            {"mode": "PPO"},  # meta_controller_0
        ]

        if optimized_params:
            self._apply_optimized_params(optimized_params)

    # =============================================================================
    # HELPER METHODS FOR DERIVED VALUES
    # =============================================================================

    def get_initial_asset_plan(self) -> Dict[str, Dict[str, float]]:
        """Get the initial asset deployment plan based on configured ownership fractions."""
        return {
            'wind': {'capacity_mw': self.owned_wind_capacity_mw},
            'solar': {'capacity_mw': self.owned_solar_capacity_mw},
            'hydro': {'capacity_mw': self.owned_hydro_capacity_mw},
            'battery': {'capacity_mwh': self.owned_battery_capacity_mwh}
        }

    def get_asset_capex(self, currency: str = 'DKK') -> Dict[str, float]:
        """Get the CAPEX values for different asset types.

        Args:
            currency: 'USD' for original market rates, 'DKK' for internal calculations
        """
        capex_usd = {
            'wind_mw': self.wind_capex_per_mw,
            'solar_mw': self.solar_capex_per_mw,
            'hydro_mw': self.hydro_capex_per_mw,
            'battery_mwh': self.battery_capex_per_mwh
        }

        if currency == 'USD':
            return capex_usd
        elif currency == 'DKK':
            # Convert USD to DKK for internal calculations
            return {key: value / self.dkk_to_usd_rate for key, value in capex_usd.items()}
        else:
            raise ValueError(f"Unsupported currency: {currency}")

    def get_expected_physical_values(self) -> Dict[str, float]:
        """Get expected values for validation checks (in DKK for environment validation)."""
        # Calculate CAPEX in DKK for environment validation
        actual_physical_capex_dkk = self.calculate_total_physical_capex(currency='DKK')
        remaining_cash_dkk = self.init_budget - actual_physical_capex_dkk

        return {
            'wind': self.owned_wind_capacity_mw,
            'solar': self.owned_solar_capacity_mw,
            'hydro': self.owned_hydro_capacity_mw,
            'battery': self.owned_battery_capacity_mwh,
            'physical_book_value': actual_physical_capex_dkk,  # CAPEX in DKK
            'cash_min': remaining_cash_dkk * 0.8  # At least 80% of remaining cash for trading
        }

    def get_trading_capital_limits(self) -> Dict[str, float]:
        """Get trading capital allocation limits for validation."""
        # Calculate allocated trading capital in DKK
        trading_capital_dkk = self.init_budget * self.financial_allocation
        trading_capital_usd = trading_capital_dkk * self.dkk_to_usd_rate

        # Maximum financial exposure with leverage
        max_financial_exposure_dkk = trading_capital_dkk * self.max_leverage
        max_financial_exposure_usd = max_financial_exposure_dkk * self.dkk_to_usd_rate

        return {
            'trading_capital_dkk': trading_capital_dkk,
            'trading_capital_usd': trading_capital_usd,
            'max_financial_exposure_dkk': max_financial_exposure_dkk,
            'max_financial_exposure_usd': max_financial_exposure_usd,
            'max_leverage': self.max_leverage
        }

    def calculate_total_physical_capex(self, currency: str = 'USD') -> float:
        """Calculate total expected CAPEX for physical assets.

        Args:
            currency: 'USD' for original market rates, 'DKK' for internal calculations
        """
        capex_values = self.get_asset_capex(currency=currency)
        return (
            self.owned_wind_capacity_mw * capex_values['wind_mw'] +
            self.owned_solar_capacity_mw * capex_values['solar_mw'] +
            self.owned_hydro_capacity_mw * capex_values['hydro_mw'] +
            self.owned_battery_capacity_mwh * capex_values['battery_mwh']
        )

    def validate_configuration(self) -> bool:
        """
        CONFIGURATION VALIDATION (NEW)

        Validates that all configuration parameters are consistent and within valid ranges.
        Raises ValueError if validation fails.

        Returns:
            True if all validations pass

        Raises:
            ValueError: If any validation fails
        """
        errors = []

        # 1. Allocation fractions should sum to 1.0
        total_allocation = self.physical_allocation + self.financial_allocation
        if abs(total_allocation - 1.0) > 0.01:
            errors.append(f"Allocation fractions sum to {total_allocation}, expected 1.0")

        # 2. Ownership fractions should be in [0, 1]
        for name, value in [
            ("wind_ownership_fraction", self.wind_ownership_fraction),
            ("solar_ownership_fraction", self.solar_ownership_fraction),
            ("hydro_ownership_fraction", self.hydro_ownership_fraction),
        ]:
            if not (0.0 <= value <= 1.0):
                errors.append(f"{name}={value} not in [0, 1]")

        # 3. Meta controller ranges should be valid
        # Allow fixed settings via min == max (useful for ablations / stabilizing training).
        if self.meta_freq_min > self.meta_freq_max:
            errors.append(f"meta_freq_min ({self.meta_freq_min}) > meta_freq_max ({self.meta_freq_max})")
        if self.meta_cap_min > self.meta_cap_max:
            errors.append(f"meta_cap_min ({self.meta_cap_min}) > meta_cap_max ({self.meta_cap_max})")

        # 4. Investment frequency ranges should be valid
        if self.min_investment_freq >= self.max_investment_freq:
            errors.append(f"min_investment_freq ({self.min_investment_freq}) >= max_investment_freq ({self.max_investment_freq})")
        if self.investment_freq < self.min_investment_freq or self.investment_freq > self.max_investment_freq:
            errors.append(f"investment_freq ({self.investment_freq}) not in [{self.min_investment_freq}, {self.max_investment_freq}]")

        # 5. Battery SOC bounds should be valid
        if self.batt_soc_min >= self.batt_soc_max:
            errors.append(f"batt_soc_min ({self.batt_soc_min}) >= batt_soc_max ({self.batt_soc_max})")
        if not (0.0 <= self.batt_soc_min <= 1.0) or not (0.0 <= self.batt_soc_max <= 1.0):
            errors.append("Battery SOC bounds not in [0, 1]")

        # 6. Battery round-trip efficiency (eta_charge * eta_discharge) should be in (0, 1]
        rt_eff = self.batt_eta_charge * self.batt_eta_discharge
        if not (0.0 < rt_eff <= 1.0):
            errors.append(f"batt_eta_charge*batt_eta_discharge ({rt_eff}) not in (0, 1]")

        # 7. Price bounds should be valid
        if self.minimum_price_floor >= self.maximum_price_cap:
            errors.append(f"minimum_price_floor ({self.minimum_price_floor}) >= maximum_price_cap ({self.maximum_price_cap})")
        if self.minimum_price_filter >= self.minimum_price_floor:
            errors.append(f"minimum_price_filter ({self.minimum_price_filter}) >= minimum_price_floor ({self.minimum_price_floor})")

        # 8. Leverage should be >= 1.0
        if self.max_leverage < 1.0:
            errors.append(f"max_leverage ({self.max_leverage}) < 1.0")

        # 9. Rate parameters should be in [0, 1]
        rate_params = [
            ("operating_cost_rate", self.operating_cost_rate),
            ("insurance_rate", self.insurance_rate),
            ("management_fee_rate", self.management_fee_rate),
            ("property_tax_rate", self.property_tax_rate),
            ("debt_service_rate", self.debt_service_rate),
            ("distribution_rate", self.distribution_rate),
            ("target_cash_ratio", self.target_cash_ratio),
            ("battery_opex_rate", self.battery_opex_rate),
            ("performance_fee_rate", self.performance_fee_rate),
            ("trading_cost_rate", self.trading_cost_rate),
        ]
        for name, value in rate_params:
            if not (0.0 <= value <= 1.0):
                errors.append(f"{name}={value} not in [0, 1]")
        if self.eval_distribution_rate is not None:
            if not (0.0 <= float(self.eval_distribution_rate) <= 1.0):
                errors.append(f"eval_distribution_rate={self.eval_distribution_rate} not in [0, 1]")
        mtm_model = str(getattr(self, "mtm_return_model", "percent_capped") or "percent_capped").strip().lower()
        if mtm_model not in {
            "percent",
            "percent_capped",
            "notional_price_diff",
            "horizon_settlement",
            "horizon_settlement_continuous",
        }:
            errors.append(f"mtm_return_model={self.mtm_return_model!r} is not supported")
        if float(getattr(self, "mtm_reference_price_dkk_per_mwh", 500.0)) <= 0.0:
            errors.append("mtm_reference_price_dkk_per_mwh must be positive")
        try:
            if int(getattr(self, "mtm_settlement_horizon_steps", 6)) <= 0:
                errors.append("mtm_settlement_horizon_steps must be positive")
        except (TypeError, ValueError):
            errors.append("mtm_settlement_horizon_steps must be an integer")
        entry_mode = str(getattr(self, "mtm_entry_price_mode", "current_price") or "").strip().lower()
        if entry_mode not in {"same_hour_prev_day", "rolling_same_hour_median", "current_price"}:
            errors.append(
                "mtm_entry_price_mode must be one of "
                "{'same_hour_prev_day', 'rolling_same_hour_median', 'current_price'}"
            )
        denom_mode = (
            str(getattr(self, "mtm_horizon_payoff_denominator_mode", "reference_price") or "")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if denom_mode not in {"reference_price", "entry_price_floor", "mwh_volume", "mwh"}:
            errors.append(
                "mtm_horizon_payoff_denominator_mode must be one of "
                "{'reference_price', 'entry_price_floor', 'mwh_volume'}"
            )
        settlement_mode = (
            str(getattr(self, "mtm_settlement_price_mode", "energy_index") or "energy_index")
            .strip()
            .lower()
            .replace("-", "_")
        )
        valid_settlement_modes = {
            "energy_index",
            "none",
            "base",
            "cross_zone_basis",
            "basis_adjusted",
            "external_series",
            "external",
            "realized",
            "real_settlement",
        }
        if settlement_mode not in valid_settlement_modes:
            errors.append(
                "mtm_settlement_price_mode must be one of "
                "{'energy_index', 'cross_zone_basis', 'external_series'}"
            )
        try:
            if float(getattr(self, "mtm_basis_scale", 0.0)) < 0.0:
                errors.append("mtm_basis_scale must be non-negative")
        except Exception:
            errors.append("mtm_basis_scale must be numeric")
        basis_centering = (
            str(getattr(self, "mtm_basis_centering_mode", "rolling_median") or "rolling_median")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if basis_centering not in {"none", "rolling_median", "expanding_median"}:
            errors.append(
                "mtm_basis_centering_mode must be one of "
                "{'none', 'rolling_median', 'expanding_median'}"
            )
        try:
            if int(getattr(self, "mtm_basis_centering_window_steps", 4320)) <= 0:
                errors.append("mtm_basis_centering_window_steps must be positive")
        except Exception:
            errors.append("mtm_basis_centering_window_steps must be an integer")
        if bool(getattr(self, "mtm_apply_price_return_cap", True)):
            if float(getattr(self, "mtm_price_return_cap_min", -0.001)) >= float(getattr(self, "mtm_price_return_cap_max", 0.001)):
                errors.append("mtm_price_return_cap_min must be < mtm_price_return_cap_max")
        sizing_base = str(getattr(self, "investor_notional_sizing_base", "initial_trading_sleeve") or "").strip().lower().replace("-", "_")
        valid_sizing_bases = {"initial_trading_sleeve", "live_trading_cash", "initial_fund_nav"}
        if sizing_base not in valid_sizing_bases:
            errors.append(
                "investor_notional_sizing_base="
                f"{getattr(self, 'investor_notional_sizing_base', None)!r} is not supported"
            )
        try:
            cap_alloc = float(getattr(self, "capital_allocation_fraction", 0.60))
            if not (0.0 <= cap_alloc <= 1.0):
                errors.append(f"capital_allocation_fraction={cap_alloc} not in [0, 1]")
        except Exception:
            errors.append("capital_allocation_fraction must be numeric")
        try:
            max_pos = float(getattr(self, "max_position_size", 0.35))
            if max_pos <= 0.0:
                errors.append(f"max_position_size={max_pos} must be positive")
        except Exception:
            errors.append("max_position_size must be numeric")
        try:
            loss_thr = float(getattr(self, "mtm_loss_exit_threshold_pct", 0.06))
            if loss_thr < 0.0:
                errors.append(f"mtm_loss_exit_threshold_pct={loss_thr} must be non-negative")
        except Exception:
            errors.append("mtm_loss_exit_threshold_pct must be numeric")
        try:
            margin_fraction = float(getattr(self, "trading_sleeve_maintenance_margin_fraction", 0.05))
            if not (0.0 <= margin_fraction <= 1.0):
                errors.append(
                    "trading_sleeve_maintenance_margin_fraction="
                    f"{margin_fraction} must be in [0, 1]"
                )
        except Exception:
            errors.append("trading_sleeve_maintenance_margin_fraction must be numeric")
        try:
            risk_min = float(getattr(self, "risk_exposure_cap_min", 0.25))
            risk_max = float(getattr(self, "risk_exposure_cap_max", 1.0))
            if not (0.0 <= risk_min <= risk_max <= 1.0):
                errors.append(
                    "risk_exposure_cap_min/max must satisfy "
                    f"0 <= min <= max <= 1, got min={risk_min}, max={risk_max}"
                )
        except Exception:
            errors.append("risk_exposure_cap_min/max must be numeric")
        forecast_mode = str(getattr(self, "forecast_prior_control_mode", "residual") or "residual").strip().lower()
        if forecast_mode not in {"residual", "distributional"}:
            errors.append(
                "forecast_prior_control_mode must be one of "
                "{'distributional', 'residual'}, "
                f"got {getattr(self, 'forecast_prior_control_mode', None)!r}"
            )
        payoff_target_mode = (
            str(getattr(self, "forecast_prior_payoff_target_mode", "auto") or "auto")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if payoff_target_mode not in {"auto", "price_return", "horizon_settlement"}:
            errors.append(
                "forecast_prior_payoff_target_mode must be one of "
                "{'auto', 'price_return', 'horizon_settlement'}"
            )
        for name in (
            "forecast_prior_max_abs_exposure",
            "forecast_prior_residual_scale",
            "forecast_prior_blend",
            "forecast_prior_directional_floor",
            "forecast_prior_residual_to_prior_ratio",
            "forecast_prior_inactive_residual_scale",
            "forecast_prior_policy_relative_hit_margin",
            "forecast_prior_residual_evidence_floor",
            "forecast_prior_beta_max_abs_exposure",
            "forecast_prior_beta_weight",
            "forecast_prior_tail_loss_budget_fraction",
            "forecast_prior_distributional_loss_budget_fraction",
            "forecast_prior_distributional_disaster_loss_budget_fraction",
            "forecast_prior_distributional_fixed_cap_abs",
            "forecast_prior_distributional_cold_start_exposure",
        ):
            try:
                value = float(getattr(self, name))
                if not (0.0 <= value <= 1.0):
                    errors.append(f"{name}={value} not in [0, 1]")
            except Exception:
                errors.append(f"{name} must be numeric")
        try:
            pr_min = int(getattr(self, "forecast_prior_policy_relative_min_samples", 50))
            if pr_min < 1:
                errors.append(
                    f"forecast_prior_policy_relative_min_samples={pr_min} must be positive"
                )
        except Exception:
            errors.append("forecast_prior_policy_relative_min_samples must be integer")
        try:
            adv_scale = float(getattr(self, "forecast_prior_policy_relative_advantage_scale", 0.02))
            if adv_scale <= 0.0:
                errors.append(
                    f"forecast_prior_policy_relative_advantage_scale={adv_scale} must be positive"
                )
        except Exception:
            errors.append("forecast_prior_policy_relative_advantage_scale must be numeric")
        try:
            beta_min = int(getattr(self, "forecast_prior_beta_min_samples", 50))
            if beta_min < 1:
                errors.append(f"forecast_prior_beta_min_samples={beta_min} must be positive")
        except Exception:
            errors.append("forecast_prior_beta_min_samples must be integer")
        try:
            beta_scale = float(getattr(self, "forecast_prior_beta_lcb_scale", 0.005))
            if beta_scale <= 0.0:
                errors.append(f"forecast_prior_beta_lcb_scale={beta_scale} must be positive")
        except Exception:
            errors.append("forecast_prior_beta_lcb_scale must be numeric")
        try:
            beta_clip = float(getattr(self, "forecast_prior_beta_return_clip", 0.25))
            if beta_clip <= 0.0:
                errors.append(f"forecast_prior_beta_return_clip={beta_clip} must be positive")
        except Exception:
            errors.append("forecast_prior_beta_return_clip must be numeric")
        try:
            tail_min = int(getattr(self, "forecast_prior_tail_min_samples", 50))
            if tail_min < 1:
                errors.append(f"forecast_prior_tail_min_samples={tail_min} must be positive")
        except Exception:
            errors.append("forecast_prior_tail_min_samples must be integer")
        try:
            tail_q = float(getattr(self, "forecast_prior_tail_quantile", 0.99))
            if not (0.50 <= tail_q <= 0.9999):
                errors.append(f"forecast_prior_tail_quantile={tail_q} must be in [0.50, 0.9999]")
        except Exception:
            errors.append("forecast_prior_tail_quantile must be numeric")
        for name in (
            "forecast_prior_tail_stress_return_floor",
            "forecast_prior_tail_default_return",
            "forecast_prior_tail_return_clip",
            "forecast_prior_distributional_conditional_tail_floor",
            "forecast_prior_distributional_disaster_tail_floor",
            "forecast_prior_distributional_default_tail_return",
            "forecast_prior_distributional_return_clip",
            "forecast_prior_distributional_edge_scale",
        ):
            try:
                value = float(getattr(self, name))
                if value <= 0.0:
                    errors.append(f"{name}={value} must be positive")
            except Exception:
                errors.append(f"{name} must be numeric")
        feasible_enabled = bool(
            getattr(self, "forecast_prior_feasible_action_mappo", False)
        )
        observation_only = bool(
            getattr(self, "forecast_prior_observation_only_mappo", False)
        )
        mirror_enabled = bool(getattr(self, "forecast_prior_mirror_mappo", False))
        if sum(bool(v) for v in (observation_only, feasible_enabled, mirror_enabled)) > 1:
            errors.append(
                "forecast_prior_observation_only_mappo and "
                "forecast_prior_feasible_action_mappo and forecast_prior_mirror_mappo "
                "are mutually exclusive"
            )
        if observation_only and not bool(getattr(self, "enable_forecast_utilization", False)):
            errors.append("forecast-observation-only MAPPO requires enable_forecast_utilization")
        if observation_only and forecast_mode != "distributional":
            errors.append(
                "forecast-observation-only MAPPO requires "
                "forecast_prior_control_mode=distributional"
            )
        if feasible_enabled and not bool(getattr(self, "enable_forecast_utilization", False)):
            errors.append("feasible-action MAPPO requires enable_forecast_utilization")
        if feasible_enabled and forecast_mode != "distributional":
            errors.append(
                "feasible-action MAPPO requires forecast_prior_control_mode=distributional"
            )
        if mirror_enabled and not bool(getattr(self, "enable_forecast_utilization", False)):
            errors.append("forecast-mirror MAPPO requires enable_forecast_utilization")
        if mirror_enabled and forecast_mode != "distributional":
            errors.append(
                "forecast-mirror MAPPO requires forecast_prior_control_mode=distributional"
            )
        if mirror_enabled and str(getattr(self, "algo", "")).strip().lower() != "mappo":
            errors.append("forecast-mirror integration requires algo=mappo")
        if mirror_enabled and not bool(getattr(self, "investor_use_beta_policy", False)):
            errors.append("forecast-mirror integration requires investor_use_beta_policy=true")
        if mirror_enabled and str(
            getattr(self, "mtm_return_model", "") or ""
        ).strip().lower() not in {"horizon_settlement", "horizon_settlement_continuous"}:
            errors.append(
                "forecast-mirror integration requires a fixed-horizon settlement return model"
            )
        if mirror_enabled and str(
            getattr(self, "mtm_horizon_payoff_denominator_mode", "") or ""
        ).strip().lower() not in {"mwh_volume", "mwh"}:
            errors.append("forecast-mirror integration requires the mwh_volume payoff")
        for name in (
            "forecast_mirror_prior_concentration",
            "forecast_mirror_trust_max",
            "forecast_mirror_trust_regularization",
            "forecast_mirror_trust_gradient_clip",
            "forecast_mirror_counterfactual_payoff_scale",
            "forecast_mirror_counterfactual_loss_aversion",
        ):
            try:
                value = float(getattr(self, name))
                if value < 0.0 or (
                    name in {
                        "forecast_mirror_trust_gradient_clip",
                        "forecast_mirror_counterfactual_payoff_scale",
                    }
                    and value <= 0.0
                ):
                    errors.append(f"{name}={value} has an invalid range")
            except Exception:
                errors.append(f"{name} must be numeric")
        try:
            trust_initial = float(getattr(self, "forecast_mirror_trust_initial", 0.10))
            if not (0.0 < trust_initial < 1.0):
                errors.append("forecast_mirror_trust_initial must be in (0, 1)")
        except Exception:
            errors.append("forecast_mirror_trust_initial must be numeric")
        for name in (
            "forecast_mirror_trust_hidden_dim",
            "forecast_mirror_trust_min_samples",
            "forecast_mirror_trust_batch_size",
            "forecast_mirror_counterfactual_grid_size",
        ):
            try:
                value = int(getattr(self, name))
                minimum = 5 if name == "forecast_mirror_counterfactual_grid_size" else 1
                if value < minimum:
                    errors.append(f"{name}={value} must be at least {minimum}")
            except Exception:
                errors.append(f"{name} must be integer")
        for name in (
            "forecast_prior_feasible_action_window",
            "forecast_prior_feasible_action_min_samples",
        ):
            try:
                value = int(getattr(self, name))
                if value < 1:
                    errors.append(f"{name}={value} must be positive")
            except Exception:
                errors.append(f"{name} must be integer")
        try:
            feasible_q = float(
                getattr(self, "forecast_prior_feasible_action_cvar_quantile", 0.90)
            )
            if not (0.50 <= feasible_q <= 0.9999):
                errors.append(
                    "forecast_prior_feasible_action_cvar_quantile="
                    f"{feasible_q} must be in [0.50, 0.9999]"
                )
        except Exception:
            errors.append("forecast_prior_feasible_action_cvar_quantile must be numeric")
        for name in (
            "forecast_prior_feasible_action_shortfall_budget",
            "forecast_prior_feasible_action_dual_lr",
            "forecast_prior_feasible_action_dual_max",
            "forecast_prior_feasible_action_reward_weight",
        ):
            try:
                value = float(getattr(self, name))
                if value < 0.0:
                    errors.append(f"{name}={value} must be non-negative")
            except Exception:
                errors.append(f"{name} must be numeric")
        for name in (
            "forecast_prior_feasible_action_payoff_scale",
            "forecast_prior_feasible_action_score_clip",
            "forecast_prior_feasible_action_evidence_power",
        ):
            try:
                value = float(getattr(self, name))
                if value <= 0.0:
                    errors.append(f"{name}={value} must be positive")
            except Exception:
                errors.append(f"{name} must be numeric")
        try:
            dist_window = int(getattr(self, "forecast_prior_distributional_window", 2000))
            if dist_window < 1:
                errors.append(
                    f"forecast_prior_distributional_window={dist_window} must be positive"
                )
        except Exception:
            errors.append("forecast_prior_distributional_window must be integer")
        try:
            dist_min = int(getattr(self, "forecast_prior_distributional_min_samples", 50))
            if dist_min < 1:
                errors.append(
                    f"forecast_prior_distributional_min_samples={dist_min} must be positive"
                )
        except Exception:
            errors.append("forecast_prior_distributional_min_samples must be integer")
        for name in (
            "forecast_prior_distributional_tail_quantile",
            "forecast_prior_distributional_disaster_quantile",
        ):
            try:
                value = float(getattr(self, name))
                if not (0.50 <= value <= 0.9999):
                    errors.append(f"{name}={value} must be in [0.50, 0.9999]")
            except Exception:
                errors.append(f"{name} must be numeric")
        try:
            dist_tail_q = float(getattr(self, "forecast_prior_distributional_tail_quantile", 0.99))
            dist_disaster_q = float(
                getattr(self, "forecast_prior_distributional_disaster_quantile", 0.999)
            )
            if dist_disaster_q < dist_tail_q:
                errors.append(
                    "forecast_prior_distributional_disaster_quantile must be >= "
                    "forecast_prior_distributional_tail_quantile"
                )
        except Exception:
            pass
        try:
            edge_hurdle = float(getattr(self, "forecast_prior_distributional_edge_hurdle", 0.0))
            if edge_hurdle < 0.0:
                errors.append("forecast_prior_distributional_edge_hurdle must be non-negative")
        except Exception:
            errors.append("forecast_prior_distributional_edge_hurdle must be numeric")
        try:
            dist_conf_power = float(
                getattr(self, "forecast_prior_distributional_confidence_power", 1.0)
            )
            if dist_conf_power < 0.0:
                errors.append("forecast_prior_distributional_confidence_power must be non-negative")
        except Exception:
            errors.append("forecast_prior_distributional_confidence_power must be numeric")
        bucket_mode = str(
            getattr(self, "forecast_prior_distributional_bucket_mode", "conditional") or "conditional"
        ).strip().lower()
        if bucket_mode not in {"conditional", "global", "directional_fixed_cap"}:
            errors.append(
                "forecast_prior_distributional_bucket_mode must be one of "
                "{'conditional', 'global', 'directional_fixed_cap'}"
            )
        try:
            conf_power = float(getattr(self, "forecast_prior_direction_confidence_power", 1.0))
            if conf_power < 0.0:
                errors.append(
                    f"forecast_prior_direction_confidence_power={conf_power} must be non-negative"
                )
        except Exception:
            errors.append("forecast_prior_direction_confidence_power must be numeric")
        if float(self.friction_cost_multiplier) < 0.0:
            errors.append(f"friction_cost_multiplier={self.friction_cost_multiplier} must be non-negative")
        market_fee_model = str(getattr(self, "market_fee_model", "legacy_notional_fixed"))
        market_fee_model = market_fee_model.strip().lower().replace("-", "_")
        if market_fee_model not in {"legacy_notional_fixed", "nord_pool_intraday_2026"}:
            errors.append(
                "market_fee_model must be one of "
                "{'legacy_notional_fixed', 'nord_pool_intraday_2026'}"
            )
        if float(getattr(self, "transaction_fee_dkk_per_mwh", 0.0)) < 0.0:
            errors.append("transaction_fee_dkk_per_mwh must be non-negative")
        if float(getattr(self, "annual_market_access_fee_dkk", 0.0)) < 0.0:
            errors.append("annual_market_access_fee_dkk must be non-negative")
        access_allocation = float(getattr(self, "market_access_fee_allocation_fraction", 1.0))
        if not 0.0 <= access_allocation <= 1.0:
            errors.append("market_access_fee_allocation_fraction must be in [0, 1]")
        if float(self.half_spread_bp) < 0.0:
            errors.append(f"half_spread_bp={self.half_spread_bp} must be non-negative")
        # market-impact task
        if float(getattr(self, "impact_coef_bp", 0.0)) < 0.0:
            errors.append(f"impact_coef_bp={self.impact_coef_bp} must be non-negative")
        if float(getattr(self, "impact_exponent", 0.5)) <= 0.0:
            errors.append(f"impact_exponent={self.impact_exponent} must be positive")
        impact_ref = str(getattr(self, "impact_ref_notional", "sleeve")).strip().lower()
        impact_volume_refs = {"volume", "market_volume", "day_ahead_volume", "liquidity_volume"}
        # market-impact task
        if impact_ref not in {"sleeve", *impact_volume_refs}:
            try:
                if float(getattr(self, "impact_ref_notional")) <= 0.0:
                    errors.append(
                        f"impact_ref_notional={self.impact_ref_notional} must be positive, 'sleeve', or 'volume'"
                    )
            except Exception:
                errors.append(
                    f"impact_ref_notional={self.impact_ref_notional!r} must be positive, 'sleeve', or 'volume'"
                )
        if impact_ref in impact_volume_refs and float(getattr(self, "impact_coef_bp", 0.0)) > 0.0:
            # market-impact task
            if not str(getattr(self, "impact_volume_data_path", "") or "").strip():
                errors.append(
                    "impact_volume_data_path is required when impact_ref_notional='volume' and impact_coef_bp > 0"
                )
        if str(getattr(self, "impact_volume_unit", "mwh")).strip().lower() not in {"mwh", "dkk"}:
            errors.append("impact_volume_unit must be 'mwh' or 'dkk'")
        try:
            if float(getattr(self, "impact_volume_max_staleness_minutes", 90.0)) <= 0.0:
                errors.append("impact_volume_max_staleness_minutes must be positive")
        except Exception:
            errors.append("impact_volume_max_staleness_minutes must be numeric")
        try:
            if float(getattr(self, "impact_volume_price_floor_dkk_per_mwh", 50.0)) <= 0.0:
                errors.append("impact_volume_price_floor_dkk_per_mwh must be positive")
        except Exception:
            errors.append("impact_volume_price_floor_dkk_per_mwh must be numeric")
        try:
            if float(getattr(self, "no_trade_threshold", 0.01)) < 0.0:
                errors.append("no_trade_threshold must be non-negative")
        except Exception:
            errors.append("no_trade_threshold must be numeric")
        no_trade_reference = str(
            getattr(self, "no_trade_threshold_reference", "executable_capacity")
            or "executable_capacity"
        ).strip().lower().replace("-", "_")
        if no_trade_reference not in {"executable_capacity", "max_position"}:
            errors.append(
                "no_trade_threshold_reference must be 'executable_capacity' or 'max_position'"
            )
        try:
            if float(getattr(self, "liquidity_participation_cap_fraction", 0.0)) < 0.0:
                errors.append("liquidity_participation_cap_fraction must be non-negative")
        except Exception:
            errors.append("liquidity_participation_cap_fraction must be numeric")
        source = str(getattr(self, "liquidity_volume_source", "load") or "load").strip().lower()
        if source not in {"load", "generation", "max_load_generation", "impact_volume"}:
            errors.append("liquidity_volume_source must be one of {'load', 'generation', 'max_load_generation', 'impact_volume'}")
        try:
            if float(getattr(self, "liquidity_volume_multiplier", 1.0)) <= 0.0:
                errors.append("liquidity_volume_multiplier must be positive")
        except Exception:
            errors.append("liquidity_volume_multiplier must be numeric")
        try:
            if float(getattr(self, "liquidity_min_volume_mwh", 1.0)) <= 0.0:
                errors.append("liquidity_min_volume_mwh must be positive")
        except Exception:
            errors.append("liquidity_min_volume_mwh must be numeric")
        try:
            if float(getattr(self, "liquidity_tail_impact_threshold_dkk_per_mwh", 5000.0)) <= 0.0:
                errors.append("liquidity_tail_impact_threshold_dkk_per_mwh must be positive")
        except Exception:
            errors.append("liquidity_tail_impact_threshold_dkk_per_mwh must be numeric")
        for name in (
            "liquidity_tail_impact_multiplier",
            "liquidity_tail_impact_power",
            "liquidity_tail_impact_max_multiplier",
            "collateral_notional_margin_fraction",
            "collateral_stress_loss_fraction",
            "collateral_stress_price_dkk_per_mwh",
            "collateral_funding_rate_annual",
            "collateral_tradeable_haircut",
        ):
            try:
                if float(getattr(self, name)) < 0.0:
                    errors.append(f"{name} must be non-negative")
            except Exception:
                errors.append(f"{name} must be numeric")

        # 10. Forecast horizons should be positive integers
        for horizon_name, horizon_steps in self.forecast_horizons.items():
            if not isinstance(horizon_steps, int) or horizon_steps <= 0:
                errors.append(f"forecast_horizons[{horizon_name}]={horizon_steps} not a positive integer")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("[OK] Configuration validation passed")
        return True

    def _apply_optimized_params(self, params: Dict[str, Any]):
        print("Applying optimized hyperparameters...")

        # Accept either 'update_every' or 'n_steps'
        self.update_every = int(params.get('update_every', params.get('n_steps', self.update_every)))

        # Learning parameters
        self.lr = float(params.get('lr', self.lr))
        self.ent_coef = params.get('ent_coef', self.ent_coef)
        self.batch_size = int(params.get('batch_size', self.batch_size))
        self.gamma = float(params.get('gamma', self.gamma))
        self.gae_lambda = float(params.get('gae_lambda', self.gae_lambda))
        self.clip_range = float(params.get('clip_range', self.clip_range))
        self.vf_coef = float(params.get('vf_coef', self.vf_coef))
        self.max_grad_norm = float(params.get('max_grad_norm', self.max_grad_norm))
        self.investor_use_beta_policy = bool(params.get('investor_use_beta_policy', self.investor_use_beta_policy))
        self.investor_beta_epsilon = float(params.get('investor_beta_epsilon', self.investor_beta_epsilon))

        # Net arch: take explicit list if provided; otherwise map from size label
        if isinstance(params.get('net_arch'), (list, tuple)):
            self.net_arch = list(params['net_arch'])
        else:
            net_arch_mapping = {
                'small': [64, 32],
                'medium': [128, 64],
                'large': [256, 128, 64]
            }
            self.net_arch = net_arch_mapping.get(params.get('net_arch_size', 'medium'), [128, 64])

        # Activation: accept 'activation' or 'activation_fn'
        self.activation_fn = params.get('activation', params.get('activation_fn', self.activation_fn))

        # Agent modes: if a full list is provided, use it; otherwise accept *_mode aliases
        if isinstance(params.get('agent_policies'), list) and params['agent_policies']:
            self.agent_policies = params['agent_policies']
        else:
            self.agent_policies = [
                {"mode": params.get('investor_mode', 'PPO')},
                {"mode": params.get('battery_mode', 'DQN')},
                {"mode": params.get('risk_mode', 'PPO')},
                {"mode": params.get('meta_mode', 'PPO')},
            ]

        print(f"   Learning rate: {self.lr:.2e}")
        print(f"   Entropy coefficient: {self.ent_coef}")
        print(f"   Network architecture: {self.net_arch}")
        print(f"   Agent modes: {[p['mode'] for p in self.agent_policies]}")
        print(f"   Update every: {self.update_every}")
        print(f"   Batch size: {self.batch_size}")



"""
Centralized Logging Module

This module provides:
1. Centralized application logging configuration

ALL logging should go through this module - no print() statements or direct logging.basicConfig() calls.
"""

import csv
import json
import os
import sys
import numpy as np
import logging
from typing import Dict, Any, Optional

# ============================================================================
# CENTRALIZED LOGGING CONFIGURATION
# ============================================================================

_logging_configured = False
_log_file_handler = None


class ProgressAwareConsoleHandler(logging.StreamHandler):
    """
    Console handler that cooperates with tqdm progress bars.

    It writes through tqdm.write() when tqdm is available so normal log lines do
    not overwrite or visually corrupt the in-place training progress display.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                from tqdm import tqdm
                tqdm.write(msg, file=stream)
            except Exception:
                stream.write(msg + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    force_reconfigure: bool = False
) -> None:
    """
    Configure centralized logging for the entire application.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
        format_string: Optional custom format string
        force_reconfigure: If True, reconfigure even if already configured (default: False)
    """
    global _logging_configured, _log_file_handler
    
    if _logging_configured and not force_reconfigure:
        # Already configured, but allow adding file handler if not present
        if log_file and _log_file_handler is None:
            root_logger = logging.getLogger()
            if format_string is None:
                format_string = '%(levelname)s: %(message)s'
            _log_file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            _log_file_handler.setLevel(level)
            _log_file_handler.setFormatter(logging.Formatter(format_string))
            root_logger.addHandler(_log_file_handler)
        return
    
    if format_string is None:
        format_string = '%(levelname)s: %(message)s'
    
    # Remove existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Close previous file handler if exists
    if _log_file_handler is not None:
        try:
            _log_file_handler.close()
        except Exception:
            pass
        _log_file_handler = None
    
    # Console handler: write to the original stderr stream (if wrapped) and use
    # a tqdm-aware handler so progress bars remain readable during training.
    console_stream = getattr(sys.stderr, "stderr", sys.stderr)
    console_handler = ProgressAwareConsoleHandler(console_stream)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        _log_file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        _log_file_handler.setLevel(level)
        _log_file_handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(_log_file_handler)
    
    root_logger.setLevel(level)
    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    if not _logging_configured:
        configure_logging()  # Auto-configure if not done
    return logging.getLogger(name)


def set_console_logging_level(level: int):
    """
    Update non-file console handler levels without touching file logging.

    Returns a list of (handler, previous_level) entries that can be passed to
    `restore_console_logging_levels`.
    """
    saved_levels = []
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        if isinstance(handler, logging.StreamHandler):
            saved_levels.append((handler, handler.level))
            handler.setLevel(level)
    return saved_levels


def restore_console_logging_levels(saved_levels) -> None:
    """Restore console handler levels captured by `set_console_logging_level`."""
    for handler, previous_level in saved_levels or []:
        try:
            handler.setLevel(previous_level)
        except Exception:
            pass


def emit_progress_lines(lines) -> None:
    """
    Write plain progress lines to both console and file handlers, bypassing
    normal log levels so training can stay quiet except for explicit snapshots.
    """
    if isinstance(lines, str):
        lines = [lines]
    else:
        lines = [str(line) for line in (lines or [])]
    if not lines:
        return

    root_logger = logging.getLogger()
    console_captures_file = False
    for handler in root_logger.handlers:
        try:
            if isinstance(handler, ProgressAwareConsoleHandler):
                stream = getattr(handler, "stream", None)
                if stream is not None and isinstance(stream, TeeOutput):
                    console_captures_file = True
                    break
        except Exception:
            continue

    for handler in root_logger.handlers:
        try:
            stream = getattr(handler, "stream", None)
            if stream is None:
                continue

            if isinstance(handler, ProgressAwareConsoleHandler):
                try:
                    from tqdm import tqdm
                    for line in lines:
                        tqdm.write(line, file=stream)
                except Exception:
                    for line in lines:
                        stream.write(line + "\n")
                    handler.flush()
                continue

            if console_captures_file and isinstance(handler, logging.FileHandler):
                continue

            for line in lines:
                stream.write(line + "\n")
            handler.flush()
        except Exception:
            continue


# ============================================================================
# TERMINAL OUTPUT TEE (for capturing print statements)
# ============================================================================

class TeeOutput:
    """
    Class to tee output to both console and file.
    Captures both print() statements and logging output.
    Also captures exceptions and tracebacks.
    
    This class can be assigned to both sys.stdout and sys.stderr,
    and will write to the original streams plus the log file.
    """
    def __init__(self, file_path: str):
        self.file = open(file_path, 'w', encoding='utf-8', buffering=1)  # Line buffered
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self.file_path = file_path
        self._is_stdout = False
        self._is_stderr = False
        
    def _set_stream_type(self, is_stdout: bool = False, is_stderr: bool = False):
        """Set which stream type this instance represents"""
        self._is_stdout = is_stdout
        self._is_stderr = is_stderr
        
    def write(self, text: str) -> None:
        """Write to both console and file"""
        try:
            # Determine which original stream to write to
            original_stream = None
            if self._is_stdout or (not self._is_stderr and not self._is_stdout):
                original_stream = self._original_stdout
            elif self._is_stderr:
                original_stream = self._original_stderr
            else:
                original_stream = self._original_stdout  # Default to stdout
            
            # Write to original stream (console)
            if original_stream:
                original_stream.write(text)
                original_stream.flush()
            
            # Always write to file
            if self.file and not self.file.closed:
                self.file.write(text)
                self.file.flush()  # Ensure immediate write
        except Exception:
            # If file write fails, at least try to write to console
            try:
                if original_stream:
                    original_stream.write(text)
                    original_stream.flush()
            except Exception:
                pass
        
    def flush(self) -> None:
        """Flush both streams"""
        try:
            if self._is_stdout or (not self._is_stderr and not self._is_stdout):
                self._original_stdout.flush()
            elif self._is_stderr:
                self._original_stderr.flush()
            if self.file and not self.file.closed:
                self.file.flush()
        except Exception:
            pass
        
    def close(self) -> None:
        """Close the file"""
        try:
            if self.file and not self.file.closed:
                self.file.flush()
                self.file.close()
        except Exception:
            pass
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def isatty(self) -> bool:
        """Return False to indicate this is not a TTY (for libraries that check)"""
        return False
    
    def fileno(self) -> int:
        """Return file descriptor for original stdout/stderr"""
        if self._is_stdout or (not self._is_stderr and not self._is_stdout):
            return self._original_stdout.fileno()
        elif self._is_stderr:
            return self._original_stderr.fileno()
        else:
            return self._original_stdout.fileno()
    
    @property
    def stdout(self):
        """Return original stdout for restoration"""
        return self._original_stdout
    
    @property
    def stderr(self):
        """Return original stderr for restoration"""
        return self._original_stderr


# ============================================================================
# REWARD LOGGER (CSV Data Logging)
# ============================================================================

class RewardLogger:
    """Logs detailed reward, forecast, and portfolio information at every timestep"""
    
    def __init__(self, log_dir: str = "debug_logs", tier_name: str = "tier1", enabled: bool = True):
        self.log_dir = log_dir
        self.tier_name = tier_name
        self.enabled = bool(enabled)
        self.flush_every_steps = 100
        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)

        # Track unknown-field warnings so we only log each unknown field once
        self._warned_unknown_fields = set()
        self.investor_health_fields = [
            'inv_mean_clip_hit',
            'inv_mean_clip_hit_rate',
            'inv_mu_abs_roll',
            'inv_mu_sign_consistency',
            'inv_mu_raw',
            'inv_sigma_raw',
            'inv_a_raw',
            'inv_action_mean',
            'inv_action_sigma',
            'inv_action_sample',
            'inv_tanh_mu',
            'inv_tanh_a',
            'inv_sat_mean',
            'inv_sat_sample',
            'inv_sat_noise_only',
            'inv_reward_step',
            'inv_value',
            'inv_value_next',
            'inv_td_error',
        ]
        self.investor_health_context_fields = [
            'episode',
            'timestep',
            'training_global_step',
            'decision_step',
            'exposure_exec',
            'position_exposure',
            'action_sign',
            'trade_signal_active',
            'trade_signal_sign',
            'investor_reward',
            'fund_nav_usd',
            'fund_nav_dkk',
            'price_return_1step',
            'price_return_invfreq',
            'training_return',
            'training_horizon_steps',
            'investor_action',
        ]

        # ---- Battery operator health ----
        self.battery_health_fields = [
            'bat_action_idx',
            'bat_q_max',
            'bat_q_chosen',
            'bat_reward_step',
        ]
        self.battery_health_context_fields = [
            'episode', 'timestep', 'training_global_step',
            'bat_decision_step', 'bat_soc', 'bat_spread',
            'bat_energy', 'bat_capacity',
            'battery_reward',
        ]

        # ---- Risk controller health ----
        self.risk_health_fields = [
            'risk_action_raw',
            'risk_reward_step',
            'risk_value',
            'risk_value_next',
            'risk_td_error',
        ]
        self.risk_health_context_fields = [
            'episode', 'timestep', 'training_global_step',
            'risk_decision_step', 'risk_multiplier',
            'risk_drawdown', 'overall_risk_snapshot',
            'risk_score',
        ]

        # ---- Meta controller health ----
        self.meta_health_fields = [
            'meta_action_raw',
            'meta_reward_step',
            'meta_value',
            'meta_value_next',
            'meta_td_error',
        ]
        self.meta_health_context_fields = [
            'episode', 'timestep', 'training_global_step',
            'meta_decision_step', 'meta_cap_fraction',
            'meta_budget_n',
            'base_reward',
        ]
        
        # Combined field order for each agent's health CSV (context + health fields in one row)
        self.investor_health_all_fields = self.investor_health_context_fields + [
            f for f in self.investor_health_fields if f not in self.investor_health_context_fields
        ]
        self.battery_health_all_fields = self.battery_health_context_fields + [
            f for f in self.battery_health_fields if f not in self.battery_health_context_fields
        ]
        self.risk_health_all_fields = self.risk_health_context_fields + [
            f for f in self.risk_health_fields if f not in self.risk_health_context_fields
        ]
        self.meta_health_all_fields = self.meta_health_context_fields + [
            f for f in self.meta_health_fields if f not in self.meta_health_context_fields
        ]

        # CSV file for step-by-step logging (will be set per episode)
        self.csv_file = None
        self.csv_writer = None
        self.csv_file_handle = None
        # Per-agent health CSV handles (written incrementally, one row per log_step)
        self.agent_health_handles = {}   # key: 'investor'|'battery'|'risk'|'meta'
        self.agent_health_writers = {}   # matching DictWriters
        
        # Category-specific CSV writers (portfolio, forecast, rewards, positions)
        self.category_writers = {}
        self.category_handles = {}
        self.category_fields = {
            'portfolio': [
                'episode', 'timestep',
                # Core portfolio metrics
                'portfolio_value_usd_millions', 'cash_dkk',
                'fund_nav_usd',
                'trading_sleeve_value_usd_thousands',
                'trading_mtm_tracker_usd_thousands',
                'operating_revenue_usd_thousands', 'mtm_pnl',
                # Fund NAV component breakdown (DKK)
                'fund_nav_dkk', 'trading_cash_dkk', 'trading_sleeve_value_dkk', 'physical_book_value_dkk',
                'accumulated_operational_revenue_dkk', 'financial_mtm_dkk', 'financial_exposure_dkk',
                'liquidity_market_volume_mwh', 'liquidity_causal_market_volume_mwh',
                'liquidity_volume_cap_mwh',
                'liquidity_requested_volume_mwh', 'liquidity_executed_volume_mwh',
                'liquidity_participation', 'liquidity_scale',
                'liquidity_tail_impact_multiplier', 'liquidity_tail_spread_dkk_per_mwh',
                'market_impact_ref_notional_dkk', 'market_impact_participation', 'market_impact_bp',
                'no_trade_threshold_dkk', 'no_trade_threshold_reference_dkk',
                'collateral_required_dkk', 'collateral_funding_cost_dkk',
                'cumulative_collateral_funding_costs_dkk',
                'collateral_open_notional_dkk', 'collateral_open_volume_mwh',
                'cumulative_trading_costs_dkk', 'cumulative_impact_costs_dkk',
                'cumulative_volume_transaction_fees_dkk',
                'cumulative_market_access_fees_dkk', 'market_access_fee_step_dkk',
                'horizon_settlement_pnl_dkk', 'horizon_settlement_count',
                'total_distributions_dkk', 'distribution_adjusted_nav_dkk',
                'distribution_adjusted_trading_sleeve_dkk',
                'depreciation_ratio', 'years_elapsed',
                # NAV attribution drivers (per-step fund NAV, not trading cash)
                'nav_start', 'nav_end', 'pnl_total', 'pnl_battery', 'pnl_generation', 'pnl_hedge', 'cash_delta_ops',
                # Battery dispatch metrics (FIX #5)
                'battery_decision', 'battery_intensity', 'battery_spread', 'battery_adjusted_hurdle', 'battery_volatility_adj',
                # Price & returns
                'price_current', 'price_return_1step', 'price_return_invfreq',
                'training_return', 'training_horizon_steps'
            ],
            'positions': [
                'episode', 'timestep',
                'position_signed', 'held_exposure_signed', 'position_exposure',
                'wind_pos_norm', 'solar_pos_norm', 'hydro_pos_norm',
                'decision_step', 'exposure_exec', 'action_sign',
                'trade_signal_active', 'trade_signal_sign',
                'risk_multiplier', 'tradeable_capital', 'mtm_exit_count',
                'forecast_prior_exposure', 'forecast_prior_target', 'forecast_prior_blend',
                'forecast_only_prior_exposure', 'forecast_beta_exposure',
                'forecast_beta_strength', 'forecast_beta_active',
                'forecast_beta_payoff_lcb', 'forecast_beta_count',
                'forecast_prior_active', 'forecast_prior_skill', 'forecast_prior_hit_lcb',
                'forecast_prior_hit_rate', 'forecast_prior_residual_q', 'forecast_prior_magnitude',
                'forecast_prior_edge_excess', 'forecast_prior_direction_confidence',
                'forecast_prior_confidence_weight', 'forecast_prior_calibration_count',
                'forecast_prior_calibration_total',
                'forecast_residual_authority', 'forecast_base_residual_authority',
                'forecast_residual_evidence_multiplier', 'forecast_residual_to_prior_ratio',
                'forecast_tail_cap_abs', 'forecast_tail_return',
                'forecast_tail_empirical_return', 'forecast_tail_count',
                'forecast_tail_budget_dkk',
                'distributional_bucket_count', 'distributional_global_count',
                'distributional_using_bucket', 'distributional_edge_mean',
                'distributional_edge_shrunk', 'distributional_edge_strength',
                'distributional_confidence_weight',
                'distributional_conditional_tail', 'distributional_disaster_tail',
                'distributional_conditional_cap_abs', 'distributional_disaster_cap_abs',
                'distributional_conditional_budget_dkk', 'distributional_disaster_budget_dkk',
                'feasible_action_enabled',
                'feasible_action_coordinate',
                'feasible_action_capacity_mwh',
                'feasible_action_capacity_fraction',
                'feasible_action_hard_capacity_mwh',
                'feasible_action_hard_capacity_fraction',
                'feasible_action_evidence_authority',
                'feasible_action_capacity_conditioned_on_evidence',
                'feasible_action_anchor_requested_mwh',
                'feasible_action_anchor_quantity_mwh',
                'feasible_action_policy_requested_mwh',
                'feasible_action_policy_quantity_mwh',
                'feasible_action_adjustment_mwh',
                'feasible_action_anchor_fraction',
                'feasible_action_target_fraction',
                'feasible_action_anchor_no_trade_hold',
                'feasible_action_policy_no_trade_hold',
                'feasible_action_liquidity_capacity_mwh',
                'feasible_action_liquidity_capacity_fraction',
                'feasible_action_collateral_headroom_fraction',
                'feasible_action_no_trade_threshold_mwh',
                'feasible_action_policy_cost_dkk',
                'feasible_action_anchor_cost_dkk',
                'feasible_action_policy_funding_cost_dkk',
                'feasible_action_anchor_funding_cost_dkk',
                'feasible_action_count',
                'feasible_action_advantage_mean',
                'feasible_action_cvar_shortfall',
                'feasible_action_dual_lambda',
                'feasible_action_last_advantage_dkk',
                'feasible_action_reward_step',
                'trading_sleeve_margin_active', 'trading_sleeve_margin_step',
                'trading_sleeve_margin_threshold_dkk', 'trading_sleeve_margin_gap_dkk',
                'forecast_prior_reason',
                'forecast_control_mode', 'forecast_mode_reason', 'forecast_policy_target',
                'forecast_counterfactual_policy_target',
                'forecast_alignment', 'forecast_strength', 'forecast_adjustment',
                'forecast_policy_hit_lcb', 'forecast_policy_relative_hit_lcb',
                'forecast_policy_relative_advantage_lcb',
                'forecast_policy_relative_strength', 'forecast_policy_relative_count',
                'investor_action', 'battery_action',
                'inv_mu_raw', 'inv_sigma_raw', 'inv_a_raw',
                'inv_action_mean', 'inv_action_sigma', 'inv_action_sample',
                'inv_tanh_mu', 'inv_tanh_a',
                'inv_mean_clip_hit', 'inv_mean_clip_hit_rate',
                'inv_mu_abs_roll', 'inv_mu_sign_consistency',
                'inv_sat_mean', 'inv_sat_sample', 'inv_sat_noise_only',
                'inv_reward_step', 'inv_value', 'inv_value_next', 'inv_td_error'
            ],
            'rewards': ['episode', 'timestep', 'base_reward', 'investor_reward', 'battery_reward',
                        'risk_score', 'operational_score']
        }
        
        # Track episode-level aggregates
        self.episode_data = []
        self.current_episode = 0
        self.category_fields['positions'] = [
            field for field in self.category_fields['positions']
            if field not in self.investor_health_fields
        ]

    @staticmethod
    def _strip_row_fields(row: dict, prefixes) -> None:
        for key in list(row.keys()):
            if any(key.startswith(prefix) for prefix in prefixes):
                row.pop(key, None)

    def _filter_main_fieldnames(self, fieldnames):
        return list(fieldnames)

    def _should_flush(self, timestep: int) -> bool:
        try:
            return (int(timestep) + 1) % int(self.flush_every_steps) == 0
        except Exception:
            return False

    def _flush_open_handles(self) -> None:
        try:
            if self.csv_file_handle is not None:
                self.csv_file_handle.flush()
        except Exception:
            pass
        for handle in self.category_handles.values():
            try:
                if handle is not None:
                    handle.flush()
            except Exception:
                pass
        for handle in self.agent_health_handles.values():
            try:
                if handle is not None:
                    handle.flush()
            except Exception:
                pass
        
    def start_episode(self, episode_num: int):
        """Initialize logging for a new episode - creates a new CSV file per episode"""
        if not self.enabled:
            self.current_episode = episode_num
            return

        # Close previous episode's file if open
        if self.csv_file_handle is not None:
            self.csv_file_handle.close()
            self.csv_file_handle = None
            self.csv_writer = None

        # Close previous category files
        for handle in self.category_handles.values():
            if handle is not None:
                handle.close()
        self.category_handles.clear()
        self.category_writers.clear()

        # Close previous agent health files
        for handle in self.agent_health_handles.values():
            if handle is not None:
                handle.close()
        self.agent_health_handles.clear()
        self.agent_health_writers.clear()

        self.current_episode = episode_num

        # Create new CSV file for this episode.
        # Desired behavior: re-running an episode overwrites its prior logs (e.g., after OOM/restart).
        self.csv_file = os.path.join(self.log_dir, f"{self.tier_name}_debug_ep{episode_num}.csv")

        try:
            # Always create new file (write mode) for each episode
            self.csv_file_handle = open(self.csv_file, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.DictWriter(self.csv_file_handle, fieldnames=self._get_fieldnames())
            self.csv_writer.writeheader()
            self.csv_file_handle.flush()
            logging.info(f"[LOGGER] Created new log file for episode {episode_num}: {self.csv_file}")

            # Create category-specific CSV files
            self._create_category_csvs(episode_num)

            # Create per-agent health CSV files (one per agent, written incrementally)
            _agent_health_specs = [
                ('investor', self.investor_health_all_fields),
                ('battery',  self.battery_health_all_fields),
                ('risk',     self.risk_health_all_fields),
                ('meta',     self.meta_health_all_fields),
            ]
            for agent_key, fields in _agent_health_specs:
                health_path = os.path.join(
                    self.log_dir,
                    f"{self.tier_name}_{agent_key}_health_ep{episode_num}.csv"
                )
                try:
                    h = open(health_path, 'w', newline='', encoding='utf-8')
                    w = csv.DictWriter(h, fieldnames=fields, extrasaction='ignore')
                    w.writeheader()
                    h.flush()
                    self.agent_health_handles[agent_key] = h
                    self.agent_health_writers[agent_key] = w
                    logging.info(f"[LOGGER] Created {agent_key} health log: {health_path}")
                except Exception as e:
                    logging.warning(f"[LOGGER] Could not create {agent_key} health CSV: {e}")
        except Exception as e:
            logging.exception(f"[LOGGER] ERROR creating log file {self.csv_file}: {e}")
    
    def _create_category_csvs(self, episode_num: int):
        """Create category-specific CSV files for better organization."""
        if not self.enabled:
            return
        try:
            for category, fields in self.category_fields.items():
                # Single canonical file name per episode/category.
                csv_file = os.path.join(self.log_dir, f"{self.tier_name}_{category}_ep{episode_num}.csv")

                try:
                    handle = open(csv_file, 'w', newline='', encoding='utf-8')
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    handle.flush()
                    self.category_handles[category] = handle
                    self.category_writers[category] = writer
                    logging.debug(f"[LOGGER] Created category CSV: {csv_file}")
                except Exception as e:
                    logging.warning(f"[LOGGER] Could not create {category} CSV: {e}")
        except Exception as e:
            logging.warning(f"[LOGGER] Could not create category CSVs: {e}")
    
    def log_step(self, 
                 timestep: int,
                 # Portfolio metrics (logged at every timestep)
                 portfolio_value_usd_millions: float = 0.0,
                 fund_nav_usd: float = 0.0,
                 cash_dkk: float = 0.0,
                 trading_sleeve_value_usd_thousands: float = 0.0,
                 trading_mtm_tracker_usd_thousands: float = 0.0,
                 operating_revenue_usd_thousands: float = 0.0,
                # Fund NAV component breakdown (DKK)
                 fund_nav_dkk: float = 0.0,
                 distribution_adjusted_nav_dkk: float = 0.0,
                 total_distributions_dkk: float = 0.0,
                 trading_cash_dkk: float = 0.0,
                 trading_sleeve_value_dkk: float = 0.0,
                 distribution_adjusted_trading_sleeve_dkk: float = 0.0,
                 physical_book_value_dkk: float = 0.0,
                 accumulated_operational_revenue_dkk: float = 0.0,
                 financial_mtm_dkk: float = 0.0,
                 financial_exposure_dkk: float = 0.0,
                 cumulative_trading_costs_dkk: float = 0.0,
                 cumulative_impact_costs_dkk: float = 0.0,
                 cumulative_volume_transaction_fees_dkk: float = 0.0,
                 cumulative_market_access_fees_dkk: float = 0.0,
                 market_access_fee_step_dkk: float = 0.0,
                 horizon_settlement_pnl_dkk: float = 0.0,
                 horizon_settlement_count: float = 0.0,
                 depreciation_ratio: float = 0.0,
                 years_elapsed: float = 0.0,
                 # Position info
                 position_signed: float = 0.0,
                 held_exposure_signed: float = 0.0,
                 position_exposure: float = 0.0,
                 decision_step: float = 0.0,
                 exposure_exec: float = 0.0,
                 action_sign: float = 0.0,
                 trade_signal_active: float = 0.0,
                 trade_signal_sign: float = 0.0,
                 risk_multiplier: float = 1.0,
                 tradeable_capital: float = 0.0,
                 mtm_exit_count: float = 0.0,
                 forecast_prior_exposure: float = 0.0,
                 forecast_only_prior_exposure: float = 0.0,
                 forecast_beta_exposure: float = 0.0,
                 forecast_beta_strength: float = 0.0,
                 forecast_beta_active: float = 0.0,
                 forecast_beta_payoff_lcb: float = 0.0,
                 forecast_beta_count: float = 0.0,
                 forecast_prior_target: float = 0.0,
                 forecast_prior_blend: float = 0.0,
                 forecast_prior_active: float = 0.0,
                 forecast_prior_skill: float = 0.0,
                 forecast_prior_hit_lcb: float = 0.5,
                 forecast_prior_hit_rate: float = 0.5,
                 forecast_prior_residual_q: float = 0.0,
                 forecast_prior_magnitude: float = 0.0,
                 forecast_prior_edge_excess: float = 0.0,
                 forecast_prior_direction_confidence: float = 0.5,
                 forecast_prior_confidence_weight: float = 0.0,
                 forecast_prior_calibration_count: float = 0.0,
                 forecast_prior_calibration_total: float = 0.0,
                 forecast_residual_authority: float = 0.0,
                 forecast_base_residual_authority: float = 0.0,
                 forecast_residual_evidence_multiplier: float = 1.0,
                 forecast_residual_to_prior_ratio: float = 0.0,
                 forecast_tail_cap_abs: float = 1.0,
                 forecast_tail_return: float = 0.0,
                 forecast_tail_empirical_return: float = 0.0,
                 forecast_tail_count: float = 0.0,
                 forecast_tail_budget_dkk: float = 0.0,
                 distributional_bucket_count: float = 0.0,
                 distributional_global_count: float = 0.0,
                 distributional_using_bucket: float = 0.0,
                 distributional_edge_mean: float = 0.0,
                 distributional_edge_shrunk: float = 0.0,
                 distributional_edge_strength: float = 0.0,
                 distributional_confidence_weight: float = 0.0,
                 distributional_conditional_tail: float = 0.0,
                 distributional_disaster_tail: float = 0.0,
                 distributional_conditional_cap_abs: float = 0.0,
                 distributional_disaster_cap_abs: float = 0.0,
                 distributional_conditional_budget_dkk: float = 0.0,
                 distributional_disaster_budget_dkk: float = 0.0,
                 feasible_action_enabled: float = 0.0,
                 feasible_action_coordinate: float = 0.0,
                 feasible_action_capacity_mwh: float = 0.0,
                 feasible_action_capacity_fraction: float = 0.0,
                 feasible_action_hard_capacity_mwh: float = 0.0,
                 feasible_action_hard_capacity_fraction: float = 0.0,
                 feasible_action_evidence_authority: float = 0.0,
                 feasible_action_capacity_conditioned_on_evidence: float = 0.0,
                 feasible_action_anchor_requested_mwh: float = 0.0,
                 feasible_action_anchor_quantity_mwh: float = 0.0,
                 feasible_action_policy_requested_mwh: float = 0.0,
                 feasible_action_policy_quantity_mwh: float = 0.0,
                 feasible_action_adjustment_mwh: float = 0.0,
                 feasible_action_anchor_fraction: float = 0.0,
                 feasible_action_target_fraction: float = 0.0,
                 feasible_action_anchor_no_trade_hold: float = 0.0,
                 feasible_action_policy_no_trade_hold: float = 0.0,
                 feasible_action_liquidity_capacity_mwh: float = 0.0,
                 feasible_action_liquidity_capacity_fraction: float = 0.0,
                 feasible_action_collateral_headroom_fraction: float = 0.0,
                 feasible_action_no_trade_threshold_mwh: float = 0.0,
                 feasible_action_policy_cost_dkk: float = 0.0,
                 feasible_action_anchor_cost_dkk: float = 0.0,
                 feasible_action_policy_funding_cost_dkk: float = 0.0,
                 feasible_action_anchor_funding_cost_dkk: float = 0.0,
                 feasible_action_count: float = 0.0,
                 feasible_action_advantage_mean: float = 0.0,
                 feasible_action_cvar_shortfall: float = 0.0,
                 feasible_action_dual_lambda: float = 0.0,
                 feasible_action_last_advantage_dkk: float = 0.0,
                 feasible_action_reward_step: float = 0.0,
                 forecast_mirror_enabled: float = 0.0,
                 forecast_mirror_action: float = 0.0,
                 forecast_mirror_requested_quantity_mwh: float = 0.0,
                 forecast_mirror_executed_quantity_mwh: float = 0.0,
                 forecast_mirror_hard_capacity_mwh: float = 0.0,
                 forecast_mirror_prior_quantity_mwh: float = 0.0,
                 forecast_mirror_prior_coordinate: float = 0.0,
                 forecast_mirror_evidence: float = 0.0,
                 forecast_mirror_trust: float = 0.0,
                 forecast_mirror_exponent: float = 0.0,
                 forecast_mirror_base_action_mean: float = 0.0,
                 forecast_mirror_final_action_mean: float = 0.0,
                 forecast_mirror_trust_gradient: float = 0.0,
                 forecast_mirror_counterfactual_score: float = 0.0,
                 forecast_mirror_pending_count: float = 0.0,
                 trading_sleeve_margin_active: float = 0.0,
                 trading_sleeve_margin_step: float = -1.0,
                 trading_sleeve_margin_threshold_dkk: float = 0.0,
                 trading_sleeve_margin_gap_dkk: float = 0.0,
                 forecast_prior_reason: str = '',
                 forecast_control_mode: str = '',
                 forecast_mode_reason: str = '',
                 forecast_policy_target: float = 0.0,
                 forecast_counterfactual_policy_target: float = 0.0,
                 forecast_alignment: float = 0.0,
                 forecast_strength: float = 0.0,
                 forecast_adjustment: float = 0.0,
                 forecast_policy_hit_lcb: float = 0.5,
                 forecast_policy_relative_hit_lcb: float = 0.5,
                 forecast_policy_relative_advantage_lcb: float = 0.0,
                 forecast_policy_relative_strength: float = 0.0,
                 forecast_policy_relative_count: float = 0.0,
                 # Price data (for forward-looking accuracy analysis)
                 price_current: float = 0.0,  # Raw price at current timestep (DKK/MWh)
                 # Price returns
                 price_return_1step: float = 0.0,
                 price_return_invfreq: float = 0.0,
                 training_return: float = 0.0,
                 training_horizon_steps: float = 0.0,
                 # Main reward components
                 operational_score: float = 0.0,
                 risk_score: float = 0.0,
                 hedging_score: float = 0.0,
                 nav_stability_score: float = 0.0,
                 # Reward weights
                 weight_operational: float = 0.0,
                 weight_risk: float = 0.0,
                 weight_hedging: float = 0.0,
                 weight_nav: float = 0.0,
                 # Final rewards
                 base_reward: float = 0.0,
                 investor_reward: float = 0.0,
                 battery_reward: float = 0.0,
                 # Financial metrics
                 mtm_pnl: float = 0.0,  # Per-step MTM PnL (not cumulative)
                 # Actions
                 investor_action: Optional[Dict] = None,
                 battery_action: Optional[Dict] = None,
                 # Reward breakdown debugging
                 unrealized_pnl_norm: float = 0.0,
                 combined_confidence: float = 0.0,
                 adaptive_multiplier: float = 0.0,
                 # Position breakdown
                 wind_pos_norm: float = 0.0,
                 solar_pos_norm: float = 0.0,
                 hydro_pos_norm: float = 0.0,
                 # NEW: Position alignment diagnostics (Bug #1 fix tracking)
                 position_alignment_status: str = 'no_strategy',
                 investor_position_ratio: float = 0.0,
                 investor_position_direction: int = 0,
                 investor_total_position: float = 0.0,
                 investor_exploration_bonus: float = 0.0,
                 # NEW: Price floor diagnostics (Bug #4 fix tracking)
                 price_raw: float = 0.0,
                 price_floor_used: float = 0.0,
                 # NEW: NAV attribution drivers (per-step fund NAV breakdown)
                 nav_start: float = 0.0,
                 nav_end: float = 0.0,
                 pnl_total: float = 0.0,
                 pnl_battery: float = 0.0,
                 pnl_generation: float = 0.0,
                 pnl_hedge: float = 0.0,
                 cash_delta_ops: float = 0.0,
                 # NEW: Battery dispatch metrics (FIX #5)
                 battery_decision: str = 'idle',
                 battery_intensity: float = 0.0,
                 battery_spread: float = 0.0,
                 battery_adjusted_hurdle: float = 0.0,
                 battery_volatility_adj: float = 0.0,
                 # NEW: Battery state metrics (BATTERY REWARD FIX)
                 battery_energy: float = 0.0,
                 battery_capacity: float = 0.0,
                 battery_soc: float = 0.0,
                 battery_cash_delta: float = 0.0,
                 battery_throughput: float = 0.0,
                 battery_degradation_cost: float = 0.0,
                 battery_eta_charge: float = 0.0,
                 battery_eta_discharge: float = 0.0,
                 # Investor health diagnostics
                 training_global_step: int = 0,
                 inv_mean_clip_hit: float = 0.0,
                 inv_mean_clip_hit_rate: float = 0.0,
                 inv_mu_abs_roll: float = 0.0,
                 inv_mu_sign_consistency: float = 0.0,
                 inv_mu_raw: float = 0.0,
                 inv_sigma_raw: float = 0.0,
                 inv_a_raw: float = 0.0,
                 inv_action_mean: float = 0.0,
                 inv_action_sigma: float = 0.0,
                 inv_action_sample: float = 0.0,
                 inv_tanh_mu: float = 0.0,
                 inv_tanh_a: float = 0.0,
                 inv_sat_mean: float = 0.0,
                 inv_sat_sample: float = 0.0,
                 inv_sat_noise_only: float = 0.0,
                 inv_reward_step: float = 0.0,
                 inv_value: float = 0.0,
                 inv_value_next: float = 0.0,
                 inv_td_error: float = 0.0,
                 # Battery operator health diagnostics
                 bat_decision_step: float = 0.0,
                 bat_action_idx: int = 0,
                 bat_q_max: float = 0.0,
                 bat_q_chosen: float = 0.0,
                 bat_reward_step: float = 0.0,
                 bat_soc: float = 0.0,
                 bat_spread: float = 0.0,
                 bat_energy: float = 0.0,
                 bat_capacity: float = 0.0,
                 # Risk controller health diagnostics
                 risk_decision_step: float = 0.0,
                 risk_action_raw: float = 0.0,
                 risk_reward_step: float = 0.0,
                 risk_value: float = 0.0,
                 risk_value_next: float = 0.0,
                 risk_td_error: float = 0.0,
                 risk_drawdown: float = 0.0,
                 overall_risk_snapshot: float = 0.0,
                 # Meta controller health diagnostics
                 meta_decision_step: float = 0.0,
                 meta_action_raw: float = 0.0,
                 meta_reward_step: float = 0.0,
                 meta_value: float = 0.0,
                 meta_value_next: float = 0.0,
                 meta_td_error: float = 0.0,
                 meta_cap_fraction: float = 0.0,
                 meta_budget_n: float = 0.0,
                 **extra_fields: Any):
        """Log detailed step information"""
        if not self.enabled:
            return

        # DEBUG to verify log_step is being called
        if timestep % 100 == 0:
            logging.debug(f"[LOGGER] log_step() called at timestep={timestep}")

        # Auto-initialize if not already done (safety check)
        if self.csv_writer is None:
            logging.warning(f"[LOGGER] csv_writer is NONE at timestep={timestep}, initializing...")
            self.start_episode(self.current_episode if self.current_episode > 0 else 0)

        if self.csv_writer is None:
            logging.error(f"[LOGGER] csv_writer STILL NONE after initialization at timestep={timestep}, SKIPPING!")
            return  # Still None after initialization attempt - skip logging

        # DEBUG to verify we're about to write
        if timestep % 100 == 0:
            logging.debug(f"[LOGGER] csv_writer EXISTS, writing row for timestep={timestep}")

        row = {
            # ===================================================================
            # ===================================================================
            # Portfolio metrics (logged at every timestep)
            'episode': self.current_episode,
            'timestep': timestep,
            'portfolio_value_usd_millions': portfolio_value_usd_millions,
            'fund_nav_usd': fund_nav_usd,
            'cash_dkk': cash_dkk,
            'trading_sleeve_value_usd_thousands': trading_sleeve_value_usd_thousands,
            'trading_mtm_tracker_usd_thousands': trading_mtm_tracker_usd_thousands,
            'operating_revenue_usd_thousands': operating_revenue_usd_thousands,
            # Primary portfolio metrics (mtm_pnl is per-step, not cumulative)
            'mtm_pnl': mtm_pnl,
            # Fund NAV component breakdown (DKK)
            'fund_nav_dkk': fund_nav_dkk,
            'distribution_adjusted_nav_dkk': distribution_adjusted_nav_dkk,
            'total_distributions_dkk': total_distributions_dkk,
            'trading_cash_dkk': trading_cash_dkk,
            'trading_sleeve_value_dkk': trading_sleeve_value_dkk,
            'distribution_adjusted_trading_sleeve_dkk': distribution_adjusted_trading_sleeve_dkk,
            'physical_book_value_dkk': physical_book_value_dkk,
            'accumulated_operational_revenue_dkk': accumulated_operational_revenue_dkk,
            'financial_mtm_dkk': financial_mtm_dkk,
            'financial_exposure_dkk': financial_exposure_dkk,
            'cumulative_trading_costs_dkk': cumulative_trading_costs_dkk,
            'cumulative_impact_costs_dkk': cumulative_impact_costs_dkk,
            'cumulative_volume_transaction_fees_dkk': cumulative_volume_transaction_fees_dkk,
            'cumulative_market_access_fees_dkk': cumulative_market_access_fees_dkk,
            'market_access_fee_step_dkk': market_access_fee_step_dkk,
            'horizon_settlement_pnl_dkk': horizon_settlement_pnl_dkk,
            'horizon_settlement_count': horizon_settlement_count,
            'depreciation_ratio': depreciation_ratio,
            'years_elapsed': years_elapsed,
            # Position info
            'position_signed': position_signed,
            'held_exposure_signed': held_exposure_signed,
            'position_exposure': position_exposure,
            'decision_step': decision_step,
            'exposure_exec': exposure_exec,
            'action_sign': action_sign,
            'trade_signal_active': trade_signal_active,
            'trade_signal_sign': trade_signal_sign,
            'risk_multiplier': risk_multiplier,
            'tradeable_capital': tradeable_capital,
            'mtm_exit_count': mtm_exit_count,
            'forecast_prior_exposure': forecast_prior_exposure,
            'forecast_only_prior_exposure': forecast_only_prior_exposure,
            'forecast_beta_exposure': forecast_beta_exposure,
            'forecast_beta_strength': forecast_beta_strength,
            'forecast_beta_active': forecast_beta_active,
            'forecast_beta_payoff_lcb': forecast_beta_payoff_lcb,
            'forecast_beta_count': forecast_beta_count,
            'forecast_prior_target': forecast_prior_target,
            'forecast_prior_blend': forecast_prior_blend,
            'forecast_prior_active': forecast_prior_active,
            'forecast_prior_skill': forecast_prior_skill,
            'forecast_prior_hit_lcb': forecast_prior_hit_lcb,
            'forecast_prior_hit_rate': forecast_prior_hit_rate,
            'forecast_prior_residual_q': forecast_prior_residual_q,
            'forecast_prior_magnitude': forecast_prior_magnitude,
            'forecast_prior_edge_excess': forecast_prior_edge_excess,
            'forecast_prior_direction_confidence': forecast_prior_direction_confidence,
            'forecast_prior_confidence_weight': forecast_prior_confidence_weight,
            'forecast_prior_calibration_count': forecast_prior_calibration_count,
            'forecast_prior_calibration_total': forecast_prior_calibration_total,
            'forecast_residual_authority': forecast_residual_authority,
            'forecast_base_residual_authority': forecast_base_residual_authority,
            'forecast_residual_evidence_multiplier': forecast_residual_evidence_multiplier,
            'forecast_residual_to_prior_ratio': forecast_residual_to_prior_ratio,
            'forecast_tail_cap_abs': forecast_tail_cap_abs,
            'forecast_tail_return': forecast_tail_return,
            'forecast_tail_empirical_return': forecast_tail_empirical_return,
            'forecast_tail_count': forecast_tail_count,
            'forecast_tail_budget_dkk': forecast_tail_budget_dkk,
            'distributional_bucket_count': distributional_bucket_count,
            'distributional_global_count': distributional_global_count,
            'distributional_using_bucket': distributional_using_bucket,
            'distributional_edge_mean': distributional_edge_mean,
            'distributional_edge_shrunk': distributional_edge_shrunk,
            'distributional_edge_strength': distributional_edge_strength,
            'distributional_confidence_weight': distributional_confidence_weight,
            'distributional_conditional_tail': distributional_conditional_tail,
            'distributional_disaster_tail': distributional_disaster_tail,
            'distributional_conditional_cap_abs': distributional_conditional_cap_abs,
            'distributional_disaster_cap_abs': distributional_disaster_cap_abs,
            'distributional_conditional_budget_dkk': distributional_conditional_budget_dkk,
            'distributional_disaster_budget_dkk': distributional_disaster_budget_dkk,
            'feasible_action_enabled': feasible_action_enabled,
            'feasible_action_coordinate': feasible_action_coordinate,
            'feasible_action_capacity_mwh': feasible_action_capacity_mwh,
            'feasible_action_capacity_fraction': feasible_action_capacity_fraction,
            'feasible_action_hard_capacity_mwh': feasible_action_hard_capacity_mwh,
            'feasible_action_hard_capacity_fraction': feasible_action_hard_capacity_fraction,
            'feasible_action_evidence_authority': feasible_action_evidence_authority,
            'feasible_action_capacity_conditioned_on_evidence': feasible_action_capacity_conditioned_on_evidence,
            'feasible_action_anchor_requested_mwh': feasible_action_anchor_requested_mwh,
            'feasible_action_anchor_quantity_mwh': feasible_action_anchor_quantity_mwh,
            'feasible_action_policy_requested_mwh': feasible_action_policy_requested_mwh,
            'feasible_action_policy_quantity_mwh': feasible_action_policy_quantity_mwh,
            'feasible_action_adjustment_mwh': feasible_action_adjustment_mwh,
            'feasible_action_anchor_fraction': feasible_action_anchor_fraction,
            'feasible_action_target_fraction': feasible_action_target_fraction,
            'feasible_action_anchor_no_trade_hold': feasible_action_anchor_no_trade_hold,
            'feasible_action_policy_no_trade_hold': feasible_action_policy_no_trade_hold,
            'feasible_action_liquidity_capacity_mwh': feasible_action_liquidity_capacity_mwh,
            'feasible_action_liquidity_capacity_fraction': feasible_action_liquidity_capacity_fraction,
            'feasible_action_collateral_headroom_fraction': feasible_action_collateral_headroom_fraction,
            'feasible_action_no_trade_threshold_mwh': feasible_action_no_trade_threshold_mwh,
            'feasible_action_policy_cost_dkk': feasible_action_policy_cost_dkk,
            'feasible_action_anchor_cost_dkk': feasible_action_anchor_cost_dkk,
            'feasible_action_policy_funding_cost_dkk': feasible_action_policy_funding_cost_dkk,
            'feasible_action_anchor_funding_cost_dkk': feasible_action_anchor_funding_cost_dkk,
            'feasible_action_count': feasible_action_count,
            'feasible_action_advantage_mean': feasible_action_advantage_mean,
            'feasible_action_cvar_shortfall': feasible_action_cvar_shortfall,
            'feasible_action_dual_lambda': feasible_action_dual_lambda,
            'feasible_action_last_advantage_dkk': feasible_action_last_advantage_dkk,
            'feasible_action_reward_step': feasible_action_reward_step,
            'forecast_mirror_enabled': forecast_mirror_enabled,
            'forecast_mirror_action': forecast_mirror_action,
            'forecast_mirror_requested_quantity_mwh': forecast_mirror_requested_quantity_mwh,
            'forecast_mirror_executed_quantity_mwh': forecast_mirror_executed_quantity_mwh,
            'forecast_mirror_hard_capacity_mwh': forecast_mirror_hard_capacity_mwh,
            'forecast_mirror_prior_quantity_mwh': forecast_mirror_prior_quantity_mwh,
            'forecast_mirror_prior_coordinate': forecast_mirror_prior_coordinate,
            'forecast_mirror_evidence': forecast_mirror_evidence,
            'forecast_mirror_trust': forecast_mirror_trust,
            'forecast_mirror_exponent': forecast_mirror_exponent,
            'forecast_mirror_base_action_mean': forecast_mirror_base_action_mean,
            'forecast_mirror_final_action_mean': forecast_mirror_final_action_mean,
            'forecast_mirror_trust_gradient': forecast_mirror_trust_gradient,
            'forecast_mirror_counterfactual_score': forecast_mirror_counterfactual_score,
            'forecast_mirror_pending_count': forecast_mirror_pending_count,
            'trading_sleeve_margin_active': trading_sleeve_margin_active,
            'trading_sleeve_margin_step': trading_sleeve_margin_step,
            'trading_sleeve_margin_threshold_dkk': trading_sleeve_margin_threshold_dkk,
            'trading_sleeve_margin_gap_dkk': trading_sleeve_margin_gap_dkk,
            'forecast_prior_reason': forecast_prior_reason,
            'forecast_control_mode': forecast_control_mode,
            'forecast_mode_reason': forecast_mode_reason,
            'forecast_policy_target': forecast_policy_target,
            'forecast_counterfactual_policy_target': forecast_counterfactual_policy_target,
            'forecast_alignment': forecast_alignment,
            'forecast_strength': forecast_strength,
            'forecast_adjustment': forecast_adjustment,
            'forecast_policy_hit_lcb': forecast_policy_hit_lcb,
            'forecast_policy_relative_hit_lcb': forecast_policy_relative_hit_lcb,
            'forecast_policy_relative_advantage_lcb': forecast_policy_relative_advantage_lcb,
            'forecast_policy_relative_strength': forecast_policy_relative_strength,
            'forecast_policy_relative_count': forecast_policy_relative_count,
            # Price data (for forward-looking accuracy analysis)
            'price_current': price_current,  # Raw price at current timestep (DKK/MWh)
            # Price returns
            'price_return_1step': price_return_1step,
            'price_return_invfreq': price_return_invfreq,
            'training_return': training_return,
            'training_horizon_steps': training_horizon_steps,
            # Main reward components
            'operational_score': operational_score,
            'risk_score': risk_score,
            'hedging_score': hedging_score,
            'nav_stability_score': nav_stability_score,
            # Reward weights
            'weight_operational': weight_operational,
            'weight_risk': weight_risk,
            'weight_hedging': weight_hedging,
            'weight_nav': weight_nav,
            # Final rewards
            'base_reward': base_reward,
            'investor_reward': investor_reward,
            'battery_reward': battery_reward,
            # Investor health diagnostics
            'training_global_step': int(training_global_step),
            'inv_mean_clip_hit': float(inv_mean_clip_hit),
            'inv_mean_clip_hit_rate': float(inv_mean_clip_hit_rate),
            'inv_mu_abs_roll': float(inv_mu_abs_roll),
            'inv_mu_sign_consistency': float(inv_mu_sign_consistency),
        }


        row.update({
            # Position breakdown
            'wind_pos_norm': wind_pos_norm,
            'solar_pos_norm': solar_pos_norm,
            'hydro_pos_norm': hydro_pos_norm,
            # Reward breakdown debugging
            'unrealized_pnl_norm': unrealized_pnl_norm,
            'combined_confidence': combined_confidence,
            'adaptive_multiplier': adaptive_multiplier,
            
            # Forecast utilization is represented only by forecast_prior_* fields.
            # NEW: Position alignment diagnostics (Bug #1 fix tracking)
            'position_alignment_status': position_alignment_status,
            'investor_position_ratio': investor_position_ratio,
            'investor_position_direction': investor_position_direction,
            'investor_total_position': investor_total_position,
            'investor_exploration_bonus': investor_exploration_bonus,
            # NEW: Price floor diagnostics (Bug #4 fix tracking)
            'price_raw': price_raw,
            'price_floor_used': price_floor_used,
            # NEW: NAV attribution drivers (per-step fund NAV breakdown)
            'nav_start': nav_start,
            'nav_end': nav_end,
            'pnl_total': pnl_total,
            'pnl_battery': pnl_battery,
            'pnl_generation': pnl_generation,
            'pnl_hedge': pnl_hedge,
            'cash_delta_ops': cash_delta_ops,
            # NEW: Battery dispatch metrics (FIX #5)
            'battery_decision': battery_decision,
            'battery_intensity': battery_intensity,
            'battery_spread': battery_spread,
            'battery_adjusted_hurdle': battery_adjusted_hurdle,
            'battery_volatility_adj': battery_volatility_adj,
            # NEW: Battery state metrics (BATTERY REWARD FIX)
            'battery_energy': battery_energy,
            'battery_capacity': battery_capacity,
            'battery_soc': battery_soc,
            'battery_cash_delta': battery_cash_delta,
            'battery_throughput': battery_throughput,
            'battery_degradation_cost': battery_degradation_cost,
            'battery_eta_charge': battery_eta_charge,
            'battery_eta_discharge': battery_eta_discharge,
            # Investor policy diagnostics: native distribution stats plus
            # deployed action-space mean/sample.
            'inv_mu_raw': inv_mu_raw,
            'inv_sigma_raw': inv_sigma_raw,
            'inv_a_raw': inv_a_raw,
            'inv_action_mean': inv_action_mean,
            'inv_action_sigma': inv_action_sigma,
            'inv_action_sample': inv_action_sample,
            'inv_tanh_mu': inv_tanh_mu,
            'inv_tanh_a': inv_tanh_a,
            'inv_sat_mean': inv_sat_mean,
            'inv_sat_sample': inv_sat_sample,
            'inv_sat_noise_only': inv_sat_noise_only,
            # Investor TD-error proxy (credit assignment diagnostics)
            'inv_reward_step': inv_reward_step,
            'inv_value': inv_value,
            'inv_value_next': inv_value_next,
            'inv_td_error': inv_td_error,
            # Battery operator health diagnostics
            'bat_decision_step': bat_decision_step,
            'bat_action_idx': int(bat_action_idx),
            'bat_q_max': bat_q_max,
            'bat_q_chosen': bat_q_chosen,
            'bat_reward_step': bat_reward_step,
            'bat_soc': bat_soc,
            'bat_spread': bat_spread,
            'bat_energy': bat_energy,
            'bat_capacity': bat_capacity,
            # Risk controller health diagnostics
            'risk_decision_step': risk_decision_step,
            'risk_action_raw': risk_action_raw,
            'risk_reward_step': risk_reward_step,
            'risk_value': risk_value,
            'risk_value_next': risk_value_next,
            'risk_td_error': risk_td_error,
            'risk_drawdown': risk_drawdown,
            'overall_risk_snapshot': overall_risk_snapshot,
            # Meta controller health diagnostics
            'meta_decision_step': meta_decision_step,
            'meta_action_raw': meta_action_raw,
            'meta_reward_step': meta_reward_step,
            'meta_value': meta_value,
            'meta_value_next': meta_value_next,
            'meta_td_error': meta_td_error,
            'meta_cap_fraction': meta_cap_fraction,
            'meta_budget_n': meta_budget_n,
        })
        
        # Add action info if available
        # CRITICAL FIX: Handle arrays and dicts properly to avoid boolean ambiguity
        def safe_get_action_value(action_dict, key, default=0.0):
            """Safely extract action value, handling arrays and dicts"""
            if action_dict is None:
                return default
            if not isinstance(action_dict, dict):
                return default
            val = action_dict.get(key, default)
            # Handle numpy arrays
            if isinstance(val, np.ndarray):
                return float(val.item() if val.size == 1 else val.flatten()[0])
            return float(val) if val is not None else default

        def serialize_action(action_dict):
            """Serialize an action dict (possibly with numpy values) into JSON for CSV logs."""
            if action_dict is None or not isinstance(action_dict, dict) or len(action_dict) == 0:
                return ''

            def _normalize(value):
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        return float(value.item())
                    return value.astype(float).tolist()
                if isinstance(value, (np.generic,)):
                    return float(value)
                if isinstance(value, dict):
                    return {k: _normalize(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [_normalize(v) for v in value]
                return float(value) if isinstance(value, (int, float)) else value

            normalized = {k: _normalize(v) for k, v in action_dict.items()}
            try:
                return json.dumps(normalized, separators=(',', ':'))
            except (TypeError, ValueError):
                return str(normalized)
        
        if investor_action is not None and isinstance(investor_action, dict):
            row['inv_wind'] = safe_get_action_value(investor_action, 'wind', 0.0)
            row['inv_solar'] = safe_get_action_value(investor_action, 'solar', 0.0)
            row['inv_hydro'] = safe_get_action_value(investor_action, 'hydro', 0.0)
        else:
            row['inv_wind'] = 0.0
            row['inv_solar'] = 0.0
            row['inv_hydro'] = 0.0
            
        if battery_action is not None and isinstance(battery_action, dict):
            row['batt_charge'] = safe_get_action_value(battery_action, 'charge', 0.0)
            row['batt_discharge'] = safe_get_action_value(battery_action, 'discharge', 0.0)
        else:
            row['batt_charge'] = 0.0
            row['batt_discharge'] = 0.0

        # Preserve raw JSON actions for category-specific logs
        row['investor_action'] = serialize_action(investor_action)
        row['battery_action'] = serialize_action(battery_action)

        investor_health_row = {
            field: row.get(field, '')
            for field in self.investor_health_context_fields
            if field in row
        }
        for field in self.investor_health_fields:
            if field in row:
                investor_health_row[field] = row.pop(field, '')
        if investor_health_row:
            _w = self.agent_health_writers.get('investor')
            if _w is not None:
                try:
                    _w.writerow(investor_health_row)
                except Exception:
                    pass

        # Battery health row
        battery_health_row = {
            field: row.get(field, '')
            for field in self.battery_health_context_fields
            if field in row
        }
        for field in self.battery_health_fields:
            if field in row:
                battery_health_row[field] = row.pop(field, '')
        # Pop agent-specific context fields that don't belong in the main CSV
        for _f in ('bat_decision_step', 'bat_soc', 'bat_spread', 'bat_energy', 'bat_capacity'):
            row.pop(_f, None)
        if battery_health_row:
            _w = self.agent_health_writers.get('battery')
            if _w is not None:
                try:
                    _w.writerow(battery_health_row)
                except Exception:
                    pass

        # Risk health row
        risk_health_row = {
            field: row.get(field, '')
            for field in self.risk_health_context_fields
            if field in row
        }
        for field in self.risk_health_fields:
            if field in row:
                risk_health_row[field] = row.pop(field, '')
        for _f in ('risk_decision_step', 'risk_drawdown', 'overall_risk_snapshot'):
            row.pop(_f, None)
        if risk_health_row:
            _w = self.agent_health_writers.get('risk')
            if _w is not None:
                try:
                    _w.writerow(risk_health_row)
                except Exception:
                    pass

        # Meta health row
        meta_health_row = {
            field: row.get(field, '')
            for field in self.meta_health_context_fields
            if field in row
        }
        for field in self.meta_health_fields:
            if field in row:
                meta_health_row[field] = row.pop(field, '')
        for _f in ('meta_decision_step', 'meta_cap_fraction', 'meta_budget_n'):
            row.pop(_f, None)
        if meta_health_row:
            _w = self.agent_health_writers.get('meta')
            if _w is not None:
                try:
                    _w.writerow(meta_health_row)
                except Exception:
                    pass


        # Environment-specific financial diagnostics arrive through
        # **extra_fields and are retained when declared in the CSV schema.
        row.update(extra_fields)
        row_to_write = row
        try:
            writer_fieldnames = getattr(self.csv_writer, 'fieldnames', None)
            if writer_fieldnames:
                allowed_fields = set(writer_fieldnames)
                extra_fields = [
                    key for key in row.keys()
                    if key not in allowed_fields
                ]
                if extra_fields:
                    new_fields = [f for f in extra_fields if f not in self._warned_unknown_fields]
                    if new_fields:
                        logging.warning(f"[LOGGER] Dropping unknown columns: {new_fields}")
                        self._warned_unknown_fields.update(new_fields)
                row_to_write = {field: row.get(field, '') for field in writer_fieldnames}
        except Exception as filter_err:
            logging.debug(f"[LOGGER] Could not align row with fieldnames: {filter_err}")
            row_to_write = row

        try:
            self.csv_writer.writerow(row_to_write)
            if self._should_flush(timestep):
                self._flush_open_handles()
            # DEBUG to verify write succeeded
            if timestep % 100 == 0:
                logging.debug(f"[LOGGER] Successfully wrote row for timestep={timestep} to {self.csv_file}")
            
            # Write to category-specific CSVs if enabled
            self._write_to_category_csvs(row)
        except Exception as e:
            logging.exception(f"[LOGGER] ERROR writing to log file: {e}")
    
    def _write_to_category_csvs(self, row: dict):
        """Write row data to category-specific CSV files (only the fields defined for each category)."""
        try:
            for category, writer in self.category_writers.items():
                if writer is None:
                    continue

                if not self._is_valid_category_row(category, row):
                    continue

                handle = self.category_handles.get(category)
                if handle is not None:
                    # Extract only fields that belong to this category
                    try:
                        fields = self.category_fields.get(category, [])
                        filtered_row = {k: row.get(k, '') for k in fields if k in row or k in ['episode', 'timestep']}
                        writer.writerow(filtered_row)
                    except Exception as e:
                        logging.debug(f"[LOGGER] Could not write to {category} CSV: {e}")
        except Exception as e:
            logging.debug(f"[LOGGER] Error in _write_to_category_csvs: {e}")

    def _is_valid_category_row(self, category: str, row: dict) -> bool:
        episode = row.get('episode')
        timestep = row.get('timestep')

        if not self._is_valid_nonnegative_number(episode, max_value=100000):
            return False

        if not self._is_valid_nonnegative_number(timestep, max_value=1000000):
            return False

        if category == 'portfolio':
            portfolio_val = row.get('portfolio_value_usd_millions')
            if not self._is_valid_numeric_value(portfolio_val):
                return False

        return True

    @staticmethod
    def _is_valid_nonnegative_number(value, max_value=1e9) -> bool:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return False
        return 0.0 <= val <= max_value and np.isfinite(val)

    @staticmethod
    def _is_valid_numeric_value(value) -> bool:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return False
        return np.isfinite(val)
    
    def _get_fieldnames(self):
        """Get CSV field names - common columns first, forecast columns at the end"""
        fieldnames = [
            # ===================================================================
            # ===================================================================
            # Portfolio metrics (logged at every timestep)
            'episode', 'timestep',
            'portfolio_value_usd_millions', 'fund_nav_usd', 'cash_dkk',
            'trading_sleeve_value_usd_thousands',
            'trading_mtm_tracker_usd_thousands', 'operating_revenue_usd_thousands',
            # Primary portfolio metrics (mtm_pnl is per-step, not cumulative)
            'mtm_pnl',  # Mark-to-market PnL (per-step, not cumulative)
            # Fund NAV component breakdown (DKK)
            'fund_nav_dkk', 'trading_cash_dkk', 'trading_sleeve_value_dkk', 'physical_book_value_dkk',
            'accumulated_operational_revenue_dkk', 'financial_mtm_dkk', 'financial_exposure_dkk',
            'cumulative_trading_costs_dkk', 'cumulative_impact_costs_dkk',
            'cumulative_volume_transaction_fees_dkk',
            'cumulative_market_access_fees_dkk', 'market_access_fee_step_dkk',
            'liquidity_market_volume_mwh', 'liquidity_causal_market_volume_mwh',
            'liquidity_volume_cap_mwh',
            'liquidity_requested_volume_mwh', 'liquidity_executed_volume_mwh',
            'liquidity_participation', 'liquidity_scale',
            'liquidity_tail_impact_multiplier', 'liquidity_tail_spread_dkk_per_mwh',
            'market_impact_ref_notional_dkk', 'market_impact_participation', 'market_impact_bp',
            'no_trade_threshold_dkk', 'no_trade_threshold_reference_dkk',
            'collateral_required_dkk', 'collateral_funding_cost_dkk',
            'cumulative_collateral_funding_costs_dkk',
            'collateral_open_notional_dkk', 'collateral_open_volume_mwh',
            'horizon_settlement_pnl_dkk', 'horizon_settlement_count',
            'total_distributions_dkk', 'distribution_adjusted_nav_dkk',
            'distribution_adjusted_trading_sleeve_dkk',
            'depreciation_ratio', 'years_elapsed',
            # Position info (common to both tiers)
            'position_signed', 'held_exposure_signed', 'position_exposure', 'decision_step', 'exposure_exec', 'action_sign',
            'trade_signal_active', 'trade_signal_sign',
            'risk_multiplier', 'tradeable_capital', 'mtm_exit_count',
            'forecast_prior_exposure', 'forecast_prior_target', 'forecast_prior_blend',
            'forecast_only_prior_exposure', 'forecast_beta_exposure',
            'forecast_beta_strength', 'forecast_beta_active',
            'forecast_beta_payoff_lcb', 'forecast_beta_count',
            'forecast_prior_active', 'forecast_prior_skill', 'forecast_prior_hit_lcb',
            'forecast_prior_hit_rate', 'forecast_prior_residual_q', 'forecast_prior_magnitude',
            'forecast_prior_edge_excess', 'forecast_prior_direction_confidence',
            'forecast_prior_confidence_weight', 'forecast_prior_calibration_count',
            'forecast_prior_calibration_total',
            'forecast_residual_authority', 'forecast_base_residual_authority',
            'forecast_residual_evidence_multiplier', 'forecast_residual_to_prior_ratio',
            'forecast_tail_cap_abs', 'forecast_tail_return',
            'forecast_tail_empirical_return', 'forecast_tail_count',
            'forecast_tail_budget_dkk',
            'distributional_bucket_count', 'distributional_global_count',
            'distributional_using_bucket', 'distributional_edge_mean',
            'distributional_edge_shrunk', 'distributional_edge_strength',
            'distributional_confidence_weight',
            'distributional_conditional_tail', 'distributional_disaster_tail',
            'distributional_conditional_cap_abs', 'distributional_disaster_cap_abs',
            'distributional_conditional_budget_dkk', 'distributional_disaster_budget_dkk',
            'feasible_action_enabled',
            'feasible_action_coordinate',
            'feasible_action_capacity_mwh',
            'feasible_action_capacity_fraction',
            'feasible_action_hard_capacity_mwh',
            'feasible_action_hard_capacity_fraction',
            'feasible_action_evidence_authority',
            'feasible_action_capacity_conditioned_on_evidence',
            'feasible_action_anchor_requested_mwh',
            'feasible_action_anchor_quantity_mwh',
            'feasible_action_policy_requested_mwh',
            'feasible_action_policy_quantity_mwh',
            'feasible_action_adjustment_mwh',
            'feasible_action_anchor_fraction',
            'feasible_action_target_fraction',
            'feasible_action_anchor_no_trade_hold',
            'feasible_action_policy_no_trade_hold',
            'feasible_action_liquidity_capacity_mwh',
            'feasible_action_liquidity_capacity_fraction',
            'feasible_action_collateral_headroom_fraction',
            'feasible_action_no_trade_threshold_mwh',
            'feasible_action_policy_cost_dkk',
            'feasible_action_anchor_cost_dkk',
            'feasible_action_policy_funding_cost_dkk',
            'feasible_action_anchor_funding_cost_dkk',
            'feasible_action_count',
            'feasible_action_advantage_mean',
            'feasible_action_cvar_shortfall',
            'feasible_action_dual_lambda',
            'feasible_action_last_advantage_dkk',
            'feasible_action_reward_step',
            'forecast_mirror_enabled',
            'forecast_mirror_action',
            'forecast_mirror_requested_quantity_mwh',
            'forecast_mirror_executed_quantity_mwh',
            'forecast_mirror_hard_capacity_mwh',
            'forecast_mirror_prior_quantity_mwh',
            'forecast_mirror_prior_coordinate',
            'forecast_mirror_evidence',
            'forecast_mirror_trust',
            'forecast_mirror_exponent',
            'forecast_mirror_base_action_mean',
            'forecast_mirror_final_action_mean',
            'forecast_mirror_trust_gradient',
            'forecast_mirror_counterfactual_score',
            'forecast_mirror_pending_count',
            'trading_sleeve_margin_active', 'trading_sleeve_margin_step',
            'trading_sleeve_margin_threshold_dkk', 'trading_sleeve_margin_gap_dkk',
            'forecast_prior_reason',
            'forecast_control_mode', 'forecast_mode_reason', 'forecast_policy_target',
            'forecast_counterfactual_policy_target',
            'forecast_alignment', 'forecast_strength', 'forecast_adjustment',
            'forecast_policy_hit_lcb', 'forecast_policy_relative_hit_lcb',
            'forecast_policy_relative_advantage_lcb',
            'forecast_policy_relative_strength', 'forecast_policy_relative_count',
            # Price data (for forward-looking accuracy analysis)
            'price_current',  # Raw price at current timestep (DKK/MWh)
            # Price returns (common to both tiers)
            'price_return_1step', 'price_return_invfreq',
            'training_return', 'training_horizon_steps',
            # Main reward components (common to both tiers)
            'operational_score', 'risk_score', 'hedging_score', 'nav_stability_score',
            # Reward weights (common to both tiers, forecast weight will be 0.0 for Tier 1)
            'weight_operational', 'weight_risk', 'weight_hedging', 'weight_nav',
            # Final rewards (common to both tiers)
            'base_reward', 'investor_reward', 'battery_reward',
            # Investor health diagnostics
            'training_global_step',
            'inv_mean_clip_hit', 'inv_mean_clip_hit_rate',
            'inv_mu_abs_roll', 'inv_mu_sign_consistency',
            # MSCD: Multi-Signal Certified Deployment runtime gate
            # FCPRO persistence telemetry + FQDP forecast-quality decay
            # Reference-model diagnostics
            # Investor policy diagnostics: native distribution stats plus
            # deployed action-space mean/sample.
            'inv_mu_raw', 'inv_sigma_raw', 'inv_a_raw',
            'inv_action_mean', 'inv_action_sigma', 'inv_action_sample',
            'inv_tanh_mu', 'inv_tanh_a',
            'inv_sat_mean', 'inv_sat_sample', 'inv_sat_noise_only',
            # Investor TD-error proxy (credit assignment diagnostics)
            'inv_reward_step', 'inv_value', 'inv_value_next', 'inv_td_error',
            # Actions (common to both tiers)
            'inv_wind', 'inv_solar', 'inv_hydro', 'batt_charge', 'batt_discharge',
            'investor_action', 'battery_action',
            # Position breakdown (common to both tiers)
            'wind_pos_norm', 'solar_pos_norm', 'hydro_pos_norm',
            # Reward breakdown debugging (common to both tiers)
            'unrealized_pnl_norm', 'combined_confidence', 'adaptive_multiplier',
            
            # NEW: Position alignment diagnostics (Bug #1 fix tracking)
            'position_alignment_status', 'investor_position_ratio', 'investor_position_direction',
            'investor_total_position', 'investor_exploration_bonus',
            # NEW: Price floor diagnostics (Bug #4 fix tracking)
            'price_raw', 'price_floor_used',
            # NEW: NAV attribution drivers (per-step financial breakdown)
            'nav_start', 'nav_end', 'pnl_total', 'pnl_battery', 'pnl_generation', 'pnl_hedge', 'cash_delta_ops',
            # NEW: Battery dispatch metrics (FIX #5)
            'battery_decision', 'battery_intensity', 'battery_spread', 'battery_adjusted_hurdle', 'battery_volatility_adj',
            # NEW: Battery state metrics for diagnostics (BATTERY REWARD FIX)
            'battery_energy', 'battery_capacity', 'battery_soc',
            'battery_cash_delta', 'battery_throughput', 'battery_degradation_cost',
            'battery_eta_charge', 'battery_eta_discharge'
        ]

        # Preserve insertion order while removing duplicates
        deduped = []
        seen = set()
        for name in fieldnames:
            if name not in seen:
                deduped.append(name)
                seen.add(name)

        # Guarantee serialized action columns are present even if future edits remove them
        for required in ('investor_action', 'battery_action'):
            if required not in seen:
                deduped.append(required)
                seen.add(required)

        filtered = self._filter_main_fieldnames(deduped)
        all_health_fields = (
            set(self.investor_health_fields)
            | set(self.battery_health_fields)
            | set(self.risk_health_fields)
            | set(self.meta_health_fields)
        )
        # Also exclude the agent-specific context fields that belong only in health CSVs
        health_only_context = {
            'bat_decision_step', 'bat_soc', 'bat_spread', 'bat_energy', 'bat_capacity',
            'risk_decision_step', 'risk_drawdown', 'overall_risk_snapshot',
            'meta_decision_step', 'meta_cap_fraction', 'meta_budget_n',
        }
        return [name for name in filtered if name not in all_health_fields and name not in health_only_context]
    
    def close(self):
        """Close log files"""
        if self.csv_file_handle:
            self.csv_file_handle.close()
            self.csv_file_handle = None
            self.csv_writer = None
        for handle in self.category_handles.values():
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass
        self.category_handles.clear()
        self.category_writers.clear()
        for handle in self.agent_health_handles.values():
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass
        self.agent_health_handles.clear()
        self.agent_health_writers.clear()

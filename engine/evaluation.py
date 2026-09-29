#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation and Analysis Script
===========================================

Complete evaluation system that combines:
- Checkpoint-based evaluation (Stable Baselines3 models)
- Agent directory evaluation (MultiESGAgent system)
- Portfolio performance analysis
- Risk and economic model analysis
- Statistical confidence assessment
- Comprehensive plotting and reporting

Features:
- Automatic latest checkpoint detection
- Direct Stable Baselines3 model loading
- MultiESGAgent system loading
- Cache-only forecast utilization flag alignment
- Comprehensive metrics calculation
- Portfolio analysis with plotting
- Flexible data input (defaults to unseen evaluation dataset)
- JSON results and analysis reports

Usage:
    # Evaluate latest checkpoint (automatic unseen data)
    python evaluation.py --mode checkpoint

    # Evaluate specific agent directory
    python evaluation.py --mode agents --trained_agents saved_agents --eval_data "evaluation_dataset/unseendata.csv"

    # Baseline comparison
    python evaluation.py --mode agents --trained_agents saved_agents --eval_data "evaluation_dataset/unseendata.csv"

    # With comprehensive analysis and plots
    python evaluation.py --mode checkpoint --analyze --plot
"""

import argparse
import builtins as _builtins
import math
import os
import random
import sys
import re


def _force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_force_utf8_stdio()
import warnings
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from config import EnhancedConfig
from runtime_contract import (
    build_runtime_contract,
    controller_contract_settings,
    engine_file_hashes,
    execution_contract_settings,
    forecast_prior_contract_settings,
    mtm_contract_settings,
    runtime_contract_hash,
    sizing_contract_settings,
)
from forecast_prior_cli import (
    add_forecast_prior_override_args,
    apply_forecast_prior_overrides,
)

# Keep evaluation aligned with training-time rolling_past bootstrap.
EVAL_ROLLING_PAST_HISTORY_DIR = "rolling_past_history_dataset"


def _rule_based_checkpoint_model(agent_name: str) -> Dict[str, str]:
    return {"mode": "RULE", "agent_name": str(agent_name)}


def _is_rule_based_checkpoint_model(model: Any) -> bool:
    return isinstance(model, dict) and str(model.get("mode", "")).upper() == "RULE"


def setup_console_encoding():
    """
    Best-effort Windows console UTF-8 configuration.

    IMPORTANT: Do NOT detach/replace sys.stdout/sys.stderr at import time.
    Import-time I/O mutation breaks logging handlers in other modules.
    """
    if sys.platform != "win32":
        return
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        # Non-fatal; keep defaults.
        pass

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Clean ASCII overrides for the console helpers above.
def _sanitize_console_text(message: str) -> str:
    """Normalize evaluation console output to simple ASCII."""
    text = str(message)
    for _ in range(2):
        if any(ch in text for ch in ("Ã", "Â", "â")):
            try:
                repaired = text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
            except Exception:
                break
            if repaired and repaired != text:
                text = repaired
                continue
        break
    replacements = {
        "âœ…": "[OK]",
        "âš ": "[WARN]",
        "âŒ": "[FAIL]",
        "ðŸ”": "[INFO]",
        "ðŸ“‚": "[LOAD]",
        "ðŸ“": "[LOAD]",
        "ðŸ“Š": "[ANALYZE]",
        "ðŸ“ˆ": "[EVAL]",
        "ðŸ’¾": "[SAVE]",
        "ðŸŽ¯": "[GOAL]",
        "ðŸ”§": "[SETUP]",
        "â€¦": "...",
        "→": "->",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = text.encode("ascii", errors="ignore").decode("ascii", errors="ignore")
    return " ".join(text.split())


def print(*args, **kwargs):
    """Module-local print wrapper that sanitizes evaluation output."""
    cleaned = [
        _sanitize_console_text(arg) if isinstance(arg, str) else arg
        for arg in args
    ]
    return _builtins.print(*cleaned, **kwargs)


def print_progress(message: str, step: int = None, total: int = None):
    """Print progress message with optional step counter."""
    message = _sanitize_console_text(message)
    timestamp = datetime.now().strftime("%H:%M:%S")
    if step is not None and total is not None:
        progress = f"[{step}/{total}]"
        print(f"[{timestamp}] {progress} {message}")
    else:
        print(f"[{timestamp}] {message}")
    sys.stdout.flush()


def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    sys.stdout.flush()


def _print_eval_loop_progress(
    completed_steps: int,
    total_steps: int,
    portfolio_values: list,
    rewards_by_agent: Dict[str, list],
    successful_inference_actions: Optional[int] = None,
    total_inference_attempts: Optional[int] = None,
):
    """Emit a live ASCII evaluation progress line."""
    if total_steps <= 0:
        return

    current_portfolio = portfolio_values[-1] if portfolio_values else 800_000_000
    portfolio_change = ((current_portfolio / 800_000_000) - 1) * 100
    total_reward = sum(sum(rewards_by_agent[agent]) for agent in rewards_by_agent)
    msg = (
        f"Progress: {completed_steps}/{total_steps} "
        f"({completed_steps / total_steps * 100:.1f}%) | "
        f"NAV: ${current_portfolio/1e6:.1f}M ({portfolio_change:+.2f}%) | "
        f"Total Reward: {total_reward:.1f}"
    )
    if successful_inference_actions is not None and total_inference_attempts is not None:
        success_rate = (
            successful_inference_actions / total_inference_attempts
            if total_inference_attempts > 0 else 0.0
        )
        msg += f" | Inference: {success_rate*100:.1f}%"
    print_progress(msg)

# Import project modules (no side effects / no prints at import time).
from environment import RenewableMultiAgentEnv
from metacontroller import MultiESGAgent
from generator import load_energy_data
from utils import configure_tf_memory

# Portfolio analysis imports
try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Matplotlib/Seaborn not available - plotting disabled")


class EvaluationConfig:
    """Lightweight config for evaluation mode."""
    def __init__(self):
        self.update_every = 128
        self.lr = 3e-4
        self.ent_coef = 0.01
        self.verbose = 1
        self.seed = 42
        self.multithreading = True
        self.agent_policies = [
            {"mode": "PPO"},  # investor_0
            {"mode": "DQN"},  # battery_operator_0
            {"mode": "RULE"},  # risk_controller_0
            {"mode": "RULE"},  # meta_controller_0
        ]
        self.battery_action_mode = "discrete"
        self.battery_discrete_action_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.battery_initial_soc = 0.50


class PortfolioAnalysisConfig:
    """Configuration for portfolio analysis."""
    def __init__(self):
        self.risk_free_annual = 0.02
        self.min_annualization = 252
        self.max_annualization = 52560
        self.equity_cols = ("portfolio_value", "equity", "total_return_nav", "portfolio_performance", "fund_performance")
        self.budget_cols = ("budget", "investment_capital")
        self.plot_title = "Hybrid Renewable Energy Fund - AI Performance Analysis"

        # Get fund parameters from config if available
        try:
            from config import EnhancedConfig
            config = EnhancedConfig()
            self.initial_fund_size = config.init_budget_usd
            self.currency_conversion = config.dkk_to_usd_rate
            self.physical_allocation = config.physical_allocation
            self.target_baseline_return = 0.0376  # 3.76% baseline
            self.target_ai_return = 0.0544        # 5.44% AI-enhanced
        except ImportError:
            self.initial_fund_size = 800_000_000
            self.currency_conversion = 0.15
            self.physical_allocation = 0.88
            self.target_baseline_return = 0.0376
            self.target_ai_return = 0.0544


class SafeDivision:
    """Safe division utility to avoid division by zero."""
    @staticmethod
    def div(numerator, denominator, default=0.0):
        return numerator / denominator if denominator != 0 else default


class PortfolioAnalyzer:
    """Integrated portfolio analyzer for evaluation results."""

    def __init__(self, results_data: Dict[str, Any], log_data: Optional[pd.DataFrame] = None):
        self.results = results_data
        self.log_data = log_data
        self.config = PortfolioAnalysisConfig()
        self.analysis_results = {}

    def analyze_performance(self, make_plots: bool = False, save_plots: bool = False, output_dir: str = "evaluation_results") -> Dict[str, Any]:
        """Perform comprehensive portfolio performance analysis."""
        print("\nÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  PERFORMING PORTFOLIO ANALYSIS")
        print("=" * 50)

        analysis = {}

        # Basic performance metrics
        analysis['basic_metrics'] = self._analyze_basic_metrics()

        # Risk analysis
        analysis['risk_analysis'] = self._analyze_risk_metrics()

        # Economic model analysis
        analysis['economic_analysis'] = self._analyze_economic_model()

        # AI performance analysis
        analysis['ai_analysis'] = self._analyze_ai_performance()

        # Statistical confidence
        analysis['confidence_analysis'] = self._analyze_statistical_confidence()

        if make_plots and HAS_PLOTTING:
            analysis['plots'] = self._create_performance_plots(save_plots, output_dir)

        # Generate summary
        analysis['summary'] = self._generate_analysis_summary(analysis)

        self.analysis_results = analysis
        print("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Portfolio analysis completed")

        return analysis

    def _analyze_basic_metrics(self) -> Dict[str, Any]:
        """Analyze basic performance metrics."""
        metrics = {}

        # Extract key metrics from results
        metrics['total_return'] = self.results.get('total_return', 0.0)
        metrics['sharpe_ratio'] = self.results.get('sharpe_ratio', 0.0)
        metrics['volatility'] = self.results.get('volatility', 0.0)
        metrics['max_drawdown'] = self.results.get('max_drawdown', 0.0)

        # Portfolio values
        initial_pv = self.results.get('initial_portfolio_value', 0.0)
        final_pv = self.results.get('final_portfolio_value', 0.0)

        if initial_pv > 0:
            metrics['absolute_return'] = final_pv - initial_pv
            metrics['return_percentage'] = (final_pv / initial_pv - 1) * 100

        # Performance vs targets
        actual_return = metrics['total_return']
        metrics['vs_baseline_target'] = actual_return - self.config.target_baseline_return
        metrics['vs_ai_target'] = actual_return - self.config.target_ai_return
        metrics['target_achievement_ratio'] = SafeDivision.div(actual_return, self.config.target_ai_return, 0.0)

        return metrics

    def _analyze_risk_metrics(self) -> Dict[str, Any]:
        """Analyze risk-related metrics."""
        risk_analysis = {}

        # Basic risk metrics
        risk_analysis['average_risk'] = self.results.get('average_risk', 0.0)
        risk_analysis['max_risk'] = self.results.get('max_risk', 0.0)
        risk_analysis['min_risk'] = self.results.get('min_risk', 0.0)

        # Risk-adjusted returns
        total_return = self.results.get('total_return', 0.0)
        avg_risk = risk_analysis['average_risk']

        if avg_risk > 0:
            risk_analysis['return_per_unit_risk'] = total_return / avg_risk
            risk_analysis['risk_efficiency'] = 'high' if risk_analysis['return_per_unit_risk'] > 0.1 else 'moderate'

        # Volatility analysis
        volatility = self.results.get('volatility', 0.0)
        risk_analysis['volatility_category'] = (
            'low' if volatility < 0.1 else
            'moderate' if volatility < 0.2 else
            'high'
        )

        return risk_analysis

    def _create_performance_plots(self, save_plots: bool = False, output_dir: str = "evaluation_results") -> Dict[str, str]:
        """Create performance visualization plots."""
        if not HAS_PLOTTING:
            return {'error': 'Plotting libraries not available'}

        plots_created = {}

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(self.config.plot_title, fontsize=16, fontweight='bold')

            ax1 = axes[0, 0]
            metrics = ['Total Return', 'Sharpe Ratio', 'Volatility', 'Max Drawdown']
            values = [
                self.results.get('total_return', 0) * 100,
                self.results.get('sharpe_ratio', 0),
                self.results.get('volatility', 0) * 100,
                self.results.get('max_drawdown', 0) * 100,
            ]
            bars = ax1.bar(metrics, values, color=['green', 'blue', 'orange', 'red'])
            ax1.set_title('Key Performance Metrics')
            ax1.set_ylabel('Value (%)')
            ax1.tick_params(axis='x', rotation=45)
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f'{value:.2f}%', ha='center', va='bottom')

            ax2 = axes[0, 1]
            agent_rewards = {}
            for key, value in self.results.items():
                if '_total_reward' in key:
                    agent_name = key.replace('_total_reward', '').replace('_0', '')
                    agent_rewards[agent_name] = value

            if agent_rewards:
                agents = list(agent_rewards.keys())
                rewards = list(agent_rewards.values())
                bars = ax2.bar(agents, rewards, color='skyblue')
                ax2.set_title('Agent Performance (Total Rewards)')
                ax2.set_ylabel('Total Reward')
                ax2.tick_params(axis='x', rotation=45)
                for bar, reward in zip(bars, rewards):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f'{reward:.1f}', ha='center', va='bottom')

            ax3 = axes[1, 0]
            risk_metrics = ['Avg Risk', 'Max Risk', 'Volatility']
            risk_values = [
                self.results.get('average_risk', 0),
                self.results.get('max_risk', 0),
                self.results.get('volatility', 0),
            ]
            ax3.bar(risk_metrics, risk_values, color='coral')
            ax3.set_title('Risk Metrics')
            ax3.set_ylabel('Risk Level')

            ax4 = axes[1, 1]
            targets = ['Baseline Target', 'AI Target', 'Actual Return']
            target_values = [
                self.config.target_baseline_return * 100,
                self.config.target_ai_return * 100,
                self.results.get('total_return', 0) * 100,
            ]
            bars = ax4.bar(targets, target_values, color=['gray', 'lightblue', 'green'])
            ax4.set_title('Performance vs Targets')
            ax4.set_ylabel('Return (%)')
            ax4.tick_params(axis='x', rotation=45)
            for bar, value in zip(bars, target_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2.0, height, f'{value:.2f}%', ha='center', va='bottom')

            plt.tight_layout()

            if save_plots:
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = os.path.join(output_dir, f"performance_analysis_{timestamp}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plots_created['performance_plot'] = plot_path
                print(f"Performance plot saved: {plot_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            print(f"Error creating plots: {e}")
            plots_created['error'] = str(e)

        return plots_created

    def _generate_analysis_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of analysis."""
        summary = {}

        # Overall performance assessment
        total_return = self.results.get('total_return', 0.0)
        ai_target = self.config.target_ai_return

        if total_return >= ai_target:
            summary['overall_assessment'] = 'EXCELLENT'
            summary['performance_grade'] = 'A'
        elif total_return >= self.config.target_baseline_return:
            summary['overall_assessment'] = 'GOOD'
            summary['performance_grade'] = 'B'
        elif total_return >= 0:
            summary['overall_assessment'] = 'MODERATE'
            summary['performance_grade'] = 'C'
        else:
            summary['overall_assessment'] = 'POOR'
            summary['performance_grade'] = 'D'

        # Key highlights
        summary['key_metrics'] = {
            'total_return_pct': f"{total_return * 100:.2f}%",
            'sharpe_ratio': f"{self.results.get('sharpe_ratio', 0):.3f}",
            'max_drawdown_pct': f"{self.results.get('max_drawdown', 0) * 100:.2f}%",
            'confidence_level': analysis.get('confidence_analysis', {}).get('confidence_level', 'unknown')
        }

        # Recommendations
        recommendations = []

        if total_return < ai_target:
            recommendations.append("Consider optimizing AI model parameters")

        if self.results.get('volatility', 0) > 0.2:
            recommendations.append("Implement additional risk management measures")

        inference_rate = analysis.get('ai_analysis', {}).get('action_inference_success_rate', None)
        if inference_rate is not None and inference_rate < 80:
            recommendations.append("Investigate policy inference failures during evaluation")

        if not recommendations:
            recommendations.append("Maintain current strategy - performance is satisfactory")

        summary['recommendations'] = recommendations

        return summary

    def _analyze_economic_model(self) -> Dict[str, Any]:
        """Analyze economic model performance."""
        economic = {}

        # Fund structure analysis
        initial_pv = self.results.get('initial_portfolio_value', 0.0)
        final_pv = self.results.get('final_portfolio_value', 0.0)

        economic['fund_size_initial'] = initial_pv
        economic['fund_size_final'] = final_pv
        economic['fund_growth'] = final_pv - initial_pv if initial_pv > 0 else 0.0
        economic['fund_growth_percentage'] = SafeDivision.div(economic['fund_growth'], initial_pv, 0.0) * 100

        # Asset allocation efficiency
        physical_target = self.config.physical_allocation
        economic['target_physical_allocation'] = physical_target * 100
        economic['target_trading_allocation'] = (1 - physical_target) * 100

        # Revenue analysis
        total_rewards = self.results.get('total_rewards', 0.0)
        economic['total_rewards'] = total_rewards
        economic['reward_efficiency'] = SafeDivision.div(total_rewards, initial_pv, 0.0) if initial_pv > 0 else 0.0

        return economic

    def _analyze_ai_performance(self) -> Dict[str, Any]:
        """Analyze AI-specific performance improvements."""
        ai_analysis = {}

        # Model inference reliability
        inference_rate = _extract_action_inference_success_rate(self.results)
        if inference_rate is not None:
            ai_analysis['action_inference_success_rate'] = inference_rate * 100
            ai_analysis['action_inference_reliability'] = (
                'excellent' if inference_rate > 0.9 else
                'good' if inference_rate > 0.7 else
                'moderate'
            )

        # Agent performance
        agent_rewards = {}
        total_agent_rewards = 0
        for key, value in self.results.items():
            if '_total_reward' in key:
                agent_name = key.replace('_total_reward', '')
                agent_rewards[agent_name] = value
                total_agent_rewards += value

        ai_analysis['agent_rewards'] = agent_rewards
        ai_analysis['total_agent_rewards'] = total_agent_rewards

        # AI enhancement assessment
        actual_return = self.results.get('total_return', 0.0)
        baseline_target = self.config.target_baseline_return
        ai_target = self.config.target_ai_return

        if actual_return > baseline_target:
            ai_analysis['ai_enhancement'] = actual_return - baseline_target
            ai_analysis['enhancement_percentage'] = SafeDivision.div(ai_analysis['ai_enhancement'], baseline_target, 0.0) * 100
        else:
            ai_analysis['ai_enhancement'] = 0.0
            ai_analysis['enhancement_percentage'] = 0.0

        ai_analysis['ai_effectiveness'] = (
            'highly_effective' if actual_return > ai_target else
            'effective' if actual_return > baseline_target else
            'needs_improvement'
        )

        return ai_analysis

    def _analyze_statistical_confidence(self) -> Dict[str, Any]:
        """Analyze statistical confidence in results."""
        confidence = {}

        # Evaluation steps and data quality
        eval_steps = self.results.get('evaluation_steps', 0)
        confidence['evaluation_steps'] = eval_steps
        confidence['data_points'] = eval_steps

        # Estimate confidence based on data size
        if eval_steps >= 10000:
            confidence['confidence_level'] = 'high'
            confidence['confidence_score'] = 0.95
        elif eval_steps >= 5000:
            confidence['confidence_level'] = 'moderate'
            confidence['confidence_score'] = 0.80
        elif eval_steps >= 1000:
            confidence['confidence_level'] = 'low'
            confidence['confidence_score'] = 0.65
        else:
            confidence['confidence_level'] = 'very_low'
            confidence['confidence_score'] = 0.50

        # Time period analysis (assuming 10-minute intervals)
        minutes_of_data = eval_steps * 10
        hours_of_data = minutes_of_data / 60
        days_of_data = hours_of_data / 24

        confidence['hours_of_data'] = hours_of_data
        confidence['days_of_data'] = days_of_data
        confidence['months_of_data'] = days_of_data / 30

        return confidence


def _annual_rate_to_step_rate(annual_rate: float, periods_per_year: float) -> float:
    annual = float(annual_rate)
    periods = float(max(periods_per_year, 1.0))
    if not np.isfinite(annual) or not np.isfinite(periods):
        return 0.0
    if annual <= -1.0:
        return -1.0
    return float((1.0 + annual) ** (1.0 / periods) - 1.0)


def _infer_periods_per_year(timestamps, default_periods_per_year: float = 52596.0) -> float:
    try:
        if timestamps is None:
            return float(default_periods_per_year)
        ts = pd.Series(timestamps).dropna()
        if ts.empty:
            return float(default_periods_per_year)
        parsed = pd.to_datetime(ts, errors="coerce").dropna().sort_values()
        if parsed.size < 2:
            return float(default_periods_per_year)
        deltas = parsed.diff().dt.total_seconds().to_numpy(dtype=np.float64)
        deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
        if deltas.size == 0:
            return float(default_periods_per_year)
        median_seconds = float(np.median(deltas))
        if median_seconds <= 0.0:
            return float(default_periods_per_year)
        seconds_per_year = 365.25 * 24.0 * 60.0 * 60.0
        inferred = seconds_per_year / median_seconds
        return float(np.clip(inferred, 1.0, 60.0 * 24.0 * 365.25))
    except Exception:
        return float(default_periods_per_year)


def _compute_annualized_risk_metrics(
    returns,
    *,
    periods_per_year: float,
    annual_risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    vals = np.asarray(returns, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    periods = float(max(periods_per_year, 1.0))
    if vals.size == 0:
        return {
            "step_mean_return": 0.0,
            "step_volatility": 0.0,
            "annualized_volatility": 0.0,
            "annualized_sharpe": 0.0,
            "per_step_risk_free_rate": _annual_rate_to_step_rate(annual_risk_free_rate, periods),
        }

    step_mean = float(np.mean(vals))
    # Use sample volatility for finite evaluation windows. Population volatility
    # slightly overstates Sharpe, especially for daily subsamples.
    step_vol = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    step_rf = _annual_rate_to_step_rate(annual_risk_free_rate, periods)
    excess_mean = float(np.mean(vals - step_rf))
    annualized_vol = float(step_vol * math.sqrt(periods))
    annualized_sharpe = float((excess_mean / step_vol) * math.sqrt(periods)) if step_vol > 0.0 else 0.0
    return {
        "step_mean_return": step_mean,
        "step_volatility": step_vol,
        "annualized_volatility": annualized_vol,
        "annualized_sharpe": annualized_sharpe,
        "per_step_risk_free_rate": float(step_rf),
    }


def _series_returns(values) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.asarray([], dtype=np.float64)
    prev = vals[:-1]
    cur = vals[1:]
    ok = np.isfinite(prev) & np.isfinite(cur) & (np.abs(prev) > 1e-12)
    return ((cur[ok] - prev[ok]) / prev[ok]).astype(np.float64)


def _subsample_path(values, step: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return vals
    stride = int(max(step, 1))
    # Use completed non-overlapping periods only. Appending the final partial
    # period would treat, for example, a few remaining 10-minute steps as a full
    # daily return and can slightly bias the reported Sharpe.
    idx = np.arange(0, vals.size, stride, dtype=int)
    if idx.size < 2 and vals.size >= 2:
        idx = np.asarray([0, vals.size - 1], dtype=int)
    return vals[idx]


def _acf(values: np.ndarray, lag: int) -> float:
    lag = int(lag)
    if values.size <= lag + 1:
        return float("nan")
    a = values[:-lag] - np.mean(values[:-lag])
    b = values[lag:] - np.mean(values[lag:])
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def _hac_variance_inflation(returns: np.ndarray, max_lag: int) -> float:
    if returns.size < 3 or int(max_lag) <= 0:
        return 1.0
    q = min(int(max_lag), returns.size - 2)
    total = 1.0
    for lag in range(1, q + 1):
        rho = _acf(returns, lag)
        if not math.isfinite(rho):
            continue
        weight = 1.0 - (lag / (q + 1.0))
        total += 2.0 * weight * rho
    return float(max(total, 1e-12))


def _path_sharpe(values, *, periods_per_year: float, annual_risk_free_rate: float) -> Dict[str, float]:
    returns = _series_returns(values)
    periods = float(max(periods_per_year, 1.0))
    if returns.size < 2:
        return {
            "n_returns": float(returns.size),
            "step_volatility": 0.0,
            "annualized_volatility": 0.0,
            "annualized_sharpe": 0.0,
            "mean_return": 0.0,
        }
    stats = _compute_annualized_risk_metrics(
        returns,
        periods_per_year=periods,
        annual_risk_free_rate=annual_risk_free_rate,
    )
    return {
        "n_returns": float(returns.size),
        "step_volatility": float(stats["step_volatility"]),
        "annualized_volatility": float(stats["annualized_volatility"]),
        "annualized_sharpe": float(stats["annualized_sharpe"]),
        "mean_return": float(np.mean(returns)),
    }


def _resolve_eval_risk_free_rate(eval_env, default: float = 0.02) -> float:
    try:
        cfg = getattr(eval_env, "config", None)
        if cfg is not None:
            val = getattr(cfg, "risk_free_rate", None)
            if val is not None:
                return float(val)
        wrapped = getattr(eval_env, "env", None)
        if wrapped is not None:
            cfg = getattr(wrapped, "config", None)
            if cfg is not None:
                val = getattr(cfg, "risk_free_rate", None)
                if val is not None:
                    return float(val)
    except Exception:
        pass
    return float(default)

def _prepare_baseline_imports() -> None:
    """Expose the single authoritative Prototype5 baseline implementation."""
    root = Path(__file__).resolve().parent
    if not (root / "baselines").is_dir():
        root = root.parent
    baseline_paths = [
        root / "baselines",
        root / "baselines" / "Baseline1_TraditionalPortfolio",
    ]
    for path in baseline_paths:
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def _load_baseline_data(eval_data_path: str, timesteps: int) -> pd.DataFrame:
    data = load_energy_data(eval_data_path)
    max_steps = int(max(1, min(int(timesteps), len(data))))
    return data.iloc[:max_steps].reset_index(drop=True)


def _baseline_summary(
    metrics: Dict[str, Any],
    *,
    method: str,
    role: str,
    baseline_id: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = dict(metrics)
    out.update(
        {
            "baseline_id": baseline_id,
            "method": method,
            "role": role,
            "status": "completed",
            "evaluation_contract": "prototype5_final_campaign_v1",
            "hybrid_benchmark_contract": True,
            "current_codebase_environment": True,
            "artifact_free_evaluation": True,
            "notes": (
                "Evaluated in-process with the current fixed hybrid-fund accounting: "
                "88% physical infrastructure, 12% trading sleeve, operating revenue, "
                "MTM, transaction costs, depreciation, and shareholder distributions."
            ),
        }
    )
    if extra:
        out.update(extra)
    return out


def _evaluate_traditional_portfolio_baseline(data: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    _prepare_baseline_imports()
    from baseline_common import HybridFundLedger
    from traditional_portfolio_optimizer import Timebase, OptimizerConfig, TraditionalPortfolioOptimizer

    tb = Timebase(time_step_hours=10.0 / 60.0)
    opt_cfg = OptimizerConfig(
        method="markowitz_mean_variance",
        risk_aversion_lambda=5.0,
        shrinkage=0.1,
        allow_short=False,
        seed=int(seed),
    )
    opt = TraditionalPortfolioOptimizer(timebase=tb, rf_annual=0.02, opt_cfg=opt_cfg)
    returns = opt.build_asset_returns(data)
    ledger = HybridFundLedger(data, seed=seed, timebase_hours=tb.time_step_hours)

    rebalance_every = max(1, int(ledger.investment_freq))
    lookback_steps = max(int(tb.steps_per_year), rebalance_every)
    initial_risky = opt._equal_weight(
        np.zeros(len(opt.risky_assets), dtype=np.float64),
        np.eye(len(opt.risky_assets), dtype=np.float64),
        opt.rf_step,
    )
    w = {a: float(weight) for a, weight in zip(opt.risky_assets, initial_risky)}
    w["cash"] = max(0.0, 1.0 - sum(w.values()))
    rebalance_count = 0
    exposure_history = []
    price_weight_history = []

    for t in range(len(returns)):
        target_exposure = None
        if t > 0 and (t % rebalance_every == 0):
            start = max(0, t - lookback_steps)
            window = returns.iloc[start:t]
            try:
                w = opt.rebalance_weights(window, method=opt_cfg.method)
                # The hybrid fund already holds fixed physical assets. Only the
                # optimizer's explicit price sleeve should become financial MTM
                # exposure; physical wind/solar/hydro weights are diagnostics.
                price_weight = float(w.get("price", 0.0))
                target_exposure = float(np.clip(price_weight, -1.0, 1.0))
                rebalance_count += 1
                price_weight_history.append(float(price_weight))
            except Exception:
                target_exposure = None
        record = ledger.step(t, target_exposure=target_exposure, battery_action="idle")
        exposure_history.append(float(record.get("current_abs_exposure_dkk", 0.0)))

    return _baseline_summary(
        ledger.performance_metrics(),
        method="Traditional Portfolio - Markowitz Mean-Variance (12% price-sleeve cap)",
        role="classical_finance_trading_sleeve",
        baseline_id="baseline_1",
        extra={
            "rebalance_count": int(rebalance_count),
            "rebalance_every_steps": int(rebalance_every),
            "mean_abs_exposure_dkk": float(np.mean(exposure_history)) if exposure_history else 0.0,
            "price_sleeve_cap": 0.12,
            "mean_price_weight": float(np.mean(price_weight_history)) if price_weight_history else 0.0,
            "max_price_weight": float(np.max(price_weight_history)) if price_weight_history else 0.0,
            "baseline_interpretation": (
                "Classical optimizer benchmark. Only its explicit price sleeve "
                "maps to financial MTM exposure; physical weights are diagnostic "
                "because the hybrid fund holds fixed physical infrastructure."
            ),
        },
    )


def _evaluate_capped_long_price_sleeve_baseline(data: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    _prepare_baseline_imports()
    from baseline_common import HybridFundLedger

    ledger = HybridFundLedger(data, seed=seed)
    price_weight = 0.12
    decision_count = 0
    exposure_history = []
    rebalance_every = max(1, int(ledger.investment_freq))

    for t in range(len(data)):
        target_exposure = None
        if t > 0 and (t % rebalance_every == 0):
            target_exposure = price_weight
            decision_count += 1
        record = ledger.step(t, target_exposure=target_exposure, battery_action="idle")
        exposure_history.append(float(record.get("current_abs_exposure_dkk", 0.0)))

    return _baseline_summary(
        ledger.performance_metrics(),
        method="Capped Long Price Sleeve (12%)",
        role="constant_long_price_sleeve_comparator",
        baseline_id="baseline_4",
        extra={
            "target_price_weight": price_weight,
            "rebalance_count": int(decision_count),
            "rebalance_every_steps": int(rebalance_every),
            "mean_abs_exposure_dkk": float(np.mean(exposure_history)) if exposure_history else 0.0,
            "baseline_interpretation": (
                "Mechanistic comparator for constant capped long exposure to the "
                "same settlement-linked financial sleeve. This is not a learned "
                "or optimized policy."
            ),
        },
    )


def _evaluate_rule_based_baseline(data: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    _prepare_baseline_imports()
    from baseline_common import HybridFundLedger

    ledger = HybridFundLedger(data, seed=seed)
    price_hist = []
    target_exposure = 0.0
    trigger_counts = {
        "long_price_momentum": 0,
        "short_price_reversion": 0,
        "battery_charge": 0,
        "battery_discharge": 0,
        "battery_idle": 0,
    }
    exposure_history = []

    prices = data.get("price", pd.Series(0.0, index=data.index)).astype(float).to_numpy()
    wind = data.get("wind", pd.Series(0.0, index=data.index)).astype(float).to_numpy()
    solar = data.get("solar", pd.Series(0.0, index=data.index)).astype(float).to_numpy()
    hydro = data.get("hydro", pd.Series(0.0, index=data.index)).astype(float).to_numpy()

    for t in range(len(data)):
        price = float(prices[t])
        price_hist.append(price)
        hist = np.asarray(price_hist[-168:], dtype=float)
        battery_action = "idle"
        decision_exposure = None

        if hist.size >= 24:
            p25 = float(np.percentile(hist, 25))
            p50 = float(np.percentile(hist, 50))
            p75 = float(np.percentile(hist, 75))
            recent = hist[-min(hist.size, 12):]
            momentum = float((recent[-1] - recent[0]) / max(abs(recent[0]), 1e-6)) if recent.size >= 2 else 0.0
            start = max(0, t - 168)
            generation_support = float(
                np.nanmean(
                    [
                        wind[t] / max(np.nanpercentile(wind[start: t + 1], 75), 1e-6),
                        solar[t] / max(np.nanpercentile(solar[start: t + 1], 75), 1e-6),
                        hydro[t] / max(np.nanpercentile(hydro[start: t + 1], 75), 1e-6),
                    ]
                )
            )

            if price <= p25:
                battery_action = "charge"
                trigger_counts["battery_charge"] += 1
            elif price >= p75:
                battery_action = "discharge"
                trigger_counts["battery_discharge"] += 1
            else:
                trigger_counts["battery_idle"] += 1

            if momentum > 0.002 and price >= p50 and generation_support >= 0.75:
                target_exposure = 0.50
                trigger_counts["long_price_momentum"] += 1
            elif momentum < -0.002 and price <= p50:
                target_exposure = -0.25
                trigger_counts["short_price_reversion"] += 1
            else:
                target_exposure *= 0.90
                if abs(target_exposure) < 0.05:
                    target_exposure = 0.0

            decision_exposure = float(np.clip(target_exposure, -0.50, 0.50))
        else:
            trigger_counts["battery_idle"] += 1

        record = ledger.step(t, target_exposure=decision_exposure, battery_action=battery_action)
        exposure_history.append(float(record.get("current_abs_exposure_dkk", 0.0)))

    return _baseline_summary(
        ledger.performance_metrics(),
        method="Rule-Based Heuristic - Fixed Hybrid Fund",
        role="expert_rules_trading_and_battery_sleeves",
        baseline_id="baseline_2",
        extra={
            "rule_triggers": trigger_counts,
            "mean_abs_exposure_dkk": float(np.mean(exposure_history)) if exposure_history else 0.0,
            "alignment_fix": (
                "This rule baseline no longer buys or sells physical infrastructure during evaluation; "
                "it uses fixed Tier1 physical assets and only controls trading exposure plus battery dispatch."
            ),
        },
    )


def _evaluate_buy_and_hold_baseline(data: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    _prepare_baseline_imports()
    from baseline_common import HybridFundLedger

    ledger = HybridFundLedger(data, seed=seed)
    for t in range(len(data)):
        ledger.step(t, target_exposure=None, battery_action="idle")

    return _baseline_summary(
        ledger.performance_metrics(),
        method="Hybrid Buy-and-Hold",
        role="passive_fixed_physical_sleeve_idle_trading_sleeve",
        baseline_id="baseline_3",
        extra={
            "active_decisions": 0,
            "market_exposure": 0.0,
            "cash_accrual": "disabled_to_match_current_environment",
        },
    )


def run_traditional_baselines(
    eval_data_path: str,
    timesteps: int = 10000,
    output_dir: str = "evaluation_results",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run final paper baselines in-process and return one aggregate JSON-ready payload."""
    print_progress("Running aligned hybrid-fund baselines in process")
    data = _load_baseline_data(eval_data_path, timesteps)

    evaluators = [
        ("baseline_1", _evaluate_traditional_portfolio_baseline),
        ("baseline_2", _evaluate_rule_based_baseline),
        ("baseline_3", _evaluate_buy_and_hold_baseline),
        ("baseline_4", _evaluate_capped_long_price_sleeve_baseline),
    ]
    baseline_results: Dict[str, Any] = {}
    for key, fn in evaluators:
        try:
            result = fn(data, seed=seed)
            baseline_results[key] = result
            final_wealth = float(result.get("final_portfolio_value", result.get("final_value_usd", 0.0)))
            total_return = float(result.get("total_return", 0.0)) * 100.0
            print_progress(
                f"   {result.get('method', key)}: final wealth ${final_wealth/1e6:.2f}M, "
                f"return {total_return:+.2f}%"
            )
        except Exception as e:
            baseline_results[key] = {
                "baseline_id": key,
                "status": "failed",
                "error": str(e),
                "evaluation_contract": "prototype5_final_campaign_v1",
            }

    return baseline_results


def find_latest_checkpoint(checkpoint_base_dir: str = "normal/checkpoints") -> Optional[str]:
    """Find the latest checkpoint directory or final models."""

    # First, try to find final_models directory (preferred)
    base_dir = os.path.dirname(checkpoint_base_dir) if checkpoint_base_dir.endswith('checkpoints') else checkpoint_base_dir
    final_models_dir = os.path.join(base_dir, "final_models")

    if os.path.exists(final_models_dir):
        # Check if final models contain the required policy files
        required_files = [
            "investor_0_policy.zip",
            "battery_operator_0_policy.zip",
        ]

        all_files_exist = all(os.path.exists(os.path.join(final_models_dir, f)) for f in required_files)

        if all_files_exist:
            print(f"ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯ Found final models directory: {final_models_dir}")
            return final_models_dir
        else:
            print("ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Final models directory exists but missing some policy files")

    # Fallback to checkpoint detection
    if not os.path.exists(checkpoint_base_dir):
        print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Checkpoint directory not found: {checkpoint_base_dir}")
        return None

    # Find all checkpoint directories
    checkpoint_pattern = os.path.join(checkpoint_base_dir, "checkpoint_*")
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ No checkpoints found in {checkpoint_base_dir}")
        return None

    # Sort by checkpoint number (extract number from checkpoint_XXXXX)
    def get_checkpoint_number(path):
        try:
            return int(os.path.basename(path).split('_')[1])
        except (IndexError, ValueError):
            return 0

    latest_checkpoint = max(checkpoints, key=get_checkpoint_number)
    checkpoint_num = get_checkpoint_number(latest_checkpoint)

    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â Found latest checkpoint: {os.path.basename(latest_checkpoint)} (step {checkpoint_num})")
    return latest_checkpoint


def load_checkpoint_models(checkpoint_dir: str) -> Dict[str, Any]:
    """Load models from checkpoint using Stable Baselines3."""
    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â¥ Loading models from checkpoint: {checkpoint_dir}")
    
    try:
        # ------------------------------------------------------------------
        # NumPy pickle compatibility shim
        # ------------------------------------------------------------------
        # Some SB3 checkpoints can embed pickled objects referencing internal NumPy module paths
        # like `numpy._core.numeric` (NumPy 2.x). If this runtime uses a different NumPy layout,
        # unpickling can fail with: "No module named 'numpy._core.numeric'".
        #
        # We install a minimal alias so older/newer checkpoints can load on this machine.
        try:
            import sys
            import types
            import numpy.core.numeric as _np_core_numeric

            if "numpy._core.numeric" not in sys.modules:
                sys.modules.setdefault("numpy._core", types.ModuleType("numpy._core"))
                sys.modules["numpy._core.numeric"] = _np_core_numeric
                # Expose as attribute for completeness (some loaders inspect numpy._core)
                try:
                    setattr(sys.modules["numpy"], "_core", sys.modules["numpy._core"])
                except Exception as alias_error:
                    print(f"Warning: NumPy shim attribute alias failed: {alias_error}")
        except Exception as shim_error:
            # Never block evaluation due to a best-effort compatibility shim.
            print(f"Warning: NumPy compatibility shim skipped: {shim_error}")

        from stable_baselines3 import PPO, SAC, TD3, DQN
        from policy import (
            BetaActorCriticPolicy,
            CentralizedCriticActorCriticPolicy,
            CentralizedCriticBetaActorCriticPolicy,
            CentralizedCriticPPO,
            CentralizedCriticRolloutBuffer,
        )
        print("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Successfully imported stable-baselines3")

        algo_map = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "DQN": DQN}
        agent_modes = {
            "investor_0": "PPO",
            "battery_operator_0": "DQN",
            "risk_controller_0": "PPO",
            "meta_controller_0": "PPO",
        }
        saved_algo = "ippo"
        config_path = os.path.join(checkpoint_dir, "training_config.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                final_cfg = saved.get("final_config", {}) or {}
                flags = saved.get("flags", {}) or {}
                runtime_contract = saved.get("runtime_contract", {}) or {}
                controller_contract = (
                    runtime_contract.get("controllers", {}) if isinstance(runtime_contract, dict) else {}
                )
                saved_algo = str(final_cfg.get("algo", flags.get("algo", "ippo")) or "ippo").strip().lower()
                saved_policies = final_cfg.get("agent_policies", None)
                agent_order = [
                    "investor_0",
                    "battery_operator_0",
                    "risk_controller_0",
                    "meta_controller_0",
                ]
                if isinstance(saved_policies, list):
                    for idx, agent_name in enumerate(agent_order):
                        if idx >= len(saved_policies):
                            break
                        policy_cfg = saved_policies[idx] or {}
                        if isinstance(policy_cfg, dict):
                            agent_modes[agent_name] = str(
                                policy_cfg.get("mode", agent_modes[agent_name])
                            ).upper()
                risk_rule = bool(
                    final_cfg.get(
                        "risk_controller_rule_based",
                        flags.get(
                            "risk_controller_rule_based",
                            controller_contract.get("risk_controller_rule_based", False)
                            if isinstance(controller_contract, dict) else False,
                        ),
                    )
                )
                meta_rule = bool(
                    final_cfg.get(
                        "meta_controller_rule_based",
                        flags.get(
                            "meta_controller_rule_based",
                            controller_contract.get("meta_controller_rule_based", False)
                            if isinstance(controller_contract, dict) else False,
                        ),
                    )
                )
                if risk_rule:
                    agent_modes["risk_controller_0"] = "RULE"
                if meta_rule:
                    agent_modes["meta_controller_0"] = "RULE"
            except Exception as e:
                print(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Could not read training_config.json for algo detection: {e}")

        _ = (
            BetaActorCriticPolicy,
            CentralizedCriticActorCriticPolicy,
            CentralizedCriticBetaActorCriticPolicy,
            CentralizedCriticPPO,
            CentralizedCriticRolloutBuffer,
        )
        ppo_loader = CentralizedCriticPPO if saved_algo == "mappo" else PPO
        # mappo-2x2 task
        algo_map["PPO"] = ppo_loader
        model_configs = [
            ("investor_0", "investor_0_policy.zip", agent_modes["investor_0"], algo_map.get(agent_modes["investor_0"], ppo_loader)),
            ("battery_operator_0", "battery_operator_0_policy.zip", agent_modes["battery_operator_0"], algo_map.get(agent_modes["battery_operator_0"], DQN)),
            ("risk_controller_0", "risk_controller_0_policy.zip", agent_modes["risk_controller_0"], algo_map.get(agent_modes["risk_controller_0"], ppo_loader)),
            ("meta_controller_0", "meta_controller_0_policy.zip", agent_modes["meta_controller_0"], algo_map.get(agent_modes["meta_controller_0"], ppo_loader)),
        ]
        
        loaded_models = {}
        
        for agent_name, model_file, agent_mode, model_class in model_configs:
            model_path = os.path.join(checkpoint_dir, model_file)

            if str(agent_mode).upper() == "RULE":
                print(f"RULE policy configured for {agent_name}; using deterministic environment controller.")
                loaded_models[agent_name] = _rule_based_checkpoint_model(agent_name)
                continue
            
            if os.path.exists(model_path):
                try:
                    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â§ Loading {model_file} with {model_class.__name__}...")
                    model = model_class.load(model_path)
                    loaded_models[agent_name] = model
                    print(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ SUCCESS: Loaded {agent_name} from checkpoint!")
                except Exception as e:
                    print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Failed to load {model_file}: {e}")
                    loaded_models[agent_name] = None
            else:
                print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Model file not found: {model_path}")
                loaded_models[agent_name] = None
        
        return loaded_models
        
    except ImportError as e:
        print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Cannot import stable-baselines3: {e}")
        return {}
    except Exception as e:
        print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error loading checkpoint models: {e}")
        return {}


def _resolve_reserved_eval_forecast_context(
    args,
    output_dir: Optional[str] = None,
    config: Optional[EnhancedConfig] = None,
) -> Dict[str, Any]:
    """Resolve the canonical evaluation forecast-cache context."""

    cache_root = str(getattr(args, "forecast_cache_dir", "forecast_cache") or "forecast_cache")
    direct_csvs = []
    if os.path.isdir(cache_root):
        direct_csvs = [
            p for p in glob.glob(os.path.join(cache_root, "precomputed_forecasts_*.csv"))
            if "_metadata" not in os.path.basename(p)
        ]
    if direct_csvs:
        cache_dir = cache_root
    else:
        eval_stem = ""
        try:
            eval_stem = os.path.splitext(os.path.basename(str(getattr(args, "eval_data", "") or "")))[0]
        except Exception:
            eval_stem = ""
        def _has_forecast_csv(path: str) -> bool:
            return bool(
                os.path.isdir(path)
                and [
                    p for p in glob.glob(os.path.join(path, "precomputed_forecasts_*.csv"))
                    if "_metadata" not in os.path.basename(p)
                ]
            )
        if eval_stem == "unseendata_v2":
            candidates = [
                os.path.join(
                    cache_root,
                    "forecast_cache_eval_episode20_2025",
                    "forecast_cache_eval_episode20_2025-unseendata_v2",
                ),
                os.path.join(
                    cache_root,
                    "forecast_cache_eval_episode20_2025_v2",
                    "forecast_cache_eval_episode20_2025-unseendata_v2",
                ),
                os.path.join(
                    cache_root,
                    "forecast_cache_eval_episode20_2025_v2",
                    "forecast_cache_eval_episode20_2025",
                    "forecast_cache_eval_episode20_2025-unseendata_v2",
                ),
                os.path.join(cache_root, "forecast_cache_eval_episode20_2025_v2"),
            ]
        else:
            candidates = [
                os.path.join(
                    cache_root,
                    "forecast_cache_eval_episode20_2025",
                    "forecast_cache_eval_episode20_2025-full",
                ),
                os.path.join(cache_root, "forecast_cache_eval_episode20_2025"),
            ]
        cache_dir = next((p for p in candidates if _has_forecast_csv(p)), candidates[0])

    return {
        "forecast_cache_dir": cache_dir,
        "output_dir": output_dir,
    }


def _apply_market_fee_overrides(cfg, args) -> None:
    if getattr(args, "market_fee_model", None) is not None:
        cfg.market_fee_model = str(args.market_fee_model).strip().lower().replace("-", "_")
    if getattr(args, "transaction_fee_dkk_per_mwh", None) is not None:
        cfg.transaction_fee_dkk_per_mwh = float(args.transaction_fee_dkk_per_mwh)
    if getattr(args, "annual_market_access_fee_dkk", None) is not None:
        cfg.annual_market_access_fee_dkk = float(args.annual_market_access_fee_dkk)
    if getattr(args, "market_access_fee_allocation_fraction", None) is not None:
        cfg.market_access_fee_allocation_fraction = float(args.market_access_fee_allocation_fraction)
    if getattr(args, "market_fee_source_id", None) is not None:
        cfg.market_fee_source_id = str(args.market_fee_source_id)


def _build_eval_config(args):
    from config import EnhancedConfig

    cfg = EnhancedConfig()
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)
    if getattr(args, "eval_distribution_rate", None) is not None:
        cfg.eval_distribution_rate = float(args.eval_distribution_rate)
    if getattr(args, "distribution_rate", None) is not None:
        cfg.distribution_rate = float(args.distribution_rate)
    if getattr(args, "mtm_return_model", None) is not None:
        cfg.mtm_return_model = str(args.mtm_return_model).strip().lower()
    if getattr(args, "mtm_reference_price_dkk_per_mwh", None) is not None:
        cfg.mtm_reference_price_dkk_per_mwh = float(args.mtm_reference_price_dkk_per_mwh)
    if getattr(args, "mtm_settlement_horizon_steps", None) is not None:
        cfg.mtm_settlement_horizon_steps = int(args.mtm_settlement_horizon_steps)
    if getattr(args, "mtm_entry_price_mode", None) is not None:
        cfg.mtm_entry_price_mode = str(args.mtm_entry_price_mode).strip().lower()
    if getattr(args, "mtm_horizon_payoff_denominator_mode", None) is not None:
        cfg.mtm_horizon_payoff_denominator_mode = (
            str(args.mtm_horizon_payoff_denominator_mode).strip().lower().replace("-", "_")
        )
    if getattr(args, "mtm_settlement_price_mode", None) is not None:
        cfg.mtm_settlement_price_mode = (
            str(args.mtm_settlement_price_mode).strip().lower().replace("-", "_")
        )
    if getattr(args, "mtm_basis_price_data_path", None) is not None:
        cfg.mtm_basis_price_data_path = str(args.mtm_basis_price_data_path)
    if getattr(args, "mtm_basis_price_column", None) is not None:
        cfg.mtm_basis_price_column = str(args.mtm_basis_price_column)
    if getattr(args, "mtm_basis_timestamp_column", None) is not None:
        cfg.mtm_basis_timestamp_column = str(args.mtm_basis_timestamp_column)
    if getattr(args, "mtm_basis_scale", None) is not None:
        cfg.mtm_basis_scale = float(args.mtm_basis_scale)
    if getattr(args, "mtm_basis_centering_mode", None) is not None:
        cfg.mtm_basis_centering_mode = (
            str(args.mtm_basis_centering_mode).strip().lower().replace("-", "_")
        )
    if getattr(args, "mtm_basis_centering_window_steps", None) is not None:
        cfg.mtm_basis_centering_window_steps = int(args.mtm_basis_centering_window_steps)
    if getattr(args, "mtm_external_settlement_price_data_path", None) is not None:
        cfg.mtm_external_settlement_price_data_path = str(args.mtm_external_settlement_price_data_path)
    if getattr(args, "mtm_external_settlement_price_column", None) is not None:
        cfg.mtm_external_settlement_price_column = str(args.mtm_external_settlement_price_column)
    if getattr(args, "mtm_external_settlement_timestamp_column", None) is not None:
        cfg.mtm_external_settlement_timestamp_column = str(args.mtm_external_settlement_timestamp_column)
    if getattr(args, "mtm_external_settlement_min_price_dkk_per_mwh", None) is not None:
        cfg.mtm_external_settlement_min_price_dkk_per_mwh = float(args.mtm_external_settlement_min_price_dkk_per_mwh)
    if getattr(args, "mtm_external_settlement_max_price_dkk_per_mwh", None) is not None:
        cfg.mtm_external_settlement_max_price_dkk_per_mwh = float(args.mtm_external_settlement_max_price_dkk_per_mwh)
    if bool(getattr(args, "disable_mtm_return_cap", False)):
        cfg.mtm_apply_price_return_cap = False
    if getattr(args, "mtm_price_return_cap_min", None) is not None:
        cfg.mtm_price_return_cap_min = float(args.mtm_price_return_cap_min)
    if getattr(args, "mtm_price_return_cap_max", None) is not None:
        cfg.mtm_price_return_cap_max = float(args.mtm_price_return_cap_max)
    if getattr(args, "investor_notional_sizing_base", None) is not None:
        cfg.investor_notional_sizing_base = str(args.investor_notional_sizing_base).strip().lower()
    if getattr(args, "max_position_size", None) is not None:
        cfg.max_position_size = float(args.max_position_size)
    if getattr(args, "capital_allocation_fraction", None) is not None:
        cfg.capital_allocation_fraction = float(args.capital_allocation_fraction)
    if getattr(args, "mtm_loss_exit_threshold_pct", None) is not None:
        cfg.mtm_loss_exit_threshold_pct = float(args.mtm_loss_exit_threshold_pct)
    if getattr(args, "friction_cost_multiplier", None) is not None:
        cfg.friction_cost_multiplier = float(args.friction_cost_multiplier)
    _apply_market_fee_overrides(cfg, args)
    if getattr(args, "no_trade_threshold", None) is not None:
        cfg.no_trade_threshold = float(args.no_trade_threshold)
    if getattr(args, "no_trade_threshold_reference", None) is not None:
        cfg.no_trade_threshold_reference = str(args.no_trade_threshold_reference)
    if getattr(args, "half_spread_bp", None) is not None:
        cfg.half_spread_bp = float(args.half_spread_bp)
    if getattr(args, "max_position_size", None) is not None:
        cfg.max_position_size = float(args.max_position_size)
    if getattr(args, "capital_allocation_fraction", None) is not None:
        cfg.capital_allocation_fraction = float(args.capital_allocation_fraction)
    if getattr(args, "mtm_loss_exit_threshold_pct", None) is not None:
        cfg.mtm_loss_exit_threshold_pct = float(args.mtm_loss_exit_threshold_pct)
    # market-impact task
    if getattr(args, "impact_coef_bp", None) is not None:
        cfg.impact_coef_bp = float(args.impact_coef_bp)
    if getattr(args, "impact_exponent", None) is not None:
        cfg.impact_exponent = float(args.impact_exponent)
    if getattr(args, "impact_ref_notional", None) is not None:
        cfg.impact_ref_notional = str(args.impact_ref_notional)
    if getattr(args, "impact_volume_data", None) is not None:
        cfg.impact_volume_data_path = str(args.impact_volume_data)
    if getattr(args, "impact_volume_column", None) is not None:
        cfg.impact_volume_column = str(args.impact_volume_column)
    if getattr(args, "impact_volume_unit", None) is not None:
        cfg.impact_volume_unit = str(args.impact_volume_unit)
    if getattr(args, "impact_volume_timestamp_column", None) is not None:
        cfg.impact_volume_timestamp_column = str(args.impact_volume_timestamp_column)
    if getattr(args, "impact_volume_max_staleness_min", None) is not None:
        cfg.impact_volume_max_staleness_minutes = float(args.impact_volume_max_staleness_min)
    if getattr(args, "impact_volume_price_floor_dkk_per_mwh", None) is not None:
        cfg.impact_volume_price_floor_dkk_per_mwh = float(args.impact_volume_price_floor_dkk_per_mwh)
    for _name in (
        "liquidity_participation_cap_fraction",
        "liquidity_volume_multiplier",
        "liquidity_min_volume_mwh",
        "liquidity_tail_impact_threshold_dkk_per_mwh",
        "liquidity_tail_impact_multiplier",
        "liquidity_tail_impact_power",
        "liquidity_tail_impact_max_multiplier",
        "collateral_notional_margin_fraction",
        "collateral_stress_loss_fraction",
        "collateral_stress_price_dkk_per_mwh",
        "collateral_funding_rate_annual",
        "collateral_tradeable_haircut",
    ):
        if getattr(args, _name, None) is not None:
            setattr(cfg, _name, float(getattr(args, _name)))
    if getattr(args, "liquidity_volume_source", None) is not None:
        cfg.liquidity_volume_source = str(args.liquidity_volume_source)
    if getattr(args, "enable_collateral_cash_drag", None) is not None:
        cfg.enable_collateral_cash_drag = bool(args.enable_collateral_cash_drag)
    if bool(getattr(args, "disable_collateral_cash_drag", False)):
        cfg.enable_collateral_cash_drag = False
    if getattr(args, "investment_freq", None) is not None:
        cfg.investment_freq = int(args.investment_freq)
    if getattr(args, "meta_freq_min", None) is not None:
        cfg.meta_freq_min = int(args.meta_freq_min)
    if getattr(args, "meta_freq_max", None) is not None:
        cfg.meta_freq_max = int(args.meta_freq_max)
    if getattr(args, "global_norm_mode", None) is not None:
        cfg.use_global_normalization = bool(str(args.global_norm_mode).strip().lower() == "global")
    if not bool(getattr(cfg, "use_global_normalization", False)):
        cfg.rolling_past_history_enable = True
        roll_dir = str(getattr(args, "rolling_past_history_dir", "") or "").strip()
        cfg.rolling_past_history_dir = roll_dir or EVAL_ROLLING_PAST_HISTORY_DIR
    # Forecast utilization opt-in: typically inherited from training_config.json via
    # _hydrate_eval_config_from_training_config; CLI override accepted for
    # symmetry with training.
    if bool(getattr(args, "enable_forecast_utilization", False)):
        cfg.enable_forecast_utilization = True
    cfg.risk_controller_rule_based = bool(getattr(args, "risk_controller_rule_based", False))
    cfg.meta_controller_rule_based = bool(getattr(args, "meta_controller_rule_based", False))
    cfg.agent_policies = [dict(p) for p in getattr(cfg, "agent_policies", [])]
    if cfg.risk_controller_rule_based and len(cfg.agent_policies) > 2:
        cfg.agent_policies[2]["mode"] = "RULE"
    if cfg.meta_controller_rule_based and len(cfg.agent_policies) > 3:
        cfg.agent_policies[3]["mode"] = "RULE"
    apply_forecast_prior_overrides(cfg, args)
    return cfg


def _running_moments_from_values(values):
    """Convert a historical series into the persistent online-state format."""
    try:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        count = float(arr.size)
        mean = float(np.mean(arr))
        var = float(np.var(arr, ddof=0))
        return {
            "count": count,
            "mean": mean,
            "m2": max(var * count, 0.0),
        }
    except Exception:
        return None


def _robust_p95_scale(values, min_scale=0.1):
    try:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return max(float(min_scale), 1.0)
        p95_val = float(np.percentile(np.abs(arr), 95))
        if np.isfinite(p95_val) and p95_val > 0.0:
            return max(p95_val, float(min_scale))
    except Exception:
        pass
    return max(float(min_scale), 1.0)


def _prime_eval_rolling_past_from_history(cfg):
    """Prime evaluation from the complete trailing causal history window."""
    history_dir = str(getattr(cfg, "rolling_past_history_dir", "") or "").strip()
    if not history_dir:
        raise RuntimeError("[ROLLING_PAST][EVAL] rolling_past mode requires rolling_past_history_dir")
    pattern = os.path.join(history_dir, "history_*.csv")
    candidates = []
    for p in glob.glob(pattern):
        name = os.path.basename(str(p))
        m = re.match(r"^history_(\d+)\.csv$", name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise RuntimeError(f"[ROLLING_PAST][EVAL] No history_*.csv file found in {history_dir}")

    tail_days = int(getattr(cfg, "rolling_past_history_tail_days", 365) or 365)
    rows_per_day = int(getattr(cfg, "rolling_past_history_rows_per_day", 144) or 144)
    tail_rows = max(max(tail_days, 1) * max(rows_per_day, 1), max(rows_per_day, 1))

    candidates.sort(key=lambda x: x[0])
    frames = []
    source_paths = []
    rows_accum = 0
    for _, history_csv_path in reversed(candidates):
        frame = load_energy_data(
            history_csv_path,
            convert_to_raw_units=False,
            config=cfg,
            mw_scale_overrides=None,
        )
        frames.insert(0, frame)
        source_paths.insert(0, history_csv_path)
        rows_accum += int(len(frame))
        if rows_accum >= tail_rows:
            break
    history_df = pd.concat(frames, axis=0, ignore_index=True)
    if "timestamp" in history_df.columns:
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], errors="raise")
        history_df = (
            history_df.sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )
    if len(history_df) > tail_rows:
        history_df = history_df.iloc[-tail_rows:].copy()

    if "price" not in history_df.columns:
        raise RuntimeError(f"Evaluation rolling_past bootstrap missing price column: {source_paths}")

    price_state = _running_moments_from_values(history_df["price"].to_numpy())
    if price_state is None:
        raise RuntimeError(f"Evaluation rolling_past bootstrap could not build price state: {source_paths}")

    cfg.rolling_past_price_state = price_state
    cfg.rolling_past_wind_scale = _robust_p95_scale(history_df["wind"].to_numpy(), min_scale=0.1)
    cfg.rolling_past_solar_scale = _robust_p95_scale(history_df["solar"].to_numpy(), min_scale=0.1)
    cfg.rolling_past_hydro_scale = _robust_p95_scale(history_df["hydro"].to_numpy(), min_scale=0.1)
    cfg.rolling_past_load_scale = _robust_p95_scale(history_df["load"].to_numpy(), min_scale=0.1)
    cfg.rolling_past_eval_history_files = [str(path) for path in source_paths]
    cfg.rolling_past_eval_history_rows = int(len(history_df))
    print(
        f"[ROLLING_PAST][EVAL] Bootstrapped from {len(source_paths)} causal files "
        f"({os.path.basename(source_paths[0])}..{os.path.basename(source_paths[-1])}, "
        f"rows={len(history_df)}, price_mean={float(price_state['mean']):.3f})"
    )


def _hydrate_eval_config_from_training_config(cfg, final_models_dir: str):
    """Load Tier-specific evaluation overrides from the saved training_config.json."""
    config_path = os.path.join(final_models_dir, "training_config.json")
    if not os.path.isfile(config_path):
        return cfg
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except Exception:
        return cfg

    final_cfg = saved.get("final_config", {}) or {}
    for key, value in dict(final_cfg).items():
        if value is None:
            continue
        if hasattr(cfg, str(key)):
            setattr(cfg, str(key), value)

    runtime_contract = saved.get("runtime_contract", {}) or {}
    if isinstance(runtime_contract, dict):
        mode = str(runtime_contract.get("global_norm_mode", "") or "").strip().lower()
        if mode:
            cfg.use_global_normalization = bool(mode == "global")
        for key in ("rolling_past_history_dir", "investment_freq", "meta_freq_min", "meta_freq_max"):
            if key in runtime_contract and hasattr(cfg, key):
                setattr(cfg, key, runtime_contract[key])
        prior_contract = runtime_contract.get("forecast_prior", {}) or {}
        if isinstance(prior_contract, dict):
            for key, value in prior_contract.items():
                if value is None:
                    continue
                if hasattr(cfg, str(key)):
                    setattr(cfg, str(key), value)
            if "forecast_prior_horizon_steps" in prior_contract:
                horizon = int(prior_contract["forecast_prior_horizon_steps"])
                horizons = dict(getattr(cfg, "forecast_horizons", {}) or {})
                horizons["short"] = horizon
                cfg.forecast_horizons = horizons
            if "forecast_prior_denom_floor" in prior_contract:
                cfg.forecast_prior_denom_floor = float(prior_contract["forecast_prior_denom_floor"])
        mtm_contract = runtime_contract.get("mtm", {}) or {}
        if isinstance(mtm_contract, dict):
            for key, value in mtm_contract.items():
                if value is None:
                    continue
                if hasattr(cfg, str(key)):
                    setattr(cfg, str(key), value)
        sizing_contract = runtime_contract.get("sizing", {}) or {}
        if isinstance(sizing_contract, dict):
            for key, value in sizing_contract.items():
                if value is None:
                    continue
                if hasattr(cfg, str(key)):
                    setattr(cfg, str(key), value)
        execution_contract = runtime_contract.get("execution", {}) or {}
        if isinstance(execution_contract, dict):
            for key, value in execution_contract.items():
                if value is None:
                    continue
                if hasattr(cfg, str(key)):
                    setattr(cfg, str(key), value)
        controller_contract = runtime_contract.get("controllers", {}) or {}
        if isinstance(controller_contract, dict):
            for key, value in controller_contract.items():
                if value is None:
                    continue
                if hasattr(cfg, str(key)):
                    setattr(cfg, str(key), bool(value))

    flags = saved.get("flags", {}) or {}
    if "enable_forecast_utilization" in flags:
        cfg.enable_forecast_utilization = bool(flags.get("enable_forecast_utilization"))
    for key in ("risk_controller_rule_based", "meta_controller_rule_based"):
        if key in flags and hasattr(cfg, key):
            setattr(cfg, key, bool(flags.get(key)))

    cfg.agent_policies = [dict(p) for p in getattr(cfg, "agent_policies", [])]
    if bool(getattr(cfg, "risk_controller_rule_based", False)) and len(cfg.agent_policies) > 2:
        cfg.agent_policies[2]["mode"] = "RULE"
    if bool(getattr(cfg, "meta_controller_rule_based", False)) and len(cfg.agent_policies) > 3:
        cfg.agent_policies[3]["mode"] = "RULE"

    return cfg


def _load_runtime_contract_hash_from_training_config(final_models_dir: str) -> str:
    config_path = os.path.join(final_models_dir, "training_config.json")
    if not os.path.isfile(config_path):
        return ""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        return str(saved.get("runtime_contract_hash", "") or "").strip().lower()
    except Exception:
        return ""


def _load_runtime_contract_from_training_config(final_models_dir: str) -> dict:
    config_path = os.path.join(final_models_dir, "training_config.json")
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        runtime_contract = saved.get("runtime_contract", {}) or {}
        return runtime_contract if isinstance(runtime_contract, dict) else {}
    except Exception:
        return {}


def run_tier_suite_evaluation(eval_data: pd.DataFrame, args) -> Dict[str, Any]:
    """Tier1 evaluation on unseen data."""
    eval_forecast_overlay = bool(getattr(args, "eval_forecast_overlay", False))
    results: Dict[str, Any] = {
        "evaluation_mode": "tier1",
        "eval_data": args.eval_data,
        "seed": int(getattr(args, "seed", 42)),
        "eval_forecast_overlay": eval_forecast_overlay,
        "tiers": {},
    }

    steps = args.eval_steps if args.eval_steps is not None else (len(eval_data) - 1)
    steps = int(max(1, steps))

    tier_out = args.output_dir
    env_log_dir = None
    log_sleeve = bool(getattr(args, "log_sleeve", False))
    # sleeve-supplement task
    if log_sleeve:
        env_log_dir = os.path.join(tier_out, "env_logs")

    run_dir = args.tier1_dir
    policy_final_models_dir = os.path.join(run_dir, "final_models")
    models = load_checkpoint_models(policy_final_models_dir)
    cfg = _hydrate_eval_config_from_training_config(_build_eval_config(args), policy_final_models_dir)
    trained_runtime_contract = _load_runtime_contract_from_training_config(policy_final_models_dir)
    trained_forecast_enabled = bool(trained_runtime_contract.get("enable_forecast_utilization", False))
    if getattr(args, "eval_distribution_rate", None) is not None:
        cfg.eval_distribution_rate = float(args.eval_distribution_rate)
    if getattr(args, "distribution_rate", None) is not None:
        cfg.distribution_rate = float(args.distribution_rate)
    if getattr(args, "mtm_return_model", None) is not None:
        cfg.mtm_return_model = str(args.mtm_return_model).strip().lower()
    if getattr(args, "mtm_reference_price_dkk_per_mwh", None) is not None:
        cfg.mtm_reference_price_dkk_per_mwh = float(args.mtm_reference_price_dkk_per_mwh)
    if getattr(args, "mtm_settlement_horizon_steps", None) is not None:
        cfg.mtm_settlement_horizon_steps = int(args.mtm_settlement_horizon_steps)
    if getattr(args, "mtm_entry_price_mode", None) is not None:
        cfg.mtm_entry_price_mode = str(args.mtm_entry_price_mode).strip().lower()
    if getattr(args, "mtm_horizon_payoff_denominator_mode", None) is not None:
        cfg.mtm_horizon_payoff_denominator_mode = (
            str(args.mtm_horizon_payoff_denominator_mode).strip().lower().replace("-", "_")
        )
    if getattr(args, "mtm_settlement_price_mode", None) is not None:
        cfg.mtm_settlement_price_mode = (
            str(args.mtm_settlement_price_mode).strip().lower().replace("-", "_")
        )
    if getattr(args, "mtm_basis_price_data_path", None) is not None:
        cfg.mtm_basis_price_data_path = str(args.mtm_basis_price_data_path)
    if getattr(args, "mtm_basis_price_column", None) is not None:
        cfg.mtm_basis_price_column = str(args.mtm_basis_price_column)
    if getattr(args, "mtm_basis_timestamp_column", None) is not None:
        cfg.mtm_basis_timestamp_column = str(args.mtm_basis_timestamp_column)
    if getattr(args, "mtm_basis_scale", None) is not None:
        cfg.mtm_basis_scale = float(args.mtm_basis_scale)
    if getattr(args, "mtm_basis_centering_mode", None) is not None:
        cfg.mtm_basis_centering_mode = (
            str(args.mtm_basis_centering_mode).strip().lower().replace("-", "_")
        )
    if getattr(args, "mtm_basis_centering_window_steps", None) is not None:
        cfg.mtm_basis_centering_window_steps = int(args.mtm_basis_centering_window_steps)
    if getattr(args, "mtm_external_settlement_price_data_path", None) is not None:
        cfg.mtm_external_settlement_price_data_path = str(args.mtm_external_settlement_price_data_path)
    if getattr(args, "mtm_external_settlement_price_column", None) is not None:
        cfg.mtm_external_settlement_price_column = str(args.mtm_external_settlement_price_column)
    if getattr(args, "mtm_external_settlement_timestamp_column", None) is not None:
        cfg.mtm_external_settlement_timestamp_column = str(args.mtm_external_settlement_timestamp_column)
    if getattr(args, "mtm_external_settlement_min_price_dkk_per_mwh", None) is not None:
        cfg.mtm_external_settlement_min_price_dkk_per_mwh = float(args.mtm_external_settlement_min_price_dkk_per_mwh)
    if getattr(args, "mtm_external_settlement_max_price_dkk_per_mwh", None) is not None:
        cfg.mtm_external_settlement_max_price_dkk_per_mwh = float(args.mtm_external_settlement_max_price_dkk_per_mwh)
    if bool(getattr(args, "disable_mtm_return_cap", False)):
        cfg.mtm_apply_price_return_cap = False
    if getattr(args, "mtm_price_return_cap_min", None) is not None:
        cfg.mtm_price_return_cap_min = float(args.mtm_price_return_cap_min)
    if getattr(args, "mtm_price_return_cap_max", None) is not None:
        cfg.mtm_price_return_cap_max = float(args.mtm_price_return_cap_max)
    if getattr(args, "investor_notional_sizing_base", None) is not None:
        cfg.investor_notional_sizing_base = str(args.investor_notional_sizing_base).strip().lower()
    if getattr(args, "max_position_size", None) is not None:
        cfg.max_position_size = float(args.max_position_size)
    if getattr(args, "capital_allocation_fraction", None) is not None:
        cfg.capital_allocation_fraction = float(args.capital_allocation_fraction)
    if getattr(args, "mtm_loss_exit_threshold_pct", None) is not None:
        cfg.mtm_loss_exit_threshold_pct = float(args.mtm_loss_exit_threshold_pct)
    if getattr(args, "friction_cost_multiplier", None) is not None:
        cfg.friction_cost_multiplier = float(args.friction_cost_multiplier)
    _apply_market_fee_overrides(cfg, args)
    if getattr(args, "no_trade_threshold", None) is not None:
        cfg.no_trade_threshold = float(args.no_trade_threshold)
    if getattr(args, "no_trade_threshold_reference", None) is not None:
        cfg.no_trade_threshold_reference = str(args.no_trade_threshold_reference)
    if getattr(args, "half_spread_bp", None) is not None:
        cfg.half_spread_bp = float(args.half_spread_bp)
    if eval_forecast_overlay:
        cfg.enable_forecast_utilization = True
        apply_forecast_prior_overrides(cfg, args)
    # market-impact task
    if getattr(args, "impact_coef_bp", None) is not None:
        cfg.impact_coef_bp = float(args.impact_coef_bp)
    if getattr(args, "impact_exponent", None) is not None:
        cfg.impact_exponent = float(args.impact_exponent)
    if getattr(args, "impact_ref_notional", None) is not None:
        cfg.impact_ref_notional = str(args.impact_ref_notional)
    if getattr(args, "impact_volume_data", None) is not None:
        cfg.impact_volume_data_path = str(args.impact_volume_data)
    if getattr(args, "impact_volume_column", None) is not None:
        cfg.impact_volume_column = str(args.impact_volume_column)
    if getattr(args, "impact_volume_unit", None) is not None:
        cfg.impact_volume_unit = str(args.impact_volume_unit)
    if getattr(args, "impact_volume_timestamp_column", None) is not None:
        cfg.impact_volume_timestamp_column = str(args.impact_volume_timestamp_column)
    if getattr(args, "impact_volume_max_staleness_min", None) is not None:
        cfg.impact_volume_max_staleness_minutes = float(args.impact_volume_max_staleness_min)
    if getattr(args, "impact_volume_price_floor_dkk_per_mwh", None) is not None:
        cfg.impact_volume_price_floor_dkk_per_mwh = float(args.impact_volume_price_floor_dkk_per_mwh)
    for _name in (
        "liquidity_participation_cap_fraction",
        "liquidity_volume_multiplier",
        "liquidity_min_volume_mwh",
        "liquidity_tail_impact_threshold_dkk_per_mwh",
        "liquidity_tail_impact_multiplier",
        "liquidity_tail_impact_power",
        "liquidity_tail_impact_max_multiplier",
        "collateral_notional_margin_fraction",
        "collateral_stress_loss_fraction",
        "collateral_stress_price_dkk_per_mwh",
        "collateral_funding_rate_annual",
        "collateral_tradeable_haircut",
    ):
        if getattr(args, _name, None) is not None:
            setattr(cfg, _name, float(getattr(args, _name)))
    if getattr(args, "liquidity_volume_source", None) is not None:
        cfg.liquidity_volume_source = str(args.liquidity_volume_source)
    if getattr(args, "enable_collateral_cash_drag", None) is not None:
        cfg.enable_collateral_cash_drag = bool(args.enable_collateral_cash_drag)
    if bool(getattr(args, "disable_collateral_cash_drag", False)):
        cfg.enable_collateral_cash_drag = False
    cfg.enable_episode_csv_logs = bool(log_sleeve)

    runtime_contract_kwargs = {
        "global_norm_mode": "global" if bool(getattr(cfg, "use_global_normalization", False)) else "rolling_past",
        "rolling_past_history_dir": str(getattr(cfg, "rolling_past_history_dir", "") or ""),
        "investment_freq": int(getattr(cfg, "investment_freq", getattr(args, "investment_freq", 6)) or 6),
        "meta_freq_min": int(getattr(cfg, "meta_freq_min", getattr(args, "meta_freq_min", 6)) or 6),
        "meta_freq_max": int(getattr(cfg, "meta_freq_max", getattr(args, "meta_freq_max", 6)) or 6),
        "enable_forecast_utilization": bool(getattr(cfg, "enable_forecast_utilization", False)),
        "forecast_prior_settings": forecast_prior_contract_settings(cfg),
        "mtm_settings": mtm_contract_settings(cfg),
        "sizing_settings": sizing_contract_settings(cfg),
        "execution_settings": execution_contract_settings(cfg),
        "controller_settings": controller_contract_settings(cfg),
    }
    eval_runtime_contract = build_runtime_contract(
        **runtime_contract_kwargs,
        log_sleeve=log_sleeve,
    )
    eval_runtime_match_contract = build_runtime_contract(
        **runtime_contract_kwargs,
        log_sleeve=False,
    )
    policy_match_kwargs = dict(runtime_contract_kwargs)
    policy_match_kwargs["enable_forecast_utilization"] = trained_forecast_enabled
    if not trained_forecast_enabled:
        policy_match_kwargs["forecast_prior_settings"] = None
    eval_policy_match_contract = build_runtime_contract(
        **policy_match_kwargs,
        log_sleeve=False,
    )
    eval_policy_match_hash = runtime_contract_hash(eval_policy_match_contract)
    eval_runtime_match_hash = runtime_contract_hash(eval_runtime_match_contract)
    eval_runtime_hash = runtime_contract_hash(eval_runtime_contract)
    train_hash = _load_runtime_contract_hash_from_training_config(policy_final_models_dir)
    overlay_contract_exception = bool(
        eval_forecast_overlay
        and train_hash
        and eval_policy_match_hash == train_hash
        and eval_runtime_match_hash != train_hash
    )
    # [SIM_AUDIT PATCH] contract mismatch may be allowed only for audit evaluations
    _sim_audit_mismatch = bool(train_hash and eval_runtime_hash != train_hash and eval_runtime_match_hash != train_hash and not overlay_contract_exception)
    if _sim_audit_mismatch and os.environ.get('SIM_AUDIT_ALLOW_PROTOCOL_MISMATCH') == '1':
        print(f'[SIM_AUDIT] protocol mismatch ALLOWED for audit evaluation: train={train_hash}, eval={eval_runtime_hash}')
    elif _sim_audit_mismatch:
        raise RuntimeError(
            "[EVAL_RUNTIME_CONTRACT] training/eval hash mismatch for tier1: "
            f"train={train_hash}, eval={eval_runtime_hash}"
        )
    if overlay_contract_exception:
        print(
            "[EVAL_RUNTIME_CONTRACT][FORECAST_OVERLAY] allowed forecast-only eval overlay "
            f"for tier1: train={train_hash}, policy_match={eval_policy_match_hash}, eval={eval_runtime_hash}"
        )
    if not bool(getattr(cfg, "use_global_normalization", False)):
        _prime_eval_rolling_past_from_history(cfg)

    eval_forecast_cache_dir = args.forecast_cache_dir
    if bool(getattr(cfg, "enable_forecast_utilization", False)):
        forecast_ctx = _resolve_reserved_eval_forecast_context(args, output_dir=tier_out, config=cfg)
        eval_forecast_cache_dir = forecast_ctx["forecast_cache_dir"]

    # [SIM_AUDIT PATCH] optional config overrides for audit evaluations
    _sim_audit_overrides = json.loads(os.environ.get('SIM_AUDIT_CFG_OVERRIDES', '') or '{}')
    for _k, _v in _sim_audit_overrides.items():
        if not hasattr(cfg, _k):
            raise AttributeError(f'[SIM_AUDIT] unknown config attribute {_k}')
        setattr(cfg, _k, _v)
    if _sim_audit_overrides:
        print(f'[SIM_AUDIT] config overrides applied: {_sim_audit_overrides}')
    env = create_evaluation_environment(
        eval_data,
        output_dir=tier_out,
        investment_freq=int(getattr(cfg, "investment_freq", args.investment_freq) or args.investment_freq),
        config=cfg,
        env_log_dir=env_log_dir,
        forecast_cache_dir=eval_forecast_cache_dir,
        fail_fast=True,
    )

    tier_metrics = run_checkpoint_evaluation(models, env, eval_data, steps)
    if tier_metrics is None:
        tier_metrics = {"status": "failed", "error": "evaluation_failed"}
    else:
        if env_log_dir:
            try:
                # sleeve-supplement task: flush debug CSVs before reading them into JSON metrics.
                _close_env_debug_logger(env)
                debug_log_path = _find_env_debug_log(env_log_dir)
                tier_metrics["env_log_dir"] = env_log_dir
                tier_metrics["env_debug_log"] = debug_log_path or ""
                if debug_log_path:
                    tier_metrics["sleeve_metrics"] = _compute_sleeve_metrics_from_env_log(
                        debug_log_path,
                        dkk_to_usd_rate=float(getattr(cfg, "dkk_to_usd_rate", 0.145) or 0.145),
                        primary_sharpe_mode=str(getattr(args, "sleeve_sharpe_mode", "daily_hac_7") or "daily_hac_7"),
                    )
            except Exception as e:
                tier_metrics["sleeve_metrics"] = {"sleeve_metrics_error": str(e)}
        # sleeve-supplement task
        if log_sleeve:
            tier_metrics["log_sleeve"] = True
            tier_metrics["eval_runtime_contract_hash"] = eval_runtime_hash
        tier_metrics["train_runtime_contract_hash"] = train_hash
        tier_metrics["eval_policy_match_contract_hash"] = eval_policy_match_hash
        tier_metrics["eval_runtime_contract_hash"] = eval_runtime_hash
        tier_metrics["engine_file_hashes"] = engine_file_hashes()
        # [SIM_AUDIT PATCH] provenance of audit evaluations
        tier_metrics['sim_audit'] = {'protocol_mismatch': bool(_sim_audit_mismatch), 'mismatch_allowed': os.environ.get('SIM_AUDIT_ALLOW_PROTOCOL_MISMATCH') == '1', 'cfg_overrides': _sim_audit_overrides, 'variant': os.environ.get('SIM_AUDIT_VARIANT', '')}
        tier_metrics["eval_forecast_overlay"] = eval_forecast_overlay
        tier_metrics["eval_runtime_contract_exception"] = (
            "forecast_overlay_only" if overlay_contract_exception else ""
        )
        tier_metrics.update({
            "status": "completed",
            "run_dir": run_dir,
            "final_models_dir": policy_final_models_dir,
            "evaluation_mode": "tier1_forecast_overlay" if eval_forecast_overlay else "tier1",
            "variant": "tier1_forecast_overlay" if eval_forecast_overlay else "tier1",
        })
    results["tiers"]["tier1"] = tier_metrics
    return results


def _pct(x: Any) -> float:
    """Convert a return-like value to percent (supports fraction or percent already)."""
    try:
        v = float(x)
    except Exception:
        return 0.0
    # Heuristic: if magnitude looks like already-percent (e.g. 4.2), keep it.
    if abs(v) > 1.5:
        return v
    return v * 100.0


def _extract_action_inference_success_rate(results: Dict[str, Any]) -> Optional[float]:
    """Read the current or compatibility action-inference success metric from result dictionaries."""
    for key in ("action_inference_success_rate", "prediction_success_rate"):
        if key not in results:
            continue
        try:
            return float(results[key])
        except Exception:
            return None
    return None


def _find_env_debug_log(env_log_dir: str) -> Optional[str]:
    """Best-effort lookup of the per-episode debug CSV produced by the env logger."""
    try:
        # Common path in this codebase.
        p = os.path.join(env_log_dir, "tier1_debug_ep0.csv")
        if os.path.isfile(p):
            return p

        # Fallback: any *debug_ep0.csv in that directory.
        candidates = glob.glob(os.path.join(env_log_dir, "*debug_ep0.csv"))
        return candidates[0] if candidates else None
    except Exception:
        return None


def _close_env_debug_logger(eval_env) -> None:
    """Flush and close the environment debug logger if sleeve logging is active."""
    seen = set()
    for obj in (eval_env, getattr(eval_env, "env", None)):
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        tracker = getattr(obj, "debug_tracker", None)
        if tracker is not None and hasattr(tracker, "close"):
            try:
                tracker.close()
            except Exception:
                pass


def _compute_sleeve_metrics_from_env_log(
    debug_csv_path: str,
    dkk_to_usd_rate: float = 0.145,
    primary_sharpe_mode: str = "daily",
) -> Dict[str, Any]:
    """Split total NAV into (operating sleeve) and (trading sleeve) from env debug logs.

    Why this exists:
      Total NAV in this hybrid-fund simulator includes the physical book value of assets.
      That makes total-NAV volatility/drawdown look tiny even when the financial trading
      sleeve takes meaningful risk. For reporting/paper figures, it is often clearer to
      decompose NAV into:
        - operating_sleeve = physical_book_value + accumulated_operational_revenue
        - trading_sleeve   = trading_cash + financial_mtm
    """
    metrics: Dict[str, Any] = {}
    if not debug_csv_path or not os.path.isfile(debug_csv_path):
        return metrics

    required = {
        "fund_nav_dkk",
        "trading_cash_dkk",
        "physical_book_value_dkk",
        "accumulated_operational_revenue_dkk",
        "financial_mtm_dkk",
    }
    optional = {
        "financial_exposure_dkk",
        "decision_step",
        "total_distributions_dkk",
        "distribution_adjusted_nav_dkk",
        "distribution_adjusted_trading_sleeve_dkk",
        "timestamp",
        "cumulative_trading_costs_dkk",
        "cumulative_impact_costs_dkk",
        "cumulative_volume_transaction_fees_dkk",
        "cumulative_market_access_fees_dkk",
        "market_access_fee_step_dkk",
        "cumulative_collateral_funding_costs_dkk",
        "collateral_required_dkk",
        "collateral_funding_cost_dkk",
        "collateral_open_notional_dkk",
        "collateral_open_volume_mwh",
        "liquidity_market_volume_mwh",
        "liquidity_causal_market_volume_mwh",
        "liquidity_volume_cap_mwh",
        "liquidity_requested_volume_mwh",
        "liquidity_executed_volume_mwh",
        "liquidity_participation",
        "liquidity_scale",
        "liquidity_tail_impact_multiplier",
        "liquidity_tail_spread_dkk_per_mwh",
        "market_impact_ref_notional_dkk",
        "market_impact_participation",
        "market_impact_bp",
        "no_trade_threshold_dkk",
        "no_trade_threshold_reference_dkk",
        "horizon_settlement_pnl_dkk",
        "horizon_settlement_count",
        "forecast_prior_active",
        "forecast_prior_exposure",
        "forecast_only_prior_exposure",
        "forecast_beta_exposure",
        "forecast_beta_strength",
        "forecast_beta_active",
        "forecast_beta_payoff_lcb",
        "forecast_beta_count",
        "forecast_prior_skill",
        "forecast_prior_hit_lcb",
        "forecast_prior_hit_rate",
        "forecast_prior_direction_confidence",
        "forecast_prior_confidence_weight",
        "forecast_prior_calibration_count",
        "forecast_prior_calibration_total",
        "forecast_residual_authority",
        "forecast_base_residual_authority",
        "forecast_residual_evidence_multiplier",
        "forecast_residual_to_prior_ratio",
        "trading_sleeve_margin_active",
        "trading_sleeve_margin_step",
        "trading_sleeve_margin_threshold_dkk",
        "trading_sleeve_margin_gap_dkk",
        "forecast_adjustment",
        "forecast_alignment",
        "forecast_mode_reason",
        "forecast_policy_hit_lcb",
        "forecast_policy_relative_hit_lcb",
        "forecast_policy_relative_advantage_lcb",
        "forecast_policy_relative_strength",
        "forecast_policy_relative_count",
        "feasible_action_enabled",
        "feasible_action_coordinate",
        "feasible_action_capacity_mwh",
        "feasible_action_capacity_fraction",
        "feasible_action_hard_capacity_mwh",
        "feasible_action_hard_capacity_fraction",
        "feasible_action_evidence_authority",
        "feasible_action_capacity_conditioned_on_evidence",
        "feasible_action_anchor_requested_mwh",
        "feasible_action_anchor_quantity_mwh",
        "feasible_action_policy_requested_mwh",
        "feasible_action_policy_quantity_mwh",
        "feasible_action_adjustment_mwh",
        "feasible_action_anchor_fraction",
        "feasible_action_target_fraction",
        "feasible_action_anchor_no_trade_hold",
        "feasible_action_policy_no_trade_hold",
        "feasible_action_liquidity_capacity_mwh",
        "feasible_action_liquidity_capacity_fraction",
        "feasible_action_collateral_headroom_fraction",
        "feasible_action_no_trade_threshold_mwh",
        "feasible_action_policy_cost_dkk",
        "feasible_action_anchor_cost_dkk",
        "feasible_action_policy_funding_cost_dkk",
        "feasible_action_anchor_funding_cost_dkk",
        "feasible_action_count",
        "feasible_action_advantage_mean",
        "feasible_action_cvar_shortfall",
        "feasible_action_dual_lambda",
        "feasible_action_last_advantage_dkk",
        "feasible_action_reward_step",
        "forecast_mirror_enabled",
        "forecast_mirror_action",
        "forecast_mirror_requested_quantity_mwh",
        "forecast_mirror_executed_quantity_mwh",
        "forecast_mirror_hard_capacity_mwh",
        "forecast_mirror_prior_quantity_mwh",
        "forecast_mirror_prior_coordinate",
        "forecast_mirror_evidence",
        "forecast_mirror_trust",
        "forecast_mirror_exponent",
        "forecast_mirror_base_action_mean",
        "forecast_mirror_final_action_mean",
        "forecast_mirror_trust_gradient",
        "forecast_mirror_counterfactual_score",
        "forecast_mirror_pending_count",
    }

    try:
        with open(debug_csv_path, "r", encoding="utf-8") as f:
            header = (f.readline() or "").strip().split(",")
        have = set(h for h in header if h)
    except Exception:
        have = set()

    if not required.issubset(have):
        # If the log schema changes, skip gracefully.
        missing = sorted(list(required - have))
        metrics["sleeve_metrics_error"] = f"missing_required_columns:{','.join(missing)}"
        return metrics

    usecols = sorted(list(required | (optional & have)))

    try:
        df = pd.read_csv(debug_csv_path, usecols=usecols)
    except Exception as e:
        metrics["sleeve_metrics_error"] = f"read_failed:{e}"
        return metrics

    # Convert to USD (all logs are in DKK). Prefer total-wealth series when
    # available; keep raw NAV as explicit reported/ex-distribution diagnostics.
    reported_nav_usd = df["fund_nav_dkk"].to_numpy(dtype=float) * dkk_to_usd_rate
    if "distribution_adjusted_nav_dkk" in df.columns:
        nav_usd = df["distribution_adjusted_nav_dkk"].to_numpy(dtype=float) * dkk_to_usd_rate
    else:
        nav_usd = reported_nav_usd

    reported_trading_usd = (
        (df["trading_cash_dkk"].to_numpy(dtype=float) + df["financial_mtm_dkk"].to_numpy(dtype=float))
        * dkk_to_usd_rate
    )
    if "distribution_adjusted_trading_sleeve_dkk" in df.columns:
        trading_usd = df["distribution_adjusted_trading_sleeve_dkk"].to_numpy(dtype=float) * dkk_to_usd_rate
    else:
        trading_usd = reported_trading_usd
    operating_usd = (
        (df["physical_book_value_dkk"].to_numpy(dtype=float) + df["accumulated_operational_revenue_dkk"].to_numpy(dtype=float))
        * dkk_to_usd_rate
    )

    if len(nav_usd) == 0:
        return metrics

    total_gain = float(nav_usd[-1] - nav_usd[0])
    trading_gain = float(trading_usd[-1] - trading_usd[0])
    operating_gain = float(operating_usd[-1] - operating_usd[0])

    metrics.update({
        "sleeve_total_initial_usd": float(nav_usd[0]),
        "sleeve_total_final_usd": float(nav_usd[-1]),
        "sleeve_total_gain_usd": total_gain,
        "sleeve_reported_nav_initial_usd": float(reported_nav_usd[0]),
        "sleeve_reported_nav_final_usd": float(reported_nav_usd[-1]),
        "sleeve_trading_initial_usd": float(trading_usd[0]),
        "sleeve_trading_final_usd": float(trading_usd[-1]),
        "sleeve_trading_gain_usd": trading_gain,
        "sleeve_reported_trading_initial_usd": float(reported_trading_usd[0]),
        "sleeve_reported_trading_final_usd": float(reported_trading_usd[-1]),
        "sleeve_operating_initial_usd": float(operating_usd[0]),
        "sleeve_operating_final_usd": float(operating_usd[-1]),
        "sleeve_operating_gain_usd": operating_gain,
    })
    if "total_distributions_dkk" in df.columns:
        metrics["sleeve_total_distributions_usd"] = float(df["total_distributions_dkk"].to_numpy(dtype=float)[-1] * dkk_to_usd_rate)

    if total_gain != 0.0:
        metrics["sleeve_trading_gain_share"] = float(trading_gain / total_gain)
    else:
        metrics["sleeve_trading_gain_share"] = 0.0

    if trading_usd[0] != 0.0:
        metrics["sleeve_trading_return_pct"] = float((trading_usd[-1] / trading_usd[0] - 1.0) * 100.0)
    else:
        metrics["sleeve_trading_return_pct"] = 0.0

    if operating_usd[0] != 0.0:
        metrics["sleeve_operating_return_pct"] = float((operating_usd[-1] / operating_usd[0] - 1.0) * 100.0)
    else:
        metrics["sleeve_operating_return_pct"] = 0.0

    # Trading sleeve risk metrics. The primary paper metric is deliberately not
    # the raw 10-minute annualized Sharpe. Report non-overlapping daily returns,
    # with daily Newey-West/HAC adjustment by default for serial correlation.
    try:
        if len(trading_usd) > 1 and float(trading_usd[0]) != 0.0:
            tr = _series_returns(trading_usd)
            periods_per_year = _infer_periods_per_year(df["timestamp"] if "timestamp" in df.columns else None)
            annual_risk_free_rate = float(getattr(PortfolioAnalysisConfig(), "risk_free_annual", 0.02))
            raw_stats = _path_sharpe(
                trading_usd,
                periods_per_year=periods_per_year,
                annual_risk_free_rate=annual_risk_free_rate,
            )
            daily_path = _subsample_path(trading_usd, 144)
            daily_returns = _series_returns(daily_path)
            daily_stats = _path_sharpe(
                daily_path,
                periods_per_year=365.25,
                annual_risk_free_rate=annual_risk_free_rate,
            )
            daily_zero_rf_stats = _path_sharpe(
                daily_path,
                periods_per_year=365.25,
                annual_risk_free_rate=0.0,
            )
            daily_hac_lag = min(7, max(int(daily_returns.size) - 2, 0))
            daily_hac_vif = _hac_variance_inflation(daily_returns, daily_hac_lag)
            daily_hac_sharpe = (
                float(daily_stats["annualized_sharpe"] / math.sqrt(daily_hac_vif))
                if daily_hac_vif > 0.0 else 0.0
            )
            daily_hac_zero_rf_sharpe = (
                float(daily_zero_rf_stats["annualized_sharpe"] / math.sqrt(daily_hac_vif))
                if daily_hac_vif > 0.0 else 0.0
            )
            hac_lag = 144
            hac_vif = _hac_variance_inflation(tr, hac_lag)
            hac_sharpe = (
                float(raw_stats["annualized_sharpe"] / math.sqrt(hac_vif))
                if hac_vif > 0.0 else 0.0
            )

            peak = np.maximum.accumulate(trading_usd)
            drawdowns = np.where(peak > 0.0, (peak - trading_usd) / peak, 0.0)
            dd = float(np.max(drawdowns)) if len(drawdowns) else 0.0
            margin_active_any = False
            if "trading_sleeve_margin_active" in df.columns:
                margin_values = pd.to_numeric(
                    df["trading_sleeve_margin_active"], errors="coerce"
                ).fillna(0.0).to_numpy(dtype=float)
                margin_active_any = bool(np.any(margin_values > 0.5))
            nonpositive_equity = bool(np.any(trading_usd <= 0.0))
            ruined = bool(margin_active_any or nonpositive_equity or dd >= 0.99)
            ruin_reasons = []
            if margin_active_any:
                ruin_reasons.append("maintenance_margin_triggered")
            if nonpositive_equity:
                ruin_reasons.append("nonpositive_sleeve_equity")
            if dd >= 0.99:
                ruin_reasons.append("drawdown_at_least_99pct")

            mode = str(primary_sharpe_mode or "daily").strip().lower().replace("-", "_")
            if mode in {"hac", "hac144", "hac_lag_144"}:
                mode = "hac_144"
            if mode in {"daily_hac", "daily_hac7", "daily_newey_west", "daily_newey_west_7"}:
                mode = "daily_hac_7"
            if mode not in {"daily", "daily_hac_7", "hac_144"}:
                mode = "daily_hac_7"

            metrics["sleeve_trading_sharpe_primary_mode"] = mode
            metrics["sleeve_trading_ruined"] = ruined
            metrics["sleeve_trading_ruin_reason"] = ";".join(ruin_reasons)
            metrics["sleeve_trading_daily_sharpe_ratio_raw"] = float(daily_stats["annualized_sharpe"])
            metrics["sleeve_trading_daily_hac7_sharpe_ratio_raw"] = float(daily_hac_sharpe)
            metrics["sleeve_trading_daily_sharpe_ratio"] = (
                None if ruined else float(daily_stats["annualized_sharpe"])
            )
            metrics["sleeve_trading_daily_hac7_sharpe_ratio"] = (
                None if ruined else float(daily_hac_sharpe)
            )
            metrics["sleeve_trading_daily_hac7_vif"] = float(daily_hac_vif)
            metrics["sleeve_trading_daily_n_returns"] = float(daily_returns.size)
            metrics["sleeve_trading_annual_risk_free_rate"] = float(annual_risk_free_rate)
            daily_zero_rf_sharpe = float(daily_zero_rf_stats["annualized_sharpe"])
            daily_hac_zero_rf_sharpe = float(daily_hac_zero_rf_sharpe)
            metrics["sleeve_trading_daily_zero_rf_sharpe_ratio_raw"] = daily_zero_rf_sharpe
            metrics["sleeve_trading_daily_hac7_zero_rf_sharpe_ratio_raw"] = daily_hac_zero_rf_sharpe
            metrics["sleeve_trading_daily_zero_rf_sharpe_ratio"] = (
                None if ruined else daily_zero_rf_sharpe
            )
            metrics["sleeve_trading_daily_hac7_zero_rf_sharpe_ratio"] = (
                None if ruined else daily_hac_zero_rf_sharpe
            )

            if mode == "hac_144":
                primary_sharpe = float(hac_sharpe)
                primary_vol = float(raw_stats["annualized_volatility"] * math.sqrt(hac_vif))
                primary_step_vol = float(raw_stats["step_volatility"] * math.sqrt(hac_vif))
            elif mode == "daily_hac_7":
                primary_sharpe = float(daily_hac_sharpe)
                primary_vol = float(daily_stats["annualized_volatility"] * math.sqrt(daily_hac_vif))
                primary_step_vol = float(daily_stats["step_volatility"] * math.sqrt(daily_hac_vif))
            else:
                primary_sharpe = float(daily_stats["annualized_sharpe"])
                primary_vol = float(daily_stats["annualized_volatility"])
                primary_step_vol = float(daily_stats["step_volatility"])

            metrics["sleeve_trading_volatility"] = primary_vol
            metrics["sleeve_trading_step_volatility"] = primary_step_vol
            metrics["sleeve_trading_sharpe_ratio_raw"] = primary_sharpe
            metrics["sleeve_trading_sharpe_ratio"] = None if ruined else primary_sharpe
            metrics["sleeve_trading_max_drawdown_pct"] = dd * 100.0
            eval_window_days = float(len(trading_usd)) / 144.0
            metrics["sleeve_eval_window_days"] = eval_window_days
            if eval_window_days > 0.0 and trading_usd[0] > 0.0 and trading_usd[-1] > 0.0:
                metrics["sleeve_trading_annualized_geometric_return_pct"] = float(
                    ((trading_usd[-1] / trading_usd[0]) ** (365.25 / eval_window_days) - 1.0) * 100.0
                )
            if len(daily_path) > 1 and daily_path[0] != 0.0:
                metrics["sleeve_trading_complete_daily_window_return_pct"] = float(
                    (daily_path[-1] / daily_path[0] - 1.0) * 100.0
                )
            if len(daily_path) and daily_path[-1] != 0.0:
                metrics["sleeve_trading_final_partial_day_return_pct"] = float(
                    (trading_usd[-1] / daily_path[-1] - 1.0) * 100.0
                )
            metrics["sleeve_sharpe_annualization_note"] = (
                f"annualized from a {eval_window_days:.0f}-day evaluation window "
                "using non-overlapping daily returns (365.25 periods/year); "
                f"primary excess-return Sharpe uses rf={annual_risk_free_rate:.4f}, "
                "with zero-rf sensitivity reported separately"
            )
    except Exception as e:
        metrics["sleeve_trading_risk_metrics_error"] = str(e)

    # Exposure diagnostics on decision steps (more interpretable than total-NAV vol when book value dominates).
    try:
        if "financial_exposure_dkk" in df.columns:
            expo = df["financial_exposure_dkk"].to_numpy(dtype=float)
            if "decision_step" in df.columns:
                dec = df["decision_step"].astype(bool).to_numpy()
                expo = expo[dec] if dec.any() else expo
            metrics["sleeve_mean_abs_exposure_dkk"] = float(np.mean(np.abs(expo))) if len(expo) else 0.0
            metrics["sleeve_max_abs_exposure_dkk"] = float(np.max(np.abs(expo))) if len(expo) else 0.0
    except Exception as e:
        metrics["sleeve_exposure_metrics_error"] = str(e)

    # Turnover, friction share of gross gain, and per-settlement-event P&L.
    try:
        if "financial_exposure_dkk" in df.columns and len(trading_usd) > 0:
            expo_dkk = pd.to_numeric(df["financial_exposure_dkk"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            initial_sleeve_dkk = float(trading_usd[0] / dkk_to_usd_rate) if dkk_to_usd_rate else 0.0
            gross_turnover_dkk = float(np.sum(np.abs(np.diff(expo_dkk)))) if len(expo_dkk) > 1 else 0.0
            metrics["sleeve_exposure_gross_turnover_dkk"] = gross_turnover_dkk
            if initial_sleeve_dkk > 0.0:
                days = max(float(len(expo_dkk)) / 144.0, 1e-9)
                metrics["sleeve_exposure_turnover_x_sleeve"] = gross_turnover_dkk / initial_sleeve_dkk
                metrics["sleeve_exposure_turnover_x_sleeve_per_day"] = (
                    gross_turnover_dkk / initial_sleeve_dkk / days
                )
        if "cumulative_trading_costs_dkk" in df.columns:
            costs = pd.to_numeric(df["cumulative_trading_costs_dkk"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            total_costs_dkk = float(costs[-1]) if len(costs) else 0.0
            metrics["sleeve_total_trading_costs_usd"] = total_costs_dkk * dkk_to_usd_rate
            if "cumulative_impact_costs_dkk" in df.columns:
                impact = pd.to_numeric(df["cumulative_impact_costs_dkk"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                metrics["sleeve_total_impact_costs_usd"] = (
                    (float(impact[-1]) if len(impact) else 0.0) * dkk_to_usd_rate
                )
            if "cumulative_volume_transaction_fees_dkk" in df.columns:
                volume_fees = pd.to_numeric(
                    df["cumulative_volume_transaction_fees_dkk"], errors="coerce"
                ).fillna(0.0).to_numpy(dtype=float)
                metrics["sleeve_total_volume_transaction_fees_usd"] = (
                    (float(volume_fees[-1]) if len(volume_fees) else 0.0)
                    * dkk_to_usd_rate
                )
            if "cumulative_market_access_fees_dkk" in df.columns:
                access_fees = pd.to_numeric(
                    df["cumulative_market_access_fees_dkk"], errors="coerce"
                ).fillna(0.0).to_numpy(dtype=float)
                metrics["sleeve_total_market_access_fees_usd"] = (
                    (float(access_fees[-1]) if len(access_fees) else 0.0)
                    * dkk_to_usd_rate
                )
            if "cumulative_collateral_funding_costs_dkk" in df.columns:
                collateral_costs = pd.to_numeric(
                    df["cumulative_collateral_funding_costs_dkk"],
                    errors="coerce",
                ).fillna(0.0).to_numpy(dtype=float)
                metrics["sleeve_total_collateral_funding_costs_usd"] = (
                    (float(collateral_costs[-1]) if len(collateral_costs) else 0.0)
                    * dkk_to_usd_rate
                )
            gross_gain_usd = float(trading_gain + total_costs_dkk * dkk_to_usd_rate)
            metrics["sleeve_gross_trading_gain_usd"] = gross_gain_usd
            if abs(gross_gain_usd) > 1e-9:
                metrics["sleeve_friction_share_of_gross_gain"] = float(
                    (total_costs_dkk * dkk_to_usd_rate) / abs(gross_gain_usd)
                )
        if "liquidity_participation" in df.columns:
            participation = pd.to_numeric(df["liquidity_participation"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["sleeve_liquidity_max_participation"] = float(np.max(participation)) if len(participation) else 0.0
            metrics["sleeve_liquidity_mean_participation"] = float(np.mean(participation)) if len(participation) else 0.0
        if "liquidity_scale" in df.columns:
            scale = pd.to_numeric(df["liquidity_scale"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
            metrics["sleeve_liquidity_bind_share"] = float(np.mean(scale < 0.999999)) if len(scale) else 0.0
            metrics["sleeve_liquidity_min_scale"] = float(np.min(scale)) if len(scale) else 1.0
        if "liquidity_tail_impact_multiplier" in df.columns:
            tail_mult = pd.to_numeric(df["liquidity_tail_impact_multiplier"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
            metrics["sleeve_liquidity_tail_impact_max_multiplier"] = float(np.max(tail_mult)) if len(tail_mult) else 1.0
            metrics["sleeve_liquidity_tail_impact_mean_multiplier"] = float(np.mean(tail_mult)) if len(tail_mult) else 1.0
        if "market_impact_participation" in df.columns:
            impact_participation = pd.to_numeric(
                df["market_impact_participation"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
            metrics["sleeve_market_impact_max_participation"] = (
                float(np.max(impact_participation)) if len(impact_participation) else 0.0
            )
            metrics["sleeve_market_impact_mean_participation"] = (
                float(np.mean(impact_participation)) if len(impact_participation) else 0.0
            )
        if "no_trade_threshold_dkk" in df.columns:
            no_trade_threshold = pd.to_numeric(
                df["no_trade_threshold_dkk"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
            metrics["sleeve_no_trade_threshold_max_dkk"] = (
                float(np.max(no_trade_threshold)) if len(no_trade_threshold) else 0.0
            )
            metrics["sleeve_no_trade_threshold_mean_dkk"] = (
                float(np.mean(no_trade_threshold)) if len(no_trade_threshold) else 0.0
            )
        if "collateral_required_dkk" in df.columns:
            collateral_required = pd.to_numeric(df["collateral_required_dkk"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["sleeve_collateral_required_max_dkk"] = float(np.max(collateral_required)) if len(collateral_required) else 0.0
            metrics["sleeve_collateral_required_mean_dkk"] = float(np.mean(collateral_required)) if len(collateral_required) else 0.0
        if "horizon_settlement_pnl_dkk" in df.columns and "horizon_settlement_count" in df.columns:
            pnl = pd.to_numeric(df["horizon_settlement_pnl_dkk"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            cnt = pd.to_numeric(df["horizon_settlement_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            events = pnl[cnt > 0.5]
            metrics["sleeve_settlement_event_count"] = float(events.size)
            if events.size:
                metrics["sleeve_settlement_pnl_mean_dkk"] = float(np.mean(events))
                metrics["sleeve_settlement_pnl_std_dkk"] = float(np.std(events, ddof=0))
                metrics["sleeve_settlement_pnl_hit_rate"] = float(np.mean(events > 0.0))
                metrics["sleeve_settlement_pnl_p5_dkk"] = float(np.percentile(events, 5))
                metrics["sleeve_settlement_pnl_p95_dkk"] = float(np.percentile(events, 95))
    except Exception as e:
        metrics["sleeve_trade_diagnostics_error"] = str(e)

    # Forecast-prior diagnostics for FoCAL runs. These are not performance
    # metrics; they prove whether the forecast layer was active, aligned,
    # damping conflict, or effectively falling back to MARL.
    try:
        decision_mask = None
        if "decision_step" in df.columns:
            decision_mask = pd.to_numeric(df["decision_step"], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.5
            metrics["focal_diagnostic_decision_rows"] = float(np.sum(decision_mask))
        if "forecast_prior_active" in df.columns:
            fp_active = pd.to_numeric(df["forecast_prior_active"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_active_share"] = float(np.mean(fp_active > 0.5)) if len(fp_active) else 0.0
            if decision_mask is not None and decision_mask.any():
                metrics["focal_active_share_decision"] = float(np.mean(fp_active[decision_mask] > 0.5))
        if "forecast_prior_exposure" in df.columns:
            fp_prior = pd.to_numeric(df["forecast_prior_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_mean_abs_prior_exposure"] = float(np.mean(np.abs(fp_prior))) if len(fp_prior) else 0.0
            metrics["focal_max_abs_prior_exposure"] = float(np.max(np.abs(fp_prior))) if len(fp_prior) else 0.0
            if decision_mask is not None and decision_mask.any():
                fp_prior_dec = fp_prior[decision_mask]
                metrics["focal_mean_abs_prior_exposure_decision"] = float(np.mean(np.abs(fp_prior_dec))) if len(fp_prior_dec) else 0.0
                metrics["focal_max_abs_prior_exposure_decision"] = float(np.max(np.abs(fp_prior_dec))) if len(fp_prior_dec) else 0.0
        for col, prefix in (
            ("forecast_only_prior_exposure", "focal_forecast_only"),
            ("forecast_beta_exposure", "focal_beta"),
        ):
            if col in df.columns:
                arr = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                metrics[f"{prefix}_mean_abs_exposure"] = float(np.mean(np.abs(arr))) if len(arr) else 0.0
                metrics[f"{prefix}_max_abs_exposure"] = float(np.max(np.abs(arr))) if len(arr) else 0.0
                if decision_mask is not None and decision_mask.any():
                    arr_dec = arr[decision_mask]
                    metrics[f"{prefix}_mean_abs_exposure_decision"] = float(np.mean(np.abs(arr_dec))) if len(arr_dec) else 0.0
                    metrics[f"{prefix}_max_abs_exposure_decision"] = float(np.max(np.abs(arr_dec))) if len(arr_dec) else 0.0
        if "forecast_beta_active" in df.columns:
            beta_active = pd.to_numeric(df["forecast_beta_active"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_beta_active_share"] = float(np.mean(beta_active > 0.5)) if len(beta_active) else 0.0
            if decision_mask is not None and decision_mask.any():
                metrics["focal_beta_active_share_decision"] = float(np.mean(beta_active[decision_mask] > 0.5))
        if "forecast_beta_strength" in df.columns:
            beta_strength = pd.to_numeric(df["forecast_beta_strength"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_beta_strength_mean"] = float(np.mean(beta_strength)) if len(beta_strength) else 0.0
        if "forecast_beta_payoff_lcb" in df.columns:
            beta_lcb = pd.to_numeric(df["forecast_beta_payoff_lcb"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_beta_payoff_lcb_final"] = float(beta_lcb[-1]) if len(beta_lcb) else 0.0
        if "forecast_beta_count" in df.columns:
            beta_count = pd.to_numeric(df["forecast_beta_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_beta_count_final"] = float(beta_count[-1]) if len(beta_count) else 0.0
        if "forecast_adjustment" in df.columns:
            fp_adj = pd.to_numeric(df["forecast_adjustment"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_mean_abs_adjustment"] = float(np.mean(np.abs(fp_adj))) if len(fp_adj) else 0.0
            metrics["focal_max_abs_adjustment"] = float(np.max(np.abs(fp_adj))) if len(fp_adj) else 0.0
            if decision_mask is not None and decision_mask.any():
                fp_adj_dec = fp_adj[decision_mask]
                metrics["focal_mean_abs_adjustment_decision"] = float(np.mean(np.abs(fp_adj_dec))) if len(fp_adj_dec) else 0.0
                metrics["focal_max_abs_adjustment_decision"] = float(np.max(np.abs(fp_adj_dec))) if len(fp_adj_dec) else 0.0
        if "forecast_alignment" in df.columns:
            fp_align = pd.to_numeric(df["forecast_alignment"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_agreement_share"] = float(np.mean(fp_align > 0.0)) if len(fp_align) else 0.0
            metrics["focal_conflict_share"] = float(np.mean(fp_align < 0.0)) if len(fp_align) else 0.0
            if decision_mask is not None and decision_mask.any():
                fp_align_dec = fp_align[decision_mask]
                metrics["focal_agreement_share_decision"] = float(np.mean(fp_align_dec > 0.0)) if len(fp_align_dec) else 0.0
                metrics["focal_conflict_share_decision"] = float(np.mean(fp_align_dec < 0.0)) if len(fp_align_dec) else 0.0
        if "forecast_prior_skill" in df.columns:
            fp_skill = pd.to_numeric(df["forecast_prior_skill"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_mean_skill"] = float(np.mean(fp_skill)) if len(fp_skill) else 0.0
        if "forecast_prior_hit_lcb" in df.columns:
            fp_hit = pd.to_numeric(df["forecast_prior_hit_lcb"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
            metrics["focal_mean_hit_lcb"] = float(np.mean(fp_hit)) if len(fp_hit) else 0.5
        if "forecast_prior_hit_rate" in df.columns:
            fp_hit_rate = pd.to_numeric(df["forecast_prior_hit_rate"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
            metrics["focal_mean_hit_rate"] = float(np.mean(fp_hit_rate)) if len(fp_hit_rate) else 0.5
        if "forecast_prior_direction_confidence" in df.columns:
            fp_conf = pd.to_numeric(df["forecast_prior_direction_confidence"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
            metrics["focal_mean_direction_confidence"] = float(np.mean(fp_conf)) if len(fp_conf) else 0.5
        if "forecast_prior_confidence_weight" in df.columns:
            fp_weight = pd.to_numeric(df["forecast_prior_confidence_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_mean_confidence_weight"] = float(np.mean(fp_weight)) if len(fp_weight) else 0.0
        if "forecast_residual_authority" in df.columns:
            fp_resid = pd.to_numeric(df["forecast_residual_authority"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_mean_residual_authority"] = float(np.mean(fp_resid)) if len(fp_resid) else 0.0
            metrics["focal_max_residual_authority"] = float(np.max(fp_resid)) if len(fp_resid) else 0.0
            if decision_mask is not None and decision_mask.any():
                fp_resid_dec = fp_resid[decision_mask]
                metrics["focal_mean_residual_authority_decision"] = float(np.mean(fp_resid_dec)) if len(fp_resid_dec) else 0.0
        if "forecast_base_residual_authority" in df.columns:
            base_resid = pd.to_numeric(df["forecast_base_residual_authority"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_mean_base_residual_authority"] = float(np.mean(base_resid)) if len(base_resid) else 0.0
            if decision_mask is not None and decision_mask.any():
                metrics["focal_mean_base_residual_authority_decision"] = float(np.mean(base_resid[decision_mask])) if len(base_resid) else 0.0
        if "forecast_residual_evidence_multiplier" in df.columns:
            resid_mult = pd.to_numeric(df["forecast_residual_evidence_multiplier"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_residual_evidence_multiplier_mean"] = float(np.mean(resid_mult)) if len(resid_mult) else 0.0
        if "forecast_residual_to_prior_ratio" in df.columns:
            fp_ratio = pd.to_numeric(df["forecast_residual_to_prior_ratio"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_residual_to_prior_ratio"] = float(np.max(fp_ratio)) if len(fp_ratio) else 0.0
        if "trading_sleeve_margin_active" in df.columns:
            margin_active = pd.to_numeric(df["trading_sleeve_margin_active"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["sleeve_margin_active_any"] = bool(np.any(margin_active > 0.5))
            metrics["sleeve_margin_active_share"] = float(np.mean(margin_active > 0.5)) if len(margin_active) else 0.0
        if "trading_sleeve_margin_step" in df.columns:
            margin_steps = pd.to_numeric(df["trading_sleeve_margin_step"], errors="coerce").fillna(-1.0).to_numpy(dtype=float)
            valid_steps = margin_steps[margin_steps >= 0.0]
            metrics["sleeve_margin_first_step"] = float(np.min(valid_steps)) if len(valid_steps) else -1.0
        if "trading_sleeve_margin_gap_dkk" in df.columns:
            margin_gap = pd.to_numeric(df["trading_sleeve_margin_gap_dkk"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["sleeve_margin_min_gap_dkk"] = float(np.min(margin_gap)) if len(margin_gap) else 0.0
        if "forecast_prior_calibration_count" in df.columns:
            fp_count = pd.to_numeric(df["forecast_prior_calibration_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_final_calibration_count"] = float(fp_count[-1]) if len(fp_count) else 0.0
        if "forecast_prior_calibration_total" in df.columns:
            fp_total = pd.to_numeric(df["forecast_prior_calibration_total"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_final_calibration_total"] = float(fp_total[-1]) if len(fp_total) else 0.0
        if "forecast_mode_reason" in df.columns:
            reasons = df["forecast_mode_reason"].fillna("").astype(str).str.strip()
            valid = reasons[reasons != ""]
            denom = float(max(len(valid), 1))
            counts = valid.value_counts()
            for reason in (
                "certified_directional_prior_bounded_residual",
                "policy_fallback_no_edge",
                "agreement_boost",
                "agreement_boost_policy_relative",
                "conflict_damping",
                "conflict_damping_policy_relative",
                "policy_fallback_conflict_no_relative_edge",
                "neutral_small_prior",
                "neutral_small_prior_policy_relative",
            ):
                metrics[f"focal_reason_share_{reason}"] = float(counts.get(reason, 0) / denom)
            if decision_mask is not None and decision_mask.any():
                reason_dec = reasons[decision_mask]
                valid_dec = reason_dec[reason_dec != ""]
                denom_dec = float(max(len(valid_dec), 1))
                counts_dec = valid_dec.value_counts()
                for reason in (
                    "policy_fallback_no_edge",
                    "agreement_boost",
                    "agreement_boost_policy_relative",
                    "conflict_damping",
                    "conflict_damping_policy_relative",
                    "policy_fallback_conflict_no_relative_edge",
                    "neutral_small_prior",
                    "neutral_small_prior_policy_relative",
                ):
                    metrics[f"focal_reason_share_{reason}_decision"] = float(counts_dec.get(reason, 0) / denom_dec)
        if "forecast_policy_relative_strength" in df.columns:
            rel = pd.to_numeric(df["forecast_policy_relative_strength"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_policy_relative_strength_mean"] = float(np.mean(rel)) if len(rel) else 0.0
        if "forecast_policy_relative_count" in df.columns:
            cnt = pd.to_numeric(df["forecast_policy_relative_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics["focal_policy_relative_final_count"] = float(cnt[-1]) if len(cnt) else 0.0
        if "forecast_policy_hit_lcb" in df.columns:
            pol = pd.to_numeric(df["forecast_policy_hit_lcb"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
            metrics["focal_policy_hit_lcb_final"] = float(pol[-1]) if len(pol) else 0.5
        if "forecast_policy_relative_hit_lcb" in df.columns:
            pri = pd.to_numeric(df["forecast_policy_relative_hit_lcb"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
            metrics["focal_prior_vs_policy_hit_lcb_final"] = float(pri[-1]) if len(pri) else 0.5
        for column, suffix in {
            "distributional_edge_strength": "edge_strength",
            "distributional_confidence_weight": "confidence_weight",
        }.items():
            if column not in df.columns:
                continue
            values = pd.to_numeric(df[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics[f"focal_distributional_{suffix}_mean"] = (
                float(np.mean(values)) if len(values) else 0.0
            )
            if decision_mask is not None and decision_mask.any():
                metrics[f"focal_distributional_{suffix}_mean_decision"] = float(
                    np.mean(values[decision_mask])
                )
        feasible_columns = {
            "feasible_action_coordinate": "coordinate",
            "feasible_action_capacity_mwh": "capacity_mwh",
            "feasible_action_capacity_fraction": "capacity_fraction",
            "feasible_action_hard_capacity_mwh": "hard_capacity_mwh",
            "feasible_action_hard_capacity_fraction": "hard_capacity_fraction",
            "feasible_action_evidence_authority": "evidence_authority",
            "feasible_action_capacity_conditioned_on_evidence": "capacity_conditioned_share",
            "feasible_action_anchor_quantity_mwh": "anchor_quantity_mwh",
            "feasible_action_policy_quantity_mwh": "policy_quantity_mwh",
            "feasible_action_adjustment_mwh": "adjustment_mwh",
            "feasible_action_anchor_fraction": "anchor_fraction",
            "feasible_action_target_fraction": "target_fraction",
            "feasible_action_anchor_no_trade_hold": "anchor_no_trade_hold_share",
            "feasible_action_policy_no_trade_hold": "policy_no_trade_hold_share",
            "feasible_action_liquidity_capacity_fraction": "liquidity_capacity_fraction",
            "feasible_action_collateral_headroom_fraction": "collateral_headroom_fraction",
            "feasible_action_policy_funding_cost_dkk": "policy_funding_cost_dkk",
            "feasible_action_anchor_funding_cost_dkk": "anchor_funding_cost_dkk",
            "feasible_action_advantage_mean": "advantage_mean",
            "feasible_action_cvar_shortfall": "cvar_shortfall",
            "feasible_action_dual_lambda": "dual_lambda",
            "feasible_action_last_advantage_dkk": "last_advantage_dkk",
            "feasible_action_reward_step": "reward_step",
        }
        for column, suffix in feasible_columns.items():
            if column not in df.columns:
                continue
            values = pd.to_numeric(df[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics[f"feasible_action_{suffix}_mean"] = (
                float(np.mean(values)) if len(values) else 0.0
            )
            if (
                decision_mask is not None
                and decision_mask.any()
                and column != "feasible_action_reward_step"
            ):
                metrics[f"feasible_action_{suffix}_mean_decision"] = float(
                    np.mean(values[decision_mask])
                )
        mirror_columns = {
            "forecast_mirror_action": "action",
            "forecast_mirror_requested_quantity_mwh": "requested_quantity_mwh",
            "forecast_mirror_executed_quantity_mwh": "executed_quantity_mwh",
            "forecast_mirror_hard_capacity_mwh": "hard_capacity_mwh",
            "forecast_mirror_prior_quantity_mwh": "prior_quantity_mwh",
            "forecast_mirror_prior_coordinate": "prior_coordinate",
            "forecast_mirror_evidence": "evidence",
            "forecast_mirror_trust": "trust",
            "forecast_mirror_exponent": "exponent",
            "forecast_mirror_base_action_mean": "base_action_mean",
            "forecast_mirror_final_action_mean": "final_action_mean",
            "forecast_mirror_trust_gradient": "trust_gradient",
            "forecast_mirror_counterfactual_score": "counterfactual_score",
            "forecast_mirror_pending_count": "pending_count",
        }
        for column, suffix in mirror_columns.items():
            if column not in df.columns:
                continue
            values = pd.to_numeric(df[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            metrics[f"forecast_mirror_{suffix}_mean"] = (
                float(np.mean(values)) if len(values) else 0.0
            )
            if decision_mask is not None and decision_mask.any():
                metrics[f"forecast_mirror_{suffix}_mean_decision"] = float(
                    np.mean(values[decision_mask])
                )
        if "feasible_action_count" in df.columns:
            values = pd.to_numeric(
                df["feasible_action_count"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
            metrics["feasible_action_final_count"] = float(values[-1]) if len(values) else 0.0
        if "feasible_action_adjustment_mwh" in df.columns:
            values = pd.to_numeric(
                df["feasible_action_adjustment_mwh"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
            mask = (
                decision_mask
                if decision_mask is not None and decision_mask.any()
                else np.ones(len(values), dtype=bool)
            )
            selected = values[mask]
            metrics["feasible_action_intervention_share_decision"] = (
                float(np.mean(np.abs(selected) > 1e-9)) if len(selected) else 0.0
            )
            metrics["feasible_action_mean_abs_adjustment_mwh_decision"] = (
                float(np.mean(np.abs(selected))) if len(selected) else 0.0
            )
        if "feasible_action_coordinate" in df.columns:
            coordinate = pd.to_numeric(
                df["feasible_action_coordinate"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
            mask = (
                decision_mask
                if decision_mask is not None and decision_mask.any()
                else np.ones(len(coordinate), dtype=bool)
            )
            selected = coordinate[mask]
            metrics["feasible_action_abstention_side_share_decision"] = (
                float(np.mean(selected < -1e-9)) if len(selected) else 0.0
            )
            metrics["feasible_action_amplification_side_share_decision"] = (
                float(np.mean(selected > 1e-9)) if len(selected) else 0.0
            )
            metrics["feasible_action_saturation_share_decision"] = (
                float(np.mean(np.abs(selected) >= 0.95)) if len(selected) else 0.0
            )
        for prefix, requested_column, executed_column in (
            (
                "anchor",
                "feasible_action_anchor_requested_mwh",
                "feasible_action_anchor_quantity_mwh",
            ),
            (
                "policy",
                "feasible_action_policy_requested_mwh",
                "feasible_action_policy_quantity_mwh",
            ),
        ):
            if requested_column not in df.columns or executed_column not in df.columns:
                continue
            requested = pd.to_numeric(
                df[requested_column], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
            executed = pd.to_numeric(
                df[executed_column], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
            mask = (
                decision_mask
                if decision_mask is not None and decision_mask.any()
                else np.ones(len(requested), dtype=bool)
            )
            metrics[f"feasible_action_{prefix}_execution_bind_share"] = (
                float(np.mean(np.abs(requested[mask] - executed[mask]) > 1e-9))
                if len(requested)
                else 0.0
            )
        required = {
            "anchor": "feasible_action_anchor_quantity_mwh",
            "policy": "feasible_action_policy_quantity_mwh",
            "capacity": "feasible_action_capacity_mwh",
        }
        if all(column in df.columns for column in required.values()):
            anchor = pd.to_numeric(
                df[required["anchor"]], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
            policy = pd.to_numeric(
                df[required["policy"]], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
            capacity = pd.to_numeric(
                df[required["capacity"]], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
            mask = (
                decision_mask
                if decision_mask is not None and decision_mask.any()
                else np.ones(len(policy), dtype=bool)
            )
            metrics["feasible_action_sign_violation_share_decision"] = (
                float(np.mean((anchor[mask] * policy[mask]) < -1e-9))
                if len(policy)
                else 0.0
            )
            metrics["feasible_action_capacity_violation_share_decision"] = (
                float(np.mean(np.abs(policy[mask]) > capacity[mask] + 1e-8))
                if len(policy)
                else 0.0
            )
    except Exception as e:
        metrics["focal_diagnostics_error"] = str(e)

    return metrics


def write_tier_report(tier_results: Dict[str, Any], output_dir: str) -> Tuple[str, str]:
    """Write a consolidated tier report (comparison or single-variant evaluation).

    Returns: (csv_path, md_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []
    preferred_order = [
        "tier1",
    ]
    variants = [v for v in preferred_order if v in tier_results]
    variants.extend([v for v in tier_results.keys() if v not in variants])
    if not variants:
        variants = preferred_order[:1]

    for variant in variants:
        r = tier_results.get(variant, {}) or {}
        s = r.get("sleeve_metrics", {}) or {}
        rows.append({
            "variant": variant,
            "status": r.get("status", "unknown"),
            "final_portfolio_value_usd": float(r.get("final_portfolio_value", 0.0)),
            "initial_portfolio_value_usd": float(r.get("initial_portfolio_value", 0.0)),
            "total_return_pct": _pct(r.get("total_return", 0.0)),
            "sharpe_ratio": float(r.get("sharpe_ratio", 0.0)),
            "max_drawdown_pct": _pct(r.get("max_drawdown", 0.0)),
            "volatility": float(r.get("volatility", 0.0)),
            "total_distributions_usd": float(r.get("total_distributions_usd", 0.0)),
            "reported_nav_final_usd": float(r.get("reported_nav_final_portfolio_value", 0.0)),
            "reported_nav_return_pct": _pct(r.get("reported_nav_total_return", 0.0)),
            "reported_nav_sharpe": float(r.get("reported_nav_sharpe_ratio", 0.0)),
            "total_rewards": float(r.get("total_rewards", 0.0)),
            "average_risk": float(r.get("average_risk", 0.0)),
            # Sleeve decomposition (from env debug logs)
            "trading_gain_usd": float(s.get("sleeve_trading_gain_usd", 0.0)),
            "operating_gain_usd": float(s.get("sleeve_operating_gain_usd", 0.0)),
            "trading_gain_share": float(s.get("sleeve_trading_gain_share", 0.0)),
            "trading_return_pct": float(s.get("sleeve_trading_return_pct", 0.0)),
            "operating_return_pct": float(s.get("sleeve_operating_return_pct", 0.0)),
            "mean_abs_exposure_dkk": float(s.get("sleeve_mean_abs_exposure_dkk", 0.0)),
            "trading_sharpe_mode": str(s.get("sleeve_trading_sharpe_primary_mode", "")),
            "trading_sharpe": (
                float(s.get("sleeve_trading_sharpe_ratio"))
                if s.get("sleeve_trading_sharpe_ratio") is not None
                else None
            ),
            "trading_max_dd_pct": float(s.get("sleeve_trading_max_drawdown_pct", 0.0)),
            "focal_decision_rows": float(s.get("focal_diagnostic_decision_rows", 0.0)),
            "focal_active_share": float(s.get("focal_active_share_decision", s.get("focal_active_share", 0.0))),
            "focal_agreement_share": float(s.get("focal_agreement_share_decision", s.get("focal_agreement_share", 0.0))),
            "focal_conflict_share": float(s.get("focal_conflict_share_decision", s.get("focal_conflict_share", 0.0))),
            "focal_mean_abs_prior_exposure": float(s.get("focal_mean_abs_prior_exposure_decision", s.get("focal_mean_abs_prior_exposure", 0.0))),
            "focal_mean_abs_adjustment": float(s.get("focal_mean_abs_adjustment_decision", s.get("focal_mean_abs_adjustment", 0.0))),
            "focal_mean_skill": float(s.get("focal_mean_skill", 0.0)),
            "focal_mean_hit_lcb": float(s.get("focal_mean_hit_lcb", 0.0)),
            "focal_mean_hit_rate": float(s.get("focal_mean_hit_rate", 0.0)),
            "focal_mean_direction_confidence": float(s.get("focal_mean_direction_confidence", 0.0)),
            "focal_mean_confidence_weight": float(s.get("focal_mean_confidence_weight", 0.0)),
            "focal_final_calibration_count": float(s.get("focal_final_calibration_count", 0.0)),
            "focal_final_calibration_total": float(s.get("focal_final_calibration_total", 0.0)),
            "run_dir": r.get("run_dir", ""),
            "final_models_dir": r.get("final_models_dir", ""),
        })

    df = pd.DataFrame(rows)
    report_scope = "single_variant"
    report_prefix = "tier_result"
    report_title = "Tier Evaluation Report"
    mode_label = "tier1"

    csv_path = os.path.join(output_dir, f"{report_prefix}_{ts}.csv")
    df.to_csv(csv_path, index=False)

    md_path = os.path.join(output_dir, f"{report_prefix}_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"## {report_title}\n\n")
        f.write(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Report scope: `{report_scope}`\n")
        f.write(f"- Evaluation mode: `{mode_label}`\n")
        f.write(f"- Output CSV: `{os.path.basename(csv_path)}`\n\n")

        f.write("### Summary Table\n\n")
        f.write("| Variant | Status | Final (USD) | Return % | Sharpe | Max DD % | Trading Sharpe | Trading Max DD % |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for _, row in df.iterrows():
            trading_sharpe_text = (
                "N/A" if pd.isna(row["trading_sharpe"]) else f"{row['trading_sharpe']:.4f}"
            )
            f.write(
                f"| {row['variant']} | {row['status']} | "
                f"{row['final_portfolio_value_usd']:.2f} | {row['total_return_pct']:.3f} | "
                f"{row['sharpe_ratio']:.4f} | {row['max_drawdown_pct']:.3f} | "
                f"{trading_sharpe_text} | {row['trading_max_dd_pct']:.3f} |\n"
            )

        f.write("\nTotal-NAV Sharpe/DD can be compressed by the physical operating sleeve.\n")
        f.write("Trading-sleeve Sharpe/DD are shown above using the selected paper-facing Sharpe mode.\n")

        f.write("\n### Evaluation Inputs / Runtime\n\n")
        for _, row in df.iterrows():
            f.write(f"- {row['variant']}:\n")
            f.write(f"  - final_models: `{row['final_models_dir']}`\n")
            if str(row.get('run_dir', '')).strip() != "":
                f.write(f"  - run_dir: `{row['run_dir']}`\n")

        f.write("\n### NAV Sleeve Decomposition (Operating vs Trading)\n\n")
        f.write("Total NAV includes physical book value, which can compress volatility/drawdown.\n")
        f.write("This section splits gains into the operating sleeve (physical + ops revenue) and\n")
        f.write("the trading sleeve (cash + MTM).\n\n")
        f.write("| Variant | Trading Gain (USD) | Operating Gain (USD) | Trading Share | Trading Return % | Mean |Exposure| (DKK) | Avg Risk |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for _, row in df.iterrows():
            f.write(
                f"| {row['variant']} | {row['trading_gain_usd']:.2f} | {row['operating_gain_usd']:.2f} | "
                f"{row['trading_gain_share']:.3f} | {row['trading_return_pct']:.2f} | "
                f"{row['mean_abs_exposure_dkk']:.2f} | {row['average_risk']:.3f} |\n"
            )

    return csv_path, md_path


def load_agent_system(trained_agents_dir: str, eval_env, enhanced: bool = False) -> Optional[Any]:
    """Load MultiESGAgent system from directory."""
    model_type = "enhanced" if enhanced else "standard"
    print(f"Loading {model_type} agent system from: {trained_agents_dir}")

    try:
        config = EvaluationConfig()
        agent_system = MultiESGAgent(
            config,
            env=eval_env,
            device="cpu",
            training=False,
            debug=False
        )

        # Load policies from disk
        loaded_count = agent_system.load_policies(trained_agents_dir)
        print(f"Loaded {loaded_count} agent policies")

        if loaded_count == 0:
            print("No agent policies loaded successfully")
            return None

        return agent_system

    except Exception as e:
        print(f"Error loading agent system: {e}")
        return None

def load_enhanced_agent_system(enhanced_models_dir: str, eval_env) -> Optional[Any]:
    """Load enhanced MultiESGAgent system with the fixed Tier1 evaluation runtime."""
    print(f"Loading enhanced agent system from: {enhanced_models_dir}")

    return load_agent_system(enhanced_models_dir, eval_env, enhanced=True)


def create_evaluation_environment(
    data: pd.DataFrame,
    log_path: Optional[str] = None,
    output_dir: str = "evaluation_results",
    investment_freq: int = 6,
    config=None,
    env_log_dir: Optional[str] = None,
    forecast_cache_dir: Optional[str] = None,
    fail_fast: bool = True,
) -> Any:
    """Create evaluation environment. Forecast utilization is cache-only."""
    print_progress("Setting up evaluation environment...")

    try:
        if forecast_cache_dir is not None and bool(getattr(config, "enable_forecast_utilization", False)):
            try:
                config.forecast_cache_dir = str(forecast_cache_dir)
            except Exception:
                pass

        base_env = RenewableMultiAgentEnv(
            data,
            investment_freq=int(investment_freq),
            config=config,
            log_dir=env_log_dir,
        )
        # Evaluation-only marker for paper metric-family stress tests.
        base_env.evaluation_mode = True
        print_progress("Environment created")
        return base_env

    except Exception as e:
        if fail_fast:
            raise
        print_progress(f"Failed to create evaluation environment: {e}")
        return None


def _coerce_action_for_space(action: np.ndarray, action_space):
    """Make sure an action matches the environment's action space."""
    if hasattr(action_space, "n"):  # Discrete
        if isinstance(action, (list, tuple, np.ndarray)):
            action = np.array(action).astype(np.int64).flatten()
            return int(action[0])
        return int(action)
    else:  # Box
        act = np.array(action, dtype=np.float32).squeeze()
        if hasattr(action_space, "shape") and action_space.shape is not None:
            target = int(np.prod(action_space.shape))
            act = act.flatten()
            if act.size != target:
                if act.size < target:
                    act = np.pad(act, (0, target - act.size))
                else:
                    act = act[:target]
        return act


def calculate_performance_metrics(portfolio_values: list, 
                                rewards_by_agent: Dict[str, list],
                                risk_levels: list,
                                evaluation_steps: int,
                                timestamps=None,
                                annual_risk_free_rate: float = 0.0) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics."""
    metrics = {}
    
    try:
        # Portfolio performance
        if portfolio_values:
            pv_array = np.array(portfolio_values)
            returns = np.diff(pv_array) / pv_array[:-1]
            periods_per_year = _infer_periods_per_year(timestamps)
            risk_stats = _compute_annualized_risk_metrics(
                returns,
                periods_per_year=periods_per_year,
                annual_risk_free_rate=annual_risk_free_rate,
            )

            metrics['total_return'] = (pv_array[-1] / pv_array[0] - 1) if len(pv_array) > 1 else 0.0
            metrics['volatility'] = float(risk_stats['annualized_volatility'])
            metrics['step_volatility'] = float(risk_stats['step_volatility'])
            metrics['sharpe_ratio'] = float(risk_stats['annualized_sharpe'])
            metrics['periods_per_year'] = float(periods_per_year)
            metrics['annual_risk_free_rate'] = float(annual_risk_free_rate)
            metrics['per_step_risk_free_rate'] = float(risk_stats['per_step_risk_free_rate'])
            if len(pv_array) > 0:
                peak = np.maximum.accumulate(pv_array)
                drawdowns = np.where(peak > 0.0, (peak - pv_array) / peak, 0.0)
                metrics['max_drawdown'] = float(np.max(drawdowns)) if len(drawdowns) else 0.0
            else:
                metrics['max_drawdown'] = 0.0
            metrics['initial_portfolio_value'] = float(pv_array[0]) if len(pv_array) > 0 else 0.0
            metrics['final_portfolio_value'] = float(pv_array[-1]) if len(pv_array) > 0 else 0.0
        
        # Agent rewards
        total_rewards = 0
        for agent, rewards in rewards_by_agent.items():
            if rewards:
                agent_total = np.sum(rewards)
                metrics[f'{agent}_total_reward'] = float(agent_total)
                metrics[f'{agent}_avg_reward'] = float(np.mean(rewards))
                total_rewards += agent_total
        
        metrics['total_rewards'] = float(total_rewards)
        
        # Risk metrics
        if risk_levels:
            metrics['average_risk'] = float(np.mean(risk_levels))
            metrics['max_risk'] = float(np.max(risk_levels))
            metrics['min_risk'] = float(np.min(risk_levels))
        
        # Evaluation info
        metrics['evaluation_steps'] = evaluation_steps
        metrics['evaluation_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    except Exception as e:
        print(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Error calculating performance metrics: {e}")
        metrics['error'] = str(e)
    
    return metrics


def _iter_env_chain(env):
    """Yield an environment and simple wrapper parents."""
    seen = set()
    cur = env
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = getattr(cur, "env", None)


def _get_total_distributions_dkk(env) -> float:
    """Cash distributions are investor wealth and must be added back to NAV."""
    for candidate in _iter_env_chain(env):
        if hasattr(candidate, "total_distributions"):
            try:
                return float(max(0.0, getattr(candidate, "total_distributions", 0.0)))
            except Exception:
                return 0.0
    for candidate in _iter_env_chain(env):
        if hasattr(candidate, "distributed_profits"):
            try:
                return float(max(0.0, getattr(candidate, "distributed_profits", 0.0)))
            except Exception:
                return 0.0
    return 0.0


def _attach_eval_override_context(metrics: Dict[str, Any], eval_env) -> Dict[str, Any]:
    cfg = getattr(eval_env, "config", None)
    override = getattr(cfg, "eval_distribution_rate", None) if cfg is not None else None
    metrics["evaluation_mode_env_flag"] = bool(getattr(eval_env, "evaluation_mode", False))
    metrics["eval_distribution_rate"] = None if override is None else float(override)
    metrics["cash_sweeper_eval_override_active"] = bool(
        getattr(eval_env, "evaluation_mode", False) and override is not None
    )
    metrics["cash_sweeper_distribution_rate_used"] = (
        float(override)
        if override is not None and bool(getattr(eval_env, "evaluation_mode", False))
        else float(getattr(cfg, "distribution_rate", 0.0) if cfg is not None else 0.0)
    )
    # Friction sweep metadata for eval-only stress tests; defaults preserve v1 behavior.
    metrics["friction_cost_multiplier"] = float(
        getattr(cfg, "friction_cost_multiplier", 1.0) if cfg is not None else 1.0
    )
    metrics["market_fee_model"] = str(
        getattr(cfg, "market_fee_model", "legacy_notional_fixed")
        if cfg is not None else "legacy_notional_fixed"
    )
    metrics["transaction_fee_dkk_per_mwh"] = float(
        getattr(cfg, "transaction_fee_dkk_per_mwh", 0.0) if cfg is not None else 0.0
    )
    metrics["annual_market_access_fee_dkk"] = float(
        getattr(cfg, "annual_market_access_fee_dkk", 0.0) if cfg is not None else 0.0
    )
    metrics["market_access_fee_allocation_fraction"] = float(
        getattr(cfg, "market_access_fee_allocation_fraction", 1.0)
        if cfg is not None else 1.0
    )
    metrics["market_fee_source_id"] = str(
        getattr(cfg, "market_fee_source_id", "legacy_unsourced_fixed_fee")
        if cfg is not None else "legacy_unsourced_fixed_fee"
    )
    metrics["half_spread_bp"] = float(getattr(cfg, "half_spread_bp", 0.0) if cfg is not None else 0.0)
    metrics["impact_coef_bp"] = float(getattr(cfg, "impact_coef_bp", 0.0) if cfg is not None else 0.0)
    metrics["impact_exponent"] = float(getattr(cfg, "impact_exponent", 0.5) if cfg is not None else 0.5)
    metrics["impact_ref_notional"] = str(
        getattr(cfg, "impact_ref_notional", "sleeve") if cfg is not None else "sleeve"
    )
    metrics["impact_volume_data_path"] = str(
        getattr(cfg, "impact_volume_data_path", "") if cfg is not None else ""
    )
    metrics["impact_volume_column"] = str(getattr(cfg, "impact_volume_column", "") if cfg is not None else "")
    metrics["impact_volume_unit"] = str(getattr(cfg, "impact_volume_unit", "mwh") if cfg is not None else "mwh")
    metrics["impact_volume_max_staleness_minutes"] = float(
        getattr(cfg, "impact_volume_max_staleness_minutes", 90.0) if cfg is not None else 90.0
    )
    metrics["impact_volume_price_floor_dkk_per_mwh"] = float(
        getattr(cfg, "impact_volume_price_floor_dkk_per_mwh", 50.0) if cfg is not None else 50.0
    )
    for name, default in (
        ("liquidity_participation_cap_fraction", 0.0),
        ("liquidity_volume_multiplier", 1.0),
        ("liquidity_min_volume_mwh", 1.0),
        ("liquidity_tail_impact_threshold_dkk_per_mwh", 5000.0),
        ("liquidity_tail_impact_multiplier", 1.0),
        ("liquidity_tail_impact_power", 1.0),
        ("liquidity_tail_impact_max_multiplier", 10.0),
        ("collateral_notional_margin_fraction", 0.02),
        ("collateral_stress_loss_fraction", 0.10),
        ("collateral_stress_price_dkk_per_mwh", 25000.0),
        ("collateral_funding_rate_annual", 0.05),
        ("collateral_tradeable_haircut", 1.0),
    ):
        metrics[name] = float(getattr(cfg, name, default) if cfg is not None else default)
    metrics["liquidity_volume_source"] = str(
        getattr(cfg, "liquidity_volume_source", "load") if cfg is not None else "load"
    )
    metrics["enable_collateral_cash_drag"] = bool(
        getattr(cfg, "enable_collateral_cash_drag", False) if cfg is not None else False
    )
    metrics["mtm_return_model"] = str(getattr(cfg, "mtm_return_model", "percent_capped") if cfg is not None else "percent_capped")
    metrics["mtm_reference_price_dkk_per_mwh"] = float(
        getattr(cfg, "mtm_reference_price_dkk_per_mwh", 500.0) if cfg is not None else 500.0
    )
    metrics["mtm_settlement_horizon_steps"] = int(
        getattr(cfg, "mtm_settlement_horizon_steps", 6) if cfg is not None else 6
    )
    metrics["mtm_entry_price_mode"] = str(
        getattr(cfg, "mtm_entry_price_mode", "current_price") if cfg is not None else "current_price"
    )
    metrics["mtm_horizon_payoff_denominator_mode"] = str(
        getattr(cfg, "mtm_horizon_payoff_denominator_mode", "reference_price")
        if cfg is not None else "reference_price"
    )
    metrics["mtm_settlement_price_mode"] = str(
        getattr(cfg, "mtm_settlement_price_mode", "energy_index")
        if cfg is not None else "energy_index"
    )
    metrics["mtm_basis_price_data_path"] = str(
        getattr(cfg, "mtm_basis_price_data_path", "") if cfg is not None else ""
    )
    metrics["mtm_basis_price_column"] = str(
        getattr(cfg, "mtm_basis_price_column", "price") if cfg is not None else "price"
    )
    metrics["mtm_basis_scale"] = float(
        getattr(cfg, "mtm_basis_scale", 0.0) if cfg is not None else 0.0
    )
    metrics["mtm_basis_centering_mode"] = str(
        getattr(cfg, "mtm_basis_centering_mode", "rolling_median")
        if cfg is not None else "rolling_median"
    )
    metrics["mtm_basis_centering_window_steps"] = int(
        getattr(cfg, "mtm_basis_centering_window_steps", 4320)
        if cfg is not None else 4320
    )
    metrics["mtm_external_settlement_price_data_path"] = str(
        getattr(cfg, "mtm_external_settlement_price_data_path", "") if cfg is not None else ""
    )
    metrics["mtm_external_settlement_price_column"] = str(
        getattr(cfg, "mtm_external_settlement_price_column", "settlement_price")
        if cfg is not None else "settlement_price"
    )
    metrics["mtm_external_settlement_timestamp_column"] = str(
        getattr(cfg, "mtm_external_settlement_timestamp_column", "timestamp")
        if cfg is not None else "timestamp"
    )
    total_impact_dkk = 0.0
    last_impact_dkk = 0.0
    last_ref_notional = 0.0
    last_participation = 0.0
    last_impact_bp = 0.0
    total_collateral_cost_dkk = 0.0
    last_collateral_required_dkk = 0.0
    last_collateral_cost_dkk = 0.0
    last_liquidity_participation = 0.0
    last_liquidity_scale = 1.0
    last_liquidity_tail_multiplier = 1.0
    liquidity_source = ""
    for candidate in _iter_env_chain(eval_env):
        if hasattr(candidate, "cumulative_market_impact_costs"):
            try:
                total_impact_dkk = float(getattr(candidate, "cumulative_market_impact_costs", 0.0))
            except Exception:
                total_impact_dkk = 0.0
        if hasattr(candidate, "_last_market_impact_cost"):
            try:
                last_impact_dkk = float(getattr(candidate, "_last_market_impact_cost", 0.0))
            except Exception:
                last_impact_dkk = 0.0
        if hasattr(candidate, "_last_market_impact_ref_notional"):
            try:
                last_ref_notional = float(getattr(candidate, "_last_market_impact_ref_notional", 0.0))
            except Exception:
                last_ref_notional = 0.0
        if hasattr(candidate, "_last_market_impact_participation"):
            try:
                last_participation = float(getattr(candidate, "_last_market_impact_participation", 0.0))
            except Exception:
                last_participation = 0.0
        if hasattr(candidate, "_last_market_impact_bp"):
            try:
                last_impact_bp = float(getattr(candidate, "_last_market_impact_bp", 0.0))
            except Exception:
                last_impact_bp = 0.0
        if hasattr(candidate, "cumulative_collateral_funding_costs"):
            try:
                total_collateral_cost_dkk = float(getattr(candidate, "cumulative_collateral_funding_costs", 0.0))
            except Exception:
                total_collateral_cost_dkk = 0.0
        if hasattr(candidate, "_last_collateral_required_dkk"):
            try:
                last_collateral_required_dkk = float(getattr(candidate, "_last_collateral_required_dkk", 0.0))
            except Exception:
                last_collateral_required_dkk = 0.0
        if hasattr(candidate, "_last_collateral_funding_cost_dkk"):
            try:
                last_collateral_cost_dkk = float(getattr(candidate, "_last_collateral_funding_cost_dkk", 0.0))
            except Exception:
                last_collateral_cost_dkk = 0.0
        if hasattr(candidate, "_last_liquidity_participation"):
            try:
                last_liquidity_participation = float(getattr(candidate, "_last_liquidity_participation", 0.0))
            except Exception:
                last_liquidity_participation = 0.0
        if hasattr(candidate, "_last_liquidity_scale"):
            try:
                last_liquidity_scale = float(getattr(candidate, "_last_liquidity_scale", 1.0))
            except Exception:
                last_liquidity_scale = 1.0
        if hasattr(candidate, "_last_liquidity_tail_impact_multiplier"):
            try:
                last_liquidity_tail_multiplier = float(getattr(candidate, "_last_liquidity_tail_impact_multiplier", 1.0))
            except Exception:
                last_liquidity_tail_multiplier = 1.0
        if hasattr(candidate, "_impact_liquidity_source"):
            liquidity_source = str(getattr(candidate, "_impact_liquidity_source", "") or "")
    dkk_to_usd = float(getattr(cfg, "dkk_to_usd_rate", 0.145) if cfg is not None else 0.145)
    # market-impact task
    metrics["total_market_impact_cost_dkk"] = float(max(total_impact_dkk, 0.0))
    metrics["total_market_impact_cost_usd"] = float(max(total_impact_dkk, 0.0) * dkk_to_usd)
    metrics["last_market_impact_cost_dkk"] = float(max(last_impact_dkk, 0.0))
    metrics["last_market_impact_ref_notional_dkk"] = float(max(last_ref_notional, 0.0))
    metrics["last_market_impact_participation"] = float(max(last_participation, 0.0))
    metrics["last_market_impact_bp"] = float(max(last_impact_bp, 0.0))
    metrics["impact_liquidity_source"] = liquidity_source
    metrics["total_collateral_funding_cost_dkk"] = float(max(total_collateral_cost_dkk, 0.0))
    metrics["total_collateral_funding_cost_usd"] = float(max(total_collateral_cost_dkk, 0.0) * dkk_to_usd)
    metrics["last_collateral_required_dkk"] = float(max(last_collateral_required_dkk, 0.0))
    metrics["last_collateral_funding_cost_dkk"] = float(max(last_collateral_cost_dkk, 0.0))
    metrics["last_liquidity_participation"] = float(max(last_liquidity_participation, 0.0))
    metrics["last_liquidity_scale"] = float(max(last_liquidity_scale, 0.0))
    metrics["last_liquidity_tail_impact_multiplier"] = float(max(last_liquidity_tail_multiplier, 1.0))
    return metrics


def _attach_distribution_adjusted_context(
    metrics: Dict[str, Any],
    reported_nav_values: list,
    risk_levels: list,
    evaluation_steps: int,
    timestamps=None,
    annual_risk_free_rate: float = 0.0,
    final_total_distributions_usd: float = 0.0,
) -> Dict[str, Any]:
    """Primary metrics use total wealth; raw NAV metrics stay visible."""
    reported = calculate_performance_metrics(
        reported_nav_values,
        {},
        risk_levels,
        evaluation_steps,
        timestamps=timestamps,
        annual_risk_free_rate=annual_risk_free_rate,
    )
    for key in (
        "total_return",
        "volatility",
        "step_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "initial_portfolio_value",
        "final_portfolio_value",
    ):
        if key in reported:
            metrics[f"reported_nav_{key}"] = reported[key]
    metrics["distribution_adjusted_evaluation"] = True
    metrics["total_distributions_usd"] = float(max(0.0, final_total_distributions_usd))
    return metrics


def run_checkpoint_evaluation(models: Dict[str, Any],
                            eval_env,
                            data: pd.DataFrame,
                            evaluation_steps: int = 8000) -> Optional[Dict[str, Any]]:
    """Run evaluation using checkpoint models (Stable Baselines3)."""
    print("ÃƒÂ°Ã…Â¸Ã…Â¡Ã¢â€šÂ¬ Starting checkpoint model evaluation...")

    if not models or not eval_env:
        raise RuntimeError("[EVAL FAIL-HARD] No models or evaluation environment available.")
        print("ÃƒÂ¢Ã‚ÂÃ…â€™ No models or environment available")
        return None

    # Count loaded models
    loaded_count = sum(1 for m in models.values() if m is not None)
    model_total = max(1, len(models))
    print(f"ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯ Checkpoint models loaded: {loaded_count}/{model_total}")

    if loaded_count == 0:
        raise RuntimeError("[EVAL FAIL-HARD] No checkpoint models loaded successfully.")
        print("ÃƒÂ¢Ã‚ÂÃ…â€™ No checkpoint models loaded successfully")
        return None

    missing_agents = [
        agent for agent in getattr(eval_env, "possible_agents", [])
        if models.get(agent) is None
    ]
    if missing_agents:
        raise RuntimeError(
            "[EVAL FAIL-HARD] Missing checkpoint model(s) for "
            f"{missing_agents}; refusing rule/sample fallback because it invalidates tier comparisons."
        )

    # Run evaluation
    obs, _ = eval_env.reset()
    steps = min(evaluation_steps, len(data) - 1)

    # Tracking metrics
    portfolio_values = []
    reported_nav_values = []
    rewards_by_agent = {agent: [] for agent in eval_env.possible_agents}
    risk_levels = []
    successful_inference_actions = 0
    total_inference_attempts = 0
    model_inference_successes = 0
    model_inference_attempts = 0
    if not hasattr(eval_env, "_calculate_fund_nav"):
        raise RuntimeError("[EVAL FAIL-HARD] Evaluation env lacks _calculate_fund_nav at reset.")
    initial_nav_dkk = float(eval_env._calculate_fund_nav())
    initial_rate = float(getattr(getattr(eval_env, "config", None), "dkk_to_usd_rate", 0.145) or 0.145)
    initial_reported_nav_usd = initial_nav_dkk * initial_rate
    initial_total_distributions_usd = _get_total_distributions_dkk(eval_env) * initial_rate
    reported_nav_values.append(initial_reported_nav_usd)
    portfolio_values.append(initial_reported_nav_usd + initial_total_distributions_usd)

    print(f"ÃƒÂ°Ã…Â¸Ã‚Â§Ã‚Âª Running evaluation for {steps} steps...")

    for step in range(steps):
        actions = {}

        # Get actions from checkpoint models
        for agent in eval_env.possible_agents:
            if agent not in obs:
                continue

            total_inference_attempts += 1
            agent_key = agent.replace('_0', '_0')  # Normalize agent name

            model = models.get(agent_key)
            if _is_rule_based_checkpoint_model(model):
                try:
                    action = eval_env.get_rule_based_agent_action(agent_key, obs[agent])
                    actions[agent] = action
                    successful_inference_actions += 1
                except Exception as e:
                    raise RuntimeError(
                        f"[EVAL FAIL-HARD] {agent} rule-based prediction crashed; "
                        "saved checkpoint declares this controller as RULE. "
                        f"Underlying error: {type(e).__name__}: {e}"
                    ) from e
            elif agent_key in models and model is not None:
                model_inference_attempts += 1
                try:
                    # Use checkpoint model for prediction
                    action, _ = model.predict(obs[agent], deterministic=True)
                    actions[agent] = action
                    successful_inference_actions += 1
                    model_inference_successes += 1
                    if (
                        agent_key == "investor_0"
                        and hasattr(getattr(model, "policy", None), "get_mirror_diagnostics")
                        and bool(
                            getattr(
                                getattr(eval_env, "config", None),
                                "forecast_prior_mirror_mappo",
                                False,
                            )
                        )
                    ):
                        # CFM-MAPPO: the environment's execution recorder requires the
                        # decision-time policy snapshot. Training writes it from the
                        # metacontroller action path; checkpoint evaluation must
                        # mirror that hand-off here after each investor predict.
                        mirror = model.policy.get_mirror_diagnostics()

                        def _mirror_scalar(name: str) -> float:
                            value = np.asarray(
                                mirror.get(name, [0.0]), dtype=np.float64
                            ).reshape(-1)
                            return float(value[0]) if value.size else 0.0

                        eval_env._forecast_mirror_policy_snapshot = {
                            "step": int(getattr(eval_env, "t", step)),
                            "observation": np.asarray(
                                obs[agent], dtype=np.float32
                            ).reshape(-1).copy(),
                            "base_alpha": _mirror_scalar("base_alpha"),
                            "base_beta": _mirror_scalar("base_beta"),
                            "prior_alpha": _mirror_scalar("prior_alpha"),
                            "prior_beta": _mirror_scalar("prior_beta"),
                            "final_alpha": _mirror_scalar("final_alpha"),
                            "final_beta": _mirror_scalar("final_beta"),
                            "evidence": _mirror_scalar("evidence"),
                            "trust": _mirror_scalar("trust"),
                            "exponent": _mirror_scalar("exponent"),
                        }
                except Exception as e:
                    raise RuntimeError(
                        f"[EVAL FAIL-HARD] {agent} model prediction crashed; "
                        "refusing rule/sample fallback because it invalidates tier comparisons. "
                        f"Underlying error: {type(e).__name__}: {e}"
                    ) from e
            else:
                raise RuntimeError(
                    f"[EVAL FAIL-HARD] {agent} model is missing; refusing rule/sample fallback."
                )

        # Execute step
        try:
            obs, rewards, dones, truncs, infos = eval_env.step(actions)

            # Track metrics
            for agent, reward in rewards.items():
                rewards_by_agent[agent].append(float(reward))

            # Extract portfolio value and risk if available
            portfolio_value_dkk = None
            extraction_method = "unknown"

            # Try multiple methods to get portfolio value (in DKK)
            if hasattr(eval_env, 'get_portfolio_value'):
                portfolio_value_dkk = eval_env.get_portfolio_value()
                extraction_method = "get_portfolio_value"
            elif hasattr(eval_env, '_calculate_fund_nav'):
                # Direct access to environment's NAV calculation
                portfolio_value_dkk = eval_env._calculate_fund_nav()
                extraction_method = "_calculate_fund_nav"
            elif hasattr(eval_env, 'env') and hasattr(eval_env.env, '_calculate_fund_nav'):
                # Handle wrapped environment case
                portfolio_value_dkk = eval_env.env._calculate_fund_nav()
                extraction_method = "wrapped._calculate_fund_nav"
            elif 'fund_nav' in infos.get(list(eval_env.possible_agents)[0], {}):
                portfolio_value_dkk = infos[list(eval_env.possible_agents)[0]]['fund_nav']
                extraction_method = "infos.fund_nav"
            elif 'portfolio_value' in infos.get(list(eval_env.possible_agents)[0], {}):
                portfolio_value_dkk = infos[list(eval_env.possible_agents)[0]]['portfolio_value']
                extraction_method = "infos.portfolio_value"
            elif hasattr(eval_env, 'equity'):
                portfolio_value_dkk = eval_env.equity
                extraction_method = "equity"
            elif hasattr(eval_env, 'env') and hasattr(eval_env.env, 'equity'):
                # Handle wrapped environment case
                portfolio_value_dkk = eval_env.env.equity
                extraction_method = "wrapped.equity"
            else:
                raise RuntimeError(
                    "[EVAL FAIL-HARD] Could not extract portfolio NAV from evaluation environment; "
                    "refusing fallback initial value."
                )

            # Convert DKK to USD for consistent analysis
            # Get conversion rate from environment or use default
            dkk_to_usd_rate = 0.145  # Default rate
            if hasattr(eval_env, 'config') and hasattr(eval_env.config, 'dkk_to_usd_rate'):
                dkk_to_usd_rate = eval_env.config.dkk_to_usd_rate
            elif hasattr(eval_env, 'env') and hasattr(eval_env.env, 'config') and hasattr(eval_env.env.config, 'dkk_to_usd_rate'):
                dkk_to_usd_rate = eval_env.env.config.dkk_to_usd_rate

            # Convert to USD for analysis. Primary evaluation uses total
            # investor wealth: reported NAV plus cumulative cash distributions.
            reported_portfolio_value_usd = portfolio_value_dkk * dkk_to_usd_rate
            total_distributions_usd = _get_total_distributions_dkk(eval_env) * dkk_to_usd_rate
            portfolio_value_usd = reported_portfolio_value_usd + total_distributions_usd
            reported_nav_values.append(reported_portfolio_value_usd)
            portfolio_values.append(portfolio_value_usd)

            # Debug output for first step
            if step == 0:
                print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â Portfolio value extraction: {portfolio_value_dkk/1e9:.2f}B DKK ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ ${portfolio_value_usd/1e6:.1f}M USD using method '{extraction_method}'")

            if hasattr(eval_env, 'get_risk_level'):
                risk_levels.append(eval_env.get_risk_level())

            completed_steps = step + 1
            if completed_steps % 1000 == 0 or completed_steps == steps:
                _print_eval_loop_progress(
                    completed_steps=completed_steps,
                    total_steps=steps,
                    portfolio_values=portfolio_values,
                    rewards_by_agent=rewards_by_agent,
                    successful_inference_actions=successful_inference_actions,
                    total_inference_attempts=total_inference_attempts,
                )

            # Handle episode termination
            if any(dones.values()) or any(truncs.values()):
                if completed_steps < steps:
                    raise RuntimeError(
                        "[EVAL FAIL-HARD] Environment terminated early at step "
                        f"{completed_steps}/{steps}; refusing partial-path metrics."
                    )

        except Exception as e:
            raise RuntimeError(
                f"[EVAL FAIL-HARD] Checkpoint evaluation failed at step {step + 1}/{steps}: {e}"
            ) from e
            print(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Step execution error: {e}")
            break

    print("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Checkpoint evaluation completed")

    # Calculate metrics
    success_rate = (
        model_inference_successes / model_inference_attempts
        if model_inference_attempts > 0 else 0.0
    )
    success_rate_all_agents = (
        successful_inference_actions / total_inference_attempts
        if total_inference_attempts > 0 else 0.0
    )
    risk_free_rate = _resolve_eval_risk_free_rate(eval_env)
    metrics = calculate_performance_metrics(
        portfolio_values,
        rewards_by_agent,
        risk_levels,
        steps,
        timestamps=data.get("timestamp"),
        annual_risk_free_rate=risk_free_rate,
    )
    metrics = _attach_distribution_adjusted_context(
        metrics,
        reported_nav_values,
        risk_levels,
        steps,
        timestamps=data.get("timestamp"),
        annual_risk_free_rate=risk_free_rate,
        final_total_distributions_usd=(
            float(portfolio_values[-1] - reported_nav_values[-1])
            if portfolio_values and reported_nav_values
            else 0.0
        ),
    )
    metrics = _attach_eval_override_context(metrics, eval_env)

    # Add checkpoint-specific metrics
    metrics['action_inference_success_rate'] = success_rate
    metrics['action_inference_success_rate_all_agents'] = success_rate_all_agents
    metrics['models_loaded'] = loaded_count
    metrics['evaluation_mode'] = 'checkpoint'

    return metrics


def run_agent_evaluation(agent_system,
                        eval_env,
                        data: pd.DataFrame,
                        evaluation_steps: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Run evaluation using MultiESGAgent system."""
    print_progress("ÃƒÂ°Ã…Â¸Ã…Â¡Ã¢â€šÂ¬ Starting agent system evaluation...")

    if not agent_system or not eval_env:
        print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ No agent system or environment available")
        return None

    # Run evaluation
    print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Resetting evaluation environment...")
    try:
        obs, _ = eval_env.reset()
        print_progress("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Environment reset successful")
    except Exception as e:
        print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Environment reset failed: {e}")
        return None

    # Pick evaluation length
    if evaluation_steps is None:
        evaluation_steps = min(len(data) - 1, 10_000)
    evaluation_steps = int(max(1, evaluation_steps))

    print_progress(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‚Â Evaluating for {evaluation_steps} steps")

    # Initialize PPO buffer if needed
    if hasattr(agent_system, 'policies'):
        for policy in agent_system.policies:
            if hasattr(policy, 'policy') and hasattr(policy.policy, 'reset_noise'):
                try:
                    policy.policy.reset_noise()
                except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                    print_progress(
                        f"[PPO] reset_noise skipped for {getattr(policy, 'agent_name', 'unknown')}: {e}"
                    )
    print_progress("[PPO] PPO BUFFER RESET: Preserving financial state at step 0")

    # Tracking metrics
    portfolio_values = []
    reported_nav_values = []
    rewards_by_agent = {agent: [] for agent in eval_env.possible_agents}
    risk_levels = []
    actions_taken = {agent: [] for agent in eval_env.possible_agents}
    if not hasattr(eval_env, "_calculate_fund_nav"):
        raise RuntimeError("[EVAL FAIL-HARD] Evaluation env lacks _calculate_fund_nav at reset.")
    initial_nav_dkk = float(eval_env._calculate_fund_nav())
    initial_rate = float(getattr(getattr(eval_env, "config", None), "dkk_to_usd_rate", 0.145) or 0.145)
    initial_reported_nav_usd = initial_nav_dkk * initial_rate
    initial_total_distributions_usd = _get_total_distributions_dkk(eval_env) * initial_rate
    reported_nav_values.append(initial_reported_nav_usd)
    portfolio_values.append(initial_reported_nav_usd + initial_total_distributions_usd)

    print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Starting evaluation loop...")

    for step in range(evaluation_steps):
        if step == 0:
            print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Processing first step...")

        actions = {}

        if step == 0:
            print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Getting actions from agents...")

        # Get actions from agent system
        for i, agent in enumerate(eval_env.possible_agents):
            if agent not in obs:
                continue

            try:
                if step == 0:
                    print_progress(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Processing agent {agent} (obs shape: {np.array(obs[agent]).shape})")

                agent_obs = np.array(obs[agent], dtype=np.float32).reshape(1, -1)
                policy = agent_system.policies[i]

                if hasattr(policy, "predict"):
                    if step == 0:
                        print_progress(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Getting prediction from {agent}...")
                    act, _ = policy.predict(agent_obs, deterministic=True)
                    if step == 0:
                        print_progress(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Got prediction from {agent}")
                else:
                    raise RuntimeError(
                        f"[EVAL FAIL-HARD] {agent} policy has no predict() method; "
                        "refusing sample fallback."
                    )

                act = _coerce_action_for_space(act, eval_env.action_space(agent))
                actions[agent] = act
                actions_taken[agent].append(np.array(act).copy() if hasattr(act, 'copy') else act)

            except Exception as e:
                print_progress(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Action prediction error for {agent}: {e}")
                raise RuntimeError(
                    f"[EVAL FAIL-HARD] {agent} policy prediction crashed in agent evaluation; "
                    f"refusing sample fallback. Underlying error: {type(e).__name__}: {e}"
                ) from e

        if step == 0:
            print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Executing first environment step...")

        # Execute step
        try:
            obs, rewards, dones, truncs, infos = eval_env.step(actions)
            if step == 0:
                print_progress("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ First environment step completed")

            # Track metrics
            for agent, reward in rewards.items():
                rewards_by_agent[agent].append(float(reward))

            # Extract portfolio value and risk if available
            portfolio_value_dkk = None

            # Try multiple methods to get portfolio value (in DKK)
            if hasattr(eval_env, 'get_portfolio_value'):
                portfolio_value_dkk = eval_env.get_portfolio_value()
            elif hasattr(eval_env, '_calculate_fund_nav'):
                # Direct access to environment's NAV calculation
                portfolio_value_dkk = eval_env._calculate_fund_nav()
            elif hasattr(eval_env, 'env') and hasattr(eval_env.env, '_calculate_fund_nav'):
                # Handle wrapped environment case
                portfolio_value_dkk = eval_env.env._calculate_fund_nav()
            elif 'fund_nav' in infos.get(list(eval_env.possible_agents)[0], {}):
                portfolio_value_dkk = infos[list(eval_env.possible_agents)[0]]['fund_nav']
            elif 'portfolio_value' in infos.get(list(eval_env.possible_agents)[0], {}):
                portfolio_value_dkk = infos[list(eval_env.possible_agents)[0]]['portfolio_value']
            elif hasattr(eval_env, 'equity'):
                portfolio_value_dkk = eval_env.equity
            elif hasattr(eval_env, 'env') and hasattr(eval_env.env, 'equity'):
                # Handle wrapped environment case
                portfolio_value_dkk = eval_env.env.equity
            else:
                raise RuntimeError(
                    "[EVAL FAIL-HARD] Could not extract portfolio NAV from evaluation environment; "
                    "refusing fallback initial value."
                )

            # Convert DKK to USD for consistent analysis
            # Get conversion rate from environment or use default
            dkk_to_usd_rate = 0.145  # Default rate
            if hasattr(eval_env, 'config') and hasattr(eval_env.config, 'dkk_to_usd_rate'):
                dkk_to_usd_rate = eval_env.config.dkk_to_usd_rate
            elif hasattr(eval_env, 'env') and hasattr(eval_env.env, 'config') and hasattr(eval_env.env.config, 'dkk_to_usd_rate'):
                dkk_to_usd_rate = eval_env.env.config.dkk_to_usd_rate

            # Convert to USD for analysis. Primary evaluation uses total
            # investor wealth: reported NAV plus cumulative cash distributions.
            reported_portfolio_value_usd = portfolio_value_dkk * dkk_to_usd_rate
            total_distributions_usd = _get_total_distributions_dkk(eval_env) * dkk_to_usd_rate
            portfolio_value_usd = reported_portfolio_value_usd + total_distributions_usd
            reported_nav_values.append(reported_portfolio_value_usd)
            portfolio_values.append(portfolio_value_usd)

            if hasattr(eval_env, 'get_risk_level'):
                risk_levels.append(eval_env.get_risk_level())

            completed_steps = step + 1
            if completed_steps % 1000 == 0 or completed_steps == evaluation_steps:
                _print_eval_loop_progress(
                    completed_steps=completed_steps,
                    total_steps=evaluation_steps,
                    portfolio_values=portfolio_values,
                    rewards_by_agent=rewards_by_agent,
                )

            # Handle episode termination
            if any(dones.values()) or any(truncs.values()):
                if completed_steps < evaluation_steps:
                    raise RuntimeError(
                        "[EVAL FAIL-HARD] Environment terminated early at step "
                        f"{completed_steps}/{evaluation_steps}; refusing partial-path metrics."
                    )

        except Exception as e:
            raise RuntimeError(
                "[EVAL FAIL-HARD] Agent evaluation failed at step "
                f"{step + 1}/{evaluation_steps}: {e}"
            ) from e
            print(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Step execution error: {e}")
            break

    print("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Agent evaluation completed")

    # Calculate metrics
    risk_free_rate = _resolve_eval_risk_free_rate(eval_env)
    metrics = calculate_performance_metrics(
        portfolio_values,
        rewards_by_agent,
        risk_levels,
        evaluation_steps,
        timestamps=data.get("timestamp"),
        annual_risk_free_rate=risk_free_rate,
    )
    metrics = _attach_distribution_adjusted_context(
        metrics,
        reported_nav_values,
        risk_levels,
        evaluation_steps,
        timestamps=data.get("timestamp"),
        annual_risk_free_rate=risk_free_rate,
        final_total_distributions_usd=(
            float(portfolio_values[-1] - reported_nav_values[-1])
            if portfolio_values and reported_nav_values
            else 0.0
        ),
    )
    metrics = _attach_eval_override_context(metrics, eval_env)
    metrics['evaluation_mode'] = 'agents'

    return metrics


def run_comprehensive_evaluation(eval_data: pd.DataFrame, args) -> Dict[str, Any]:
    """Run comprehensive evaluation comparing all configurations with separate environments."""
    print_section_header("COMPREHENSIVE EVALUATION: All Configurations")
    print_progress("ÃƒÂ°Ã…Â¸Ã…Â¡Ã¢â€šÂ¬ Starting comprehensive evaluation of 5 systems...")

    comprehensive_results = {
        'evaluation_type': 'comprehensive',
        'timestamp': datetime.now().isoformat(),
        'eval_data_path': args.eval_data,
        'eval_steps': args.eval_steps,
        'configurations': {}
    }

    # 1. Evaluate Baselines (use basic environment)
    print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  [1/3] EVALUATING BASELINES...", 1, 3)
    try:
        baseline_results = run_traditional_baselines(
            args.eval_data,
            args.eval_steps or 10000,
            args.output_dir,
            seed=getattr(args, "seed", 42),
        )
        if baseline_results:
            comprehensive_results['configurations']['baselines'] = baseline_results
            print_progress("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Baseline evaluation completed")
        else:
            print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ Baseline evaluation failed")
            comprehensive_results['configurations']['baselines'] = {'error': 'Baseline evaluation failed'}
    except Exception as e:
        print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Baseline evaluation error: {e}")
        comprehensive_results['configurations']['baselines'] = {'error': str(e)}

    # 2. Evaluate Normal Models (no forecasts, no Tier1 runtime) - Create basic environment
    print_progress("[2/3] EVALUATING NORMAL MODELS", 2, 3)
    try:
        if os.path.exists(args.normal_models):
            # Create basic environment for normal models.
            print_progress("ÃƒÂ°Ã…Â¸Ã‚ÂÃ¢â‚¬â€ÃƒÂ¯Ã‚Â¸Ã‚Â Creating basic environment for normal models...")
            normal_cfg = _build_eval_config(args)
            normal_cfg.enable_forecast_utilization = False
            normal_eval_env = create_evaluation_environment(
                eval_data,
                output_dir=args.output_dir,
                investment_freq=int(getattr(normal_cfg, "investment_freq", args.investment_freq) or args.investment_freq),
                config=normal_cfg,
                fail_fast=True,
            )

            if normal_eval_env:
                # Load normal models without enhanced features
                print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Loading normal agent system...")
                normal_agent_system = load_agent_system(args.normal_models, normal_eval_env, enhanced=False)
                if normal_agent_system:
                    print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Running normal agent evaluation...")
                    normal_results = run_agent_evaluation(normal_agent_system, normal_eval_env, eval_data, args.eval_steps)
                    if normal_results:
                        normal_results['model_type'] = 'normal'
                        normal_results['features'] = {'forecast_utilization': False, 'tier1_reward_shaping': False}
                        comprehensive_results['configurations']['normal_agents'] = normal_results
                        print_progress("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Normal agent evaluation completed")
                    else:
                        print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ Normal agent evaluation failed")
                        comprehensive_results['configurations']['normal_agents'] = {'error': 'Normal agent evaluation failed'}
                else:
                    print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ Failed to load normal agent system")
                    comprehensive_results['configurations']['normal_agents'] = {'error': 'Failed to load normal agent system'}
            else:
                print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ Failed to create basic evaluation environment")
                comprehensive_results['configurations']['normal_agents'] = {'error': 'Failed to create basic evaluation environment'}
        else:
            print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Normal models directory not found: {args.normal_models}")
            comprehensive_results['configurations']['normal_agents'] = {'error': f'Directory not found: {args.normal_models}'}
    except Exception as e:
        print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Normal agent evaluation error: {e}")
        comprehensive_results['configurations']['normal_agents'] = {'error': str(e)}

    # 3. Evaluate Full Models with the fixed Tier1 runtime.
    print_progress("[3/3] EVALUATING FULL MODELS", 3, 3)
    try:
        if os.path.exists(args.full_models):
            # Create enhanced evaluation environment with the fixed-observation runtime.
            print_progress("Creating enhanced environment for full models...")
            full_cfg = _build_eval_config(args)
            forecast_ctx = None
            if bool(getattr(full_cfg, "enable_forecast_utilization", False)):
                forecast_ctx = _resolve_reserved_eval_forecast_context(args, output_dir=args.output_dir, config=full_cfg)
            enhanced_eval_env = create_evaluation_environment(
                eval_data,
                output_dir=args.output_dir,
                investment_freq=int(getattr(full_cfg, "investment_freq", args.investment_freq) or args.investment_freq),
                config=full_cfg,
                forecast_cache_dir=forecast_ctx["forecast_cache_dir"] if forecast_ctx else args.forecast_cache_dir,
                fail_fast=True,
            )

            if enhanced_eval_env:
                # Load enhanced models with all features
                print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Loading enhanced agent system...")
                full_agent_system = load_enhanced_agent_system(args.full_models, enhanced_eval_env)
                if full_agent_system:
                    print_progress("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Running full agent evaluation...")
                    full_results = run_agent_evaluation(full_agent_system, enhanced_eval_env, eval_data, args.eval_steps)
                    if full_results:
                        full_results['model_type'] = 'enhanced'
                        full_results['features'] = {
                            'forecast_utilization': bool(getattr(full_cfg, "enable_forecast_utilization", False)),
                            'tier1_reward_shaping': False,
                        }
                        comprehensive_results['configurations']['full_agents'] = full_results
                        print_progress("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Full agent evaluation completed")
                    else:
                        print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ Full agent evaluation failed")
                        comprehensive_results['configurations']['full_agents'] = {'error': 'Full agent evaluation failed'}
                else:
                    print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ Failed to load full agent system")
                    comprehensive_results['configurations']['full_agents'] = {'error': 'Failed to load full agent system'}
            else:
                print_progress("ÃƒÂ¢Ã‚ÂÃ…â€™ Failed to create enhanced evaluation environment")
                comprehensive_results['configurations']['full_agents'] = {'error': 'Failed to create enhanced evaluation environment'}
        else:
            print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Full models directory not found: {args.full_models}")
            comprehensive_results['configurations']['full_agents'] = {'error': f'Directory not found: {args.full_models}'}
    except Exception as e:
        print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Full agent evaluation error: {e}")
        comprehensive_results['configurations']['full_agents'] = {'error': str(e)}

    print_progress("ÃƒÂ°Ã…Â¸Ã…Â½Ã¢â‚¬Â° Comprehensive evaluation completed!")
    return comprehensive_results


def save_results(results: Dict[str, Any], output_dir: str, mode: str, analysis: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str]]:
    """Save evaluation results and analysis to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save main results
    results_file = os.path.join(output_dir, f"evaluation_{mode}_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â¾ Results saved to: {results_file}")

    # Save analysis if provided
    analysis_file = None
    if analysis:
        analysis_file = os.path.join(output_dir, f"analysis_{mode}_{timestamp}.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Analysis saved to: {analysis_file}")

    return results_file, analysis_file


def analyze_comprehensive_results(comprehensive_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze comprehensive evaluation results across all configurations."""
    print("\nÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  COMPREHENSIVE ANALYSIS")
    print("="*50)

    analysis = {
        'summary': {},
        'performance_ranking': [],
        'feature_impact': {},
        'statistical_comparison': {}
    }

    configurations = comprehensive_results.get('configurations', {})

    # Extract performance metrics for comparison
    performance_data = {}

    # ========================================
    # ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  FINAL PORTFOLIO VALUES TABLE
    # ========================================
    print("\n" + "=" * 80)
    print("ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  FINAL PORTFOLIO VALUES - ALL CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Rank':<4} {'Configuration':<25} {'Final Value':<15} {'Return':<10} {'Sharpe':<8} {'Features':<20}")
    print("-" * 80)

    for config_name, config_results in configurations.items():
        if 'error' in config_results:
            print(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â {config_name}: {config_results['error']}")
            continue

        # Extract key metrics
        if config_name == 'baselines':
            # Handle baseline results structure
            for baseline_name, baseline_data in config_results.items():
                if isinstance(baseline_data, dict) and baseline_data.get('status') != 'failed':
                    config_display_name = baseline_data.get('method', baseline_name)

                    # Extract portfolio values correctly
                    final_value = baseline_data.get('final_portfolio_value', baseline_data.get('final_value_usd', 800_000_000))
                    initial_value = baseline_data.get('initial_portfolio_value', baseline_data.get('initial_value_usd', 800_000_000))
                    total_return = baseline_data.get('total_return', 0) * 100  # Convert to percentage

                    performance_data[config_display_name] = {
                        'total_return': total_return,
                        'sharpe_ratio': baseline_data.get('sharpe_ratio', 0),
                        'max_drawdown': baseline_data.get('max_drawdown', 0) * 100,  # Convert to percentage
                        'final_value_usd': final_value,
                        'initial_value_usd': initial_value,
                        'type': 'baseline'
                    }
        else:
            # Handle agent results structure
            if 'total_return' in config_results or 'final_portfolio_value' in config_results:
                features = config_results.get('features', {})
                config_display_name = "Normal Agents" if config_name == 'normal_agents' else "Full Agents"

                # Extract portfolio values correctly
                final_value = config_results.get('final_portfolio_value', 800_000_000)
                initial_value = config_results.get('initial_portfolio_value', 800_000_000)
                total_return = config_results.get('total_return', 0) * 100  # Convert to percentage

                performance_data[config_display_name] = {
                    'total_return': total_return,
                    'sharpe_ratio': config_results.get('sharpe_ratio', 0),
                    'max_drawdown': config_results.get('max_drawdown', 0) * 100,  # Convert to percentage
                    'final_value_usd': final_value,
                    'initial_value_usd': initial_value,
                    'type': 'agent',
                    'forecast_utilization': features.get('forecast_utilization', False),
                    'tier1_reward_shaping': features.get('tier1_reward_shaping', False)
                }

    # Sort by final portfolio value for the main table
    ranked_by_value = sorted(performance_data.items(), key=lambda x: x[1]['final_value_usd'], reverse=True)

    for i, (config_name, metrics) in enumerate(ranked_by_value, 1):
        final_value = metrics['final_value_usd']
        initial_value = metrics['initial_value_usd']
        return_pct = ((final_value - initial_value) / initial_value * 100) if initial_value > 0 else 0.0

        # Format features
        features_str = ""
        if metrics['type'] == 'agent':
            features = []
            if metrics.get('forecast_utilization'): features.append("Forecast Utilization")
            if metrics.get('tier1_reward_shaping', False): features.append("Tier1 Reward Shaping")
            features_str = ', '.join(features) if features else 'Basic RL'
        else:
            features_str = 'Traditional'

        # Color coding for top performers
        rank_symbol = "ÃƒÂ°Ã…Â¸Ã‚Â¥Ã¢â‚¬Â¡" if i == 1 else "ÃƒÂ°Ã…Â¸Ã‚Â¥Ã‹â€ " if i == 2 else "ÃƒÂ°Ã…Â¸Ã‚Â¥Ã¢â‚¬Â°" if i == 3 else f"{i}."

        print(f"{rank_symbol:<4} {config_name:<25} ${final_value/1e6:>10.1f}M {return_pct:>+7.2f}% {metrics['sharpe_ratio']:>6.3f} {features_str:<20}")

    print("-" * 80)
    print(f"{'Initial Portfolio Value:':<41} $800.0M")
    if ranked_by_value:
        print(f"{'Best Performer:':<41} {ranked_by_value[0][0]} (${ranked_by_value[0][1]['final_value_usd']/1e6:.1f}M)")
        improvement = ((ranked_by_value[0][1]['final_value_usd'] - 800_000_000) / 800_000_000 * 100)
        print(f"{'Best Return:':<41} +{improvement:.2f}%")
    print("=" * 80)

    # Rank by Sharpe ratio for detailed analysis
    ranked_configs = sorted(performance_data.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)

    print("\nÃƒÂ°Ã…Â¸Ã‚ÂÃ¢â‚¬Â  PERFORMANCE RANKING (by Sharpe Ratio):")
    for i, (config_name, metrics) in enumerate(ranked_configs, 1):
        features_str = ""
        if metrics['type'] == 'agent':
            features = []
            if metrics.get('forecast_utilization'): features.append("Forecast Utilization")
            if metrics.get('tier1_reward_shaping', False): features.append("Tier1 Reward Shaping")
            features_str = f" ({', '.join(features) if features else 'Basic'})"

        print(f"   {i}. {config_name}{features_str}")
        print(f"      Return: {metrics['total_return']:.2f}% | Sharpe: {metrics['sharpe_ratio']:.3f} | Drawdown: {metrics['max_drawdown']:.2f}%")

    analysis['performance_ranking'] = ranked_configs

    # Feature impact analysis
    if len([x for x in performance_data.values() if x['type'] == 'agent']) >= 2:
        print("\nÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â¬ FEATURE IMPACT ANALYSIS:")

        # Find normal vs full agents
        normal_metrics = None
        full_metrics = None

        for config_name, metrics in performance_data.items():
            if metrics['type'] == 'agent':
                if not metrics.get('forecast_utilization') and not metrics.get('tier1_reward_shaping', False):
                    normal_metrics = metrics
                elif metrics.get('forecast_utilization') and metrics.get('tier1_reward_shaping', False):
                    full_metrics = metrics

        if normal_metrics and full_metrics:
            return_improvement = full_metrics['total_return'] - normal_metrics['total_return']
            sharpe_improvement = full_metrics['sharpe_ratio'] - normal_metrics['sharpe_ratio']

            print("   Enhanced Features Impact:")
            print(f"      Return Improvement: {return_improvement:+.2f}%")
            print(f"      Sharpe Improvement: {sharpe_improvement:+.3f}")

            analysis['feature_impact'] = {
                'return_improvement_pct': return_improvement,
                'sharpe_improvement': sharpe_improvement,
                'normal_performance': normal_metrics,
                'enhanced_performance': full_metrics
            }

    # Best performer summary
    if ranked_configs:
        best_config = ranked_configs[0]
        analysis['summary']['best_performer'] = {
            'name': best_config[0],
            'metrics': best_config[1]
        }

        print(f"\nÃƒÂ°Ã…Â¸Ã‚Â¥Ã¢â‚¬Â¡ BEST PERFORMER: {best_config[0]}")
        print(f"   Sharpe Ratio: {best_config[1]['sharpe_ratio']:.3f}")
        print(f"   Total Return: {best_config[1]['total_return']:.2f}%")

    return analysis


def main():
    """Main evaluation function."""
    setup_console_encoding()
    print_progress("Starting Comprehensive Evaluation System")
    parser = argparse.ArgumentParser(description="Unified Evaluation Script")

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["checkpoint", "agents", "baselines", "compare", "comprehensive", "tiers"],
        required=True,
        help=(
            "Evaluation mode: "
            "'checkpoint' for latest checkpoint, "
            "'agents' for custom agent directory, "
            "'baselines' for traditional methods, "
            "'compare' for comprehensive comparison, "
            "'comprehensive' for all configurations, "
            "'tiers' to evaluate Tier1/baseline models on unseen data using final_models/"
        ),
    )

    # Data and model paths
    parser.add_argument("--eval_data", type=str, default="evaluation_dataset/unseendata.csv",
                       help="Path to evaluation data CSV")
    parser.add_argument("--trained_agents", type=str, default=None,
                       help="Directory with saved agent policies (required for 'agents' mode)")
    parser.add_argument("--checkpoint_dir", type=str, default="normal/checkpoints",
                       help="Base directory for checkpoints (for 'checkpoint' mode)")

    # Enhanced model support
    parser.add_argument("--enhanced_models", type=str, default=None,
                       help="Directory with enhanced models")

    # Comprehensive evaluation paths
    parser.add_argument("--normal_models", type=str, default="normal/final_models",
                       help="Directory with normal agent policies")
    parser.add_argument("--full_models", type=str, default="full/final_models",
                       help="Directory with full agent policies")

    # Evaluation options
    parser.add_argument("--eval_steps", type=int, default=None,
                       help="Number of timesteps to evaluate (default: auto)")
    parser.add_argument(
        "--eval-distribution-rate",
        "--eval_distribution_rate",
        dest="eval_distribution_rate",
        type=float,
        default=None,
        help="Evaluation-only cash-sweeper distribution rate override. Omit to preserve training/live behavior.",
    )
    parser.add_argument(
        "--distribution-rate",
        "--distribution_rate",
        dest="distribution_rate",
        type=float,
        default=None,
        help="Set the base cash-sweeper distribution rate before any evaluation-only override.",
    )
    parser.add_argument(
        "--friction-cost-multiplier",
        "--friction_cost_multiplier",
        dest="friction_cost_multiplier",
        type=float,
        default=None,
        help="Evaluation-only multiplier for trading transaction costs. Omit to preserve saved-config behavior.",
    )
    parser.add_argument(
        "--market-fee-model", "--market_fee_model",
        dest="market_fee_model",
        choices=["legacy_notional_fixed", "nord_pool_intraday_2026"],
        default=None,
    )
    parser.add_argument(
        "--transaction-fee-dkk-per-mwh", "--transaction_fee_dkk_per_mwh",
        dest="transaction_fee_dkk_per_mwh", type=float, default=None,
    )
    parser.add_argument(
        "--annual-market-access-fee-dkk", "--annual_market_access_fee_dkk",
        dest="annual_market_access_fee_dkk", type=float, default=None,
    )
    parser.add_argument(
        "--market-access-fee-allocation-fraction", "--market_access_fee_allocation_fraction",
        dest="market_access_fee_allocation_fraction", type=float, default=None,
    )
    parser.add_argument(
        "--market-fee-source-id", "--market_fee_source_id",
        dest="market_fee_source_id", type=str, default=None,
    )
    parser.add_argument(
        "--no-trade-threshold",
        "--no_trade_threshold",
        dest="no_trade_threshold",
        type=float,
        default=None,
        help="Fraction of executable capacity (or raw DKK when above 1) below which rebalances are ignored.",
    )
    parser.add_argument(
        "--no-trade-threshold-reference",
        "--no_trade_threshold_reference",
        dest="no_trade_threshold_reference",
        choices=["executable_capacity", "max_position"],
        default=None,
    )
    parser.add_argument(
        "--half-spread-bp",
        "--half_spread_bp",
        dest="half_spread_bp",
        type=float,
        default=None,
        help="Evaluation-only half-spread cost in basis points on traded notional. Omit to preserve saved-config behavior.",
    )
    parser.add_argument(
        "--impact-coef-bp",
        "--impact_coef_bp",
        dest="impact_coef_bp",
        type=float,
        default=None,
        help="Evaluation-only temporary market-impact coefficient in bp at full-sleeve participation.",
    )
    parser.add_argument(
        "--impact-exponent",
        "--impact_exponent",
        dest="impact_exponent",
        type=float,
        default=None,
        help="Evaluation-only market-impact exponent; 0.5 is the square-root law.",
    )
    parser.add_argument(
        "--impact-ref-notional",
        "--impact_ref_notional",
        dest="impact_ref_notional",
        type=str,
        default=None,
        help="Evaluation-only market-impact Q_ref in DKK, 'sleeve' for current trading cash, or 'volume' for market-volume calibration.",
    )
    parser.add_argument(
        "--impact-volume-data",
        "--impact_volume_data",
        dest="impact_volume_data",
        type=str,
        default=None,
        help="CSV with timestamped market volume for --impact-ref-notional volume.",
    )
    parser.add_argument(
        "--impact-volume-column",
        "--impact_volume_column",
        dest="impact_volume_column",
        type=str,
        default=None,
        help="Volume column name, or comma-separated columns to sum. Blank autodetects a single volume column.",
    )
    parser.add_argument(
        "--impact-volume-unit",
        "--impact_volume_unit",
        dest="impact_volume_unit",
        type=str,
        choices=["mwh", "dkk"],
        default=None,
        help="Unit of impact volume data: MWh converted via spot price, or DKK notional.",
    )
    parser.add_argument(
        "--impact-volume-timestamp-column",
        "--impact_volume_timestamp_column",
        dest="impact_volume_timestamp_column",
        type=str,
        default=None,
        help="Timestamp column in --impact-volume-data.",
    )
    parser.add_argument(
        "--impact-volume-max-staleness-min",
        "--impact_volume_max_staleness_min",
        dest="impact_volume_max_staleness_min",
        type=float,
        default=None,
        help="Maximum minutes to forward-fill hourly market volume onto the 10-minute eval grid.",
    )
    parser.add_argument(
        "--impact-volume-price-floor-dkk-per-mwh",
        "--impact_volume_price_floor_dkk_per_mwh",
        dest="impact_volume_price_floor_dkk_per_mwh",
        type=float,
        default=None,
        help="Price floor used when converting MWh volume to DKK notional for volume-calibrated impact.",
    )
    parser.add_argument("--liquidity-participation-cap-fraction", "--liquidity_participation_cap_fraction", dest="liquidity_participation_cap_fraction", type=float, default=None)
    parser.add_argument("--liquidity-volume-source", "--liquidity_volume_source", dest="liquidity_volume_source", choices=["load", "generation", "max_load_generation", "impact_volume"], default=None)
    parser.add_argument("--liquidity-volume-multiplier", "--liquidity_volume_multiplier", dest="liquidity_volume_multiplier", type=float, default=None)
    parser.add_argument("--liquidity-min-volume-mwh", "--liquidity_min_volume_mwh", dest="liquidity_min_volume_mwh", type=float, default=None)
    parser.add_argument("--liquidity-tail-impact-threshold-dkk-per-mwh", "--liquidity_tail_impact_threshold_dkk_per_mwh", dest="liquidity_tail_impact_threshold_dkk_per_mwh", type=float, default=None)
    parser.add_argument("--liquidity-tail-impact-multiplier", "--liquidity_tail_impact_multiplier", dest="liquidity_tail_impact_multiplier", type=float, default=None)
    parser.add_argument("--liquidity-tail-impact-power", "--liquidity_tail_impact_power", dest="liquidity_tail_impact_power", type=float, default=None)
    parser.add_argument("--liquidity-tail-impact-max-multiplier", "--liquidity_tail_impact_max_multiplier", dest="liquidity_tail_impact_max_multiplier", type=float, default=None)
    parser.add_argument("--enable-collateral-cash-drag", "--enable_collateral_cash_drag", dest="enable_collateral_cash_drag", action="store_true", default=None)
    parser.add_argument("--disable-collateral-cash-drag", "--disable_collateral_cash_drag", dest="disable_collateral_cash_drag", action="store_true", default=False)
    parser.add_argument("--collateral-notional-margin-fraction", "--collateral_notional_margin_fraction", dest="collateral_notional_margin_fraction", type=float, default=None)
    parser.add_argument("--collateral-stress-loss-fraction", "--collateral_stress_loss_fraction", dest="collateral_stress_loss_fraction", type=float, default=None)
    parser.add_argument("--collateral-stress-price-dkk-per-mwh", "--collateral_stress_price_dkk_per_mwh", dest="collateral_stress_price_dkk_per_mwh", type=float, default=None)
    parser.add_argument("--collateral-funding-rate-annual", "--collateral_funding_rate_annual", dest="collateral_funding_rate_annual", type=float, default=None)
    parser.add_argument("--collateral-tradeable-haircut", "--collateral_tradeable_haircut", dest="collateral_tradeable_haircut", type=float, default=None)
    parser.add_argument(
        "--log-sleeve",
        dest="log_sleeve",
        action="store_true",
        default=False,
        help="Write per-step env debug CSVs for trading-sleeve supplement metrics.",
    )
    parser.add_argument(
        "--sleeve-sharpe-mode",
        "--sleeve_sharpe_mode",
        dest="sleeve_sharpe_mode",
        choices=["daily", "daily_hac_7", "hac_144"],
        default="daily_hac_7",
        help="Primary trading-sleeve Sharpe stored as sleeve_trading_sharpe_ratio.",
    )
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for deterministic baseline diagnostics")
    parser.add_argument("--investment_freq", type=int, default=6,
                       help="Investor action frequency for evaluation (should match training; default 6)")
    parser.add_argument("--meta_freq_min", type=int, default=None,
                       help="Minimum live investor trade cadence for evaluation (should match training).")
    parser.add_argument("--meta_freq_max", type=int, default=None,
                       help="Maximum live investor trade cadence for evaluation (should match training).")
    parser.add_argument(
        "--global_norm_mode",
        type=str,
        default="rolling_past",
        choices=["rolling_past", "global"],
        help="Normalization mode for evaluation config; keep aligned with training defaults.",
    )
    parser.add_argument(
        "--rolling_past_history_dir",
        type=str,
        default="",
        help=(
            "Directory with history_*.csv for rolling_past eval bootstrap when global_norm_mode=rolling_past "
            f"(default: {EVAL_ROLLING_PAST_HISTORY_DIR})."
        ),
    )

    # Tier1/baseline evaluation.
    parser.add_argument("--tier1_dir", type=str, default="tier1_seed789",
                       help="Tier1 run directory containing final_models/")
    parser.add_argument(
        "--tiers_only",
        type=str,
        default="tier1",
        choices=[
            "tier1",
            "baseline",
        ],
        help="Run only a subset in --mode tiers.",
    )
    parser.set_defaults(tier1_dir=None)
    parser.add_argument(
        "--forecast_cache_dir",
        type=str,
        default="forecast_cache",
        help="Base directory for forecast caches.",
    )
    parser.add_argument(
        "--mtm_return_model",
        "--mtm-return-model",
        dest="mtm_return_model",
        type=str,
        default=None,
        choices=[
            "percent",
            "percent_capped",
            "notional_price_diff",
            "horizon_settlement",
            "horizon_settlement_continuous",
        ],
        help=(
            "Financial MTM payoff model. Use horizon_settlement for strict "
            "fixed-horizon settlement against a causal entry benchmark."
        ),
    )
    parser.add_argument(
        "--mtm_reference_price_dkk_per_mwh",
        "--mtm-reference-price-dkk-per-mwh",
        dest="mtm_reference_price_dkk_per_mwh",
        type=float,
        default=None,
        help="Reference price used by notional_price_diff MTM.",
    )
    parser.add_argument("--mtm_settlement_horizon_steps", "--mtm-settlement-horizon-steps", dest="mtm_settlement_horizon_steps", type=int, default=None)
    parser.add_argument(
        "--mtm_entry_price_mode",
        "--mtm-entry-price-mode",
        dest="mtm_entry_price_mode",
        type=str,
        default=None,
        choices=["same_hour_prev_day", "rolling_same_hour_median", "current_price"],
    )
    parser.add_argument(
        "--mtm_horizon_payoff_denominator_mode",
        "--mtm-horizon-payoff-denominator-mode",
        dest="mtm_horizon_payoff_denominator_mode",
        type=str,
        default=None,
        choices=["reference_price", "entry_price_floor", "mwh_volume", "mwh"],
        help=(
            "Denominator for horizon_settlement payoff. reference_price preserves "
            "price_diff/reference; entry_price_floor uses price_diff/max(abs(entry_price), reference); "
            "mwh_volume stores explicit MWh volume and pays volume*(settlement-entry)."
        ),
    )
    parser.add_argument(
        "--mtm_settlement_price_mode",
        "--mtm-settlement-price-mode",
        dest="mtm_settlement_price_mode",
        type=str,
        default=None,
        choices=[
            "energy_index",
            "none",
            "base",
            "cross_zone_basis",
            "basis_adjusted",
            "external_series",
            "external",
            "realized",
            "real_settlement",
        ],
        help="Settlement price process for horizon_settlement MTM.",
    )
    parser.add_argument("--mtm_basis_price_data_path", "--mtm-basis-price-data-path", dest="mtm_basis_price_data_path", type=str, default=None)
    parser.add_argument("--mtm_basis_price_column", "--mtm-basis-price-column", dest="mtm_basis_price_column", type=str, default=None)
    parser.add_argument("--mtm_basis_timestamp_column", "--mtm-basis-timestamp-column", dest="mtm_basis_timestamp_column", type=str, default=None)
    parser.add_argument("--mtm_basis_scale", "--mtm-basis-scale", dest="mtm_basis_scale", type=float, default=None)
    parser.add_argument(
        "--mtm_basis_centering_mode",
        "--mtm-basis-centering-mode",
        dest="mtm_basis_centering_mode",
        type=str,
        default=None,
        choices=["none", "rolling_median", "expanding_median"],
    )
    parser.add_argument("--mtm_basis_centering_window_steps", "--mtm-basis-centering-window-steps", dest="mtm_basis_centering_window_steps", type=int, default=None)
    parser.add_argument("--mtm_external_settlement_price_data_path", "--mtm-external-settlement-price-data-path", dest="mtm_external_settlement_price_data_path", type=str, default=None)
    parser.add_argument("--mtm_external_settlement_price_column", "--mtm-external-settlement-price-column", dest="mtm_external_settlement_price_column", type=str, default=None)
    parser.add_argument("--mtm_external_settlement_timestamp_column", "--mtm-external-settlement-timestamp-column", dest="mtm_external_settlement_timestamp_column", type=str, default=None)
    parser.add_argument("--mtm_external_settlement_min_price_dkk_per_mwh", "--mtm-external-settlement-min-price-dkk-per-mwh", dest="mtm_external_settlement_min_price_dkk_per_mwh", type=float, default=None)
    parser.add_argument("--mtm_external_settlement_max_price_dkk_per_mwh", "--mtm-external-settlement-max-price-dkk-per-mwh", dest="mtm_external_settlement_max_price_dkk_per_mwh", type=float, default=None)
    parser.add_argument(
        "--disable_mtm_return_cap",
        "--disable-mtm-return-cap",
        dest="disable_mtm_return_cap",
        action="store_true",
        help="Disable MTM return clipping.",
    )
    parser.add_argument("--mtm_price_return_cap_min", "--mtm-price-return-cap-min", dest="mtm_price_return_cap_min", type=float, default=None)
    parser.add_argument("--mtm_price_return_cap_max", "--mtm-price-return-cap-max", dest="mtm_price_return_cap_max", type=float, default=None)
    parser.add_argument(
        "--investor_notional_sizing_base",
        "--investor-notional-sizing-base",
        dest="investor_notional_sizing_base",
        type=str,
        default=None,
        choices=["initial_trading_sleeve", "live_trading_cash", "initial_fund_nav"],
        help="Base used to convert normalized investor exposure into DKK notional.",
    )
    parser.add_argument("--max_position_size", "--max-position-size", dest="max_position_size", type=float, default=None)
    parser.add_argument("--capital_allocation_fraction", "--capital-allocation-fraction", dest="capital_allocation_fraction", type=float, default=None)
    parser.add_argument("--mtm_loss_exit_threshold_pct", "--mtm-loss-exit-threshold-pct", dest="mtm_loss_exit_threshold_pct", type=float, default=None)
    parser.add_argument(
        "--risk_controller_rule_based",
        "--risk-controller-rule-based",
        dest="risk_controller_rule_based",
        action="store_true",
        default=False,
        help="Evaluate with the deterministic rule-based risk controller when the saved protocol declares it.",
    )
    parser.add_argument(
        "--meta_controller_rule_based",
        "--meta-controller-rule-based",
        dest="meta_controller_rule_based",
        action="store_true",
        default=False,
        help="Evaluate with the deterministic rule-based meta allocator when the saved protocol declares it.",
    )
    parser.add_argument(
        "--enable_forecast_utilization",
        action="store_true",
        default=False,
        help=(
            "Single switch for ANN forecast-cache utilization during evaluation. "
            "Usually inherited from training_config.json; exposed here so "
            "per-variant eval CLIs stay symmetric with training."
        ),
    )
    parser.add_argument(
        "--eval_forecast_overlay",
        "--eval-forecast-overlay",
        dest="eval_forecast_overlay",
        action="store_true",
        default=False,
        help=(
            "Evaluate an existing trained policy with the forecast-prior action layer enabled only at inference. "
            "The runtime contract exception is allowed only when the policy-matching contract still matches training."
        ),
    )
    parser.add_argument(
        "--write_tier_report",
        action="store_true",
        default=False,
        help="Also write tier_result CSV/Markdown reports. By default evaluation writes JSON only.",
    )
    add_forecast_prior_override_args(parser)

    # Analysis options
    parser.add_argument("--analyze", action="store_true",
                       help="Perform comprehensive portfolio analysis")
    parser.add_argument("--plot", action="store_true",
                       help="Generate performance plots (requires --analyze)")
    parser.add_argument("--save_plots", action="store_true",
                       help="Save plots to files instead of displaying (requires --plot)")

    args = parser.parse_args()
    seed_value = int(getattr(args, "seed", 42))
    os.environ.setdefault("PYTHONHASHSEED", str(seed_value))
    random.seed(seed_value)
    np.random.seed(seed_value)
    try:
        import torch
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
    except Exception:
        pass

    # Enable GPU memory growth before any TF graph operations.
    configure_tf_memory()

    print_progress("Parsing arguments and validating paths...")

    # Auto-detect trained agents for agents mode if not specified
    if args.mode == "agents" and args.trained_agents is None and args.enhanced_models is None:
        # Try common locations
        candidate_dirs = [
            "normal/final_models",
            "enhanced/final_models",  # New location for enhanced models
            "training_agent_results/final_models",
            "final_models",
            "saved_agents"
        ]

        for candidate in candidate_dirs:
            if os.path.exists(candidate):
                required_files = [
                    "investor_0_policy.zip", "battery_operator_0_policy.zip",
                ]
                if all(os.path.exists(os.path.join(candidate, f)) for f in required_files):
                    # Check if this is an enhanced model directory
                    is_enhanced = False

                    # Method 1: Check directory name
                    if "enhanced" in candidate.lower():
                        is_enhanced = True

                    # Method 2: Check training config for enhanced features
                    config_file = os.path.join(candidate, "training_config.json")
                    if os.path.exists(config_file):
                        try:
                            with open(config_file, 'r') as f:
                                config = json.load(f)
                            enhanced_features = config.get('enhanced_features', {})
                            if (
                                enhanced_features.get('tier1_enabled')
                                or enhanced_features.get('tier1_reward_shaping_enabled')
                            ):
                                is_enhanced = True
                        except Exception as e:
                            print(f"ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Could not parse training config for auto-detection ({config_file}): {e}")

                    if is_enhanced:
                        args.enhanced_models = candidate
                        print(f"ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯ Auto-detected enhanced models: {candidate}")
                    else:
                        args.trained_agents = candidate
                        print(f"ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯ Auto-detected trained agents: {candidate}")
                    break

        if args.trained_agents is None and args.enhanced_models is None:
            print("ÃƒÂ¢Ã‚ÂÃ…â€™ --trained_agents or --enhanced_models is required when using 'agents' mode")
            print("   Searched locations: normal/final_models, enhanced/final_models, training_agent_results/final_models")
            sys.exit(1)

    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â¥ UNIFIED EVALUATION - MODE: {args.mode.upper()}")
    print("=" * 60)

    # Load evaluation data
    print_progress(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Loading evaluation data from: {args.eval_data}")
    try:
        eval_data = load_energy_data(args.eval_data)
        print_progress(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Loaded evaluation data: {eval_data.shape}")
        if "timestamp" in eval_data.columns and eval_data["timestamp"].notna().any():
            ts = eval_data["timestamp"].dropna()
            print_progress(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Â¦ Date range: {ts.iloc[0]} ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ {ts.iloc[-1]}")
    except Exception as e:
        print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Error loading evaluation data: {e}")
        sys.exit(1)

    # Tier suite mode evaluates one or more tier variants on unseen data using final_models/
    if args.mode == "tiers":
        try:
            results = run_tier_suite_evaluation(eval_data, args)
            # Optional consolidated report (CSV + Markdown); JSON is always written.
            if bool(getattr(args, "write_tier_report", False)):
                csv_path, md_path = write_tier_report(results.get("tiers", {}), args.output_dir)
                results["tier_report_csv"] = csv_path
                results["tier_report_md"] = md_path
                results["tier_report_scope"] = "single_variant"

            analysis = None
            if args.analyze:
                analysis = analyze_comprehensive_results({"configurations": results.get("tiers", {})})

            save_results(results, args.output_dir, mode="tiers", analysis=analysis)
            if not bool(getattr(args, "write_tier_report", False)):
                return
            print_progress(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Å¾ Tier report written: {md_path}")
            print_progress(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Å¾ Tier report CSV written: {csv_path}")
            return
        except Exception as e:
            print_progress(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Tier suite evaluation failed: {e}")
            raise

    # Create evaluation environment for other modes. Forecast utilization is
    # cache-prior execution logic only.
    eval_cfg = _build_eval_config(args)
    eval_forecast_cache_dir = args.forecast_cache_dir
    if bool(getattr(eval_cfg, "enable_forecast_utilization", False)):
        forecast_ctx = _resolve_reserved_eval_forecast_context(args, output_dir=args.output_dir, config=eval_cfg)
        eval_forecast_cache_dir = forecast_ctx["forecast_cache_dir"]

    eval_env = create_evaluation_environment(
        eval_data,
        output_dir=args.output_dir,
        investment_freq=int(getattr(eval_cfg, "investment_freq", args.investment_freq) or args.investment_freq),
        config=eval_cfg,
        forecast_cache_dir=eval_forecast_cache_dir,
        fail_fast=True,
    )

    if eval_env is None:
        print("ÃƒÂ¢Ã‚ÂÃ…â€™ Failed to create evaluation environment")
        sys.exit(1)

    # Run evaluation based on mode
    results = None

    if args.mode == "checkpoint":
        # Checkpoint mode: find latest checkpoint and load SB3 models
        latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint is None:
            print("ÃƒÂ¢Ã‚ÂÃ…â€™ No checkpoints found")
            sys.exit(1)

        checkpoint_models = load_checkpoint_models(latest_checkpoint)
        if not checkpoint_models or not any(m is not None for m in checkpoint_models.values()):
            print("ÃƒÂ¢Ã‚ÂÃ…â€™ No checkpoint models loaded successfully")
            sys.exit(1)

        results = run_checkpoint_evaluation(
            checkpoint_models,
            eval_env,
            eval_data,
            args.eval_steps or 8000
        )

        if results:
            results['checkpoint_path'] = latest_checkpoint

    elif args.mode == "agents":
        # Agents mode: load MultiESGAgent system (standard or enhanced)

        # Determine which model type to load
        if args.enhanced_models:
            if not os.path.exists(args.enhanced_models):
                print(f"Enhanced models directory not found: {args.enhanced_models}")
                sys.exit(1)

            print("Loading enhanced models with the Tier1 evaluation runtime...")

            agent_system = load_enhanced_agent_system(args.enhanced_models, eval_env)
            model_path = args.enhanced_models
        else:
            if not os.path.exists(args.trained_agents):
                print(f"ÃƒÂ¢Ã‚ÂÃ…â€™ Trained agents directory not found: {args.trained_agents}")
                sys.exit(1)

            agent_system = load_agent_system(args.trained_agents, eval_env)
            model_path = args.trained_agents

        if agent_system is None:
            print("ÃƒÂ¢Ã‚ÂÃ…â€™ Failed to load agent system")
            sys.exit(1)

        results = run_agent_evaluation(
            agent_system,
            eval_env,
            eval_data,
            args.eval_steps
        )

        if results:
            results['trained_agents_path'] = model_path
            results['model_type'] = 'enhanced' if args.enhanced_models else 'standard'

    elif args.mode == "baselines":
        # Baselines mode: run traditional baseline methods
        print("ÃƒÂ°Ã…Â¸Ã‚ÂÃ¢â‚¬ÂºÃƒÂ¯Ã‚Â¸Ã‚Â Running traditional baseline evaluation...")

        baseline_results = run_traditional_baselines(
            args.eval_data,
            args.eval_steps or 10000,  # Use same default as agents
            args.output_dir,
            seed=getattr(args, "seed", 42),
        )

        if baseline_results and 'error' not in baseline_results:
            results = {
                'evaluation_type': 'traditional_baselines',
                'baselines': baseline_results,
                'eval_data_path': args.eval_data,
                'eval_steps': args.eval_steps or 10000,
                'baseline_scope': 'baseline_1_2_3_current_hybrid_fund',
                'artifact_policy': 'single_aggregate_json'
            }
        else:
            print("ÃƒÂ¢Ã‚ÂÃ…â€™ Traditional baseline evaluation failed")
            sys.exit(1)

    elif args.mode == "compare":
        # Compare mode: run both AI and traditional baselines
        print("ÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ¢â‚¬Å¾ Running comprehensive comparison...")

        # Initialize ai_results
        ai_results = None

        # Run AI evaluation first (enhanced or standard models)
        if args.enhanced_models:
            agent_system = load_enhanced_agent_system(args.enhanced_models, eval_env)
            if agent_system:
                ai_results = run_agent_evaluation(agent_system, eval_env, eval_data, args.eval_steps)
                if ai_results:
                    ai_results['model_type'] = 'enhanced'
                    ai_results['trained_agents_path'] = args.enhanced_models
        elif args.trained_agents:
            # Use standard models
            agent_system = load_agent_system(args.trained_agents, eval_env)
            if agent_system:
                ai_results = run_agent_evaluation(agent_system, eval_env, eval_data, args.eval_steps)
                if ai_results:
                    ai_results['model_type'] = 'standard'
                    ai_results['trained_agents_path'] = args.trained_agents
        else:
            # Auto-detect or use latest checkpoint
            latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
            if latest_checkpoint and "final_models" in latest_checkpoint:
                args.trained_agents = latest_checkpoint
                agent_system = load_agent_system(args.trained_agents, eval_env)
                if agent_system:
                    ai_results = run_agent_evaluation(agent_system, eval_env, eval_data, args.eval_steps)
                    if ai_results:
                        ai_results['model_type'] = 'standard'
                        ai_results['trained_agents_path'] = args.trained_agents
            elif latest_checkpoint:
                checkpoint_models = load_checkpoint_models(latest_checkpoint)
                if checkpoint_models:
                    ai_results = run_checkpoint_evaluation(checkpoint_models, eval_env, eval_data, args.eval_steps or 8000)
            else:
                print("ÃƒÂ¢Ã‚ÂÃ…â€™ No AI models found for comparison")

        # Run traditional baselines with same steps as AI agents
        baseline_results = run_traditional_baselines(
            args.eval_data,
            args.eval_steps or 10000,
            args.output_dir,
            seed=getattr(args, "seed", 42),
        )

        # Combine results
        if ai_results and baseline_results and 'error' not in baseline_results:
            results = {
                'evaluation_type': 'comprehensive_comparison',
                'ai_results': ai_results,
                'baseline_results': baseline_results,
                'eval_data_path': args.eval_data,
                'eval_steps': args.eval_steps or 8000
            }
        elif ai_results and 'error' in baseline_results:
            print("ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â Traditional baselines failed, showing AI results only")
            results = ai_results
            results['baseline_error'] = baseline_results.get('error', 'Unknown error')
        elif baseline_results and 'error' not in baseline_results and not ai_results:
            print("ÃƒÂ¢Ã…Â¡Ã‚Â ÃƒÂ¯Ã‚Â¸Ã‚Â AI evaluation failed, showing baseline results only")
            results = {
                'evaluation_type': 'baselines_only',
                'baseline_results': baseline_results,
                'ai_error': 'AI evaluation failed'
            }
        else:
            print("ÃƒÂ¢Ã‚ÂÃ…â€™ Both AI and baseline evaluations failed")
            sys.exit(1)

    elif args.mode == "comprehensive":
        # Comprehensive mode: evaluate all configurations
        print("ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯ Running comprehensive evaluation of all configurations...")

        results = run_comprehensive_evaluation(eval_data, args)

        if not results or not results.get('configurations'):
            print("ÃƒÂ¢Ã‚ÂÃ…â€™ Comprehensive evaluation failed")
            sys.exit(1)

    # Save and display results
    if results:
        # Add common metadata
        results['eval_data_path'] = args.eval_data
        results['forecast_utilization_enabled'] = bool(getattr(eval_cfg, "enable_forecast_utilization", False))
        results['eval_steps_requested'] = args.eval_steps

        # Perform analysis if requested
        analysis_results = None
        if args.analyze:
            print("\nÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  PERFORMING COMPREHENSIVE ANALYSIS...")

            # Handle different result types
            if results.get('evaluation_type') == 'comprehensive':
                # For comprehensive results, use specialized analysis
                analysis_results = analyze_comprehensive_results(results)
            elif results.get('evaluation_type') == 'comprehensive_comparison':
                # For comparison results, analyze the AI results part
                ai_results = results.get('ai_results', {})
                if ai_results:
                    analyzer = PortfolioAnalyzer(ai_results)
                    analysis_results = analyzer.analyze_performance(
                        make_plots=args.plot,
                        save_plots=args.save_plots,
                        output_dir=args.output_dir
                    )
                    # Add baseline comparison info to analysis
                    if 'baseline_results' in results:
                        analysis_results['baseline_comparison'] = results['baseline_results']
                else:
                    # Fallback for comparison without AI results
                    analysis_results = {
                        'basic_metrics': {'total_return': 0.0, 'sharpe_ratio': 0.0},
                        'summary': {'overall_assessment': 'MODERATE', 'performance_grade': 'C'}
                    }
            else:
                # For regular results (agents, baselines, checkpoints)
                analyzer = PortfolioAnalyzer(results)
                analysis_results = analyzer.analyze_performance(
                    make_plots=args.plot,
                    save_plots=args.save_plots,
                    output_dir=args.output_dir
                )

        # Save results and analysis
        results_file, analysis_file = save_results(results, args.output_dir, args.mode, analysis_results)

        # Display summary
        print("\nÃƒÂ°Ã…Â¸Ã…Â½Ã¢â‚¬Â° EVALUATION SUCCESS!")

        # Handle different result types for display
        if results.get('evaluation_type') == 'comprehensive':
            # For comprehensive results, show summary from analysis
            if analysis_results and 'summary' in analysis_results:
                best_performer = analysis_results['summary'].get('best_performer', {})
                if best_performer:
                    print(f"ÃƒÂ°Ã…Â¸Ã‚Â¥Ã¢â‚¬Â¡ BEST PERFORMER: {best_performer['name']}")
                    metrics = best_performer['metrics']
                    print(f"   ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‹â€  Return: {metrics['total_return']:.2f}%")
                    print(f"   ÃƒÂ¢Ã…Â¡Ã‚Â¡ Sharpe: {metrics['sharpe_ratio']:.3f}")
                    print(f"   ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Â° Drawdown: {metrics['max_drawdown']:.2f}%")

                if 'feature_impact' in analysis_results:
                    impact = analysis_results['feature_impact']
                    print("\nÃƒÂ°Ã…Â¸Ã¢â‚¬ÂÃ‚Â¬ ENHANCED FEATURES IMPACT:")
                    print(f"   ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‹â€  Return Improvement: {impact['return_improvement_pct']:+.2f}%")
                    print(f"   ÃƒÂ¢Ã…Â¡Ã‚Â¡ Sharpe Improvement: {impact['sharpe_improvement']:+.3f}")
            else:
                print("ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Comprehensive evaluation completed - see detailed results in JSON file")

        elif results.get('evaluation_type') == 'comprehensive_comparison':
            # For comparison results, show AI results
            ai_results = results.get('ai_results', {})
            baseline_results = results.get('baseline_results', {})

            if ai_results:
                print("ÃƒÂ°Ã…Â¸Ã‚Â¤Ã¢â‚¬â€œ AI AGENTS PERFORMANCE:")
                print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‹â€  Final Return: {ai_results.get('total_return', 0):+.2%}")
                print(f"ÃƒÂ¢Ã…Â¡Ã‚Â¡ Sharpe Ratio: {ai_results.get('sharpe_ratio', 0):.3f}")
                print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Total Rewards: {ai_results.get('total_rewards', 0):.2f}")

                if 'initial_portfolio_value' in ai_results and 'final_portfolio_value' in ai_results:
                    initial = ai_results['initial_portfolio_value']
                    final = ai_results['final_portfolio_value']
                    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â° Portfolio: ${initial/1e6:.1f}M ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ ${final/1e6:.1f}M (${(final-initial)/1e6:+.1f}M)")
                    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Volatility: {ai_results.get('volatility', 0)*100:.2f}%")
                    print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Â° Max Drawdown: {ai_results.get('max_drawdown', 0)*100:.2f}%")

            if baseline_results:
                print("\nÃƒÂ°Ã…Â¸Ã‚ÂÃ¢â‚¬ÂºÃƒÂ¯Ã‚Â¸Ã‚Â BASELINE COMPARISON:")
                for baseline_name, baseline_data in baseline_results.items():
                    if isinstance(baseline_data, dict) and 'total_return' in baseline_data:
                        method = baseline_data.get('method', baseline_name)
                        return_pct = baseline_data.get('total_return', 0) * 100
                        sharpe = baseline_data.get('sharpe_ratio', 0)
                        print(f"   {method}: {return_pct:+.2f}% return, {sharpe:.2f} Sharpe")
        else:
            # For regular results
            print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‹â€  Final Return: {results.get('total_return', 0):+.2%}")
            print(f"ÃƒÂ¢Ã…Â¡Ã‚Â¡ Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
            print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Total Rewards: {results.get('total_rewards', 0):.2f}")
            print(f"ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯ Average Risk: {results.get('average_risk', 0):.3f}")

            # Portfolio performance details
            if 'initial_portfolio_value' in results and 'final_portfolio_value' in results:
                initial = results['initial_portfolio_value']
                final = results['final_portfolio_value']
                print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â° Portfolio: ${initial/1e6:.1f}M ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ ${final/1e6:.1f}M (${(final-initial)/1e6:+.1f}M)")
                print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Volatility: {results.get('volatility', 0)*100:.2f}%")
                print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Â° Max Drawdown: {results.get('max_drawdown', 0)*100:.2f}%")

        inference_rate = _extract_action_inference_success_rate(results)
        if inference_rate is not None:
            print(f"ÃƒÂ°Ã…Â¸Ã…Â½Ã‚Â¯ Action Inference Success: {inference_rate:.1%}")
        if 'models_loaded' in results:
            print(f"ÃƒÂ°Ã…Â¸Ã‚Â¤Ã¢â‚¬â€œ Models Loaded: {results['models_loaded']}")

        # Display analysis summary if available
        if analysis_results and 'summary' in analysis_results:
            summary = analysis_results['summary']
            print("\nÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  ANALYSIS SUMMARY:")
            print(f"ÃƒÂ°Ã…Â¸Ã‚ÂÃ¢â‚¬Â  Overall Assessment: {summary.get('overall_assessment', 'N/A')}")
            print(f"ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‚Â Performance Grade: {summary.get('performance_grade', 'N/A')}")

            if 'recommendations' in summary:
                print("ÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â¡ Recommendations:")
                for i, rec in enumerate(summary['recommendations'], 1):
                    print(f"   {i}. {rec}")

        print("\nÃƒÂ°Ã…Â¸Ã¢â‚¬â„¢Ã‚Â¾ Files saved:")
        print(f"   ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã¢â‚¬Å¾ Results: {results_file}")
        if analysis_file:
            print(f"   ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…Â  Analysis: {analysis_file}")

    else:
        print("ÃƒÂ¢Ã‚ÂÃ…â€™ Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

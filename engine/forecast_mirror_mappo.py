"""Pure counterfactual utilities for Counterfactual Forecast-Mirror MAPPO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MirrorCounterfactualOutcome:
    trust_gradient: float
    expected_score: float
    score_std: float
    prior_log_score_std: float


def _positive_finite(name: str, value: float) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return parsed


def beta_grid_weights(alpha: float, beta: float, unit_grid: np.ndarray) -> np.ndarray:
    """Return stable normalized Beta-density weights on an interior grid."""
    a = _positive_finite("alpha", alpha)
    b = _positive_finite("beta", beta)
    raw_grid = np.asarray(unit_grid, dtype=np.float64)
    if raw_grid.ndim != 1 or raw_grid.size == 0 or not np.isfinite(raw_grid).all():
        raise ValueError("unit_grid must be a non-empty finite one-dimensional array")
    x = np.clip(raw_grid, 1e-8, 1.0 - 1e-8)
    log_weights = (a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x)
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Beta grid normalization produced invalid weights")
    return weights / total


def counterfactual_trust_gradient(
    *,
    unit_grid: np.ndarray,
    scores: np.ndarray,
    final_alpha: float,
    final_beta: float,
    prior_alpha: float,
    prior_beta: float,
    evidence: float,
) -> MirrorCounterfactualOutcome:
    """Differentiate expected local utility with respect to forecast trust.

    For ``pi_exec proportional to pi_base * pi_prior ** (evidence * trust)``,
    the exact score-function derivative is
    ``evidence * Cov_pi_exec(score, log(pi_prior))``.  Constant terms in the
    prior log density cancel from the covariance.
    """
    x = np.clip(np.asarray(unit_grid, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    score = np.asarray(scores, dtype=np.float64)
    if x.shape != score.shape or x.ndim != 1:
        raise ValueError("unit_grid and scores must be finite one-dimensional peers")
    if x.size == 0:
        raise ValueError("counterfactual grid must not be empty")
    if not np.isfinite(x).all() or not np.isfinite(score).all():
        raise ValueError("counterfactual grid and scores must be finite")
    prior_a = _positive_finite("prior_alpha", prior_alpha)
    prior_b = _positive_finite("prior_beta", prior_beta)
    evidence_f = float(evidence)
    if not np.isfinite(evidence_f):
        raise ValueError(f"evidence must be finite, got {evidence!r}")
    weights = beta_grid_weights(final_alpha, final_beta, x)
    log_prior = (
        (prior_a - 1.0) * np.log(x)
        + (prior_b - 1.0) * np.log1p(-x)
    )
    score_mean = float(np.sum(weights * score))
    log_prior_mean = float(np.sum(weights * log_prior))
    covariance = float(
        np.sum(weights * (score - score_mean) * (log_prior - log_prior_mean))
    )
    return MirrorCounterfactualOutcome(
        trust_gradient=float(np.clip(evidence_f, 0.0, 1.0)) * covariance,
        expected_score=score_mean,
        score_std=float(np.sqrt(np.sum(weights * (score - score_mean) ** 2))),
        prior_log_score_std=float(
            np.sqrt(np.sum(weights * (log_prior - log_prior_mean) ** 2))
        ),
    )


__all__ = [
    "MirrorCounterfactualOutcome",
    "beta_grid_weights",
    "counterfactual_trust_gradient",
]

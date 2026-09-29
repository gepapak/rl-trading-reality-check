"""Load and causally calibrate the investor's cached ANN forecast signal.

The processor is shared by deterministic, observation-only, feasible-action,
and forecast-mirror variants. It supplies forecast direction, confidence, and
strictly matured online evidence. Exposure construction remains the
responsibility of the selected action-layer mechanism in ``environment.py``.
"""

from __future__ import annotations

import glob
import logging
import os
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


FORECAST_PRIOR_FEATURE_COLS = [
    "price_short_expert_ann_pred_return",
    "price_short_expert_ann_direction_prob",
    "price_short_expert_ann_direction_margin",
    "price_short_expert_ann_uncertainty",
    "price_short_expert_ann_quality",
]

FORECAST_PRIOR_OPTIONAL_COLS = [
    "price_short_expert_ann",
    "price_forecast_short",
    "price_short_expert_ann_latent_norm",
    "price_short_expert_ann_latent_0",
    "price_short_expert_ann_latent_1",
    "price_short_expert_ann_latent_2",
    "price_short_expert_ann_latent_3",
]

FORECAST_PRIOR_ALL_COLS = FORECAST_PRIOR_FEATURE_COLS + FORECAST_PRIOR_OPTIONAL_COLS
PRED_RETURN_IDX = 0
DIRECTION_PROB_IDX = 1
DIRECTION_MARGIN_IDX = 2
UNCERTAINTY_IDX = 3
QUALITY_IDX = 4
PRED_PRICE_IDXS = (5, 6)


def load_forecast_prior_features(cache_dir: str, episode_num: int) -> Optional[np.ndarray]:
    """
    Load ANN forecast-cache columns for one episode.

    Returns an array with columns matching ``FORECAST_PRIOR_ALL_COLS``. Missing
    optional latent columns are zero-filled; missing required columns fail closed
    and return ``None``.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("[FORECAST_PRIOR] pandas is required to read forecast cache")
        return None

    ep_dir = os.path.join(str(cache_dir), f"episode_{int(episode_num)}")
    if not os.path.isdir(ep_dir):
        ep_dir = str(cache_dir)

    csv_paths = sorted(glob.glob(os.path.join(ep_dir, "precomputed_forecasts_*.csv")))
    csv_paths = [p for p in csv_paths if "_metadata" not in os.path.basename(p)]
    if not csv_paths:
        return None
    if len(csv_paths) != 1:
        logger.error(
            "[FORECAST_PRIOR] Expected one forecast CSV for episode %s in '%s'; found %d: %s",
            int(episode_num),
            ep_dir,
            len(csv_paths),
            [os.path.basename(path) for path in csv_paths],
        )
        return None

    csv_path = csv_paths[0]
    try:
        header = pd.read_csv(csv_path, nrows=0)
        available = set(str(c) for c in header.columns)
        missing_required = [c for c in FORECAST_PRIOR_FEATURE_COLS if c not in available]
        if missing_required:
            logger.error(
                "[FORECAST_PRIOR] Cache '%s' missing required columns: %s",
                csv_path,
                missing_required,
            )
            return None

        read_cols = [c for c in FORECAST_PRIOR_ALL_COLS if c in available]
        df = pd.read_csv(csv_path, usecols=read_cols)
        for col in FORECAST_PRIOR_ALL_COLS:
            if col not in df.columns:
                df[col] = np.nan if col in {"price_short_expert_ann", "price_forecast_short"} else 0.0
        df = df[FORECAST_PRIOR_ALL_COLS]
    except Exception as e:
        logger.error("[FORECAST_PRIOR] Failed to read '%s': %s", csv_path, e)
        return None

    if df.empty:
        return None

    arr = df.to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        bad = ~np.isfinite(arr)
        for j in range(arr.shape[1]):
            col = FORECAST_PRIOR_ALL_COLS[j]
            if col in {"price_short_expert_ann", "price_forecast_short"}:
                default = np.nan
            elif col == "price_short_expert_ann_direction_prob":
                default = 0.5
            elif col == "price_short_expert_ann_uncertainty":
                default = 1.0
            else:
                default = 0.0
            arr[bad[:, j], j] = default
    return arr


class ConformalForecastPrior:
    """
    Stateful online evidence-calibrated exposure prior for ANN forecast-cache
    signals.

    Despite the legacy class name, this is NOT conformal prediction: it keeps
    a rolling window of absolute forecast errors (a conformal-style
    nonconformity quantile used as an edge hurdle) plus a Wald
    lower-confidence-bound on the directional hit rate. No distribution-free
    coverage guarantee is claimed or provided.

    At time ``t`` it first updates calibration from the forecast made at
    ``t - horizon_steps`` using the now-realized raw DKK price.  In the paper
    protocol this calibration update is restricted to the investor decision
    grid, avoiding artificial sample inflation from forward-filled ten-minute
    rows.  It then emits a conservative exposure prior for the current cache
    row.  The prior is bounded to [-max_abs_exposure, max_abs_exposure] and can
    be weighted by the live directional confidence edge.
    """

    __slots__ = (
        "raw",
        "decision_freq",
        "horizon",
        "window_size",
        "min_samples",
        "hit_lcb_z",
        "residual_quantile",
        "default_residual",
        "edge_gain",
        "error_hurdle",
        "skill_power",
        "directional_floor",
        "max_abs_exposure",
        "use_direction_confidence",
        "direction_confidence_power",
        "calibrate_on_decision_grid",
        "denom_floor",
        "target_mode",
        "entry_price_mode",
        "settlement_reference_price",
        "payoff_denominator_mode",
        "steps_per_day",
        "vol_half_life_steps",
        "vol_target",
        "_hits",
        "_abs_errors",
        "_calibration_events_total",
        "_prices",
        "_calibrated",
        "_ema_logret2",
        "_last_price",
        "_last_vol_price",
        "_last_t",
        "_last_signal",
    )

    def __init__(
        self,
        raw_features: Optional[np.ndarray],
        *,
        decision_freq: int = 6,
        horizon_steps: int = 6,
        window_size: int = 500,
        min_samples: int = 50,
        hit_lcb_z: float = 1.64,
        residual_quantile: float = 0.70,
        default_residual: float = 0.10,
        edge_gain: float = 3.0,
        error_hurdle: float = 0.50,
        skill_power: float = 1.0,
        directional_floor: float = 0.50,
        max_abs_exposure: float = 0.60,
        use_direction_confidence: bool = True,
        direction_confidence_power: float = 1.0,
        calibrate_on_decision_grid: bool = True,
        denom_floor: float = 50.0,
        target_mode: str = "price_return",
        entry_price_mode: str = "current_price",
        settlement_reference_price: float = 500.0,
        payoff_denominator_mode: str = "reference_price",
        steps_per_day: int = 144,
        vol_half_life_steps: int = 288,
        vol_target: float = 0.15,
    ) -> None:
        self.raw = raw_features
        self.decision_freq = max(int(decision_freq), 1)
        self.horizon = max(int(horizon_steps), 1)
        self.window_size = max(int(window_size), 1)
        self.min_samples = max(int(min_samples), 1)
        self.hit_lcb_z = float(max(hit_lcb_z, 0.0))
        self.residual_quantile = float(np.clip(residual_quantile, 0.50, 0.99))
        self.default_residual = float(max(default_residual, 1e-6))
        self.edge_gain = float(max(edge_gain, 0.0))
        self.error_hurdle = float(max(error_hurdle, 0.0))
        self.skill_power = float(max(skill_power, 1e-6))
        self.directional_floor = float(np.clip(directional_floor, 0.0, 0.95))
        self.max_abs_exposure = float(np.clip(max_abs_exposure, 0.0, 1.0))
        self.use_direction_confidence = bool(use_direction_confidence)
        self.direction_confidence_power = float(max(direction_confidence_power, 0.0))
        self.calibrate_on_decision_grid = bool(calibrate_on_decision_grid)
        self.denom_floor = float(max(denom_floor, 1e-6))
        self.target_mode = str(target_mode or "price_return").strip().lower().replace("-", "_")
        self.entry_price_mode = str(entry_price_mode or "current_price").strip().lower().replace("-", "_")
        self.settlement_reference_price = float(max(settlement_reference_price, 1e-6))
        self.payoff_denominator_mode = (
            str(payoff_denominator_mode or "reference_price").strip().lower().replace("-", "_")
        )
        self.steps_per_day = max(int(steps_per_day), 1)
        self.vol_half_life_steps = max(int(vol_half_life_steps), 1)
        self.vol_target = float(max(vol_target, 1e-6))

        self._hits = deque(maxlen=self.window_size)
        self._abs_errors = deque(maxlen=self.window_size)
        self._calibration_events_total = 0
        self._prices: Dict[int, float] = {}
        self._calibrated = set()
        self._ema_logret2 = (1e-3) ** 2
        self._last_price: Optional[float] = None
        self._last_vol_price: Optional[float] = None
        self._last_t: Optional[int] = None
        self._last_signal = self._empty_signal(reason="cold")

    @staticmethod
    def _sgn(x: float) -> float:
        if x > 0.0:
            return 1.0
        if x < 0.0:
            return -1.0
        return 0.0

    def _empty_signal(self, *, reason: str, phase: float = 0.0) -> Dict[str, Any]:
        return {
            "step": -1,
            "prior_exposure": 0.0,
            "forecast_sign": 0.0,
            "pred_return": 0.0,
            "direction_margin": 0.0,
            "direction_confidence": 0.5,
            "direction_confidence_edge": 0.0,
            "confidence_weight": 1.0 if not self.use_direction_confidence else 0.0,
            "hit_lcb": 0.5,
            "hit_rate": 0.5,
            "gate_type": "online_directional_hit_lcb",
            "skill": 0.0,
            "residual_q": self.default_residual,
            "edge_excess": 0.0,
            "magnitude": 0.0,
            "quality": 0.0,
            "uncertainty": 1.0,
            "sigma_h": 0.0,
            "calibration_count": len(self._abs_errors),
            "calibration_total": self._calibration_events_total,
            "directional_hit_count": len(self._hits),
            "phase": float(phase),
            "return_target_mode": self.target_mode,
            "target_entry_price": 0.0,
            "target_predicted_price": 0.0,
            "decision_grid_calibration": self.calibrate_on_decision_grid,
            "calibration_event": False,
            "active": False,
            "reason": str(reason),
        }

    def _update_price_state(self, t: int, price: float) -> None:
        if not np.isfinite(price):
            return
        self._prices[int(t)] = float(price)
        if price > 0.0:
            self._last_price = float(price)

    def _update_volatility_state(self, value: float) -> None:
        """Update risk scaling from the realized payoff/settlement index."""
        if not np.isfinite(value):
            return
        if self._last_vol_price is not None and np.isfinite(self._last_vol_price):
            # Settlement prices may be zero or negative. A floor-normalized
            # arithmetic change remains defined across sign changes, unlike a
            # log return, and is only used by the legacy exposure scaler.
            lr = float(
                (float(value) - self._last_vol_price)
                / max(abs(self._last_vol_price), self.denom_floor)
            )
            alpha = 1.0 - 0.5 ** (1.0 / float(self.vol_half_life_steps))
            self._ema_logret2 = float(
                (1.0 - alpha) * self._ema_logret2 + alpha * (lr * lr)
            )
        self._last_vol_price = float(value)

    def _entry_price_for_step(self, step: int, current_price: float) -> float:
        current = float(current_price) if np.isfinite(current_price) else 0.0
        if self.target_mode != "horizon_settlement":
            return current
        if self.entry_price_mode == "current_price":
            return current

        step = int(step)
        if self.entry_price_mode == "same_hour_prev_day":
            value = self._prices.get(step - self.steps_per_day)
            if value is not None and np.isfinite(value):
                return float(value)

        values = []
        cursor = step - self.steps_per_day
        while cursor >= 0 and len(values) < 30:
            value = self._prices.get(cursor)
            if value is not None and np.isfinite(value):
                values.append(float(value))
            cursor -= self.steps_per_day
        if values:
            return float(np.median(np.asarray(values, dtype=np.float64)))
        return current

    def _payoff_denominator(self, entry_price: float) -> float:
        ref = float(max(self.settlement_reference_price, 1e-6))
        if self.payoff_denominator_mode == "entry_price_floor":
            entry_abs = abs(float(entry_price)) if np.isfinite(entry_price) else 0.0
            return float(max(entry_abs, ref))
        return ref

    def _uses_mwh_volume_payoff(self) -> bool:
        return self.payoff_denominator_mode in {"mwh_volume", "mwh"}

    def _predicted_price_for_step(self, step: int, current_price: float) -> float:
        if self.raw is not None and 0 <= int(step) < self.raw.shape[0]:
            for idx in PRED_PRICE_IDXS:
                if idx < self.raw.shape[1]:
                    value = float(self.raw[int(step), idx])
                    if np.isfinite(value):
                        return value
            pred_return = float(self.raw[int(step), PRED_RETURN_IDX])
            if np.isfinite(pred_return) and np.isfinite(current_price):
                denom = max(abs(float(current_price)), self.denom_floor)
                return float(current_price) + pred_return * denom
        return float(current_price) if np.isfinite(current_price) else 0.0

    def _forecast_return_for_step(self, step: int, current_price: float) -> tuple[float, float, float]:
        if self.raw is None or step < 0 or step >= self.raw.shape[0]:
            return 0.0, 0.0, 0.0
        if self.target_mode == "horizon_settlement":
            entry_price = self._entry_price_for_step(step, current_price)
            predicted_price = self._predicted_price_for_step(step, current_price)
            if self._uses_mwh_volume_payoff():
                forecast_return = float(predicted_price - entry_price)
            else:
                denom = self._payoff_denominator(entry_price)
                forecast_return = float((predicted_price - entry_price) / denom)
            return forecast_return, float(entry_price), float(predicted_price)
        pred_return = float(self.raw[int(step), PRED_RETURN_IDX])
        return pred_return if np.isfinite(pred_return) else 0.0, float(current_price), 0.0

    def _realized_return_for_origin(
        self,
        origin_step: int,
        origin_price: float,
        settlement_price: float,
    ) -> float:
        if self.target_mode == "horizon_settlement":
            entry_price = self._entry_price_for_step(origin_step, origin_price)
            if self._uses_mwh_volume_payoff():
                return float(float(settlement_price) - entry_price)
            denom = self._payoff_denominator(entry_price)
            return float((float(settlement_price) - entry_price) / denom)
        return float(
            (float(settlement_price) - float(origin_price))
            / max(abs(float(origin_price)), self.denom_floor)
        )

    def _update_calibration(self, t: int, settlement_price: float) -> bool:
        if self.calibrate_on_decision_grid and (int(t) % self.decision_freq) != 0:
            return False
        k = int(t) - self.horizon
        if k < 0 or k in self._calibrated:
            return False
        if self.calibrate_on_decision_grid and (int(k) % self.decision_freq) != 0:
            return False
        if self.raw is None or k >= self.raw.shape[0]:
            return False
        past_price = self._prices.get(k)
        if past_price is None or not np.isfinite(past_price):
            return False
        if not np.isfinite(settlement_price):
            return False

        self._calibrated.add(k)
        realized = self._realized_return_for_origin(k, past_price, float(settlement_price))
        pred_return, _, _ = self._forecast_return_for_step(k, past_price)
        margin = float(np.clip(self.raw[k, DIRECTION_MARGIN_IDX], -1.0, 1.0))
        if self.target_mode == "horizon_settlement":
            forecast_sign = self._sgn(pred_return)
        else:
            forecast_sign = self._sgn(margin) or self._sgn(pred_return)
        realized_sign = self._sgn(realized)
        if forecast_sign != 0.0 and realized_sign != 0.0:
            self._hits.append(1.0 if forecast_sign == realized_sign else 0.0)
        if np.isfinite(realized) and np.isfinite(pred_return):
            self._abs_errors.append(abs(realized - pred_return))
            self._calibration_events_total += 1
        return True

    def update_and_compute(
        self,
        t: int,
        price: float,
        *,
        settlement_price: Optional[float] = None,
        settlement_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        t = int(t)
        phase = float(t % self.decision_freq) / float(self.decision_freq)
        if self._last_t == t:
            return dict(self._last_signal)

        price_f = float(price) if np.isfinite(price) else 0.0
        settlement_f = (
            float(settlement_price)
            if settlement_price is not None and np.isfinite(settlement_price)
            else price_f
        )
        self._update_price_state(t, price_f)
        self._update_volatility_state(settlement_f if self.target_mode == "horizon_settlement" else price_f)
        # Signal time and observation time are distinct in the simulator: the
        # action for t is selected before the settlement for t is booked.  The
        # caller therefore supplies the latest strictly observed settlement
        # step; retaining t as the default preserves compatibility for callers
        # whose event ordering makes the contemporaneous value observable.
        calibration_t = t if settlement_step is None else int(settlement_step)
        calibration_event = self._update_calibration(calibration_t, settlement_f)

        if self.raw is None or t < 0 or t >= self.raw.shape[0]:
            out = self._empty_signal(reason="no_cache", phase=phase)
            out["step"] = t
            self._last_t = t
            self._last_signal = out
            return dict(out)

        pred_return, entry_price, predicted_price = self._forecast_return_for_step(t, price_f)
        direction_prob = float(np.clip(self.raw[t, DIRECTION_PROB_IDX], 0.0, 1.0))
        margin = float(np.clip(self.raw[t, DIRECTION_MARGIN_IDX], -1.0, 1.0))
        uncertainty = float(max(0.0, self.raw[t, UNCERTAINTY_IDX]))
        quality = float(np.clip(self.raw[t, QUALITY_IDX], 0.0, 1.0))

        if self.target_mode == "horizon_settlement":
            forecast_sign = self._sgn(pred_return)
            # ANN v3 predicts P(settlement-entry > 0). Confidence is measured
            # relative to the signed point-forecast action.
            if forecast_sign > 0.0:
                direction_confidence = direction_prob
            elif forecast_sign < 0.0:
                direction_confidence = 1.0 - direction_prob
            else:
                direction_confidence = 0.5
        else:
            forecast_sign = self._sgn(margin) or self._sgn(pred_return)
            if forecast_sign > 0.0:
                direction_confidence = direction_prob
            elif forecast_sign < 0.0:
                direction_confidence = 1.0 - direction_prob
            else:
                direction_confidence = 0.5
        direction_confidence = float(np.clip(direction_confidence, 0.0, 1.0))
        direction_confidence_edge = float(max(0.0, 2.0 * direction_confidence - 1.0))
        if self.use_direction_confidence:
            confidence_weight = float(direction_confidence_edge ** self.direction_confidence_power)
        else:
            confidence_weight = 1.0

        hit_rate = 0.5
        if len(self._hits) >= self.min_samples:
            p_hat = float(np.mean(np.asarray(self._hits, dtype=np.float64)))
            hit_rate = p_hat
            se = float(np.sqrt(max(p_hat * (1.0 - p_hat), 1e-6) / max(len(self._hits), 1)))
            hit_lcb = float(np.clip(p_hat - self.hit_lcb_z * se, 0.0, 1.0))
        else:
            # Stay flat until the prior has causal evidence from this episode.
            # Direction-probability alone made the old prior active nearly
            # everywhere and allowed high exposure before error calibration.
            hit_lcb = 0.5

        if len(self._abs_errors) >= self.min_samples:
            residual_q = float(
                np.quantile(np.asarray(self._abs_errors, dtype=np.float64), self.residual_quantile)
            )
        else:
            residual_q = self.default_residual
        residual_q = float(max(residual_q, 1e-6))

        directional_edge = float(
            np.clip(
                (hit_lcb - self.directional_floor) / max(1.0 - self.directional_floor, 1e-6),
                0.0,
                1.0,
            )
        )
        skill = float(directional_edge ** self.skill_power)
        edge_excess = float(max(0.0, abs(pred_return) - self.error_hurdle * residual_q))
        magnitude = float(edge_excess / (abs(pred_return) + residual_q))
        point_edge_taper = float(0.50 + 0.50 * np.clip(magnitude, 0.0, 1.0))
        confidence_taper = float(
            (0.25 + 0.75 * confidence_weight)
            if self.use_direction_confidence
            else 1.0
        )
        sigma_step = float(max(self._ema_logret2, 1e-12) ** 0.5)
        sigma_h = float(max(sigma_step * (self.horizon ** 0.5), 1e-6))
        vol_shrink = float(min(1.0, self.vol_target / sigma_h))

        raw_prior = (
            forecast_sign
            * self.edge_gain
            * self.max_abs_exposure
            * skill
            * point_edge_taper
            * confidence_taper
            * quality
            * float(np.exp(-uncertainty))
            * vol_shrink
        )
        prior_exposure = float(np.clip(raw_prior, -self.max_abs_exposure, self.max_abs_exposure))

        active = bool(abs(prior_exposure) > 1e-9 and forecast_sign != 0.0 and skill > 0.0)
        out = {
            "step": t,
            "prior_exposure": prior_exposure,
            "forecast_sign": float(forecast_sign),
            "pred_return": pred_return,
            "direction_margin": margin,
            "direction_confidence": direction_confidence,
            "direction_confidence_edge": direction_confidence_edge,
            "confidence_weight": confidence_weight,
            "hit_lcb": hit_lcb,
            "hit_rate": hit_rate,
            "gate_type": "online_directional_hit_lcb",
            "skill": skill,
            "directional_floor": self.directional_floor,
            "directional_edge": directional_edge,
            "residual_q": residual_q,
            "edge_excess": edge_excess,
            "magnitude": magnitude,
            "point_edge_taper": point_edge_taper,
            "confidence_taper": confidence_taper,
            "quality": quality,
            "uncertainty": uncertainty,
            "sigma_h": sigma_h,
            "calibration_count": len(self._abs_errors),
            "calibration_total": self._calibration_events_total,
            "directional_hit_count": len(self._hits),
            "phase": phase,
            "return_target_mode": self.target_mode,
            "target_entry_price": float(entry_price),
            "target_predicted_price": float(predicted_price),
            "decision_grid_calibration": self.calibrate_on_decision_grid,
            "calibration_event": bool(calibration_event),
            "active": active,
            "reason": "active" if active else "zero_edge",
        }
        self._last_t = t
        self._last_signal = out
        return dict(out)


__all__ = [
    "FORECAST_PRIOR_FEATURE_COLS",
    "FORECAST_PRIOR_OPTIONAL_COLS",
    "FORECAST_PRIOR_ALL_COLS",
    "load_forecast_prior_features",
    "ConformalForecastPrior",
]

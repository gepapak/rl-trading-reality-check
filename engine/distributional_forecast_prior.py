"""Distributional tail-budgeted forecast-prior sizing for FoCAL v3.

This module converts cached settlement-aligned forecast signals into a target
normalized investor exposure.  It is intentionally action-layer logic: the
forecast defines the anchor direction, while causal realized settlement payoffs
define how much exposure the sleeve is allowed to take under a conditional
tail-loss budget and a wider disaster-loss budget.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, Iterable, Tuple

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _quantile(values: Iterable[float], q: float, default: float = 0.0) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, float(np.clip(q, 0.0, 1.0))))


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.mean(arr))


class DistributionalTailBudgetSizer:
    """Causal conditional payoff sizer for forecast-prior exposure.

    Each matured decision contributes one realized unit payoff to a conditional
    bucket keyed by forecast direction, confidence, and predicted payoff
    magnitude.  Sizing uses forecast-signed expected payoff for the matching
    bucket and limits notional through two tail budgets:

    * conditional budget: normal risk budget for the current evidence bucket;
    * disaster budget: wider guardrail against pooled historical tail prints.

    The result is a forecast-prior target, not a learned policy action.
    """

    def __init__(
        self,
        *,
        window_size: int = 2000,
        min_samples: int = 50,
        tail_quantile: float = 0.99,
        disaster_quantile: float = 0.999,
        conditional_loss_budget_fraction: float = 0.05,
        disaster_loss_budget_fraction: float = 0.20,
        conditional_tail_floor: float = 2.0,
        disaster_tail_floor: float = 25.0,
        default_tail_return: float = 25.0,
        return_clip: float = 250.0,
        edge_scale: float = 0.05,
        edge_hurdle: float = 0.0,
        confidence_power: float = 1.0,
        bucket_mode: str = "conditional",
        use_confidence_weight: bool = True,
        fixed_cap_abs: float = 0.10,
        max_abs_exposure: float = 0.60,
        cold_start_exposure: float = 0.0,
    ) -> None:
        self.window_size = int(max(1, window_size))
        self.min_samples = int(max(1, min_samples))
        self.tail_quantile = float(np.clip(tail_quantile, 0.50, 0.9999))
        self.disaster_quantile = float(np.clip(disaster_quantile, self.tail_quantile, 0.9999))
        self.conditional_loss_budget_fraction = float(
            np.clip(conditional_loss_budget_fraction, 0.0, 1.0)
        )
        self.disaster_loss_budget_fraction = float(np.clip(disaster_loss_budget_fraction, 0.0, 1.0))
        self.conditional_tail_floor = float(max(conditional_tail_floor, 1e-9))
        self.disaster_tail_floor = float(max(disaster_tail_floor, 1e-9))
        self.default_tail_return = float(max(default_tail_return, 1e-9))
        self.return_clip = float(max(return_clip, 1e-9))
        self.edge_scale = float(max(edge_scale, 1e-9))
        self.edge_hurdle = float(max(edge_hurdle, 0.0))
        self.confidence_power = float(max(confidence_power, 0.0))
        mode = str(bucket_mode or "conditional").strip().lower()
        self.bucket_mode = mode if mode in {"conditional", "global", "directional_fixed_cap"} else "conditional"
        self.use_confidence_weight = bool(use_confidence_weight)
        self.fixed_cap_abs = float(np.clip(fixed_cap_abs, 0.0, 1.0))
        self.max_abs_exposure = float(np.clip(max_abs_exposure, 0.0, 1.0))
        self.cold_start_exposure = float(np.clip(cold_start_exposure, 0.0, self.max_abs_exposure))
        self._spread_scale_bins = bool(
            max(
                self.conditional_tail_floor,
                self.disaster_tail_floor,
                self.default_tail_return,
                self.return_clip,
                self.edge_scale,
            )
            >= 100.0
        )

        self._signed_by_bucket: Dict[Tuple[int, int, int], Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._abs_by_bucket: Dict[Tuple[int, int, int], Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._signed_global: Deque[float] = deque(maxlen=self.window_size)
        self._abs_global: Deque[float] = deque(maxlen=self.window_size)

    @staticmethod
    def _sign_bin(forecast_sign: float) -> int:
        sign = float(np.sign(_safe_float(forecast_sign, 0.0)))
        if sign > 0:
            return 1
        if sign < 0:
            return -1
        return 0

    @staticmethod
    def _confidence_bin(confidence: float) -> int:
        c = float(np.clip(_safe_float(confidence, 0.5), 0.0, 1.0))
        if c >= 0.85:
            return 4
        if c >= 0.75:
            return 3
        if c >= 0.65:
            return 2
        if c >= 0.55:
            return 1
        return 0

    def _magnitude_bin(self, predicted_payoff: float) -> int:
        mag = abs(_safe_float(predicted_payoff, 0.0))
        if self._spread_scale_bins:
            # Prototype3 mwh_volume mode uses DKK/MWh spreads as unit payoffs.
            # Keep these thresholds monotonic: small spreads stay in low
            # evidence buckets, while extreme settlement tails get isolated.
            if mag >= 5000.0:
                return 4
            if mag >= 1000.0:
                return 3
            if mag >= 250.0:
                return 2
            if mag >= 50.0:
                return 1
            return 0
        # Legacy percentage-return bins.
        if mag >= 1.0:
            return 4
        if mag >= 0.25:
            return 3
        if mag >= 0.05:
            return 2
        if mag >= 0.01:
            return 1
        return 0

    def bucket_key(
        self,
        *,
        forecast_sign: float,
        confidence: float,
        predicted_payoff: float,
    ) -> Tuple[int, int, int]:
        return (
            self._sign_bin(forecast_sign),
            self._confidence_bin(confidence),
            self._magnitude_bin(predicted_payoff),
        )

    def update(
        self,
        *,
        forecast_sign: float,
        confidence: float,
        predicted_payoff: float,
        realized_payoff: float,
    ) -> None:
        sign = self._sign_bin(forecast_sign)
        payoff = float(np.clip(_safe_float(realized_payoff, 0.0), -self.return_clip, self.return_clip))
        signed_payoff = float(sign * payoff) if sign != 0 else float(payoff)
        abs_payoff = float(abs(payoff))
        key = self.bucket_key(
            forecast_sign=sign,
            confidence=confidence,
            predicted_payoff=predicted_payoff,
        )
        self._signed_by_bucket[key].append(signed_payoff)
        self._abs_by_bucket[key].append(abs_payoff)
        self._signed_global.append(signed_payoff)
        self._abs_global.append(abs_payoff)

    def size(
        self,
        *,
        forecast_sign: float,
        confidence: float,
        predicted_payoff: float,
        max_position_notional_dkk: float,
        sleeve_value_dkk: float,
        margin_surplus_dkk: float,
    ) -> Dict[str, float]:
        sign = self._sign_bin(forecast_sign)
        max_position = float(max(_safe_float(max_position_notional_dkk, 0.0), 0.0))
        sleeve_value = float(max(_safe_float(sleeve_value_dkk, 0.0), 0.0))
        margin_surplus = float(max(_safe_float(margin_surplus_dkk, 0.0), 0.0))
        key = self.bucket_key(
            forecast_sign=sign,
            confidence=confidence,
            predicted_payoff=predicted_payoff,
        )
        bucket_signed = list(self._signed_by_bucket.get(key, ()))
        bucket_abs = list(self._abs_by_bucket.get(key, ()))
        global_signed = list(self._signed_global)
        global_abs = list(self._abs_global)
        bucket_n = len(bucket_signed)
        global_n = len(global_signed)
        global_tail_empirical = _quantile(global_abs, self.tail_quantile, default=0.0)
        global_tail = float(
            max(
                self.conditional_tail_floor,
                global_tail_empirical,
                self.default_tail_return,
                1e-9,
            )
        )
        global_budget = float(
            min(
                sleeve_value * self.conditional_loss_budget_fraction,
                margin_surplus * 0.90,
            )
        )
        global_cap = (
            global_budget / (max_position * global_tail)
            if global_n >= self.min_samples
            and global_budget > 0.0
            and max_position > 0.0
            else self.cold_start_exposure
        )
        global_cap = float(np.clip(global_cap, 0.0, self.max_abs_exposure))
        use_bucket = self.bucket_mode == "conditional" and bucket_n >= self.min_samples
        evidence_signed = bucket_signed if use_bucket else global_signed
        evidence_abs = bucket_abs if use_bucket else global_abs
        evidence_n = bucket_n if use_bucket else global_n

        active = False
        if sign != 0 and max_position > 0.0 and sleeve_value > 0.0 and evidence_n >= self.min_samples:
            edge_mean = _mean(evidence_signed)
            shrink = float(np.sqrt(evidence_n / max(evidence_n + self.min_samples, 1)))
            if self.use_confidence_weight:
                # ``confidence`` is already aligned to ``forecast_sign`` by
                # ForecastPriorProcessor: values below 0.5 mean that the
                # classifier supports the opposite direction. Treating the
                # distance from 0.5 symmetrically would turn strong opposition
                # into strong sizing confidence for the selected sign.
                confidence_weight = float(
                    np.clip(2.0 * _safe_float(confidence, 0.5) - 1.0, 0.0, 1.0)
                )
                if self.confidence_power != 1.0:
                    confidence_weight = float(confidence_weight ** self.confidence_power)
            else:
                confidence_weight = 1.0
            shrunk_edge = float(edge_mean * shrink)
            edge_excess = float(max(0.0, shrunk_edge - self.edge_hurdle))
            edge_strength = float(np.clip(edge_excess / self.edge_scale, 0.0, 1.0))
            conditional_tail_empirical = _quantile(evidence_abs, self.tail_quantile, default=0.0)
            conditional_tail = float(
                max(self.conditional_tail_floor, conditional_tail_empirical, 1e-9)
            )
            disaster_tail_empirical = _quantile(global_abs, self.disaster_quantile, default=0.0)
            disaster_tail = float(
                max(self.disaster_tail_floor, disaster_tail_empirical, self.default_tail_return, 1e-9)
            )
            conditional_budget = float(
                min(
                    sleeve_value * self.conditional_loss_budget_fraction,
                    margin_surplus * 0.90,
                )
            )
            disaster_budget = float(
                min(
                    sleeve_value * self.disaster_loss_budget_fraction,
                    margin_surplus * 0.90,
                )
            )
            conditional_cap = (
                conditional_budget / (max_position * conditional_tail)
                if conditional_budget > 0.0
                else 0.0
            )
            disaster_cap = (
                disaster_budget / (max_position * disaster_tail)
                if disaster_budget > 0.0
                else 0.0
            )
            tail_cap_abs = float(
                np.clip(min(conditional_cap, disaster_cap, self.max_abs_exposure), 0.0, self.max_abs_exposure)
            )
            if self.bucket_mode == "directional_fixed_cap":
                target_abs = float(
                    np.clip(
                        min(self.fixed_cap_abs, disaster_cap, self.max_abs_exposure),
                        0.0,
                        self.max_abs_exposure,
                    )
                )
                # Keep a basic evidence gate: if the pooled signed payoff is
                # negative after shrinkage, the directional comparator stands down.
                if edge_strength <= 0.0:
                    target_abs = 0.0
            else:
                target_abs = float(tail_cap_abs * edge_strength * confidence_weight)
            exposure = float(sign * target_abs)
            active = bool(abs(exposure) > 1e-9)
        else:
            edge_mean = _mean(evidence_signed) if evidence_n else 0.0
            shrink = 0.0
            confidence_weight = 0.0
            shrunk_edge = 0.0
            edge_excess = 0.0
            edge_strength = 0.0
            conditional_tail_empirical = 0.0
            conditional_tail = self.default_tail_return
            disaster_tail_empirical = 0.0
            disaster_tail = self.default_tail_return
            conditional_budget = 0.0
            disaster_budget = 0.0
            conditional_cap = 0.0
            disaster_cap = 0.0
            tail_cap_abs = self.cold_start_exposure if sign != 0 else 0.0
            exposure = float(sign * tail_cap_abs)

        return {
            "prior_exposure": float(np.clip(exposure, -self.max_abs_exposure, self.max_abs_exposure)),
            "prior_exposure_uncapped": float(exposure),
            "active": float(1.0 if active else 0.0),
            "bucket_count": float(bucket_n),
            "global_count": float(global_n),
            "global_tail_return": float(global_tail),
            "global_tail_empirical": float(global_tail_empirical),
            "global_cap_abs": float(global_cap),
            "global_budget_dkk": float(global_budget),
            "evidence_count": float(evidence_n),
            "using_bucket": float(1.0 if use_bucket else 0.0),
            "edge_mean": float(edge_mean),
            "edge_shrink": float(shrink),
            "edge_shrunk": float(shrunk_edge),
            "edge_excess": float(edge_excess),
            "edge_strength": float(edge_strength),
            "confidence_weight": float(confidence_weight),
            "conditional_tail_return": float(conditional_tail),
            "conditional_tail_empirical": float(conditional_tail_empirical),
            "disaster_tail_return": float(disaster_tail),
            "disaster_tail_empirical": float(disaster_tail_empirical),
            "conditional_cap_abs": float(conditional_cap),
            "disaster_cap_abs": float(disaster_cap),
            "tail_cap_abs": float(tail_cap_abs),
            "conditional_budget_dkk": float(conditional_budget),
            "disaster_budget_dkk": float(disaster_budget),
        }

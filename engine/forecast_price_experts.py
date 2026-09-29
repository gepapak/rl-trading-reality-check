from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Concatenate, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model, load_model


logger = logging.getLogger(__name__)

PRICE_SHORT_EXPERT_METHODS: Tuple[str, ...] = (
    "ann",
)

PRICE_SHORT_EXPERT_LABELS: Dict[str, str] = {
    "ann": "ANN",
}

PRICE_SHORT_EXPERT_VERSION = "3.0.0"
PRICE_SHORT_EXPERT_TARGET = "price"
PRICE_SHORT_EXPERT_HORIZON = "short"
PRICE_SHORT_EXPERT_DENOM_FLOOR = 50.0
# Decision-focused training (Tier-B variant): selected per-run via the
# FORECAST_TRAINING_LOSS environment variable so the deep call chain
# (trainer wrapper -> forecast_engine -> expert bank) needs no signature
# changes. "statistical" reproduces the frozen v2 behaviour exactly.
PRICE_SHORT_EXPERT_TRAINING_LOSS_ENV = "FORECAST_TRAINING_LOSS"
PRICE_SHORT_EXPERT_TRAINING_LOSSES = {"statistical", "decision_focused_v1"}
# Pre-registered single configuration (no sweeps): tau matches the prior's
# distributional edge scale (250 DKK/MWh); beta keeps the price head anchored
# to the DKK scale the anchor's magnitude mapping and calibration expect.
PRICE_SHORT_EXPERT_DF_TAU_DKK = 250.0
PRICE_SHORT_EXPERT_DF_MSE_BETA = 0.25


def _price_short_expert_training_loss() -> str:
    mode = str(os.environ.get(PRICE_SHORT_EXPERT_TRAINING_LOSS_ENV, "statistical")).strip().lower()
    if mode not in PRICE_SHORT_EXPERT_TRAINING_LOSSES:
        raise ValueError(
            f"Unsupported {PRICE_SHORT_EXPERT_TRAINING_LOSS_ENV}={mode!r}; "
            f"expected one of {sorted(PRICE_SHORT_EXPERT_TRAINING_LOSSES)}"
        )
    return mode
PRICE_SHORT_EXPERT_ANN_LATENT_DIM = 4


def _set_deterministic_seed(seed: int) -> None:
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, anchors: np.ndarray) -> float:
    denom = np.maximum(np.abs(anchors), PRICE_SHORT_EXPERT_DENOM_FLOOR)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.abs(y_true - y_pred) / denom
    return float(np.nanmean(vals) * 100.0)


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, anchors: np.ndarray) -> float:
    actual_ret = np.asarray(y_true, dtype=np.float32) - np.asarray(anchors, dtype=np.float32)
    pred_ret = np.asarray(y_pred, dtype=np.float32) - np.asarray(anchors, dtype=np.float32)
    mask = np.abs(actual_ret) > 1e-8
    if not np.any(mask):
        return 0.5
    return float(np.mean(np.sign(actual_ret[mask]) == np.sign(pred_ret[mask])))


def _residual_risk(y_true: np.ndarray, y_pred: np.ndarray, anchors: np.ndarray) -> float:
    denom = np.maximum(np.abs(anchors), PRICE_SHORT_EXPERT_DENOM_FLOOR)
    actual_ret = (np.asarray(y_true, dtype=np.float32) - anchors) / denom
    pred_ret = (np.asarray(y_pred, dtype=np.float32) - anchors) / denom
    residuals = np.abs(actual_ret - pred_ret)
    if residuals.size <= 0:
        return 0.5
    tail = float(np.quantile(residuals, 0.90))
    return float(np.clip(np.tanh(tail / 0.10), 0.0, 1.0))


def _create_horizon_windows(
    series: np.ndarray,
    look_back: int,
    horizon_steps: int,
    reference_series: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(series, dtype=np.float32).reshape(-1)
    references = values if reference_series is None else np.asarray(reference_series, dtype=np.float32).reshape(-1)
    if references.size != values.size:
        raise ValueError(
            f"reference_series length mismatch: got {references.size}, expected {values.size}"
        )
    n = int(values.size)
    end = n - int(look_back) - int(horizon_steps) + 1
    if end <= 0:
        return (
            np.zeros((0, int(look_back)), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    x = np.zeros((end, int(look_back)), dtype=np.float32)
    y = np.zeros(end, dtype=np.float32)
    anchors = np.zeros(end, dtype=np.float32)
    for idx in range(end):
        window = values[idx: idx + int(look_back)]
        target_idx = idx + int(look_back) + int(horizon_steps) - 1
        x[idx, :] = window
        y[idx] = float(values[target_idx])
        anchors[idx] = float(references[target_idx])
    return x, y, anchors


def _ann_input_view(windows: np.ndarray) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values.astype(np.float32)


def _split_series_three_way(series: np.ndarray, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(series, dtype=np.float32).reshape(-1)
    n = int(values.size)
    train_size = int(n * float(train_ratio))
    val_size = int(n * float(val_ratio))
    train = values[:train_size]
    val = values[train_size: train_size + val_size]
    test = values[train_size + val_size:]
    return train, val, test


def _fit_standard_scalers(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[StandardScaler, StandardScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x_train_scaled = sc_x.fit_transform(x_train)
    y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    x_val_scaled = sc_x.transform(x_val)
    y_val_scaled = sc_y.transform(y_val.reshape(-1, 1)).ravel()
    x_test_scaled = sc_x.transform(x_test)
    y_test_scaled = sc_y.transform(y_test.reshape(-1, 1)).ravel()
    return (
        sc_x,
        sc_y,
        x_train_scaled.astype(np.float32),
        y_train_scaled.astype(np.float32),
        x_val_scaled.astype(np.float32),
        y_val_scaled.astype(np.float32),
        x_test_scaled.astype(np.float32),
        y_test_scaled.astype(np.float32),
    )


def _decision_focused_price_loss(tau_scaled: float, beta: float):
    """Profit-weighted price loss on the same-delivery spread.

    ``y_true`` carries two columns in scaled target space: the settlement
    target and the entry anchor. The differentiable soft position
    ``tanh(predicted_spread / tau)`` is paid the realized spread, so errors on
    large spreads dominate exactly as they do in the trading P&L; the MSE term
    keeps the head anchored to the DKK scale the anchor's magnitude mapping
    and online calibration consume.
    """
    tau = tf.constant(max(float(tau_scaled), 1e-6), dtype=tf.float32)
    beta_c = tf.constant(max(float(beta), 0.0), dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        target = y_true[:, 0:1]
        anchor = y_true[:, 1:2]
        pred_spread = (y_pred - anchor) / tau
        real_spread = (target - anchor) / tau
        utility = tf.tanh(pred_spread) * real_spread
        mse = tf.square(y_pred - target)
        return tf.reduce_mean(-utility + beta_c * mse, axis=-1)

    return loss


def _compile_keras_forecast_model(
    model: Any,
    *,
    df_price_loss: Any = None,
) -> Any:
    output_names = [str(name) for name in list(getattr(model, "output_names", []) or [])]
    if {"price", "direction", "uncertainty"}.issubset(set(output_names)):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
            loss={
                "price": df_price_loss if df_price_loss is not None else "mse",
                "direction": "binary_crossentropy",
                "uncertainty": "mse",
            },
            loss_weights={
                "price": 1.0,
                "direction": 0.18,
                "uncertainty": 0.12,
            },
        )
    else:
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4))
    return model


def _build_ann_model(input_dim: int, *, df_price_loss: Any = None) -> Model:
    inp = tf.keras.Input(shape=(int(input_dim),), name="ann_window")
    x = Dense(256, activation="gelu", name="ann_stem")(inp)
    x = Dropout(0.08, name="ann_stem_do")(x)
    for idx in range(3):
        residual = Dense(256, activation="gelu", name=f"ann_block_{idx}_d1")(x)
        residual = Dropout(0.06, name=f"ann_block_{idx}_do")(residual)
        residual = Dense(256, name=f"ann_block_{idx}_d2")(residual)
        x = LayerNormalization(epsilon=1e-6, name=f"ann_block_{idx}_ln")(x + residual)
        x = Activation("gelu", name=f"ann_block_{idx}_act")(x)
    trunk = Dense(128, activation="gelu", name="ann_trunk")(x)
    latent_hidden = Dense(32, activation="gelu", name="ann_latent_hidden")(trunk)
    latent = Dense(PRICE_SHORT_EXPERT_ANN_LATENT_DIM, activation="tanh", name="latent")(latent_hidden)
    shared = Concatenate(name="ann_shared_concat")([trunk, latent])

    price_hidden = Dense(64, activation="gelu", name="ann_price_hidden")(shared)
    price_out = Dense(1, name="price")(price_hidden)

    direction_hidden = Dense(48, activation="gelu", name="ann_direction_hidden")(shared)
    direction_out = Dense(1, activation="sigmoid", name="direction")(direction_hidden)

    uncertainty_hidden = Dense(48, activation="gelu", name="ann_uncertainty_hidden")(shared)
    uncertainty_out = Dense(1, activation="softplus", name="uncertainty")(uncertainty_hidden)

    model = Model(
        inputs=inp,
        outputs={
            "price": price_out,
            "direction": direction_out,
            "uncertainty": uncertainty_out,
            "latent": latent,
        },
        name="price_short_ann_multitask",
    )
    return _compile_keras_forecast_model(model, df_price_loss=df_price_loss)


def _fit_keras_model(
    model: Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_path: str,
    history_path: str,
    epochs: int = 120,
    batch_size: int = 64,
    patience: int = 12,
    sample_weight: Any = None,
) -> Model:
    ckpt = ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=0,
    )
    early = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=max(4, int(patience)),
        restore_best_weights=True,
        verbose=0,
    )
    rlrop = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=max(2, int(patience) // 3),
        min_lr=1e-5,
        verbose=0,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=int(epochs),
        batch_size=int(batch_size),
        shuffle=False,
        callbacks=[ckpt, early, rlrop],
        sample_weight=sample_weight,
        verbose=0,
    )
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            fh,
            indent=2,
        )
    best_model = load_model(model_path, compile=False)
    _compile_keras_forecast_model(best_model)
    return best_model


def _unpack_model_predictions(
    model: Any,
    raw_pred: Any,
) -> Dict[str, np.ndarray]:
    if isinstance(raw_pred, dict):
        out = {
            str(k): np.asarray(v, dtype=np.float32).reshape((len(v), -1)) if np.asarray(v).ndim > 1 else np.asarray(v, dtype=np.float32).reshape(-1, 1)
            for k, v in raw_pred.items()
        }
    elif isinstance(raw_pred, (list, tuple)):
        names = [str(name) for name in list(getattr(model, "output_names", []) or [])]
        out = {}
        for idx, value in enumerate(raw_pred):
            key = names[idx] if idx < len(names) else f"output_{idx}"
            arr = np.asarray(value, dtype=np.float32)
            out[key] = arr.reshape((arr.shape[0], -1)) if arr.ndim > 1 else arr.reshape(-1, 1)
    else:
        arr = np.asarray(raw_pred, dtype=np.float32)
        out = {"price": arr.reshape((arr.shape[0], -1)) if arr.ndim > 1 else arr.reshape(-1, 1)}
    if "price" not in out and out:
        first_key = next(iter(out))
        out["price"] = np.asarray(out[first_key], dtype=np.float32)
    if "direction" not in out:
        out["direction"] = np.full_like(out["price"], 0.5, dtype=np.float32)
    if "uncertainty" not in out:
        out["uncertainty"] = np.zeros_like(out["price"], dtype=np.float32)
    if "latent" not in out:
        out["latent"] = np.zeros((int(out["price"].shape[0]), PRICE_SHORT_EXPERT_ANN_LATENT_DIM), dtype=np.float32)
    return out


def get_price_short_expert_root(episode_dir: str) -> str:
    return os.path.join(str(episode_dir), "price_short_experts")


def get_price_short_expert_paths(episode_dir: str, method: str) -> Dict[str, str]:
    method_key = str(method).strip().lower()
    if method_key not in PRICE_SHORT_EXPERT_LABELS:
        raise ValueError(f"Unknown price-short expert method: {method}")
    root = _ensure_dir(os.path.join(get_price_short_expert_root(episode_dir), PRICE_SHORT_EXPERT_LABELS[method_key]))
    return {
        "root": root,
        "model_path": os.path.join(root, f"{method_key}_model.keras"),
        "scaler_x_path": os.path.join(root, f"{method_key}_scaler_x.pkl"),
        "scaler_y_path": os.path.join(root, f"{method_key}_scaler_y.pkl"),
        "history_path": os.path.join(root, f"{method_key}_history.json"),
        "metadata_path": os.path.join(root, f"{method_key}_metadata.json"),
    }


def price_short_expert_bank_exists(episode_dir: str) -> bool:
    try:
        for method in PRICE_SHORT_EXPERT_METHODS:
            paths = get_price_short_expert_paths(episode_dir, method)
            if not os.path.isfile(paths["metadata_path"]):
                return False
            with open(paths["metadata_path"], "r", encoding="utf-8-sig") as fh:
                metadata = json.load(fh)
            if str(metadata.get("version", "")) != PRICE_SHORT_EXPERT_VERSION:
                return False
            if not os.path.isfile(paths["model_path"]):
                return False
            if not os.path.isfile(paths["scaler_x_path"]):
                return False
            if not os.path.isfile(paths["scaler_y_path"]):
                return False
        return True
    except Exception:
        return False


def _evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anchors: np.ndarray,
) -> Dict[str, float]:
    return {
        "mape": _safe_mape(y_true, y_pred, anchors),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "directional_accuracy": _directional_accuracy(y_true, y_pred, anchors),
        "residual_risk": _residual_risk(y_true, y_pred, anchors),
    }


def _build_metadata_quality(metrics: Dict[str, float]) -> float:
    dir_acc = float(np.clip(metrics.get("directional_accuracy", 0.5), 0.0, 1.0))
    mape = float(max(metrics.get("mape", 100.0), 0.0))
    residual_risk = float(np.clip(metrics.get("residual_risk", 0.5), 0.0, 1.0))
    q_mape = 1.0 / (1.0 + mape / 25.0)
    return float(np.clip(0.50 * dir_acc + 0.30 * q_mape + 0.20 * (1.0 - residual_risk), 0.0, 1.0))


@dataclass
class ExpertArtifacts:
    method: str
    model: Any
    scaler_x: StandardScaler
    scaler_y: StandardScaler
    metadata: Dict[str, Any]


class PriceShortExpertBank:
    """ANN-only short-horizon price forecast bank used by the active forecast path."""

    def __init__(
        self,
        episode_dir: str,
        look_back: int,
        horizon_steps: int,
        verbose: bool = False,
        refresh_stride: int = 6,
    ):
        self.episode_dir = str(episode_dir)
        self.look_back = int(look_back)
        self.horizon_steps = int(horizon_steps)
        self.verbose = bool(verbose)
        self.refresh_stride = max(1, int(refresh_stride))
        self.artifacts: Dict[str, ExpertArtifacts] = {}
        self._load()

    def _load(self) -> None:
        self.artifacts.clear()
        for method in PRICE_SHORT_EXPERT_METHODS:
            paths = get_price_short_expert_paths(self.episode_dir, method)
            if not os.path.isfile(paths["metadata_path"]):
                continue
            with open(paths["metadata_path"], "r", encoding="utf-8-sig") as fh:
                metadata = json.load(fh)
            if str(metadata.get("version", "")) != PRICE_SHORT_EXPERT_VERSION:
                logger.warning(
                    "Price-short expert %s in %s uses metadata version %s; current version is %s.",
                    method,
                    self.episode_dir,
                    metadata.get("version", "unknown"),
                    PRICE_SHORT_EXPERT_VERSION,
                )
                continue
            scaler_x = joblib.load(paths["scaler_x_path"])
            scaler_y = joblib.load(paths["scaler_y_path"])
            model = load_model(paths["model_path"], compile=False)
            self.artifacts[method] = ExpertArtifacts(
                method=method,
                model=model,
                scaler_x=scaler_x,
                scaler_y=scaler_y,
                metadata=dict(metadata),
            )
        if self.verbose:
            logger.info(
                "Loaded %s/%s ANN short-forecast artifacts from %s",
                len(self.artifacts),
                len(PRICE_SHORT_EXPERT_METHODS),
                self.episode_dir,
            )

    def is_complete(self) -> bool:
        return all(method in self.artifacts for method in PRICE_SHORT_EXPERT_METHODS)

    def methods(self) -> Tuple[str, ...]:
        return tuple(method for method in PRICE_SHORT_EXPERT_METHODS if method in self.artifacts)

    def get_metadata_quality(self, method: str) -> float:
        art = self.artifacts.get(str(method).strip().lower())
        if art is None:
            return 0.5
        return float(np.clip(art.metadata.get("metadata_quality", 0.5), 0.0, 1.0))

    def get_metadata_metrics(self, method: str) -> Dict[str, float]:
        art = self.artifacts.get(str(method).strip().lower())
        if art is None:
            return {}
        metrics = art.metadata.get("test_metrics", {}) or {}
        return {str(k): float(v) for k, v in metrics.items() if np.isfinite(v)}

    def get_cache_validation_info(self) -> Dict[str, Dict[str, float | str]]:
        info: Dict[str, Dict[str, float | str]] = {}
        for method in self.methods():
            paths = get_price_short_expert_paths(self.episode_dir, method)
            info[method] = {
                "model_path": paths["model_path"],
                "model_mtime": float(os.path.getmtime(paths["model_path"])) if os.path.isfile(paths["model_path"]) else 0.0,
                "metadata_path": paths["metadata_path"],
                "metadata_mtime": float(os.path.getmtime(paths["metadata_path"])) if os.path.isfile(paths["metadata_path"]) else 0.0,
            }
        return info

    def _build_series_windows(self, series: np.ndarray) -> np.ndarray:
        values = np.asarray(series, dtype=np.float32).reshape(-1)
        t_count = int(values.size)
        if t_count <= 0:
            return np.zeros((0, self.look_back), dtype=np.float32)
        windows = np.zeros((t_count, self.look_back), dtype=np.float32)
        ann_metadata = self.artifacts.get("ann").metadata if "ann" in self.artifacts else {}
        mean_val = float(ann_metadata.get("training_price_mean", 0.0))
        if not np.isfinite(mean_val):
            raise ValueError("ANN metadata contains a non-finite training_price_mean")
        for t in range(t_count):
            # The origin-t action is chosen before settlement_t is known.
            # Every window therefore ends at t-1.
            start = max(0, t - self.look_back)
            hist = values[start:t]
            if hist.size <= 0:
                windows[t, :] = mean_val
            elif hist.size < self.look_back:
                windows[t, : self.look_back - hist.size] = mean_val
                windows[t, self.look_back - hist.size :] = hist
            else:
                windows[t, :] = hist[-self.look_back :]
        return windows

    def _predict_ann_details_batch(
        self,
        windows: np.ndarray,
        entry_prices: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        art = self.artifacts["ann"]
        model_input = _ann_input_view(windows)
        requires_entry = bool(art.metadata.get("ann_input_includes_entry_price", False))
        if requires_entry:
            if entry_prices is None:
                raise ValueError("ANN v3 inference requires current delivery entry prices")
            entry = np.asarray(entry_prices, dtype=np.float32).reshape(-1)
            if entry.size != len(model_input) or not np.all(np.isfinite(entry)):
                raise ValueError("ANN v3 entry prices must be finite and match the inference batch")
            model_input = np.concatenate([model_input, entry.reshape(-1, 1)], axis=1)
        x_scaled = art.scaler_x.transform(model_input).astype(np.float32)
        raw_outputs = _unpack_model_predictions(art.model, art.model.predict(x_scaled, verbose=0))
        price_scaled = np.asarray(raw_outputs.get("price"), dtype=np.float32).reshape(-1)
        pred_price = art.scaler_y.inverse_transform(price_scaled.reshape(-1, 1))[:, 0].astype(np.float32)
        anchors = (
            np.asarray(entry_prices, dtype=np.float32).reshape(-1)
            if entry_prices is not None
            else np.asarray(windows[:, -1], dtype=np.float32).reshape(-1)
        )
        denom = np.maximum(np.abs(anchors), PRICE_SHORT_EXPERT_DENOM_FLOOR)
        pred_return = np.clip((pred_price - anchors) / denom, -0.25, 0.25).astype(np.float32)
        direction_prob = np.clip(
            np.asarray(raw_outputs.get("direction"), dtype=np.float32).reshape(-1),
            0.0,
            1.0,
        ).astype(np.float32)
        uncertainty = np.clip(
            np.asarray(raw_outputs.get("uncertainty"), dtype=np.float32).reshape(-1),
            0.0,
            1.0,
        ).astype(np.float32)
        direction_margin = np.clip(2.0 * direction_prob - 1.0, -1.0, 1.0).astype(np.float32)
        latent = np.asarray(raw_outputs.get("latent"), dtype=np.float32)
        latent = latent.reshape((len(pred_price), -1))
        if latent.shape[1] < PRICE_SHORT_EXPERT_ANN_LATENT_DIM:
            latent = np.pad(latent, ((0, 0), (0, PRICE_SHORT_EXPERT_ANN_LATENT_DIM - latent.shape[1])))
        elif latent.shape[1] > PRICE_SHORT_EXPERT_ANN_LATENT_DIM:
            latent = latent[:, :PRICE_SHORT_EXPERT_ANN_LATENT_DIM]
        latent_norm = np.clip(
            np.linalg.norm(latent, axis=1) / max(np.sqrt(float(PRICE_SHORT_EXPERT_ANN_LATENT_DIM)), 1e-6),
            0.0,
            1.0,
        ).astype(np.float32)
        quality = np.clip(
            0.60 * (1.0 - uncertainty) + 0.40 * np.abs(direction_margin),
            0.0,
            1.0,
        ).astype(np.float32)
        return {
            "pred_price": pred_price,
            "pred_return": pred_return,
            "direction_prob": direction_prob,
            "direction_margin": direction_margin,
            "uncertainty": uncertainty,
            "quality": quality,
            "latent": latent.astype(np.float32),
            "latent_norm": latent_norm,
        }

    def predict_from_window(
        self,
        method: str,
        window: np.ndarray,
        entry_price: Optional[float] = None,
    ) -> float:
        values = np.asarray(window, dtype=np.float32).reshape(-1)
        if values.size != self.look_back:
            raise ValueError(f"Expected window length {self.look_back}, got {values.size}")
        method_key = str(method).strip().lower()
        if method_key not in self.artifacts:
            raise KeyError(f"Price-short expert not loaded: {method_key}")
        windows = values.reshape(1, -1)
        entry = None if entry_price is None else np.asarray([entry_price], dtype=np.float32)
        details = self._predict_ann_details_batch(windows, entry_prices=entry)
        return float(details["pred_price"][0])

    def predict_details_from_window(
        self,
        method: str,
        window: np.ndarray,
        entry_price: Optional[float] = None,
    ) -> Dict[str, float]:
        values = np.asarray(window, dtype=np.float32).reshape(-1)
        if values.size != self.look_back:
            raise ValueError(f"Expected window length {self.look_back}, got {values.size}")
        method_key = str(method).strip().lower()
        if method_key not in self.artifacts:
            raise KeyError(f"Price-short expert not loaded: {method_key}")
        windows = values.reshape(1, -1)
        entry = None if entry_price is None else np.asarray([entry_price], dtype=np.float32)
        details = self._predict_ann_details_batch(windows, entry_prices=entry)
        out: Dict[str, float] = {
            "pred_price": float(details["pred_price"][0]),
            "pred_return": float(details["pred_return"][0]),
            "direction_prob": float(details["direction_prob"][0]),
            "direction_margin": float(details["direction_margin"][0]),
            "uncertainty": float(details["uncertainty"][0]),
            "quality": float(details["quality"][0]),
            "latent_norm": float(details["latent_norm"][0]),
        }
        latent = np.asarray(details["latent"][0], dtype=np.float32).reshape(-1)
        for idx in range(min(latent.size, PRICE_SHORT_EXPERT_ANN_LATENT_DIM)):
            out[f"latent_{idx}"] = float(np.clip(latent[idx], -1.0, 1.0))
        return out

    def predict_all_from_window(
        self,
        window: np.ndarray,
        entry_price: Optional[float] = None,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for method in self.methods():
            out[method] = self.predict_from_window(method, window, entry_price=entry_price)
        return out

    def precompute_details_for_series(
        self,
        series: np.ndarray,
        entry_series: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        values = np.asarray(series, dtype=np.float32).reshape(-1)
        entries = None if entry_series is None else np.asarray(entry_series, dtype=np.float32).reshape(-1)
        if entries is not None and (entries.size != values.size or not np.all(np.isfinite(entries))):
            raise ValueError("entry_series must be finite and have the same length as series")
        windows = self._build_series_windows(values)
        if windows.shape[0] <= 0:
            return {
                method: {"pred_price": np.zeros(0, dtype=np.float32)}
                for method in self.methods()
            }
        details: Dict[str, Dict[str, np.ndarray]] = {}
        for method in self.methods():
            ann_details = self._predict_ann_details_batch(windows, entry_prices=entries)
            method_details: Dict[str, np.ndarray] = {
                "pred_price": np.asarray(ann_details["pred_price"], dtype=np.float32),
                "pred_return": np.asarray(ann_details["pred_return"], dtype=np.float32),
                "direction_prob": np.asarray(ann_details["direction_prob"], dtype=np.float32),
                "direction_margin": np.asarray(ann_details["direction_margin"], dtype=np.float32),
                "uncertainty": np.asarray(ann_details["uncertainty"], dtype=np.float32),
                "quality": np.asarray(ann_details["quality"], dtype=np.float32),
                "latent_norm": np.asarray(ann_details["latent_norm"], dtype=np.float32),
            }
            latent = np.asarray(ann_details["latent"], dtype=np.float32)
            for idx in range(min(latent.shape[1], PRICE_SHORT_EXPERT_ANN_LATENT_DIM)):
                method_details[f"latent_{idx}"] = np.asarray(latent[:, idx], dtype=np.float32)
            details[method] = method_details
        return details

    def precompute_for_series(
        self,
        series: np.ndarray,
        entry_series: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        details = self.precompute_details_for_series(series, entry_series=entry_series)
        return {
            method: np.asarray(method_details.get("pred_price", np.zeros(0, dtype=np.float32)), dtype=np.float32)
            for method, method_details in details.items()
        }


def _save_metadata(
    metadata_path: str,
    payload: Dict[str, Any],
) -> None:
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _save_standard_artifacts(
    method: str,
    episode_dir: str,
    model: Any,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    test_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    train_count: int,
    val_count: int,
    test_count: int,
    sampled_counts: Dict[str, int],
    look_back: int,
    horizon_steps: int,
    history_path: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    paths = get_price_short_expert_paths(episode_dir, method)
    model.save(paths["model_path"], include_optimizer=True)
    joblib.dump(scaler_x, paths["scaler_x_path"])
    joblib.dump(scaler_y, paths["scaler_y_path"])
    metadata = {
        "version": PRICE_SHORT_EXPERT_VERSION,
        "method": method,
        "target": PRICE_SHORT_EXPERT_TARGET,
        "horizon": PRICE_SHORT_EXPERT_HORIZON,
        "look_back": int(look_back),
        "horizon_steps": int(horizon_steps),
        "input_view": "raw_price_window",
        "train_count": int(train_count),
        "val_count": int(val_count),
        "test_count": int(test_count),
        "sampled_counts": {k: int(v) for k, v in sampled_counts.items()},
        "model_path": paths["model_path"],
        "scaler_x_path": paths["scaler_x_path"],
        "scaler_y_path": paths["scaler_y_path"],
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "metadata_quality": _build_metadata_quality(test_metrics),
    }
    if history_path and os.path.isfile(history_path):
        metadata["history_path"] = history_path
    if isinstance(extra_metadata, dict):
        for key, value in extra_metadata.items():
            metadata[str(key)] = value
    _save_metadata(paths["metadata_path"], metadata)
    return paths


def _train_ann_expert(
    episode_dir: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    a_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    a_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    a_test: np.ndarray,
    look_back: int,
    horizon_steps: int,
    seed: int,
) -> Dict[str, Any]:
    paths = get_price_short_expert_paths(episode_dir, "ann")
    x_train_view = np.concatenate(
        [_ann_input_view(x_train), np.asarray(a_train, dtype=np.float32).reshape(-1, 1)],
        axis=1,
    )
    x_val_view = np.concatenate(
        [_ann_input_view(x_val), np.asarray(a_val, dtype=np.float32).reshape(-1, 1)],
        axis=1,
    )
    x_test_view = np.concatenate(
        [_ann_input_view(x_test), np.asarray(a_test, dtype=np.float32).reshape(-1, 1)],
        axis=1,
    )
    sc_x, sc_y, x_train_s, y_train_s, x_val_s, y_val_s, _, _ = _fit_standard_scalers(
        x_train_view, y_train, x_val_view, y_val, x_test_view, y_test
    )
    x_test_s = sc_x.transform(x_test_view).astype(np.float32)
    denom_train = np.maximum(np.abs(a_train), PRICE_SHORT_EXPERT_DENOM_FLOOR)
    denom_val = np.maximum(np.abs(a_val), PRICE_SHORT_EXPERT_DENOM_FLOOR)
    dir_train = ((y_train - a_train) > 0.0).astype(np.float32)
    dir_val = ((y_val - a_val) > 0.0).astype(np.float32)
    sigma_train = np.clip(np.abs(y_train - a_train) / denom_train, 0.0, 1.0).astype(np.float32)
    sigma_val = np.clip(np.abs(y_val - a_val) / denom_val, 0.0, 1.0).astype(np.float32)
    _set_deterministic_seed(seed)
    training_loss = _price_short_expert_training_loss()
    df_price_loss = None
    price_train_target: np.ndarray = y_train_s
    price_val_target: np.ndarray = y_val_s
    fit_sample_weight = None
    if training_loss == "decision_focused_v1":
        # Anchor (entry) in the same scaled target space as the price head.
        a_train_s = sc_y.transform(np.asarray(a_train, dtype=np.float64).reshape(-1, 1)).astype(np.float32)
        a_val_s = sc_y.transform(np.asarray(a_val, dtype=np.float64).reshape(-1, 1)).astype(np.float32)
        y_scale = float(np.ravel(getattr(sc_y, "scale_", [1.0]))[0]) or 1.0
        tau_scaled = PRICE_SHORT_EXPERT_DF_TAU_DKK / max(abs(y_scale), 1e-9)
        df_price_loss = _decision_focused_price_loss(tau_scaled, PRICE_SHORT_EXPERT_DF_MSE_BETA)
        price_train_target = np.concatenate(
            [np.asarray(y_train_s, dtype=np.float32).reshape(-1, 1), a_train_s], axis=1
        )
        price_val_target = np.concatenate(
            [np.asarray(y_val_s, dtype=np.float32).reshape(-1, 1), a_val_s], axis=1
        )
        # Cost-sensitive direction head: each sample weighted by its realized
        # spread magnitude, so directional accuracy is optimized where it pays.
        spread_train = np.abs(np.asarray(y_train, dtype=np.float64) - np.asarray(a_train, dtype=np.float64))
        w_train = (spread_train / max(float(np.mean(spread_train)), 1e-9)).astype(np.float32)
        fit_sample_weight = {"direction": w_train}
    model = _build_ann_model(x_train_view.shape[1], df_price_loss=df_price_loss)
    best_model = _fit_keras_model(
        model,
        x_train_s,
        {
            "price": price_train_target,
            "direction": dir_train,
            "uncertainty": sigma_train,
        },
        x_val_s,
        {
            "price": price_val_target,
            "direction": dir_val,
            "uncertainty": sigma_val,
        },
        paths["model_path"],
        paths["history_path"],
        epochs=100,
        batch_size=64,
        patience=10,
        sample_weight=fit_sample_weight,
    )
    val_outputs = _unpack_model_predictions(best_model, best_model.predict(x_val_s, verbose=0))
    test_outputs = _unpack_model_predictions(best_model, best_model.predict(x_test_s, verbose=0))
    val_pred = sc_y.inverse_transform(np.ravel(val_outputs["price"]).reshape(-1, 1))[:, 0]
    test_pred = sc_y.inverse_transform(np.ravel(test_outputs["price"]).reshape(-1, 1))[:, 0]
    val_metrics = _evaluate_forecast(y_val, val_pred, a_val)
    test_metrics = _evaluate_forecast(y_test, test_pred, a_test)
    val_uncertainty = float(np.mean(np.clip(np.asarray(val_outputs["uncertainty"]).reshape(-1), 0.0, 1.0)))
    test_uncertainty = float(np.mean(np.clip(np.asarray(test_outputs["uncertainty"]).reshape(-1), 0.0, 1.0)))
    val_direction_conf = float(np.mean(np.abs(2.0 * np.asarray(val_outputs["direction"]).reshape(-1) - 1.0)))
    test_direction_conf = float(np.mean(np.abs(2.0 * np.asarray(test_outputs["direction"]).reshape(-1) - 1.0)))
    _save_standard_artifacts(
        "ann",
        episode_dir,
        best_model,
        sc_x,
        sc_y,
        test_metrics,
        val_metrics,
        x_train.shape[0],
        x_val.shape[0],
        x_test.shape[0],
        {"train": x_train.shape[0], "val": x_val.shape[0], "test": x_test.shape[0]},
        look_back,
        horizon_steps,
        history_path=paths["history_path"],
        extra_metadata={
            "architecture": "multitask_residual_ann_forecaster",
            "training_loss": training_loss,
            "ann_input_includes_entry_price": True,
            "forecast_information_set": "settlement_history_strictly_before_origin_plus_current_day_ahead_entry",
            "direction_target": "same_delivery_settlement_minus_entry_index",
            "target_delivery_alignment": "same_delivery_hour",
            "ann_latent_dim": int(PRICE_SHORT_EXPERT_ANN_LATENT_DIM),
            "val_mean_uncertainty": val_uncertainty,
            "test_mean_uncertainty": test_uncertainty,
            "val_direction_confidence": val_direction_conf,
            "test_direction_confidence": test_direction_conf,
        },
    )
    return {"method": "ann", "success": True, "test_metrics": test_metrics, "val_metrics": val_metrics}


def train_price_short_expert_bank(
    episode_num: int,
    data_filtered: Any,
    output_base_dir: str = "forecast_models",
    look_back: int = 24,
    horizon_steps: int = 6,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 1234,
) -> Dict[str, Any]:
    episode_dir = os.path.join(str(output_base_dir), f"episode_{int(episode_num)}")
    _ensure_dir(get_price_short_expert_root(episode_dir))

    if PRICE_SHORT_EXPERT_TARGET not in data_filtered.columns:
        raise ValueError(f"'{PRICE_SHORT_EXPERT_TARGET}' column not found for ANN short-forecast training.")

    series = np.asarray(data_filtered[PRICE_SHORT_EXPERT_TARGET].astype(np.float32).values, dtype=np.float32)
    reference_column = next(
        (name for name in ("entry_price_index", "energy_index_price") if name in data_filtered.columns),
        None,
    )
    if reference_column is None:
        raise ValueError(
            "Settlement-aligned ANN training requires 'entry_price_index' or 'energy_index_price'."
        )
    reference_series = np.asarray(
        data_filtered[reference_column].astype(np.float32).values,
        dtype=np.float32,
    )
    if reference_series.size != series.size or not np.all(np.isfinite(reference_series)):
        raise ValueError("Settlement-aligned entry-price series is incomplete or non-finite")
    if "Date" in data_filtered.columns:
        raw_dates = [str(x) for x in data_filtered["Date"].tolist()]
    elif "timestamp" in data_filtered.columns:
        raw_dates = [str(x) for x in data_filtered["timestamp"].tolist()]
    else:
        raw_dates = []
    train_series, val_series, test_series = _split_series_three_way(series, train_ratio=train_ratio, val_ratio=val_ratio)
    train_ref, val_ref, test_ref = _split_series_three_way(
        reference_series,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    x_train, y_train, a_train = _create_horizon_windows(
        train_series, look_back, horizon_steps, reference_series=train_ref
    )
    x_val, y_val, a_val = _create_horizon_windows(
        val_series, look_back, horizon_steps, reference_series=val_ref
    )
    x_test, y_test, a_test = _create_horizon_windows(
        test_series, look_back, horizon_steps, reference_series=test_ref
    )
    if min(x_train.shape[0], x_val.shape[0], x_test.shape[0]) <= 0:
        raise ValueError("Insufficient price data to train the ANN short-horizon forecast bank.")

    results = []
    n_total = len(series)
    train_cut = int(n_total * float(train_ratio))
    val_cut = train_cut + int(n_total * float(val_ratio))
    training_start = raw_dates[0] if raw_dates else None
    training_end = raw_dates[max(0, train_cut - 1)] if raw_dates and train_cut > 0 else None
    validation_end = raw_dates[max(0, min(val_cut, n_total) - 1)] if raw_dates and val_cut > 0 else None
    test_end = raw_dates[-1] if raw_dates else None
    print("[PRICE_SHORT_EXPERT_BANK] Training 1 expert: ANN")
    try:
        result = _train_ann_expert(
            episode_dir,
            x_train,
            y_train,
            a_train,
            x_val,
            y_val,
            a_val,
            x_test,
            y_test,
            a_test,
            int(look_back),
            int(horizon_steps),
            int(seed),
        )
    except Exception as exc:
        result = {"method": "ann", "success": False, "error": str(exc)}
    status = "OK" if result.get("success") else f"FAILED ({result.get('error', 'unknown')})"
    print(f"  [1/1] ANN: {status}", flush=True)
    if result.get("success"):
        try:
            paths = get_price_short_expert_paths(episode_dir, "ann")
            with open(paths["metadata_path"], "r", encoding="utf-8-sig") as fh:
                metadata = json.load(fh)
            metadata["training_start"] = training_start
            metadata["training_end"] = training_end
            metadata["validation_end"] = validation_end
            metadata["test_end"] = test_end
            metadata["training_price_mean"] = float(np.mean(train_series))
            metadata["entry_reference_column"] = str(reference_column)
            metadata["forecast_information_set"] = (
                "settlement_history_strictly_before_origin_plus_current_day_ahead_entry"
            )
            metadata["direction_target"] = "same_delivery_settlement_minus_entry_index"
            metadata["target_delivery_alignment"] = "same_delivery_hour"
            with open(paths["metadata_path"], "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)
        except Exception as exc:
            result = {
                "method": "ann",
                "success": False,
                "error": f"metadata_update_failed: {exc}",
            }
    results.append(result)
    successful = [r for r in results if r.get("success")]
    return {
        "episode_num": int(episode_num),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "results": results,
    }

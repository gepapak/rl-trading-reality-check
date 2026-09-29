import json
import os
import numpy as np
import pandas as pd
import logging
from utils import _get_tf

logger = logging.getLogger(__name__)

# =========================
# TensorFlow setup (optional)
# =========================


def _is_capacity_factor_data(df: pd.DataFrame) -> bool:
    """Detect whether renewable generation columns look like capacity factors."""
    renewable_cols = ["wind", "solar", "hydro"]
    for col in renewable_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            max_val = float(np.nanmax(s.values)) if len(s) else 0.0
            if max_val > 2.0:
                return False
    return True


def _convert_to_raw_mw_values(df: pd.DataFrame, config=None, mw_scale_overrides=None) -> pd.DataFrame:
    """Convert capacity-factor data to raw MW values for direct forecasting."""
    if config and hasattr(config, "mw_conversion_scales"):
        capacity_mw = config.mw_conversion_scales.copy()
    else:
        capacity_mw = {
            "wind": 1103,
            "solar": 100,
            "hydro": 534,
            "load": 2999,
        }

    if mw_scale_overrides:
        for key, value in mw_scale_overrides.items():
            if value is not None and key in capacity_mw:
                capacity_mw[key] = float(value)
                logger.info("[OVERRIDE] Using CLI override for %s: %s MW", key, value)

    df_converted = df.copy()
    logger.info("[INFO] Converting to raw MW values (no normalization):")

    def _looks_like_capacity_factor(series: pd.Series) -> bool:
        s = pd.to_numeric(series, errors="coerce")
        if len(s) == 0:
            return False
        max_val = float(np.nanmax(s.values))
        min_val = float(np.nanmin(s.values))
        return (max_val <= 2.0) and (min_val >= -0.1)

    for col, capacity in capacity_mw.items():
        if col in df_converted.columns:
            original_range = f"[{df[col].min():.3f}, {df[col].max():.3f}]"
            if _looks_like_capacity_factor(df[col]):
                df_converted[col] = df[col] * capacity
                new_range = f"[{df_converted[col].min():.1f}, {df_converted[col].max():.1f}] MW"
                logger.info("  %s: %s -> %s", col, original_range, new_range)
                logger.info("  %s: %s (kept as-is; already looks like MW)", col, original_range)

    if "price" in df_converted.columns:
        price_range = f"[{df_converted['price'].min():.1f}, {df_converted['price'].max():.1f}] $/MWh"
        logger.info("  price: %s (no conversion)", price_range)

    logger.info("[OK] Raw MW conversion complete - ready for direct forecasting")
    return df_converted


def load_energy_data(csv_path: str, convert_to_raw_units: bool = True, config=None, mw_scale_overrides=None) -> pd.DataFrame:
    """
    Load energy time series data from CSV with optional MW conversion.

    Requires at least: wind, solar, hydro, price, load.
    Keeps extra columns if present and parses timestamp when possible.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif {"date", "time"}.issubset(df.columns):
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce",
        )

    required = ["wind", "solar", "hydro", "price", "load"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    numeric_extra = [c for c in ["risk", "revenue", "battery_energy", "npv"] if c in df.columns]
    for col in required + numeric_extra:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)

    if convert_to_raw_units and _is_capacity_factor_data(df):
        logger.info("[INFO] Converting capacity factors to raw MW values for direct forecasting...")
        df = _convert_to_raw_mw_values(df, config=config, mw_scale_overrides=mw_scale_overrides)
        logger.info("[OK] Raw MW conversion completed - forecasts will work directly with these units")
    else:
        logger.info("[INFO] Data already in raw MW units - ready for direct forecasting")

    return df


class MultiHorizonForecastGenerator:
    """
    Compatibility wrapper for the active ANN short-horizon forecast bank.

    The newer forecast_engine.py expects a generator object with
    is_complete_stack() and precompute_offline(). The current project only uses
    the ANN short price expert columns consumed by forecast_prior.py, so this
    wrapper keeps the legacy interface while delegating prediction to
    forecast_price_experts.PriceShortExpertBank.
    """

    def __init__(
        self,
        model_dir: str,
        scaler_dir: str,
        metadata_dir: str,
        look_back: int = 24,
        expert_refresh_stride: int = 6,
        verbose: bool = False,
        fallback_mode: bool = False,
        config=None,
    ):
        self.model_dir = str(model_dir)
        self.scaler_dir = str(scaler_dir)
        self.metadata_dir = str(metadata_dir)
        self.look_back = int(look_back)
        self.expert_refresh_stride = int(expert_refresh_stride)
        self.verbose = bool(verbose)
        self.fallback_mode = bool(fallback_mode)
        self.config = config
        self.episode_dir = os.path.dirname(os.path.abspath(self.model_dir))
        self._bank = None

    def _load_bank(self):
        if self._bank is None:
            from forecast_price_experts import PriceShortExpertBank

            horizon_steps = 6
            if self.config is not None:
                horizons = getattr(self.config, "forecast_horizons", {}) or {}
                if isinstance(horizons, dict):
                    horizon_steps = int(horizons.get("short", horizon_steps) or horizon_steps)
            self._bank = PriceShortExpertBank(
                episode_dir=self.episode_dir,
                look_back=self.look_back,
                horizon_steps=horizon_steps,
                verbose=self.verbose,
                refresh_stride=self.expert_refresh_stride,
            )
        return self._bank

    def is_complete_stack(self) -> bool:
        try:
            bank = self._load_bank()
            return bool(bank.is_complete())
        except Exception as exc:
            if self.verbose:
                logger.warning("Forecast expert stack is incomplete: %s", exc)
            return False

    @staticmethod
    def _cache_stem(df: pd.DataFrame, timestamp_col: str) -> str:
        if timestamp_col in df.columns:
            ts = pd.to_datetime(df[timestamp_col], errors="coerce")
            if ts.notna().any():
                start = ts.min().strftime("%Y%m%d")
                end = ts.max().strftime("%Y%m%d")
                return f"precomputed_forecasts_{start}_to_{end}_{len(df)}rows"
        return f"precomputed_forecasts_{len(df)}rows"

    def precompute_offline(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        batch_size: int = 8192,
        cache_dir: str = "forecast_cache",
    ) -> str:
        del batch_size
        if "price" not in df.columns:
            raise ValueError("Forecast precompute requires a 'price' column")
        bank = self._load_bank()
        if not bank.is_complete():
            raise RuntimeError(f"Forecast expert stack is incomplete: {self.episode_dir}")

        prices = pd.to_numeric(df["price"], errors="coerce").to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(prices)):
            raise ValueError("Forecast precompute found non-finite settlement prices; causal filling is not permitted")
        entry_column = next(
            (name for name in ("entry_price_index", "energy_index_price") if name in df.columns),
            None,
        )
        if entry_column is None:
            raise ValueError(
                "ANN v3 forecast precompute requires 'entry_price_index' or 'energy_index_price'"
            )
        entry_prices = pd.to_numeric(df[entry_column], errors="coerce").to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(entry_prices)):
            raise ValueError(f"Forecast precompute found non-finite values in {entry_column}")
        details = bank.precompute_details_for_series(prices, entry_series=entry_prices)
        ann = details.get("ann", {})
        n = int(len(df))

        def arr(name: str, default: float = 0.0) -> np.ndarray:
            values = np.asarray(ann.get(name, np.full(n, default, dtype=np.float32)), dtype=np.float32).reshape(-1)
            if values.size != n:
                out = np.full(n, default, dtype=np.float32)
                out[: min(n, values.size)] = values[: min(n, values.size)]
                values = out
            return values

        out = pd.DataFrame()
        if timestamp_col in df.columns:
            out["timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            out["timestamp"] = np.arange(n, dtype=np.int64)
        out["price_forecast_short"] = arr("pred_price")
        out["price_short_expert_ann"] = arr("pred_price")
        out["price_short_expert_ann_pred_return"] = arr("pred_return")
        out["price_short_expert_ann_direction_prob"] = arr("direction_prob", 0.5)
        out["price_short_expert_ann_direction_margin"] = arr("direction_margin")
        out["price_short_expert_ann_uncertainty"] = arr("uncertainty", 1.0)
        out["price_short_expert_ann_quality"] = arr("quality")
        out["price_short_expert_ann_latent_norm"] = arr("latent_norm")
        for idx in range(4):
            out[f"price_short_expert_ann_latent_{idx}"] = arr(f"latent_{idx}")

        os.makedirs(cache_dir, exist_ok=True)
        stem = self._cache_stem(df, timestamp_col)
        csv_path = os.path.join(cache_dir, f"{stem}.csv")
        meta_path = os.path.join(cache_dir, f"{stem}_metadata.json")
        out.to_csv(csv_path, index=False)

        metadata = {
            "look_back": int(self.look_back),
            "targets": ["price"],
            "horizons": ["short"],
            "horizon_offsets": {"short": int(bank.horizon_steps)},
            "forecast_alignment": "origin_timestamp",
            "forecast_information_set": "strictly_before_origin_plus_current_day_ahead_entry",
            "target_delivery_alignment": "same_delivery_hour",
            "target_alignment_version": "same_delivery_causal_v3",
            "entry_reference_column": str(entry_column),
            "price_short_expert_version": str(
                bank.artifacts.get("ann").metadata.get("version", "")
            ),
            "rows": n,
            "forecast_model_dir": self.episode_dir,
            "expert_model_paths": {
                "ann": os.path.join(self.episode_dir, "price_short_experts", "ANN", "ann_model.keras")
            },
            "cache_csv": csv_path,
        }
        try:
            metadata["expert_cache_validation"] = bank.get_cache_validation_info()
        except Exception:
            pass
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        return csv_path

# LAZY TensorFlow initialization (use _get_tf() from utils)
# Suppress TensorFlow warnings globally
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import logging as _logging
    _logging.getLogger("tensorflow").setLevel(_logging.ERROR)
except Exception:
    pass

def initialize_tensorflow(device="cuda"):
    """Initialize TensorFlow based on device setting with enhanced memory management (lazy)."""
    use_gpu = device.lower() == "cuda"
    tf = _get_tf()  # Lazy initialization from utils

    if tf is None:
        logging.warning("TensorFlow not available, cannot initialize")
        return None

    # Configure TensorFlow for memory efficiency - CONSERVATIVE APPROACH
    try:
        # HIGH: Fix GPU Initialization Conflict - rely solely on memory_growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and use_gpu:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Remove hardcoded VirtualDeviceConfiguration(memory_limit=X) to prevent instability
        logging.info(f"TensorFlow configured for {'GPU' if use_gpu else 'CPU'}-only mode with dynamic memory allocation")
    except Exception as e:
        logging.warning(f"Failed to configure TensorFlow memory settings: {e}")

    return tf


"""Build a trivial slope-based forecast cache with the ANN-prior schema.

This is a robustness baseline: it replaces the ANN cache with a causal
price-slope signal while keeping the existing conformal-prior execution path.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def build_slope_cache(
    *,
    eval_data: Path,
    output_cache_dir: Path,
    horizon_steps: int = 6,
    denom_floor: float = 10.0,
    return_scale: float = 1.0,
    margin_gain: float = 4.0,
    uncertainty: float = 0.10,
    quality: float = 0.80,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if not eval_data.is_file():
        raise FileNotFoundError(f"eval_data not found: {eval_data}")

    cache_dir = output_cache_dir / "forecast_cache_eval_episode20_2025" / "forecast_cache_eval_episode20_2025-full"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_csv = cache_dir / f"precomputed_forecasts_slope_prior_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out_meta = Path(str(out_csv).replace(".csv", "_metadata.json"))
    if out_csv.exists() and not overwrite:
        raise FileExistsError(f"cache CSV exists: {out_csv}")

    df = pd.read_csv(eval_data, parse_dates=["timestamp"])
    if "price" not in df.columns:
        raise ValueError(f"{eval_data} must contain a price column")

    price = pd.to_numeric(df["price"], errors="coerce").to_numpy(dtype=float)
    n = len(price)
    h = max(1, int(horizon_steps))

    prev = np.r_[np.repeat(price[:1], h), price[:-h]]
    denom = np.maximum(np.abs(prev), float(denom_floor))
    slope_ret = (price - prev) / denom
    pred_return = np.clip(float(return_scale) * slope_ret, -0.25, 0.25)
    margin = np.clip(np.tanh(float(margin_gain) * pred_return), -1.0, 1.0)
    direction_prob = np.clip(0.5 + 0.5 * margin, 0.0, 1.0)
    pred_price = price * (1.0 + pred_return)

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "price_forecast_short": pred_price.astype(np.float32),
            "price_short_expert_ann": pred_price.astype(np.float32),
            "price_short_expert_ann_pred_return": pred_return.astype(np.float32),
            "price_short_expert_ann_direction_prob": direction_prob.astype(np.float32),
            "price_short_expert_ann_direction_margin": margin.astype(np.float32),
            "price_short_expert_ann_uncertainty": np.full(n, float(uncertainty), dtype=np.float32),
            "price_short_expert_ann_quality": np.full(n, float(quality), dtype=np.float32),
            "price_short_expert_ann_latent_norm": np.zeros(n, dtype=np.float32),
            "price_short_expert_ann_latent_0": np.zeros(n, dtype=np.float32),
            "price_short_expert_ann_latent_1": np.zeros(n, dtype=np.float32),
            "price_short_expert_ann_latent_2": np.zeros(n, dtype=np.float32),
            "price_short_expert_ann_latent_3": np.zeros(n, dtype=np.float32),
        }
    )
    out.to_csv(out_csv, index=False)

    metadata: Dict[str, Any] = {
        "look_back": int(horizon_steps),
        "targets": ["price"],
        "horizons": ["short"],
        "horizon_offsets": {"short": int(horizon_steps)},
        "forecast_alignment": "origin_timestamp",
        "forecast_tails": {"price": {"short": []}},
        "model_modification_times": {},
        "model_paths": {},
        "expert_model_modification_times": {},
        "expert_model_paths": {},
        "expert_metadata_modification_times": {},
        "expert_metadata_paths": {},
        "expert_refresh_stride": int(horizon_steps),
        "cache_created": datetime.now().isoformat(timespec="seconds"),
        "slope_prior_baseline": {
            "eval_data": str(eval_data),
            "horizon_steps": int(horizon_steps),
            "denom_floor": float(denom_floor),
            "return_scale": float(return_scale),
            "margin_gain": float(margin_gain),
            "uncertainty": float(uncertainty),
            "quality": float(quality),
        },
    }
    out_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    future = price[h:] - price[: max(0, n - h)]
    pred = pred_return[: max(0, n - h)]
    pred_s = np.sign(pred)
    real_s = np.sign(future)
    mask = np.isfinite(pred_s) & np.isfinite(real_s) & (pred_s != 0.0) & (real_s != 0.0)
    hit = float((pred_s[mask] == real_s[mask]).mean()) if int(mask.sum()) else None

    return {
        "eval_data": str(eval_data),
        "cache_dir": str(cache_dir),
        "csv": str(out_csv),
        "metadata": str(out_meta),
        "rows": int(n),
        "nonzero_signal_frac": float(np.mean(pred_return != 0.0)),
        "horizon_sign_hit": hit,
        "horizon_sign_n": int(mask.sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a slope-prior forecast cache for robustness evaluation.")
    parser.add_argument("--eval_data", default="robustness_ffill/evaluation_dataset/unseendata.csv")
    parser.add_argument("--output_cache_dir", default="robustness_ffill_slope/forecast_cache")
    parser.add_argument("--horizon_steps", type=int, default=6)
    parser.add_argument("--denom_floor", type=float, default=10.0)
    parser.add_argument("--return_scale", type=float, default=1.0)
    parser.add_argument("--margin_gain", type=float, default=4.0)
    parser.add_argument("--uncertainty", type=float, default=0.10)
    parser.add_argument("--quality", type=float, default=0.80)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_slope_cache(
        eval_data=_resolve(args.eval_data),
        output_cache_dir=_resolve(args.output_cache_dir),
        horizon_steps=int(args.horizon_steps),
        denom_floor=float(args.denom_floor),
        return_scale=float(args.return_scale),
        margin_gain=float(args.margin_gain),
        uncertainty=float(args.uncertainty),
        quality=float(args.quality),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

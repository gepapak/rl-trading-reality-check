#!/usr/bin/env python3
"""Run final deterministic and temporal baselines for both price regions."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from protocol_preflight import assert_final_data_contract


PROJECT_ROOT = Path(__file__).resolve().parent
MARL_SUITE_DIR = "batch_tier_phase_runs/prototype5_mappo_marl_final_v1"
FOCAL_SUITE_DIR = (
    "Ablations/batch_tier_phase_runs/"
    "prototype5_mechanism_ablations_final_v1/focal_anchor"
)
RESULT_ROOT = "results/prototype5_baselines_final_v1"
BASELINE_OUTPUT_ROOT = "baseline_results/prototype5_final_paper_baselines_v1"
TEMPORAL_BASELINE_OUTPUT_ROOT = "baseline_results/prototype5_temporal_baselines_v1"

# Harmonized EU balancing-energy technical price bounds (±15,000 EUR/MWh at
# ~7.45 DKK/EUR). Wider than every observed raw settlement print, so the bound
# acts as a technical guard rather than a distribution-shaping clip.
SETTLEMENT_CLIP_MIN_DKK_DEFAULT = -111750.0
SETTLEMENT_CLIP_MAX_DKK_DEFAULT = 111750.0
MARKET_FEE_ARGS = [
    "--market_fee_model", "nord_pool_intraday_2026",
    "--transaction_fee_dkk_per_mwh", "0.9238",
    "--annual_market_access_fee_dkk", "160175.0",
    "--market_access_fee_allocation_fraction", "1.0",
    "--market_fee_source_id", "nord_pool_nordic_baltic_2026_standard_participant",
]


def _force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


_force_utf8_stdio()


REGIONS = {
    "original": {
        "eval_data": "evaluation_dataset_ffill/unseendata.csv",
    },
    "unseendata_v2": {
        "eval_data": "evaluation_dataset_ffill/unseendata_v2.csv",
    },
}




def _run(cmd: list[str], label: str, log_path: Path, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = subprocess.list2cmdline(cmd)
    print("")
    print("=" * 100)
    print(f"[RUN] {label}")
    print(line)
    print(f"Log: {log_path}")
    print("=" * 100)
    if dry_run:
        return
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] START {label}\n{line}\n")
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for out_line in process.stdout:
            print(out_line, end="", flush=True)
            log.write(out_line)
            log.flush()
        rc = process.wait()
        log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] END {label} exit={rc}\n")
    if rc != 0:
        raise RuntimeError(f"{label} failed with exit code {rc}. See {log_path}")


def _assert_settlement_clip_consistency(args: argparse.Namespace) -> None:
    """Refuse to run when settlement files were built with a tighter clip than configured."""
    if bool(getattr(args, "energy_index_settlement", False)) or bool(getattr(args, "basis_risk", False)):
        return
    import pandas as pd

    clip_min = float(args.settlement_clip_min_dkk)
    clip_max = float(args.settlement_clip_max_dkk)
    candidates = [
        PROJECT_ROOT / str(args.settlement_eval_original),
        PROJECT_ROOT / str(args.settlement_eval_v2),
    ]
    stale = []
    for path in candidates:
        if not path.is_file():
            continue
        prices = pd.to_numeric(
            pd.read_csv(path, usecols=[str(args.settlement_price_column)])[str(args.settlement_price_column)],
            errors="coerce",
        ).dropna()
        if prices.empty:
            continue
        lo, hi = float(prices.min()), float(prices.max())
        at_legacy_bounds = abs(lo + 1000.0) < 1e-6 or abs(hi - 10000.0) < 1e-6
        if at_legacy_bounds and (clip_min < -1000.0 - 1e-6 or clip_max > 10000.0 + 1e-6):
            stale.append(f"  {path} (min={lo:.1f}, max={hi:.1f})")
    if stale:
        raise RuntimeError(
            "Settlement files below still carry the legacy [-1000, 10000] DKK clip but the "
            f"protocol now uses [{clip_min:.0f}, {clip_max:.0f}]. Rebuild them first:\n"
            + "\n".join(stale)
            + "\n\n  python scripts/build_real_settlement_protocol_v2.py "
            + f"--settlement_clip_min {clip_min:.0f} --settlement_clip_max {clip_max:.0f} --overwrite\n"
            "Then retrain the forecast models and rebuild the hourly forecast cache "
            "(scripts/train_hourly_forecast_models_v2.py, scripts/precompute_hourly_forecast_cache_v2.py) "
            "because the forecast targets share the settlement clip."
        )


def _assert_liquidity_volume_files(args: argparse.Namespace) -> None:
    candidates = [
        PROJECT_ROOT / str(args.liquidity_volume_eval_original),
        PROJECT_ROOT / str(args.liquidity_volume_eval_v2),
    ]
    missing = [str(path) for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Real liquidity-volume files are missing:\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\n\nBuild them first with:\n"
            "  python scripts/build_real_liquidity_volume_protocol_v1.py --overwrite"
        )


def _run_baselines(args: argparse.Namespace, logs_dir: Path) -> None:
    cmd = [
        args.python,
        "scripts/run_final_paper_baselines.py",
        "--output_root", args.baseline_output_root,
        "--timesteps", "39311",
        "--seed", str(args.seed),
    ]
    if bool(getattr(args, "basis_risk", False)):
        cmd.extend(
            [
                "--basis_risk",
                "--basis_eval_original", str(args.basis_eval_original),
                "--basis_eval_v2", str(args.basis_eval_v2),
                "--basis_scale", str(float(args.basis_scale)),
                "--basis_centering_mode", str(args.basis_centering_mode),
                "--basis_centering_window_steps", str(int(args.basis_centering_window_steps)),
            ]
        )
    elif bool(getattr(args, "energy_index_settlement", False)):
        cmd.append("--energy_index_settlement")
    else:
        cmd.extend(
            [
                "--settlement_eval_original", str(args.settlement_eval_original),
                "--settlement_eval_v2", str(args.settlement_eval_v2),
                "--settlement_price_column", str(args.settlement_price_column),
                "--settlement_timestamp_column", str(args.settlement_timestamp_column),
                "--settlement_clip_min_dkk", str(float(args.settlement_clip_min_dkk)),
                "--settlement_clip_max_dkk", str(float(args.settlement_clip_max_dkk)),
                "--liquidity_volume_eval_original", str(args.liquidity_volume_eval_original),
                "--liquidity_volume_eval_v2", str(args.liquidity_volume_eval_v2),
                "--liquidity_volume_column", str(args.liquidity_volume_column),
                "--liquidity_volume_unit", str(args.liquidity_volume_unit),
                "--liquidity_volume_timestamp_column", str(args.liquidity_volume_timestamp_column),
                "--liquidity_volume_max_staleness_min", str(float(args.liquidity_volume_max_staleness_min)),
                "--impact_volume_price_floor_dkk_per_mwh", str(float(args.impact_volume_price_floor_dkk_per_mwh)),
            ]
        )
    _run(cmd, "Final baselines on both price regions", logs_dir / "final_baselines.log", args.dry_run)


def _evaluation_region(path: Path, payload: dict) -> str:
    eval_data = str(payload.get("eval_data", "") or "").replace("\\", "/").lower()
    path_text = str(path).replace("\\", "/").lower()
    if "unseendata_v2" in eval_data or "evaluations_2025_v2" in path_text:
        return "unseendata_v2"
    return "original"


def _first_tier(payload: dict) -> dict:
    tiers = payload.get("tiers", {})
    if isinstance(tiers, dict):
        for tier in tiers.values():
            if isinstance(tier, dict):
                return tier
    return {}


def _mean_completed_exposure_dkk(suite_dir: Path, region: str) -> tuple[float | None, int]:
    # A seed can have multiple timestamped evaluation JSONs after a resumed or
    # repeated evaluation. Use only the newest completed result per seed.
    latest_by_seed: dict[int, tuple[float, float]] = {}
    if not suite_dir.exists():
        return None, 0
    for path in suite_dir.rglob("evaluation_tiers_*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        if _evaluation_region(path, payload) != region:
            continue
        tier = _first_tier(payload)
        if str(tier.get("status", "completed")).lower() not in {"completed", ""}:
            continue
        sleeve = tier.get("sleeve_metrics", {})
        raw = sleeve.get("sleeve_mean_abs_exposure_dkk")
        try:
            value = float(raw)
        except Exception:
            continue
        if math.isfinite(value) and value > 0.0:
            try:
                seed = int(payload.get("seed"))
            except Exception:
                match = re.search(r"seed(\d+)", str(path).replace("\\", "/"), flags=re.IGNORECASE)
                if match is None:
                    continue
                seed = int(match.group(1))
            mtime = float(path.stat().st_mtime)
            previous = latest_by_seed.get(seed)
            if previous is None or mtime > previous[0]:
                latest_by_seed[seed] = (mtime, value)
    values = [item[1] for item in latest_by_seed.values()]
    if not values:
        return None, 0
    return float(sum(values) / len(values)), len(values)


def _protocol_max_notional_dkk() -> float:
    from config import EnhancedConfig

    cfg = EnhancedConfig()
    initial_fund_dkk = float(getattr(cfg, "init_budget", 0.0))
    financial_allocation = float(getattr(cfg, "financial_allocation", 0.12))
    initial_sleeve_dkk = initial_fund_dkk * financial_allocation
    return initial_sleeve_dkk * 0.60 * 0.10


def _temporal_target_norm(args: argparse.Namespace, profile: str, region: str) -> tuple[float, str]:
    if profile == "stress":
        norm = float(args.temporal_rule_target_norm)
        return norm, f"fixed stress norm={norm:.8f}"

    if profile == "focal_matched":
        suite_dir = PROJECT_ROOT / args.temporal_focal_suite_dir
        fallback = float(args.temporal_focal_fallback_norm)
        label = "FoCAL"
    elif profile == "marl_matched":
        suite_dir = PROJECT_ROOT / args.temporal_marl_suite_dir
        fallback = float(args.temporal_marl_fallback_norm)
        label = "MARL"
    else:
        raise ValueError(f"Unknown temporal profile: {profile}")

    exposure, count = _mean_completed_exposure_dkk(suite_dir, region)
    if exposure is None:
        if not (bool(args.allow_temporal_match_fallback) or bool(args.dry_run)):
            raise RuntimeError(
                f"Cannot build temporal {profile} baseline for {region}: no completed "
                f"evaluation JSONs with sleeve_mean_abs_exposure_dkk under {suite_dir}. "
                "Run the matched MARL/FoCAL evaluations first, or pass "
                "--allow_temporal_match_fallback to use the configured fallback norm."
            )
        return fallback, f"{label} fallback norm={fallback:.8f}"

    max_notional = _protocol_max_notional_dkk()
    if max_notional <= 0.0:
        raise RuntimeError("Protocol max notional is non-positive; cannot match temporal exposure.")
    norm = exposure / max_notional
    norm = max(float(args.temporal_min_target_norm), min(float(args.temporal_max_target_norm), norm))
    return norm, f"{label}-matched mean_abs_exposure={exposure:,.0f} DKK from n={count}"


def _run_temporal_baselines(args: argparse.Namespace, logs_dir: Path) -> None:
    for region, spec in REGIONS.items():
        for profile in args.temporal_profiles:
            target_norm, match_note = _temporal_target_norm(args, profile, region)
            cmd = [
                args.python,
                "scripts/evaluate_temporal_price_baselines.py",
                "--eval_data", str(spec["eval_data"]),
                "--output_dir", str(Path(args.temporal_baseline_output_root) / profile / region),
                "--rules", "prev_hour_momentum", "same_hour_yesterday", "combined_agreement",
                "--seed", str(args.seed),
                "--decision_freq", "6",
                "--horizon_steps", "6",
                "--capital_allocation_fraction", "0.6",
                "--max_position_size", "0.1",
                "--rule_target_norm", f"{target_norm:.12g}",
                "--payoff_reference_price", "500",
                "--payoff_mode", "mwh_volume",
                "--friction_cost_multiplier", "1.0",
                *MARKET_FEE_ARGS,
                "--no_trade_threshold", "0.01",
                "--half_spread_bp", "5.0",
                "--impact_coef_bp", "20.0",
                "--impact_exponent", "0.5",
                "--liquidity_participation_cap_fraction", "0.25",
                "--liquidity_volume_source", "impact_volume",
                "--impact_volume_data", str(args.liquidity_volume_eval_v2 if region == "unseendata_v2" else args.liquidity_volume_eval_original),
                "--impact_volume_column", str(args.liquidity_volume_column),
                "--impact_volume_unit", str(args.liquidity_volume_unit),
                "--impact_volume_timestamp_column", str(args.liquidity_volume_timestamp_column),
                "--impact_volume_max_staleness_min", str(float(args.liquidity_volume_max_staleness_min)),
                "--impact_volume_price_floor_dkk_per_mwh", str(float(args.impact_volume_price_floor_dkk_per_mwh)),
                "--liquidity_volume_multiplier", "1.0",
                "--liquidity_min_volume_mwh", "1.0",
                "--liquidity_tail_impact_threshold_dkk_per_mwh", "5000.0",
                "--liquidity_tail_impact_multiplier", "3.0",
                "--liquidity_tail_impact_power", "1.0",
                "--liquidity_tail_impact_max_multiplier", "10.0",
                "--enable_collateral_cash_drag",
                "--collateral_notional_margin_fraction", "0.02",
                "--collateral_stress_loss_fraction", "0.10",
                "--collateral_stress_price_dkk_per_mwh", "25000.0",
                "--collateral_funding_rate_annual", "0.05",
                "--collateral_tradeable_haircut", "1.0",
            ]
            if bool(getattr(args, "basis_risk", False)):
                basis_path = args.basis_eval_v2 if region == "unseendata_v2" else args.basis_eval_original
                cmd.extend(
                    [
                        "--settlement_price_data", str(basis_path),
                        "--basis_scale", str(float(args.basis_scale)),
                        "--basis_centering_mode", str(args.basis_centering_mode),
                        "--basis_centering_window_steps", str(int(args.basis_centering_window_steps)),
                    ]
                )
            elif not bool(getattr(args, "energy_index_settlement", False)):
                settlement_path = args.settlement_eval_v2 if region == "unseendata_v2" else args.settlement_eval_original
                cmd.extend(
                    [
                        "--settlement_price_data", str(settlement_path),
                        "--settlement_price_mode", "external_series",
                        "--settlement_price_column", str(args.settlement_price_column),
                        "--settlement_timestamp_column", str(args.settlement_timestamp_column),
                        "--settlement_clip_min_dkk", str(float(args.settlement_clip_min_dkk)),
                        "--settlement_clip_max_dkk", str(float(args.settlement_clip_max_dkk)),
                    ]
                )
            _run(
                cmd,
                f"Temporal price-direction baselines for {region} ({profile}; {match_note})",
                logs_dir / f"temporal_baselines_{profile}_{region}.log",
                args.dry_run,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", default=RESULT_ROOT)
    parser.add_argument("--baseline_output_root", default=BASELINE_OUTPUT_ROOT)
    parser.add_argument("--temporal_baseline_output_root", default=TEMPORAL_BASELINE_OUTPUT_ROOT)
    parser.add_argument(
        "--basis_risk",
        action="store_true",
        help="Use cross-zone basis/tracking-error settlement instead of real external settlement.",
    )
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
    parser.add_argument(
        "--basis_eval_original",
        default="evaluation_dataset_ffill/unseendata_v2.csv",
        help="Peer price file used when evaluating the original DK1 evaluation dataset.",
    )
    parser.add_argument(
        "--basis_eval_v2",
        default="evaluation_dataset_ffill/unseendata.csv",
        help="Peer price file used when evaluating the DK2 evaluation dataset.",
    )
    parser.add_argument("--basis_scale", type=float, default=0.50)
    parser.add_argument(
        "--basis_centering_mode",
        choices=["none", "rolling_median", "expanding_median"],
        default="rolling_median",
    )
    parser.add_argument("--basis_centering_window_steps", type=int, default=4320)
    parser.add_argument(
        "--temporal_rule_target_norm",
        type=float,
        default=0.125,
        help=(
            "Temporal-rule notional as a fraction of the final protocol max notional. "
            "Used only by the stress profile unless matching fallbacks are enabled."
        ),
    )
    parser.add_argument(
        "--temporal_profiles",
        nargs="+",
        choices=["focal_matched", "marl_matched", "stress"],
        default=["focal_matched", "marl_matched"],
        help=(
            "Temporal baseline exposure profiles. focal_matched and marl_matched read completed "
            "evaluation JSONs and match sleeve_mean_abs_exposure_dkk region-by-region. stress keeps "
            "the fixed --temporal_rule_target_norm protocol."
        ),
    )
    parser.add_argument("--temporal_focal_suite_dir", default=FOCAL_SUITE_DIR)
    parser.add_argument("--temporal_marl_suite_dir", default=MARL_SUITE_DIR)
    parser.add_argument("--temporal_min_target_norm", type=float, default=0.0)
    parser.add_argument("--temporal_max_target_norm", type=float, default=1.0)
    parser.add_argument(
        "--allow_temporal_match_fallback",
        action="store_true",
        help="Allow fallback target norms if the matched MARL/FoCAL evaluation JSONs are unavailable.",
    )
    parser.add_argument("--temporal_focal_fallback_norm", type=float, default=0.004)
    parser.add_argument("--temporal_marl_fallback_norm", type=float, default=0.05)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_temporal_baselines", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assert_final_data_contract(PROJECT_ROOT)
    result_root = PROJECT_ROOT / args.result_root
    logs_dir = result_root / "logs"
    if not args.dry_run:
        _assert_settlement_clip_consistency(args)
        _assert_liquidity_volume_files(args)

    if not args.skip_baselines:
        _run_baselines(args, logs_dir)
    if not args.skip_temporal_baselines:
        _run_temporal_baselines(args, logs_dir)

    print("\nDone. Baseline outputs:")
    print("  baselines:", PROJECT_ROOT / args.baseline_output_root)
    print("  temporal baselines:", PROJECT_ROOT / args.temporal_baseline_output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

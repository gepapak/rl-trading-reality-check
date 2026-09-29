#!/usr/bin/env python3
"""Run Markowitz/capped-long exposure sensitivity checks for Prototype5.

This is a paper hygiene runner, not a replacement for the main baselines.
It evaluates whether the strong Markowitz/capped-long result is driven by
larger passive tail exposure by rerunning the same strict real-settlement
ledger with:

- the original 12% price-sleeve exposure;
- the same tail-loss budget used by FoCAL;
- FoCAL mean-exposure and max-exposure matched caps;
- maintenance-margin sensitivity.

The script reuses the frozen data/accounting protocol so settlement,
costs, sizing base, horizon, and margin mechanics match the reported
baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for extra in (
    PROJECT_ROOT / "baselines",
    PROJECT_ROOT / "baselines" / "Baseline1_TraditionalPortfolio",
):
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

import run_final_paper_baselines as final_baselines  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "baseline_results" / "prototype5_markowitz_sensitivity_final_v1"
)
DEFAULT_FOCAL_SUITE_DIR = (
    PROJECT_ROOT
    / "Ablations"
    / "batch_tier_phase_runs"
    / "prototype5_mechanism_ablations_final_v1"
    / "focal_anchor"
)
DEFAULT_TIMESTEPS = 39311
DEFAULT_SEED = 42

REGIONS = {
    "original": {
        "eval_data": PROJECT_ROOT / "evaluation_dataset_ffill" / "unseendata.csv",
        "settlement_data": PROJECT_ROOT
        / "evaluation_dataset_ffill"
        / "unseendata_settlement_real_v2.csv",
        "eval_folder": "evaluations_2025",
    },
    "unseendata_v2": {
        "eval_data": PROJECT_ROOT / "evaluation_dataset_ffill" / "unseendata_v2.csv",
        "settlement_data": PROJECT_ROOT
        / "evaluation_dataset_ffill"
        / "unseendata_v2_settlement_real_v2.csv",
        "eval_folder": "evaluations_2025_v2",
    },
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _load_eval_data(path: Path, timesteps: int) -> pd.DataFrame:
    from evaluation import _load_baseline_data

    return _load_baseline_data(str(path), int(timesteps))


def _set_protocol_for_region(
    *,
    region: str,
    margin_fraction: float,
    settlement_clip_min_dkk: float,
    settlement_clip_max_dkk: float,
) -> Dict[str, Any]:
    spec = REGIONS[region]
    final_baselines.FINAL_PROTOCOL["mtm_settlement_price_mode"] = "external_series"
    final_baselines.FINAL_PROTOCOL["mtm_basis_price_data_path"] = ""
    final_baselines.FINAL_PROTOCOL["mtm_external_settlement_price_data_path"] = str(
        spec["settlement_data"]
    )
    final_baselines.FINAL_PROTOCOL["mtm_external_settlement_price_column"] = "settlement_price"
    final_baselines.FINAL_PROTOCOL["mtm_external_settlement_timestamp_column"] = "timestamp"
    final_baselines.FINAL_PROTOCOL[
        "mtm_external_settlement_min_price_dkk_per_mwh"
    ] = float(settlement_clip_min_dkk)
    final_baselines.FINAL_PROTOCOL[
        "mtm_external_settlement_max_price_dkk_per_mwh"
    ] = float(settlement_clip_max_dkk)
    final_baselines.FINAL_PROTOCOL["trading_sleeve_maintenance_margin_fraction"] = float(
        margin_fraction
    )
    return dict(final_baselines.FINAL_PROTOCOL)


def _latest_json(path: Path) -> Optional[Path]:
    files = [p for p in path.glob("evaluation_tiers_*.json") if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _collect_focal_exposure_reference(focal_suite_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Collect completed FoCAL sleeve exposure statistics by evaluation region."""
    out: Dict[str, Dict[str, Any]] = {
        "original": {
            "seed_count": 0,
            "mean_exposure_values_dkk": [],
            "max_exposure_values_dkk": [],
        },
        "unseendata_v2": {
            "seed_count": 0,
            "mean_exposure_values_dkk": [],
            "max_exposure_values_dkk": [],
        },
    }
    if not focal_suite_dir.exists():
        return out

    for region, spec in REGIONS.items():
        eval_folder = str(spec["eval_folder"])
        for seed_dir in sorted(focal_suite_dir.glob("seed*")):
            result_dir = seed_dir / eval_folder / "tier1_forecast_utilization"
            result_path = _latest_json(result_dir)
            if result_path is None:
                continue
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8-sig"))
                sleeve = payload["tiers"]["tier1"].get("sleeve_metrics", {})
            except Exception:
                continue
            mean_exp = _safe_float(sleeve.get("sleeve_mean_abs_exposure_dkk"), math.nan)
            max_exp = _safe_float(sleeve.get("sleeve_max_abs_exposure_dkk"), math.nan)
            if math.isfinite(mean_exp) and mean_exp >= 0.0:
                out[region]["mean_exposure_values_dkk"].append(mean_exp)
            if math.isfinite(max_exp) and max_exp >= 0.0:
                out[region]["max_exposure_values_dkk"].append(max_exp)
            if math.isfinite(mean_exp) or math.isfinite(max_exp):
                out[region]["seed_count"] += 1

    for region, stats in out.items():
        means = stats["mean_exposure_values_dkk"]
        maxes = stats["max_exposure_values_dkk"]
        stats["mean_exposure_dkk"] = float(np.mean(means)) if means else None
        stats["max_exposure_dkk"] = float(np.max(maxes)) if maxes else None
        stats["mean_of_seed_max_exposure_dkk"] = float(np.mean(maxes)) if maxes else None
    return out


def _max_position_notional_dkk(ledger: Any) -> float:
    return float(max(ledger._tradeable_capital_dkk() * ledger.max_position_size, 1.0))


def _tail_budget_cap(
    ledger: Any,
    *,
    tail_loss_budget_fraction: float,
    tail_stress_return: float,
) -> float:
    sleeve_value = float(max(ledger._current_trading_sleeve_value_dkk(), 0.0))
    margin_threshold = float(max(ledger._trading_sleeve_margin_threshold_dkk(), 0.0))
    margin_surplus = float(max(sleeve_value - margin_threshold, 0.0))
    budget_fraction = float(np.clip(tail_loss_budget_fraction, 0.0, 1.0))
    loss_budget = float(min(sleeve_value * budget_fraction, margin_surplus * 0.90))
    denom = _max_position_notional_dkk(ledger) * max(float(tail_stress_return), 1e-9)
    if loss_budget <= 0.0 or denom <= 0.0:
        return 0.0
    return float(np.clip(loss_budget / denom, 0.0, 1.0))


def _summarize_result(
    *,
    region: str,
    arm: str,
    strategy: str,
    result: Dict[str, Any],
    protocol: Dict[str, Any],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    row = final_baselines._normalize_row(region, arm, result)
    row.update(
        {
            "arm": arm,
            "strategy": strategy,
            "margin_fraction": _safe_float(
                protocol.get("trading_sleeve_maintenance_margin_fraction")
            ),
            "settlement_mode": str(protocol.get("mtm_settlement_price_mode", "")),
            "entry_mode": str(protocol.get("mtm_entry_price_mode", "")),
            "payoff_denominator_mode": str(
                protocol.get("mtm_horizon_payoff_denominator_mode", "")
            ),
            **extra,
        }
    )
    return row


def _ledger_metrics_with_extra(ledger: Any, extra: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(ledger.performance_metrics())
    result.update(extra)
    return result


def _run_capped_long(
    data: pd.DataFrame,
    *,
    seed: int,
    cap_fn: Callable[[Any, int], float],
    method: str,
    baseline_id: str,
) -> Dict[str, Any]:
    from baseline_common import HybridFundLedger

    ledger = HybridFundLedger(data, seed=seed)
    decision_count = 0
    exposure_history: List[float] = []
    target_history: List[float] = []
    rebalance_every = max(1, int(ledger.investment_freq))

    for t in range(len(data)):
        target_exposure = None
        if t > 0 and (t % rebalance_every == 0):
            cap = float(np.clip(cap_fn(ledger, t), 0.0, 1.0))
            target_exposure = cap
            decision_count += 1
            target_history.append(cap)
        record = ledger.step(t, target_exposure=target_exposure, battery_action="idle")
        exposure_history.append(float(record.get("current_abs_exposure_dkk", 0.0)))

    return _ledger_metrics_with_extra(
        ledger,
        {
            "baseline_id": baseline_id,
            "method": method,
            "role": "capped_long_sensitivity_reference",
            "status": "completed",
            "rebalance_count": int(decision_count),
            "rebalance_every_steps": int(rebalance_every),
            "mean_target_exposure": float(np.mean(target_history)) if target_history else 0.0,
            "max_target_exposure": float(np.max(target_history)) if target_history else 0.0,
            "mean_abs_exposure_dkk": float(np.mean(exposure_history)) if exposure_history else 0.0,
            "sensitivity_note": (
                "Constant long comparator evaluated with the final Prototype5 "
                "real-settlement ledger and an externally specified exposure cap."
            ),
        },
    )


def _run_markowitz_with_price_cap(
    data: pd.DataFrame,
    *,
    seed: int,
    cap_fn: Callable[[Any, int], float],
    method: str,
    baseline_id: str,
) -> Dict[str, Any]:
    from baseline_common import HybridFundLedger
    from traditional_portfolio_optimizer import (
        OptimizerConfig,
        Timebase,
        TraditionalPortfolioOptimizer,
    )

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
    exposure_history: List[float] = []
    raw_price_weight_history: List[float] = []
    capped_price_weight_history: List[float] = []
    cap_history: List[float] = []
    rebalance_count = 0

    for t in range(len(returns)):
        target_exposure = None
        if t > 0 and (t % rebalance_every == 0):
            start = max(0, t - lookback_steps)
            window = returns.iloc[start:t]
            try:
                w = opt.rebalance_weights(window, method=opt_cfg.method)
                raw_price_weight = float(w.get("price", 0.0))
                cap = float(np.clip(cap_fn(ledger, t), 0.0, 1.0))
                target_exposure = float(np.clip(raw_price_weight, -cap, cap))
                rebalance_count += 1
                raw_price_weight_history.append(raw_price_weight)
                capped_price_weight_history.append(target_exposure)
                cap_history.append(cap)
            except Exception:
                target_exposure = None
        record = ledger.step(t, target_exposure=target_exposure, battery_action="idle")
        exposure_history.append(float(record.get("current_abs_exposure_dkk", 0.0)))

    return _ledger_metrics_with_extra(
        ledger,
        {
            "baseline_id": baseline_id,
            "method": method,
            "role": "markowitz_price_sleeve_sensitivity_reference",
            "status": "completed",
            "rebalance_count": int(rebalance_count),
            "rebalance_every_steps": int(rebalance_every),
            "raw_mean_price_weight": (
                float(np.mean(raw_price_weight_history)) if raw_price_weight_history else 0.0
            ),
            "raw_max_price_weight": (
                float(np.max(raw_price_weight_history)) if raw_price_weight_history else 0.0
            ),
            "mean_target_exposure": (
                float(np.mean(capped_price_weight_history))
                if capped_price_weight_history
                else 0.0
            ),
            "max_target_exposure": (
                float(np.max(capped_price_weight_history))
                if capped_price_weight_history
                else 0.0
            ),
            "mean_exposure_cap": float(np.mean(cap_history)) if cap_history else 0.0,
            "max_exposure_cap": float(np.max(cap_history)) if cap_history else 0.0,
            "mean_abs_exposure_dkk": float(np.mean(exposure_history)) if exposure_history else 0.0,
            "sensitivity_note": (
                "The optimizer is unchanged; only the price sleeve exposure "
                "that maps to financial MTM trades is post-capped. Physical "
                "weights remain diagnostic in the hybrid fund."
            ),
        },
    )


def _constant_cap(value: float) -> Callable[[Any, int], float]:
    fixed = float(np.clip(value, 0.0, 1.0))
    return lambda _ledger, _t: fixed


def _focal_cap_from_reference(
    *,
    region: str,
    match: str,
    focal_reference: Dict[str, Dict[str, Any]],
    max_position_notional: float,
) -> Tuple[float, Dict[str, Any]]:
    stats = focal_reference.get(region, {})
    key = "mean_exposure_dkk" if match == "mean" else "max_exposure_dkk"
    exposure_dkk = stats.get(key)
    if exposure_dkk is None:
        raise RuntimeError(
            f"Cannot run FoCAL-{match}-matched sensitivity for {region}: "
            f"no completed FoCAL exposure metrics found under {DEFAULT_FOCAL_SUITE_DIR}."
        )
    cap = float(np.clip(float(exposure_dkk) / max(max_position_notional, 1.0), 0.0, 1.0))
    return cap, {
        f"focal_reference_{match}_exposure_dkk": float(exposure_dkk),
        "focal_reference_seed_count": int(stats.get("seed_count", 0)),
        "focal_reference_mean_of_seed_max_exposure_dkk": (
            None
            if stats.get("mean_of_seed_max_exposure_dkk") is None
            else float(stats["mean_of_seed_max_exposure_dkk"])
        ),
    }


def _build_arms(
    *,
    region: str,
    data: pd.DataFrame,
    seed: int,
    focal_reference: Dict[str, Dict[str, Any]],
    margins: Iterable[float],
    tail_loss_budget_fraction: float,
    tail_stress_return: float,
    selected_arms: List[str],
) -> List[Dict[str, Any]]:
    from baseline_common import HybridFundLedger

    probe_ledger = HybridFundLedger(data, seed=seed)
    max_position_notional = _max_position_notional_dkk(probe_ledger)
    arms: List[Dict[str, Any]] = []

    def add_pair(
        *,
        arm: str,
        cap_fn: Callable[[Any, int], float],
        extra: Dict[str, Any],
    ) -> None:
        if "all" not in selected_arms and arm not in selected_arms:
            return
        arms.append(
            {
                "arm": arm,
                "strategy": "markowitz",
                "runner": _run_markowitz_with_price_cap,
                "cap_fn": cap_fn,
                "extra": dict(extra),
            }
        )
        arms.append(
            {
                "arm": arm,
                "strategy": "capped_long",
                "runner": _run_capped_long,
                "cap_fn": cap_fn,
                "extra": dict(extra),
            }
        )

    add_pair(
        arm="current_12pct",
        cap_fn=_constant_cap(0.12),
        extra={
            "exposure_cap_mode": "current_12pct",
            "target_cap": 0.12,
            "max_position_notional_dkk": max_position_notional,
        },
    )

    add_pair(
        arm="stress_loss_matched",
        cap_fn=lambda ledger, _t: _tail_budget_cap(
            ledger,
            tail_loss_budget_fraction=tail_loss_budget_fraction,
            tail_stress_return=tail_stress_return,
        ),
        extra={
            "exposure_cap_mode": "dynamic_tail_budget",
            "tail_loss_budget_fraction": float(tail_loss_budget_fraction),
            "tail_stress_return": float(tail_stress_return),
            "max_position_notional_dkk": max_position_notional,
        },
    )

    for match in ("max", "mean"):
        focal_arm = f"focal_{match}_exposure_matched"
        if "all" not in selected_arms and focal_arm not in selected_arms:
            continue
        cap, ref_extra = _focal_cap_from_reference(
            region=region,
            match=match,
            focal_reference=focal_reference,
            max_position_notional=max_position_notional,
        )
        add_pair(
            arm=focal_arm,
            cap_fn=_constant_cap(cap),
            extra={
                "exposure_cap_mode": f"focal_{match}_exposure_matched",
                "target_cap": cap,
                "max_position_notional_dkk": max_position_notional,
                **ref_extra,
            },
        )

    for margin in margins:
        arm = f"margin_{str(float(margin)).replace('.', 'p')}"
        if abs(float(margin) - 0.05) < 1e-12:
            continue
        add_pair(
            arm=arm,
            cap_fn=_constant_cap(0.12),
            extra={
                "exposure_cap_mode": "current_12pct_margin_sweep",
                "target_cap": 0.12,
                "max_position_notional_dkk": max_position_notional,
                "sweep_margin_fraction": float(margin),
            },
        )

    return arms


def _write_outputs(output_root: Path, payload: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_root / f"markowitz_sensitivity_{stamp}.json"
    latest_json = output_root / "markowitz_sensitivity_latest.json"
    csv_path = output_root / f"markowitz_sensitivity_summary_{stamp}.csv"
    latest_csv = output_root / "markowitz_sensitivity_summary_latest.csv"

    text = json.dumps(payload, indent=2, sort_keys=True)
    json_path.write_text(text, encoding="utf-8")
    latest_json.write_text(text, encoding="utf-8")

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    latest_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

    print("")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote CSV:  {csv_path}")
    print(f"Latest JSON: {latest_json}")
    print(f"Latest CSV:  {latest_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Prototype5 Markowitz/capped-long risk sensitivity checks."
    )
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--focal_suite_dir", default=str(DEFAULT_FOCAL_SUITE_DIR))
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=sorted(REGIONS.keys()),
        default=["original", "unseendata_v2"],
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "current_12pct",
            "stress_loss_matched",
            "focal_max_exposure_matched",
            "focal_mean_exposure_matched",
            "margin_0p1",
            "margin_0p2",
        ],
    )
    parser.add_argument("--margins", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--tail_loss_budget_fraction", type=float, default=0.05)
    parser.add_argument("--tail_stress_return", type=float, default=25.0)
    parser.add_argument(
        "--settlement_clip_min_dkk",
        type=float,
        default=final_baselines.SETTLEMENT_CLIP_MIN_DKK_DEFAULT,
    )
    parser.add_argument(
        "--settlement_clip_max_dkk",
        type=float,
        default=final_baselines.SETTLEMENT_CLIP_MAX_DKK_DEFAULT,
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned region/arm/strategy combinations without evaluating.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    focal_suite_dir = Path(args.focal_suite_dir)
    if not focal_suite_dir.is_absolute():
        focal_suite_dir = PROJECT_ROOT / focal_suite_dir

    for region in args.regions:
        spec = REGIONS[region]
        if not spec["eval_data"].is_file():
            raise SystemExit(f"Missing evaluation data for {region}: {spec['eval_data']}")
        if not spec["settlement_data"].is_file():
            raise SystemExit(f"Missing settlement data for {region}: {spec['settlement_data']}")

    final_baselines.install_final_protocol_config_patch()
    focal_reference = _collect_focal_exposure_reference(focal_suite_dir)

    payload: Dict[str, Any] = {
        "evaluation_type": "markowitz_capped_long_sensitivity",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "focal_suite_dir": str(focal_suite_dir),
        "timesteps": int(args.timesteps),
        "seed": int(args.seed),
        "selected_regions": list(args.regions),
        "selected_arms": list(args.arms),
        "focal_exposure_reference": focal_reference,
        "results": {},
    }
    rows: List[Dict[str, Any]] = []

    for region in args.regions:
        data = _load_eval_data(REGIONS[region]["eval_data"], int(args.timesteps))
        region_results: Dict[str, Any] = {}

        base_protocol = _set_protocol_for_region(
            region=region,
            margin_fraction=0.05,
            settlement_clip_min_dkk=float(args.settlement_clip_min_dkk),
            settlement_clip_max_dkk=float(args.settlement_clip_max_dkk),
        )
        planned_arms = _build_arms(
            region=region,
            data=data,
            seed=int(args.seed),
            focal_reference=focal_reference,
            margins=list(args.margins),
            tail_loss_budget_fraction=float(args.tail_loss_budget_fraction),
            tail_stress_return=float(args.tail_stress_return),
            selected_arms=list(args.arms),
        )

        print("")
        print("=" * 100)
        print(f"Region: {region}")
        print(f"Eval data: {REGIONS[region]['eval_data']}")
        print(f"Settlement: {REGIONS[region]['settlement_data']}")
        print(f"Planned runs: {len(planned_arms)}")
        print("=" * 100)

        for item in planned_arms:
            arm = str(item["arm"])
            strategy = str(item["strategy"])
            margin = _safe_float(item["extra"].get("sweep_margin_fraction"), 0.05)
            protocol = _set_protocol_for_region(
                region=region,
                margin_fraction=margin,
                settlement_clip_min_dkk=float(args.settlement_clip_min_dkk),
                settlement_clip_max_dkk=float(args.settlement_clip_max_dkk),
            )
            label = f"{region} | {arm} | {strategy} | margin={margin:.3f}"
            print(label)
            if args.dry_run:
                continue

            result = item["runner"](
                data,
                seed=int(args.seed),
                cap_fn=item["cap_fn"],
                method=f"{strategy} sensitivity: {arm}",
                baseline_id=f"{strategy}_{arm}",
            )
            result["final_paper_protocol"] = dict(protocol)
            result["eval_data_path"] = str(REGIONS[region]["eval_data"])
            region_results[f"{strategy}_{arm}"] = {
                "result": result,
                "extra": dict(item["extra"]),
                "protocol": dict(protocol),
            }
            rows.append(
                _summarize_result(
                    region=region,
                    arm=f"{strategy}_{arm}",
                    strategy=strategy,
                    result=result,
                    protocol=protocol,
                    extra=dict(item["extra"]),
                )
            )

        payload["results"][region] = {
            "protocol": dict(base_protocol),
            "eval_data_path": str(REGIONS[region]["eval_data"]),
            "settlement_data_path": str(REGIONS[region]["settlement_data"]),
            "arms": region_results,
        }

    if args.dry_run:
        print("")
        print("Dry run only; no output files written.")
        return 0

    _write_outputs(output_root, payload, rows)

    print("")
    print("Summary")
    print("-" * 120)
    print(
        f"{'region':<15} {'arm':<42} {'sleeve ret %':>13} "
        f"{'HAC7 Sharpe':>12} {'DD %':>10} {'mean exp DKK':>15}"
    )
    for row in rows:
        print(
            f"{row.get('region', ''):<15} {row.get('arm', ''):<42} "
            f"{_safe_float(row.get('sleeve_trading_return_pct')):>13.3f} "
            f"{_safe_float(row.get('sleeve_trading_sharpe_ratio')):>12.3f} "
            f"{_safe_float(row.get('sleeve_trading_max_drawdown_pct')):>10.3f} "
            f"{_safe_float(row.get('sleeve_mean_abs_exposure_dkk')):>15.0f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

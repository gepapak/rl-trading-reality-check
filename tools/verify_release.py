"""Recompute the headline numbers of README.md and docs/METHODS_AND_RESULTS.md from the shipped result files.

Needs only this repository (no engine, no network).

Usage: python tools/verify_release.py
Exit code 0 = every check passes, 1 = at least one check fails.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R = pd.read_csv(REPO / "results" / "campaign" / "results_long.csv")
FAILS: list[str] = []


def check(name: str, value: float, expected: float, tol: float) -> None:
    ok = bool(np.isfinite(value) and abs(value - expected) <= tol)
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {value:,.3f} (expected {expected:,.3f} ± {tol})")
    if not ok:
        FAILS.append(name)


def sel(variant: str, agent: str, region: str | None = None, seeds: list[int] | None = None) -> pd.DataFrame:
    x = R[(R.variant == variant) & (R.agent == agent)]
    if region:
        x = x[x.region == region]
    if seeds:
        x = x[x.seed.isin(seeds)]
    return x


def iqm(v: np.ndarray) -> float:
    v = np.sort(v)
    k = int(np.floor(len(v) * 0.25))
    return float(v[k:len(v) - k].mean())


def main() -> int:
    print("== L1 price-taker (ledger, seed means)")
    check("anchor L1 DK1", sel("L1_price_taker", "anchor", "original").investor_return_pct.mean(), 154.47, 0.05)
    check("anchor L1 DK2", sel("L1_price_taker", "anchor", "v2").investor_return_pct.mean(), 480.22, 0.05)
    check("feasible L1 DK1", sel("L1_price_taker", "feasible", "original").investor_return_pct.mean(), 140.24, 0.05)
    check("MARL L1 DK1", sel("L1_price_taker", "marl", "original").investor_return_pct.mean(), -96.76, 0.05)
    check("MARL L1 sleeve-ruined share", sel("L1_price_taker", "marl").sleeve_ruined.mean(), 1.0, 1e-9)

    print("== L3 loss forgiveness (MARL, 20 runs)")
    l3 = sel("L3_price_taker_no_solvency", "marl")
    check("runs", len(l3), 20, 0)
    check("ledger min %", l3.investor_return_pct.min(), -3338.07, 0.1)
    check("ledger max %", l3.investor_return_pct.max(), -737.11, 0.1)
    check("runs reporting a positive return", int((l3.sleeve_return_pct > 0).sum()), 13, 0)
    check("max reported fund return %", l3.fund_return_pct.max(), 12.54, 0.05)

    print("== L7 thesis-like, MARL 10 seeds")
    for region, exp in (("original", (269.9, 73.0, -101.2, 40.5, -534.8)), ("v2", (121.8, -6.2, -100.7, 23.0, -430.7))):
        x = sel("L7_thesis_like", "marl", region)
        check(f"{region} n", len(x), 10, 0)
        check(f"{region} reported mean", x.sleeve_return_pct.mean(), exp[0], 0.1)
        check(f"{region} reported IQM", iqm(x.sleeve_return_pct.values), exp[1], 0.1)
        check(f"{region} reported median", x.sleeve_return_pct.median(), exp[2], 0.1)
        check(f"{region} fund mean", x.fund_return_pct.mean(), exp[3], 0.1)
        check(f"{region} ledger mean", x.investor_return_pct.mean(), exp[4], 0.1)
    check("anchor L7 DK1 reported", sel("L7_thesis_like", "anchor", "original").sleeve_return_pct.mean(), -35.9, 0.1)
    check("anchor L7 DK2 reported", sel("L7_thesis_like", "anchor", "v2").sleeve_return_pct.mean(), -41.5, 0.1)

    print("== leave-one-out (MARL seeds 7/42/123, both regions)")
    for v, rep, led in (("L7_minus_percent_mtm", 437.5, -2980.2), ("L7_minus_no_solvency", -95.2, -94.8),
                        ("L7_minus_price_taker", -0.2, -0.0), ("L7_minus_interp", 533.3, -2817.8), ("L7_minus_sweeper", 417.8, -300.0)):
        x = sel(v, "marl")
        check(f"{v} reported", x.sleeve_return_pct.mean(), rep, 0.1)
        check(f"{v} ledger", x.investor_return_pct.mean(), led, 0.1)
    ns = sel("L7_minus_no_solvency", "marl")
    check("L7_minus_no_solvency max |reported-ledger| pp", (ns.sleeve_return_pct - ns.investor_return_pct).abs().max(), 0.42, 0.01)

    print("== ledger agreement under sound accounting (L0-L6, all agents)")
    sound = R[R.variant.isin(["L0_strict", "L1_price_taker", "L2_no_solvency", "L4_load_proxy_liquidity", "L5_percent_payoff",
                              "L6_interpolated_prices"])]
    check("runs (138 re-runs + 36 frozen strict baselines)", len(sound), 174, 0)
    check("max |reported-ledger| pp", (sound.sleeve_return_pct - sound.investor_return_pct).abs().max(), 1.26, 0.01)

    print("== ledger_check.py on the shipped examples")
    for f, exp_rc in (("L0_strict_marl_seed7_DK1", 0), ("L3_marl_seed7_DK1", 1), ("L7_marl_seed42_DK1", 0), ("L7_marl_seed7_DK1", 1)):
        rc = subprocess.run([sys.executable, str(REPO / "tools" / "ledger_check.py"), str(REPO / "results" / "ledger_examples" / f"{f}.csv.gz")],
                            capture_output=True, text=True).returncode
        check(f"{f} exit code", rc, exp_rc, 0)

    print("== prevalence")
    c = pd.read_csv(REPO / "code_audit" / "coding.csv", sep="|", keep_default_na=False)
    check("code audit: repositories", len(c), 21, 0)
    check("code audit: capital account present", int((c.C0_money == "Y").sum()), 1, 0)
    check("code audit: solvency modelled", int((c.C2_solvency == "Y").sum()), 0, 0)
    check("code audit: percent payoff", int((c.C4_payoff == "b").sum()), 0, 0)
    check("code audit: price-taker", int((c.C1_price_taker == "Y").sum()), 11, 0)
    lit = pd.read_csv(REPO / "literature" / "literature_survey.csv")
    lit = lit[lit.included.str.startswith("yes")]
    check("literature: papers", len(lit), 21, 0)
    check("literature: price-taker (explicit/implicit)", int(lit.C1_liquidity.str.contains("price-taker", case=False).sum()), 14, 0)
    check("literature: solvency enforced", int(lit.C3_solvency.str.contains("enforced").sum()), 0, 0)

    print("== cross-market (shipped results)")
    xm = pd.read_csv(REPO / "results" / "cross_market" / "cross_market_liquidity_test.csv")
    check("min R (value-vs-volume fill)", xm.R_value_vs_volume_fill.min(), 1.131, 0.002)
    check("zones", len(xm), 8, 0)

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else f'{len(FAILS)} CHECK(S) FAILED: ' + ', '.join(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())

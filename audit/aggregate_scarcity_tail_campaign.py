"""Aggregate the learned-agent audit campaign into the "four illusions" evidence tables and check every
pre-registered prediction in PREREGISTRATION.md. Reads only files inside Paper1.
Outputs: campaign_runs/results_long.csv, campaign_runs/results_summary.csv, campaign_runs/PREDICTION_VERDICTS.md
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

AUD = Path(__file__).resolve().parent
RUNS = AUD / "campaign_runs"
P5 = AUD.parent / "Prototype5"
SETTLE = {"original": P5 / "evaluation_dataset_ffill" / "unseendata_settlement_real_v2.csv",
          "v2": P5 / "evaluation_dataset_ffill" / "unseendata_v2_settlement_real_v2.csv"}
COLS = ["timestep", "decision_step", "price_current", "position_signed", "liquidity_executed_volume_mwh",
        "horizon_settlement_pnl_dkk", "mtm_pnl", "cumulative_volume_transaction_fees_dkk", "cumulative_market_access_fees_dkk",
        "cumulative_impact_costs_dkk", "cumulative_collateral_funding_costs_dkk", "trading_sleeve_value_dkk"]


def investor_metrics(debug_csv: str, region: str) -> dict:
    d = pd.read_csv(debug_csv, usecols=lambda c: c in COLS)
    sleeve0 = float(d.trading_sleeve_value_dkk.iloc[0])
    cost_cols = [c for c in COLS if c.startswith("cumulative_") and c in d]
    costs = d[cost_cols].fillna(0).sum(axis=1)
    # booked trading P&L: mtm_pnl equals horizon_settlement_pnl under settlement payoffs, and is the only booked
    # channel under the thesis-like percent-MTM payoff (where horizon settlement is zero)
    booked = d.mtm_pnl if "mtm_pnl" in d else d.horizon_settlement_pnl_dkk
    settle_share = float(d.horizon_settlement_pnl_dkk.abs().sum() / max(booked.abs().sum(), 1e-9))
    step = booked.fillna(0).values - np.diff(np.r_[0.0, costs.values])
    eq = sleeve0 + np.cumsum(step)
    dd = 1 - eq / np.maximum.accumulate(eq)
    day = np.arange(len(d)) // 144
    daily = pd.Series(step).groupby(day).sum()
    sh = lambda x: float(x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else np.nan
    # spike-removal sensitivity: drop the 3 largest-|pnl| settlement events
    ev = booked.fillna(0).values
    idx = np.argsort(-np.abs(ev))[:3]
    step3 = step.copy(); step3[idx] = 0.0
    daily3 = pd.Series(step3).groupby(day).sum()
    return dict(investor_return_pct=100 * (eq[-1] / sleeve0 - 1), investor_max_dd_pct=100 * float(dd.max()),
                investor_min_equity_pct=100 * (eq.min() / sleeve0 - 1), investor_daily_sharpe=sh(daily),
                investor_daily_sharpe_ex_top3=sh(daily3), path_ruined=bool(eq.min() <= 0.05 * sleeve0),
                settlement_share_of_booked_pnl=settle_share, sleeve_cash_floor_hit=bool((d.trading_sleeve_value_dkk <= 1.0).any()),
                long_share=float((d.position_signed > 0).mean()), short_share=float((d.position_signed < 0).mean()))


ALL_SEEDS = [7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202]
FROZEN = {"marl": P5 / "batch_tier_phase_runs" / "prototype5_mappo_marl_final_v1" / "seed{s}" / "{ev}" / "tier1",
          "feasible": P5 / "Ablations" / "batch_tier_phase_runs" / "prototype5_mechanism_ablations_final_v1" / "feasible_action_full"
                      / "seed{s}" / "{ev}" / "tier1_forecast_utilization"}


def row_from_tier(t: dict, json_path: str, job: dict, source: str) -> dict | None:
    if t.get("status") != "completed":
        return None
    sm = t.get("sleeve_metrics", {}) or {}
    r = dict(variant=job["variant"], agent=job["agent"], seed=int(job["seed"]), region=job["region"], l0_source=source,
             fund_return_pct=100 * float(t.get("total_return", np.nan)), fund_10min_sharpe=float(t.get("sharpe_ratio", np.nan)),
             sleeve_return_pct=float(sm.get("sleeve_trading_return_pct", np.nan)),
             sleeve_ruined=bool(sm.get("sleeve_trading_ruined", False)),
             sleeve_max_dd_pct=float(sm.get("sleeve_trading_max_drawdown_pct", np.nan)),
             mismatch_allowed=bool((t.get("sim_audit") or {}).get("mismatch_allowed", False)))
    dbg = t.get("env_debug_log") or ""
    if not (dbg and Path(dbg).is_file()):  # frozen runs record the pre-move path; the log sits next to the JSON
        dbg = str(Path(json_path).parent / "env_logs" / "tier1_debug_ep0.csv")
    if Path(dbg).is_file():
        r.update(investor_metrics(dbg, job["region"]))
    return r


def collect() -> pd.DataFrame:
    rows = []
    for jf in glob.glob(str(RUNS / "evals" / "*" / "audit_job.json")):
        job = json.load(open(jf, encoding="utf-8"))
        outs = sorted(glob.glob(str(Path(jf).parent / "**" / "evaluation_tiers_*.json"), recursive=True))
        if not outs:
            continue
        r = row_from_tier(json.load(open(outs[-1], encoding="utf-8"))["tiers"]["tier1"], outs[-1], job, "audit_rerun")
        if r:
            rows.append(r)
    # L0 = frozen final campaign (the validation gate showed the audit engine reproduces it to 1e-8 pp), so strict
    # baselines for seeds not re-run are read from the frozen outputs; seed sets then match on both sides of every comparison.
    have = {(r["agent"], r["seed"], r["region"]) for r in rows if r["variant"] == "L0_strict"}
    for agent, pat in FROZEN.items():
        for s in ALL_SEEDS:
            for region, ev in (("original", "evaluations_2025"), ("v2", "evaluations_2025_v2")):
                if (agent, s, region) in have:
                    continue
                fs = sorted(glob.glob(str(Path(str(pat).format(s=s, ev=ev)) / "evaluation_tiers_*.json")))
                if fs:
                    job = dict(variant="L0_strict", agent=agent, seed=s, region=region)
                    r = row_from_tier(json.load(open(fs[-1], encoding="utf-8"))["tiers"]["tier1"], fs[-1], job, "frozen_campaign")
                    if r:
                        rows.append(r)
    return pd.DataFrame(rows)


def verdicts(R: pd.DataFrame) -> list[str]:
    out = []
    def m(v, a, reg, col="investor_return_pct"):
        x = R[(R.variant == v) & (R.agent == a) & (R.region == reg)][col]
        return float(x.mean()) if len(x) else np.nan
    regs = ["original", "v2"]
    # P1
    anc = [m("L1_price_taker", "anchor", g) / m("L0_strict", "anchor", g) if m("L0_strict", "anchor", g) else np.nan for g in regs]
    marl_up = [m("L1_price_taker", "marl", g) > m("L0_strict", "marl", g) for g in regs]
    out.append(f"P1 liquidity illusion: anchor L1/L0 ratio {np.round(anc,1)} (need >=50 both); MARL mean higher under L1 {marl_up} (need both) -> "
               + ("SUPPORTED" if all(x >= 50 for x in anc if x == x) and len([x for x in anc if x == x]) == 2 and all(marl_up) else "NOT SUPPORTED"))
    # P2
    l1 = R[(R.variant == "L1_price_taker") & (R.agent == "marl")]
    bad = float(((l1.sleeve_ruined) | (l1.investor_max_dd_pct >= 50) | (l1.sleeve_max_dd_pct >= 50)).mean()) if len(l1) else np.nan
    # PREREGISTRATION.md words P2(b) as the "reported return" (the engine's sleeve_trading_return_pct). The first
    # aggregator version used the reconstructed ledger instead; corrected at ~22:00 on 28 Sep 2026, after results were seen.
    # Both readings are printed.
    l3_gt = [m("L3_price_taker_no_solvency", "marl", g, "sleeve_return_pct") > m("L1_price_taker", "marl", g, "sleeve_return_pct") for g in regs]
    l3_gt_ledger = [m("L3_price_taker_no_solvency", "marl", g) > m("L1_price_taker", "marl", g) for g in regs]
    out.append(f"P2 solvency illusion: MARL L1 ruined-or-DD>=50% share {bad:.2f} (need >=0.20); L3 reported-return mean > L1 {l3_gt} (need >=1) "
               f"[ledger reading: {l3_gt_ledger}] -> " + ("SUPPORTED" if bad >= 0.2 and any(l3_gt) else "NOT SUPPORTED"))
    # P3
    tr = "marl_trained_price_taker"
    base_l1 = R[(R.variant == "L1_price_taker") & (R.agent == "marl") & (R.seed.isin([7, 42, 123]))].investor_return_pct.mean()
    base_l0 = R[(R.variant == "L0_strict") & (R.agent == "marl") & (R.seed.isin([7, 42, 123]))].investor_return_pct.mean()
    tr_l1 = R[(R.variant == "L1_price_taker") & (R.agent == tr)].investor_return_pct.mean()
    tr_l0 = R[(R.variant == "L0_strict") & (R.agent == tr)].investor_return_pct.mean()
    out.append(f"P3 learning exploits illusion: retrained@L1 {tr_l1:.3f} vs strict-trained@L1 {base_l1:.3f} (need >); retrained@L0 {tr_l0:.3f} vs strict-trained@L0 {base_l0:.3f} (need <=) -> "
               + ("SUPPORTED" if tr_l1 > base_l1 and tr_l0 <= base_l0 else "NOT SUPPORTED / INCOMPLETE"))
    # P4
    ratio = (R.fund_10min_sharpe / R.investor_daily_sharpe.abs()).replace([np.inf, -np.inf], np.nan)
    share = float((ratio > 10).mean()) if len(ratio) else np.nan
    flips = 0
    for (v, g), grp in R[R.variant == "L0_strict"].groupby(["variant", "region"]):
        a = grp.groupby("agent")[["investor_daily_sharpe", "investor_daily_sharpe_ex_top3"]].mean()
        if len(a) >= 2 and list(a.investor_daily_sharpe.sort_values().index) != list(a.investor_daily_sharpe_ex_top3.sort_values().index):
            flips += 1
    out.append(f"P4 metric illusion: share of runs with fund 10-min Sharpe >10x investor daily Sharpe {share:.2f} (need >=0.8); Sharpe-rank flips after removing top-3 events: {flips} (need >=1) -> "
               + ("SUPPORTED" if share >= 0.8 and flips >= 1 else "NOT SUPPORTED"))
    # P5
    three = ["anchor", "feasible", "marl"]
    base_rank = {g: list(R[(R.variant == "L0_strict") & (R.region == g) & (R.agent.isin(three))]
                        .groupby("agent").investor_return_pct.mean().sort_values(ascending=False).index) for g in regs}
    flipped, flipped_ledger = [], []
    for (v, g), grp in R[(R.variant != "L0_strict") & (R.agent.isin(three))].groupby(["variant", "region"]):
        # L0 ranking on the same seed set as the shortcut run (side variants use 3 seeds, key variants 10)
        l0 = R[(R.variant == "L0_strict") & (R.region == g) & (R.agent.isin(three))]
        l0 = l0[[s in set(grp[grp.agent == a].seed) for a, s in zip(l0.agent, l0.seed)]]
        for col, bucket in (("sleeve_return_pct", flipped), ("investor_return_pct", flipped_ledger)):
            rk = list(grp.groupby("agent")[col].mean().sort_values(ascending=False).index)
            base = list(l0.groupby("agent")[col].mean().sort_values(ascending=False).index)
            if len(rk) == 3 and len(base) == 3 and rk != base:
                bucket.append(f"{v}/{g}: {base} -> {rk}")
    out.append(f"P5 ranking flip (reported investor return): flipped under {flipped if flipped else 'none'} -> "
               + ("SUPPORTED" if flipped else "NOT SUPPORTED") + f" [ledger reading: {flipped_ledger if flipped_ledger else 'none'}]")
    out.append("Not applicable: L5_percent_payoff x feasible-action. The engine refuses this combination "
               "([TRADE_EXEC_FATAL] feasible-action MAPPO requires the mwh_volume payoff protocol), so its 6 runs exit=1 by design.")
    return out


def addenda_verdicts(R: pd.DataFrame) -> list[str]:
    """PREREGISTRATION.md addendum 2 (Q1-Q4, L7 ten-seed extension) and addendum 3 (R1-R5, L7 leave-one-out)."""
    out, regs = [], ["original", "v2"]
    new_seeds = [2025, 3007, 5001, 8102, 9005, 10001, 11202]
    l7 = R[(R.variant == "L7_thesis_like") & (R.agent == "marl")]
    nw = l7[l7.seed.isin(new_seeds)]
    if len(nw) == 14:
        dom = np.maximum(nw.long_share, nw.short_share)
        out.append(f"Q1 direction saturation: min dominant share {dom.min():.3f} over 14 new runs (need >=0.90 each) -> "
                   + ("SUPPORTED" if (dom >= 0.9).all() else "NOT SUPPORTED"))
        sgn = np.where(nw.long_share >= nw.short_share, 1, -1)
        ok = (np.sign(nw.investor_return_pct) == sgn)
        out.append(f"Q2 drift mechanism: sign(ledger)==dominant direction in {int(ok.sum())}/14 new runs (need 14) -> "
                   + ("SUPPORTED" if ok.all() else "NOT SUPPORTED"))
        los = l7[l7.investor_return_pct < -100]
        q3 = bool(((los.sleeve_return_pct >= -120) & (los.fund_return_pct >= -10)).all())
        out.append(f"Q3 zero-floor loss cap: {len(los)} L7 MARL runs with ledger < -100%; min reported {los.sleeve_return_pct.min():.1f}%, "
                   f"min fund {los.fund_return_pct.min():.1f}% (need >=-120 and >=-10) -> " + ("SUPPORTED" if q3 else "NOT SUPPORTED"))
        anc = R[(R.variant == "L7_thesis_like") & (R.agent == "anchor")].set_index("region")
        a1 = [l7[l7.region == g].sleeve_return_pct.mean() > anc.loc[g, "sleeve_return_pct"] for g in regs]
        a2 = [l7[l7.region == g].investor_return_pct.mean() < anc.loc[g, "investor_return_pct"] for g in regs]
        out.append(f"Q4 headline at 10 seeds: MARL reported mean {[round(l7[l7.region == g].sleeve_return_pct.mean(), 1) for g in regs]} vs anchor "
                   f"{[round(anc.loc[g, 'sleeve_return_pct'], 1) for g in regs]} (> both: {a1}); MARL ledger mean "
                   f"{[round(l7[l7.region == g].investor_return_pct.mean(), 1) for g in regs]} vs anchor {[round(anc.loc[g, 'investor_return_pct'], 1) for g in regs]} "
                   f"(< both: {a2}) -> " + ("SUPPORTED" if all(a1) and all(a2) else "NOT SUPPORTED")
                   + f"; long-saturated seeds: {int((l7.groupby('seed').long_share.mean() >= 0.9).sum())}/{l7.seed.nunique()}")
    else:
        out.append(f"Q1-Q4: incomplete ({len(nw)}/14 new L7 runs)")

    def inversion(v):
        res = []
        for g in regs:
            mm = R[(R.variant == v) & (R.agent == "marl") & (R.region == g)].sleeve_return_pct
            aa = R[(R.variant == v) & (R.agent == "anchor") & (R.region == g)].sleeve_return_pct
            res.append(bool(len(mm) == 3 and len(aa) == 1 and mm.mean() > aa.iloc[0]) if len(mm) == 3 and len(aa) == 1 else None)
        return res

    def means(v):
        return {g: (round(R[(R.variant == v) & (R.agent == "marl") & (R.region == g)].sleeve_return_pct.mean(), 1),
                    round(R[(R.variant == v) & (R.agent == "anchor") & (R.region == g)].sleeve_return_pct.mean(), 1)) for g in regs}
    loo = [v for v in R.variant.unique() if v.startswith("L7_minus_") or v == "L8_pctmtm_nosolv"]
    if len(loo) < 7 or any(None in inversion(v) for v in loo):
        out.append(f"R1-R5: incomplete ({len(R[R.variant.isin(loo)])}/56 leave-one-out runs)")
        return out
    inv = {v: inversion(v) for v in loo}
    out.append(f"R1 payoff necessary: L7_minus_percent_mtm inversion {inv['L7_minus_percent_mtm']} (marl, anchor) {means('L7_minus_percent_mtm')} (need none) -> "
               + ("SUPPORTED" if not any(inv["L7_minus_percent_mtm"]) else "NOT SUPPORTED"))
    ns = R[R.variant == "L7_minus_no_solvency"]
    gap = (ns.sleeve_return_pct - ns.investor_return_pct).abs().max()
    out.append(f"R2 floor hides ledger: L7_minus_no_solvency max |reported-ledger| {gap:.2f} pp (need <=5) -> " + ("SUPPORTED" if gap <= 5 else "NOT SUPPORTED"))
    out.append(f"R3 interpolation not necessary: L7_minus_interp inversion {inv['L7_minus_interp']} {means('L7_minus_interp')} (need >=1) -> "
               + ("SUPPORTED" if any(inv["L7_minus_interp"]) else "NOT SUPPORTED"))
    r4 = all(inv["L7_minus_legacy_fee"]) and all(inv["L7_minus_sweeper"])
    out.append(f"R4 fee/sweeper not necessary: legacy_fee {inv['L7_minus_legacy_fee']} {means('L7_minus_legacy_fee')}, sweeper {inv['L7_minus_sweeper']} "
               f"{means('L7_minus_sweeper')} (need all) -> " + ("SUPPORTED" if r4 else "NOT SUPPORTED"))
    out.append(f"R5 minimal pair: L8_pctmtm_nosolv inversion {inv['L8_pctmtm_nosolv']} {means('L8_pctmtm_nosolv')} (need >=1) -> "
               + ("SUPPORTED" if any(inv["L8_pctmtm_nosolv"]) else "NOT SUPPORTED"))
    out.append(f"Reported only: L7_minus_price_taker inversion {inv['L7_minus_price_taker']} {means('L7_minus_price_taker')}")
    return out


def main() -> int:
    R = collect()
    if R.empty:
        print("no completed evaluations yet")
        return 0
    R.to_csv(RUNS / "results_long.csv", index=False)
    S = R.groupby(["variant", "agent", "region"]).agg(
        n=("seed", "size"), investor_return_mean=("investor_return_pct", "mean"), investor_return_min=("investor_return_pct", "min"),
        investor_return_max=("investor_return_pct", "max"), investor_max_dd_mean=("investor_max_dd_pct", "mean"),
        ruined_share=("path_ruined", "mean"), sleeve_ruined_share=("sleeve_ruined", "mean"),
        investor_daily_sharpe=("investor_daily_sharpe", "mean"), fund_10min_sharpe=("fund_10min_sharpe", "mean")).reset_index()
    S.to_csv(RUNS / "results_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string(index=False))
    R["reported_minus_ledger_pp"] = R.sleeve_return_pct - R.investor_return_pct
    G = R.groupby(["variant", "agent"]).agg(n=("seed", "size"), reported_mean=("sleeve_return_pct", "mean"),
                                            ledger_mean=("investor_return_pct", "mean"),
                                            max_abs_gap_pp=("reported_minus_ledger_pp", lambda x: float(x.abs().max())),
                                            reported_positive_while_ledger_below_minus100=(
                                                "reported_minus_ledger_pp", lambda x: int(((R.loc[x.index, "sleeve_return_pct"] > 0)
                                                                                           & (R.loc[x.index, "investor_return_pct"] < -100)).sum())),
                                            settlement_share=("settlement_share_of_booked_pnl", "mean")).reset_index()
    G.to_csv(RUNS / "ledger_vs_reported.csv", index=False)
    print(G.round(3).to_string(index=False))
    V = verdicts(R) + addenda_verdicts(R)
    (RUNS / "PREDICTION_VERDICTS.md").write_text("# Pre-registered prediction verdicts\n\n" + "\n".join(f"- {v}" for v in V) + "\n", encoding="utf-8")
    print("\n".join(V))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

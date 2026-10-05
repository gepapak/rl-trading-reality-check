"""Study B aggregation: the pre-registered checks B1-B5 (PREREGISTRATION_B.md), applied verbatim.

    python aggregate_b.py            # full run (test period)
    python aggregate_b.py --smoke    # logic check on the smoke outputs (train split, tiny budgets, one seed)

Outputs (results/): runs_b.csv, cells_b.csv, VERDICTS_B.md
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
N_BOOT = 10_000


def load(smoke: bool) -> pd.DataFrame:
    rows = [json.load(open(f)) for f in glob.glob(str(RES / ("jobs_smoke" if smoke else "jobs") / "*.json"))]
    R = pd.DataFrame(rows)
    ne = R.market == "NE"
    # NE outcomes are means over the sign draws; RM outcomes come from the single real path
    R["ML"] = np.where(ne, R.get("drawmean_mean_abs_lev_open"), R.mean_abs_lev_open)
    R["rep"] = np.where(ne, R.get("drawmean_reported_return_pct"), R.reported_return_pct)
    R["led"] = np.where(ne, R.get("drawmean_ledger_return_pct"), R.ledger_return_pct)
    R["inflation"] = R.rep - R.led
    R["RS"] = R["probe_e0.1"] - R["probe_e1"]
    return R


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    x = np.asarray(x, float)
    means = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cells(R: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (market, algo, region, liab), g in R.groupby(["market", "algo", "region", "liability"]):
        lo, hi = boot_ci(g.rep.values, rng)
        rows.append(dict(market=market, algo=algo, region=region, liability=liab, n_seeds=len(g),
                         ML=g.ML.mean(), RS=g.RS.mean(), rep=g.rep.mean(), rep_lo=lo, rep_hi=hi, led=g.led.mean(),
                         rep_median=g.rep.median(), led_median=g.led.median(), inflation=g.inflation.mean(),
                         floor_share=g.floor_bound.mean(), closed_share=g.closed.mean(),
                         **{c: g[c].mean() for c in g.columns if c.startswith("probe_e")}))
    return pd.DataFrame(rows)


def checks(C: pd.DataFrame, R: pd.DataFrame) -> list[str]:
    W = C.pivot_table(index=["market", "algo", "region"], columns="liability", values=["ML", "RS", "rep_lo"])
    ne = W.loc["NE"] if "NE" in W.index.get_level_values(0) else W.iloc[0:0]
    n_all, n_ne = len(W), len(ne)
    cnt = lambda s: int(s.sum())  # noqa: E731
    out = [f"cells available: {n_all} of 8 (NE {n_ne} of 4); seeds per cell: {sorted(C.n_seeds.unique().tolist())}"]
    # B1
    b1a, b1b = cnt(ne.ML.ZF > ne.ML.FL), cnt(ne.ML.LL > ne.ML.FL)
    out.append(f"B1 NE: ML(ZF) > ML(FL) in {b1a}/{n_ne} (need 4/4); ML(LL) > ML(FL) in {b1b}/{n_ne} (need >=3) -> "
               + ("SUPPORTED" if n_ne == 4 and b1a == 4 and b1b >= 3 else "NOT SUPPORTED"))
    # B2
    b2a, b2b, b2c = cnt(W.RS.ZF > W.RS.FL), cnt(W.RS.ZF > 0), cnt(W.RS.LL > W.RS.FL)
    out.append(f"B2 RS(ZF) > RS(FL) in {b2a}/{n_all} (need >=6); RS(ZF) > 0 in {b2b}/{n_all} (need >=6) -> "
               + ("SUPPORTED" if b2a >= 6 and b2b >= 6 else "NOT SUPPORTED")
               + f"; secondary RS(LL) > RS(FL) in {b2c}/{n_all} (need >=6) -> " + ("SUPPORTED" if b2c >= 6 else "NOT SUPPORTED"))
    # B3
    b3a, b3b = cnt(ne.rep_lo.ZF > 0), cnt(ne.rep_lo.FL <= 0)
    out.append(f"B3 NE: ZF reported lower bound > 0 in {b3a}/{n_ne} (need >=3); FL reported lower bound <= 0 in {b3b}/{n_ne} (need >=3) -> "
               + ("SUPPORTED" if b3a >= 3 and b3b >= 3 else "NOT SUPPORTED"))
    # B4
    z = R[R.liability.isin(["ZF", "LL"])]
    rho = float(spearmanr(z.ML, z.inflation).correlation) if len(z) > 2 else float("nan")
    out.append(f"B4 Spearman(ML, inflation) over {len(z)} ZF/LL runs = {rho:.2f} (need > 0.5) -> "
               + ("SUPPORTED" if rho > 0.5 else "NOT SUPPORTED"))
    # B5
    b5a, b5b = cnt(W.RS.LL < W.RS.ZF), cnt(ne.ML.LL < ne.ML.ZF)
    out.append(f"B5 RS(LL) < RS(ZF) in {b5a}/{n_all} (need >=6); ML(LL) < ML(ZF) in {b5b}/{n_ne} NE cells (need >=3) -> "
               + ("SUPPORTED" if b5a >= 6 and b5b >= 3 else "NOT SUPPORTED"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    R = load(a.smoke)
    C = cells(R, np.random.default_rng(20260929))
    tag = "_SMOKE" if a.smoke else ""
    R.drop(columns=[c for c in R.columns if c.startswith("draws_")]).to_csv(RES / f"runs_b{tag}.csv", index=False)
    C.to_csv(RES / f"cells_b{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(C[["market", "algo", "region", "liability", "n_seeds", "ML", "RS", "rep", "rep_lo", "rep_hi", "led", "inflation",
             "floor_share", "closed_share"]].round(2).to_string(index=False))
    out = checks(C, R)
    print("\n".join(out))
    (RES / f"VERDICTS_B{tag}.md").write_text("# Study B verdicts" + (" (SMOKE: train split, tiny budgets)" if a.smoke else "")
                                             + "\n\n" + "\n".join(f"- {c}" for c in out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

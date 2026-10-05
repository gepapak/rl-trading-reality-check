"""Study H aggregation: pre-registered checks H1-H5 (PREREGISTRATION_H.md), written and hashed before the run.

    python aggregate_h.py [--smoke]
Outputs: results_h/runs_h.csv, results_h/curves_h.csv, results_h/VERDICTS_H.md
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results_h"
rng = np.random.default_rng(20261003)


def boot_mean(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (10_000, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def boot_diff(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a[rng.integers(0, len(a), (10_000, len(a)))].mean(axis=1) - b[rng.integers(0, len(b), (10_000, len(b)))].mean(axis=1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rows = [json.load(open(f)) for f in glob.glob(str(RES / ("jobs_smoke" if a.smoke else "jobs") / "*.json"))]
    R = pd.DataFrame(rows)
    for k in rows[0]["lev"]:
        R[f"L{k}"] = R.lev.apply(lambda d: d[k])
    R["inflation"] = R.drawmean_reported_return_pct - R.drawmean_ledger_return_pct
    L = lambda rule, obs, e: R[(R.rule == rule) & (R.obsset == obs)][f"L{e}"].to_numpy()  # noqa: E731
    out = [f"runs: {len(R)}; per rule x obs: {R.groupby(['rule','obsset']).size().to_dict()}"]
    z, f = L("ZF", "EQ", "0.02"), L("FL", "EQ", "0.02")
    g, (lo, hi) = float(z.mean() - f.mean()), boot_diff(z, f)
    out.append(f"H1 EQ: mean L_ZF(0.02) - L_FL(0.02) = {g:.2f} [{lo:.2f}, {hi:.2f}] (need >= 8, lower bound > 4) -> "
               + ("SUPPORTED" if g >= 8 and lo > 4 else "NOT SUPPORTED"))
    h2 = int(((L("FL", "EQ", "0.02") <= 1) & (L("FL", "EQ", "1") <= 1)).sum())
    out.append(f"H2 EQ: L_FL(0.02) <= 1 and L_FL(1) <= 1 in {h2}/{len(f)} runs (need >= 8 of 10) -> " + ("SUPPORTED" if h2 >= 8 else "NOT SUPPORTED"))
    h3 = int((L("ZF", "EQ", "0.02") > L("ZF", "EQ", "1")).sum())
    out.append(f"H3 EQ: L_ZF(0.02) > L_ZF(1) in {h3}/{len(z)} runs (need >= 8 of 10) -> " + ("SUPPORTED" if h3 >= 8 else "NOT SUPPORTED"))
    zf, ff = L("ZF", "FULL", "0.02"), L("FL", "FULL", "0.02")
    g4, (lo4, hi4) = float(zf.mean() - ff.mean()), boot_diff(zf, ff)
    out.append(f"H4 FULL: mean L_ZF(0.02) - L_FL(0.02) = {g4:.2f} [{lo4:.2f}, {hi4:.2f}] (need lower bound > 0) -> "
               + ("SUPPORTED" if lo4 > 0 else "NOT SUPPORTED"))
    inf = R[(R.rule == "ZF") & (R.obsset == "EQ")].inflation.to_numpy()
    lo5, hi5 = boot_mean(inf)
    out.append(f"H5 EQ: ZF reported - ledger = {inf.mean():.1f} pp [{lo5:.1f}, {hi5:.1f}] (need lower bound > 0) -> "
               + ("SUPPORTED" if lo5 > 0 else "NOT SUPPORTED"))
    tag = "_SMOKE" if a.smoke else ""
    R.drop(columns=["lev", "draws_reported_return_pct"]).to_csv(RES / f"runs_h{tag}.csv", index=False)
    cols = [c for c in R.columns if c.startswith("L")]
    C = R.groupby(["obsset", "rule"])[cols + ["drawmean_reported_return_pct", "drawmean_ledger_return_pct"]].mean()
    C.to_csv(RES / f"curves_h{tag}.csv")
    pd.set_option("display.width", 220)
    print(C.round(2).to_string())
    print("\n".join(out))
    (RES / f"VERDICTS_H{tag}.md").write_text("# Study H verdicts" + (" (SMOKE)" if a.smoke else "") + "\n\n"
                                            + "\n".join(f"- {o}" for o in out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

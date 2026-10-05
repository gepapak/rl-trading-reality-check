"""Study F aggregation: pre-registered checks F1-F6 (PREREGISTRATION_F.md), written and hashed before the run.

    python aggregate_f.py            # results_f/jobs
    python aggregate_f.py --smoke    # logic check on smoke outputs
Outputs: results_f/runs_f.csv, results_f/curves_f.csv, results_f/VERDICTS_F.md
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results_f"
N_BOOT = 10_000
rng = np.random.default_rng(20261003)


def boot_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(N_BOOT)]
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def boot_diff_of_diffs(a1, b1, a2, b2) -> tuple[float, float]:
    d = [(rng.choice(a1, len(a1)).mean() - rng.choice(b1, len(b1)).mean())
         - (rng.choice(a2, len(a2)).mean() - rng.choice(b2, len(b2)).mean()) for _ in range(N_BOOT)]
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rows = [json.load(open(f)) for f in glob.glob(str(RES / ("jobs_smoke" if a.smoke else "jobs") / "*.json"))]
    R = pd.DataFrame(rows)
    for k in rows[0]["lev"]:
        R[f"L{k}"] = R.lev.apply(lambda d: d[k])
    L = lambda rule, obs, e: R[(R.rule == rule) & (R.obsset == obs)][f"L{e}"].to_numpy()  # noqa: E731
    n = lambda rule, obs: int(((R.rule == rule) & (R.obsset == obs)).sum())  # noqa: E731
    out = [f"runs: {len(R)}; per rule x obs: { {f'{r}-{o}': n(r, o) for r in ('FL','LL','ZF') for o in ('EQ','FULL')} }"]
    f1 = int((L("ZF", "EQ", "0.1") == 16).sum())
    out.append(f"F1 EQ: L_ZF(0.1) = 16 in {f1}/{n('ZF','EQ')} runs (need >=8 of 10) -> " + ("SUPPORTED" if f1 >= 8 else "NOT SUPPORTED"))
    f2 = int((L("FL", "EQ", "0.1") <= 1).sum())
    out.append(f"F2 EQ: L_FL(0.1) <= 1 in {f2}/{n('FL','EQ')} runs (need >=8 of 10) -> " + ("SUPPORTED" if f2 >= 8 else "NOT SUPPORTED"))
    z, f = L("ZF", "EQ", "0.1"), L("FL", "EQ", "0.1")
    gap = float(z.mean() - f.mean()); lo, hi = boot_diff(z, f)
    out.append(f"F3 EQ: mean L_ZF(0.1) - L_FL(0.1) = {gap:.2f} [{lo:.2f}, {hi:.2f}] (need >= 8 and lower bound > 4) -> "
               + ("SUPPORTED" if gap >= 8 and lo > 4 else "NOT SUPPORTED"))
    f4 = int((L("ZF", "EQ", "0.1") > L("ZF", "EQ", "2")).sum()) if n("ZF", "EQ") else 0
    out.append(f"F4 EQ: L_ZF(0.1) > L_ZF(2) in {f4}/{n('ZF','EQ')} runs (need >=8 of 10) -> " + ("SUPPORTED" if f4 >= 8 else "NOT SUPPORTED"))
    zf_full, fl_full = L("ZF", "FULL", "0.1"), L("FL", "FULL", "0.1")
    gap_full = float(zf_full.mean() - fl_full.mean()); lo5, hi5 = boot_diff_of_diffs(z, f, zf_full, fl_full)
    out.append(f"F5 gap EQ {gap:.2f} vs FULL {gap_full:.2f}; difference {gap - gap_full:.2f} [{lo5:.2f}, {hi5:.2f}] (need lower bound > 0) -> "
               + ("SUPPORTED" if lo5 > 0 else "NOT SUPPORTED"))
    ll = L("LL", "EQ", "0.1"); d6 = float(z.mean() - ll.mean()); lo6, hi6 = boot_diff(z, ll)
    out.append(f"F6 EQ: mean L_ZF(0.1) - L_LL(0.1) = {d6:.2f} [{lo6:.2f}, {hi6:.2f}] (need lower bound > 0) -> "
               + ("SUPPORTED" if lo6 > 0 else "NOT SUPPORTED"))
    tag = "_SMOKE" if a.smoke else ""
    R.drop(columns=["lev", "draws_reported_return_pct"]).to_csv(RES / f"runs_f{tag}.csv", index=False)
    cols = [c for c in R.columns if c.startswith("L")]
    C = R.groupby(["obsset", "rule"])[cols + ["drawmean_reported_return_pct", "drawmean_ledger_return_pct"]].mean()
    C.to_csv(RES / f"curves_f{tag}.csv")
    pd.set_option("display.width", 220)
    print(C.round(2).to_string())
    print("\n".join(out))
    (RES / f"VERDICTS_F{tag}.md").write_text("# Study F verdicts" + (" (SMOKE)" if a.smoke else "") + "\n\n"
                                            + "\n".join(f"- {o}" for o in out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

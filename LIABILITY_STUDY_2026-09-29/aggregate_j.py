"""Study J aggregation: pre-registered checks J1-J5 (PREREGISTRATION_J.md), written and hashed before the run.

    python aggregate_j.py            # results_j/jobs
    python aggregate_j.py --smoke    # logic check on smoke outputs
Outputs: results_j/runs_j.csv, results_j/curves_j.csv, results_j/VERDICTS_J.md
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results_j"
N_BOOT = 10_000
rng = np.random.default_rng(20261005)


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
    L = lambda zone, rule, e: R[(R.region == zone) & (R.rule == rule)][f"L{e}"].to_numpy()  # noqa: E731
    n = {f"{z}-{r}": int(((R.region == z) & (R.rule == r)).sum()) for z in ("FI", "NO2") for r in ("FL", "LL", "ZF")}
    out = [f"runs: {len(R)}; per zone x rule: {n}"]
    z, f, ll = L("FI", "ZF", "0.1"), L("FI", "FL", "0.1"), L("FI", "LL", "0.1")
    gap = float(z.mean() - f.mean()); lo, hi = boot_diff(z, f)
    out.append(f"J1 FI: mean L_ZF(0.1) - L_FL(0.1) = {gap:.2f} [{lo:.2f}, {hi:.2f}] (need >= 8 and lower bound > 4) -> "
               + ("SUPPORTED" if gap >= 8 and lo > 4 else "NOT SUPPORTED"))
    j2 = int((L("FI", "ZF", "0.1") > L("FI", "ZF", "2")).sum())
    out.append(f"J2 FI: L_ZF(0.1) > L_ZF(2) in {j2}/{len(z)} runs (need >= 8 of 10) -> " + ("SUPPORTED" if j2 >= 8 else "NOT SUPPORTED"))
    d3 = float(z.mean() - ll.mean()); lo3, hi3 = boot_diff(z, ll)
    out.append(f"J3 FI: mean L_ZF(0.1) - L_LL(0.1) = {d3:.2f} [{lo3:.2f}, {hi3:.2f}] (need lower bound > 0) -> "
               + ("SUPPORTED" if lo3 > 0 else "NOT SUPPORTED"))
    a1, b1, a2, b2 = L("FI", "ZF", "0.25"), L("FI", "FL", "0.25"), L("NO2", "ZF", "0.25"), L("NO2", "FL", "0.25")
    g_fi, g_no = float(a1.mean() - b1.mean()), float(a2.mean() - b2.mean()); lo4, hi4 = boot_diff_of_diffs(a1, b1, a2, b2)
    out.append(f"J4 gap at e=0.25: FI {g_fi:.2f} vs NO2 {g_no:.2f}; difference {g_fi - g_no:.2f} [{lo4:.2f}, {hi4:.2f}] (need lower bound > 0) -> "
               + ("SUPPORTED" if lo4 > 0 else "NOT SUPPORTED"))
    z5, f5 = L("NO2", "ZF", "0.02"), L("NO2", "FL", "0.02")
    d5 = float(z5.mean() - f5.mean()); lo5, hi5 = boot_diff(z5, f5)
    out.append(f"J5 NO2: mean L_ZF(0.02) - L_FL(0.02) = {d5:.2f} [{lo5:.2f}, {hi5:.2f}] (need lower bound > 0) -> "
               + ("SUPPORTED" if lo5 > 0 else "NOT SUPPORTED"))
    tag = "_SMOKE" if a.smoke else ""
    R.drop(columns=["lev", "draws_reported_return_pct"]).to_csv(RES / f"runs_j{tag}.csv", index=False)
    cols = [c for c in R.columns if c.startswith("L")]
    C = R.groupby(["region", "rule"])[cols + ["drawmean_reported_return_pct", "drawmean_ledger_return_pct"]].mean()
    C.to_csv(RES / f"curves_j{tag}.csv")
    pd.set_option("display.width", 220)
    print(C.round(2).to_string())
    print("\n".join(out))
    (RES / f"VERDICTS_J{tag}.md").write_text("# Study J verdicts" + (" (SMOKE)" if a.smoke else "") + "\n\n"
                                            + "\n".join(f"- {o}" for o in out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

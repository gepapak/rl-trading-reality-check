"""Study I aggregation: pre-registered checks I1-I3 per learner configuration (PREREGISTRATION_I.md), written and hashed before the run.

    python aggregate_i.py            # results_i/jobs
    python aggregate_i.py --smoke    # logic check on smoke outputs
Outputs: results_i/runs_i.csv, results_i/curves_i.csv, results_i/VERDICTS_I.md
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results_i"
N_BOOT = 10_000
rng = np.random.default_rng(20261004)
CONFIGS = ("A2C-n256", "A2C-n64", "DQN-default")


def boot_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(N_BOOT)]
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rows = [json.load(open(f)) for f in glob.glob(str(RES / ("jobs_smoke" if a.smoke else "jobs") / "*.json"))]
    R = pd.DataFrame(rows)
    for k in rows[0]["lev"]:
        R[f"L{k}"] = R.lev.apply(lambda d: d[k])
    out = []
    for algo in CONFIGS:
        S = R[R.config == algo]
        L = lambda rule, e: S[S.rule == rule][f"L{e}"].to_numpy()  # noqa: E731
        n = {r: int((S.rule == r).sum()) for r in ("FL", "LL", "ZF")}
        out.append(f"{algo} runs per rule: {n}")
        z, f, ll = L("ZF", "0.1"), L("FL", "0.1"), L("LL", "0.1")
        if min(len(z), len(f), len(ll)) == 0:
            out.append(f"{algo}: missing cells, verdicts not computed")
            continue
        gap = float(z.mean() - f.mean()); lo, hi = boot_diff(z, f)
        out.append(f"I1 {algo}: mean L_ZF(0.1) - L_FL(0.1) = {gap:.2f} [{lo:.2f}, {hi:.2f}] (need >= 8 and lower bound > 4) -> "
                   + ("SUPPORTED" if gap >= 8 and lo > 4 else "NOT SUPPORTED"))
        i2 = int((L("ZF", "0.1") > L("ZF", "2")).sum())
        out.append(f"I2 {algo}: L_ZF(0.1) > L_ZF(2) in {i2}/{n['ZF']} runs (need >= 8 of 10) -> "
                   + ("SUPPORTED" if i2 >= 8 else "NOT SUPPORTED"))
        d3 = float(z.mean() - ll.mean()); lo3, hi3 = boot_diff(z, ll)
        out.append(f"I3 {algo}: mean L_ZF(0.1) - L_LL(0.1) = {d3:.2f} [{lo3:.2f}, {hi3:.2f}] (need lower bound > 0) -> "
                   + ("SUPPORTED" if lo3 > 0 else "NOT SUPPORTED"))
        out.append(f"   exploratory {algo}: L_ZF(0.1)=16 in {int((z == 16).sum())}/{len(z)}; L_FL(0.1)<=1 in {int((f <= 1).sum())}/{len(f)}; "
                   f"Cohen d (ZF vs FL at 0.1) = {gap / np.sqrt((z.var(ddof=1) + f.var(ddof=1)) / 2) if (z.var(ddof=1) + f.var(ddof=1)) > 0 else float('nan'):.2f}")
    tag = "_SMOKE" if a.smoke else ""
    R.drop(columns=["lev", "draws_reported_return_pct"]).to_csv(RES / f"runs_i{tag}.csv", index=False)
    cols = [c for c in R.columns if c.startswith("L")]
    C = R.groupby(["config", "rule"])[cols + ["drawmean_reported_return_pct", "drawmean_ledger_return_pct"]].mean()
    C.to_csv(RES / f"curves_i{tag}.csv")
    pd.set_option("display.width", 220)
    print(C.round(2).to_string())
    print("\n".join(out))
    (RES / f"VERDICTS_I{tag}.md").write_text("# Study I verdicts" + (" (SMOKE)" if a.smoke else "") + "\n\n"
                                            + "\n".join(f"- {o}" for o in out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

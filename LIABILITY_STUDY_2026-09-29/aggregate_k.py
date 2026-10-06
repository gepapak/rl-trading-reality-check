"""Study K aggregation: pre-registered checks K1-K3 (PREREGISTRATION_K.md), written and hashed before the run.

Gambling region of a run: T = the largest probed equity e such that the greedy leverage is at least 8 (half the
maximum) at every probed equity up to e; T = 0 if the leverage at zero equity is below 8.

    python aggregate_k.py            # results_k/jobs
    python aggregate_k.py --smoke    # logic check on smoke outputs
Outputs: results_k/runs_k.csv, results_k/curves_k.csv, results_k/VERDICTS_K.md
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results_k"
N_BOOT = 10_000
N_PERM = 10_000
rng = np.random.default_rng(20261006)


def region_T(lev: dict) -> float:
    es = sorted(lev, key=float)
    T = 0.0
    for e in es:
        if lev[e] >= 8:
            T = float(e)
        else:
            break
    return T


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rows = [json.load(open(f)) for f in glob.glob(str(RES / ("jobs_smoke" if a.smoke else "jobs") / "*.json"))]
    R = pd.DataFrame(rows)
    R["T"] = R.lev.apply(region_T)
    for k in rows[0]["lev"]:
        R[f"L{k}"] = R.lev.apply(lambda d: d[k])
    Z = R[R.rule == "ZF"]
    out = [f"runs: {len(R)}; per condition: {R.groupby(['rule', 'gamma']).size().to_dict()}"]
    g, T = Z.gamma.to_numpy(float), Z["T"].to_numpy(float)
    rho = spearman(g, T)
    perm = np.array([spearman(g, rng.permutation(T)) for _ in range(N_PERM)])
    p = float((np.sum(perm >= rho - 1e-12) + 1) / (N_PERM + 1))
    out.append(f"K1 Spearman(gamma, T) over zero-floor runs = {rho:.3f}, one-sided permutation p = {p:.4f} (need rho > 0 and p < 0.05) -> "
               + ("SUPPORTED" if rho > 0 and p < 0.05 else "NOT SUPPORTED"))
    hi, lo = Z[Z.gamma == 0.995]["T"].to_numpy(float), Z[Z.gamma == 0.95]["T"].to_numpy(float)
    if len(hi) and len(lo):
        d = float(hi.mean() - lo.mean())
        bs = [rng.choice(hi, len(hi)).mean() - rng.choice(lo, len(lo)).mean() for _ in range(N_BOOT)]
        blo, bhi = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
        out.append(f"K2 mean T(0.995) - T(0.95) = {d:.3f} [{blo:.3f}, {bhi:.3f}] (need lower bound > 0) -> "
                   + ("SUPPORTED" if blo > 0 else "NOT SUPPORTED"))
        out.append(f"K3 mean T(0.995) = {hi.mean():.3f} vs 1.5 x mean T(0.95) = {1.5 * lo.mean():.3f} (need T(0.995) >= 1.5 T(0.95)) -> "
                   + ("SUPPORTED" if hi.mean() >= 1.5 * lo.mean() and hi.mean() > 0 else "NOT SUPPORTED"))
    out.append("   exploratory: mean T by condition " + str(R.groupby(["rule", "gamma"])["T"].mean().round(3).to_dict()))
    tag = "_SMOKE" if a.smoke else ""
    R.drop(columns=["lev", "draws_reported_return_pct"]).to_csv(RES / f"runs_k{tag}.csv", index=False)
    cols = [c for c in R.columns if c.startswith("L")] + ["T", "drawmean_reported_return_pct", "drawmean_ledger_return_pct"]
    C = R.groupby(["rule", "gamma"])[cols].mean()
    C.to_csv(RES / f"curves_k{tag}.csv")
    pd.set_option("display.width", 260)
    print(C.round(2).to_string())
    print("\n".join(out))
    (RES / f"VERDICTS_K{tag}.md").write_text("# Study K verdicts" + (" (SMOKE)" if a.smoke else "") + "\n\n"
                                            + "\n".join(f"- {o}" for o in out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

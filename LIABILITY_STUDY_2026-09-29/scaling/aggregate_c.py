"""Study C aggregation: the pre-registered checks C1-C5 (PREREGISTRATION_C.md), applied verbatim.
Written after all 24 runs had finished (disclosed); the decision rules are those of the hashed pre-registration.

    python aggregate_c.py      # writes results/snapshots_c.csv, results/cells_c.csv, results/VERDICTS_C.md

Cell = (algorithm, zone); seed means over seeds 10, 11, 12. RS = P(0.1) - P(1).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIRST, LAST, MID = 100_000, 1_600_000, 400_000


def load() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(RES / "jobs" / "*.json")):
        if f.endswith(".partial.json"):
            continue
        d = json.load(open(f))
        for s in d["snapshots"]:
            rows.append(dict(liability=d["liability"], algo=d["algo"], seed=d["seed"], region=d["region"], step=s["step"],
                             RS=s["probe_e0.1"] - s["probe_e1"], P01=s["probe_e0.1"], P1=s["probe_e1"],
                             rep=s["drawmean_reported_return_pct"], led=s["drawmean_ledger_return_pct"],
                             ML=s["drawmean_mean_abs_lev_open"]))
    S = pd.DataFrame(rows)
    S["inflation"] = S.rep - S.led
    return S


def main() -> int:
    S = load()
    n = S.groupby(["liability", "algo", "region"]).seed.nunique()
    C = S.groupby(["liability", "algo", "region", "step"])[["RS", "P01", "P1", "rep", "led", "inflation", "ML"]].mean().reset_index()
    cells = sorted({(a, r) for a, r in zip(C.algo, C.region)})
    at = lambda l, a, r, st, col: float(C[(C.liability == l) & (C.algo == a) & (C.region == r) & (C.step == st)][col].iloc[0])  # noqa: E731
    out = [f"runs: {S.groupby(['liability','algo','seed','region']).ngroups} of 24; seeds per cell: {sorted(n.unique().tolist())}; cells: {len(cells)}"]
    # C1
    c1a = [at("ZF", a, r, LAST, "RS") > at("ZF", a, r, FIRST, "RS") for a, r in cells]
    rho = []
    for a, r in cells:
        g = C[(C.liability == "ZF") & (C.algo == a) & (C.region == r)].sort_values("step")
        rho.append(float(spearmanr(np.log(g.step), g.RS).correlation))
    c1b = [x > 0 for x in rho]
    out.append(f"C1 RS(ZF) 1.6M > 100k in {sum(c1a)}/4 (need >=3); Spearman(log steps, RS(ZF)) > 0 in {sum(c1b)}/4 (need >=3) "
               f"[rho by cell {dict(zip([f'{a}-{r}' for a, r in cells], [round(x, 2) for x in rho]))}] -> "
               + ("SUPPORTED" if sum(c1a) >= 3 and sum(c1b) >= 3 else "NOT SUPPORTED"))
    # C2
    c2 = [at("ZF", a, r, LAST, "P01") > at("ZF", a, r, FIRST, "P01") for a, r in cells]
    out.append(f"C2 P(0.1)(ZF) 1.6M > 100k in {sum(c2)}/4 (need >=3) -> " + ("SUPPORTED" if sum(c2) >= 3 else "NOT SUPPORTED"))
    # C3
    d = lambda l, a, r: at(l, a, r, LAST, "RS") - at(l, a, r, FIRST, "RS")  # noqa: E731
    c3 = [d("ZF", a, r) > d("FL", a, r) for a, r in cells]
    out.append(f"C3 change in RS larger under ZF than FL in {sum(c3)}/4 (need >=3) -> " + ("SUPPORTED" if sum(c3) >= 3 else "NOT SUPPORTED"))
    # C4
    c4 = [at("ZF", a, r, LAST, "inflation") > at("ZF", a, r, FIRST, "inflation") for a, r in cells]
    out.append(f"C4 inflation(ZF) 1.6M > 100k in {sum(c4)}/4 (need >=3) -> " + ("SUPPORTED" if sum(c4) >= 3 else "NOT SUPPORTED"))
    # C5
    c5 = [at("ZF", a, r, MID, "RS") > at("FL", a, r, MID, "RS") for a, r in cells]
    out.append(f"C5 RS(ZF) > RS(FL) at 400k in {sum(c5)}/4 (need >=3) -> " + ("SUPPORTED" if sum(c5) >= 3 else "NOT SUPPORTED"))
    S.to_csv(RES / "snapshots_c.csv", index=False)
    C.to_csv(RES / "cells_c.csv", index=False)
    pd.set_option("display.width", 220)
    print(C.pivot_table(index=["algo", "region", "liability"], columns="step", values="RS").round(2).to_string())
    print(C.pivot_table(index=["algo", "region", "liability"], columns="step", values="P01").round(1).to_string())
    print(C[C.liability == "ZF"].pivot_table(index=["algo", "region"], columns="step", values=["rep", "led"]).round(0).to_string())
    print("\n".join(out))
    (RES / "VERDICTS_C.md").write_text("# Study C verdicts\n\n" + "\n".join(f"- {o}" for o in out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

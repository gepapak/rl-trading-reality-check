"""Study E aggregation: pre-registered checks E1-E5 (PREREGISTRATION_E.md), written and hashed before the run.

    python aggregate_e.py            # results_e/jobs
    python aggregate_e.py --smoke    # logic check on the smoke outputs
Outputs: results_e/cells_e.csv, results_e/runs_e.csv, results_e/VERDICTS_E.md
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results_e"
MAXLEV = {"LOW": 16.0, "HIGH": 64.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rows = [json.load(open(f)) for f in glob.glob(str(RES / ("jobs_smoke" if a.smoke else "jobs") / "*.json"))]
    R = pd.DataFrame(rows)
    R["NML"] = R.drawmean_mean_abs_lev_open / R.levset.map(MAXLEV)
    R["TOP"] = R.drawmean_share_top_lev_open
    R["BUST"] = R.drawshare_bust
    C = R.groupby(["levset", "rule", "algo", "region"])[["NML", "TOP", "BUST", "drawmean_reported_return_pct",
                                                         "drawmean_ledger_return_pct"]].mean().reset_index()
    C["n_seeds"] = R.groupby(["levset", "rule", "algo", "region"]).seed.nunique().values
    W = C.pivot_table(index=["levset", "algo", "region"], columns="rule", values=["NML", "TOP", "BUST"])
    hi = W.loc["HIGH"] if "HIGH" in W.index.get_level_values(0) else W.iloc[0:0]
    lo = W.loc["LOW"] if "LOW" in W.index.get_level_values(0) else W.iloc[0:0]
    cnt = lambda s: int(s.sum())  # noqa: E731
    out = [f"cells HIGH {len(hi)}, LOW {len(lo)}; seeds per cell {sorted(C.n_seeds.unique().tolist())}"]
    e1a, e1b = cnt(hi.NML.LINLL > hi.NML.LINFL), cnt(hi.TOP.LINLL > hi.TOP.LINFL)
    out.append(f"E1 HIGH: NML(LINLL) > NML(LINFL) in {e1a}/4 and TOP(LINLL) > TOP(LINFL) in {e1b}/4 (need >=3 each) -> "
               + ("SUPPORTED" if e1a >= 3 and e1b >= 3 else "NOT SUPPORTED"))
    d_hi = hi.NML.LINLL - hi.NML.LINFL
    d_lo = (lo.NML.LINLL - lo.NML.LINFL).reindex(d_hi.index)
    e2 = cnt(d_hi > d_lo)
    out.append(f"E2 NML(LINLL)-NML(LINFL) larger in HIGH than LOW in {e2}/4 (need >=3) -> " + ("SUPPORTED" if e2 >= 3 else "NOT SUPPORTED"))
    e3a, e3b = cnt(hi.NML.LOGNF < hi.NML.LINLL), cnt(hi.TOP.LOGNF < hi.TOP.LINLL)
    out.append(f"E3 HIGH: NML(LOGNF) < NML(LINLL) in {e3a}/4 and TOP(LOGNF) < TOP(LINLL) in {e3b}/4 (need >=3 each) -> "
               + ("SUPPORTED" if e3a >= 3 and e3b >= 3 else "NOT SUPPORTED"))
    e4a, e4b = cnt(hi.BUST.LINLL > hi.BUST.LINFL), cnt(hi.BUST.LINLL > hi.BUST.LOGNF)
    out.append(f"E4 HIGH: BUST(LINLL) > BUST(LINFL) in {e4a}/4 and > BUST(LOGNF) in {e4b}/4 (need >=3 each) -> "
               + ("SUPPORTED" if e4a >= 3 and e4b >= 3 else "NOT SUPPORTED"))
    e5 = cnt(hi.NML.LOGNF <= hi.NML.LINFL + 0.05)
    out.append(f"E5 HIGH: NML(LOGNF) <= NML(LINFL) + 0.05 in {e5}/4 (need >=3) -> " + ("SUPPORTED" if e5 >= 3 else "NOT SUPPORTED"))
    tag = "_SMOKE" if a.smoke else ""
    R.drop(columns=[c for c in R.columns if c.startswith("draws_")]).to_csv(RES / f"runs_e{tag}.csv", index=False)
    C.to_csv(RES / f"cells_e{tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print(C.round(3).to_string(index=False))
    print("\n".join(out))
    (RES / f"VERDICTS_E{tag}.md").write_text("# Study E verdicts" + (" (SMOKE)" if a.smoke else "") + "\n\n"
                                            + "\n".join(f"- {o}" for o in out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

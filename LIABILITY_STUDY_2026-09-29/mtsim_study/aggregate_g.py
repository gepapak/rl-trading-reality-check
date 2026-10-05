"""Study G aggregation: pre-registered checks G1-G3 (PREREGISTRATION_G.md), written and hashed before the run.

    python aggregate_g.py [--smoke]
Outputs: results_g/runs_g.csv, results_g/VERDICTS_G.md
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results_g"
rng = np.random.default_rng(20261003)


def boot_diff(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a[rng.integers(0, len(a), (10_000, len(a)))].mean(axis=1) - b[rng.integers(0, len(b), (10_000, len(b)))].mean(axis=1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def boot_mean(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (10_000, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rows = [json.load(open(f)) for f in glob.glob(str(RES / ("jobs_smoke" if a.smoke else "jobs") / "*.json"))]
    R = pd.DataFrame(rows)
    for k in rows[0]["exposure"]:
        R[f"x{k}"] = R.exposure.apply(lambda d: d[k])
    R["xlow"] = R[["x0.02", "x0.05", "x0.1", "x0.25"]].mean(axis=1)
    R["gap"] = R.reported_return_pct - R.booked_return_pct
    o, f = R[R.condition == "ORIG"], R[R.condition == "FLPATCH"]
    out = [f"runs: ORIG {len(o)}, FLPATCH {len(f)}"]
    d1, (l1, h1) = o["x0.1"].mean() - f["x0.1"].mean(), boot_diff(o["x0.1"], f["x0.1"])
    out.append(f"G1 x_ORIG(0.1) - x_FLPATCH(0.1) = {d1:.3f} [{l1:.3f}, {h1:.3f}] (need lower bound > 0) -> " + ("SUPPORTED" if l1 > 0 else "NOT SUPPORTED"))
    d2, (l2, h2) = o.xlow.mean() - f.xlow.mean(), boot_diff(o.xlow, f.xlow)
    out.append(f"G2 mean x over e<=0.25, ORIG - FLPATCH = {d2:.3f} [{l2:.3f}, {h2:.3f}] (need lower bound > 0) -> " + ("SUPPORTED" if l2 > 0 else "NOT SUPPORTED"))
    l3, h3 = boot_mean(o.gap)
    out.append(f"G3 ORIG reported - booked = {o.gap.mean():.2f} pp [{l3:.2f}, {h3:.2f}] (need lower bound > 0) -> " + ("SUPPORTED" if l3 > 0 else "NOT SUPPORTED"))
    tag = "_SMOKE" if a.smoke else ""
    R.drop(columns=["exposure", "draws_reported", "draws_booked"]).to_csv(RES / f"runs_g{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.groupby("condition")[[c for c in R.columns if c.startswith("x")] + ["reported_return_pct", "booked_return_pct", "wiped_share"]].mean().round(3).to_string())
    print("\n".join(out))
    (RES / f"VERDICTS_G{tag}.md").write_text("# Study G verdicts" + (" (SMOKE)" if a.smoke else "") + "\n\n"
                                            + "\n".join(f"- {x}" for x in out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

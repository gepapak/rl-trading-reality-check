"""Study I pilot: distance of each learner configuration's final (1M-step) policy from the risk-neutral optimum.

Optimum (Table tab:dp, dp_theory.py): FL 0 at every equity level; ZF 16 at e <= 1 and 0 at e >= 2.
Distance: mean |L(e) - L*(e)| over the equity grid, averaged over FL and ZF. Training split, seed 0, DK1.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = ["0", "0.02", "0.05", "0.1", "0.25", "0.5", "1", "2", "3"]
OPT = {"FL": {e: 0.0 for e in GRID}, "ZF": {e: (16.0 if float(e) <= 1 else 0.0) for e in GRID}}
rows = {}
for f in glob.glob(os.path.join(HERE, "results_pilot_i", "*.json")):
    d = json.load(open(f))
    lev = d["snapshots"][-1]["lev"]
    mad = float(np.mean([abs(lev[e] - OPT[d["rule"]][e]) for e in GRID]))
    rows.setdefault((d["algo"], d["variant"]), {})[d["rule"]] = (mad, lev["0.1"], lev["2"], d["seconds"])
for (algo, var), r in sorted(rows.items()):
    fl, zf = r["FL"], r["ZF"]
    print(f"{algo:4s} {var:8s} MAD FL {fl[0]:5.2f}  ZF {zf[0]:5.2f}  mean {(fl[0] + zf[0]) / 2:5.2f} | "
          f"L(0.1) FL {fl[1]:5.2f} ZF {zf[1]:5.2f} | L_ZF(2) {zf[2]:5.2f} | seconds {max(fl[3], zf[3]):.0f}")

"""Verify the pre-registration hashes of Studies A-K and recompute the paper's headline numbers from the shipped results.

Needs only this repository (no training, no network). Run from the repository root:

    python tools/verify_studies.py

Part 1 checks every file listed in the two PREREGISTRATION_HASHES.txt files against its recorded SHA-256. A file hashed
more than once (a registration amended before its run) must match one of its recorded hashes; the eSett-derived panels
of Study J are not shipped and are reported as "not shipped" until rebuilt (LIABILITY_STUDY_2026-09-29/get_esett_j.py
and build_panel_j.py). Part 2 recomputes headline numbers from the aggregated result files.
Exit code 0 = every check passes, 1 = at least one check fails.
"""
from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LIAB = REPO / "LIABILITY_STUDY_2026-09-29"
GEN = REPO / "GENERALIZATION_STUDY_2026-09-29"
NOT_SHIPPED = {"data_panel_j/panel_FI.pkl", "data_panel_j/panel_NO2.pkl"}   # derived from eSett data (not redistributed)
FAILS: list[str] = []


def check_hashes(folder: Path) -> None:
    recorded: dict[str, set[str]] = defaultdict(set)
    for line in (folder / "PREREGISTRATION_HASHES.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1].startswith("*"):
            recorded[parts[-1][1:]].add(parts[-2])
    for rel, hashes in sorted(recorded.items()):
        p = folder / rel
        if not p.exists():
            status = "not shipped (rebuild to verify)" if rel in NOT_SHIPPED else "MISSING"
            if status == "MISSING":
                FAILS.append(f"{folder.name}/{rel}")
            print(f"[{'info' if rel in NOT_SHIPPED else 'FAIL'}] {folder.name}/{rel}: {status}")
            continue
        ok = hashlib.sha256(p.read_bytes()).hexdigest() in hashes
        print(f"[{'ok' if ok else 'FAIL'}] {folder.name}/{rel}")
        if not ok:
            FAILS.append(f"{folder.name}/{rel}")


def check(name: str, value: float, expected: float, tol: float) -> None:
    ok = bool(np.isfinite(value) and abs(value - expected) <= tol)
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {value:,.2f} (expected {expected:,.3f} +/- {tol})")
    if not ok:
        FAILS.append(name)


def headlines() -> None:
    c = pd.read_csv(LIAB / "results" / "cells_b.csv")
    z = c[(c.market == "NE") & (c.liability == "ZF")]
    check("B3 zero floor, placebo: lowest reported return (%)", z.rep.min(), 38.44, 0.01)
    check("B3 zero floor, placebo: highest reported return (%)", z.rep.max(), 147.83, 0.01)
    check("B3 zero floor, placebo: highest booked return (%)", z.led.max(), -15.08, 0.01)
    check("B3 zero floor, placebo: lowest booked return (%)", z.led.min(), -57.35, 0.01)
    cc = pd.read_csv(LIAB / "scaling" / "results" / "cells_c.csv")
    zc = cc[(cc.liability == "ZF") & (cc.step == 1_600_000)]
    check("C zero floor at 1.6M steps: lowest reported return (%)", zc.rep.min(), 124.72, 0.01)
    check("C zero floor at 1.6M steps: highest reported return (%)", zc.rep.max(), 182.82, 0.01)
    F = pd.read_csv(LIAB / "results_f" / "runs_f.csv")
    F = F[F.obsset == "EQ"]
    zf, fl = F[F.rule == "ZF"]["L0.1"], F[F.rule == "FL"]["L0.1"]
    check("F3 gap L_ZF(0.1) - L_FL(0.1)", zf.mean() - fl.mean(), 11.10, 0.005)
    check("F1 floor-trained agents at leverage 16 (of 10)", float((zf == 16).sum()), 10, 0)
    I = pd.read_csv(LIAB / "results_i" / "runs_i.csv")
    for cfg, exp in (("A2C-n256", 8.53), ("A2C-n64", 11.95), ("DQN-default", 7.15)):
        g = I[I.config == cfg]
        check(f"I1 gap {cfg}", g[g.rule == "ZF"]["L0.1"].mean() - g[g.rule == "FL"]["L0.1"].mean(), exp, 0.005)
    check("I pooled gap (exploratory)", I[I.rule == "ZF"]["L0.1"].mean() - I[I.rule == "FL"]["L0.1"].mean(), 9.21, 0.005)
    J = pd.read_csv(LIAB / "results_j" / "runs_j.csv")

    def L(zone: str, rule: str, e: str) -> pd.Series:
        return J[(J.region == zone) & (J.rule == rule)].sort_values("seed")[f"L{e}"]
    check("J1 Finland gap at 0.1", L("FI", "ZF", "0.1").mean() - L("FI", "FL", "0.1").mean(), 8.18, 0.005)
    check("J2 Finland L_ZF(0.1) > L_ZF(2) (runs of 10)", float((L("FI", "ZF", "0.1").values > L("FI", "ZF", "2").values).sum()), 9, 0)
    check("J4 Finland gap at 0.25", L("FI", "ZF", "0.25").mean() - L("FI", "FL", "0.25").mean(), 8.47, 0.005)
    check("J4 Norway gap at 0.25", L("NO2", "ZF", "0.25").mean() - L("NO2", "FL", "0.25").mean(), 0.625, 0.005)
    H = pd.read_csv(LIAB / "results_h" / "runs_h.csv")
    h = H[(H.obsset == "EQ") & (H.rule == "FL")]
    cols = [col for col in H.columns if col.startswith("L")]
    check("H2 full-liability agents at leverage <= 0.25 everywhere (of 10)", float((h[cols].max(axis=1) <= 0.25).sum()), 10, 0)
    G = pd.read_csv(LIAB / "mtsim_study" / "results_g" / "runs_g.csv")
    check("G gym-mtsim runs", float(len(G)), 20, 0)
    check("G largest exposure of any agent", float(G[[col for col in G.columns if col.startswith("x")]].abs().max().max()), 0.0, 0)
    K = pd.read_csv(LIAB / "results_k" / "runs_k.csv")
    for g, exp in ((0.95, 1.35), (0.98, 1.475), (0.99, 1.85), (0.995, 1.425)):
        check(f"K mean gambling region, zero floor, gamma {g}", K[(K.rule == "ZF") & (K.gamma == g)]["T"].mean(), exp, 0.005)
    check("K full-liability agents with no gambling region (of 20)", float((K[K.rule == "FL"]["T"] == 0).sum()), 17, 0)
    BM = pd.read_csv(LIAB / "results_k" / "best_moment.csv")
    check("Theorem 1: largest KS distance, reported vs running maximum (four zones)", BM.ks_D.max(), 0.0785, 0.0005)
    check("Theorem 1: DK1 mean reported equity (allocations)", BM[BM.zone == "DK1"].mean_reported_K.item(), 2.551, 0.001)
    D = pd.read_csv(LIAB / "env_audit" / "coding_d.csv")
    d = D[D.included == "yes"]
    check("D included environments", float(len(d)), 18, 0)
    check("D environments meeting both conditions", float((d.gambling_prone.str.upper() == "YES").sum()), 1, 0)


def main() -> int:
    print("== Part 1: pre-registration hashes")
    check_hashes(GEN)
    check_hashes(LIAB)
    print("\n== Part 2: headline numbers")
    headlines()
    print("\nall checks pass" if not FAILS else f"\n{len(FAILS)} check(s) FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

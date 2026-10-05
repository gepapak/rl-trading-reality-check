"""Study A aggregation: verdicts, false positives and the pre-registered checks A1-A5 (PREREGISTRATION_A.md).

    python aggregate_study.py            # full run (test split)
    python aggregate_study.py --smoke    # logic check on the smoke outputs (train split, tiny budgets)

Outputs (results/): verdicts.csv, rl_seed_level.csv, VERDICTS_A.md
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DESIGNS = {"E1": ["E1-small", "E1-large"], "E2": ["E2"], "E3": ["E3"]}


def iqm(v) -> float:
    v = np.sort(np.asarray(v, float))
    k = int(np.floor(len(v) * 0.25))
    return float(v[k:len(v) - k].mean()) if len(v) else np.nan


def load(smoke: bool):
    rl = pd.DataFrame([json.load(open(f)) for f in glob.glob(str(RES / ("jobs_smoke" if smoke else "jobs") / "*.json"))])
    rules = pd.read_csv(RES / ("rules_SMOKE_train_split.csv" if smoke else "rules.csv"))
    return rl, rules


def verdict_table(rl: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (env, region, regime), g in rl.groupby(["env", "region", "regime"]):
        means = g.groupby("algo")[["reported", "reality"]].mean()
        r = rules[(rules.env == env) & (rules.region == region) & (rules.regime == regime)]
        if r.empty:
            continue
        best_rl_rep, best_rl_real = float(means.reported.max()), float(means.reality.max())
        best_rule_rep, best_rule_real = float(r.reported.max()), float(r.reality.max())
        rep_win, real_win = best_rl_rep > best_rule_rep, best_rl_real > best_rule_real
        rows.append(dict(env=env, region=region, regime=regime, n_rl=len(g),
                         best_rl_algo_reported=means.reported.idxmax(), best_rl_reported=best_rl_rep,
                         best_rule_reported=best_rule_rep, best_rule_name_reported=r.loc[r.reported.idxmax(), "policy"],
                         best_rl_reality=best_rl_real, best_rule_reality=best_rule_real,
                         reported_verdict_rl_wins=rep_win, reality_verdict_rl_wins=real_win,
                         false_positive=bool(regime != "S0" and rep_win and not real_win),
                         rl_reported_median=float(g.reported.median()), rl_reported_iqm=iqm(g.reported),
                         rl_reality_median=float(g.reality.median())))
    return pd.DataFrame(rows)


def checks(V: pd.DataFrame, rl: pd.DataFrame, rules: pd.DataFrame) -> list[str]:
    out = []
    fp = lambda env, reg: V[(V.env == env) & (V.regime == reg)].false_positive.tolist()
    # A1
    a1 = fp("E2", "S3")
    out.append(f"A1 E2 under S3: false positive by region {dict(zip(V[(V.env=='E2')&(V.regime=='S3')].region, a1))} (need >=1) -> "
               + ("SUPPORTED" if any(a1) else "NOT SUPPORTED"))
    # A2: ledger check on every E2 evaluation (RL reported runs + rules reported runs)
    e2 = pd.concat([rl[rl.env == "E2"][["regime", "rep_floor_bound", "rep_max_gap_pp"]],
                    rules[rules.env == "E2"][["regime", "rep_floor_bound", "rep_max_gap_pp"]]], ignore_index=True)
    bound = e2[e2.rep_floor_bound.astype(bool)]
    s0 = e2[e2.regime == "S0"]
    share_bound = float((bound.rep_max_gap_pp > 5).mean()) if len(bound) else np.nan
    share_s0 = float((s0.rep_max_gap_pp > 5).mean()) if len(s0) else np.nan
    out.append(f"A2 ledger check: flagged {share_bound if share_bound == share_bound else 'n/a'} of {len(bound)} floor-bound E2 runs (need >=0.9); "
               f"flagged {share_s0} of {len(s0)} S0 runs (need 0) -> "
               + ("SUPPORTED" if len(bound) and share_bound >= 0.9 and share_s0 == 0 else
                  ("NOT TESTABLE (floor never bound)" if not len(bound) else "NOT SUPPORTED")))
    # A3: best reported (max over RL seed-means and rules), S1 vs S0
    def best(env, region, regime):
        g = rl[(rl.env == env) & (rl.region == region) & (rl.regime == regime)]
        r = rules[(rules.env == env) & (rules.region == region) & (rules.regime == regime)]
        return max(float(g.groupby("algo").reported.mean().max()), float(r.reported.max()))
    ratios = {(env, reg): best(env, reg, "S1") / best(env, reg, "S0") for env in ("E1-small", "E1-large") for reg in ("DK1", "DK2")}
    ok3 = all(ratios[("E1-large", r)] >= 2 for r in ("DK1", "DK2")) and all(ratios[("E1-small", r)] < 1.10 for r in ("DK1", "DK2"))
    out.append("A3 price-taker ratio S1/S0 of best reported profit: " + ", ".join(f"{e} {r} {v:.2f}" for (e, r), v in ratios.items())
               + " (need large >=2, small <1.10, both regions) -> " + ("SUPPORTED" if ok3 else "NOT SUPPORTED"))
    # A4: look-ahead false positives in E1-large, E2, E3
    a4 = {d: any(fp(d, "S4")) for d in ("E1-large", "E2", "E3")}
    out.append(f"A4 look-ahead false positive per design {a4} (need >=2 of 3) -> " + ("SUPPORTED" if sum(a4.values()) >= 2 else "NOT SUPPORTED"))
    # A5: any shortcut false positive per design
    a5 = {d: bool(V[V.env.isin(envs)].false_positive.any()) for d, envs in DESIGNS.items()}
    out.append(f"A5 any-shortcut false positive per design {a5} (need >=2 of 3) -> " + ("SUPPORTED" if sum(a5.values()) >= 2 else "NOT SUPPORTED"))
    # reported without prediction: strict verdicts
    s0v = V[V.regime == "S0"][["env", "region", "best_rl_reported", "best_rule_reported", "reported_verdict_rl_wins"]]
    out.append("Strict (S0) verdicts, RL wins?: " + ", ".join(f"{a} {b}: {c}" for a, b, c in s0v[["env", "region", "reported_verdict_rl_wins"]].values))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rl, rules = load(a.smoke)
    V = verdict_table(rl, rules)
    tag = "_SMOKE" if a.smoke else ""
    V.to_csv(RES / f"verdicts{tag}.csv", index=False)
    rl.to_csv(RES / f"rl_seed_level{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(V[["env", "region", "regime", "n_rl", "best_rl_reported", "best_rule_reported", "best_rl_reality", "best_rule_reality",
             "reported_verdict_rl_wins", "reality_verdict_rl_wins", "false_positive"]].round(1).to_string(index=False))
    C = checks(V, rl, rules)
    print("\n".join(C))
    (RES / f"VERDICTS_A{tag}.md").write_text("# Study A verdicts" + (" (SMOKE: train split, tiny budgets)" if a.smoke else "") + "\n\n"
                                             + "\n".join(f"- {c}" for c in C) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

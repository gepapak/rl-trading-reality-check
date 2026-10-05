"""Study E manuscript table (from results_e/runs_e.csv): normalized leverage, top-level share and bankruptcy share by
rule and leverage set, pooled and per algorithm, with seed-level permutation tests (exploratory) for the key contrasts.

    python make_studyE_outputs.py   -> sections/tab_studyE.tex and printed statistics
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = pd.read_csv(HERE / "results_e" / "runs_e.csv")
OUT = next((HERE.parent / "Overleaf Projects (1 items)").glob("*"), HERE / "outputs") / "sections" / "tab_studyE.tex"
RULES = [("LINFL", "Linear, full liability"), ("LINLL", "Linear, capped loss"), ("LOGNF", "Log, bust unrewarded")]
rng = np.random.default_rng(20261002)


def perm(a: np.ndarray, b: np.ndarray, n: int = 20_000) -> float:
    obs, pool, c = a.mean() - b.mean(), np.r_[a, b], 0
    for _ in range(n):
        p = rng.permutation(pool)
        c += (p[: len(a)].mean() - p[len(a):].mean()) >= obs
    return (c + 1) / (n + 1)


def main() -> int:
    lines = [r"\begin{table}[!htbp]", r"\centering",
             r"\caption{Equity-scaled study in the placebo market: normalized mean leverage (NML, mean $|\ell|$ over the set's maximum), share of quarter-hours at the maximum level (TOP) and share of evaluation draws ending in bankruptcy (BUST), means over 20 agents per rule and leverage set (two algorithms, five seeds, two zones), with PPO and DQN means in brackets.}",
             r"\label{tab:studyE}", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{@{}llccc@{}}", r"\toprule", r"Set & Rule & NML [PPO, DQN] & TOP [PPO, DQN] & BUST [PPO, DQN] \\", r"\midrule"]
    for lev in ("LOW", "HIGH"):
        for i, (rule, name) in enumerate(RULES):
            g = R[(R.levset == lev) & (R.rule == rule)]
            cell = lambda m: f"{g[m].mean():.2f} [{g[g.algo=='PPO'][m].mean():.2f}, {g[g.algo=='DQN'][m].mean():.2f}]"  # noqa: E731
            lines.append(f"{lev if i == 0 else ''} & {name} & {cell('NML')} & {cell('TOP')} & {cell('BUST')}" + r" \\")
        if lev == "LOW":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(OUT)
    for lev in ("HIGH", "LOW"):
        g = R[R.levset == lev]
        for a, b in (("LINLL", "LINFL"), ("LINLL", "LOGNF")):
            for m in ("NML", "TOP", "BUST"):
                x, y = g[g.rule == a][m].to_numpy(), g[g.rule == b][m].to_numpy()
                print(f"{lev} {m} {a}-{b}: {x.mean() - y.mean():+.3f}, one-sided p = {perm(x, y):.3f}")
    h = R[R.levset == "HIGH"]
    print("HIGH reported - ledger (pp), mean by rule:", h.assign(g=h.drawmean_reported_return_pct - h.drawmean_ledger_return_pct)
          .groupby("rule").g.mean().round(1).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

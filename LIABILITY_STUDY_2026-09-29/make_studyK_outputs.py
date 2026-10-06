"""Study K figure (from results_k/runs_k.csv, results_k/dp_full_b2000_g4.csv and results_k/theory_k.csv).

    fig_patience.pdf  gambling region T of every floor-trained and full-liability agent by discount factor, the mean of
                      the floor-trained agents with 95% bootstrap intervals, the optimal region (full-distribution DP,
                      mean of DK1 and DK2) and the closed-form reflected-diffusion barrier.
Palette as fig_wellposed.pdf.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = next((HERE.parent / "Overleaf Projects (1 items)").glob("*"), HERE / "outputs") / "figures" / "fig_patience.pdf"
R = pd.read_csv(HERE / "results_k" / "runs_k.csv")
D = pd.read_csv(HERE / "results_k" / "dp_full_b2000_g4.csv")
C = pd.read_csv(HERE / "results_k" / "theory_k.csv")
GAMMAS = [0.95, 0.98, 0.99, 0.995]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ZF_C, FL_C, OPT_C = "#1baf7a", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
rng = np.random.default_rng(20261005)


def main() -> int:
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(GAMMAS))
    means, lo, hi = [], [], []
    for i, g in enumerate(GAMMAS):
        t = R[(R.rule == "ZF") & (R.gamma == g)]["T"].to_numpy()
        ax.scatter(np.full(len(t), i) + rng.uniform(-0.12, 0.12, len(t)), t, s=10, color=ZF_C, alpha=0.45, lw=0)
        bs = [rng.choice(t, len(t)).mean() for _ in range(10000)]
        means.append(t.mean()); lo.append(np.quantile(bs, 0.025)); hi.append(np.quantile(bs, 0.975))
        f = R[(R.rule == "FL") & (R.gamma == g)]["T"].to_numpy()
        if len(f):
            ax.scatter(np.full(len(f), i) + 0.28 + rng.uniform(-0.05, 0.05, len(f)), f, s=10, color=FL_C, alpha=0.6, lw=0,
                       marker="s", label="Full liability, each agent" if i == 0 else None)
    ax.errorbar(x, means, yerr=[np.array(means) - lo, np.array(hi) - means], color=ZF_C, lw=1.4, marker="^", ms=5,
                mec="white", mew=0.5, capsize=2.5, label="Zero floor, learned (mean, 95% interval)")
    opt = [D[(D.gamma == g) & D.zone.isin(["DK1", "DK2"])].threshold_K.mean() for g in GAMMAS]
    cf = [C[C.gamma == g].closed_form_barrier_K.mean() for g in GAMMAS]
    ax.plot(x, opt, color=OPT_C, lw=1.4, ls="--", marker="o", ms=3.5, label="Optimal region (dynamic programming)")
    ax.plot(x, cf, color=MUTED, lw=1.0, ls=":", label="Closed-form barrier (Proposition 8)")
    ax.set_xticks(x, [f"{g:g}" for g in GAMMAS])
    ax.set_xlabel(r"Discount factor $\gamma$")
    ax.set_ylabel("Gambling region (allocations)")
    ax.set_ylim(-0.2, 5.3)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT, {g: round(m, 2) for g, m in zip(GAMMAS, means)}, "opt", [round(o, 2) for o in opt], "cf", [round(c, 2) for c in cf])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

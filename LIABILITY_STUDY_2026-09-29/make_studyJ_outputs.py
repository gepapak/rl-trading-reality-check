"""Study J figure (from results_j/runs_j.csv and results_j/dp_j.csv).

    fig_markets.pdf  learned leverage by equity (equity as the only input) in FI and NO2: FL / LL / ZF, mean of 10 runs
                     with 95% bootstrap bands, and each zone's risk-neutral optimum (dashed).
Palette and styling as fig_wellposed.pdf (make_studyF_outputs.py).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = pd.read_csv(HERE / "results_j" / "runs_j.csv")
# optimum from the full training distribution (dp_full.py); the pre-registered dp_j.csv used a 3,000-quarter-hour sample
D = pd.read_csv(HERE / "results_k" / "dp_full_b2000_g4.csv")
D = D[D.gamma == 0.99]
OUT = next((HERE.parent / "Overleaf Projects (1 items)").glob("*"), HERE / "outputs") / "figures" / "fig_markets.pdf"
RULE = {"FL": ("Full liability", "#2a78d6", "o"), "LL": ("Capped loss, stop at zero", "#eb6834", "s"),
        "ZF": ("Zero floor, keep trading", "#1baf7a", "^")}
E = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
rng = np.random.default_rng(20261005)


def band(x: np.ndarray) -> tuple[float, float]:
    m = x[rng.integers(0, len(x), (10_000, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, zone, title in zip(axes, ("FI", "NO2"), ("(a) Finland, mean |spread| 60 EUR/MWh",
                                                      "(b) Norway NO2, mean |spread| 12 EUR/MWh")):
        for rule, (name, color, mk) in RULE.items():
            g = R[(R.region == zone) & (R.rule == rule)]
            M = np.array([[r[f"L{e:g}"] for e in E] for _, r in g.iterrows()])
            lo, hi = np.array([band(M[:, j]) for j in range(len(E))]).T
            ax.fill_between(E, lo, hi, color=color, alpha=0.15, lw=0)
            ax.plot(E, M.mean(axis=0), color=color, lw=1.4, marker=mk, ms=4, mec="white", mew=0.5, label=f"{name}, learned")
        for rule in ("ZF", "FL"):
            opt = [float(D[D.zone == zone][f"opt_{e:g}"].iloc[0]) if rule == "ZF" else 0.0 for e in E]
            ax.plot(E, opt, color=RULE[rule][1], lw=1.0, ls="--", label=f"{RULE[rule][0]}, optimum")
        ax.set_xscale("log")
        ax.set_xticks(E, ["0.02", "0.05", "0.1", "0.25", "0.5", "1", "2", "3"], fontsize=6.5)
        ax.minorticks_off()
        ax.set_xlabel("Equity / allocation")
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_ylim(-0.5, 16.8)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("|Leverage| chosen")
    axes[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(1.05, -0.24), ncol=3)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

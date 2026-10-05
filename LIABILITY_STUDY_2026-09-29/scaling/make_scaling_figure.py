"""Study C figure (from results/snapshots_c.csv): resurrection slope and placebo-market returns over training.

    fig_training_scale.pdf  (a) RS by training steps, zero floor vs full liability, PPO and DQN (six runs each: 3 seeds x 2 zones)
                            (b) zero floor: reported vs booked return in the placebo market by training steps

Palette: slots 1 (full liability) and 3 (zero floor) of the dataviz reference palette, as in the Study B figures;
algorithms differ by line style and marker, reported vs booked by filled vs hollow markers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
S = pd.read_csv(HERE / "results" / "snapshots_c.csv")
OUT = next((HERE.parents[1] / "Overleaf Projects (1 items)").glob("*"), HERE.parent / "outputs") / "figures" / "fig_training_scale.pdf"
COL = {"FL": ("Full liability", "#2a78d6"), "ZF": ("Zero floor", "#1baf7a")}
ALG = {"PPO": ("-", "o"), "DQN": ("--", "s")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
rng = np.random.default_rng(20260930)
steps = sorted(S.step.unique())


def band(x: np.ndarray) -> tuple[float, float]:
    m = x[rng.integers(0, len(x), (10_000, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    for rule, (rname, color) in COL.items():
        for algo, (ls, mk) in ALG.items():
            g = S[(S.liability == rule) & (S.algo == algo)]
            mean = [g[g.step == s].RS.mean() for s in steps]
            lo, hi = np.array([band(g[g.step == s].RS.to_numpy()) for s in steps]).T
            ax.fill_between(steps, lo, hi, color=color, alpha=0.10, lw=0)
            ax.plot(steps, mean, color=color, ls=ls, marker=mk, ms=3.5, lw=1.3, mec="white", mew=0.4, label=f"{rname}, {algo}")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel("Resurrection slope $P(0.1)-P(1)$")
    ax.set_title("(a) Extra leverage near zero equity", fontsize=8, color=INK)
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    ax = axes[1]
    z = S[S.liability == "ZF"]
    for algo, (ls, mk) in ALG.items():
        g = z[z.algo == algo]
        for col, face, lab in (("rep", COL["ZF"][1], "reported"), ("led", "white", "booked")):
            mean = [g[g.step == s][col].mean() for s in steps]
            ax.plot(steps, mean, color=COL["ZF"][1], ls=ls, marker=mk, ms=4, mfc=face, mec=COL["ZF"][1], lw=1.2,
                    label=f"{algo}, {lab}")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel("Return in the placebo market (%)")
    ax.set_title("(b) Zero floor: reported vs booked", fontsize=8, color=INK)
    ax.legend(fontsize=6, loc="center right", ncol=2)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(steps, ["100k", "200k", "400k", "800k", "1.6M"], fontsize=6.5)
        ax.minorticks_off()
        ax.set_xlabel("Training steps")
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

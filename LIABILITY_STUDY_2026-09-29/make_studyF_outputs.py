"""Study F figure and statistics (from results_f/runs_f.csv and results/dp_theory.csv).

    fig_wellposed.pdf  learned leverage by equity (equity-only input), FL / LL / ZF, mean of 10 runs with 95% bootstrap
                       bands, and the risk-neutral optimum (dashed); second panel: the same with all 11 inputs.
Palette: categorical slots 1-3 of the dataviz reference palette, one per liability rule (as in the Study B figures).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = pd.read_csv(HERE / "results_f" / "runs_f.csv")
D = pd.read_csv(HERE / "results" / "dp_theory.csv")
OUT = next((HERE.parent / "Overleaf Projects (1 items)").glob("*"), HERE / "outputs") / "figures" / "fig_wellposed.pdf"
RULE = {"FL": ("Full liability", "#2a78d6", "o"), "LL": ("Capped loss, stop at zero", "#eb6834", "s"),
        "ZF": ("Zero floor, keep trading", "#1baf7a", "^")}
E = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
rng = np.random.default_rng(20261003)


def band(x: np.ndarray) -> tuple[float, float]:
    m = x[rng.integers(0, len(x), (10_000, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def cohen_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s)


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, obs, title in zip(axes, ("EQ", "FULL"), ("(a) Equity as the only input", "(b) All 11 inputs")):
        for rule, (name, color, mk) in RULE.items():
            g = R[(R.obsset == obs) & (R.rule == rule)]
            M = np.array([[r[f"L{e:g}"] for e in E] for _, r in g.iterrows()])
            lo, hi = np.array([band(M[:, j]) for j in range(len(E))]).T
            ax.fill_between(E, lo, hi, color=color, alpha=0.15, lw=0)
            ax.plot(E, M.mean(axis=0), color=color, lw=1.4, marker=mk, ms=4, mec="white", mew=0.5, label=f"{name}, learned")
        if obs == "EQ":
            for rule in ("ZF", "FL"):
                opt = [D[(D.rule == rule) & np.isclose(D.equity, e)].opt_abs_lev.mean() for e in E]
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
    eq = R[R.obsset == "EQ"]
    z, f, l = (eq[eq.rule == r]["L0.1"].to_numpy() for r in ("ZF", "FL", "LL"))
    print(f"F3 Cohen d ZF vs FL at 0.1: {cohen_d(z, f):.2f}; ZF vs LL: {cohen_d(z, l):.2f}")
    for rule in ("FL", "LL", "ZF"):
        g = eq[eq.rule == rule]
        print(rule, "L(0.1) by run:", sorted(g["L0.1"].round(2).tolist()), "| per zone mean:", g.groupby("region")["L0.1"].mean().round(2).to_dict())
    print("placebo returns EQ (reported, ledger) by rule:", eq.groupby("rule")[["drawmean_reported_return_pct", "drawmean_ledger_return_pct"]].mean().round(1).to_dict("index"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

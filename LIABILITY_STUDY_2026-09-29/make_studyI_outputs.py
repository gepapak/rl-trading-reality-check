"""Study I figure and statistics (from results_i/runs_i.csv, results_f/runs_f.csv and results/dp_theory.csv).

    fig_learners.pdf  learned leverage by equity (equity as the only input) for four learners: PPO (Study F, EQ),
                      A2C-n256, A2C-n64 and DQN-default (Study I); FL / LL / ZF, mean of 10 runs with 95% bootstrap
                      bands, and the risk-neutral optimum (dashed).
Palette and styling as fig_wellposed.pdf (make_studyF_outputs.py).

    python make_studyI_outputs.py            # final results
    python make_studyI_outputs.py --smoke    # layout check on smoke outputs (writes fig_learners_SMOKE.pdf here)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = pd.read_csv(HERE / "results" / "dp_theory.csv")
FIGDIR = next((HERE.parent / "Overleaf Projects (1 items)").glob("*"), HERE / "outputs") / "figures"
RULE = {"FL": ("Full liability", "#2a78d6", "o"), "LL": ("Capped loss, stop at zero", "#eb6834", "s"),
        "ZF": ("Zero floor, keep trading", "#1baf7a", "^")}
PANELS = [("PPO", "(a) PPO (Study F)"), ("A2C-n256", "(b) A2C, rollout 256"), ("A2C-n64", "(c) A2C, rollout 64"),
          ("DQN-default", "(d) DQN, library defaults")]
E = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
rng = np.random.default_rng(20261004)


def band(x: np.ndarray) -> tuple[float, float]:
    m = x[rng.integers(0, len(x), (10_000, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def cohen_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    I = pd.read_csv(HERE / "results_i" / ("runs_i_SMOKE.csv" if a.smoke else "runs_i.csv"))
    F = pd.read_csv(HERE / "results_f" / "runs_f.csv")
    F = F[F.obsset == "EQ"].assign(config="PPO")
    R = pd.concat([F, I], ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), sharey=True, sharex=True)
    for ax, (cfg, title) in zip(axes.ravel(), PANELS):
        for rule, (name, color, mk) in RULE.items():
            g = R[(R.config == cfg) & (R.rule == rule)]
            if g.empty:
                continue
            M = np.array([[r[f"L{e:g}"] for e in E] for _, r in g.iterrows()])
            lo, hi = np.array([band(M[:, j]) for j in range(len(E))]).T
            ax.fill_between(E, lo, hi, color=color, alpha=0.15, lw=0)
            ax.plot(E, M.mean(axis=0), color=color, lw=1.4, marker=mk, ms=4, mec="white", mew=0.5, label=f"{name}, learned")
        for rule in ("ZF", "FL"):
            opt = [D[(D.rule == rule) & np.isclose(D.equity, e)].opt_abs_lev.mean() for e in E]
            ax.plot(E, opt, color=RULE[rule][1], lw=1.0, ls="--", label=f"{RULE[rule][0]}, optimum")
        ax.set_xscale("log")
        ax.set_xticks(E, ["0.02", "0.05", "0.1", "0.25", "0.5", "1", "2", "3"], fontsize=6.5)
        ax.minorticks_off()
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_ylim(-0.5, 16.8)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("Equity / allocation")
    for ax in axes[:, 0]:
        ax.set_ylabel("|Leverage| chosen")
    axes[1, 0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(1.05, -0.24), ncol=3)
    fig.subplots_adjust(wspace=0.10, hspace=0.28)
    out = (HERE / "fig_learners_SMOKE.pdf") if a.smoke else (FIGDIR / "fig_learners.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)
    for cfg, _ in PANELS:
        g = R[R.config == cfg]
        z, f, l = (g[g.rule == r]["L0.1"].to_numpy() for r in ("ZF", "FL", "LL"))
        if min(len(z), len(f), len(l)) < 2:
            continue
        print(f"{cfg}: L(0.1) ZF {z.mean():.2f} FL {f.mean():.2f} LL {l.mean():.2f}; d(ZF,FL) {cohen_d(z, f):.2f}; "
              f"ZF=16 {int((z == 16).sum())}/{len(z)}; FL<=1 {int((f <= 1).sum())}/{len(f)}; "
              f"ZF L(0.1)>L(2) {int((g[g.rule == 'ZF']['L0.1'].to_numpy() > g[g.rule == 'ZF']['L2'].to_numpy()).sum())}/{len(z)}")
        print("   curve ZF:", g[g.rule == "ZF"][[f"L{e:g}" for e in E]].mean().round(1).tolist(),
              "| FL:", g[g.rule == "FL"][[f"L{e:g}" for e in E]].mean().round(1).tolist())
        print("   placebo returns (reported, ledger):",
              g.groupby("rule")[["drawmean_reported_return_pct", "drawmean_ledger_return_pct"]].mean().round(1).to_dict("index"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Manuscript outputs for Study B and the theory check, built only from results/runs_b.csv, results/cells_b.csv and
results/dp_theory.csv.

    fig_leverage_curves.pdf   learned leverage at fixed equity (PPO, DQN) and the theoretical incentive, NE market
    fig_false_profit.pdf      reported vs ledger return in the NE market, by liability rule and algorithm
    tab_dp.tex                risk-neutral optimal leverage and incentive by equity level (dp_theory.py)
    tab_false_profit.tex      NE-market reported/ledger returns per cell (appendix)

Palette: first three categorical slots of the dataviz reference palette (validated all-pairs, light mode), one per
liability rule everywhere; rules also differ by marker shape, and reported vs ledger by filled vs hollow markers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OVERLEAF = next((HERE.parent / "Overleaf Projects (1 items)").glob("*"), HERE / "outputs")
FIG, SEC = OVERLEAF / "figures", OVERLEAF / "sections"
R = pd.read_csv(HERE / "results" / "runs_b.csv")
C = pd.read_csv(HERE / "results" / "cells_b.csv")
D = pd.read_csv(HERE / "results" / "dp_theory.csv")
RULE = {"FL": ("Full liability", "#2a78d6", "o"), "LL": ("Capped loss, stop at zero", "#eb6834", "s"),
        "ZF": ("Zero floor, keep trading", "#1baf7a", "^")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
E = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
rng = np.random.default_rng(20260930)


def boot(x: np.ndarray, n: int = 10_000) -> tuple[float, float]:
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def draw_matrices(liab: str, algo: str, col: str) -> list[np.ndarray]:
    """Placebo market: one seeds x draws matrix per zone (the 20 sign draws are shared by all seeds of a zone)."""
    import glob
    import json
    mats = []
    for region in ("DK1", "DK2"):
        rows = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / "results" / "jobs" / f"{liab}__NE__{algo}__s*__{region}.json")))]
        mats.append(np.array([r[f"draws_{col}"] for r in rows]))
    return mats


def boot2(mats: list[np.ndarray], n: int = 10_000) -> tuple[float, float]:
    """Two-way bootstrap: resample seeds and shared draws within each zone, average the zone means."""
    out = np.empty(n)
    for b in range(n):
        ms = []
        for m in mats:
            i, j = rng.integers(0, m.shape[0], m.shape[0]), rng.integers(0, m.shape[1], m.shape[1])
            ms.append(m[np.ix_(i, j)].mean())
        out[b] = np.mean(ms)
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def fig_leverage_curves() -> None:
    ne = R[R.market == "NE"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, algo in zip(axes[:2], ("PPO", "DQN")):
        for rule, (name, color, mk) in RULE.items():
            g = ne[(ne.algo == algo) & (ne.liability == rule)]
            M = np.array([[r[f"probe_e{e:g}"] for e in E] for _, r in g.iterrows()])
            mean = M.mean(axis=0)
            lo, hi = np.array([boot(M[:, j]) for j in range(len(E))]).T
            ax.fill_between(E, lo, hi, color=color, alpha=0.15, lw=0)
            ax.plot(E, mean, color=color, lw=1.4, marker=mk, ms=4, mec="white", mew=0.5, label=name)
        ax.set_title(f"({'a' if algo == 'PPO' else 'b'}) {algo}, learned", fontsize=8, color=INK)
        ax.set_ylim(0, 16.5)
    ax = axes[2]
    for rule, (name, color, mk) in RULE.items():
        v = [D[(D.rule == rule) & (np.isclose(D.equity, e))].gain_over_flat_eur.mean() for e in E]
        ax.plot(E, v, color=color, lw=1.4, marker=mk, ms=4, mec="white", mew=0.5, label=name)
    ax.set_title("(c) Theory: value over staying flat", fontsize=8, color=INK)
    ax.set_ylabel("EUR")
    axes[0].set_ylabel("Mean |leverage| chosen")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(E, ["0.02", "0.05", "0.1", "0.25", "0.5", "1", "2"], fontsize=6.5)
        ax.minorticks_off()
        ax.set_xlabel("Equity / allocation")
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(1.75, -0.28), ncol=3)
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(FIG / "fig_leverage_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_false_profit() -> None:
    ne = R[R.market == "NE"]
    rows = [(rule, algo) for rule in ("FL", "LL", "ZF") for algo in ("PPO", "DQN")]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for i, (rule, algo) in enumerate(rows):
        name, color, mk = RULE[rule]
        g = ne[(ne.liability == rule) & (ne.algo == algo)]
        y = len(rows) - 1 - i
        for k, (col, face, dy) in enumerate((("rep", color, 0.14), ("led", "white", -0.14))):
            x = g[col].to_numpy()
            lo, hi = boot2(draw_matrices(rule, algo, "reported_return_pct" if col == "rep" else "ledger_return_pct"))
            ax.plot([lo, hi], [y + dy, y + dy], color=color, lw=1.0)
            ax.plot(x.mean(), y + dy, marker=mk, ms=5, mfc=face, mec=color, mew=1.0, ls="")
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(rows)), [f"{RULE[r][0].split(',')[0]} / {a}" for r, a in rows][::-1], fontsize=6.5)
    ax.set_xlabel("Return over the test period, mean of 20 market draws (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", mfc=MUTED, mec=MUTED, ms=4, label="Reported"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, ms=4, label="Ledger (booked)")]
    ax.legend(handles=h, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_false_profit.pdf", bbox_inches="tight")
    plt.close(fig)


def tab_dp() -> str:
    cols = [0.02, 0.1, 0.25, 0.5, 1.0, 2.0]
    lines = [r"\begin{table}[!htbp]", r"\centering",
             r"\caption{Risk-neutral optimal policy of the no-edge sleeve by value iteration (discount 0.99, empirical DK1 and DK2 spread and liquidity-cap distributions, mean of the two zones). Top: optimal $|$leverage$|$; bottom: value of the optimal leverage over staying flat (EUR, allocation 20,000~EUR). The two zones give the same optimal leverage at every equity level.}",
             r"\label{tab:dp}", r"\footnotesize", r"\begin{tabular}{@{}l" + "r" * len(cols) + r"@{}}", r"\toprule",
             r"Equity / allocation & " + " & ".join(f"{c:g}" for c in cols) + r" \\", r"\midrule",
             r"\multicolumn{" + str(len(cols) + 1) + r"}{@{}l}{\emph{Optimal $|$leverage$|$}} \\"]
    for rule in ("FL", "LL", "ZF"):
        v = [D[(D.rule == rule) & np.isclose(D.equity, c)].opt_abs_lev.mean() for c in cols]
        lines.append(RULE[rule][0] + " & " + " & ".join(f"{x:g}" for x in v) + r" \\")
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{" + str(len(cols) + 1) + r"}{@{}l}{\emph{Value over staying flat (EUR)}} \\")
    for rule in ("FL", "LL", "ZF"):
        v = [D[(D.rule == rule) & np.isclose(D.equity, c)].gain_over_flat_eur.mean() for c in cols]
        lines.append(RULE[rule][0] + " & " + " & ".join(f"{x:.1f}" for x in v) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_false_profit() -> str:
    ne = C[C.market == "NE"]
    f = lambda x: f"${x:+.0f}$"  # noqa: E731
    lines = [r"\begin{table}[!htbp]", r"\centering",
             r"\caption{No-edge market: seed-mean return over the test period (mean of 20 independent sign draws; 5 seeds per cell), reported and booked (ledger), with two-way bootstrap 95\% intervals for the reported return (seeds and the 20 shared market draws resampled jointly). Every policy has an expected return of at most zero in this market.}",
             r"\label{tab:falseprofit}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{1}{c}{Full liability} & \multicolumn{2}{c}{Capped loss} & \multicolumn{2}{c}{Zero floor} \\",
             r"\cmidrule(lr){3-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Algorithm & Zone & Reported $=$ ledger & Reported & Ledger & Reported & Ledger \\", r"\midrule"]
    for algo in ("PPO", "DQN"):
        for region in ("DK1", "DK2"):
            c = {l: ne[(ne.algo == algo) & (ne.region == region) & (ne.liability == l)].iloc[0] for l in ("FL", "LL", "ZF")}
            zi = 0 if region == "DK1" else 1

            def ci(r, l):
                lo, hi = boot2([draw_matrices(l, algo, "reported_return_pct")[zi]])
                return f"{f(r.rep)} $[{lo:+.0f}, {hi:+.0f}]$"
            lines.append(f"{algo} & {region} & {ci(c['FL'], 'FL')} & {f(c['LL'].rep)} & {f(c['LL'].led)} & {ci(c['ZF'], 'ZF')} & {f(c['ZF'].led)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def stats() -> None:
    ne = R[R.market == "NE"]
    print("share of NE runs with RS > 0:", {l: f"{int((g.RS > 0).sum())}/{len(g)}" for l, g in ne.groupby("liability")})
    for algo in ("PPO", "DQN"):
        for rule in ("ZF", "LL"):
            for region in ("DK1", "DK2"):
                c = C[(C.market == "NE") & (C.algo == algo) & (C.region == region) & (C.liability == rule)].iloc[0]
                learned = [c[f"probe_e{e:g}"] for e in E]
                inc = [D[(D.region == region) & (D.rule == rule) & np.isclose(D.equity, e)].gain_over_flat_eur.iloc[0] for e in E]
                print(f"spearman learned vs incentive {algo} {rule} {region}: {spearmanr(learned, inc).correlation:.2f}")
    for algo in ("PPO", "DQN"):
        g = ne[ne.algo == algo]
        print(algo, "probe at 0.1 / 1 / 2 by rule:", {l: tuple(round(float(x[f'probe_e{e:g}'].mean()), 2) for e in (0.1, 1.0, 2.0)) for l, x in g.groupby("liability")})
    rm = R[R.market == "RM"]
    print("RM mean ML / reported / ledger / floor share by rule:", rm.groupby("liability")[["ML", "rep", "led", "floor_bound"]].mean().round(2).to_dict("index"))
    print("NE ZF floor-bound share over draws:", ne[ne.liability == "ZF"].drawshare_floor_bound.mean().round(2) if "drawshare_floor_bound" in ne else "n/a")


def main() -> int:
    fig_leverage_curves()
    fig_false_profit()
    (SEC / "tab_dp.tex").write_text(tab_dp(), encoding="utf-8", newline="\n")
    (SEC / "tab_false_profit.tex").write_text(tab_false_profit(), encoding="utf-8", newline="\n")
    stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

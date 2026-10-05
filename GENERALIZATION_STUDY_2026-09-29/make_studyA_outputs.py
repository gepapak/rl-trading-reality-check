"""Manuscript outputs for Study A: the verdict table (LaTeX), a margin figure and seed-bootstrap false-positive shares.

    python make_studyA_outputs.py                 # final: needs all 280 RL jobs; writes into the Overleaf project and results/
    python make_studyA_outputs.py --preview DIR   # any time: writes everything into DIR (partial results allowed)

Advantage of learning in a cell = (best RL seed-mean - best rule) / |best rule|, on the design's own metric
(E1, E3: profit in EUR; E2: sleeve return in %). "Reported" uses the cell's shortcut regime, "real" evaluates the same
policies under S0 (definition as in aggregate_study.verdict_table). A false positive is reported > 0 and real <= 0.

Palette: first three categorical slots of the dataviz reference palette (validated all-pairs, light mode), one per design
family; the two battery sizes share slot 1 and differ by marker fill, so identity never rests on colour alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aggregate_study as A

HERE = Path(__file__).resolve().parent
OVERLEAF = next((HERE.parent / "Overleaf Projects (1 items)").glob("*"), HERE / "outputs")
N_JOBS = 280
N_BOOT = 10_000
ENVS = ["E1-small", "E1-large", "E2", "E3"]
ENV_LABEL = {"E1-small": "Battery 1 MW", "E1-large": "Battery 50 MW", "E2": "Sleeve", "E3": "Wind 100 MW"}
REGIME_LABEL = {"S0": "Strict", "S1": "No liquidity cap", "S2": "Zero floor, no ruin", "S3": "No cap + zero floor",
                "S4": "Look-ahead", "S5": "Percent payoff"}
STYLE = {"E1-small": ("#2a78d6", "o", "white"), "E1-large": ("#2a78d6", "o", None), "E2": ("#eb6834", "s", None),
         "E3": ("#1baf7a", "^", None)}  # colour, marker, face override (white = hollow)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def adv(rl: float, rule: float) -> float:
    return (rl - rule) / abs(rule)


def cells(rl: pd.DataFrame, rules: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    V = A.verdict_table(rl, rules)
    rows = []
    for v in V.itertuples():
        g = rl[(rl.env == v.env) & (rl.region == v.region) & (rl.regime == v.regime)]
        # seed bootstrap: resample jobs within each algorithm (reported and real stay paired), take the best algorithm's mean
        fp = []
        by_algo = [d[["reported", "reality"]].to_numpy() for _, d in g.groupby("algo")]
        for _ in range(N_BOOT):
            means = np.array([x[rng.integers(0, len(x), len(x))].mean(axis=0) for x in by_algo])
            fp.append(means[:, 0].max() > v.best_rule_reported and means[:, 1].max() <= v.best_rule_reality)
        rows.append(dict(env=v.env, region=v.region, regime=v.regime, n_rl=v.n_rl,
                         adv_reported=adv(v.best_rl_reported, v.best_rule_reported),
                         adv_real=adv(v.best_rl_reality, v.best_rule_reality),
                         false_positive=bool(v.false_positive), p_fp_boot=float(np.mean(fp)),
                         rl_reported_median=v.rl_reported_median, rl_reality_median=v.rl_reality_median))
    C = pd.DataFrame(rows)
    C["env"] = pd.Categorical(C.env, ENVS, ordered=True)
    return C.sort_values(["env", "regime", "region"]).reset_index(drop=True)


def fmt_pct(x: float) -> str:
    p = x * 100
    if abs(p) < 10:
        return f"${p:+.1f}$"
    if abs(p) < 1e6:
        return f"${p:+,.0f}$".replace(",", "{,}")
    m, e = f"{p:+.1e}".split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


def latex_table(C: pd.DataFrame) -> str:
    lines = [r"\begin{table}[!htbp]", r"\centering",
             r"\caption{Generalization study: advantage of the best learned policy over the best rule (\%), reported under each shortcut and real (the same policies evaluated strictly). A false positive (FP) is a positive reported and a non-positive real advantage; $P_{\mathrm{FP}}$ is its share over 10,000 seed-bootstrap resamples.}",
             r"\label{tab:studyA}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}llrrcrrc@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{DK1} & \multicolumn{3}{c}{DK2} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Design & Shortcut & Reported & Real & FP ($P_{\mathrm{FP}}$) & Reported & Real & FP ($P_{\mathrm{FP}}$) \\", r"\midrule"]
    for env, ge in C.groupby("env", observed=True):
        first = True
        for regime, gr in ge.groupby("regime"):
            cellv = []
            for region in ("DK1", "DK2"):
                r = gr[gr.region == region]
                if r.empty:
                    cellv += ["--", "--", "--"]
                    continue
                r = r.iloc[0]
                fp = "--" if regime == "S0" else (("yes" if r.false_positive else "no") + f" ({r.p_fp_boot:.2f})")
                cellv += [fmt_pct(r.adv_reported), fmt_pct(r.adv_real), fp]
            lines.append((ENV_LABEL[env] if first else "") + " & " + REGIME_LABEL[regime] + " & " + " & ".join(cellv) + r" \\")
            first = False
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def figure(C: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    D = C[C.regime != "S0"]
    lo = min(-1.0, D.adv_real.min() * 1.3, D.adv_reported.min() * 1.3)
    hi = max(10.0, D.adv_reported.max() * 1.5, D.adv_real.max() * 1.5)
    ax.fill_between([lo, 0], 0, hi, color=GRID, lw=0, zorder=0)
    ax.text(lo * 0.9 if lo < -1 else -0.9, hi * 0.6, "false\npositive", fontsize=6.5, color=MUTED, va="top", ha="left")
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.6, ls="--", zorder=1)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    for env in ENVS:
        d = D[D.env == env]
        if d.empty:
            continue
        color, marker, face = STYLE[env]
        ax.scatter(d.adv_real, d.adv_reported, s=22, marker=marker, facecolor=face or color, edgecolor=color if face else "white",
                   linewidth=0.8 if face else 0.4, label=ENV_LABEL[env], zorder=3)
    for r in D[D.false_positive].itertuples():
        ax.annotate(f"{REGIME_LABEL[r.regime]}, {r.region}", (r.adv_real, r.adv_reported), xytext=(4, 3),
                    textcoords="offset points", fontsize=5.5, color=INK)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [t for t in (-100, -10, -1, -0.5, 0, 0.5, 1, 3, 10, 100, 1000, 10000) if lo <= t <= hi]
    labels = ["0" if t == 0 else f"{t * 100:+,.0f}%".replace("-", "−") for t in ticks]
    for axis, skip in ((ax.xaxis, {3}), (ax.yaxis, set())):  # +300% crowds +1,000% on the shorter x axis
        keep = [i for i, t in enumerate(ticks) if t not in skip]
        axis.set_ticks([ticks[i] for i in keep])
        axis.set_ticklabels([labels[i] for i in keep], fontsize=6.5)
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Real advantage of learning over the best rule")
    ax.set_ylabel("Reported advantage under the shortcut")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, columnspacing=1.0)
    ax.grid(True, color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=Path, help="write all outputs into this folder; partial results allowed")
    a = ap.parse_args()
    rl, rules = A.load(False)
    if a.preview is None and len(rl) < N_JOBS:
        raise SystemExit(f"only {len(rl)}/{N_JOBS} RL jobs finished; use --preview DIR for an interim view")
    C = cells(rl, rules, np.random.default_rng(20260929))
    fig_path = (a.preview or OVERLEAF / "figures") / "fig_studyA_margins.pdf"
    tab_path = (a.preview or OVERLEAF / "sections") / "tab_studyA.tex"
    csv_path = (a.preview or A.RES) / "studyA_cells.csv"
    for p in (fig_path, tab_path, csv_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    C.to_csv(csv_path, index=False)
    tab_path.write_text(latex_table(C), encoding="utf-8", newline="\n")
    figure(C, fig_path)
    pd.set_option("display.width", 200)
    print(f"{len(rl)}/{N_JOBS} RL jobs")
    print(C[["env", "region", "regime", "n_rl", "adv_reported", "adv_real", "false_positive", "p_fp_boot"]].round(3).to_string(index=False))
    print("written:", fig_path, tab_path, csv_path, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

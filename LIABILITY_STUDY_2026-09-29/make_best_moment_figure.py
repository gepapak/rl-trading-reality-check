"""EXPLORATORY (not pre-registered): data and figure for the best-moment theorem.

For each zone, a fixed position (leverage 16, liquidity cap applied) is held from zero equity under the zero floor in
the placebo market over the test period, for N independent sign draws. Writes
    results_k/best_moment.csv     per zone: KS distance and p-value (reported vs running maximum), CV, mean reported
                                  equity, prediction with fee drift, power of n-draw placebo tests
    outputs/figures/fig_best_moment.pdf   (a) DK1: reported equity vs hindsight-best exit vs the |B_T| law;
                                          (b) power of the placebo test against the number of draws
Palette as fig_wellposed.pdf.

    python make_best_moment_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, stats

from envs_b import FEE, K0, Q1, load_market
from envs_j import load_market_j

HERE = Path(__file__).resolve().parent
OUT = next((HERE.parent / "Overleaf Projects (1 items)").glob("*"), HERE / "outputs") / "figures" / "fig_best_moment.pdf"
N = 2000
NS = (2, 3, 4, 5, 6, 8, 10)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ZCOL = {"DK1": "#2a78d6", "DK2": "#1baf7a", "FI": "#eb6834", "NO2": "#8a5cd0"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})
rng = np.random.default_rng(20261005)


def power(R: np.ndarray, n: int, reps: int = 20000) -> float:
    S = R[rng.integers(0, len(R), (reps, n))]
    sd = S.std(1, ddof=1)
    t = np.where(sd > 0, S.mean(1) / np.where(sd > 0, sd, 1) * np.sqrt(n), np.inf)
    return float(np.mean(t > stats.t.ppf(0.95, n - 1)))


def main() -> int:
    rows, keep = [], {}
    for region in ("DK1", "DK2", "FI", "NO2"):
        lm = load_market if region in ("DK1", "DK2") else load_market_j
        m = lm(region, "test", "RM")
        q = np.minimum(16 * Q1, m.cap_mwh[200:])
        mag, fee = q * np.abs(m.spread[200:]), FEE * q
        T = len(mag)
        W, L, Lmax = np.zeros(N), np.zeros(N), np.zeros(N)
        for t in range(T):
            x = rng.choice([-1.0, 1.0], N) * mag[t] - fee[t]
            W = np.maximum(W + x, 0.0)
            L = L + x
            Lmax = np.maximum(Lmax, L)
        sigma, mu = float(np.sqrt(np.mean(mag ** 2))), -float(np.mean(fee))
        s, mT = sigma * np.sqrt(T), mu * T
        P = lambda a: 1 - stats.norm.cdf((a - mT) / s) + np.exp(2 * mu * a / sigma ** 2) * stats.norm.cdf((-a - mT) / s)  # noqa: E731
        EM, _ = integrate.quad(P, 0, 20 * s, limit=200)
        ks = stats.ks_2samp(W, Lmax)
        rows.append(dict(zone=region, T=T, sigma=sigma, ks_D=ks.statistic, ks_p=ks.pvalue, cv=W.std() / W.mean(),
                         mean_reported_K=W.mean() / K0, predicted_K=EM / K0, driftless_K=sigma * np.sqrt(2 * T / np.pi) / K0,
                         mean_booked_K=L.mean() / K0, **{f"power_n{n}": power(W / K0, n) for n in NS}))
        keep[region] = (W, Lmax, s)
        print(rows[-1], flush=True)
    D = pd.DataFrame(rows)
    (HERE / "results_k").mkdir(exist_ok=True)
    D.to_csv(HERE / "results_k" / "best_moment.csv", index=False)
    folded = np.abs(np.random.default_rng(1).standard_normal(200000))
    theory_power = {n: power(folded, n) for n in NS}

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    W, Lmax, s = keep["DK1"]
    bins = np.linspace(0, np.quantile(np.r_[W, Lmax], 0.995) / K0, 40)
    a.hist(W / K0, bins=bins, density=True, color="#2a78d6", alpha=0.35, label="Reported equity under the floor")
    a.hist(Lmax / K0, bins=bins, density=True, histtype="step", color=INK, lw=1.0, label="Best exit in hindsight (running maximum)")
    xx = np.linspace(0, bins[-1], 300)
    a.plot(xx, 2 * stats.norm.pdf(xx, scale=s / K0), color="#eb6834", lw=1.2, ls="--", label=r"$|B_T|$ law (no fees)")
    a.set_xlabel("Final equity / allocation (DK1, held from zero equity)")
    a.set_ylabel("Density")
    a.set_title("(a) The floor reports the best moment", fontsize=8, color=INK)
    a.legend(fontsize=6, loc="upper right")
    b.plot(NS, [theory_power[n] for n in NS], color=INK, lw=1.4, label=r"Theory ($|B_T|$, any market)")
    for region in ("DK1", "DK2", "FI", "NO2"):
        b.plot(NS, [D[D.zone == region][f"power_n{n}"].item() for n in NS], color=ZCOL[region], lw=1.0, marker="o", ms=3,
               mec="white", mew=0.4, label=region)
    b.axhline(0.8, color=MUTED, lw=0.6, ls=":")
    b.set_xlabel("Placebo draws")
    b.set_ylabel("Probability the test flags the floor")
    b.set_title("(b) A few placebo draws suffice", fontsize=8, color=INK)
    b.set_ylim(0, 1.02)
    b.yaxis.grid(True, color=GRID, lw=0.5)
    b.set_axisbelow(True)
    b.legend(fontsize=6, loc="lower right")
    fig.subplots_adjust(wspace=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT)
    print("theoretical power:", {n: round(v, 3) for n, v in theory_power.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

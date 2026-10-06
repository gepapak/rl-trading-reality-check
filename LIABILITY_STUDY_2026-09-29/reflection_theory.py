"""EXPLORATORY (not pre-registered): closed-form theory of the zero floor as a Skorokhod reflection.

Under the zero floor the reported equity follows Lindley's recursion E_{t+1} = max(E_t + x_t, 0), i.e. the ledger
reflected at zero; the cumulative forgiven loss is the reflection's regulator L_t. For a diffusion approximation of a
position held at leverage l (step standard deviation sigma = q_l * sd(spread), drift mu = -fee * q_l per step) with
discount rate lam = 1 - gamma per step, the expected discounted forgiven loss from equity e is

    u(e) = exp(-theta * e) / theta,   theta = (mu + sqrt(mu^2 + 2 lam sigma^2)) / sigma^2      (mu < 0 here)

so the value of holding leverage l forever over staying flat is G(e) = mu / lam + u(e), and the gambling threshold is
e* = ln(lam / (theta * fee * q_l)) / theta.  This script compares e* and G with the dynamic-programming optimum
(dp_theory.py, dp_j.py) in DK1, DK2, FI and NO2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from envs_b import FEE, K0, LEV, Q1, load_market
from envs_j import load_market_j

LAM = 1 - 0.99
QMAX = float(np.abs(LEV).max() * Q1)          # 4 MWh at leverage 16


def spread_stats(region: str) -> dict:
    m = load_market(region, "train", "RM") if region in ("DK1", "DK2") else load_market_j(region, "train", "RM")
    s = m.spread[200:]
    q = np.minimum(QMAX, m.cap_mwh[200:])          # the liquidity cap may bind
    pnl_sd = float(np.std(q * s))
    mad_sd = float(1.4826 * np.median(np.abs(s - np.median(s))) * np.mean(q))
    return dict(region=region, sd_spread=float(np.std(s)), mean_abs=float(np.mean(np.abs(s))),
                sigma=pnl_sd, sigma_robust=mad_sd, q_mean=float(np.mean(q)))


def closed_form(sigma: float, q: float) -> tuple[float, callable]:
    mu = -FEE * q
    theta = (mu + np.sqrt(mu ** 2 + 2 * LAM * sigma ** 2)) / sigma ** 2
    e_star = np.log(LAM / (theta * FEE * q)) / theta
    return e_star, (lambda e: mu / LAM + np.exp(-theta * e) / theta)


def main() -> int:
    rows = []
    for r in ("DK1", "DK2", "FI", "NO2"):
        st = spread_stats(r)
        for kind in ("sigma", "sigma_robust"):
            e_star, G = closed_form(st[kind], st["q_mean"])
            rows.append(dict(**{k: round(v, 2) if isinstance(v, float) else v for k, v in st.items()}, kind=kind,
                             e_star_over_K=round(e_star / K0, 3),
                             G_002=round(G(0.02 * K0), 1), G_01=round(G(0.1 * K0), 1), G_025=round(G(0.25 * K0), 1),
                             G_05=round(G(0.5 * K0), 1), G_1=round(G(1.0 * K0), 1)))
    D = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(D.to_string(index=False))
    dp = pd.concat([pd.read_csv("results/dp_theory.csv"), pd.read_csv("results_j/dp_j.csv")])
    z = dp[dp.rule == "ZF"].pivot_table(index="region", columns="equity", values="opt_abs_lev")
    g = dp[dp.rule == "ZF"].pivot_table(index="region", columns="equity", values="gain_over_flat_eur").round(1)
    print("\nDP optimal |leverage| under ZF:\n", z.to_string())
    print("\nDP value of the best action over flat (EUR):\n", g.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

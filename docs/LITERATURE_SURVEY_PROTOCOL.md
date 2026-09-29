# Literature prevalence survey: evaluation shortcuts in AI trading for electricity markets

**Purpose:** show whether the shortcuts audited in this folder are common in published work. This answers the reviewer objection "you audited only your own simulator."

**Inclusion:** peer-reviewed or arXiv papers (2019–2026) in which an RL, deep-learning or ML agent/policy trades or bids in electricity markets (day-ahead, intraday, balancing/imbalance, storage arbitrage, renewable bidding) and reports monetary performance from a backtest or simulation. Full text must be accessible; coding uses the full text, never the abstract alone.

**Coding sheet** (`literature_survey.csv`), one row per paper:

| Code | Question | Values |
|---|---|---|
| C1 liquidity | Are traded volumes limited by observed market volume/liquidity (or price impact modelled)? | yes / no (price-taker, unlimited) / partial (fixed MW limit only) / n.r. |
| C2 costs | Transaction cost model | none / fixed / volume-based / order-book / n.r. |
| C3 solvency | Is path-dependent solvency (margin, ruin, capital constraint) enforced, or is P&L summed? | enforced / summed / n.a. (physical asset only) / n.r. |
| C4 look-ahead | Any look-ahead risk (same-period realised price in features, interpolated sub-hourly prices, perfect-foresight features)? | none stated / possible / yes / n.r. |
| C5 spikes | Treatment of price spikes | kept / clipped or removed / n.r. |
| C6 metrics | Main metric; Sharpe annualisation frequency if Sharpe used | text |
| C7 seeds | Number of training seeds reported | integer / n.r. |
| C8 baselines | Strongest baseline | none / naive / tuned rule / optimisation / n.r. |
| C9 claim | Headline performance claim | text |

n.r. = not reported. Coding is conservative: a shortcut is recorded only when the paper states it or it follows directly from the stated setup.

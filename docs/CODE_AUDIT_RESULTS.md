# Code-level prevalence audit: results (28 Sep 2026, ~23:15)

**Protocol:** `CODE_AUDIT_PROTOCOL.md` (search fixed, then Deviation 1 and Clarification 1, both recorded before screening).

**Pipeline:**
- **Search:** 30 GitHub queries → 337 unique repositories (`candidates.csv`).
- **Stage 1** (name and description, Python / Jupyter, not a fork) → 42 (`stage1_keep.txt`).
- **Stage 2** (README plus file tree; criteria b and c) → **21 included** (`coding.csv`).
- **Coding:** at a pinned commit SHA (`stage2_repo_summary.csv`), from the automatically flagged lines (`evidence_flags.csv`) and a manual reading of each environment's `step` / reward code (`evidence/`). No third-party code was executed.

**Stage-2 exclusions (21):**
- no learning agent or no monetary price outcome: vessim, battery_model, SafeOR-Gym, psst-archive, Energy_Broker_GA (genetic algorithm only), e2e-model-learning, XGBoost+Pyomo, Laveet, powersync;
- < 5 stars and no publication link: Q-learning-electricity-market, MultiAgent-Evolution-DQN-EV, ABM-and-RL-LEM, Battery-storage-optimization, pv_trading (only a planned submission), battery-arbitrage-RL (ekahorsu), Grid-Scale-Battery-Dispatch-DRL, Solar_Arbitrage_Agent, Contextual-Optimization (thesis without a link), mibel-trading, Zzyang666, hybrid-vpp-rl.

## Prevalence (n = 21; Wilson 95% intervals; `prevalence_results.csv`)

| Item | k / n | Share (95% CI) | Recorded expectation | Met? |
|---|---|---|---|---|
| C0 capital / cash account present | 1 / 21 | 5% (1–23%) | – | – |
| C1 price-taker, all repositories | 11 / 21 | 52% (32–72%) | ≥ 70% | **No** |
| — single-agent storage / PV / bidding environments on historical prices *(post-hoc split)* | 11 / 12 | 92% (65–99%) | – | – |
| — multi-agent market-clearing simulators *(post-hoc split)* | 0 / 9 | 0% (0–30%) | – | – |
| C2 solvency modelled | 0 / 21 | 0% (0–15%) | ≤ 20% | Yes |
| C3 zero floor on money (repositories with a money state) | 0 / 1 | – | ≥ 1 repository | **No** |
| C4(b) percent-of-price payoff | 0 / 21 | 0% (0–15%) | ≥ 1 repository | **No** |
| C4(a) cash settlement (volume × price) | 20 / 21 | 95% | – | – |
| C5 look-ahead visible in static code | 3 / 21 | 14% (5–35%); 15 / 21 unclear | – | – |
| C6 Sharpe reported | 1 / 21 | 5% | – | – |

The three look-ahead cases (C5):
- future prices used as features via `shift(-n)` (energy-py example);
- a rolling window of true future prices in the observation (battery-optimisation-with-drl, `price_track='true'`);
- the episode's future price minimum and maximum used for thresholds and normalisation (VoltXChange).

## What this means for the paper (honest reading)

**1. The two mechanisms behind the reproduced false positive were not found in public electricity-RL code.**
- The mechanisms are zero-floor loss forgiveness (Finding A) and a percent-of-price payoff (Finding B).
- 20 of 21 environments have no capital account at all. The reward is per-step cash, volume × price, and results are reported as absolute profit or cumulative reward.
- These artifacts belong to the finance-style "capital account + percent return" simulator design that the thesis engine followed, not to the typical public electricity-RL environment.
- The generality claim for Findings A and B must therefore be limited to finance-style trading simulators; for electricity environments it is **not supported** by this audit.

**2. What is general.**
- No environment models solvency or capital (0 / 21), so ruin, capital-relative returns and leverage are undefined in public electricity-RL evaluation.
- Single-agent environments on historical prices are almost all price-takers (11 / 12). In those environments, however, positions are bounded by physical asset ratings (battery MW / MWh), so the ×300–1,000 inflation measured in the controlled audit, which came from *capital-sized* positions, does not transfer directly.

**3. Look-ahead is at least occasional (3 / 21) and mostly not checkable statically (15 / 21 unclear).**

**Limitations.**
- GitHub repository search only, 21 included repositories, and a single coder with no second rater.
- Static reading only.
- The storage vs. market-clearing split was defined after coding and is descriptive.

# Code-level prevalence audit of public electricity-trading RL environments: protocol

Written 28 Sep 2026, ~22:39 (file times: search script 22:40:09), before any repository search or code reading.

## Purpose
Estimate how common the evaluation artifacts identified in the controlled audit (CONTRIBUTION.md §3c-quater) are in public code.
The artifacts are:
- the price-taker assumption;
- missing solvency;
- zero-floor loss forgiveness;
- percent-of-price payoffs on a non-storable commodity;
- look-ahead.

## Search (fixed before running; `search_repos.py`)
- Source: the GitHub repository search API, unauthenticated.
- Ranking: sorted by stars, top 30 results per query.
- Snapshot date: 28 Sep 2026.
- Queries:
  1. reinforcement learning electricity market trading
  2. reinforcement learning energy trading environment
  3. battery arbitrage reinforcement learning
  4. electricity price arbitrage reinforcement learning
  5. intraday electricity market reinforcement learning
  6. imbalance market reinforcement learning
  7. energy trading gym
  8. power trading deep reinforcement learning
  9. bidding strategy reinforcement learning electricity
  10. energy storage arbitrage deep reinforcement learning

## Screening (metadata, README, file tree)
**Include** a repository if all of the following hold:
- (a) it contains Python code;
- (b) it implements a simulation environment in which an agent's actions determine a monetary outcome from electricity prices (trading, bidding, storage arbitrage, P2P trading at market prices);
- (c) it has ≥ 5 stars, or its README links a publication (arXiv, DOI, or a journal or conference name).

**Exclude:**
- forecasting-only repositories;
- control without monetary settlement;
- awesome-lists, course material and tutorials without their own environment;
- forks and exact duplicates;
- repositories without environment code.

**Sample:** all included repositories, up to 40 ordered by stars.

## Coding (per repository, at the default-branch HEAD commit, SHA recorded)
- **C0 money state:** the environment keeps a cash, balance or equity account (Y/N). If N, C2 and C3 are NA.
- **C1 price-taker:** trades clear at exogenous historical prices with no volume limit relative to market volume and no price impact (Y/N).
- **C2 solvency modelled:** any constraint that stops trading or ends the episode on insufficient or non-positive equity, or any margin requirement (Y/N/NA).
- **C3 loss floor:** cash, balance or equity is clipped at zero, or losses are not booked when equity is insufficient (Y/N/NA).
- **C4 payoff type:**
  - (a) cash settlement, volume × price;
  - (b) percent return on the price level (position × Δp/p, or equity × price return);
  - (c) other.
- **C5 look-ahead risk (static check only):** observations or rewards use future prices, full-series normalisation statistics, or interpolation (Y/N/unclear).
- **C6 reported metric:** cumulative reward, profit, percent return, Sharpe, or other.

**Process.**
- An automated pattern scan (`scan_repos.py`) flags candidate lines.
- The final codes are assigned by reading the flagged lines and the environment `step`/`reset` functions.
- Every code carries evidence (file:line @ SHA). Uncertain cases are coded "unclear" and reported separately.

**Safety and scope.**
- Third-party code is only read, never executed.
- Source files are fetched into memory; only short evidence excerpts and SHAs are stored.
- A dynamic ledger reconciliation of third-party environments is out of scope.

## Claims rules (fixed now)
- Proportions are reported over the applicable included repositories, with Wilson 95% intervals.
- A pattern is called "common" only if it is present in ≥ 30% of the applicable repositories.
- Recorded expectations, which are not hypotheses and are reported whatever the outcome:
  - C1 price-taker ≥ 70%;
  - C2 solvency ≤ 20%;
  - C3 loss floor in ≥ 1 repository;
  - C4(b) percent payoff in ≥ 1 repository.

## Deviation 1 (28 Sep 2026, ~22:42), recorded before screening any candidate
- **Problem.** The 10 fixed queries returned only 40 unique repositories (0–22 per query). GitHub repository search requires every term to match in the name, description or topics, which is too strict for 5–6-word queries.
- **Fix.** Before any screening:
  - the same 10 queries are re-run with the `in:name,description,topics,readme` qualifier;
  - 10 shorter supplementary queries are added, with the same qualifier, stars sort and top 30.
- **Supplementary queries:**
  - electricity trading reinforcement
  - energy trading reinforcement
  - battery arbitrage
  - electricity market reinforcement
  - energy arbitrage reinforcement
  - power market reinforcement learning
  - day-ahead market reinforcement
  - virtual power plant reinforcement
  - energy storage reinforcement learning price
  - bidding reinforcement electricity market
- The union of all 30 result sets is screened. The screening and coding rules are unchanged.

## Clarification 1 (28 Sep 2026, 22:46), recorded before any README or code was read
- **(b) means a learning agent.** Criterion (b) requires an environment in which a *learning* agent acts: RL, or a learned (DL) policy. Pure mathematical-programming dispatch models and forecasting pipelines without an acting agent are excluded.
- **Stage-1 screening on name and description.** The screener keeps every repository that plausibly contains an electricity trading, bidding, storage-arbitrage or local-market agent. It excludes lists, digests, stock/crypto/forex bots, non-energy projects, and non-Python code (language field empty or not Python / Jupyter). Stage-2 screening (README plus file tree) then applies (b) and (c).
- **Version identifier.** The commit SHA of the default branch at audit time, obtained with `git ls-remote`. Files are fetched from raw.githubusercontent.com at that SHA.

_Timestamp note (28 Sep 2026, 22:48): the header times above were corrected to match file modification times. No rule text was changed._

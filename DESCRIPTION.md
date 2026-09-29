# Repository description

## Suggested repository name
`simulator-decides-the-verdict`

## GitHub "About" text (≤ 350 characters)
Pre-registered audit of how simulator shortcuts (no liquidity cap, no solvency, zero-floor accounting) turn losing RL trading agents into apparent winners in Nordic balancing markets. Includes a ledger-reconciliation check, an evaluation checklist and a code audit of 21 public energy-RL environments.

## Suggested topics
`reinforcement-learning` `multi-agent-reinforcement-learning` `electricity-markets` `energy-trading` `balancing-market` `imbalance-settlement` `backtesting` `evaluation-methodology` `reproducibility` `pre-registration`

## Abstract
AI trading agents for electricity markets are judged in simulators, and the simulator's accounting can decide the verdict.

**The case.** A multi-agent reinforcement-learning fund trades Danish imbalance-settlement exposure on held-out 2025 data (DK1, DK2). We re-evaluate its frozen checkpoints in a validated engine under one evaluation shortcut at a time.

**What the shortcuts do.**
- Removing the liquidity cap inflates a deterministic forecast rule's return ×300–1,000, but ruins the learned policy.
- Removing solvency lets the engine's zero cash floor silently forgive losses. The learned policy's booked P&L loses 7–33× its capital, while the reported return is positive in 13 of 20 runs.
- A protocol resembling the original thesis reproduces its "MARL wins" claim from checkpoints that come last under strict evaluation: a +270% reported mean vs −535% booked, over 10 seeds.
- A 56-run leave-one-out decomposition shows the two necessary components are the missing liquidity cap and the missing solvency. The percent-of-price payoff is not necessary.

**Detection.** Reconciling reported returns with booked P&L separates sound from unsound accounting in every run.

**Generality.** A pre-registered audit of 21 public electricity-RL environments finds that none model capital or solvency and none use zero-floor accounting. The artifact therefore threatens finance-style simulators applied to energy rather than typical energy-RL code.

**Reporting.** Every prediction was written into timestamped files before the runs (internal pre-registration), and failed predictions are reported.

## Key facts
| | |
|---|---|
| Market | Danish imbalance settlement (DK1, DK2); cross-market checks on FI and NO1–NO5 |
| Evaluation window | Held-out 2025 data; market data 2025-03-04 → 2026-09-28 |
| Agents | Plain MAPPO (10 seeds), feasible-action MAPPO (10 seeds), deterministic forecast anchor |
| Runs | 264 completed engine evaluations (6 further combinations refused by the engine by design), including 3 MARL seeds retrained under the shortcut; engine validated to 1e-8 pp against the frozen campaign |
| Pre-registered predictions | 18 in the experiments (10 supported, one of them only trivially; 8 not supported) + 4 recorded code-audit expectations (1 met) |
| Code audit | 337 repositories searched → 21 included and coded at pinned commits |
| License | Code: MIT. Docs and results: CC BY 4.0. Third-party data: see `THIRD_PARTY_LICENSES.md` |

## Status
Accompanies a manuscript in preparation. Model checkpoints and raw per-run logs (~105 GB) are not included in this repository.

# Study B pre-registration: does simulated limited liability teach learning agents to gamble?

Internal pre-registration, written and hashed (SHA-256, PREREGISTRATION_HASHES.txt) before any evaluation on the test
period. Code: `envs_b.py`, `run_b.py`; aggregation: `aggregate_b.py` (written before the full run finishes, applying the
rules below verbatim).

## 1. Motivation and theory

The engine audit (SIMULATOR_AUDIT_2026-09-28) found a zero cash floor that forgives losses beyond the sleeve's cash.
Study A (GENERALIZATION_STUDY_2026-09-29) could not test this mechanism: its sleeve sized positions from current equity,
so equity never reached the floor (A2 not testable).

Corporate-finance theory predicts what a floor does to a risk-neutral decision maker. Limited liability turns the
payoff into max(W, 0), which is convex in terminal wealth W (Jensen and Meckling 1976, asset substitution). The value of
added risk is largest when equity is small relative to the position ("gambling for resurrection").

Two conditions are needed:
- the position is not scaled down with equity (a fixed trading allocation, as in the audited engine or a bank with fixed
  assets);
- some losses beyond equity are forgiven.

A reinforcement-learning agent maximizing the simulator's reported reward is such a decision maker. We therefore
predict that simulated limited liability teaches it:
1. to take more risk, and
2. to take more risk the closer it is to zero equity.

Standard evaluation of the reported return then rewards this behavior.

## 2. Design

### Environment
A sleeve trading the Danish imbalance spread (imbalance price minus day-ahead price) each quarter-hour, on the same data
and causal features as Study A (DK1, DK2; train March–December 2025, test January–August 2026).

- **Allocation:** K0 = 20,000 EUR. The position is sized from this fixed allocation, not from current equity.
- **Action:** a signed leverage level in {−16, −4, −1, −0.25, 0, 0.25, 1, 4, 16}. Leverage 1 is 0.25 MWh per
  quarter-hour, so the maximum position is 4 MWh.
- **Liquidity cap:** the executed volume is capped at 25% of the balancing volume (as in Study A S0), so price-taking is
  NOT a factor in this study.
- **Fee:** 0.10 EUR/MWh.
- **Observation (11 dimensions):**
  - day-ahead price;
  - five causal lagged spread features;
  - reported equity / K0, clipped to [0, 5];
  - four calendar features.
- **Reward:** the change in reported equity, scaled by 100 / K0.
- **Training episodes:** one week (672 quarter-hours) at a random start. Initial equity is drawn log-uniformly in
  [0.05, 2] × K0, identically in all conditions, so that every liability rule is trained on the same range of states.
- **Evaluation:** the whole test period from equity = K0.

### Liability rules
The market, sizing, cap, fee and observation are identical across rules; only the accounting of losses beyond equity
differs. The ledger always books every loss.

- **FL (full liability):** every loss is booked in reported equity. The sleeve closes when equity reaches zero, and
  the overshoot is booked.
- **LL (limited liability):** the loss that would take equity below zero is capped at the remaining equity, and the
  sleeve closes. This is the common "episode ends at bankruptcy" rule, without negative equity.
- **ZF (zero floor):** reported equity is floored at zero every quarter-hour and trading continues. This is the audited
  engine's rule.

### Markets
- **RM (real market):** the real spreads. These contain a real edge (lag-2 sign persistence).
- **NE (no edge):** the same spread magnitudes with an independent random sign each quarter-hour. The lagged features
  are rebuilt from the sign-flipped series. By construction E[payoff | past] = 0 for every policy, so no policy can beat
  staying flat in expectation.
  - The sign seeds are fixed per region and split (envs_b.NE_SEED).
  - Every NE policy is evaluated on 20 independent sign draws of the test period (draw k uses seed NE_SEED + 1000k).
  - NE outcomes are means over the 20 draws.

### Learners and baselines
- **Learners:** PPO and DQN (Stable-Baselines3 defaults, network 64-64, γ = 0.99). 400,000 steps each; seeds 0–4.
- **Runs:** 3 liability rules × 2 markets × 2 algorithms × 5 seeds × 2 regions = 120 runs.
- **Rule baselines (descriptive only, no hypothesis):**
  - flat;
  - lag-2 persistence at leverage 0.25 / 1 / 4 / 16;
  - constant long or short at the same levels.

  The latter two families are tuned on the training period within each rule and market.

### Measurements
- **ML:** mean |leverage| over the quarter-hours in which the sleeve is open during evaluation (NE: mean over draws).
- **Probe P(e):** mean |leverage| of the trained policy's greedy action over every 8th test quarter-hour, with the
  equity input set to e × K0, for e in {0, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2}. It uses the test market's features
  (NE: draw 0).
- **Resurrection slope:** RS = P(0.1) − P(1).
- **Returns:** reported and ledger returns (% of K0), floor-bound flag, closed flag and maximum reported–ledger gap.
- **Cells and units:**
  - a cell is (market, algorithm, region): 8 cells, of which 4 are NE;
  - seed means are over the 5 seeds;
  - bootstraps resample seeds (10,000 resamples, 95% percentile intervals).

## 3. Hypotheses and decision rules

- **B1 (limited liability raises risk-taking).** In the NE market:
  - ML(ZF) > ML(FL) in all 4 NE cells; and
  - ML(LL) > ML(FL) in at least 3 of 4.
- **B2 (gambling for resurrection).** RS(ZF) > RS(FL) in at least 6 of 8 cells, and RS(ZF) > 0 in at least 6 of 8 cells.
  - Secondary: RS(LL) > RS(FL) in at least 6 of 8 cells.
- **B3 (a false positive in expectation).** In the NE market:
  - under ZF, the seed-mean reported return (mean over draws) is above 0, which is the flat benchmark and the best
    achievable expectation. Its bootstrap 95% lower bound must be > 0 in at least 3 of 4 NE cells;
  - under FL, the same statistic is NOT significantly above 0 (lower bound ≤ 0) in at least 3 of 4 NE cells.
- **B4 (inflation tracks learned risk-taking).** Across all ZF and LL runs (80 runs), the Spearman correlation between
  ML and the inflation is above 0.5.
  - Inflation is reported minus ledger return (NE: mean over draws).
- **B5 (forgiveness, not termination, drives gambling).** LL forgives only the overshoot of the losing quarter-hour,
  whereas ZF forgives every loss while at zero, so LL should produce less gambling:
  - RS(LL) < RS(ZF) in at least 6 of 8 cells; and
  - ML(LL) < ML(ZF) in at least 3 of 4 NE cells.

Each hypothesis is reported as supported or not supported, whatever the result.

**Exploratory, and labelled as such:**
- probe curves;
- RM results against rule baselines;
- seed medians;
- the reconciliation check;
- per-seed behavior.

## 4. Disclosures
- **Smoke test before registration.** A development smoke test ran before this registration: 3,000 steps, seed 0,
  DK1, evaluation on the TRAINING split only. No test-period outcome was computed before the hash.
- **Changes made after the smoke test, before registration:**
  - the training budget was raised from 200,000 to 400,000 steps, to reduce the risk of a false negative from
    under-training;
  - the NE evaluation was extended to 20 sign draws, because a single draw of a no-edge market has outcome variance
    far larger than any expected effect.
- **Rationale for fixed-allocation sizing.** Positions are sized from a fixed allocation because, with sizing from
  current equity, limited liability can bind only through single-period jumps (the Study A result). This is a design
  choice, stated as a scope condition.
- **Same data as before.** The study uses the same data as Study A. Its test period has been used before, in Study A,
  but not in this environment or under these rules.

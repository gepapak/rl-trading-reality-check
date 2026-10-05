# Study H pre-registration: does learned gambling survive realistic frictions?

Internal pre-registration, hashed (SHA-256, PREREGISTRATION_HASHES.txt) before any test-period evaluation of this study.
Code:
- `envs_h.py`, Study B's environment with frictions;
- `run_h.py`, Study F's runner with the friction environment and rules FL and ZF only;
- `aggregate_h.py`, written and hashed before the run.

## 1. Motivation
Studies B–F used a stylized sleeve: a fee of 0.10 EUR/MWh, a liquidity cap of 25% of the balancing volume, and no price
impact. A reviewer may ask whether the learned gambling is an artifact of frictionless trading.

This study adds:
- a fee of 1.0 EUR/MWh, a common magnitude for intraday execution costs;
- a liquidity cap of 10% of the balancing volume;
- a quadratic price impact of κ q²/cap with κ = 50 EUR/MWh, so trading the full executable volume costs a further
  50 EUR/MWh.

All costs enter the quarter-hour P&L, so the zero floor also forgives costs incurred below zero equity.

## 2. Design calculation (before registration)
`dp_frictions.py` computes the risk-neutral optimum under these frictions (training data, discount 0.99,
`results/dp_frictions.csv`):

| Rule | Optimum by equity level | Gain over staying flat (EUR) |
|---|---|---|
| Full liability | flat everywhere | — |
| Zero floor | 16 at equity ≤ 0.1 × K0; flat at ≥ 0.25 × K0 (both zones) | DK1: 18.8, 15.0, 9.8, 2.4 at equity 0, 0.02, 0.05, 0.1; DK2: 17.4, 13.6, 8.5, 1.5 |

Frictions therefore confine the incentive to near-bankruptcy states, and give full liability a strong reason to stay
flat.

Design note: with the same frictions, the real-market persistence rule still earns +5,000% to +8,000% of the allocation
on the training period. The real-market payoff is a stylization (positions settled against the day-ahead price), so this
study uses the placebo market only.

## 3. Design
- **Environment:** envs_h.FrictionEnv (fixed sizing, K0 = 20,000 EUR, Study B leverage levels, initial training equity
  log-uniform in [0.05, 2] × K0).
- **Market and rules:** placebo market only; rules FL and ZF.
- **Observation sets:** EQ (equity only) and FULL (Study B's 11 inputs).
- **Learner:** PPO as in Study F (8 parallel environments, reward normalization, rollout 1,024, batch 256, 64-64,
  1M steps).
- **Runs:** seeds 0–4, DK1/DK2: 2 rules × 2 observation sets × 10 = 40 runs.
- **Measures:**
  - leverage curve L(e) as in Study F, at e in {0, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 3};
  - placebo returns over 20 test-period sign draws from equity K0.
- **Statistics:** the unit is the run; bootstrap intervals resample runs (10,000 resamples, percentile 95%).

## 4. Hypotheses and decision rules
- **H1 (gambling near bankruptcy survives frictions).** In EQ, mean L_ZF(0.02) − mean L_FL(0.02) ≥ 8, with a 95%
  bootstrap lower bound > 4.
- **H2 (frictions make full liability learnable).** In EQ, L_FL(0.02) ≤ 1 and L_FL(1) ≤ 1 in ≥ 8 of 10 runs.
- **H3 (gambling only near bankruptcy).** In EQ, L_ZF(0.02) > L_ZF(1) in ≥ 8 of 10 runs.
- **H4 (survives all inputs).** In FULL, mean L_ZF(0.02) − mean L_FL(0.02) > 0, with a 95% bootstrap lower bound > 0.
- **H5 (false profit survives frictions).** In EQ, the mean over ZF runs of (reported − ledger) placebo return is > 0,
  with a 95% bootstrap lower bound > 0.

Each hypothesis is reported as supported or not supported. Exploratory: complete curves vs the frictional optimum, FULL
curves, returns.

## 5. Disclosures
- **Prior knowledge.** Designed after Studies B–F were known.
- **Smoke test.** A smoke test ran before registration: 16,384 steps, seed 0, DK1, training split.
- **Launch method.** The run is launched as an independent process, because harness background tasks are stopped after
  a time limit.

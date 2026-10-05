# Study E pre-registration: reward curvature, liability and leverage with equity-scaled positions

Internal pre-registration, hashed (SHA-256, PREREGISTRATION_HASHES.txt) before any test-period evaluation of this study.
Code: `envs_e.py`, `run_e.py`; aggregation `aggregate_e.py` (written and hashed before the run finishes).

## 1. Motivation and theory
Studies B and C showed learned gambling under a zero floor with positions sized from a fixed allocation. Theory (paper,
Section 3) says that with positions scaled to equity, limited liability can bind only through single-period busts, i.e.
when leverage is high enough for one period's loss to exceed equity. Two common environment conventions then differ:

- **LINLL**, a linear (equity-change) reward with the loss capped at the remaining equity on the bust step;
- **LOGNF**, a log reward with the bust step left unrewarded (reward 0), as in Gym-Trading-Env when leverage or shorts
  are enabled.

The reference rule is **LINFL**: a linear reward, with the bust step booking the full loss including the overshoot.

### Pre-computed incentives
The incentives below were computed from the Danish training data before this registration (design calculation, no
learning). The figure is the one-period expected reward in units of equity at equity = allocation, in the placebo
market, with the liquidity cap on.

| Leverage | LINLL | LINFL | LOGNF | P(bust per quarter-hour) |
|---|---|---|---|---|
| 16 | +5.6e-6 (DK1) / −1.4e-5 (DK2) | < 0 | −1.0e-3 / −1.2e-3 | 5e-5 / 0 |
| 64 | +1.2e-3 / +1.1e-3 | < 0 | −3.1e-3 / −1.4e-3 | 1.2e-3 / 9e-4 |

- The LOGNF incentive is negative at every leverage up to 256 in both zones, with or without the cap.
- Prediction: the linear capped-loss rule creates an incentive to gamble only above a leverage threshold (between 16 and
  64 here); full liability and the log rule create none.

## 2. Design
- **Environment:** Study B's sleeve (K0 = 20,000 EUR, Q1 = 0.25 MWh, fee 0.10 EUR/MWh, liquidity cap on, same 11-dim
  observation, same training episodes with initial equity log-uniform in [0.05, 2] × K0). The position is
  `q = clip(lev · Q1 · E/K0, ±cap)`, so it is **scaled to current reported equity**.
- **Rules:** LINFL, LINLL, LOGNF (envs_e.py). The linear reward is scaled by 100/K0 and the log reward by 100.
- **Leverage sets (9 levels each):**
  - LOW: {0, ±0.25, ±1, ±4, ±16};
  - HIGH: {0, ±1, ±4, ±16, ±64}.
- **Market:** the placebo (no-edge) market only. The sign of the spread is randomized independently per quarter-hour.
  Evaluation is on the 20 sign draws of the test period used in Study B (seeds NE_SEED + 1000k), from equity K0.
- **Learners:** PPO and DQN, SB3 defaults, 64-64, γ = 0.99, 400,000 steps; seeds 0–4; DK1 and DK2.
- **Runs:** 3 rules × 2 leverage sets × 2 algorithms × 5 seeds × 2 zones = 120 runs.
- **Measures** (means over the 20 draws, then seed means per cell; a cell is algorithm × zone, 4 cells per rule and
  set):
  - NML: mean |leverage| over open quarter-hours divided by the set's maximum level (comparable across sets);
  - TOP: share of open quarter-hours at the maximum level;
  - BUST: share of draws ending in a bust;
  - reported and ledger returns; the equity probe (exploratory).

## 3. Hypotheses and decision rules
- **E1 (gambling above the bust threshold).** In HIGH, NML(LINLL) > NML(LINFL) in ≥ 3 of 4 cells, and TOP(LINLL) >
  TOP(LINFL) in ≥ 3 of 4 cells.
- **E2 (not below it).** NML(LINLL) − NML(LINFL) is larger in HIGH than in LOW in ≥ 3 of 4 cells.
- **E3 (a log reward neutralizes it).** In HIGH, NML(LOGNF) < NML(LINLL) in ≥ 3 of 4 cells, and TOP(LOGNF) <
  TOP(LINLL) in ≥ 3 of 4 cells.
- **E4 (bankruptcy).** In HIGH, BUST(LINLL) > BUST(LINFL) in ≥ 3 of 4 cells, and BUST(LINLL) > BUST(LOGNF) in ≥ 3 of
  4 cells.
- **E5 (the log rule is no riskier than full liability).** In HIGH, NML(LOGNF) ≤ NML(LINFL) + 0.05 in ≥ 3 of 4 cells.

Each hypothesis is reported as supported or not supported. Exploratory: reported vs ledger returns, equity probes, LOW-set
comparisons other than E2, per-seed behavior.

## 4. Disclosures
- **Smoke test.** A development smoke test ran before registration: 3,000 steps, seed 0, DK1, TRAINING split only,
  3 draws.
- **Design calculation.** The incentives in Section 1 were computed before registration and motivated the design.
  Originally a test of "strategic bankruptcy" under LOGNF was considered, but the calculation predicted no gambling under
  LOGNF at any leverage, so LOGNF became the contrast condition.
- **Equity probe.** It uses the same probe points as Study B, reduced to {0.1, 0.25, 1, 2}.

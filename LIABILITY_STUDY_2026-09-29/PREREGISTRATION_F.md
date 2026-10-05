# Study F pre-registration: do learning agents reach the theoretical liability policy when learning is well posed?

Internal pre-registration, hashed (SHA-256, PREREGISTRATION_HASHES.txt) before any test-period evaluation of this study.
Code: `run_f.py` (reuses `envs_b.py` and `run_b.py` unchanged); aggregation `aggregate_f.py` (written and hashed before
the run).

## 1. Motivation
In Studies B and C, agents moved in the direction the theory predicts, but stayed far from the optimal policy:
- at 10% equity, zero-floor agents chose about 8× against the optimum of 16×;
- full-liability agents chose about 6× against the optimum of 0×.

Diagnosis: a low signal-to-noise ratio.
- In the placebo market, the per-quarter-hour reward varies by roughly ±700 EUR at maximal leverage.
- Under full liability, the only reason to stay flat is the fee, about 0.4 EUR per quarter-hour.
- Every input except equity is pure noise by construction, so the policy must also learn to ignore 10 of its 11 inputs.

## 2. Exploratory pilot (disclosed; training split only)
One seed, DK1, evaluated on the TRAINING split; results in `results_pilot_f/`. PPO was trained with:
- equity as its only input;
- 8 parallel environments;
- reward normalization (VecNormalize, gamma 0.99);
- 1M steps.

It learned:

| | Leverage at equity ≤ 0.5 | Leverage at equity 1–3 |
|---|---|---|
| Zero floor | 16 | 0.25 |
| Full liability | 0.25 at equity ≤ 1 | 4 at equity 2–3 |

Without reward normalization the policies did not depend on equity. These pilot results motivated the design below.

## 3. Design
- **Environment:** Study B's fixed-sizing sleeve (envs_b.LiabilityEnv: K0 = 20,000 EUR, leverage levels
  {0, ±0.25, ±1, ±4, ±16} × 0.25 MWh, liquidity cap, fee 0.10 EUR/MWh, initial training equity log-uniform in
  [0.05, 2] × K0). Placebo market only.
- **Rules:** FL, LL, ZF (as in Study B).
- **Observation sets:**
  - EQ: reported equity / K0 is the only input;
  - FULL: Study B's 11 inputs.
- **Learner:** PPO (Stable-Baselines3).
  - 8 parallel environments, rollout 1,024 steps per environment, batch 256;
  - reward normalization (VecNormalize, norm_obs = False, gamma 0.99);
  - network 64-64, all other settings at defaults;
  - 1,000,000 steps.
- **Runs:** seeds 0–4; zones DK1, DK2; 3 rules × 2 observation sets × 5 seeds × 2 zones = 60 runs.
- **Measures:**
  - leverage curve L(e): the greedy |leverage| at equity input e × K0, for e in
    {0, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 3}. EQ: exact. FULL: mean over every 8th test quarter-hour with the equity
    input set to e, as in Study B.
  - Placebo returns: reported and ledger returns over 20 sign draws of the TEST period (Study B's NE draws) from
    equity K0.
- **Unit of analysis:** the run (10 runs per rule and observation set). Bootstrap intervals resample runs
  (10,000 resamples, percentile 95%).

## 4. Hypotheses and decision rules (effect-size based)
- **F1 (zero floor reaches the optimum at low equity).** In EQ, L_ZF(0.1) = 16 in ≥ 8 of 10 runs.
- **F2 (full liability nearly flat).** In EQ, L_FL(0.1) ≤ 1 in ≥ 8 of 10 runs.
- **F3 (a large effect).** In EQ, mean L_ZF(0.1) − mean L_FL(0.1) ≥ 8 (half the theoretical gap of 16), and the 95%
  bootstrap lower bound of the difference is > 4.
- **F4 (gambling for resurrection: more leverage near zero).** In EQ, L_ZF(0.1) > L_ZF(2) in ≥ 8 of 10 runs.
- **F5 (noise inputs dilute learning).** The mean gap L_ZF(0.1) − L_FL(0.1) is larger in EQ than in FULL. The 95%
  bootstrap interval of the difference of gaps (EQ minus FULL; runs resampled within each set) lies above 0.
- **F6 (capped loss teaches less gambling within this budget).** In EQ, mean L_ZF(0.1) − mean L_LL(0.1) > 0, with a
  95% bootstrap lower bound > 0. Theory: both optima are 16 at e ≤ 1, but the LL incentive is about a quarter of ZF's,
  so it should be learned more slowly.

Each hypothesis is reported as supported or not supported.

**Exploratory:**
- complete leverage curves and their agreement with dp_theory.py;
- placebo-market reported vs ledger returns;
- FULL-condition curves;
- per-seed results.

## 5. Disclosures
- **Pilot.** The pilot of Section 2 (training split only) motivated the design.
- **Smoke test.** A smoke test ran before registration (16,384 steps, seed 0, DK1, training split).
- **Comparison with Studies B and C.** The learner configuration differs from Studies B and C (parallel environments
  and reward normalization). This study asks what agents learn when learning is well posed, not what default settings
  learn.

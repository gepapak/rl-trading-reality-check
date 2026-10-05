# Study G pre-registration: learned behavior inside a widely used public simulator (gym-mtsim)

Internal pre-registration, hashed (SHA-256, ../PREREGISTRATION_HASHES.txt) before any evaluation on the test price paths.
Code: `mtsim_common.py`, `run_g.py`, `aggregate_g.py` (written and hashed before the run).

## 1. Motivation
The audit of public RL trading environments (Study D) found that gym-mtsim (v2.0.0, AminHP; MetaTrader-5 simulator and
Gymnasium environment) floors the account balance at zero after a stop-out. Orders are sized in absolute lots, at a
default leverage of 100, and the reward is the change in equity. This study trains agents inside gym-mtsim's own code, to
see whether that rule changes what they learn.

## 2. Pre-registration analysis of gym-mtsim's accounting
- **The rule type.** After a stop-out with a negative balance, the balance is set to zero. The margin check then refuses
  new orders, so the account cannot trade again. In the paper's terms this is the **capped-loss rule**: only the
  overshoot of the losing step is forgiven.
- **Sizing.** The margin check limits the position to roughly equity × leverage, so exposure partly shrinks with equity.
- **Prediction from theory (Propositions 3–4).** A weak incentive, confined to near-maximal leverage where single-bar
  losses exceed equity.
- **Design check** (synthetic placebo prices, constant positions, published code with the fee set to zero):
  - at about 70× leverage, 15% of 500-hour episodes are wiped out, and the floor hides 8.4 percentage points of losses
    on average (reported −65.7% vs booked −74.1%);
  - at 4× and 18× leverage the floor never binds.

## 3. Design
- **Simulator.** gym-mtsim 2.0.0 installed unchanged in an isolated environment (pandas 2.0.3 there, to read the
  bundled symbol data).
  - Symbol: EURUSD's bundled contract specification.
  - Leverage 100, stop-out level 0.2, hedge mode, `symbol_max_orders` = 1, the environment's published default fee.
- **Conditions:**
  - ORIG: gym-mtsim as published;
  - FLPATCH: a subclass whose `tick()` is the published code minus the two lines that floor the balance (full
    liability).
- **Market.** Synthetic placebo prices: hourly log returns i.i.d. Student-t(3), scaled to 0.15% per hour, with no drift.
  - Training path: seed 1000, 30,000 hours.
  - Test paths: seeds 2000–2019, 2,000 hours each, from a balance of 10,000 USD.
- **Observation (wrapper):** [equity, balance, margin, open volume, open profit], scaled (price features carry no
  information in the placebo market).
- **Training episodes:** 500 hours at a random start, with initial balance log-uniform in [0.05, 2] × 10,000.
  Episode handling and the observation are the only wrapper changes; all accounting is gym-mtsim's.
- **Learner:** PPO (Stable-Baselines3 2.7.0).
  - 8 parallel environments, observation and reward normalization (VecNormalize);
  - rollout 1,024, batch 256, network 64-64;
  - 1,000,000 steps.
- **Runs:** seeds 0–9 for each condition: 20 runs.
- **Measures:**
  - **Exposure x(e):** the fraction of the margin-feasible maximum position that the deterministic policy opens at
    equity e × 10,000 (0 if it holds), for e in {0.02, 0.05, 0.1, 0.25, 0.5, 1, 2}.
  - **Test returns:** reported (ORIG) and booked (the same policy on the same test path under FLPATCH) returns.

## 4. Hypotheses and decision rules (runs as units, 95% percentile bootstrap, 10,000 resamples)
- **G1 (the floor induces risk-taking near bankruptcy).** Mean x_ORIG(0.1) − mean x_FLPATCH(0.1) > 0, with a lower bound
  > 0.
- **G2 (the floor induces more risk-taking overall).** The mean over e ≤ 0.25 of x_ORIG minus that of x_FLPATCH is > 0,
  with a lower bound > 0.
- **G3 (reporting gap).** For ORIG-trained agents, the mean of (reported − booked) test return is > 0, with a lower bound
  > 0.

Each hypothesis is reported as supported or not supported. **The theory predicts small effects; null results will be
reported as such.** Exploratory: exposure curves, wipe-out shares, per-seed results.

## 5. Disclosures
- **Smoke test.** A smoke test (16,384 steps, seed 0, both conditions, 2 short test paths) and the design check of
  Section 2 ran before registration.
- **Fee.** The design check used a fee of zero; the study uses the published default fee.
- **Prior knowledge.** Designed after Studies B–F and during Study H.

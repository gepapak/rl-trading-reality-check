# Study I pre-registration: is the well-posed result of Study F specific to PPO?

Internal pre-registration, hashed (SHA-256, PREREGISTRATION_HASHES.txt) before any test-period evaluation of this study.
Code: `run_i.py` (reuses `envs_b.py` and `run_b.py` unchanged); aggregation `aggregate_i.py` (written and hashed before
the run). Pilot: `pilot_i.py` (training split only), results in `results_pilot_i/`.

## 1. Motivation
In Study F, all ten zero-floor PPO agents with equity as their only input learned the threshold policy of the theory
(L(0.1) = 16; gap to full liability 11.1 [7.1, 14.3]). A reviewer can ask whether this is specific to PPO. This study
repeats Study F's EQ condition with two learners that differ from PPO in kind:
- A2C: on-policy actor-critic without a clipped objective, epochs or minibatches;
- DQN: off-policy and value-based (Q-learning with replay and a target network).

## 2. Exploratory pilot (disclosed; training split only)
One seed (0), DK1, rules FL and ZF, evaluated on the TRAINING split at 250k, 500k and 1M steps. All runs use Study F's
set-up: equity as the only input, 8 parallel environments, reward normalization (VecNormalize, gamma 0.99), 64-64
network.

Pilot settings tried (final 1M-step policies; leverage at equity 0, 0.05, 0.1, 0.25, 0.5, 1, 2, 3 × K0):

| Configuration | ZF policy | FL policy | Distance from optimum (FL, ZF, mean) |
|---|---|---|---|
| A2C, SB3 defaults (n_steps 5) | 16 everywhere | 16 everywhere | 16.00, 3.56, 9.78 |
| A2C, n_steps 256, GAE 0.95, advantage normalization | 16 everywhere | 0.25 everywhere | 0.25, 3.56, 1.90 |
| A2C, n_steps 64, GAE 0.95, advantage normalization | 16 at e ≤ 1, 1 at e ≥ 2 | 4 everywhere | 4.00, 0.22, 2.11 |
| DQN, SB3 defaults | 16 at e ≤ 2, 1 at e = 3 | 4 at e ≤ 0.5, 16 at e = 1–2, 1 at e = 3 | 6.33, 1.89, 4.11 |
| DQN, 8 gradient steps per update | 16 at e ≤ 0.05, 1–4 above | 1–4 | 2.67, 5.56, 4.11 |

Distance: mean |L(e) − L*(e)| over the equity grid {0, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 3}, with the risk-neutral
optimum L* of dp_theory.py (FL: 0 everywhere; ZF: 16 at e ≤ 1, 0 at e ≥ 2); computed by `select_pilot_i.py`.
DQN policies changed markedly between the 250k, 500k and 1M snapshots under both DQN settings.

**Selection rule (applied after the pilot, stated here before any test-period run).** For each learner, the
configuration with the smallest mean distance is used; where two configurations are within 0.25 of each other, both
are run, and where they tie, the cheaper (library default) configuration is used. This gives:
- A2C-n256 (primary A2C configuration, distance 1.90) and A2C-n64 (within 0.21; run as a pre-registered sensitivity
  configuration). The two differ qualitatively in the pilot: n256 learned near-flat full liability but no equity
  dependence under the floor; n64 learned the threshold shape under the floor but 4× under full liability.
- DQN-default (tie with the 8-gradient-step setting at 4.11; the default is three times cheaper and is the setting
  of Studies B, C and E).

The pilot used seed 0 and DK1 on the TRAINING split; the study evaluates seeds 0–4 in DK1 and DK2 on the TEST split.

## 3. Design
- **Environment:** Study B's fixed-sizing sleeve (envs_b.LiabilityEnv), placebo market, equity as the only input; as in
  Study F's EQ condition.
- **Rules:** FL, LL, ZF.
- **Learners:**
  - **A2C-n256:** A2C, n_steps 256 per environment, gae_lambda 0.95, normalize_advantage True, other settings at
    SB3 2.7.0 defaults (learning rate 7e-4, RMSprop);
  - **A2C-n64:** as A2C-n256 with n_steps 64 (sensitivity configuration);
  - **DQN-default:** DQN at SB3 2.7.0 defaults (buffer 1e6, learning starts 100, batch 32, train frequency 4,
    1 gradient step, target update 10,000, exploration 1.0 → 0.05 over 10% of steps, learning rate 1e-4);
  - all: 8 parallel environments (DummyVecEnv), reward normalization (VecNormalize, norm_obs False, gamma 0.99),
    64-64 network, 1,000,000 steps.
- **Runs:** 3 learner configurations × 3 rules × seeds 0–4 × zones DK1, DK2 = 90 runs, 1,000,000 steps each.
- **Measures:** as Study F. Leverage curve L(e): greedy |leverage| at equity input e × K0 for e in
  {0, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 3} (exact). Placebo returns over the 20 sign draws of the TEST period from
  equity K0.
- **Unit of analysis:** the run (10 per configuration and rule). Bootstrap intervals resample runs (10,000 resamples,
  percentile 95%).

## 4. Hypotheses and decision rules (per configuration; effect-size based, as Study F)
Each hypothesis is evaluated separately for A2C-n256, A2C-n64 and DQN-default (nine verdicts). A2C-n256 is the
primary A2C configuration; A2C-n64 is reported as a sensitivity configuration with its own verdicts.
- **I1 (a large effect).** Mean L_ZF(0.1) − mean L_FL(0.1) ≥ 8 (half the theoretical gap of 16), and the 95% bootstrap
  lower bound of the difference is > 4. (Study F's F3.)
- **I2 (gambling for resurrection: more leverage near zero).** L_ZF(0.1) > L_ZF(2) in ≥ 8 of 10 runs. (Study F's F4.)
- **I3 (capped loss teaches less gambling).** Mean L_ZF(0.1) − mean L_LL(0.1) has a 95% bootstrap lower bound > 0.
  (Study F's F6.)

Each hypothesis is reported as supported or not supported for each configuration.

**Exploratory:** complete leverage curves; L_ZF(0.1) = 16 counts; L_FL(0.1) ≤ 1 counts; Cohen's d; placebo-market
reported vs ledger returns; comparison with Study F's PPO results.

## 5. Disclosures
- **Pilot.** The pilot of Section 2 (training split only, one seed) chose the learner configurations by the rule
  stated there. All configurations tried, including the failed defaults, are reported there. The selection rule was
  written after the pilot results were seen.
- **Smoke test.** A smoke test ran before registration (16,384 steps, seed 0, DK1, training split; results_i/jobs_smoke).
- **Relation to Study F.** The hypotheses and thresholds are Study F's F3, F4 and F6, unchanged.

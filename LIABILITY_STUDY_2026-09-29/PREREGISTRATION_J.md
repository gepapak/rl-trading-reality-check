# Study J pre-registration: does the well-posed result hold in other markets, and does it scale as the theory predicts?

Internal pre-registration, hashed (SHA-256, PREREGISTRATION_HASHES.txt) before any test-period evaluation of this study.
Code: `run_j.py` (Study F's run_f.py with the market and seeds changed; learner, environment and evaluation unchanged),
`envs_j.py` (loader), `build_panel_j.py` (data), `dp_j.py` (theory check), aggregation `aggregate_j.py` (written and
hashed before the run). No pilot: the learner configuration is Study F's, unchanged.

## 1. Motivation
All learning studies so far used the Danish imbalance spread (DK1, DK2). A reviewer can ask whether the result is
specific to Denmark. This study repeats Study F's well-posed condition (PPO, equity as the only input) in two other
Nordic bidding zones, chosen before any modelling:
- **FI** (Finland, Fingrid): the only Finnish zone; mean |spread| 60 EUR/MWh, close to DK1's 66;
- **NO2** (southern Norway, Statnett): the Norwegian zone with the largest mean absolute imbalance volume;
  hydro-dominated, mean |spread| 12 EUR/MWh, about one fifth of the Danish and Finnish level.

## 2. Data
eSett Open Data already in Paper1 (INVESTOR_ACCURACY_TRAP_2026-09-28/data_external_esett), copied unchanged to
data_esett_j/ (hashes printed by build_panel_j.py and matching the original download manifest):
- EXP14 (imbalance price; imbalance-spot difference, from which the day-ahead price is derived; validated on DK1,
  where the derived day-ahead price equals Energinet's in 100% of quarter-hours);
- EXP13 (net area imbalance volume, used for the liquidity cap as |imbalance| × 4 MW, the analogue of Denmark's
  |SatisfiedDemand|).
Splits as for Denmark: training 2025-03-04 to 2025-12-31, test 2026-01-01 to 2026-08-17. 51,067 quarter-hours per zone.

## 3. Theory check (run before registration; dp_j.py, results_j/dp_j.csv)
dp_theory.py's model and solver, unchanged, on each zone's training distribution. Risk-neutral optimal |leverage|:

| Zone, rule | e=0 | 0.02 | 0.05 | 0.1 | 0.25 | 0.5 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|---|---|
| FI, FL | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| FI, LL | 16 | 16 | 16 | 16 | 16 | 16 | 0 | 0 | 0 |
| FI, ZF | 16 | 16 | 16 | 16 | 16 | 16 | 0 | 0 | 0 |
| NO2, FL | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| NO2, LL | 16 | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 0 |
| NO2, ZF | 16 | 16 | 16 | 16 | 0 | 0 | 0 | 0 | 0 |

Value of the optimal action over staying flat under ZF (EUR): FI 28.1, 24.8, 20.7, 15.5, 6.9, 1.8, 0, 0, 0;
NO2 5.35, 2.81, 1.06, 0.10, 0, 0, 0, 0, 0 (same equity levels). For comparison, DK1/DK2 (Table tab:dp): 36.8 at
e = 0.02, 26.9 at e = 0.1, 15.8 at e = 0.25.

The theory therefore predicts that FI behaves like Denmark, and that in NO2 the gambling region shrinks toward zero
equity: the floor matters only where a few quarter-hours of small spreads can exhaust equity.

## 4. Design
- **Environment:** envs_b.LiabilityEnv (unchanged), placebo market (sign seeds 9501/9502 FI, 9601/9602 NO2),
  equity as the only input.
- **Rules:** FL, LL, ZF.
- **Learner:** Study F's PPO, unchanged (8 parallel environments, rollout 1,024 per environment, batch 256, reward
  normalization, 64-64 network, 1,000,000 steps).
- **Runs:** seeds 0–9 in each zone (ten runs per zone and rule, as Study F's ten per rule): 3 rules × 10 seeds ×
  2 zones = 60 runs.
- **Measures:** as Study F (leverage curve L(e), exact; placebo returns over 20 sign draws of the TEST period).
- **Unit of analysis:** the run; bootstrap intervals resample runs (10,000 resamples, percentile 95%).

## 5. Hypotheses and decision rules (effect-size based)
- **J1 (FI replicates F3).** FI: mean L_ZF(0.1) − mean L_FL(0.1) ≥ 8, with 95% bootstrap lower bound > 4.
- **J2 (FI replicates F4).** FI: L_ZF(0.1) > L_ZF(2) in ≥ 8 of 10 runs.
- **J3 (FI replicates F6).** FI: mean L_ZF(0.1) − mean L_LL(0.1) has a 95% bootstrap lower bound > 0.
- **J4 (the gambling region shrinks with spread volatility).** At e = 0.25, where the optimum is 16 in FI and 0 in
  NO2, the gap L_ZF − L_FL is larger in FI than in NO2: the 95% bootstrap interval of (gap FI − gap NO2), runs
  resampled within each zone and rule, lies above 0.
- **J5 (gambling persists near zero equity in the calm market).** NO2: mean L_ZF(0.02) − mean L_FL(0.02) has a
  95% bootstrap lower bound > 0.

Each hypothesis is reported as supported or not supported.

**Exploratory:** complete curves against dp_j.py; NO2 analogues of J1–J3; placebo-market reported vs ledger returns;
FL nearly flat counts.

## 6. Disclosures
- **No pilot.** The learner configuration is Study F's. A smoke test ran before registration (16,384 steps, seed 0,
  both zones, training split; results_j/jobs_smoke).
- **Theory check before registration.** dp_j.py ran before registration; its results set the hypotheses J4 and J5.
- **Zone choice.** FI and NO2 were chosen from the zones with local eSett price and volume data (FI, NO1–NO5) by the
  rule stated in Section 1, after their spread statistics had been computed but before any agent was trained.

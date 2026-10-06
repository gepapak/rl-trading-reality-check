# Study K pre-registration: does patience make floor-trained agents gamble over a wider equity range?

Internal pre-registration, hashed (SHA-256, PREREGISTRATION_HASHES.txt) before any test-period evaluation of this study.
Code: `run_k.py` (Study F's run_f.py with the discount factor varied and a wider training range of initial equity),
aggregation `aggregate_k.py`, theory `theory_k.py`, `reflection_barrier.py`, `reflection_theory.py`; predictions in
`results_k/theory_k.csv`. No pilot: the learner configuration is Study F's.

## 1. Theory and prediction
Under the zero floor, reported equity follows Lindley's recursion E_{t+1} = max(E_t + x_t, 0): the ledger reflected at
zero, whose regulator is the cumulative forgiven loss. In the diffusion limit the agent controls the volatility of a
reflected process that is recapitalized for free. Volatility and fees both scale with the position, the value is
convex, and the optimal control is bang-bang: maximal leverage below a barrier b*, flat above. The barrier scales with
the length scale sigma / sqrt(2 (1 - gamma)): more patient agents value the stream of future forgiven losses more, so
the gambling region widens with the discount factor gamma. Under full liability the optimum is flat at every gamma.

Theory check before registration (`theory_k.py`; DP on the empirical training distribution, equity grid to 6K):

| Zone | gamma | DP gambling region (K) | Closed-form barrier (K) |
|---|---|---|---|
| DK1 | 0.95 / 0.98 / 0.99 / 0.995 | 1.21 / 1.59 / 2.01 / 2.60 | 0.51 / 0.87 / 1.27 / 1.85 |
| DK2 | 0.95 / 0.98 / 0.99 / 0.995 | 1.06 / 1.38 / 1.74 / 2.25 | 0.52 / 0.87 / 1.29 / 1.86 |

Predicted ratio of the gambling region at gamma 0.995 to 0.95: about 2.1 (DP) to 3.6 (closed form). Learned regions are
expected to be smaller than the optimum (Study F's agents reached about half the DP region at gamma 0.99).

## 2. Design
- **Environment:** envs_b.LiabilityEnv (unchanged), placebo market, equity as the only input. Initial training equity
  is drawn log-uniformly in [0.05, 6] allocations instead of [0.05, 2] (set in run_k.py via envs_b.E0_TRAIN), so that
  wide gambling regions can be learned and observed. The equity input is clipped at 5 as before.
- **Conditions:** zero floor at gamma 0.95, 0.98, 0.99, 0.995; full liability at gamma 0.95 and 0.995 (control).
- **Learner:** Study F's PPO (8 parallel environments, rollout 1,024, batch 256, reward normalization, 64-64 network,
  1,000,000 steps), with PPO's gamma and VecNormalize's gamma both set to the condition's gamma.
- **Runs:** seeds 0-4, zones DK1 and DK2: 6 conditions x 5 seeds x 2 zones = 60 runs.
- **Measures:** leverage curve L(e) at e in {0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5};
  gambling region T of a run: the largest probed e such that L(e') >= 8 for every probed e' <= e (T = 0 if L(0) < 8);
  placebo returns over 20 test-period sign draws, as Study F.

## 3. Hypotheses and decision rules
- **K1 (patience widens the gambling region).** Over the 40 zero-floor runs, the Spearman correlation between gamma and
  T is positive with a one-sided permutation p < 0.05 (10,000 permutations).
- **K2 (the extremes differ).** Mean T at gamma 0.995 minus mean T at gamma 0.95 (zero floor, ten runs each) has a 95%
  bootstrap lower bound > 0.
- **K3 (the magnitude is substantial).** Mean T at gamma 0.995 is at least 1.5 times mean T at gamma 0.95 (and > 0).

Each hypothesis is reported as supported or not supported.

**Exploratory:** full-liability thresholds and curves at both gamma values; learned versus DP and closed-form regions;
collapse of the learned curves when equity is rescaled by the theoretical length scale (also with Studies F and J);
placebo returns.

## 4. Disclosures
- **No pilot.** A smoke test ran before registration (16,384 steps, seed 0, DK1, training split; results_k/jobs_smoke).
- **Theory before registration.** The DP and closed-form predictions (results_k/theory_k.csv) were computed before
  registration and set the hypotheses; the closed-form theory (reflection_barrier.py) was developed on 5 October 2026
  after Studies B-J, using their DP benchmarks.
- **Changed training range.** The wider initial-equity range differs from Studies B-J and is the same in all
  conditions of this study.

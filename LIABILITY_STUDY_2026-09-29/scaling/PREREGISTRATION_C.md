# Study C pre-registration: does learned gambling grow with training?

Internal pre-registration, hashed (SHA-256, ../PREREGISTRATION_HASHES.txt) before any test-period evaluation of this study.
Code: `run_c.py`, which reuses `../envs_b.py` and `../run_b.py` (evaluate, probe) unchanged.

## Motivation
Study B found that agents trained under a zero floor (ZF) choose more leverage the closer they are to zero equity
(the resurrection slope RS > 0, B2 supported). The shape matches the theoretical incentive (`../dp_theory.py`).

However, the learned level (6–10×) is far below the risk-neutral optimum (16×). Two questions follow:
- Is this under-exploitation a matter of training?
- Does the liability effect grow as the learner gets stronger?

Reward hacking is known to grow with agent capability (Pan, Bhatia and Steinhardt 2022). In human experiments, the
moral hazard of limited liability grows with repetition.

## Design
- **Environment and market:** Study B's environment in the no-edge market (NE) only, where the ground truth is known
  (optimal full-liability policy: flat; optimal zero-floor policy: maximum leverage up to one allocation of equity).
- **Rules:** FL (full liability) and ZF (zero floor).
- **Algorithms:** PPO and DQN (Stable-Baselines3 defaults, network 64-64).
- **Seeds:** fresh seeds 10, 11 and 12 (Study B used 0–4).
- **Regions:** DK1, DK2.
- **Runs:** 2 rules × 2 algorithms × 3 seeds × 2 regions = 24 runs.
- **Training:** one run of 1,600,000 steps per job, with snapshots at 100k, 200k, 400k, 800k and 1.6M steps. DQN's
  exploration schedule therefore spans the first 160k steps, so the 400k snapshot is not identical to Study B's
  400k run.
- **Evaluation at each snapshot**, identical to Study B:
  - evaluation on 20 independent sign draws of the test period, starting at equity = K0;
  - the probe P(e) at e in {0, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2};
  - RS = P(0.1) − P(1).

A cell is (algorithm, region), giving 4 cells per rule. Seed means are over the 3 seeds.

## Hypotheses and decision rules
- **C1 (gambling grows with training under ZF).** The seed-mean RS under ZF at 1.6M exceeds that at 100k in at least 3
  of 4 cells, and the Spearman correlation between log(steps) and the seed-mean RS under ZF across the 5 snapshots is
  positive in at least 3 of 4 cells.
- **C2 (towards the optimum).** The seed-mean P(0.1) under ZF at 1.6M exceeds that at 100k in at least 3 of 4 cells.
- **C3 (the growth is specific to limited liability).** The change in RS from 100k to 1.6M is larger under ZF than under
  FL in at least 3 of 4 cells.
- **C4 (the false profit grows).** Under ZF, the seed-mean inflation (reported minus ledger return, mean over draws) at
  1.6M exceeds that at 100k in at least 3 of 4 cells.
- **C5 (replication of B2 on fresh seeds).** At the 400k snapshot, RS(ZF) > RS(FL) in at least 3 of 4 cells.

Each hypothesis is reported as supported or not supported.

**Exploratory:**
- the full-liability trajectory (whether FL agents learn to stay flat with more training);
- per-seed trajectories;
- reported and ledger returns per snapshot.

## Disclosures
- A development smoke test ran before registration (milestones at 1k/2k/3k steps, seed 10, DK1, TRAINING split only).
- Study B's results (including B2 and the theory check) were known when this study was designed. That is its purpose:
  a follow-up on training scale and a fresh-seed replication.

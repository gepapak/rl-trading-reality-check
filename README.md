# When the simulator forgives losses

**Reinforcement-learning trading agents learn to gamble for resurrection in electricity balancing markets** — code, data pipelines, pre-registrations and aggregated results.

## The question
Trading simulators often floor an account's cash at zero, or end an episode at bankruptcy, as a guard against negative balances. Corporate finance has long known what a truncated downside does to a decision maker: limited liability rewards risk, most strongly near zero equity ("gambling for resurrection"). This repository asks whether reinforcement-learning (RL) agents *learn* that incentive from a simulator's accounting, and whether standard evaluation then reports the behavior as profit.

## Main results
- **Theory.** Five propositions with proofs: in a market without edge, full liability makes staying flat optimal, while a zero floor on fixed positions makes maximal leverage optimal, most valuably near zero equity. Capping only the final loss forgives at most one overshoot. With equity-scaled positions the hazard appears only above a leverage threshold, and a log reward removes it under a computable condition.
- **Origin.** In an audit of a validated electricity-market engine, a legacy protocol reported a learned controller first (**+270%**) while its booked P&L was **−535%**. A missing liquidity cap and missing solvency rules were jointly necessary.
- **Agents learn the incentive** (placebo market with sign-randomized imbalance spreads, where no policy can profit):
  - with default training, floor-trained agents raised leverage as equity fell and reported **+38% to +148%** while booking **−15% to −57%** (Study B); this replicated on fresh seeds and grew with training (Study C);
  - when learning was well posed, **all ten floor-trained PPO agents learned the theory's threshold policy**, 11.1 leverage units of 16 above full-liability agents (Cohen's d = 2.6, one-sided permutation p = 0.0004; Study F);
  - with A2C and DQN the floor raised leverage near zero equity by 9.2 units pooled, though about a third of their floor-trained agents gambled at every equity level (Study I);
  - in Finland PPO agents again learned the threshold policy, exactly where the Finnish optimum is maximal; in calm southern Norway, where the theory makes the incentive negligible, the effect vanished (Study J).
- **Boundaries.** With realistic trading costs (Study H), inside the public simulator gym-mtsim (Study G) and with equity-scaled positions (Study E), agents did not learn to gamble, or did so only weakly. Reported returns still exceed booked ones wherever a floor binds.
- **Prevalence.** One of the 18 most-starred public RL trading environments meets both enabling conditions (Study D); FinRL and TensorTrade do not; 20 of 21 public electricity-RL environments have no capital account.
- **Tools.** A reconciliation check of reported against booked returns (flagged 53 of 54 affected engine runs, none of 246 others), a placebo-market test, and an evaluation checklist (`CHECKLIST.md`).

All pre-registered hypotheses are reported, including the 19 of 43 liability hypotheses that were not supported.

## Studies and where to find them
| Study | Question | Folder | Pre-registration | Aggregated results |
|---|---|---|---|---|
| Engine audit | Which evaluation shortcuts decide the verdict on frozen controllers? | `engine/`, `audit/`, `results/` | `docs/PREREGISTRATION.md` | `results/` |
| A | When do simulator shortcuts reverse "learning beats the rule" (three environment designs)? | `GENERALIZATION_STUDY_2026-09-29/` | `PREREGISTRATION_A.md` | `results/` there |
| B | Do agents learn to gamble under a zero floor? | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_B.md` | `results/` |
| C | Does the gambling grow with training (fresh seeds)? | `LIABILITY_STUDY_2026-09-29/scaling/` | `PREREGISTRATION_C.md` | `scaling/results/` |
| D | How common are the enabling conditions in public trading environments? | `LIABILITY_STUDY_2026-09-29/env_audit/` | `PREREGISTRATION_D.md` | `coding_d.csv` |
| E | Equity-scaled positions and the log reward | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_E.md` | `results_e/` |
| F | Well-posed learning (PPO, equity as the only input) | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_F.md` | `results_f/`, pilot `results_pilot_f/` |
| G | A public simulator (gym-mtsim) | `LIABILITY_STUDY_2026-09-29/mtsim_study/` | `PREREGISTRATION_G.md` | `results_g/` |
| H | Realistic trading costs | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_H.md` | `results_h/` |
| I | Other learners (A2C, DQN) | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_I.md` | `results_i/`, pilot `results_pilot_i/` |
| J | Other markets (Finland, southern Norway) | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_J.md` | `results_j/` |

Each study folder also holds `PREREGISTRATION_HASHES.txt` (SHA-256 of every registered file, with UTC timestamps) and `RUN_LOG.md` (launches, interruptions, relaunches and changes of job order). Theory solvers: `dp_theory.py` (Denmark), `dp_frictions.py` (Study H), `dp_j.py` (Study J). Placebo-market test: `placebo_market.py`. Exploratory analyses are in `review_rigor.py` and `explore_i.py`.

## Repository layout
```
README.md, DESCRIPTION.md        overview; short description and repository metadata
CITATION.cff                     citation metadata
CHECKLIST.md                     evaluation checklist for RL trading agents and their simulators
LICENSE, LICENSE-docs.md         MIT (code); CC BY 4.0 (docs, results, derived tables)
THIRD_PARTY_LICENSES.md          sources, licenses and attribution for all third-party material
requirements*.txt                standalone analyses; engine stack; Studies A-J; Study G (gym-mtsim)
tools/verify_studies.py          checks every pre-registration hash of Studies A-J and recomputes their headline numbers
tools/verify_release.py          recomputes every headline number of the engine audit
tools/ledger_check.py            standalone reported-vs-booked-P&L reconciliation
GENERALIZATION_STUDY_2026-09-29/ Study A: environments, runner, aggregation, Energinet data and panels, results
LIABILITY_STUDY_2026-09-29/      Studies B-J: environments, runners, aggregation, theory solvers, results
engine/, audit/                  engine audit: simulator source (patched audit copy) and audit code
analysis/, code_audit/           cross-market liquidity test; code audit of 21 electricity-RL repositories
literature/                      pilot literature survey coding (no PDFs)
results/                         engine-audit results
data/                            Energinet data (CC BY 4.0), engine liquidity series, derived tables;
                                 fetch_esett.py downloads the eSett files (not redistributed)
docs/                            engine-audit protocols, pre-registration and methods
```

## Reproducing
Python 3.10. CPU only.

**Verification without training (minutes).**
```
pip install -r requirements-studies.txt
python tools/verify_studies.py      # pre-registration hashes of Studies A-J and their headline numbers
python tools/verify_release.py      # headline numbers of the engine audit
python tools/ledger_check.py results/ledger_examples/L3_marl_seed7_DK1.csv.gz   # a flagged run
```

**Theory and aggregation from the shipped per-run results.** In `LIABILITY_STUDY_2026-09-29/`:
```
python dp_theory.py                 # optimal policies, Denmark (Section 3 of the paper)
python aggregate_b.py               # and aggregate_e.py, aggregate_f.py, aggregate_h.py, aggregate_i.py, aggregate_j.py
python placebo_market.py --demo     # the placebo-market test on a floor and a full-liability agent
```
Figure and table scripts (`make_*.py`) write to `outputs/` inside each study folder.

**Retraining (hours to days on a multi-core CPU).** Each runner is resume-safe and skips finished jobs.
```
cd GENERALIZATION_STUDY_2026-09-29 && python run_study.py --phase rules && python run_study.py --phase rl --workers 6
cd LIABILITY_STUDY_2026-09-29
python run_b.py --phase rules && python run_b.py --phase rl --workers 6   # Study B (the study relaunched with run_b_order.py; see RUN_LOG.md)
python scaling/run_c.py --workers 7                                       # Study C
python run_e.py --workers 7 ; python run_f.py --workers 7 ; python run_h.py --workers 7 ; python run_i.py --workers 7
```
Study B took about 8 hours and Study I about 2 hours with 7 workers on an 8-core CPU.

**Study J (Finland, southern Norway).** eSett data are not redistributed:
```
python data/fetch_esett.py                                   # from the repository root; verifies content hashes
cd LIABILITY_STUDY_2026-09-29
python get_esett_j.py && python build_panel_j.py && python dp_j.py && python run_j.py --workers 7
python ../tools/verify_studies.py                            # the rebuilt panels must match their registered hashes
```

**Study G (gym-mtsim)** needs its own environment, because gym-mtsim's bundled data load only with pandas 2.0.x:
```
cd LIABILITY_STUDY_2026-09-29/mtsim_study
python -m venv .venv && .venv\Scripts\pip install -r ../../requirements-mtsim.txt
.venv\Scripts\python run_g.py --workers 7 && .venv\Scripts\python aggregate_g.py
```

**Engine audit (hours).** The engine source is in `engine/`; its datasets and trained checkpoints (about 57 GB) are not included. `pip install -r requirements-engine.txt`, then see `docs/RUNNING_THE_CAMPAIGN.md`. `audit/setup_audit_engine.py` copies the engine and applies the three-block patch in `audit/engine_patch/`, which bypasses the train/eval contract-hash check only when `SIM_AUDIT_ALLOW_PROTOCOL_MISMATCH=1` and applies config overrides from `SIM_AUDIT_CFG_OVERRIDES`.

## Pre-registration
All pre-registrations are internal: written and hashed before the corresponding test-period evaluation, not deposited with an independent registry. Their records disclose smoke tests on training data, design changes made before registration, a change of job submission order during Study B, a configuration-selection rule for Study I written after its pilot had been seen, aggregation scripts written after the runs for Study C, and, in the engine audit, timestamp corrections and one post-hoc correction of an aggregation script. `tools/verify_studies.py` checks every registered file against its recorded hash.

## Not included
- Trained checkpoints and raw per-run logs (size). Per-run summaries are in each study's results folder (`jobs/` and `runs_*.csv`).
- Engine datasets and checkpoints (about 57 GB) and raw engine logs (about 105 GB).
- Raw eSett Open Data and the Study J panels derived from them (no explicit redistribution license); `data/fetch_esett.py` re-downloads them.
- Third-party source code from the audits (copyright); the scan scripts regenerate the excerpts from pinned commits.
- The paper PDFs of the literature survey (copyright).

## Data sources
- **Energinet, Energi Data Service** (`ImbalancePrice`, `ProductionConsumptionSettlement`, `Forecasts_Hour`; `RegulatingBalancePowerdata` for the engine's activation-volume series). CC BY 4.0. Source: Energinet (www.energidataservice.dk).
- **eSett Open Data** (EXP14 imbalance prices, EXP13 imbalance volumes; FI, NO1–NO5, SE1–SE4). Public under eSett's terms of use, which grant no explicit redistribution license.

Details and required attribution: `THIRD_PARTY_LICENSES.md`.

## Limitations
- Three Nordic markets, one allocation size and a deliberately simple trading sleeve; the real-market payoff is stylized, so the claims rest on the placebo market.
- The learned gambling appears in low-cost simulators with a zero floor; the evaluation hazard appears wherever a floor binds.
- The threshold policy was reliable for PPO only; full-liability agents rarely learned to stay flat.
- Audits were coded by a single coder from static code; the literature survey is a pilot.
- Pre-registration was internal (see above).

## License
- **Code** (`*.py`): MIT (`LICENSE`).
- **Documentation, results and derived tables:** CC BY 4.0 (`LICENSE-docs.md`).
- **Third-party material** keeps its own terms (`THIRD_PARTY_LICENSES.md`).

## Citation
See `CITATION.cff`. A manuscript reference will be added when available.

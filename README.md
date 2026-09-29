# The simulator decides the verdict

**A pre-registered audit of evaluation shortcuts for AI trading agents in electricity balancing markets** (working title)

## The question
A multi-agent RL fund trades Danish imbalance-settlement exposure (DK1 / DK2, 2025 held-out data). Over one year of development, the verdict "does the learned agent beat the rules?" flipped several times as the simulator changed. This repository asks *which evaluation shortcuts decide that verdict*. It answers with controlled, pre-registered experiments on a validated engine, and with a reconciliation check that detects the failure.

## Main results

**1. The audit engine is validated.** Under the strict protocol it reproduces the frozen final campaign to within 1e-8 percentage points (MARL, feasible-action MAPPO and a deterministic anchor, both regions).

**2. No liquidity cap (price-taker).**
- The anchor's return grows ×307 / ×997 (+154% / +480%). Feasible-action MAPPO averages +140% / +426%.
- Plain MAPPO (MARL) instead loses 95–100% in all 20 runs.
- The same shortcut inflates some policies by orders of magnitude and ruins others.

**3. Silent loss forgiveness (no liquidity cap + no solvency).**
- The engine floors trading cash at zero.
- With margin and loss exit switched off, MARL's booked P&L loses **7–33× its capital** (−737% to −3,338%).
- Yet the engine reports a *positive* return in 13 of 20 runs, with a fund return up to +12.5%, above every agent's strict-protocol 8.1–8.5%.

**4. A thesis-like protocol reproduces "MARL wins" from checkpoints that come last under strict evaluation (10 seeds).**

| | DK1 | DK2 |
|---|---|---|
| MARL reported, **mean** | **+270%** | **+122%** |
| MARL reported, IQM | +73% | −6% |
| MARL reported, median | −101% | −101% |
| MARL fund return, mean | **+40.5%** | **+23.0%** |
| MARL booked P&L (ledger), mean | **−535%** | **−431%** |
| Anchor reported | −36% | −41% |

The mean and the IQM of seeds are fooled; only the median is not.

**5. Leave-one-out decomposition (56 runs).**
- Two components are **necessary**: no liquidity cap *and* no solvency.
- Removing either one eliminates the false positive and closes the reported-vs-ledger gap (to within 0.4 pp).
- The percent-of-price payoff is *not* necessary. The pre-registered prediction that it was (R1) failed.

**6. Ledger reconciliation detects the failure.** Reported and booked-P&L returns agree within 1.3 pp in all 174 runs under sound accounting (138 re-runs plus 36 frozen strict baselines), and diverge by thousands of pp exactly where losses are forgiven (`tools/ledger_check.py`).

**7. Prevalence in public code (21 electricity-RL repositories, pre-registered, static reading).**
- 20 of 21 have no capital account; 0 of 21 model solvency.
- **0 of 21** have a zero floor or a percent-of-price payoff.
- 11 of 12 single-agent environments on historical prices are price-takers.
- **Findings 3–5 are therefore hazards of finance-style simulators (capital account + percent returns); they are not a documented pattern in public electricity-RL code.**

## Pre-registration scorecard
**Internal pre-registration.** Every prediction was written into timestamped files before the corresponding runs (`docs/PREREGISTRATION.md`, `docs/CODE_AUDIT_PROTOCOL.md`). The files were not deposited with an independent registry; their header times were later corrected to match file-modification times, as noted in the files. Every outcome is reported: 10 of 18 predictions were supported (one trivially) and 8 were not.

| Set | Supported | Not supported |
|---|---|---|
| Main campaign P1–P5 | P2, P5 (see the wording note below) | P1, P3, P4 |
| Cross-market X1–X4 | X2 (negligible size) | X1, X3, X4 (the spike-thin-liquidity mechanism is refuted) |
| 10-seed extension Q1–Q4 | Q2, Q3, Q4 | Q1 |
| Leave-one-out R1–R5 | R2, R3, R4; R5 technically met but trivial (<0.5 pp) | R1 |
| Code-audit expectations | solvency ≤ 20% | price-taker ≥ 70%, zero floor ≥ 1, percent payoff ≥ 1 |

*Wording note.* P2 and P5 are worded on the *reported* return. The first aggregator version tested them on the reconstructed ledger; it was corrected after results were seen. Both readings are printed in `results/campaign/PREDICTION_VERDICTS.md`.

## Repository layout
```
README.md, DESCRIPTION.md        overview; short description, abstract and repository metadata
CITATION.cff                     citation metadata
CHECKLIST.md                     evaluation checklist for AI trading agents
LICENSE, LICENSE-docs.md         MIT (code); CC BY 4.0 (docs, results, derived tables)
THIRD_PARTY_LICENSES.md          sources, licenses and attribution for all third-party material
requirements.txt                 standalone analyses; requirements-engine.txt adds the engine stack
docs/METHODS_AND_RESULTS.md      full study design, every protocol, prediction and result
docs/                            also: pre-registration, code-audit protocol and results, campaign guide
tools/ledger_check.py            standalone reported-vs-booked-P&L reconciliation (no engine needed)
tools/verify_release.py          recomputes every headline number from the shipped results
engine/                          multi-agent simulator source (the patched audit copy that produced the results)
analysis/                        standalone cross-market liquidity test (runs on data/ only)
audit/                           engine-dependent audit code (verbatim record of what ran) + engine patch
code_audit/                      search, scan and coding of 21 public repositories (own coding, pinned SHAs)
literature/                      pilot literature survey coding (no PDFs)
results/                         campaign, variant, decision-replay, cross-market and ledger-example results
data/                            Energinet data (CC BY 4.0), engine liquidity series, derived tables;
                                 fetch_esett.py downloads the eSett files (not redistributed)
```

## Reproducing
Python 3.10 (`pip install -r requirements.txt`).

**Without the engine (minutes):**
```
python tools/verify_release.py                                                      # recomputes all headline numbers
python tools/ledger_check.py results/ledger_examples/L3_marl_seed7_DK1.csv.gz      # reported +21%, ledger -3,332%: flagged
python tools/ledger_check.py results/ledger_examples/L0_strict_marl_seed7_DK1.csv.gz  # reconciled
python data/fetch_esett.py                                                          # downloads eSett data, verifies hashes
python analysis/cross_market_liquidity_test.py                                      # regenerates results/cross_market/
python code_audit/search_repos.py && python code_audit/scan_repos.py                # network; re-fetches public code
```

**With the engine (hours).** The engine source is in `engine/`. Its datasets and trained checkpoints (about 57 GB) are not included. `pip install -r requirements-engine.txt`.
- `audit/setup_audit_engine.py` copies the engine and applies the three-block patch in `audit/engine_patch/`.
- The patch has two effects:
  - it bypasses the train/eval contract-hash check, only when `SIM_AUDIT_ALLOW_PROTOCOL_MISMATCH=1`;
  - it applies config overrides from `SIM_AUDIT_CFG_OVERRIDES`.
- `audit/run_scarcity_tail_campaign.py` runs the phases: validate (gate) → evals → retrain → l7_ext → l7_loo → aggregate.

See `docs/RUNNING_THE_CAMPAIGN.md`.

## Not included
- Engine datasets and trained checkpoints (~57 GB) and raw per-run logs (~105 GB). Aggregated results are in `results/`.
- The 22 paper PDFs and their extracted text (copyright).
- Verbatim third-party code excerpts from the code audit (copyright). `code_audit/scan_repos.py` regenerates them from the pinned commits.
- Raw eSett Open Data files: public, but with no explicit redistribution license. `data/fetch_esett.py` re-downloads them.

## Data sources
- **Energinet, Energi Data Service** (`ImbalancePrice`; `RegulatingBalancePowerdata` for the engine's activation-volume series). CC BY 4.0. Source: Energinet (www.energidataservice.dk). Subsetted and, for the liquidity series, aggregated by the authors.
- **eSett Open Data** (EXP13 imbalance volumes, EXP14 imbalance prices; FI and NO1–NO5). Public under eSett's terms of use, which grant no explicit redistribution license, so the files are fetched rather than shipped. `data/fetch_esett.py` verifies each download against hashes recorded in the study.

Details and required attribution: `THIRD_PARTY_LICENSES.md`.

## Limitations
- One engine and one agent family.
- The false positive reproduced is the authors' own earlier result.
- The L7 headline rests on 3 long-saturated seeds out of 10.
- Code-audit coding was done by a single coder from static reading.
- The literature survey is a 21-paper pilot, not a systematic review.
- Pre-registration was internal (see above).

## License
- **Code** (`*.py`): MIT (`LICENSE`).
- **Documentation, results and derived tables:** CC BY 4.0 (`LICENSE-docs.md`).
- **Third-party material** keeps its own terms (`THIRD_PARTY_LICENSES.md`).

## Citation
See `CITATION.cff`. A manuscript reference will be added when available.

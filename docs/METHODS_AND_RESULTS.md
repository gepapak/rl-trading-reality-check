# Methods and results

This document describes the study design, every protocol and prediction, and all results, including failed predictions. The numbers can be recomputed from the files in `results/` with `tools/verify_release.py`.

## 1. Setting

**Market.** Danish single-price imbalance settlement after the mFRR energy-activation-market go-live (4 Mar 2025), bidding zones DK1 ("original") and DK2 ("v2"), 2025 held-out evaluation windows at 10-minute resolution.

**Engine.** `engine/` is the multi-agent renewable-fund simulator used for the final campaign (the *Prototype5* engine). It models:
- same-delivery causal settlement;
- execution capped at 25% of observed balancing-activation volume;
- market fees, impact and collateral funding;
- trading-sleeve margin and a loss exit.

The audit applies a three-block patch (`audit/engine_patch/`). The patch touches only evaluation and has two effects:
- it allows train/evaluation protocol mismatches when `SIM_AUDIT_ALLOW_PROTOCOL_MISMATCH=1`;
- it applies config overrides from `SIM_AUDIT_CFG_OVERRIDES`.

**Agents** (frozen final checkpoints):
- plain MAPPO (**MARL**), 10 seeds;
- **feasible-action MAPPO** (MAPPO restricted to a feasible action set), 10 seeds;
- a deterministic forecast **anchor**, seed 7; only its battery agent is learned.

**Validation gate.** Before any shortcut run, the audit engine re-evaluated seed 7 of every agent in both regions under the strict protocol. It reproduced the frozen sleeve returns to 8 decimals, within the pre-set tolerance of 1e-6 pp:
- MARL −0.24948388 / −0.23127154;
- feasible-action 0.50540156 / 0.47575644;
- anchor 0.50273786 / 0.48175369.

Strict baselines for seeds that were not re-run are therefore taken from the frozen campaign.

## 2. Protocols

Each protocol changes the strict protocol (L0) in the ways listed.

| Protocol | Changes |
|---|---|
| L0 strict | none |
| L1 price-taker | liquidity participation cap off (all desired volume fills) |
| L2 no solvency | trading-sleeve margin off, loss exit off |
| L3 | L1 + L2 |
| L4 load-proxy liquidity | liquidity from a load proxy, impact referenced to the sleeve (earlier protocol) |
| L5 percent payoff | horizon payoff divided by a reference price |
| L6 interpolated prices | hourly prices linearly interpolated to 10 minutes (look-ahead) |
| L7 thesis-like | L6 + percent-capped MTM payoff + L1 + L2 + legacy flat fee + 10% cash distribution ("sweeper") |
| L7 − *c* | L7 without component *c* ∈ {interpolation, percent MTM, price-taker, no solvency, legacy fee, sweeper} |
| L8 | percent MTM + no solvency only |

Coverage:
- **L1–L3:** MARL and feasible-action, 10 seeds each, plus the anchor.
- **L4–L7:** MARL and feasible-action, 3 seeds each (7, 42, 123), plus the anchor. L7 for MARL was later extended to all 10 seeds.
- **Leave-one-out (L7 − *c*) and L8:** MARL, 3 seeds, plus the anchor.
- **Not applicable:** L5 × feasible-action. The engine refuses the combination, because feasible-action MAPPO requires the MWh payoff.
- **Total:** 264 completed evaluations.

**Classical strategies** (Markowitz, capped-long, rule-based heuristic) were also run, through the engine's baseline code, under single shortcuts V00–V12 (`results/controlled_variants/`).

## 3. Measurement

- **Reported return:** the engine's `sleeve_trading_return_pct`. This is distribution-adjusted, and it is what a paper would print.
- **Ledger return:** rebuilt from the per-step booked trading P&L (`mtm_pnl`) minus all booked costs (transaction and market-access fees, impact, collateral funding), starting from the initial sleeve equity. `tools/ledger_check.py` implements the same comparison for any per-step log.
- **Ruin:** the ledger equity path reaches ≤ 5% of initial capital.
- **Sharpe:** investor-only daily Sharpe (annualised by √365) vs the engine's fund-level Sharpe (annualised from 10-minute steps).
- **Seed aggregates:** mean, interquartile mean (IQM) and median.

## 4. Pre-registration record

Predictions were written into timestamped files *before* the corresponding runs:
- `docs/PREREGISTRATION.md`: P1–P5; addendum X1–X4; addendum 2 (Q1–Q4); addendum 3 (R1–R5);
- `docs/CODE_AUDIT_PROTOCOL.md`: code-audit rules and expectations.

**This is internal pre-registration.** The files were not deposited with an independent registry, so their timing rests on file records.

**Disclosures:**
1. **Timestamps.** The header times of addenda 2–3 and of the code-audit protocol were corrected afterwards to match file-modification and launch times. A note in each file records this; no prediction text was changed.
2. **Wording of P2 and P5.** Both are worded on the *reported* return. The first aggregator version evaluated them on the ledger; it was corrected after results were seen, and both readings are reported.
3. **Frozen L0 baselines.** L0 values for seeds that were not re-run come from the frozen campaign (see the validation gate above).
4. **Cross-market test.** The SE zones were excluded because no public quarter-hour volume data exists. This was recorded before any zone metric was computed.
5. **Code-audit search.** The search was broadened (Deviation 1) before any candidate was screened.

## 5. Results

### 5.1 Context: the verdict changed across protocol generations (uncontrolled)
Over the development year, the question "does MARL or a forecast-driven method beat non-learning strategies?" was answered under successive protocols:

| Protocol generation | MARL outcome |
|---|---|
| Interpolated prices + cash sweeper (thesis) | ranked 1st, +22% fund |
| Forward-filled prices | a Markowitz baseline ranked 1st |
| Percent-return sleeve | +818% (a forecast method reached +2,920%) |
| Real settlement prices with percent payoff | ruined |
| Final strict protocol | −0.08% (anchor +0.50%) |

Code, agents and data changed together, so this record motivates the controlled experiments but attributes nothing by itself. Its sources are not part of this repository.

### 5.2 Classical strategies under single shortcuts (V00–V12)
V00 reproduces the frozen baselines exactly.

**Shortcuts with an effect:**
- **Percent payoff** (V01) and **percent-capped MTM** (V11) ruin all active strategies (−95% to −100%).
- **No liquidity cap** (V02) also ruins them (≈ −95%).
- **Load-proxy liquidity** (V03) gives Markowitz / capped-long ×105 in DK1 (+30%) but −3.7% in DK2.
- **Interpolation** (V08) changes the top-ranked strategy.
- **The thesis-like combination** (V12) is explosive: capped-long +4.2×10⁸ %.

**Shortcuts with no effect** on these strategies: collateral / margin off, impact / spread off, legacy settlement clip, cash sweeper, continuous MTM.

### 5.3 Decision replay (identical decisions, one shortcut)
Logged decisions were re-priced against the real settlement price. The replay reproduces booked settlement P&L within 0.1% for almost all runs; the worst case is 1.5%, on a near-zero-profit seed.

**Letting all requested volume fill:**
- the anchor goes from +0.50% / +0.48% to +115% / +210%;
- MARL (10-seed mean) goes from −0.08% / −0.09% to +29% / +82%.

**With path solvency:** MARL's mean worst drawdown is 124% / 74% and 35% of MARL runs are ruined, while the anchor's drawdown stays at 7–8%.

The anchor fills only 5.6–7.5% of its desired volume in top-decile spread states, against 15–36% near zero spread. §5.9 shows this reflects the anchor *requesting* more volume in spike states, not thinner markets.

### 5.4 Learned agents in the engine, one shortcut at a time
Ledger return (%), DK1 / DK2. Seed means; ranges in brackets. For these protocols the ledger and the reported return agree within 1.3 pp in all 174 runs (138 re-runs plus 36 frozen strict baselines).

| Protocol | Anchor | Feasible-action | MARL |
|---|---|---|---|
| L0 strict | +0.50 / +0.48 | +0.50 / +0.48 | −0.08 / −0.09 |
| L1 price-taker | **+154 / +480** | **+140 / +426** (68–223 / 172–712) | **−96.8 / −97.0**, ruined in 20 / 20 |
| L2 no solvency | = L0 | = L0 | = L0 |
| L4 load proxy | +63 / +99 | +62 / +97 | −30 / +1.9 |
| L5 percent payoff | −0.02 / −0.02 | not applicable | −95.9 / −95.4 |
| L6 interpolation | +0.50 / +0.18 | +0.50 / +0.18 | −0.18 / −0.19 |

The same shortcut (L1) inflates the anchor ×307 / ×997 and ruins MARL.

### 5.5 Silent loss forgiveness (L3 = L1 + L2)
- The engine floors trading cash at zero at every booking site (`engine/environment.py`, e.g. line 2725; `engine/financial_engine.py:233`; `engine/baselines/baseline_common.py`).
- Under the strict protocol the margin and loss exit stop the sleeve before the floor binds.
- With both switched off:
  - MARL's ledger loses **737–3,338%** of sleeve capital;
  - the engine reports a **positive** sleeve return in **13 of 20** runs (up to +34%);
  - the fund return reaches up to +12.5%, above every agent's strict-protocol fund return (8.1–8.5%).
- The ruin flag is set in all 20 runs but never enters the headline return, the NAV or the Sharpe.

### 5.6 The thesis-like protocol reproduces "MARL wins" (L7, 10 MARL seeds)

| | DK1 | DK2 |
|---|---|---|
| MARL reported, mean | **+270%** | **+122%** |
| MARL reported, IQM | +73% | −6% |
| MARL reported, median | −101% | −101% |
| MARL fund return, mean | **+40.5%** | **+23.0%** |
| MARL ledger, mean / median | −535% / −1,331% | −431% / −945% |
| Feasible-action reported (3 seeds) | −35% | −41% |
| Anchor reported | −36% | −41% |

- Under the reported metric the strict ranking (anchor > feasible-action > MARL) fully inverts to MARL > feasible-action > anchor in both regions.
- Under L7 the payoff is a capped percent return on the spot-price level, which rises from 16 to 691 DKK/MWh across the window.
- 9 of 10 MARL checkpoints hold one direction on ≥ 96% of steps (seed 5001 is mixed):
  - the 3 long-saturated seeds gain +469% to +1,277%, with a directional hit rate below 0.5;
  - every short-dominant seed loses 428–1,417% in the ledger, but reports only −57% to −114% because of the zero floor.
- The mean and the IQM are therefore fooled; the median is not.

### 5.7 Which shortcuts are necessary (leave-one-out, 56 runs)

| Variant | MARL reported (ledger) | Anchor | Inversion | max \|reported − ledger\| |
|---|---|---|---|---|
| L7 | +278% (−440%) | −39% | yes | 1,287 pp |
| − percent MTM | +438% (−2,980%) | +101% | yes | 4,050 pp |
| − no solvency | −95% (−95%) | −16% | **no** | 0.4 pp |
| − price-taker | −0.2% (−0.0%) | −0.2% | **no** | 0.3 pp |
| − interpolation | +533% (−2,818%) | −83% | yes | 6,588 pp |
| − legacy fee | = L7 | = L7 | yes | – |
| − sweeper | +418% (−300%) | −39% | yes | 1,287 pp |
| L8 (percent MTM + no solvency) | −0.3% | −0.5% / −0.3% | tie (<0.5 pp) | 0.3 pp |

- **Two components are necessary: no liquidity cap *and* no solvency.** Removing either one eliminates the inversion and closes the reported-vs-ledger gap.
- **The two together are not sufficient.** In L3 the price-taker anchor earns +317% and stays first.
- **The third ingredient is interchangeable:** any two of {percent MTM, interpolation, sweeper} complete the inversion.
- **The percent-of-price payoff is not necessary.**

### 5.8 Retraining under the shortcut and metric inflation
- **Retraining.** MARL retrained for 20 episodes under L1 (seeds 7, 42, 123) was still ruined under L1: −96.0%, vs −97.5% for strict-trained MARL on the same seeds. Under L0 it earned +0.004%, vs −0.20%. Learning neither exploited the shortcut nor avoided ruin.
- **Sharpe.** Under the strict protocol, every run's fund-level 10-minute Sharpe (171–177) exceeds its investor-only daily Sharpe (−2.6 to 3.8) by more than 10×. Across all shortcut runs the share is 36% (main campaign), because shortcut-dominated sleeves drive both metrics.

### 5.9 Cross-market liquidity test (X1–X4)
- **Design.** Eight zones (DK1, DK2, FI, NO1–NO5). An executable-volume proxy of 25% of |balancing volume| is compared with a desired size of 10× its median.
  - DK uses Energinet |SatisfiedDemand|; FI and NO use eSett EXP13.
- **Price-taker inflation I:** 4.8–7.1 across zones.
- **Value-vs-volume fill ratio R:** 1.13–1.64, i.e. **> 1 everywhere**. Balancing volume is *higher* when spreads are large.
- **Verdict.** The hypothesised "spikes coincide with thin liquidity" mechanism is **refuted**. The inflation comes from capital-sized desired exposure exceeding balancing volumes at all times, not from thin markets at spikes.
- **Proxy validation.** The Danish volume proxy correlates only weakly with the engine's activation series (Spearman 0.23 / 0.10).

### 5.10 Prevalence

**Public code** (`docs/CODE_AUDIT_RESULTS.md`). 337 GitHub repositories were searched and 21 electricity-RL environments included, each coded by static reading at a pinned commit:
- 20 of 21 have no capital account; 0 of 21 model solvency;
- 0 of 21 use a zero floor on money or a percent-of-price payoff;
- 11 of 21 are price-takers (11 of 12 single-agent environments on historical prices, 0 of 9 market-clearing simulators);
- 3 of 21 show look-ahead in static code, and 15 of 21 could not be judged statically.

**Published papers** (pilot survey, 21 papers, `literature/literature_survey.csv`):
- 14 (67%) are explicitly or implicitly price-takers (16, or 76%, counting 2 "not modelled" cases);
- none enforces path-dependent solvency; all sum P&L, 16 of them with asset-backed positions.

**Consequence.** The artifacts behind §5.5–5.7 belong to *finance-style* simulators (capital account + percent returns) applied to energy. They are not a documented pattern in public electricity-RL code, whose environments settle volume × price per step and have no notion of capital.

## 6. Prediction outcomes

| ID | Prediction (short) | Outcome |
|---|---|---|
| P1 | Anchor ≥ 50× under L1 **and** MARL higher under L1 | Not supported (anchor ×307 / ×997; MARL ruined) |
| P2 | ≥ 20% of MARL L1 runs ruined or DD ≥ 50%, and L3 reported mean > L1 | Supported on the reported wording (1.00; −0.2% vs −97.1%); the ledger reading does not support it |
| P3 | Retrained > strict-trained under L1, and ≤ under L0 | Not supported |
| P4 | ≥ 80% of runs with Sharpe inflation > 10×, plus a rank flip | Not supported (36%; 1 flip) |
| P5 | Ranking flip under some shortcut | Supported (L7, both regions, reported return) |
| X1 | Mean R lower in thin-market group | Not supported |
| X2 | Mean I higher in thin-market group | Supported (6.40 vs 6.25, negligible) |
| X3 | DK: I > 1 and R < 1 | Not supported (R > 1) |
| X4 | ρ(tail index, I) < 0 | Not supported (+0.76) |
| Q1 | Each new L7 seed holds one direction ≥ 90% | Not supported (seed 5001: 74% / 87%) |
| Q2 | Ledger sign = dominant direction | Supported (14 / 14) |
| Q3 | Floor caps reported losses (≥ −120%, fund ≥ −10%) | Supported (14 runs; min −114%, fund −5.6%) |
| Q4 | 10-seed headline holds | Supported (robustness check, largely implied by Q3) |
| R1 | Percent MTM necessary | **Not supported** |
| R2 | Floor hides the ledger (gap ≤ 5 pp with solvency on) | Supported (0.4 pp) |
| R3 | Interpolation not necessary | Supported |
| R4 | Fee and sweeper not necessary | Supported |
| R5 | Minimal pair inverts | Technically met, trivial (<0.5 pp) |

In total, 10 of 18 were supported (one trivially) and 8 were not. Of the 4 recorded code-audit expectations, 1 was met (solvency ≤ 20%).

## 7. Limitations
- **Scope:** one engine, one agent family, one market.
- **Own result:** the false positive reproduced is the authors' own earlier result.
- **L7 sample:** the L7 headline rests on 3 long-saturated seeds out of 10.
- **Retraining:** only 20 episodes, 3 seeds.
- **Code audit:** a single coder, static reading, 21 repositories.
- **Literature survey:** a pilot, not a systematic review.
- **Pre-registration:** internal (see §4).

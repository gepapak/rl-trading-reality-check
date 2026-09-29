# Pre-registration: learned-agent audit and retraining under shortcuts ("one tail, four illusions")

Written 28 Sep 2026, before any learned-agent evaluation under a shortcut protocol and before any retraining.
Results produced by `run_scarcity_tail_campaign.py` will be compared against these statements verbatim, and all outcomes will be reported.

## Agents and data
- MARL: plain MAPPO, Paper1/Prototype5 final checkpoints, seeds {7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202}.
- Feasible-action MAPPO: the same ten seeds.
- Deterministic anchor: seed 7 (only its battery agent is learned).
- Held-out 2025 data: DK1 ("original") and DK2 ("v2").
- Investor-only P&L is computed from the per-step logs (booked settlement P&L minus fees, impact and collateral funding), with the path check on the sleeve equity.

## Protocols (one shortcut each; everything else = strict final protocol)
- L0 strict.
- L1 price-taker (liquidity cap disabled).
- L2 no solvency (maintenance margin off, loss exit off).
- L3 = L1 + L2.
- L4 load-proxy liquidity.
- L5 percent payoff (reference-price denominator).
- L6 interpolated hourly prices (look-ahead).
- L7 thesis-like combination (L6 + percent-capped MTM + L1 + L2 + legacy fee + 10% cash sweeper).

## Validation gate
L0 must reproduce the frozen campaign's sleeve trading return within 1e-6 percentage points for every validated run. If it does not, the campaign stops and no shortcut result is used.

## Predictions

**P1 — liquidity illusion.** Under L1, the anchor's investor-only return is at least 50× its L0 value in both regions. The MARL seed-mean investor return is higher under L1 than under L0 in both regions.

**P2 — solvency illusion.**
- Under L1, at least 20% of MARL seed-region runs are ruined or have a sleeve drawdown ≥ 50%.
- Under L3, the MARL seed-mean reported return exceeds its L1 value in at least one region.

**P3 — learning exploits the illusion (retraining).** MARL retrained under L1 (seeds 7, 42, 123):
- (a) when evaluated under L1, it reports a higher mean investor return than strict-trained MARL evaluated under L1;
- (b) when evaluated under L0 (reality), it earns no more than strict-trained MARL under L0, i.e. the exploited "skill" does not transfer.

**P4 — metric illusion.** In at least 80% of runs, the fund-level 10-minute-annualised Sharpe exceeds the investor-only daily Sharpe by more than 10×. For at least one pair of methods, the Sharpe ranking changes when the 3 largest-|spread| settlement events are removed.

**P5 — ranking flip.** For at least one shortcut protocol, the seed-mean ranking of {anchor, feasible-action, MARL} by reported investor return differs from their L0 ranking.

## Addendum (28 Sep 2026, ~15:55): cross-market liquidity-illusion test

Written before downloading eSett imbalance volumes (EXP13) and before computing any zone result.

**Data.**
- eSett EXP14 spreads s (`imblSpotDifferencePrice`) and EXP13 net imbalance volumes, all 12 Nordic zones, 2025-03-04 → 2026-09-27.
- Executable-volume proxy per quarter: c_t = 0.25 × |net imbalance_t| (MWh), mirroring the engine's 25% participation cap on activated balancing volume.
- The proxy is validated on DK1/DK2 by correlation with the Energinet activation-volume series used in the engine. This is a diagnostic, not a prediction.

**Strategy for the mechanism test.**
- A fixed desired size q per zone = 10 × the zone's median c_t.
- Direction a_t = sign(s_t): the value at stake, i.e. an upper bound on what any strategy can extract.
- Secondary: a lag-8 (2-hour) persistence direction, reported only.

**Metrics.**
- Price-taker inflation I = Σ q|s| / Σ min(q, c)|s|.
- Value-vs-volume fill ratio R = [Σ min(q, c)|s| / Σ q|s|] / [Σ min(q, c) / Σ q]. R < 1 means spikes coincide with relatively thin executable volume.

**Groups** (as pre-registered earlier): T = {DK1, DK2, SE4, FI}, H = {NO1–NO5, SE1, SE2}.

**Predictions.**
- **X1:** mean R over T < mean R over H.
- **X2:** mean I over T > mean I over H.
- **X3:** in both DK1 and DK2, I > 1 and R < 1.
- **X4:** across all 12 zones, the Spearman correlation between the upper-tail Hill index (from `cross_market_structure.csv`) and I is negative.

**Deviation recorded before any zone metric was computed (28 Sep 2026, ~16:05).**
- eSett EXP13 publishes imbalance volumes only for FI and NO1–NO5 (checked via `EXP13/MBAOptions`); DK and SE are not covered.
- DK1/DK2 therefore use Energinet's own balancing volume, |SatisfiedDemand| from the ImbalancePrice dataset (2025-03-04 → 2026-08-17). This series is validated against the engine's activation-volume series.
- SE1–SE4 are excluded: no public quarter-hour volume source is available in Paper1.
- Groups become T = {DK1, DK2, FI} and H = {NO1–NO5}. Predictions X1–X4 are unchanged in form; X4 uses the 9 remaining zones.

## Interpretation rules
- Mechanism claims about spike states use the fill-ratio-by-|spread| analysis already reported in `CONTRIBUTION.md`.
- A failed prediction is reported as failed; no protocol, seed set or threshold is changed after results are seen.

## Addendum 2 (28 Sep 2026, ~22:22, just before the l7_ext launch at 22:22:58): L7 ten-seed extension, written before any of these runs

**Context (already seen).**
- L7 results for MARL seeds 7, 42 and 123.
- Anchor seed 7 and feasible-action seeds 7, 42 and 123 under L7.
- Strict-protocol direction shares for all 10 MARL seeds, which do *not* predict L7 direction (seed 42: 47% long under L0, 99% long under L7).

**New runs.** L7_thesis_like for MARL seeds 2025, 3007, 5001, 8102, 9005, 10001, 11202 in both regions (14 runs, `--phase l7_ext`). Nothing else changes.

**Predictions.**
- **Q1 (direction saturation):** in each new run, one direction (long or short) is held on ≥ 90% of steps.
- **Q2 (drift mechanism):** in every new run, the sign of the ledger return equals the dominant direction (long gains, short loses).
- **Q3 (zero-floor loss cap):** every L7 MARL run (all 10 seeds) whose ledger return is < −100% has a reported sleeve return ≥ −120% and a fund return ≥ −10%.
- **Q4 (headline at 10 seeds):**
  - the MARL reported seed-mean sleeve return exceeds the anchor's L7 reported return in both regions;
  - the MARL ledger seed-mean is below the anchor's L7 ledger return in both regions.
  - Disclosure: given the 3 seeds already seen, Q4 largely follows from Q3, so it is a robustness check, not an independent test.
- **Reported only (no prediction):** the share of long-saturated seeds, and the MARL reported seed mean.

## Addendum 3 (28 Sep 2026, 22:25; file time 22:25:34): leave-one-out decomposition of L7, written before any of these runs

**Components of L7.**
1. `interp` (interpolated hourly prices)
2. `percent_mtm` (`--mtm_return_model percent_capped`)
3. `price_taker` (liquidity cap off)
4. `no_solvency` (margin off, loss exit off)
5. `legacy_fee`
6. `sweeper` (10% distribution rate)

**Runs.**
- `L7_minus_<component>` for each of the 6 components (all other L7 components kept).
- One minimal pair, `L8_pctmtm_nosolv` (only `percent_mtm` + `no_solvency`).
- Each variant for MARL seeds 7, 42, 123 and anchor seed 7, both regions: 56 runs (`--phase l7_loo`).

**Definition.** "Inversion" = the MARL reported seed-mean sleeve return exceeds the anchor's reported sleeve return in the same variant and region.

**Mechanism reading behind the predictions (already seen).**
- The window starts near the price floor (16 DKK/MWh vs a median of about 650).
- An always-long position under a percent-price payoff therefore earns roughly p_end / p_start (≈ 43×, before caps).

**Predictions.**
- **R1 (payoff is necessary):** under `L7_minus_percent_mtm` there is no inversion in either region.
- **R2 (floor hides the ledger):** under `L7_minus_no_solvency`, |reported − ledger| ≤ 5 pp in every run.
- **R3 (interpolation not necessary):** under `L7_minus_interp`, the inversion persists in at least one region.
- **R4 (fee and sweeper not necessary):** under `L7_minus_legacy_fee` and under `L7_minus_sweeper`, the inversion persists in both regions.
- **R5 (minimal pair):** under `L8_pctmtm_nosolv`, the inversion appears in at least one region.
- **Reported only, no prediction:** `L7_minus_price_taker`.

_Timestamp note (28 Sep 2026, 22:48): the header times of addenda 2 and 3 were corrected to match file modification and launch times. No prediction text was changed._

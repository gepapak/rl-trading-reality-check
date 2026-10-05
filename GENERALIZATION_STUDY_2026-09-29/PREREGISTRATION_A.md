# Pre-registration, Study A: do evaluation shortcuts flip the AI-vs-rules verdict across independent environment designs?

Written 29 Sep 2026, ~10:44, before any environment code, training or evaluation. Internal pre-registration (timestamped file; no external registry). All outcomes will be reported; nothing below will be changed after results are seen. Any deviation will be appended with a timestamp.

## Motivation
The simulator audit (Paper1/SIMULATOR_AUDIT_2026-09-28) showed, in one engine, that evaluation shortcuts can invert the verdict on a learned agent. That leaves an obvious objection: "this is a bug in your own simulator." Study A therefore tests whether shortcuts change the verdict in three **independently implemented** environment designs. The designs mirror the families found in the code audit of 21 public electricity-RL environments: storage arbitrage, renewable bidding, and a finance-style capital-account sleeve.

## Data
Energinet (Energi Data Service, CC BY 4.0), DK1 and DK2, 15-minute resolution, copied into `data_energinet/`:
- `ImbalancePrice`: imbalance price, spot price, |SatisfiedDemand| as the balancing volume;
- `Forecasts_Hour`: day-ahead wind forecast (onshore + offshore);
- `ProductionConsumptionSettlement`: actual wind production.

Split, chronological, fixed:
- **train** 2025-03-04 → 2025-12-31;
- **test** 2026-01-01 → 2026-08-17.

Every agent is trained per region.

## Environments (own code; discrete action set of 5 levels for all)

**E1 Battery arbitrage (asset-backed; imbalance-price settlement).**
- Action: power ∈ {−1, −0.5, 0, 0.5, 1} × P. Energy per quarter = 0.25·P; state of charge within [0, E]; 90% round-trip efficiency, applied on discharge.
- Payoff: discharged energy × imbalance price − charged energy × imbalance price.
- Two sizes: **E1-small** (1 MW / 2 MWh) and **E1-large** (50 MW / 100 MWh).
- No capital account.

**E2 Capital-account spread sleeve (finance-style).**
- Starting capital K0 = 1,000,000 EUR. Action a ∈ {−1, −0.5, 0, 0.5, 1}.
- Desired volume q = a · Q, where Q = 20 MWh per quarter per 1 M EUR of sizing capital.
- Payoff per quarter: q · (imbalance − spot) − 0.10 EUR/MWh · |q|.

**E3 Wind bidding (asset-backed).**
- 100 MW wind farm: actual and day-ahead forecast production are the region's wind series scaled to a 100 MW peak.
- Action: shading s ∈ {−0.3, −0.15, 0, 0.15, 0.3}; schedule = clip(forecast · (1 + s), 0, 100 MW).
- Revenue: schedule · spot + (actual − schedule) · imbalance price.

## Protocols

**Strict protocol S0** ("reality") applies to every environment:
- **Causal information.** Observations contain only information available at decision time:
  - lagged imbalance prices from t−2 and earlier;
  - the current spot price, which is known day-ahead;
  - time features, and the state of charge (E1) or equity (E2);
  - the day-ahead forecast (E3);
  - E3 decisions use only imbalance prices from ≥ 1 day earlier.
- **Liquidity cap.** Executed |volume| ≤ 25% of the quarter's balancing volume, |SatisfiedDemand| · 0.25 h. This applies to battery energy (E1) and to q (E2). E3 is exempt: its deviation is physical forecast error.
- **E2 solvency.**
  - Sizing uses current equity.
  - Losses are booked in full; equity may go negative.
  - If equity ≤ 5% of K0, the sleeve is ruined: trading stops and the remaining periods are zero.

**Shortcuts** (each removes one realism component):

| Shortcut | Applies to | Change |
|---|---|---|
| S1 price-taker | E1, E2 | no liquidity cap |
| S2 no-solvency / zero floor | E2 | sizing from K0 (not equity), no ruin stop, cash floored at 0 (losses below zero forgiven), trading continues |
| S3 | E2 | S1 + S2 |
| S4 look-ahead | E1, E2, E3 | observation additionally contains the current-quarter imbalance price (E1, E2), or the current actual production and imbalance price (E3) |
| S5 percent payoff | E2 | payoff = a · equity_sizing · clip(imbalance / spot − 1, −0.5, 0.5); |spot| < 1 EUR/MWh treated as 1 |

Regimes:
- E1-small and E1-large: {S0, S1, S4}.
- E2: {S0, S1, S2, S3, S4, S5}.
- E3: {S0, S4}.

## Agents
- **RL:** PPO and DQN (Stable-Baselines3, default hyper-parameters, MLP 64×64), **5 seeds each**, 200,000 training steps on the training period (random 7-day episodes).
  - One set of agents is trained **under each regime**, as a researcher using that simulator would.
  - Each trained policy is evaluated deterministically over the whole test period:
    - under its own regime (**reported**);
    - under S0 (**reality**).
- **Rules**, with parameters tuned on the training period only; each rule is evaluated under every regime:
  - E1: a lagged-price quantile threshold rule (grid-tuned), and idle;
  - E2: persistence (sign of the spread at t−2, size 0.5), and flat;
  - E3: bid the forecast (s = 0), and the best constant shading.

## Outcomes and definitions
- **Test profit:** EUR over the test period (E1, E3). For E2, the return on K0: **reported** as the equity the environment reports (floored under S2/S3), and the **ledger** as booked P&L without the floor.
- **Verdict** for one environment × region × protocol: "RL wins" if the best RL algorithm's seed-mean test profit exceeds the best rule's test profit under that protocol. The best algorithm or rule is the one with the higher test value; the same rule applies under every protocol.
- **False positive:** the reported verdict (trained and evaluated under shortcut S) says "RL wins", while the reality verdict (the same policies and rules evaluated under S0) says "RL loses".
- **Ledger check (E2):** |reported − ledger| > 5 pp of K0 at any step flags a run.

## Predictions
- **A1 (false positive outside the thesis engine).** In E2 under S3, the reported verdict is "RL wins" in at least one region, and the reality verdict for the same policies is "RL loses" in that region.
- **A2 (detection).** The ledger check flags ≥ 90% of E2 runs in which the zero floor binds (reported equity at 0 while the ledger is negative), and 0% of S0 runs.
- **A3 (price-taker scales with size).** In E1, compare the best reported test profit (maximum over RL seed-means and rules) of the S1 regime (trained and evaluated without the cap) with that of the S0 regime. The ratio S1 / S0 is ≥ 2 for E1-large and < 1.10 for E1-small, in both regions.
- **A4 (look-ahead).** In at least 2 of the 3 designs (E1-large, E2, E3), agents trained and evaluated under S4 report "RL wins", while the reality verdict for the same policies is "RL loses", in at least one region.
- **A5 (generality).** In at least 2 of the 3 designs, at least one shortcut produces a false positive (definition above) in at least one region.

**Reported without prediction:**
- the strict verdict (RL vs rules under S0, trained under S0);
- all seed-level results, including the median and IQM;
- the size of each shortcut's inflation.

## Interpretation rules
- A null result (no false positives) is reported as evidence *against* the generality of the thesis-engine finding.
- No environment parameter, protocol, seed count, training budget or metric will be changed after any test result is seen.
- Bugs found after results are seen will be fixed. Every such fix, with its effect, will be logged below with a timestamp, and both the pre-fix and post-fix results will be reported.

## Clarification 1 (29 Sep 2026, 10:45), before any code was written
- **Rule tuning.** Rule parameters (the E1 thresholds, the E3 constant shading) are tuned on the training period *within each regime*, mirroring RL agents trained within each regime. The E2 persistence rule has no free parameter.
- **Rule inputs.** Rule families never use look-ahead features. Under S4 only the RL agents' observations contain the leaked information, as with typical baselines in published work.
- **E3 reward.** The E3 training reward is revenue relative to bidding the forecast, (schedule − forecast schedule) · (spot − imbalance). This is a baseline shift that does not change the optimal policy. Evaluation reports total revenue.
- **Smoke tests.** Development smoke tests evaluate on the training period only. The test period is touched only by the full run.

## Clarification 2 (29 Sep 2026, 10:47), before any training or evaluation run
- **Reality evaluation of S4-trained agents.** An agent trained under S4 has extra observation slots. Under S0 (reality), each leaked slot is filled with the most recent causally available value, as a deployed system would do:
  - E1, E2: the imbalance price (spread) at t−2 instead of t;
  - E3: the day-ahead forecast instead of actual production, and the imbalance price at t−96 (one day earlier) instead of t.
- **S5 and the liquidity cap.** The cap still applies under S5. The position fraction is scaled by the executable share, fill = executable q / desired q, as a volume cap would scale a notional position.
- **Worker count.** Parallel training uses 6 worker processes, because one unrelated process is occupying a core. This affects wall time only.

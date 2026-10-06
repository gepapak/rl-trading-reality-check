# Repository description

## Suggested repository name
`simulator-forgives-losses`

## GitHub "About" text (≤ 350 characters)
A zero cash floor in a trading simulator teaches RL agents to gamble for resurrection, and standard evaluation reports it as profit. Theory (the floor reports the best moment in hindsight), twelve pre-registered studies on Nordic imbalance spreads (PPO, A2C, DQN), a public-environment audit, a reconciliation check, a placebo-market test and a checklist.

## Suggested topics
`reinforcement-learning` `limited-liability` `reward-misspecification` `simulation` `electricity-markets` `energy-trading` `imbalance-settlement` `backtesting` `evaluation-methodology` `pre-registration` `reproducibility`

## Abstract
Reinforcement-learning trading agents are trained and accepted in simulators whose accounting is rarely examined. One common convention, flooring an account's cash at zero, gives an agent limited liability, and limited liability rewards risk most strongly near zero equity.

**Theory.** Eight propositions and a theorem with proofs state when simulated limited liability rewards risk: losses beyond equity must be forgiven, and exposure must not shrink with equity. Under a zero floor, an agent trading from the floor is credited, in distribution, with its best moment in hindsight: a free lookback option on its own profit.

**Origin.** In an audit of a validated electricity-market engine, a legacy protocol reported a learned controller first (+270%) while its booked P&L was −535%.

**Learning.** In a sign-randomized placebo market, where no policy can profit, floor-trained agents raised their leverage as equity fell and reported +38% to +148% while booking −15% to −57%. When learning was well posed, all ten floor-trained PPO agents learned the theory's threshold policy (11.1 leverage units of 16 above full-liability agents). A2C and DQN agents also gambled near zero equity, often at every equity level. In Finland the threshold policy reappeared; in calm southern Norway, where the theory makes the incentive negligible, the effect vanished.

**Boundaries.** With realistic trading costs, inside the public simulator gym-mtsim, and with equity-scaled positions, agents did not learn to gamble, or did so only weakly. Reported returns still exceed booked ones wherever a floor binds.

**Reporting.** All studies were pre-registered internally; 22 of 46 liability hypotheses were not supported, and all are reported.

## Key facts
| | |
|---|---|
| Markets | Danish (DK1, DK2), Finnish (FI) and southern Norwegian (NO2) imbalance spreads, March 2025 → August 2026 |
| Learners | PPO, A2C and DQN (Stable-Baselines3 2.7.0); frozen MAPPO controllers in the engine audit |
| Studies | Engine audit; environment-design study A (280 agents); liability studies B–K (594 agents) |
| Pre-registered liability hypotheses | 46 (24 supported, 22 not supported) |
| Audits | 18 public RL trading environments (plus FinRL and TensorTrade); 21 public electricity-RL environments |
| License | Code: MIT. Docs and results: CC BY 4.0. Third-party data: see `THIRD_PARTY_LICENSES.md` |

## Status
Accompanies a manuscript in preparation for submission. Trained checkpoints and raw per-run logs are not included.

# Evaluation checklist for RL trading agents and their simulators

For anyone who trains or accepts a learned trading controller on the basis of simulated performance. Each item points to the evidence in this repository that shows why it matters.

## Accounting
- [ ] **Declare the liability rule.** State how the simulator books losses beyond equity, how positions are sized (fixed allocation or fraction of equity), and what happens at bankruptcy, alongside every reported result. A zero floor or a capped final loss is a modelling decision with behavioral consequences, not a safeguard.
  - Under a zero floor, agents learned to take maximal leverage near zero equity: 11.1 leverage units of 16 above full-liability agents (`LIABILITY_STUDY_2026-09-29/results_f/`).
- [ ] **Never floor cash silently; book the debt.** Either end the episode on ruin with the full loss booked, or book any recapitalization as debt in the reward and in the reported result. Real settlement systems require collateral precisely because a distressed party may stop paying.
  - Capping only the final loss taught far less gambling than a floor that keeps trading (Studies B, F, I, J).
- [ ] **Reconcile reported equity with booked P&L** for every run, and report the largest gap (`tools/ledger_check.py`).
  - Sound accounting agrees within about 1 percentage point. In the engine audit the check flagged 53 of 54 affected runs and none of 246 others, without knowledge of the mechanism (`results/`).
- [ ] **State whether position size scales with equity, and report ruin** (the share of runs whose booked equity reaches zero) alongside returns.

## Tests
- [ ] **Run a placebo market** (`LIABILITY_STUDY_2026-09-29/placebo_market.py`). Train and evaluate the agent on the same data with the sign of the payoff randomized independently in every period. No policy can profit there in expectation, so any positive expected reported return is a false positive.
  - Floor-trained agents reported +38% to +148% in the placebo market while booking −15% to −57% (`LIABILITY_STUDY_2026-09-29/results/`).
  - For an agent trading from the floor, five draws flag it about 90% of the time (Theorem 1 and Corollary 2 of the paper). An agent evaluated from a full allocation reaches the floor in only some draws: use 20 or more draws, or several agents.
- [ ] **Probe the equity response.** Set the equity input of a trained agent to a low and a normal value and compare the risk it takes. The probe needs no access to the simulator's accounting; it is a screening signal, not proof.
- [ ] **Test sensitivity to simulator assumptions.** Accept a learned controller only if its advantage over the rule survives the removal of each shortcut, and treat an advantage that appears only in markets without a real edge with particular suspicion (`GENERALIZATION_STUDY_2026-09-29/`).

## Market realism
- [ ] **Model realistic trading costs.** With a realistic fee, liquidity cap and price impact, agents did not learn to exploit the floor (`LIABILITY_STUDY_2026-09-29/results_h/`), although reported returns still exceeded booked ones when the floor bound.
- [ ] **Cap executed volume by executable market volume**, not by capital, and report the ratio of desired to executed volume.
  - Without a cap, one rule's return grew ×300–1,000 while a learned policy was ruined (engine audit).
- [ ] **Model solvency** (margin, loss exit or a budget constraint) for any account that can lose more than its capital.
  - A missing liquidity cap and missing solvency rules were jointly necessary for the engine audit's false positive.
- [ ] **Settle on the actual payoff** of the traded product and **respect information timing**: no interpolation between future prices, no normalization statistics fitted on test data, and gate-closure times that match the market.

## Statistics and reporting
- [ ] **Report every seed, and the median alongside the mean**, together with per-seed exposure and direction. Under zero-floor accounting the mean (+270%) and the interquartile mean (+73%) were fooled; the median (−101%) was not.
- [ ] **Annualize risk metrics at the decision frequency** and separate investor-level from fund-level metrics.
- [ ] **Pre-register protocol changes and report failed predictions.**

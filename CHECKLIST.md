# Evaluation checklist for AI trading agents in electricity markets

Each item points to the evidence in this repository that shows why it matters.

## Accounting
- [ ] **Reconcile the reported return with booked P&L.**
  - Rebuild equity from per-step booked P&L minus booked costs, and compare it with what the simulator reports (`tools/ledger_check.py`).
  - Sound accounting agrees within ~1 pp. Here, divergences of 1,000–6,600 pp flagged every false positive.
- [ ] **Never floor equity silently.**
  - If cash can hit zero, either end the episode (ruin), or book the debt and recapitalise explicitly.
  - A zero floor turns losses beyond capital into a free option: reported +21% vs booked −3,332% (`results/ledger_examples/`).
- [ ] **Report ruin.** State the share of runs whose booked equity path reaches ≤ 5% of initial capital, alongside returns.

## Market realism
- [ ] **Cap positions by executable volume**, not by capital.
  - Report the ratio of desired to executed volume.
  - Without a cap, one rule's return grew ×300–1,000 while a learned policy was ruined.
- [ ] **Model solvency:** margin, loss exit, or a hard budget constraint.
  - Removing solvency is one of the two necessary components of the reproduced false positive.
- [ ] **Settle on the actual payoff** (volume × settlement price for the traded product).
  - Avoid percent-of-price payoffs on a non-storable commodity; they reward a static long position against the window's price drift.
- [ ] **Respect information timing.**
  - No interpolation between future prices; no normalisation statistics fitted on test data.
  - Gate-closure times must match the market (e.g. mFRR request published by T−15).

## Statistics and reporting
- [ ] **Report every seed, and the median alongside the mean and IQM.**
  - Under zero-floor accounting the mean (+270%) and the IQM (+73%) were fooled; the median (−101%) was not.
- [ ] **Show per-seed exposure and direction.** A seed that is 99% long is a directional bet, not a strategy.
- [ ] **Include static and rule baselines**, including always-long and always-short, under the *same* protocol.
- [ ] **Separate investor-only metrics from fund-level metrics.** Annualise at the decision frequency: 10-minute Sharpe ≈ 175 vs investor daily Sharpe ≈ 3.7.
- [ ] **Pre-register protocol changes and report failed predictions.**

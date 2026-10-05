# Study D pre-registration: liability rules in public RL trading environments (static code audit)

Internal protocol, hashed (SHA-256, ../PREREGISTRATION_HASHES.txt) before the repository search was run.

## Question
Studies B and C show that learning agents learn to gamble when two conditions hold together:
1. **loss forgiveness:** losses beyond equity are not booked in the reward;
2. **sizing not scaled to equity:** position size does not shrink with equity.

How common are these conditions in the public reinforcement-learning trading environments that researchers reuse?

## Search (fixed in advance)
The GitHub repository search API (unauthenticated, public data), sorted by stars, is queried with `language:Python`
added to each of these queries:
1. `trading environment reinforcement learning`
2. `gym trading`
3. `trading gym environment`
4. `stock trading reinforcement learning`
5. `crypto trading reinforcement learning`
6. `forex reinforcement learning`
7. `portfolio reinforcement learning environment`
8. `futures trading reinforcement learning`
9. `gymnasium trading`
10. `algorithmic trading deep reinforcement learning`

The top 30 results per query are kept, and duplicates are merged.

## Inclusion
- At least 200 stars at search time.
- The repository defines a trading environment of its own: a class with a step method that maps agent actions on
  price data to an account, portfolio or profit outcome.
- Repositories that only train agents on an environment imported from another package are coded once, under that
  package.
- LLM-agent repositories, data-only repositories and tutorial collections without an environment are excluded.

Screening uses the name, description and file tree first, then the code.

## Coding (static reading at the recorded commit, with file-and-line evidence; no third-party code is executed)
- **Account:**
  - NOACC: no capital account; the reward is position × price change;
  - ACC: an account with cash, balance, net worth or equity.
- **Borrowing:** whether the agent can short, use leverage or margin, or otherwise lose more than its equity (yes/no).
- **Loss accounting (ACC with borrowing only):**
  - FL-cont: equity may go negative and trading continues;
  - FL-term: the episode ends at a threshold and the full loss is booked;
  - LL: the loss beyond equity is capped or floored and trading stops;
  - ZF: equity is floored and trading continues;
  - NF: non-finite or clipped rewards hide the loss (e.g. an inf or NaN reward replaced by 0).
- **Unlevered accounts:** ACC without borrowing is coded UNLEV. Loss forgiveness is impossible there.
- **Reward:**
  - LIN: change in equity or profit;
  - LOG: log return;
  - RISK: a risk-adjusted ratio;
  - OTHER.
- **Sizing:**
  - ABS: an absolute quantity from the action;
  - EQ: a fraction of current equity or net worth;
  - CASH: bounded by current cash;
  - FIXED-NOTIONAL: a fixed allocation.

## Outcomes (descriptive; no hypothesis test)
- **Counts of each convention.**
- **Primary outcome:** the number of included environments that are *gambling-prone*, meaning both of:
  - loss forgiveness (LL, ZF or NF);
  - sizing not scaled to equity (ABS or FIXED-NOTIONAL).

  It is reported with a Wilson 95% interval.

## Disclosures
- **Known before registration:** during scoping, four repositories were read — gym-mtsim (balance floored at zero after
  a stop-out; absolute order volume), Gym-Trading-Env, RLTrader and TensorTrade's stopper. They are coded like every
  other repository if the search returns them.
- **Coder:** a single coder (the assistant), by static reading. Behavior that depends on runtime configuration is coded
  at the repository's defaults.

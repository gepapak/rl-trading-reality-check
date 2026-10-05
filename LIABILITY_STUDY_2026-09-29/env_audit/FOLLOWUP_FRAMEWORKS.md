# Follow-up check: FinRL and TensorTrade (not pre-registered)

The pre-registered search of Study D (`PREREGISTRATION_D.md`) used GitHub's `language:Python` filter, which excluded the
two largest RL trading frameworks. Their default stock-trading environments were read separately on 4 October 2026
(static reading of the default branch, no execution), with the coding scheme of `coding_d.csv`.

| Framework | Stars (4 Oct 2026) | Files read | Account | Borrowing / shorting | Loss accounting | Sizing | Both conditions |
|---|---|---|---|---|---|---|---|
| AI4Finance-Foundation/FinRL | about 16,600 | `finrl/meta/env_stock_trading/env_stocktrading.py` | ACC | no: purchases capped at available cash (`available_amount = cash // (price * (1 + buy_cost))`, `buy_num_shares = min(available_amount, action)`); sales only of held shares | UNLEV | ABS (actions × `hmax` shares) | no |
| tensortrade-org/tensortrade | about 7,200 | `tensortrade/env/default/actions.py`, `tensortrade/env/default/stoppers.py` | ACC | no: order size `balance * proportion`, capped at the wallet balance (`size = min(balance, size)`) | UNLEV; `MaxLossStopper` ends the episode when the loss exceeds `max_allowed_loss` | EQ (fraction of wallet balance) | no |

Neither framework meets the conditions at its defaults: both are unlevered, so losses cannot exceed equity and loss
forgiveness cannot arise.
The result is reported in Section 9 of the paper as a follow-up outside the pre-registered sample.

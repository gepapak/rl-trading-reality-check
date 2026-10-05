# Study C verdicts

- runs: 24 of 24; seeds per cell: [3]; cells: 4
- C1 RS(ZF) 1.6M > 100k in 3/4 (need >=3); Spearman(log steps, RS(ZF)) > 0 in 2/4 (need >=3) [rho by cell {'DQN-DK1': -0.9, 'DQN-DK2': 0.0, 'PPO-DK1': 0.8, 'PPO-DK2': 1.0}] -> NOT SUPPORTED
- C2 P(0.1)(ZF) 1.6M > 100k in 3/4 (need >=3) -> SUPPORTED
- C3 change in RS larger under ZF than FL in 3/4 (need >=3) -> SUPPORTED
- C4 inflation(ZF) 1.6M > 100k in 4/4 (need >=3) -> SUPPORTED
- C5 RS(ZF) > RS(FL) at 400k in 4/4 (need >=3) -> SUPPORTED

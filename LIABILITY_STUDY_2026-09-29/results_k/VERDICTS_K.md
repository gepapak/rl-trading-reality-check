# Study K verdicts

- runs: 60; per condition: {('FL', 0.95): 10, ('FL', 0.995): 10, ('ZF', 0.95): 10, ('ZF', 0.98): 10, ('ZF', 0.99): 10, ('ZF', 0.995): 10}
- K1 Spearman(gamma, T) over zero-floor runs = -0.070, one-sided permutation p = 0.6625 (need rho > 0 and p < 0.05) -> NOT SUPPORTED
- K2 mean T(0.995) - T(0.95) = 0.075 [-0.650, 1.025] (need lower bound > 0) -> NOT SUPPORTED
- K3 mean T(0.995) = 1.425 vs 1.5 x mean T(0.95) = 2.025 (need T(0.995) >= 1.5 T(0.95)) -> NOT SUPPORTED
-    exploratory: mean T by condition {('FL', 0.95): 0.85, ('FL', 0.995): 0.0, ('ZF', 0.95): 1.35, ('ZF', 0.98): 1.475, ('ZF', 0.99): 1.85, ('ZF', 0.995): 1.425}

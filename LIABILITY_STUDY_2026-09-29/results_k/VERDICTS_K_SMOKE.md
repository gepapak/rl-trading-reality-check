# Study K verdicts (SMOKE)

- runs: 6; per condition: {('FL', 0.95): 1, ('FL', 0.995): 1, ('ZF', 0.95): 1, ('ZF', 0.98): 1, ('ZF', 0.99): 1, ('ZF', 0.995): 1}
- K1 Spearman(gamma, T) over zero-floor runs = 0.000, one-sided permutation p = 1.0000 (need rho > 0 and p < 0.05) -> NOT SUPPORTED
- K2 mean T(0.995) - T(0.95) = 0.000 [0.000, 0.000] (need lower bound > 0) -> NOT SUPPORTED
- K3 mean T(0.995) = 0.000 vs 1.5 x mean T(0.95) = 0.000 (need T(0.995) >= 1.5 T(0.95)) -> NOT SUPPORTED
-    exploratory: mean T by condition {('FL', 0.95): 0.0, ('FL', 0.995): 0.0, ('ZF', 0.95): 0.0, ('ZF', 0.98): 0.0, ('ZF', 0.99): 0.0, ('ZF', 0.995): 0.0}

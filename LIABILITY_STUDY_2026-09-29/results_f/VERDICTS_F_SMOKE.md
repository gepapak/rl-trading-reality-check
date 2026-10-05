# Study F verdicts (SMOKE)

- runs: 6; per rule x obs: {'FL-EQ': 1, 'FL-FULL': 1, 'LL-EQ': 1, 'LL-FULL': 1, 'ZF-EQ': 1, 'ZF-FULL': 1}
- F1 EQ: L_ZF(0.1) = 16 in 0/1 runs (need >=8 of 10) -> NOT SUPPORTED
- F2 EQ: L_FL(0.1) <= 1 in 1/1 runs (need >=8 of 10) -> NOT SUPPORTED
- F3 EQ: mean L_ZF(0.1) - L_FL(0.1) = -0.75 [-0.75, -0.75] (need >= 8 and lower bound > 4) -> NOT SUPPORTED
- F4 EQ: L_ZF(0.1) > L_ZF(2) in 0/1 runs (need >=8 of 10) -> NOT SUPPORTED
- F5 gap EQ -0.75 vs FULL 1.50; difference -2.25 [-2.25, -2.25] (need lower bound > 0) -> NOT SUPPORTED
- F6 EQ: mean L_ZF(0.1) - L_LL(0.1) = -3.75 [-3.75, -3.75] (need lower bound > 0) -> NOT SUPPORTED

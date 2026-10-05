# Study F verdicts

- runs: 60; per rule x obs: {'FL-EQ': 10, 'FL-FULL': 10, 'LL-EQ': 10, 'LL-FULL': 10, 'ZF-EQ': 10, 'ZF-FULL': 10}
- F1 EQ: L_ZF(0.1) = 16 in 10/10 runs (need >=8 of 10) -> SUPPORTED
- F2 EQ: L_FL(0.1) <= 1 in 4/10 runs (need >=8 of 10) -> NOT SUPPORTED
- F3 EQ: mean L_ZF(0.1) - L_FL(0.1) = 11.10 [7.12, 14.25] (need >= 8 and lower bound > 4) -> SUPPORTED
- F4 EQ: L_ZF(0.1) > L_ZF(2) in 8/10 runs (need >=8 of 10) -> SUPPORTED
- F5 gap EQ 11.10 vs FULL 4.50; difference 6.60 [2.57, 10.08] (need lower bound > 0) -> SUPPORTED
- F6 EQ: mean L_ZF(0.1) - L_LL(0.1) = 10.43 [5.85, 14.55] (need lower bound > 0) -> SUPPORTED

# Study H verdicts

- runs: 40; per rule x obs: {('FL', 'EQ'): 10, ('FL', 'FULL'): 10, ('ZF', 'EQ'): 10, ('ZF', 'FULL'): 10}
- H1 EQ: mean L_ZF(0.02) - L_FL(0.02) = -0.02 [-0.12, 0.08] (need >= 8, lower bound > 4) -> NOT SUPPORTED
- H2 EQ: L_FL(0.02) <= 1 and L_FL(1) <= 1 in 10/10 runs (need >= 8 of 10) -> SUPPORTED
- H3 EQ: L_ZF(0.02) > L_ZF(1) in 0/10 runs (need >= 8 of 10) -> NOT SUPPORTED
- H4 FULL: mean L_ZF(0.02) - L_FL(0.02) = 0.18 [0.08, 0.31] (need lower bound > 0) -> SUPPORTED
- H5 EQ: ZF reported - ledger = 0.0 pp [0.0, 0.0] (need lower bound > 0) -> NOT SUPPORTED

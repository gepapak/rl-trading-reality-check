# Study H verdicts (SMOKE)

- runs: 4; per rule x obs: {('FL', 'EQ'): 1, ('FL', 'FULL'): 1, ('ZF', 'EQ'): 1, ('ZF', 'FULL'): 1}
- H1 EQ: mean L_ZF(0.02) - L_FL(0.02) = 0.00 [0.00, 0.00] (need >= 8, lower bound > 4) -> NOT SUPPORTED
- H2 EQ: L_FL(0.02) <= 1 and L_FL(1) <= 1 in 1/1 runs (need >= 8 of 10) -> NOT SUPPORTED
- H3 EQ: L_ZF(0.02) > L_ZF(1) in 0/1 runs (need >= 8 of 10) -> NOT SUPPORTED
- H4 FULL: mean L_ZF(0.02) - L_FL(0.02) = 0.34 [0.34, 0.34] (need lower bound > 0) -> SUPPORTED
- H5 EQ: ZF reported - ledger = 0.0 pp [0.0, 0.0] (need lower bound > 0) -> NOT SUPPORTED

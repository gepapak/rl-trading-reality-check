# Study J verdicts (SMOKE)

- runs: 6; per zone x rule: {'FI-FL': 1, 'FI-LL': 1, 'FI-ZF': 1, 'NO2-FL': 1, 'NO2-LL': 1, 'NO2-ZF': 1}
- J1 FI: mean L_ZF(0.1) - L_FL(0.1) = 0.00 [0.00, 0.00] (need >= 8 and lower bound > 4) -> NOT SUPPORTED
- J2 FI: L_ZF(0.1) > L_ZF(2) in 0/1 runs (need >= 8 of 10) -> NOT SUPPORTED
- J3 FI: mean L_ZF(0.1) - L_LL(0.1) = 0.00 [0.00, 0.00] (need lower bound > 0) -> NOT SUPPORTED
- J4 gap at e=0.25: FI 0.00 vs NO2 0.00; difference 0.00 [0.00, 0.00] (need lower bound > 0) -> NOT SUPPORTED
- J5 NO2: mean L_ZF(0.02) - L_FL(0.02) = 0.00 [0.00, 0.00] (need lower bound > 0) -> NOT SUPPORTED

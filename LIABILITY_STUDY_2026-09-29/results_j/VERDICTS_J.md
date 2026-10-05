# Study J verdicts

- runs: 60; per zone x rule: {'FI-FL': 10, 'FI-LL': 10, 'FI-ZF': 10, 'NO2-FL': 10, 'NO2-LL': 10, 'NO2-ZF': 10}
- J1 FI: mean L_ZF(0.1) - L_FL(0.1) = 8.18 [3.90, 12.38] (need >= 8 and lower bound > 4) -> NOT SUPPORTED
- J2 FI: L_ZF(0.1) > L_ZF(2) in 9/10 runs (need >= 8 of 10) -> SUPPORTED
- J3 FI: mean L_ZF(0.1) - L_LL(0.1) = 8.97 [4.38, 13.60] (need lower bound > 0) -> SUPPORTED
- J4 gap at e=0.25: FI 8.47 vs NO2 0.62; difference 7.85 [0.25, 15.27] (need lower bound > 0) -> SUPPORTED
- J5 NO2: mean L_ZF(0.02) - L_FL(0.02) = 0.90 [-5.42, 7.17] (need lower bound > 0) -> NOT SUPPORTED

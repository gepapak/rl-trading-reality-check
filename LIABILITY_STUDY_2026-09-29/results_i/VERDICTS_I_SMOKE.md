# Study I verdicts (SMOKE)

- A2C-n256 runs per rule: {'FL': 1, 'LL': 1, 'ZF': 1}
- I1 A2C-n256: mean L_ZF(0.1) - L_FL(0.1) = 12.00 [12.00, 12.00] (need >= 8 and lower bound > 4) -> SUPPORTED
- I2 A2C-n256: L_ZF(0.1) > L_ZF(2) in 0/1 runs (need >= 8 of 10) -> NOT SUPPORTED
- I3 A2C-n256: mean L_ZF(0.1) - L_LL(0.1) = 12.00 [12.00, 12.00] (need lower bound > 0) -> SUPPORTED
-    exploratory A2C-n256: L_ZF(0.1)=16 in 1/1; L_FL(0.1)<=1 in 0/1; Cohen d (ZF vs FL at 0.1) = nan
- A2C-n64 runs per rule: {'FL': 1, 'LL': 1, 'ZF': 1}
- I1 A2C-n64: mean L_ZF(0.1) - L_FL(0.1) = -4.00 [-4.00, -4.00] (need >= 8 and lower bound > 4) -> NOT SUPPORTED
- I2 A2C-n64: L_ZF(0.1) > L_ZF(2) in 0/1 runs (need >= 8 of 10) -> NOT SUPPORTED
- I3 A2C-n64: mean L_ZF(0.1) - L_LL(0.1) = -4.00 [-4.00, -4.00] (need lower bound > 0) -> NOT SUPPORTED
-    exploratory A2C-n64: L_ZF(0.1)=16 in 0/1; L_FL(0.1)<=1 in 0/1; Cohen d (ZF vs FL at 0.1) = nan
- DQN-default runs per rule: {'FL': 1, 'LL': 1, 'ZF': 1}
- I1 DQN-default: mean L_ZF(0.1) - L_FL(0.1) = 0.00 [0.00, 0.00] (need >= 8 and lower bound > 4) -> NOT SUPPORTED
- I2 DQN-default: L_ZF(0.1) > L_ZF(2) in 1/1 runs (need >= 8 of 10) -> NOT SUPPORTED
- I3 DQN-default: mean L_ZF(0.1) - L_LL(0.1) = 0.00 [0.00, 0.00] (need lower bound > 0) -> NOT SUPPORTED
-    exploratory DQN-default: L_ZF(0.1)=16 in 1/1; L_FL(0.1)<=1 in 0/1; Cohen d (ZF vs FL at 0.1) = nan

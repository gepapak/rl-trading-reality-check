# Study I verdicts

- A2C-n256 runs per rule: {'FL': 10, 'LL': 10, 'ZF': 10}
- I1 A2C-n256: mean L_ZF(0.1) - L_FL(0.1) = 8.53 [2.75, 13.45] (need >= 8 and lower bound > 4) -> NOT SUPPORTED
- I2 A2C-n256: L_ZF(0.1) > L_ZF(2) in 5/10 runs (need >= 8 of 10) -> NOT SUPPORTED
- I3 A2C-n256: mean L_ZF(0.1) - L_LL(0.1) = 4.35 [-1.72, 10.28] (need lower bound > 0) -> NOT SUPPORTED
-    exploratory A2C-n256: L_ZF(0.1)=16 in 8/10; L_FL(0.1)<=1 in 5/10; Cohen d (ZF vs FL at 0.1) = 1.35
- A2C-n64 runs per rule: {'FL': 10, 'LL': 10, 'ZF': 10}
- I1 A2C-n64: mean L_ZF(0.1) - L_FL(0.1) = 11.95 [8.08, 14.97] (need >= 8 and lower bound > 4) -> SUPPORTED
- I2 A2C-n64: L_ZF(0.1) > L_ZF(2) in 5/10 runs (need >= 8 of 10) -> NOT SUPPORTED
- I3 A2C-n64: mean L_ZF(0.1) - L_LL(0.1) = 5.63 [-0.07, 10.73] (need lower bound > 0) -> NOT SUPPORTED
-    exploratory A2C-n64: L_ZF(0.1)=16 in 8/10; L_FL(0.1)<=1 in 8/10; Cohen d (ZF vs FL at 0.1) = 2.86
- DQN-default runs per rule: {'FL': 10, 'LL': 10, 'ZF': 10}
- I1 DQN-default: mean L_ZF(0.1) - L_FL(0.1) = 7.15 [1.65, 12.10] (need >= 8 and lower bound > 4) -> NOT SUPPORTED
- I2 DQN-default: L_ZF(0.1) > L_ZF(2) in 4/10 runs (need >= 8 of 10) -> NOT SUPPORTED
- I3 DQN-default: mean L_ZF(0.1) - L_LL(0.1) = 7.97 [2.80, 12.55] (need lower bound > 0) -> SUPPORTED
-    exploratory DQN-default: L_ZF(0.1)=16 in 9/10; L_FL(0.1)<=1 in 4/10; Cohen d (ZF vs FL at 0.1) = 1.13

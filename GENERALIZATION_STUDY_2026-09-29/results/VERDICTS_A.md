# Study A verdicts

- A1 E2 under S3: false positive by region {'DK1': True, 'DK2': False} (need >=1) -> SUPPORTED
- A2 ledger check: flagged n/a of 0 floor-bound E2 runs (need >=0.9); flagged 0.0 of 24 S0 runs (need 0) -> NOT TESTABLE (floor never bound)
- A3 price-taker ratio S1/S0 of best reported profit: E1-small DK1 0.94, E1-small DK2 1.03, E1-large DK1 1.00, E1-large DK2 1.28 (need large >=2, small <1.10, both regions) -> NOT SUPPORTED
- A4 look-ahead false positive per design {'E1-large': False, 'E2': True, 'E3': False} (need >=2 of 3) -> NOT SUPPORTED
- A5 any-shortcut false positive per design {'E1': False, 'E2': True, 'E3': False} (need >=2 of 3) -> NOT SUPPORTED
- Strict (S0) verdicts, RL wins?: E1-large DK1: True, E1-large DK2: True, E1-small DK1: True, E1-small DK2: True, E2 DK1: False, E2 DK2: False, E3 DK1: True, E3 DK2: False

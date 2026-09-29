# Pre-registered prediction verdicts

- P1 liquidity illusion: anchor L1/L0 ratio [306.9 996.6] (need >=50 both); MARL mean higher under L1 [False, False] (need both) -> NOT SUPPORTED
- P2 solvency illusion: MARL L1 ruined-or-DD>=50% share 1.00 (need >=0.20); L3 reported-return mean > L1 [True, True] (need >=1) [ledger reading: [False, False]] -> SUPPORTED
- P3 learning exploits illusion: retrained@L1 -95.965 vs strict-trained@L1 -97.539 (need >); retrained@L0 0.004 vs strict-trained@L0 -0.201 (need <=) -> NOT SUPPORTED / INCOMPLETE
- P4 metric illusion: share of runs with fund 10-min Sharpe >10x investor daily Sharpe 0.41 (need >=0.8); Sharpe-rank flips after removing top-3 events: 1 (need >=1) -> NOT SUPPORTED
- P5 ranking flip (reported investor return): flipped under ["L7_thesis_like/original: ['anchor', 'feasible', 'marl'] -> ['marl', 'feasible', 'anchor']", "L7_thesis_like/v2: ['anchor', 'feasible', 'marl'] -> ['marl', 'feasible', 'anchor']"] -> SUPPORTED [ledger reading: ["L4_load_proxy_liquidity/original: ['feasible', 'anchor', 'marl'] -> ['anchor', 'feasible', 'marl']", "L6_interpolated_prices/original: ['feasible', 'anchor', 'marl'] -> ['anchor', 'feasible', 'marl']", "L7_thesis_like/v2: ['anchor', 'feasible', 'marl'] -> ['feasible', 'anchor', 'marl']"]]
- Not applicable: L5_percent_payoff x feasible-action. The engine refuses this combination ([TRADE_EXEC_FATAL] feasible-action MAPPO requires the mwh_volume payoff protocol), so its 6 runs exit=1 by design.
- Q1 direction saturation: min dominant share 0.741 over 14 new runs (need >=0.90 each) -> NOT SUPPORTED
- Q2 drift mechanism: sign(ledger)==dominant direction in 14/14 new runs (need 14) -> SUPPORTED
- Q3 zero-floor loss cap: 14 L7 MARL runs with ledger < -100%; min reported -114.3%, min fund -5.6% (need >=-120 and >=-10) -> SUPPORTED
- Q4 headline at 10 seeds: MARL reported mean [269.9, 121.8] vs anchor [-35.9, -41.5] (> both: [True, True]); MARL ledger mean [-534.8, -430.7] vs anchor [-35.8, -41.3] (< both: [True, True]) -> SUPPORTED; long-saturated seeds: 3/10
- R1 payoff necessary: L7_minus_percent_mtm inversion [True, True] (marl, anchor) {'original': (390.1, 130.3), 'v2': (484.8, 71.5)} (need none) -> NOT SUPPORTED
- R2 floor hides ledger: L7_minus_no_solvency max |reported-ledger| 0.42 pp (need <=5) -> SUPPORTED
- R3 interpolation not necessary: L7_minus_interp inversion [True, True] {'original': (407.2, -75.6), 'v2': (659.3, -89.7)} (need >=1) -> SUPPORTED
- R4 fee/sweeper not necessary: legacy_fee [True, True] {'original': (353.0, -35.9), 'v2': (202.6, -41.5)}, sweeper [True, True] {'original': (546.2, -35.9), 'v2': (289.5, -41.5)} (need all) -> SUPPORTED
- R5 minimal pair: L8_pctmtm_nosolv inversion [True, True] {'original': (-0.3, -0.7), 'v2': (-0.3, -0.3)} (need >=1) -> SUPPORTED
- Reported only: L7_minus_price_taker inversion [True, False] {'original': (-0.3, -0.3), 'v2': (-0.2, -0.2)}

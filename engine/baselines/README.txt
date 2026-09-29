Prototype5 Final-Campaign Baselines
===================================

The root wrapper is the single authoritative baseline entry point:

  python baselines.py

Protocol
--------

  payoff:             one-hour horizon-settlement MWh-volume payoff
  entry:              current day-ahead price
  settlement:         official external settlement series
  settlement bounds:  [-111750, 111750] DKK/MWh technical guard
  execution:          25% activation-volume participation cap
  liquidity floor:    1 MWh
  costs:              sourced per-MWh fee, access fee, 5 bp half spread,
                      and volume impact
  tail impact:        enabled above 5000 DKK/MWh
  collateral:         stress margin plus 5% annual funding cost
  sleeve risk:        maintenance-margin ruin handling
  Sharpe:             daily HAC(7), reported as N/A after ruin

Default outputs
---------------

  baseline_results/prototype5_final_paper_baselines_v1
  baseline_results/prototype5_temporal_baselines_v1

Forecast-corruption controls are owned by Harness 1:

  python harness1.py

Use `python baselines.py --dry_run` to inspect commands without executing them.

# FoCAL: Risk-Controlled Forecast Utilization for Hybrid Renewable-Energy Asset Management

This repository contains the code and evaluation artifacts for **FoCAL**, a hierarchical multi-agent reinforcement learning controller for hybrid renewable-energy asset management in day-ahead electricity markets.

FoCAL studies how imperfect day-ahead price forecasts can be used safely in operational decision-making. Instead of feeding raw forecasts directly into the policy, FoCAL converts an ANN forecast into a **conformal action prior**: a bounded exposure target filtered by directional reliability, residual uncertainty, and volatility shrinkage. The investor policy then learns only a small residual around this prior.

The work is prepared for submission to **IEEE Transactions on Smart Grid**.

## Core Idea

The simulated portfolio is initialized at `$800M` and split by a fixed capital partition:

- `88%` physical sleeve: renewable generation and battery storage
- `12%` financial sleeve: day-ahead market exposure

Four agents act in a shared environment:

- Investor / exposure agent: PPO
- Battery operator: DQN
- Risk controller: PPO
- Meta allocation controller: PPO

The main claim is risk-adjusted rather than return-maximizing: FoCAL improves Sharpe ratio, drawdown, volatility, trading-sleeve risk, and execution-cost robustness while preserving comparable retained NAV.

## Key Evaluation Protocol

The paper uses a strict **forward-filled no-sweeper** evaluation protocol:

- Hourly day-ahead prices are forward-filled onto the 10-minute control grid.
- Linear interpolation is avoided to remove artificial short-horizon predictability.
- Evaluation disables the cash sweeper with `eval_distribution_rate = 0.0`.
- Metrics are annualized with `periods_per_year = 52596`.
- The financial sleeve starts at `$96M`, i.e., `12%` of `$800M`.

## Main Robustness Results

The repository includes scripts and outputs for:

- Forward-filled no-sweeper evaluation
- Trading-sleeve risk decomposition
- MAPPO 2x2 comparison: IPPO/MAPPO by prior/no-prior
- Slope-prior controls
- Weak-gate operating point
- DK2 held-zone robustness evaluation
- Square-root order-size market-impact sensitivity

## Repository Structure

```text
.
|-- environment.py
|-- financial_engine.py
|-- evaluation.py
|-- forecast_prior.py
|-- metacontroller.py
|-- runtime_contract.py
|-- run_tier_phase_multi_seed.py
|-- scripts/
|   |-- run_ffill_sleeve_eval_50.py
|   |-- run_ffill_mappo_eval.py
|   |-- run_impact_sweep_eval.py
|   `-- run_dk2_cross_zone_eval.py
|-- dataset generation/
|-- results/
|   |-- ffill_robustness/
|   |-- ffill_sleeve/
|   |-- ffill_mappo/
|   |-- ffill_impact/
|   `-- dk2_cross_zone/
|-- eval_v2_ffill_retrained/
|-- robustness_ffill_retrained/
`-- Paper Template/
```

## Reproducibility Notes

The trained policy checkpoints are reused for evaluation. The paper results are evaluation-only unless training scripts are explicitly run.

Important checkpoints are expected under:

```text
batch_tier_phase_runs/<variant>/seed<N>/<run_name>/final_models/
```

The main forward-filled evaluation data are expected under:

```text
robustness_ffill_retrained/evaluation_dataset/unseendata.csv
robustness_ffill_retrained/forecast_cache/
```

DK2 held-zone data and forecast cache are generated separately to avoid mixing price zones.

## Example Commands

Aggregate an already completed market-impact sweep:

```bash
python scripts/run_impact_sweep_eval.py --aggregate_only
```

Run or aggregate the DK2 held-zone evaluation:

```bash
python scripts/run_dk2_cross_zone_eval.py
```

## Disclaimer

This repository is research code for simulation and reproducibility. It is not financial advice, trading infrastructure, or a production electricity-market bidding system. The market-impact model is a robustness sensitivity test and does not reconstruct confidential Danish day-ahead auction order books.

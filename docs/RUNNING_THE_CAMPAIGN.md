# Running the engine campaign

The standalone checks (`tools/`, `analysis/`, `code_audit/`) run from this repository alone; see the README. This guide covers the **engine-dependent** experiments in `audit/`, which produced `results/campaign/`.

## What is and is not in this repository

| Needed for the campaign | In this repository? |
|---|---|
| Engine source (the patched audit copy that produced the results) | **yes**: `engine/` |
| Audit patch, as a diff against the unpatched engine | **yes**: `audit/engine_patch/` |
| Campaign runner, aggregator, variant and replay scripts | **yes**: `audit/` |
| Engine datasets: training / evaluation windows, settlement, liquidity and forecast inputs | no; the protocol-dataset builders are in `engine/scripts/build_*.py`, and their provenance manifests in `engine/*manifest*.json` |
| Trained checkpoints (MARL, feasible-action, anchor; 10 seeds) | no (~57 GB with datasets) |
| Raw per-run logs | no (~105 GB); aggregated results are in `results/campaign/` |

## Environment
Python 3.10 with `requirements.txt` plus `requirements-engine.txt`: PyTorch, TensorFlow, Stable-Baselines3, Gymnasium, PettingZoo.

## Expected layout
The audit scripts were run in this workspace layout, and their paths assume it:
```
<workspace>/
  Prototype5/                    unpatched engine source + dataset folders + trained checkpoints
  SIMULATOR_AUDIT_2026-09-28/    the scripts in audit/ (plus a data/ folder for interpolated prices)
```
- To recreate `Prototype5/`, run `git apply -R audit/engine_patch/evaluation_py_sim_audit.diff` from the repository root. This restores the unpatched `engine/evaluation.py`; then copy `engine/` to `Prototype5/`.
- Then add the dataset folders and checkpoints.
- `setup_audit_engine.py` then builds the patched audit copy, links the dataset folders, and checks that every other file is byte-identical to `Prototype5/`.

## Phases
```
python run_scarcity_tail_campaign.py --phase setup
python run_scarcity_tail_campaign.py --phase validate --workers 5   # gate: strict re-evaluation must match the frozen campaign within 1e-6 pp
python run_scarcity_tail_campaign.py --phase evals --workers 5      # L1-L7 (182 jobs)
python run_scarcity_tail_campaign.py --phase retrain                # MARL retrained under L1 (3 seeds), then evaluated under L1 and L0
python run_scarcity_tail_campaign.py --phase l7_ext --workers 7     # L7 for the other 7 MARL seeds (pre-registration addendum 2)
python run_scarcity_tail_campaign.py --phase l7_loo --workers 7     # leave-one-out of L7 (56 jobs; addendum 3)
python run_scarcity_tail_campaign.py --phase aggregate
```

**Runtimes on an 8-core CPU** (one thread per job):
- evaluations take 10–35 minutes per job;
- retraining takes about 4–6 hours for 3 seeds in parallel.

The runner is resume-safe. It skips completed jobs.

**How overrides are applied.** Every audit evaluation sets:
- `SIM_AUDIT_VARIANT`;
- `SIM_AUDIT_ALLOW_PROTOCOL_MISMATCH=1` (all protocols except L0);
- where needed, `SIM_AUDIT_CFG_OVERRIDES`.

The applied overrides are recorded under `sim_audit` in each result JSON.

## Outputs
`aggregate_scarcity_tail_campaign.py` writes:
- `results_long.csv`: one row per run, with reported and ledger returns, drawdown, ruin, Sharpe, and direction shares;
- `results_summary.csv`;
- `ledger_vs_reported.csv`;
- `PREDICTION_VERDICTS.md`: P1–P5, Q1–Q4, R1–R5.

## Classical strategies and decision replay
`audit/run_all_variants.py` and `audit/audit_run_variant.py` run the engine's classical baselines under the V00–V12 shortcuts; `audit/aggregate_variants.py` summarises them. `audit/decision_replay.py` re-prices logged decisions against the real settlement price. `audit/make_interpolated_eval_data.py` builds the interpolated-price datasets used by L6 and L7.

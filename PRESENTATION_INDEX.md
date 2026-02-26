# cs224n-spark Presentation Index

This repo contains the Spark pivot experiments (A/B/C/D) and local copies of `spark-pilot-results` from Modal.

## Repo Rename
- Previous local name: `fork`
- Current local name: `cs224n-spark`

## Downloaded Modal Data
Root: `data/modal_runs/`

## Chronological Run Groups

### Stage 1: Detection calibration
- `data/modal_runs/spark_pilot_20260222_110122/`
- `data/modal_runs/recal_20260222_192254/`

### Stage 2: Training plumbing and viability
- `data/modal_runs/spark_smoke/`
- `data/modal_runs/spark_full/`
- `data/modal_runs/spark_viability/`
- `data/modal_runs/math_smoke/`

### Stage 3: Routing efficacy
- `data/modal_runs/routing_smoke/`
- `data/modal_runs/routing_viability/`

### Diagnostics
- `data/modal_runs/trigger_diagnostic/trigger_diagnostic_report.json`
- `data/modal_runs/spark_sweep/sweep_report.json`

## Key Files for Presentation
- `*/summary.txt` and `*/results.json` for stage summaries.
- `*/condition_*/diagnostics_*.json` for per-step mechanism checks.
- `*/condition_*/st_trajectory_*.json` and `mass_balance_*.json` for trajectory diagnostics.

# cs224n-spark

Standalone repo for the Spark pivot experiments (A/B/C/D) and downloaded Modal outputs.

## What is here
- `pilot/`: Spark experiment code and local pilot outputs
- `data/modal_runs/`: downloaded results from Modal volume `spark-pilot-results`

## Run groups
- Stage 1 (detection calibration):
  - `data/modal_runs/spark_pilot_20260222_110122/`
  - `data/modal_runs/recal_20260222_192254/`
- Stage 2 (training plumbing / viability):
  - `data/modal_runs/spark_smoke/`
  - `data/modal_runs/spark_full/`
  - `data/modal_runs/spark_viability/`
  - `data/modal_runs/math_smoke/`
- Stage 3 (routing efficacy):
  - `data/modal_runs/routing_smoke/`
  - `data/modal_runs/routing_viability/`
- Diagnostics:
  - `data/modal_runs/trigger_diagnostic/trigger_diagnostic_report.json`
  - `data/modal_runs/spark_sweep/sweep_report.json`

## Notes
- `docs/` and `.claude/` are intentionally excluded from git.

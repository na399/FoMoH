# FoMoH Evaluation Manual

This manual is a dataset-agnostic operator guide for running FoMoH-style evaluation. It describes the reusable workflow, artifact layout, and verification gates. Dataset-specific paths, task lists, exclusions, and caveats belong in `spec/` files and generated `runs/` reports.

Use this order:

1. [Data And Tasks](01_data_and_tasks.md)
2. [Environments And Exports](02_environments_and_exports.md)
3. [Smoke Runs](03_smoke_runs.md)
4. [Full-Active Benchmarks](04_full_active_benchmarks.md)
5. [Metrics And Reporting](05_metrics_and_reporting.md)
6. [Troubleshooting](06_troubleshooting.md)

## Inputs

Define a run contract before launching model work:

```shell
export DATASET_NAME="<dataset_id>"
export MEDS_DIR="<path_to_meds>"
export MEDS_READER_DIR="<path_to_meds_reader>"
export OMOP_DUCKDB="<path_to_omop_duckdb>"
export ATHENA_VOCAB_DUCKDB="<path_to_athena_vocab_duckdb>"
export TASK_MANIFEST="<path_to_active_task_manifest_json>"
export RUN_ROOT="runs/${DATASET_NAME}"
export TEMP_ROOT="temp/${DATASET_NAME}"
export TMPFS_ROOT="/dev/shm/${DATASET_NAME}"
```

Only use real local paths in a dataset-specific run note, not in this manual.

## Hydra Runner

Use the Hydra runner to compose reproducible command plans before running model work. Plans are dry-run by default and are written as JSON plus shell commands under `${RUN_ROOT}/hydra_plans/`.

```shell
uv run python -m ehr_foundation_model_benchmark.fomoh_mimic.hydra_app \
  dataset=<dataset_config> \
  phase=<validate|smoke|full_active|report> \
  model=<model_config> \
  paths.run_root="${RUN_ROOT}" \
  paths.temp_root="${TEMP_ROOT}" \
  paths.tmpfs_root="${TMPFS_ROOT}" \
  --dry-run
```

Use `--execute` only after reviewing the emitted plan and confirming the required resource gate. Dataset configs should reference environment variable names or portable placeholders; do not commit private data paths or credentials.

## Artifact Roots

| Purpose | Convention |
| --- | --- |
| Transient logs and helper scripts | `${TEMP_ROOT}/` |
| Large scratch, venvs, caches, temporary checkpoints | `${TMPFS_ROOT}/` |
| Durable manifests, metrics, reports, checkpoints | `${RUN_ROOT}/` |
| Reproducible patch records | `patches/` or `${RUN_ROOT}/patches/` |

Keep raw data, generated labels, features, checkpoints, logs, and large scratch artifacts out of git.

## Scope Rules

Every run should explicitly record:

- dataset identifier;
- active task manifest;
- excluded or non-identifiable tasks;
- model stacks attempted;
- model stacks skipped by instruction or infeasibility;
- whether each result is reportable, smoke-only, blocked, skipped, or non-identifiable.

## GPU Gate

Before CUDA work:

```shell
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
```

Launch GPU work only when the run-specific GPU policy passes. Do not stop another GPU job without explicit approval.

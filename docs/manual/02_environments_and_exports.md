# 02. Environments And Exports

This page creates isolated model environments and model-specific data exports in a reusable way.

## Step 1: Use tmpfs For Heavy Scratch

```shell
export TMPFS_ROOT="/dev/shm/${DATASET_NAME}"
```

Use `${TMPFS_ROOT}` for isolated virtual environments, package caches, tokenizer caches, prepared data, temporary checkpoints, and large intermediate feature files. Keep durable summaries and manifests under `${RUN_ROOT}`.

## Step 1a: Emit A Validation Plan

Before creating environments or exports, emit a validate-phase plan so operators can review the dataset contract and output roots.

```shell
uv run python -m ehr_foundation_model_benchmark.fomoh_mimic.hydra_app \
  dataset=<dataset_config> \
  phase=validate \
  model=<model_config> \
  paths.run_root="${RUN_ROOT}" \
  paths.temp_root="${TEMP_ROOT}" \
  paths.tmpfs_root="${TMPFS_ROOT}" \
  --dry-run
```

Review `${RUN_ROOT}/hydra_plans/validate_plan.json` before running data preparation.

## Step 2: Create Isolated Environments

Use one `uv` environment per fragile model stack:

```shell
uv run python -m ehr_foundation_model_benchmark.fomoh_mimic.model_envs   --models "${MODEL_LIST}"   --tmpfs-root "${TMPFS_ROOT}"   --summary-json "${RUN_ROOT}/env_manifests/env_setup_summary.json"
```

Do not use the repo `.venv` for fragile external model stacks. Record interpreter path, package versions, install command, import smoke status, and log path.

## Step 3: Export Vocabulary Assets

If a model requires Athena or vocabulary CSVs, export them from the run-specific vocabulary source:

```shell
uv run python -m ehr_foundation_model_benchmark.fomoh_mimic.cli export-athena-vocab   --duckdb "${ATHENA_VOCAB_DUCKDB}"   --output-dir "${TMPFS_ROOT}/athena_vocab_csv"   --summary-json "${RUN_ROOT}/athena_vocab_summary.json"

uv run python -m ehr_foundation_model_benchmark.fomoh_mimic.cli validate-athena-vocab   --export-dir "${TMPFS_ROOT}/athena_vocab_csv"   --output-json "${RUN_ROOT}/athena_vocab_validation.json"
```

## Step 4: Export Model-Specific Layouts

Export only the layouts needed by selected models:

```shell
uv run python -m ehr_foundation_model_benchmark.fomoh_mimic.cli export-omop-smoke-layouts   --duckdb "${OMOP_DUCKDB}"   --output-root "${TMPFS_ROOT}/omop_smoke_layouts"   --max-persons "${SMOKE_MAX_PERSONS}"   --summary-json "${RUN_ROOT}/omop_smoke_layouts_summary.json"
```

For a new dataset, document each model input contract: sequence reader, table folder, flat event table, tokenizer files, vocabulary files, cohort files, and label format.

## Step 5: Preserve Compatibility Patches

If a model stack needs local compatibility edits, preserve a patch and manifest explaining target package, files changed, reason, verification command, and reapply procedure.

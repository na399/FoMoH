# 01. Data And Tasks

This page prepares dataset resources and task labels without assuming a particular dataset or task set.

## Step 1: Write A Run Contract

Create or update a dataset-specific contract that records:

- `${DATASET_NAME}`;
- `${MEDS_DIR}`;
- `${MEDS_READER_DIR}`;
- `${OMOP_DUCKDB}`;
- `${ATHENA_VOCAB_DUCKDB}`;
- `${TASK_MANIFEST}`;
- active task families;
- excluded tasks and the reason for each exclusion;
- output roots.

The run contract should live under `${RUN_ROOT}/` or `spec/`, depending on whether it is durable evidence or implementation planning.

## Step 2: Validate Source Resources

Run the repository resource validator or a dataset-specific validator:

```shell
uv run python -m ehr_foundation_model_benchmark.fomoh_mimic.cli validate-resources   --output-json "${TEMP_ROOT}/resource_summary.json"
```

If a built-in validator has local defaults, write a dataset-specific validator that emits the same kind of evidence: source path, split names, row counts, schema summary, readable sample rows, and missing required inputs.

## Step 3: Generate Labels

Generate labels for every active task and store them under `${RUN_ROOT}`:

```shell
uv run python -m ehr_foundation_model_benchmark.fomoh_mimic.cli generate-labels   --meds-dir "${MEDS_DIR}"   --output-dir "${RUN_ROOT}/task_labels/<task_family>"   --summary-json "${RUN_ROOT}/label_counts.json"
```

For task families generated from OMOP or another structured source:

```shell
uv run python -m ehr_foundation_model_benchmark.fomoh_mimic.cli generate-phenotype-labels   --duckdb "${OMOP_DUCKDB}"   --output-dir "${RUN_ROOT}/task_labels/<phenotype_family>"   --summary-json "${RUN_ROOT}/label_counts_<phenotype_family>.json"
```

## Step 4: Build The Active Task Manifest

```shell
uv run python -m ehr_foundation_model_benchmark.fomoh_mimic.cli task-bundle-manifest   --labels-dir "${RUN_ROOT}/task_labels"   --output-json "${TASK_MANIFEST}"
```

Record excluded tasks separately with reasons such as zero positives, missing source concepts, single-class held-out split, deferred scope, or package unsupported.

## Step 5: Freeze Label Evidence

Before model work, confirm each active task has train, tuning, and held-out labels; split counts; prediction-time semantics; identifiers; and no held-out leakage into training.

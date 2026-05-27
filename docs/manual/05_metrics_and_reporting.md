# 05. Metrics And Reporting

This page regenerates the benchmark matrix, runtime profile, and final summary for any dataset.

## Step 0: Emit A Report Plan

Use the report phase to record the reporting commands before regenerating summaries.

```shell
uv run python -m ehr_foundation_model_benchmark.fomoh_mimic.hydra_app \
  dataset=<dataset_config> \
  phase=report \
  model=<model_config> \
  paths.run_root="${RUN_ROOT}" \
  paths.temp_root="${TEMP_ROOT}" \
  paths.tmpfs_root="${TMPFS_ROOT}" \
  --dry-run
```

Keep the generated report plan with the benchmark matrix so readers can trace how the summary was produced.

## Step 1: Regenerate The Benchmark Matrix

```shell
uv run python "${STATUS_REPORT_SCRIPT}"   --dataset-name "${DATASET_NAME}"   --task-manifest "${TASK_MANIFEST}"   --run-root "${RUN_ROOT}"   --output-json "${RUN_ROOT}/benchmark_scale_status.json"   --output-md "${RUN_ROOT}/benchmark_scale_status.md"
```

If the current status writer has dataset-specific defaults, wrap it with a dataset-specific script and keep the output schema stable.

The matrix should include model, task, status, run type, AUROC, AUPRC, Brier, calibration metrics, workload metrics, and evidence path.

## Step 2: Regenerate The Runtime Profile

```shell
uv run python "${PROFILE_SCRIPT}"   --run-root "${RUN_ROOT}"   --output-json "${RUN_ROOT}/benchmark_profile.json"   --output-md "${RUN_ROOT}/benchmark_profile.md"
```

The profile should separate optimization evidence from model-quality claims.

## Step 3: Regenerate The Final Summary

```shell
uv run python -m ehr_foundation_model_benchmark.fomoh_mimic.cli phase1-summary   --status-json "${RUN_ROOT}/benchmark_scale_status.json"   --output-md "${RUN_ROOT}/phase1_benchmark_summary.md"
```

The summary must separate reportable full-active rows, smoke-only rows, blocked rows, skipped rows, and non-identifiable rows.

## Step 4: Interpret Feasible Metrics

For reportable rows, include all feasible metrics: AUROC when both classes are present, AUPRC when positives are present, Brier score, expected calibration error, PR lift, PPV at fixed workload, and sensitivity at fixed workload.

If a metric is mathematically undefined, write `undefined` or leave it blank and explain why. Do not fabricate metrics for missing predictions or single-class slices.

## Step 5: Record Dataset-Specific Caveats Outside This Manual

Dataset-specific caveats belong in the generated summary or implementation tracker. Caveat categories include `empty_predictions`, zero-positive task, single-class held-out split, unsupported model/task format, skipped model by user instruction, package runtime blocker, and GPU or kernel incompatibility.

## Step 6: Verify

```shell
uv run --with pytest pytest   tests/test_fomoh_mimic_phase1_summary.py   tests/test_fomoh_mimic_manual_docs.py   tests/test_fomoh_mimic_medstab_metrics.py   tests/test_fomoh_mimic_smoke_infra.py   tests/test_fomoh_mimic_labels.py   tests/test_fomoh_mimic_metrics.py -q

uv run python -m py_compile src/ehr_foundation_model_benchmark/fomoh_mimic/*.py
```

Record verification in the implementation tracker.

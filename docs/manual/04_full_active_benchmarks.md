# 04. Full-Active Benchmarks

Full-active runs use the active task manifest and full labels for `${DATASET_NAME}`. These runs are the source for reportable benchmark rows.

## Step 1: Confirm Active Scope

```shell
echo "${DATASET_NAME}"
echo "${TASK_MANIFEST}"
```

The active manifest should exclude dataset-specific non-identifiable tasks and record those exclusions outside this manual.

## Step 2: Run Baseline Or Tabular Models

For each baseline `${MODEL_NAME}`:

```shell
uv run python "${BASELINE_SCRIPT}"   --dataset-name "${DATASET_NAME}"   --task-manifest "${TASK_MANIFEST}"   --labels-root "${RUN_ROOT}/task_labels"   --output-dir "${RUN_ROOT}/model_benchmarks/${MODEL_NAME}"   --tmp-dir "${TMPFS_ROOT}/${MODEL_NAME}"
```

If the package emits no held-out predictions for a task, record an `empty_predictions` row and preserve the package output.

## Step 3: Run Sequence Foundation Models

For each sequence model `${MODEL_NAME}`:

```shell
/usr/bin/env MODEL_NAME="${MODEL_NAME}" DATASET_NAME="${DATASET_NAME}"   TASK_MANIFEST="${TASK_MANIFEST}" RUN_ROOT="${RUN_ROOT}" TEMP_ROOT="${TEMP_ROOT}"   TMPFS_ROOT="${TMPFS_ROOT}" WANDB_MODE=disabled   "${TMPFS_ROOT}/venvs/${MODEL_NAME}/bin/python" "${FOUNDATION_BENCHMARK_SCRIPT}"
```

The benchmark script should write feature files, probe outputs, model/task summaries, and persistent logs under the configured run roots.

## Step 4: Configure GPU Kernel-Specific Models

Some models need validated acceleration kernels. Record kernel setup in a model-specific manifest:

```shell
MAX_JOBS="${MAX_JOBS}" CUDA_HOME="${CUDA_HOME}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"   UV_CACHE_DIR="${TMPFS_ROOT}/uv_cache"   uv pip install --python "${TMPFS_ROOT}/venvs/${MODEL_NAME}/bin/python"   --no-deps --no-build-isolation "${KERNEL_PACKAGE_1}" "${KERNEL_PACKAGE_2}"
```

Then verify import and a finite forward pass before the full-active run.

## Step 5: Use Validated Batch Settings

```shell
export FOMOH_MODEL_BATCH_SIZE="${VALIDATED_BATCH_SIZE}"
export FOMOH_MODEL_AUTOCAST="${AUTOCAST_ENABLED}"
export FOMOH_MODEL_AUTOCAST_DTYPE="${AUTOCAST_DTYPE}"
export FOMOH_MODEL_TOKENIZE_BATCH_SIZE="${TOKENIZE_BATCH_SIZE}"
export FOMOH_MODEL_READER_THREADS="${READER_THREADS}"
```

If a larger batch OOMs, document it as a rejected setting and keep the validated setting as the default.

## Step 6: Run Smoke-Only Downstream Adapters Separately

If a model stack only has capped or smoke-sized downstream evidence, keep it out of reportable rows:

```shell
uv run python "${DOWNSTREAM_EXPORT_SCRIPT}"   --dataset-name "${DATASET_NAME}"   --task-manifest "${TASK_MANIFEST}"   --output-root "${TMPFS_ROOT}/downstream_active"

uv run python "${DOWNSTREAM_SMOKE_SCRIPT}"   --dataset-name "${DATASET_NAME}"   --task-manifest "${TASK_MANIFEST}"   --output-json "${RUN_ROOT}/model_benchmarks/downstream_active_smokes.json"
```

Single-class smoke slices should be `non-identifiable`, not benchmark failures.

# 03. Smoke Runs

Smoke runs prove that installation, data adapters, feature extraction, and probes work. They are not benchmark claims.

## Step 1: Run The GPU Gate

```shell
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
```

Use CPU-only smoke paths where possible before launching GPU work.

## Step 1a: Emit A Smoke Plan

Use Hydra to write the smoke command plan before launching model-specific commands.

```shell
uv run python -m ehr_foundation_model_benchmark.fomoh_mimic.hydra_app \
  dataset=<dataset_config> \
  phase=smoke \
  model="${MODEL_NAME}" \
  paths.run_root="${RUN_ROOT}" \
  paths.temp_root="${TEMP_ROOT}" \
  paths.tmpfs_root="${TMPFS_ROOT}" \
  --dry-run
```

Review the plan JSON and shell file under `${RUN_ROOT}/hydra_plans/`. If the smoke requires GPU, execute only after the resource gate passes.

## Step 2: Run Import And CLI Smokes

For each `${MODEL_NAME}`:

```shell
"${TMPFS_ROOT}/venvs/${MODEL_NAME}/bin/python" - <<'PY'
import importlib
import os
package = os.environ["MODEL_IMPORT_NAME"]
importlib.import_module(package)
print(f"import_ok {package}")
PY
```

Record the command, return code, and log under `${TEMP_ROOT}`. Write a durable manifest under `${RUN_ROOT}/env_manifests/`.

## Step 3: Run Minimal Data Preparation

Run the smallest preparation step that exercises the model input contract:

```shell
"${TMPFS_ROOT}/venvs/${MODEL_NAME}/bin/python" "${MODEL_PREP_SCRIPT}"   --input "${MODEL_INPUT_ROOT}"   --labels "${TASK_MANIFEST}"   --output "${TMPFS_ROOT}/${MODEL_NAME}/prepared_smoke"   --max-rows "${SMOKE_MAX_ROWS}"
```

The smoke should confirm schema compatibility, split handling, tokenizer or vocabulary loading, label joins, and deterministic output paths.

## Step 4: Run Minimal Training Or Checkpoint Loading

```shell
WANDB_MODE=disabled "${TMPFS_ROOT}/venvs/${MODEL_NAME}/bin/python" "${MODEL_TRAIN_SCRIPT}"   --prepared-data "${TMPFS_ROOT}/${MODEL_NAME}/prepared_smoke"   --output-dir "${TMPFS_ROOT}/${MODEL_NAME}/train_smoke"   --max-steps "${SMOKE_MAX_STEPS}"
```

For pretrained-only stacks, load the checkpoint and run one forward pass or feature extraction batch.

## Step 5: Run Feature And Probe Smoke

```shell
"${TMPFS_ROOT}/venvs/${MODEL_NAME}/bin/python" "${MODEL_FEATURE_SCRIPT}"   --checkpoint "${MODEL_CHECKPOINT}"   --task-manifest "${TASK_MANIFEST}"   --output-dir "${RUN_ROOT}/features/${MODEL_NAME}_smoke"

uv run python "${PROBE_SCRIPT}"   --features-dir "${RUN_ROOT}/features/${MODEL_NAME}_smoke"   --output-dir "${RUN_ROOT}/probes/${MODEL_NAME}_smoke"
```

Record single-class or zero-positive slices as `non-identifiable`; do not fail the whole smoke unless the adapter crashes.

## Step 6: Update The Tracker

After every smoke, update the implementation tracker with command, log path, output path, evidence, status, and lesson learned.

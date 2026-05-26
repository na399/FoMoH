# 06. Troubleshooting

Use this page for reusable failure handling. Record dataset-specific failures and fixes in the implementation tracker.

## GPU Memory Is Occupied

```shell
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
```

Wait if another job is using most memory. Do not stop someone else's job without explicit approval.

## tmpfs Was Cleared

Recreate isolated `uv` venvs, vocabulary exports, model-specific source-data exports, prepared scratch data, tokenizer caches, and benchmark caches. Then rerun smoke checks before full-active benchmarks.

## A Kernel Install Changes The Core Framework

If a kernel package tries to replace the existing Torch or CUDA stack, stop and reinstall with a pinned, no-dependency command:

```shell
uv pip install --python "${TMPFS_ROOT}/venvs/${MODEL_NAME}/bin/python"   --no-deps --no-build-isolation "${KERNEL_PACKAGE_1}" "${KERNEL_PACKAGE_2}"
```

Verify framework version, CUDA version, import status, and a finite forward pass before training or feature extraction.

## A Model Tokenizer Fails

Check special token names, visit boundary tokens, vocabulary size, code normalization, tokenizer cache key, and metadata serialization. Preserve compatibility fixes in a reproducible patch file.

## Feature Extraction Skips Rows

Check whether model feature timestamps exactly match label prediction timestamps. If the model emits nearest-event or sequence-end features, use a documented alignment policy and preserve the alignment summary.

## A Package Emits Empty Predictions

Do not fabricate metrics. Keep the task status as `empty_predictions` and preserve official package output, task labels, compute summary, log file, and rerun command.

## A Metric Is Undefined

If held-out labels are single-class, AUROC is undefined. Preserve other metrics only where mathematically defined and mark the row `non-identifiable` if the task cannot support the intended evaluation.

## A New Failure Appears

Immediately update the implementation tracker with command, log path, exact error, affected model/task, and whether the failure is blocked, non-identifiable, skipped, or fixed.

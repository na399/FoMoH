# FoMoH Agent Instructions

## Prime Directive

- Do not stop at partial analysis when the user has asked for an implementation.
- Avoid quick hacks and hidden technical debt. Prefer small, correct, verifiable changes.
- Use test-driven development whenever practical: write or update tests first, confirm the failure, implement, then rerun tests.
- Put disposable files in `temp/`. Do not delete the `temp/` directory.
- Ask before irreversible actions such as deleting data, stopping someone else's job, or overwriting large artifacts.
- Use `uv` for Python environment management.
- Use `pnpm` for JavaScript package management if JavaScript tooling is introduced.
- Keep scripts under 1000 lines where possible; if a script grows past that, plan modularization.

## Repository Scope

This repository is a FoMoH benchmark workspace. Treat the upstream FoMoH code as a benchmark specification and selective source of reusable assets, not as a turnkey pipeline.

Current local focus:

- Phase 1: build a reproducible FoMoH-style MIMIC-IV run.
- Primary tracker: `spec/03_implementation_tasks.md`.
- MIMIC MEDS assets: `/home/natthawut/MEDS/mimiciv`.
- MIMIC MEDS Reader assets: `/home/natthawut/MEDS/mimiciv_reader`.
- First trainable model target: `hf_ehr`/CLMBR-style GPT-small MIMIC path.
- LeJEPAtient/BRANCHUS alignment is future scope unless the user explicitly moves it into active implementation.

## Execution Rules

- Before coding, inspect the live files and existing conventions.
- Keep diffs minimal and scoped to the requested task.
- Use structured parsers and library APIs instead of ad hoc text manipulation when practical.
- Prefer `fd` for file discovery and `rg` for text search. Use `ast-grep` for syntax-aware code searches.
- Use `jq` for JSON and `yq` for YAML/XML when available.
- If `rtk` is available, prefix shell commands with `rtk` as described in `/home/natthawut/.codex/RTK.md`. If it is unavailable, use deterministic direct commands and note that fallback.
- Do not run formatters that rewrite files unless formatting is part of the task.
- Never revert or overwrite user changes unless explicitly requested.

## FoMoH-on-MIMIC Workflow

- Keep `spec/03_implementation_tasks.md` updated after every meaningful run, failed run, patch, or discovered blocker.
- Use `temp/fomoh_mimic/` for transient logs, probes, and one-off scripts.
- Use `runs/fomoh_mimic/` for durable run outputs, metrics, reports, checkpoints, and manifests.
- Preserve logs for every smoke, training, feature extraction, linear probe, and failed run.
- Do not report metrics unless the evidence chain is complete:
  - data path and schema summary;
  - label counts and prevalence by split;
  - token coverage and non-PAD length summary;
  - training command, log path, and checkpoint path;
  - feature shape and feature-diversity summary;
  - prediction path and metric JSON path;
  - AUROC, AUPRC, PR lift, Brier, calibration summary, and workload metric summary.

## GPU Training Discipline

- An NVIDIA L40S is available, but check `nvidia-smi` before launching any GPU job.
- Do not start new GPU training while an unrelated job is using most GPU memory.
- Default launch gate: no unrelated process above 10 GiB and at least 35 GiB free.
- Do not stop an existing GPU job unless the user explicitly asks.
- Run CPU/tokenization smoke before any full GPU training.
- Run a short L40S training smoke before a full MIMIC model run.
- Use persistent logs and record checkpoint evidence for every GPU run.

## Data and Artifact Safety

- Treat MIMIC-derived data, labels, features, checkpoints, and logs as restricted local artifacts.
- Keep raw data, generated labels, features, checkpoints, W&B runs, and large outputs out of git.
- Use tmpfs at `/dev/shm/fomoh_mimic` for large scratch data, isolated `uv` virtualenvs, package caches, smoke exports, tokenizer caches, and temporary checkpoints when disk pressure is possible.
- Keep repo-visible pointers, durable run summaries, metrics, manifests, and final reports under `runs/fomoh_mimic/`; keep transient logs and scripts under `temp/fomoh_mimic/`.
- Do not print secrets or secret-backed environment variable values.
- Prefer passing secret names or paths through env-var indirection rather than echoing values.
- If disk pressure matters, inspect space before launching large conversion, featurization, or training jobs.

## Testing and Verification

- For Python changes, prefer focused `uv run pytest ...` tests before broad test runs.
- Use `uv run ruff check ...` for lint checks when relevant.
- For benchmark changes, add small deterministic tests that do not require restricted datasets.
- For pipeline changes, run a smoke path first and record:
  - command;
  - log path;
  - inputs;
  - outputs;
  - pass/fail evidence;
  - next action.

## Documentation Expectations

- Keep docs concise and operational.
- Update `spec/03_implementation_tasks.md` as the source of truth for implementation status.
- Document lessons learned directly in the tracker after each failed or completed run.
- When documenting future LeJEPAtient/BRANCHUS work, keep it clearly marked as deferred unless actively requested.

@/home/natthawut/.codex/RTK.md

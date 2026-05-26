from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

METRIC_KEYS = ("auroc", "auprc", "brier", "ece", "pr_lift", "ppv_at_100", "sensitivity_at_100")
REPORTABLE_RUN_TYPES = {
    "active_phenotype_labels",
    "benchmark_full_active",
    "official_full_active_full_metrics",
    "official_full_active_prediction_blocked",
    "reportable_full_outcome",
}
CEHR_CORE_SMOKE_RUN_TYPES = {"official_downstream_active_smoke", "official_downstream_smoke"}


@dataclass(frozen=True)
class Phase1Partitions:
    reportable: list[dict[str, Any]]
    smoke_only: list[dict[str, Any]]
    other: list[dict[str, Any]]


def partition_rows(rows: Iterable[dict[str, Any]]) -> Phase1Partitions:
    reportable: list[dict[str, Any]] = []
    smoke_only: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for row in rows:
        run_type = row.get("run_type")
        if run_type in REPORTABLE_RUN_TYPES:
            reportable.append(row)
        elif run_type in CEHR_CORE_SMOKE_RUN_TYPES:
            smoke_only.append(row)
        else:
            other.append(row)
    return Phase1Partitions(reportable=reportable, smoke_only=smoke_only, other=other)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "undefined"
        return f"{value:.4f}"
    return str(value)


def _mean(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if _finite(value)]
    return sum(finite) / len(finite) if finite else None


def summarize_models(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    summary = []
    for model, model_rows in sorted(grouped.items()):
        summary.append(
            {
                "model": model,
                "rows": len(model_rows),
                "metric_rows": sum(1 for row in model_rows if any(_finite(row.get(key)) for key in METRIC_KEYS)),
                "mean_auroc": _mean(row.get("auroc") for row in model_rows),
                "mean_auprc": _mean(row.get("auprc") for row in model_rows),
                "mean_brier": _mean(row.get("brier") for row in model_rows),
                "statuses": ", ".join(
                    f"{status}={count}"
                    for status, count in sorted(
                        {
                            status: sum(1 for row in model_rows if row.get("status") == status)
                            for status in {row.get("status", "unknown") for row in model_rows}
                        }.items()
                    )
                ),
                "run_types": ", ".join(sorted({row.get("run_type", "unknown") for row in model_rows})),
            }
        )
    return summary


def _table(lines: list[str], headers: list[str], rows: Iterable[list[str]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")


def _metric_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    _table(
        lines,
        ["Model", "Task", "Status", "AUROC", "AUPRC", "Brier", "ECE", "PR lift", "PPV@100", "Sens@100"],
        (
            [
                row.get("model", ""),
                row.get("task", ""),
                row.get("status", ""),
                _format_value(row.get("auroc")),
                _format_value(row.get("auprc")),
                _format_value(row.get("brier")),
                _format_value(row.get("ece")),
                _format_value(row.get("pr_lift")),
                _format_value(row.get("ppv_at_100")),
                _format_value(row.get("sensitivity_at_100")),
            ]
            for row in sorted(rows, key=lambda item: (item.get("model", ""), item.get("task", "")))
        ),
    )


def render_phase1_summary(payload: dict[str, Any]) -> str:
    rows = payload.get("rows", [])
    partitions = partition_rows(rows)
    excluded = payload.get("excluded_tasks", [])
    reportable_summary = summarize_models(partitions.reportable)
    smoke_summary = summarize_models(partitions.smoke_only)

    lines = [
        "# FoMoH MIMIC Phase 1 Benchmark Summary",
        "",
        f"Source status generated: {payload.get('generated_at', 'unknown')}",
        "",
        "## Scope",
        "",
        f"Active MIMIC tasks: {payload.get('active_task_count', 'unknown')}.",
        f"Excluded tasks: {', '.join(excluded) if excluded else 'none'}.",
        "",
        "This summary separates reportable full-active evidence from smoke-only foundation-stack coverage.",
        "",
        "## Reportable Full-Active Results",
        "",
        "These rows use full active MIMIC labels or reportable full-outcome/active-phenotype baselines. MEDS-TAB rows marked `empty_predictions` are retained as package-output caveats, not fabricated metrics.",
        "",
    ]
    _table(
        lines,
        ["Model", "Rows", "Metric rows", "Mean AUROC", "Mean AUPRC", "Mean Brier", "Statuses", "Run types"],
        (
            [
                row["model"],
                str(row["rows"]),
                str(row["metric_rows"]),
                _format_value(row["mean_auroc"]),
                _format_value(row["mean_auprc"]),
                _format_value(row["mean_brier"]),
                row["statuses"],
                row["run_types"],
            ]
            for row in reportable_summary
        ),
    )
    lines.extend(["", "### Reportable Per-Task Metrics", ""])
    _metric_table(lines, partitions.reportable)
    lines.extend(
        [
            "",
            "## Smoke-Only CEHR/CORE Rows",
            "",
            "These rows prove CEHR-BERT, CEHR-GPT, and CORE-BEHRT package wiring, feature extraction, and probe execution. They are not benchmark claims because the cohorts are smoke-sized.",
            "",
        ]
    )
    _table(
        lines,
        ["Model", "Rows", "Metric rows", "Mean AUROC", "Mean AUPRC", "Mean Brier", "Statuses", "Run types"],
        (
            [
                row["model"],
                str(row["rows"]),
                str(row["metric_rows"]),
                _format_value(row["mean_auroc"]),
                _format_value(row["mean_auprc"]),
                _format_value(row["mean_brier"]),
                row["statuses"],
                row["run_types"],
            ]
            for row in smoke_summary
        ),
    )
    lines.extend(["", "### Smoke-Only Per-Task Metrics", ""])
    _metric_table(lines, partitions.smoke_only)
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Official MEDS-TAB emitted empty held-out prediction frames for celiac, CLL, osteoporosis, and pancreatic cancer; those tasks are recorded as `empty_predictions` with no fabricated metrics.",
            "- Some official MEDS-TAB phenotype rows have undefined AUROC because the emitted held-out labels are single-class; AUPRC, Brier, calibration, and workload metrics are still reported where mathematically defined.",
            "- CEHR-BERT, CEHR-GPT, and CORE-BEHRT all-task downstream rows are smoke-only. Six tasks per stack are `non_identifiable` because the smoke held-out slice is single-class.",
            "- `phenotypes/ischemic_stroke` is excluded for MIMIC Phase 1 by user instruction because the current simple MIMIC mapping is zero-positive; revisit it on another dataset or with a revised phenotype definition.",
            "- Mamba-Transport remains skipped by user instruction.",
            "",
            "## Evidence",
            "",
            "- Source matrix: `runs/fomoh_mimic/benchmark_scale_status.json` and `runs/fomoh_mimic/benchmark_scale_status.md`.",
            "- Runtime profile: `runs/fomoh_mimic/benchmark_profile.md`.",
            "- Active task manifest: `runs/fomoh_mimic/task_bundle_mimic_identifiable_manifest.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_phase1_summary(status_json: Path, output_md: Path) -> None:
    payload = json.loads(status_json.read_text())
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_phase1_summary(payload))

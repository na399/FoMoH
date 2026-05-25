from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def write_report(
    report_path: Path,
    *,
    resource_summary: dict,
    label_counts: dict,
    tokenization_summary: dict,
    feature_counts: dict,
    metrics: dict,
    training_summary: dict,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FoMoH-on-MIMIC Run Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Resource Summary",
        "",
        "```json",
        json.dumps(resource_summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Label Counts",
        "",
        "```json",
        json.dumps(label_counts, indent=2, sort_keys=True),
        "```",
        "",
        "## Tokenization Summary",
        "",
        "```json",
        json.dumps(tokenization_summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Feature Counts",
        "",
        "```json",
        json.dumps(feature_counts, indent=2, sort_keys=True),
        "```",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
        "",
        "## Training Summary",
        "",
        "```json",
        json.dumps(training_summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation",
        "",
        "This report is reportable only if every section above contains real paths and non-empty evidence. Smoke runs must not be compared as full benchmark results.",
    ]
    report_path.write_text("\n".join(lines) + "\n")


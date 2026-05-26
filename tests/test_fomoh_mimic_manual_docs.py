from pathlib import Path


def test_manual_pages_cover_data_agnostic_rerun_steps():
    manual_dir = Path("docs/manual")
    expected = [
        "00_overview.md",
        "01_data_and_tasks.md",
        "02_environments_and_exports.md",
        "03_smoke_runs.md",
        "04_full_active_benchmarks.md",
        "05_metrics_and_reporting.md",
        "06_troubleshooting.md",
    ]

    for filename in expected:
        assert (manual_dir / filename).exists(), filename

    overview = (manual_dir / "00_overview.md").read_text()
    for filename in expected[1:]:
        assert f"]({filename})" in overview

    combined = "\n".join((manual_dir / filename).read_text() for filename in expected)
    forbidden_fragments = [
        "MIMIC",
        "/home/natthawut",
        "ischemic_stroke",
        "patient_outcomes/death",
        "celiac",
        "pancreatic cancer",
        "Mamba-Transport remains skipped",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in combined

    full_active = (manual_dir / "04_full_active_benchmarks.md").read_text()
    assert "${DATASET_NAME}" in full_active
    assert "${MODEL_NAME}" in full_active
    assert "${TASK_MANIFEST}" in full_active

    reporting = (manual_dir / "05_metrics_and_reporting.md").read_text()
    assert "reportable" in reporting
    assert "smoke-only" in reporting
    assert "non-identifiable" in reporting
    assert "dataset-specific" in reporting

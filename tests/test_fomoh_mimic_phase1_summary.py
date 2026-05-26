from ehr_foundation_model_benchmark.fomoh_mimic.phase1_summary import partition_rows, render_phase1_summary


def test_phase1_summary_separates_reportable_from_smoke_rows():
    payload = {
        "active_task_count": 13,
        "excluded_tasks": ["phenotypes/ischemic_stroke"],
        "generated_at": "2026-05-26T00:00:00+00:00",
        "rows": [
            {
                "model": "hf_ehr_gpt",
                "task": "patient_outcomes/death",
                "status": "done",
                "run_type": "benchmark_full_active",
                "auroc": 0.8,
                "auprc": 0.4,
                "brier": 0.1,
                "ece": 0.02,
                "pr_lift": 2.0,
                "ppv_at_100": 0.5,
                "sensitivity_at_100": 0.6,
            },
            {
                "model": "cehrbert",
                "task": "patient_outcomes/death",
                "status": "non_identifiable",
                "run_type": "official_downstream_active_smoke",
            },
            {
                "model": "official_medstab_xgboost",
                "task": "phenotypes/celiac",
                "status": "empty_predictions",
                "run_type": "official_full_active_prediction_blocked",
            },
        ],
    }

    partitions = partition_rows(payload["rows"])
    assert [row["model"] for row in partitions.reportable] == ["hf_ehr_gpt", "official_medstab_xgboost"]
    assert [row["model"] for row in partitions.smoke_only] == ["cehrbert"]

    markdown = render_phase1_summary(payload)
    assert "## Reportable Full-Active Results" in markdown
    assert "## Smoke-Only CEHR/CORE Rows" in markdown
    assert "official_medstab_xgboost" in markdown
    assert "phenotypes/ischemic_stroke" in markdown
    assert "empty held-out prediction frames" in markdown

from pathlib import Path

import polars as pl

from ehr_foundation_model_benchmark.fomoh_mimic.medstab_metrics import latest_xgboost_trial
from ehr_foundation_model_benchmark.fomoh_mimic.medstab_metrics import summarize_prediction_frame


def test_latest_xgboost_trial_selects_newest_complete_trial(tmp_path: Path) -> None:
    old = tmp_path / "2026-01-01_00-00-00" / "sweep_results" / "old"
    new = tmp_path / "2026-01-01_00-00-01" / "sweep_results" / "new"
    incomplete = tmp_path / "2026-01-01_00-00-02" / "sweep_results" / "incomplete"
    for path in (old, new, incomplete):
        path.mkdir(parents=True)
    (old / "xgboost.json").write_text("{}")
    (old / "config.log").write_text("path: {}\n")
    (new / "xgboost.json").write_text("{}")
    (new / "config.log").write_text("path: {}\n")
    (incomplete / "xgboost.json").write_text("{}")

    assert latest_xgboost_trial(tmp_path) == new


def test_latest_xgboost_trial_returns_none_without_complete_trial(tmp_path: Path) -> None:
    assert latest_xgboost_trial(tmp_path) is None



def test_summarize_prediction_frame_returns_full_metrics() -> None:
    predictions = pl.DataFrame(
        {
            "boolean_value": [True, False, True, False],
            "predicted_boolean_probability": [0.9, 0.1, 0.8, 0.4],
        }
    )

    summary = summarize_prediction_frame(predictions)

    assert summary["status"] == "passed"
    assert summary["rows"] == 4
    assert summary["metrics"]["auroc"] == 1.0
    assert summary["metrics"]["auprc"] == 1.0
    assert summary["metrics"]["brier"] > 0.0


def test_summarize_prediction_frame_records_empty_predictions() -> None:
    predictions = pl.DataFrame(
        {
            "boolean_value": pl.Series([], dtype=pl.Boolean),
            "predicted_boolean_probability": pl.Series([], dtype=pl.Float64),
        }
    )

    summary = summarize_prediction_frame(predictions)

    assert summary == {
        "status": "empty_predictions",
        "rows": 0,
        "metrics": None,
        "blocker": "official MEDS-TAB XGBoost emitted zero held_out predictions",
    }

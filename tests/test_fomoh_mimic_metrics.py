import math

import polars as pl

from ehr_foundation_model_benchmark.fomoh_mimic.diagnostics import _js_divergence, _safe_metrics
from ehr_foundation_model_benchmark.fomoh_mimic.metrics import auprc, auroc, summarize_metrics
from ehr_foundation_model_benchmark.fomoh_mimic.probe import _feature_rows, fit_logistic_regression, predict_probabilities
from ehr_foundation_model_benchmark.fomoh_mimic.tabular_models import run_tabular_model
from ehr_foundation_model_benchmark.fomoh_mimic.tokenization import normalize_code_candidates


def test_metrics_rank_clear_signal():
    y_true = [False, False, True, True]
    y_score = [0.1, 0.2, 0.8, 0.9]

    assert auroc(y_true, y_score) == 1.0
    assert auprc(y_true, y_score) == 1.0
    summary = summarize_metrics(y_true, y_score)
    assert summary.pr_lift == 2.0
    assert summary.brier < 0.05


def test_probe_learns_simple_separable_signal():
    x = [[0.0], [0.1], [1.0], [1.2]]
    y = [False, False, True, True]
    bias, weights = fit_logistic_regression(x, y, epochs=300, learning_rate=0.3)
    scores = predict_probabilities(x, bias, weights)

    assert max(scores[:2]) < min(scores[2:])


def test_normalize_code_candidates_includes_interval_and_single_slash_forms():
    candidates = normalize_code_candidates("LOINC//123//start")

    assert "LOINC//123//start" in candidates
    assert "LOINC/123//start" in candidates
    assert "LOINC//123" in candidates
    assert "LOINC/123" in candidates
    assert not any(math.isnan(len(candidate)) for candidate in candidates)



def test_feature_rows_accepts_dense_x_columns():
    df = pl.DataFrame(
        {
            "x10": [10.0, 11.0],
            "x2": [2.0, 3.0],
            "x0": [0.0, 1.0],
            "boolean_value": [False, True],
        }
    )

    assert _feature_rows(df) == [[0.0, 2.0, 10.0], [1.0, 3.0, 11.0]]


def test_safe_metrics_marks_single_class_slices():
    df = pl.DataFrame(
        {
            "boolean_value": [False, False],
            "predicted_boolean_probability": [0.1, 0.2],
        }
    )

    summary = _safe_metrics(df)

    assert summary["status"] == "single_class"
    assert summary["positives"] == 0


def test_js_divergence_is_zero_for_identical_distributions():
    counts = {"A": 10, "B": 5}

    assert _js_divergence(counts, counts) == 0.0
    assert _js_divergence(counts, {"A": 5, "C": 10}) > 0.0


def test_run_tabular_model_writes_predictions(tmp_path):
    features_dir = tmp_path / "features"
    for split, xs, ys in (
        ("train", [[0.0], [0.1], [1.0], [1.2]], [False, False, True, True]),
        ("tuning", [[0.2], [1.1]], [False, True]),
        ("held_out", [[0.05], [1.3]], [False, True]),
    ):
        split_dir = features_dir / "death" / "features_with_label"
        split_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "subject_id": list(range(len(xs))),
                "prediction_time": ["2020-01-01"] * len(xs),
                "boolean_value": ys,
                "features": xs,
            }
        ).write_parquet(split_dir / f"{split}.parquet")

    metrics = run_tabular_model(features_dir, tmp_path / "probes", task="death", model_name="femr_count_lr")

    assert metrics["auroc"] == 1.0
    assert (
        tmp_path / "probes" / "death" / "femr_count_lr" / "test_predictions" / "predictions.parquet"
    ).exists()

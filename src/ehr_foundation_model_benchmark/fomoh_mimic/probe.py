from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from ehr_foundation_model_benchmark.fomoh_mimic.metrics import summarize_metrics


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def _sigmoid(value: float) -> float:
    return float(_sigmoid_array(np.asarray([value], dtype=np.float64))[0])


def _standardize(train_x: list[list[float]] | np.ndarray, test_x: list[list[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_x, dtype=np.float64)
    test = np.asarray(test_x, dtype=np.float64)
    if train.size == 0:
        return train, test
    means = train.mean(axis=0)
    stds = train.std(axis=0)
    stds[stds == 0.0] = 1.0
    return (train - means) / stds, (test - means) / stds


def fit_logistic_regression(
    features: list[list[float]] | np.ndarray,
    labels: list[bool] | np.ndarray,
    *,
    epochs: int = 600,
    learning_rate: float = 0.05,
    l2: float = 0.001,
) -> tuple[float, list[float]]:
    x = np.asarray(features, dtype=np.float64)
    if x.size == 0:
        raise ValueError("No training features provided")
    y = np.asarray(labels, dtype=np.float64)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    positives = float(y.sum())
    bias = math.log((positives + 0.5) / (float(y.shape[0]) - positives + 0.5))
    n = float(y.shape[0])
    for _ in range(epochs):
        pred = _sigmoid_array(bias + x @ weights)
        error = pred - y
        bias -= learning_rate * float(error.mean())
        weights -= learning_rate * (((x.T @ error) / n) + l2 * weights)
    return float(bias), weights.astype(float).tolist()


def predict_probabilities(features: list[list[float]] | np.ndarray, bias: float, weights: list[float] | np.ndarray) -> list[float]:
    x = np.asarray(features, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return _sigmoid_array(bias + x @ w).astype(float).tolist()


def _feature_rows(df: pl.DataFrame) -> list[list[float]]:
    if "features" in df.columns:
        return [list(map(float, row)) for row in df["features"].to_list()]
    feature_cols = sorted(
        [col for col in df.columns if col.startswith("x") and col[1:].isdigit()],
        key=lambda col: int(col[1:]),
    )
    if not feature_cols:
        raise ValueError("Expected a `features` column or dense x0..xN feature columns")
    return df.select(feature_cols).to_numpy().astype(float).tolist()


def run_probe(
    features_dir: Path,
    output_dir: Path,
    *,
    task: str,
    model_name: str = "count_lr",
) -> dict[str, float]:
    train = pl.concat(
        [
            pl.read_parquet(features_dir / task / "features_with_label" / "train.parquet"),
            pl.read_parquet(features_dir / task / "features_with_label" / "tuning.parquet"),
        ],
        how="vertical",
    )
    test = pl.read_parquet(features_dir / task / "features_with_label" / "held_out.parquet")
    train_x = _feature_rows(train)
    test_x = _feature_rows(test)
    train_y = [bool(value) for value in train["boolean_value"].to_list()]
    test_y = [bool(value) for value in test["boolean_value"].to_list()]
    train_x, test_x = _standardize(train_x, test_x)
    bias, weights = fit_logistic_regression(train_x, train_y)
    scores = predict_probabilities(test_x, bias, weights)
    metrics = asdict(summarize_metrics(test_y, scores))

    task_output_dir = output_dir / task / model_name
    prediction_dir = task_output_dir / "test_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "subject_id": test["subject_id"].to_list(),
            "prediction_time": test["prediction_time"].to_list(),
            "predicted_boolean_probability": scores,
            "predicted_boolean_value": [score >= 0.5 for score in scores],
            "boolean_value": test_y,
        }
    ).write_parquet(prediction_dir / "predictions.parquet")
    (task_output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics

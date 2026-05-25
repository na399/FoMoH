from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from ehr_foundation_model_benchmark.fomoh_mimic.metrics import summarize_metrics
from ehr_foundation_model_benchmark.fomoh_mimic.probe import (
    _feature_rows,
    _standardize,
    fit_logistic_regression,
    predict_probabilities,
)


def _load_splits(features_dir: Path, task: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    task_dir = features_dir / task / "features_with_label"
    train = pl.concat(
        [pl.read_parquet(task_dir / "train.parquet"), pl.read_parquet(task_dir / "tuning.parquet")],
        how="vertical",
    )
    test = pl.read_parquet(task_dir / "held_out.parquet")
    return train, test


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, model_name: str, seed: int) -> list[float]:
    if model_name == "femr_count_lr":
        bias, weights = fit_logistic_regression(train_x, train_y)
        return predict_probabilities(test_x, bias, weights)
    if model_name == "sklearn_histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, l2_regularization=0.01, random_state=seed)
        model.fit(train_x, train_y.astype(int))
        return model.predict_proba(test_x)[:, 1].astype(float).tolist()
    if model_name == "medstab_xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=8,
        )
        model.fit(train_x, train_y.astype(int))
        return model.predict_proba(test_x)[:, 1].astype(float).tolist()
    if model_name == "femr_lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=250,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=8,
            verbosity=-1,
        )
        model.fit(train_x, train_y.astype(int))
        return model.predict_proba(test_x)[:, 1].astype(float).tolist()
    raise ValueError(f"Unsupported tabular model: {model_name}")


def run_tabular_model(
    features_dir: Path,
    output_dir: Path,
    *,
    task: str,
    model_name: str,
    seed: int = 123,
) -> dict[str, float]:
    train, test = _load_splits(features_dir, task)
    train_x = np.asarray(_feature_rows(train), dtype=np.float64)
    test_x = np.asarray(_feature_rows(test), dtype=np.float64)
    train_y = np.asarray([bool(value) for value in train["boolean_value"].to_list()], dtype=bool)
    test_y = [bool(value) for value in test["boolean_value"].to_list()]
    if len(set(train_y.tolist())) < 2:
        raise ValueError(f"Task {task} has a single training class and cannot train {model_name}")
    train_x, test_x = _standardize(train_x, test_x)
    scores = _fit_predict(train_x, train_y, test_x, model_name, seed)
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

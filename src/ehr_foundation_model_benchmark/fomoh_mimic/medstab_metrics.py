from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import polars as pl


def latest_xgboost_trial(model_dir: Path) -> Path | None:
    """Return the newest MEDS-TAB XGBoost trial directory with model/config files."""
    candidates = [
        path
        for path in model_dir.glob("*/sweep_results/*")
        if (path / "xgboost.json").exists() and (path / "config.log").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "xgboost.json").stat().st_mtime)



from .metrics import summarize_metrics


def summarize_prediction_frame(predictions: pl.DataFrame) -> dict[str, object]:
    """Compute FoMoH metrics from official MEDS-TAB held-out predictions."""
    if predictions.height == 0:
        return {
            "status": "empty_predictions",
            "rows": 0,
            "metrics": None,
            "blocker": "official MEDS-TAB XGBoost emitted zero held_out predictions",
        }
    y_true = [bool(value) for value in predictions["boolean_value"].to_list()]
    y_score = [float(value) for value in predictions["predicted_boolean_probability"].to_list()]
    return {
        "status": "passed",
        "rows": predictions.height,
        "metrics": asdict(summarize_metrics(y_true, y_score)),
    }

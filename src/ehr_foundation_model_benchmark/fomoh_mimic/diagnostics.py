from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from ehr_foundation_model_benchmark.fomoh_mimic.metrics import summarize_metrics


MALE_CONCEPT_ID = 8507
FEMALE_CONCEPT_ID = 8532


def _safe_metrics(df: pl.DataFrame) -> dict[str, Any]:
    if df.height == 0:
        return {"status": "empty", "n": 0, "positives": 0}
    y = [bool(value) for value in df["boolean_value"].to_list()]
    positives = int(sum(y))
    if positives == 0 or positives == len(y):
        return {
            "status": "single_class",
            "n": len(y),
            "positives": positives,
            "prevalence": positives / len(y),
        }
    scores = [float(value) for value in df["predicted_boolean_probability"].to_list()]
    payload = asdict(summarize_metrics(y, scores))
    payload["status"] = "ok"
    payload["n"] = len(y)
    payload["positives"] = positives
    return payload


def _load_person(duckdb_path: Path) -> pl.DataFrame:
    import duckdb

    with duckdb.connect(str(duckdb_path), read_only=True) as conn:
        result = conn.execute(
            """
            SELECT person_id AS subject_id, gender_concept_id, year_of_birth
            FROM person
            """
        )
        return pl.DataFrame(result.fetchall(), schema=[column[0] for column in result.description], orient="row")


def _sex_label(gender_concept_id: int | None) -> str:
    if gender_concept_id == MALE_CONCEPT_ID:
        return "male"
    if gender_concept_id == FEMALE_CONCEPT_ID:
        return "female"
    return "unknown"


def _age_bin(age: int | None) -> str:
    if age is None:
        return "unknown"
    if age < 45:
        return "lt45"
    if age < 65:
        return "45_64"
    if age < 80:
        return "65_79"
    return "80_plus"


def _tertile_labels(values: list[float]) -> list[str]:
    if not values:
        return []
    series = pl.Series(values)
    q1 = float(series.quantile(1 / 3, interpolation="nearest"))
    q2 = float(series.quantile(2 / 3, interpolation="nearest"))
    labels = []
    for value in values:
        if value <= q1:
            labels.append("low")
        elif value <= q2:
            labels.append("mid")
        else:
            labels.append("high")
    return labels


def _with_subgroups(predictions: pl.DataFrame, features: pl.DataFrame, person: pl.DataFrame) -> pl.DataFrame:
    feature_view = features.select(
        [
            "subject_id",
            "prediction_time",
            "boolean_value",
            pl.col("features").list.get(0).alias("event_count"),
            pl.col("features").list.get(2).alias("history_days"),
        ]
    )
    df = predictions.join(
        feature_view,
        on=["subject_id", "prediction_time", "boolean_value"],
        how="left",
    ).join(person, on="subject_id", how="left")
    years = df["prediction_time"].dt.year().to_list()
    births = df["year_of_birth"].to_list()
    ages = [None if year is None or birth is None else int(year) - int(birth) for year, birth in zip(years, births)]
    sex = [_sex_label(value) for value in df["gender_concept_id"].to_list()]
    event_counts = [float(value or 0.0) for value in df["event_count"].to_list()]
    history_days = [float(value or 0.0) for value in df["history_days"].to_list()]
    return df.with_columns(
        [
            pl.Series("age_years", ages),
            pl.Series("age_bin", [_age_bin(age) for age in ages]),
            pl.Series("sex", sex),
            pl.Series("utilization_tertile", _tertile_labels(event_counts)),
            pl.Series("history_tertile", _tertile_labels(history_days)),
        ]
    )


def write_subgroup_diagnostics(
    predictions_root: Path,
    features_root: Path,
    duckdb_path: Path,
    output_json: Path,
    *,
    tasks: list[str],
    model_name: str = "count_lr",
) -> dict[str, Any]:
    person = _load_person(duckdb_path)
    report: dict[str, Any] = {}
    for task in tasks:
        prediction_path = predictions_root / task / model_name / "test_predictions" / "predictions.parquet"
        feature_path = features_root / task / "features_with_label" / "held_out.parquet"
        if not prediction_path.exists() or not feature_path.exists():
            report[task] = {
                "status": "missing_inputs",
                "prediction_path": str(prediction_path),
                "feature_path": str(feature_path),
            }
            continue
        df = _with_subgroups(pl.read_parquet(prediction_path), pl.read_parquet(feature_path), person)
        task_report: dict[str, Any] = {"overall": _safe_metrics(df)}
        for column in ("sex", "age_bin", "utilization_tertile", "history_tertile"):
            task_report[column] = {}
            for value in sorted(v for v in df[column].unique().to_list() if v is not None):
                task_report[column][str(value)] = _safe_metrics(df.filter(pl.col(column) == value))
        report[task] = task_report
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return report


def _distribution(counts: dict[str, int]) -> dict[str, float]:
    total = float(sum(counts.values()))
    if total == 0.0:
        return {}
    return {key: value / total for key, value in counts.items()}


def _kl_divergence(p: dict[str, float], m: dict[str, float]) -> float:
    value = 0.0
    for key, p_value in p.items():
        m_value = m.get(key, 0.0)
        if p_value > 0.0 and m_value > 0.0:
            value += p_value * math.log(p_value / m_value, 2)
    return value


def _js_divergence(p_counts: dict[str, int], q_counts: dict[str, int]) -> float:
    p = _distribution(p_counts)
    q = _distribution(q_counts)
    keys = set(p) | set(q)
    m = {key: 0.5 * p.get(key, 0.0) + 0.5 * q.get(key, 0.0) for key in keys}
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _code_counts(meds_dir: Path, split: str, max_events: int) -> dict[str, int]:
    paths = [str(path) for path in sorted((meds_dir / "data" / split).glob("*.parquet"))]
    df = pl.scan_parquet(paths).select("code").limit(max_events).collect()
    counts = df.group_by("code").len().sort("len", descending=True)
    return {str(row["code"]): int(row["len"]) for row in counts.iter_rows(named=True)}


def write_transport_diagnostics(
    meds_dir: Path,
    output_json: Path,
    *,
    max_events_per_split: int = 500_000,
) -> dict[str, Any]:
    counts = {split: _code_counts(meds_dir, split, max_events_per_split) for split in ("train", "tuning", "held_out")}
    train_codes = set(counts["train"])
    report: dict[str, Any] = {"max_events_per_split": max_events_per_split, "splits": {}, "comparisons": {}}
    for split, split_counts in counts.items():
        total = sum(split_counts.values())
        unseen = sum(value for code, value in split_counts.items() if code not in train_codes)
        report["splits"][split] = {
            "events": total,
            "unique_codes": len(split_counts),
            "top_codes": sorted(split_counts.items(), key=lambda item: item[1], reverse=True)[:20],
            "unseen_vs_train_events": unseen,
            "unseen_vs_train_fraction": (unseen / total) if total else 0.0,
        }
    for split in ("tuning", "held_out"):
        report["comparisons"][f"train_vs_{split}"] = {
            "js_divergence_bits": _js_divergence(counts["train"], counts[split])
        }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return report


def write_diagnostics_bundle(
    predictions_root: Path,
    features_root: Path,
    duckdb_path: Path,
    meds_dir: Path,
    output_json: Path,
    *,
    tasks: list[str],
    model_name: str = "count_lr",
    max_events_per_split: int = 500_000,
) -> dict[str, Any]:
    payload = {
        "subgroups": write_subgroup_diagnostics(
            predictions_root,
            features_root,
            duckdb_path,
            output_json.with_name(output_json.stem + "_subgroups.json"),
            tasks=tasks,
            model_name=model_name,
        ),
        "transport": write_transport_diagnostics(
            meds_dir,
            output_json.with_name(output_json.stem + "_transport.json"),
            max_events_per_split=max_events_per_split,
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return payload

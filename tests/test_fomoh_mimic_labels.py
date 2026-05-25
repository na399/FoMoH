import csv
from pathlib import Path
import datetime as dt

import polars as pl

from ehr_foundation_model_benchmark.fomoh_mimic.ehrshot_labels import export_task_to_ehrshot_csv
from ehr_foundation_model_benchmark.fomoh_mimic.features import (
    _feature_vector,
    _features_for_subject,
    _windowed_features_for_subject,
)
from ehr_foundation_model_benchmark.fomoh_mimic.labels import generate_outcome_labels_from_rows
from ehr_foundation_model_benchmark.fomoh_mimic.phenotypes import load_phenotype_concept_spec


def row(subject_id, time, code, visit_id=None):
    return {
        "subject_id": subject_id,
        "time": time,
        "code": code,
        "visit_occurrence_id": visit_id,
    }


def test_generate_outcome_labels_death_los_and_readmission():
    start = dt.datetime(2020, 1, 1, 8)
    end = dt.datetime(2020, 1, 10, 8)
    readmit_start = dt.datetime(2020, 1, 20, 8)
    readmit_end = dt.datetime(2020, 1, 22, 8)
    rows = [
        row(1, start - dt.timedelta(days=800), "SNOMED//old"),
        row(1, start, "Visit//9201//start", 10),
        row(1, end, "Visit//9201//end", 10),
        row(1, start + dt.timedelta(days=3), "MEDS_DEATH"),
        row(1, readmit_start, "Visit//9201//start", 11),
        row(1, readmit_end, "Visit//9201//end", 11),
        row(1, end + dt.timedelta(days=40), "SNOMED//future"),
    ]

    labels = generate_outcome_labels_from_rows(rows)
    by_task = {label.task: label for label in labels[:3]}

    assert by_task["death"].boolean_value is True
    assert by_task["long_los"].boolean_value is True
    assert by_task["readmission"].boolean_value is True
    assert by_task["death"].prediction_time == start + dt.timedelta(hours=48)


def test_generate_outcome_labels_requires_prior_history():
    start = dt.datetime(2020, 1, 1, 8)
    rows = [
        row(1, start - dt.timedelta(days=10), "SNOMED//recent"),
        row(1, start, "Visit//9201//start", 10),
        row(1, start + dt.timedelta(days=9), "Visit//9201//end", 10),
    ]

    assert generate_outcome_labels_from_rows(rows) == []



def test_export_task_to_ehrshot_csv(tmp_path):
    labels_dir = tmp_path / "labels"
    for split in ("train", "tuning", "held_out"):
        task_dir = labels_dir / "patient_outcomes" / "death"
        task_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "subject_id": [1],
                "prediction_time": [dt.datetime(2020, 1, 3, 8)],
                "boolean_value": [split == "held_out"],
            }
        ).write_parquet(task_dir / f"{split}.parquet")

    output_path = export_task_to_ehrshot_csv(labels_dir, "death", tmp_path / "ehrshot" / "death")
    with output_path.open() as f:
        rows = list(csv.DictReader(f))

    assert output_path.name == "all_labels.csv"
    assert len(rows) == 3
    assert rows[0] == {
        "patient_id": "1",
        "prediction_time": "2020-01-03T08:00",
        "value": "false",
        "label_type": "boolean",
    }
    assert rows[-1]["value"] == "true"


def test_load_phenotype_concept_spec_uses_primary_codeset():
    spec = load_phenotype_concept_spec(
        Path("src/ehr_foundation_model_benchmark/phenotypes/cohort_definitions/ami_case.json")
    )

    assert 4329847 in spec.include_concept_ids
    assert 314666 in spec.exclude_concept_ids
    assert 9201 not in spec.include_concept_ids


def test_vectorized_subject_features_match_reference_loop():
    start = dt.datetime(2020, 1, 1, 8)
    rows = [
        {"subject_id": 1, "time": start, "code": "SNOMED//1", "numeric_value": None},
        {"subject_id": 1, "time": start + dt.timedelta(days=1), "code": "LOINC//2", "numeric_value": 1.2},
        {"subject_id": 1, "time": start + dt.timedelta(days=2), "code": "LOINC//2", "numeric_value": None},
        {"subject_id": 1, "time": start + dt.timedelta(days=3), "code": "RxNorm//3", "numeric_value": None},
    ]
    labels = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "prediction_time": [start + dt.timedelta(days=1, hours=1), start + dt.timedelta(days=3)],
            "boolean_value": [False, True],
        }
    )

    observed = [row["features"] for row in _features_for_subject(pl.DataFrame(rows), labels)]
    expected = [_feature_vector(rows, value) for value in labels["prediction_time"].to_list()]

    assert observed == expected


def test_windowed_features_keep_recent_and_all_history_counts():
    start = dt.datetime(2020, 1, 1, 8)
    rows = [
        {"subject_id": 1, "time": start, "code": "SNOMED//1", "numeric_value": None},
        {"subject_id": 1, "time": start + dt.timedelta(days=50), "code": "LOINC//2", "numeric_value": 1.2},
        {"subject_id": 1, "time": start + dt.timedelta(days=760), "code": "RxNorm//3", "numeric_value": None},
    ]
    labels = pl.DataFrame(
        {
            "subject_id": [1],
            "prediction_time": [start + dt.timedelta(days=780)],
            "boolean_value": [True],
        }
    )

    feature = _windowed_features_for_subject(pl.DataFrame(rows), labels)[0]["features"]

    assert feature[:9] == _feature_vector(rows, labels["prediction_time"][0])
    assert len(feature) == 9 + 4 * 8
    assert feature[9:17] == [1.0, 0.0, 20.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert feature[-8:] == [3.0, 1.0, 780.0, 1.0, 1.0, 1.0, 0.0, 0.0]

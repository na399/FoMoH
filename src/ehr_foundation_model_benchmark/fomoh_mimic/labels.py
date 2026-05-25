from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl


ADMISSION_BASE_CODES = ("Visit//9201", "Visit//262")
DEATH_CODE = "MEDS_DEATH"
TASKS = ("death", "long_los", "readmission")
LABEL_COLUMNS = ("subject_id", "prediction_time", "boolean_value")


@dataclass(frozen=True)
class OutcomeLabel:
    task: str
    subject_id: int
    prediction_time: dt.datetime
    boolean_value: bool


@dataclass(frozen=True)
class Admission:
    subject_id: int
    start: dt.datetime
    end: dt.datetime
    base_code: str
    visit_occurrence_id: int | None = None


def _strip_interval_suffix(code: str) -> str:
    if code.endswith("//start"):
        return code[: -len("//start")]
    if code.endswith("//end"):
        return code[: -len("//end")]
    return code


def _is_start(code: str) -> bool:
    return code.endswith("//start")


def _is_end(code: str) -> bool:
    return code.endswith("//end")


def _pair_admissions(rows: list[dict]) -> list[Admission]:
    starts: dict[tuple[int, str, int | None], list[dt.datetime]] = defaultdict(list)
    ends: dict[tuple[int, str, int | None], list[dt.datetime]] = defaultdict(list)
    for row in rows:
        code = row["code"]
        base = _strip_interval_suffix(code)
        if base not in ADMISSION_BASE_CODES or row["time"] is None:
            continue
        key = (int(row["subject_id"]), base, row.get("visit_occurrence_id"))
        if _is_start(code):
            starts[key].append(row["time"])
        elif _is_end(code):
            ends[key].append(row["time"])

    admissions: list[Admission] = []
    for key, start_times in starts.items():
        end_times = sorted(ends.get(key, []))
        for start, end in zip(sorted(start_times), end_times):
            if end > start:
                admissions.append(
                    Admission(
                        subject_id=key[0],
                        start=start,
                        end=end,
                        base_code=key[1],
                        visit_occurrence_id=key[2],
                    )
                )
    return sorted(admissions, key=lambda value: (value.subject_id, value.start, value.end))


def generate_outcome_labels_from_rows(
    rows: Iterable[dict],
    *,
    prediction_hours_after_admission: float = 48.0,
    long_los_days: float = 7.0,
    min_prior_history_days: float = 730.0,
    readmission_days: float = 30.0,
) -> list[OutcomeLabel]:
    rows_by_subject: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_subject[int(row["subject_id"])].append(row)

    labels: list[OutcomeLabel] = []
    prediction_offset = dt.timedelta(hours=prediction_hours_after_admission)
    long_los = dt.timedelta(days=long_los_days)
    min_prior_history = dt.timedelta(days=min_prior_history_days)
    readmission_window = dt.timedelta(days=readmission_days)

    for subject_id, subject_rows in rows_by_subject.items():
        subject_rows = sorted(
            subject_rows,
            key=lambda row: (row["time"] is None, row["time"] or dt.datetime.min),
        )
        times = [row["time"] for row in subject_rows if row["time"] is not None]
        if not times:
            continue
        first_observed = min(times)
        last_observed = max(times)
        death_times = [
            row["time"]
            for row in subject_rows
            if row["code"] == DEATH_CODE and row["time"] is not None
        ]
        death_time = min(death_times) if death_times else None
        admissions = _pair_admissions(subject_rows)

        for idx, admission in enumerate(admissions):
            prediction_time = admission.start + prediction_offset
            if prediction_time >= admission.end:
                continue
            if death_time is not None and prediction_time >= death_time:
                continue
            if first_observed > prediction_time - min_prior_history:
                continue
            labels.append(
                OutcomeLabel(
                    "death",
                    subject_id,
                    prediction_time,
                    bool(death_time and admission.start <= death_time <= admission.end),
                )
            )
            labels.append(
                OutcomeLabel(
                    "long_los",
                    subject_id,
                    prediction_time,
                    admission.end - admission.start > long_los,
                )
            )

            readmission_prediction_time = admission.end
            if first_observed > readmission_prediction_time - min_prior_history:
                continue
            if last_observed < readmission_prediction_time + readmission_window:
                continue
            next_admissions = [
                other
                for other in admissions[idx + 1 :]
                if other.start > admission.end
                and other.start <= admission.end + readmission_window
                and other.start.date() != admission.end.date()
            ]
            labels.append(
                OutcomeLabel(
                    "readmission",
                    subject_id,
                    readmission_prediction_time,
                    bool(next_admissions),
                )
            )
    return labels


def _event_files(meds_dir: Path, split: str) -> list[Path]:
    return sorted((meds_dir / "data" / split).glob("*.parquet"))


def _load_file_rows(path: Path, remaining_subjects: int | None) -> tuple[list[dict], int]:
    df = pl.read_parquet(
        path, columns=["subject_id", "time", "code", "visit_occurrence_id"]
    )
    if remaining_subjects is not None:
        subjects = df["subject_id"].unique().head(remaining_subjects).to_list()
        df = df.filter(pl.col("subject_id").is_in(subjects))
        return df.to_dicts(), len(subjects)
    return df.to_dicts(), 0


def write_labels_by_split(
    meds_dir: Path,
    output_dir: Path,
    *,
    max_subjects_per_split: int | None = None,
) -> dict[str, dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {task: {} for task in TASKS}
    for split in ("train", "tuning", "held_out"):
        for task in TASKS:
            (output_dir / "patient_outcomes" / task).mkdir(parents=True, exist_ok=True)
        labels_by_task: dict[str, list[OutcomeLabel]] = {task: [] for task in TASKS}
        consumed_subjects = 0
        for event_file in _event_files(meds_dir, split):
            remaining = (
                None
                if max_subjects_per_split is None
                else max(max_subjects_per_split - consumed_subjects, 0)
            )
            if remaining == 0:
                break
            rows, n_subjects = _load_file_rows(event_file, remaining)
            consumed_subjects += n_subjects
            for label in generate_outcome_labels_from_rows(rows):
                labels_by_task[label.task].append(label)
        for task in TASKS:
            task_labels = labels_by_task[task]
            task_dir = output_dir / "patient_outcomes" / task
            df = pl.DataFrame(
                {
                    "subject_id": [label.subject_id for label in task_labels],
                    "prediction_time": [label.prediction_time for label in task_labels],
                    "boolean_value": [label.boolean_value for label in task_labels],
                },
                schema={
                    "subject_id": pl.Int64,
                    "prediction_time": pl.Datetime("us"),
                    "boolean_value": pl.Boolean,
                },
            )
            df.write_parquet(task_dir / f"{split}.parquet")
            counts[task][split] = len(df)
    return counts


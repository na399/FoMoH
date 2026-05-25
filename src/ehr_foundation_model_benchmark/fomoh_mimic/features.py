from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


DOMAIN_PREFIXES = {
    "dx_count": ("SNOMED//", "ICD9CM//", "ICD10CM//"),
    "med_count": ("RxNorm//",),
    "proc_count": ("ICD9Proc//", "ICD10PCS//", "CPT4//", "HCPCS//"),
    "lab_count": ("LOINC//",),
    "visit_count": ("Visit//",),
}
DOMAIN_NAMES = tuple(sorted(DOMAIN_PREFIXES))


def _feature_vector(rows: list[dict], prediction_time: Any) -> list[float]:
    counts = {name: 0 for name in DOMAIN_PREFIXES}
    event_count = 0
    numeric_count = 0
    first_time = None
    last_time = None
    unique_codes = set()
    for row in rows:
        event_time = row["time"]
        if event_time is None or event_time > prediction_time:
            continue
        event_count += 1
        unique_codes.add(row["code"])
        first_time = event_time if first_time is None or event_time < first_time else first_time
        last_time = event_time if last_time is None or event_time > last_time else last_time
        if row.get("numeric_value") is not None:
            numeric_count += 1
        for name, prefixes in DOMAIN_PREFIXES.items():
            if row["code"].startswith(prefixes):
                counts[name] += 1
                break
    history_days = ((last_time - first_time).days if first_time and last_time else 0)
    return [
        float(event_count),
        float(len(unique_codes)),
        float(history_days),
        float(numeric_count),
        *[float(counts[name]) for name in DOMAIN_NAMES],
    ]


def _datetime64_us(values: list[Any]) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[us]")


def _domain_index(code: str) -> int | None:
    for idx, name in enumerate(DOMAIN_NAMES):
        if code.startswith(DOMAIN_PREFIXES[name]):
            return idx
    return None


def _features_for_subject(event_df: pl.DataFrame, label_df: pl.DataFrame) -> list[dict[str, Any]]:
    if label_df.is_empty():
        return []
    label_rows = label_df.to_dicts()
    if event_df.is_empty():
        zero = [0.0] * (4 + len(DOMAIN_NAMES))
        return [
            {
                "subject_id": row["subject_id"],
                "prediction_time": row["prediction_time"],
                "boolean_value": row["boolean_value"],
                "features": zero,
            }
            for row in label_rows
        ]

    event_df = event_df.sort("time")
    event_times = _datetime64_us(event_df["time"].to_list())
    label_times = _datetime64_us(label_df["prediction_time"].to_list())
    cutoffs = np.searchsorted(event_times, label_times, side="right")

    codes = event_df["code"].to_list()
    first_positions: dict[str, int] = {}
    domain_events = np.zeros((len(codes), len(DOMAIN_NAMES)), dtype=np.float64)
    for idx, code in enumerate(codes):
        first_positions.setdefault(code, idx)
        domain_idx = _domain_index(code)
        if domain_idx is not None:
            domain_events[idx, domain_idx] = 1.0
    unique_first_positions = np.asarray(sorted(first_positions.values()), dtype=np.int64)

    numeric_values = event_df["numeric_value"].to_list()
    numeric_events = np.asarray([value is not None for value in numeric_values], dtype=np.float64)
    numeric_cumsum = np.concatenate(([0.0], np.cumsum(numeric_events)))
    domain_cumsum = np.vstack([np.zeros((1, len(DOMAIN_NAMES))), np.cumsum(domain_events, axis=0)])

    features: list[list[float]] = []
    for cutoff in cutoffs:
        event_count = float(cutoff)
        if cutoff == 0:
            history_days = 0.0
            unique_count = 0.0
        else:
            history_days = float((event_times[cutoff - 1] - event_times[0]) / np.timedelta64(1, "D"))
            unique_count = float(np.searchsorted(unique_first_positions, cutoff, side="left"))
        features.append(
            [
                event_count,
                unique_count,
                history_days,
                float(numeric_cumsum[cutoff]),
                *domain_cumsum[cutoff].astype(float).tolist(),
            ]
        )

    return [
        {
            "subject_id": row["subject_id"],
            "prediction_time": row["prediction_time"],
            "boolean_value": row["boolean_value"],
            "features": feature,
        }
        for row, feature in zip(label_rows, features)
    ]


WINDOW_DAYS = (30, 180, 730, None)


def _windowed_features_for_subject(event_df: pl.DataFrame, label_df: pl.DataFrame) -> list[dict[str, Any]]:
    if label_df.is_empty():
        return []
    label_rows = label_df.to_dicts()
    base_rows = _features_for_subject(event_df, label_df)
    if event_df.is_empty():
        zeros = [0.0] * (len(WINDOW_DAYS) * (3 + len(DOMAIN_NAMES)))
        return [dict(row, features=row["features"] + zeros) for row in base_rows]

    event_df = event_df.sort("time")
    event_times = _datetime64_us(event_df["time"].to_list())
    codes = event_df["code"].to_list()
    domain_events = np.zeros((len(codes), len(DOMAIN_NAMES)), dtype=np.float64)
    for idx, code in enumerate(codes):
        domain_idx = _domain_index(code)
        if domain_idx is not None:
            domain_events[idx, domain_idx] = 1.0
    numeric_values = event_df["numeric_value"].to_list()
    numeric_events = np.asarray([value is not None for value in numeric_values], dtype=np.float64)
    numeric_cumsum = np.concatenate(([0.0], np.cumsum(numeric_events)))
    domain_cumsum = np.vstack([np.zeros((1, len(DOMAIN_NAMES))), np.cumsum(domain_events, axis=0)])

    label_times = _datetime64_us(label_df["prediction_time"].to_list())
    cutoffs = np.searchsorted(event_times, label_times, side="right")
    enriched: list[dict[str, Any]] = []
    for base_row, row, label_time, cutoff in zip(base_rows, label_rows, label_times, cutoffs):
        window_features: list[float] = []
        for days in WINDOW_DAYS:
            if days is None:
                left = 0
            else:
                left_time = label_time - np.timedelta64(days, "D")
                left = int(np.searchsorted(event_times, left_time, side="left"))
            event_count = float(max(cutoff - left, 0))
            if cutoff > left:
                days_since_first = float((label_time - event_times[left]) / np.timedelta64(1, "D"))
            else:
                days_since_first = 0.0
            domain_counts = (domain_cumsum[cutoff] - domain_cumsum[left]).astype(float).tolist()
            numeric_count = float(numeric_cumsum[cutoff] - numeric_cumsum[left])
            window_features.extend([event_count, numeric_count, days_since_first, *domain_counts])
        enriched.append(
            {
                "subject_id": row["subject_id"],
                "prediction_time": row["prediction_time"],
                "boolean_value": row["boolean_value"],
                "features": base_row["features"] + window_features,
            }
        )
    return enriched


def write_count_features_for_labels(
    meds_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    *,
    task: str,
    label_family: str = "patient_outcomes",
    feature_set: str = "count",
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "tuning", "held_out"):
        labels = pl.read_parquet(labels_dir / label_family / task / f"{split}.parquet")
        if labels.is_empty():
            counts[split] = 0
            continue
        subjects = labels["subject_id"].unique().to_list()
        event_files = [str(path) for path in sorted((meds_dir / "data" / split).glob("*.parquet"))]
        events = (
            pl.scan_parquet(event_files)
            .select("subject_id", "time", "code", "numeric_value")
            .filter(pl.col("subject_id").is_in(subjects))
            .filter(pl.col("time").is_not_null())
            .collect()
        )
        events_by_subject = events.partition_by("subject_id", as_dict=True, maintain_order=False)
        labels_by_subject = labels.sort(["subject_id", "prediction_time"]).partition_by(
            "subject_id", as_dict=True, maintain_order=True
        )
        feature_rows: list[dict[str, Any]] = []
        for key, subject_labels in labels_by_subject.items():
            subject_id = key[0] if isinstance(key, tuple) else key
            subject_events = events_by_subject.get((subject_id,), events_by_subject.get(subject_id, pl.DataFrame()))
            feature_fn = _windowed_features_for_subject if feature_set == "windowed" else _features_for_subject
            feature_rows.extend(feature_fn(subject_events, subject_labels))
        split_dir = output_dir / task / "features_with_label"
        split_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(feature_rows).write_parquet(split_dir / f"{split}.parquet")
        counts[split] = len(feature_rows)
    return counts


def write_count_features_for_tasks(
    meds_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    *,
    tasks: list[str],
    label_family: str = "patient_outcomes",
    feature_set: str = "count",
) -> dict[str, dict[str, int]]:
    """Write count features for several tasks while scanning each MEDS split once."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {task: {} for task in tasks}
    for split in ("train", "tuning", "held_out"):
        labels_by_task: dict[str, pl.DataFrame] = {}
        subject_ids: set[int] = set()
        for task in tasks:
            labels = pl.read_parquet(labels_dir / label_family / task / f"{split}.parquet")
            labels_by_task[task] = labels
            if labels.is_empty():
                counts[task][split] = 0
            else:
                subject_ids.update(int(value) for value in labels["subject_id"].unique().to_list())
        if not subject_ids:
            continue
        event_files = [str(path) for path in sorted((meds_dir / "data" / split).glob("*.parquet"))]
        events = (
            pl.scan_parquet(event_files)
            .select("subject_id", "time", "code", "numeric_value")
            .filter(pl.col("subject_id").is_in(sorted(subject_ids)))
            .filter(pl.col("time").is_not_null())
            .collect()
        )
        events_by_subject = events.partition_by("subject_id", as_dict=True, maintain_order=False)
        for task, labels in labels_by_task.items():
            if labels.is_empty():
                continue
            labels_by_subject = labels.sort(["subject_id", "prediction_time"]).partition_by(
                "subject_id", as_dict=True, maintain_order=True
            )
            feature_rows: list[dict[str, Any]] = []
            for key, subject_labels in labels_by_subject.items():
                subject_id = key[0] if isinstance(key, tuple) else key
                subject_events = events_by_subject.get((subject_id,), events_by_subject.get(subject_id, pl.DataFrame()))
                feature_fn = _windowed_features_for_subject if feature_set == "windowed" else _features_for_subject
                feature_rows.extend(feature_fn(subject_events, subject_labels))
            split_dir = output_dir / task / "features_with_label"
            split_dir.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(feature_rows).write_parquet(split_dir / f"{split}.parquet")
            counts[task][split] = len(feature_rows)
    return counts

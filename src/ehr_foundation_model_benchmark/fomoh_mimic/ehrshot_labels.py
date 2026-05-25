from __future__ import annotations

import csv
from pathlib import Path

import polars as pl


def export_task_to_ehrshot_csv(labels_dir: Path, task: str, output_dir: Path) -> Path:
    """Export split parquet labels to the CSV shape consumed by hf_ehr EHRSHOT tools."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_labels.csv"
    rows: list[dict[str, str]] = []
    for split in ("train", "tuning", "held_out"):
        path = labels_dir / "patient_outcomes" / task / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pl.read_parquet(path)
        for row in df.iter_rows(named=True):
            prediction_time = row["prediction_time"]
            rows.append(
                {
                    "patient_id": str(int(row["subject_id"])),
                    "prediction_time": prediction_time.isoformat(timespec="minutes"),
                    "value": str(bool(row["boolean_value"])).lower(),
                    "label_type": "boolean",
                }
            )
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["patient_id", "prediction_time", "value", "label_type"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


DEFAULT_MEDS_DIR = Path("/home/natthawut/MEDS/mimiciv")
DEFAULT_MEDS_READER_DIR = Path("/home/natthawut/MEDS/mimiciv_reader")
DEFAULT_OMOP_DUCKDB = Path("/home/natthawut/MIMIC/data/mimiciv_omop.duckdb")
DEFAULT_HF_EHR_RUN = Path("/home/natthawut/hf_ehr/runs/gpt_small_clmbr_mimiciv_full")
DEFAULT_TOKENIZER_CONFIG = (
    DEFAULT_MEDS_DIR / "tokenizer_clmbr_smoke" / "tokenizer_config.json"
)
EXPECTED_SPLITS = ("train", "tuning", "held_out")


@dataclass(frozen=True)
class ResourceSummary:
    meds_dir: str
    meds_reader_dir: str
    split_counts: dict[str, int]
    subject_split_counts: dict[str, int]
    event_schema: dict[str, str]
    omop_duckdb_exists: bool
    checkpoint_paths: list[str]
    tokenizer_config_exists: bool


def _first_parquet(path: Path) -> Path:
    try:
        return next(path.glob("*.parquet"))
    except StopIteration as exc:
        raise FileNotFoundError(f"No parquet files under {path}") from exc


def summarize_resources(
    meds_dir: Path = DEFAULT_MEDS_DIR,
    meds_reader_dir: Path = DEFAULT_MEDS_READER_DIR,
    omop_duckdb: Path = DEFAULT_OMOP_DUCKDB,
    hf_ehr_run: Path = DEFAULT_HF_EHR_RUN,
    tokenizer_config: Path = DEFAULT_TOKENIZER_CONFIG,
) -> ResourceSummary:
    if not meds_dir.exists():
        raise FileNotFoundError(f"MEDS directory does not exist: {meds_dir}")
    if not meds_reader_dir.exists():
        raise FileNotFoundError(f"MEDS Reader directory does not exist: {meds_reader_dir}")

    split_counts: dict[str, int] = {}
    event_schema: dict[str, str] | None = None
    for split in EXPECTED_SPLITS:
        split_dir = meds_dir / "data" / split
        first_file = _first_parquet(split_dir)
        split_counts[split] = len(list(split_dir.glob("*.parquet")))
        if event_schema is None:
            event_schema = {
                key: str(value)
                for key, value in pl.read_parquet(first_file, n_rows=0).schema.items()
            }

    subject_splits = pl.read_parquet(meds_dir / "metadata" / "subject_splits.parquet")
    subject_split_counts = dict(
        zip(
            subject_splits.group_by("split").len().sort("split")["split"].to_list(),
            subject_splits.group_by("split").len().sort("split")["len"].to_list(),
        )
    )
    checkpoint_paths = (
        [str(path) for path in sorted((hf_ehr_run / "ckpts").glob("*.ckpt"))]
        if hf_ehr_run.exists()
        else []
    )
    return ResourceSummary(
        meds_dir=str(meds_dir),
        meds_reader_dir=str(meds_reader_dir),
        split_counts=split_counts,
        subject_split_counts=subject_split_counts,
        event_schema=event_schema or {},
        omop_duckdb_exists=omop_duckdb.exists(),
        checkpoint_paths=checkpoint_paths,
        tokenizer_config_exists=tokenizer_config.exists(),
    )


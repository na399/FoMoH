from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ehr_foundation_model_benchmark.fomoh_mimic.duckdb_source import DuckDBConnectionSpec, connect_duckdb_source


PHENOTYPE_NAMES = (
    "ami",
    "celiac",
    "cll",
    "htn",
    "ischemic_stroke",
    "masld",
    "osteoporosis",
    "pancreatic_cancer",
    "schizophrenia",
    "sle",
    "t2dm",
)


@dataclass(frozen=True)
class PhenotypeConceptSpec:
    name: str
    include_concept_ids: tuple[int, ...]
    exclude_concept_ids: tuple[int, ...]


def _primary_codeset_ids(payload: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for criterion in payload.get("PrimaryCriteria", {}).get("CriteriaList", []):
        for body in criterion.values():
            codeset_id = body.get("CodesetId")
            if codeset_id is not None:
                ids.add(int(codeset_id))
    return ids


def load_phenotype_concept_spec(path: Path) -> PhenotypeConceptSpec:
    payload = json.loads(path.read_text())
    primary_ids = _primary_codeset_ids(payload)
    include: list[int] = []
    exclude: list[int] = []
    for concept_set in payload.get("ConceptSets", []):
        if primary_ids and int(concept_set["id"]) not in primary_ids:
            continue
        for item in concept_set.get("expression", {}).get("items", []):
            concept = item.get("concept", {})
            concept_id = concept.get("CONCEPT_ID")
            if concept_id is None:
                continue
            if item.get("isExcluded"):
                exclude.append(int(concept_id))
            else:
                include.append(int(concept_id))
    name = path.name.removesuffix("_case.json")
    return PhenotypeConceptSpec(
        name=name,
        include_concept_ids=tuple(sorted(set(include))),
        exclude_concept_ids=tuple(sorted(set(exclude))),
    )


def _id_list(ids: tuple[int, ...]) -> str:
    if not ids:
        return "NULL"
    return ", ".join(str(value) for value in ids)


EVENT_TABLE_SPECS = {
    "condition_occurrence": (
        "condition_concept_id",
        "COALESCE(condition_start_datetime, CAST(condition_start_date AS TIMESTAMP))",
    ),
    "drug_exposure": (
        "drug_concept_id",
        "COALESCE(drug_exposure_start_datetime, CAST(drug_exposure_start_date AS TIMESTAMP))",
    ),
    "procedure_occurrence": (
        "procedure_concept_id",
        "COALESCE(procedure_datetime, CAST(procedure_date AS TIMESTAMP))",
    ),
    "measurement": (
        "measurement_concept_id",
        "COALESCE(measurement_datetime, CAST(measurement_date AS TIMESTAMP))",
    ),
    "observation": (
        "observation_concept_id",
        "COALESCE(observation_datetime, CAST(observation_date AS TIMESTAMP))",
    ),
}


def _case_event_unions(available_tables: set[str]) -> str:
    selects = []
    for table, (concept_col, time_expr) in EVENT_TABLE_SPECS.items():
        if table not in available_tables:
            continue
        selects.append(
            f"""    SELECT person_id, {time_expr} AS event_time, {concept_col} AS concept_id
    FROM {table}
    WHERE {concept_col} IN (SELECT concept_id FROM included_concepts)"""
        )
    if not selects:
        return "    SELECT NULL::BIGINT AS person_id, NULL::TIMESTAMP AS event_time, NULL::BIGINT AS concept_id WHERE FALSE"
    return "\n    UNION ALL\n".join(selects)


def phenotype_labels_sql(spec: PhenotypeConceptSpec, *, available_tables: set[str]) -> str:
    include_ids = _id_list(spec.include_concept_ids)
    exclude_ids = _id_list(spec.exclude_concept_ids)
    exclude_clause = (
        "AND concept_id NOT IN (SELECT concept_id FROM excluded_concepts)"
        if spec.exclude_concept_ids
        else ""
    )
    case_events = _case_event_unions(available_tables)
    return f"""
WITH included_concepts AS (
    SELECT descendant_concept_id AS concept_id
    FROM concept_ancestor
    WHERE ancestor_concept_id IN ({include_ids})
    UNION
    SELECT concept_id FROM concept WHERE concept_id IN ({include_ids})
),
excluded_concepts AS (
    SELECT descendant_concept_id AS concept_id
    FROM concept_ancestor
    WHERE ancestor_concept_id IN ({exclude_ids})
    UNION
    SELECT concept_id FROM concept WHERE concept_id IN ({exclude_ids})
),
case_events AS (
{case_events}
),
case_events_filtered AS (
    SELECT person_id, event_time
    FROM case_events
    WHERE event_time IS NOT NULL
    {exclude_clause}
),
eligible_visits AS (
    SELECT DISTINCT
        vo.person_id AS subject_id,
        vo.visit_start_datetime AS prediction_time
    FROM visit_occurrence vo
    JOIN observation_period op ON op.person_id = vo.person_id
    WHERE vo.visit_start_datetime IS NOT NULL
      AND vo.visit_start_datetime >= CAST(op.observation_period_start_date AS TIMESTAMP) + INTERVAL 730 DAY
      AND CAST(op.observation_period_end_date AS TIMESTAMP) >= vo.visit_start_datetime + INTERVAL 365 DAY
      AND EXISTS (
        SELECT 1
        FROM condition_occurrence co
        WHERE co.person_id = vo.person_id
          AND COALESCE(co.condition_start_datetime, CAST(co.condition_start_date AS TIMESTAMP))
              BETWEEN vo.visit_start_datetime - INTERVAL 730 DAY AND vo.visit_start_datetime
      )
)
SELECT
    ev.subject_id,
    ev.prediction_time,
    CASE WHEN EXISTS (
        SELECT 1
        FROM case_events_filtered ce
        WHERE ce.person_id = ev.subject_id
          AND ce.event_time > ev.prediction_time
          AND ce.event_time <= ev.prediction_time + INTERVAL 365 DAY
    ) THEN TRUE ELSE FALSE END AS boolean_value
FROM eligible_visits ev
WHERE NOT EXISTS (
    SELECT 1
    FROM case_events_filtered prior
    WHERE prior.person_id = ev.subject_id
      AND prior.event_time <= ev.prediction_time
)
"""


def _empty_label_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": [],
            "prediction_time": [],
            "boolean_value": [],
        },
        schema={
            "subject_id": pl.Int64,
            "prediction_time": pl.Datetime("us"),
            "boolean_value": pl.Boolean,
        },
    )


def _write_empty_task(task_dir: Path) -> None:
    for split in ("train", "tuning", "held_out"):
        _empty_label_frame().write_parquet(task_dir / f"{split}.parquet")


def write_simple_phenotype_labels_from_connection(
    conn: Any,
    subject_splits_path: Path,
    cohort_definitions_dir: Path,
    output_dir: Path,
    *,
    max_rows_per_task: int | None = None,
    max_negative_rows_per_task: int | None = None,
    tasks: list[str] | None = None,
) -> dict[str, dict[str, int | str]]:
    splits = pl.read_parquet(subject_splits_path)
    subject_col = "subject_id" if "subject_id" in splits.columns else "patient_id"
    splits = splits.select(
        [
            pl.col(subject_col).cast(pl.Int64).alias("subject_id"),
            pl.col("split").cast(pl.Utf8).str.replace("tuning", "tuning").alias("split"),
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    available_tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    summary: dict[str, dict[str, int | str]] = {}
    required_tables = {"concept", "concept_ancestor", "condition_occurrence", "visit_occurrence", "observation_period"}
    missing_required = sorted(required_tables - available_tables)
    for name in (tasks or list(PHENOTYPE_NAMES)):
        spec = load_phenotype_concept_spec(cohort_definitions_dir / f"{name}_case.json")
        task_dir = output_dir / "phenotypes" / name
        task_dir.mkdir(parents=True, exist_ok=True)
        if missing_required:
            _write_empty_task(task_dir)
            summary[name] = {
                "status": "blocked_missing_tables",
                "missing_tables": ",".join(missing_required),
                "train": 0,
                "tuning": 0,
                "held_out": 0,
            }
            continue
        if not spec.include_concept_ids:
            _write_empty_task(task_dir)
            summary[name] = {"status": "skipped_no_case_concepts", "train": 0, "tuning": 0, "held_out": 0}
            continue
        sql = phenotype_labels_sql(spec, available_tables=available_tables)
        if max_negative_rows_per_task is not None:
            sql = f"""
WITH labels AS (
{sql}
), ranked AS (
    SELECT *, row_number() OVER (PARTITION BY boolean_value ORDER BY subject_id, prediction_time) AS rn
    FROM labels
)
SELECT subject_id, prediction_time, boolean_value
FROM ranked
WHERE boolean_value = TRUE OR (boolean_value = FALSE AND rn <= {int(max_negative_rows_per_task)})
"""
        elif max_rows_per_task is not None:
            sql += f"\nLIMIT {int(max_rows_per_task)}"
        result = conn.execute(sql)
        columns = [column[0] for column in result.description]
        df = pl.DataFrame(result.fetchall(), schema=columns, orient="row")
        if df.is_empty():
            joined = df.with_columns(pl.lit("").alias("split"))
        else:
            joined = df.join(splits, on="subject_id", how="inner")
        task_counts: dict[str, int | str] = {"status": "generated"}
        for split in ("train", "tuning", "held_out"):
            split_df = joined.filter(pl.col("split") == split).select(
                ["subject_id", "prediction_time", "boolean_value"]
            )
            split_df.write_parquet(task_dir / f"{split}.parquet")
            task_counts[split] = split_df.height
            task_counts[f"{split}_positives"] = int(split_df["boolean_value"].sum()) if not split_df.is_empty() else 0
        summary[name] = task_counts
    return summary


def write_simple_phenotype_labels_from_spec(
    duckdb_spec: DuckDBConnectionSpec,
    subject_splits_path: Path,
    cohort_definitions_dir: Path,
    output_dir: Path,
    *,
    max_rows_per_task: int | None = None,
    max_negative_rows_per_task: int | None = None,
    tasks: list[str] | None = None,
) -> dict[str, dict[str, int | str]]:
    with connect_duckdb_source(duckdb_spec) as conn:
        return write_simple_phenotype_labels_from_connection(
            conn,
            subject_splits_path,
            cohort_definitions_dir,
            output_dir,
            max_rows_per_task=max_rows_per_task,
            max_negative_rows_per_task=max_negative_rows_per_task,
            tasks=tasks,
        )


def write_simple_phenotype_labels(
    duckdb_path: Path,
    subject_splits_path: Path,
    cohort_definitions_dir: Path,
    output_dir: Path,
    *,
    max_rows_per_task: int | None = None,
    max_negative_rows_per_task: int | None = None,
    tasks: list[str] | None = None,
) -> dict[str, dict[str, int | str]]:
    return write_simple_phenotype_labels_from_spec(
        DuckDBConnectionSpec(path=duckdb_path, public_path=str(duckdb_path)),
        subject_splits_path,
        cohort_definitions_dir,
        output_dir,
        max_rows_per_task=max_rows_per_task,
        max_negative_rows_per_task=max_negative_rows_per_task,
        tasks=tasks,
    )

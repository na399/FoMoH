from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ehr_foundation_model_benchmark.fomoh_mimic.smoke_infra import write_json


CEHR_TABLES = (
    "condition_occurrence",
    "procedure_occurrence",
    "drug_exposure",
    "person",
    "visit_occurrence",
    "death",
    "observation_period",
)
CORE_EVENT_TABLES = ("condition_occurrence", "procedure_occurrence", "drug_exposure")
CEHR_PARQUET_TABLES = CEHR_TABLES + ("concept",)


def _quote_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _copy_query_to_csv(conn: Any, sql: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(f"COPY ({sql}) TO '{_quote_path(output_path)}' (HEADER, DELIMITER ',')")


def _copy_query_to_parquet(conn: Any, sql: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(f"COPY ({sql}) TO '{_quote_path(output_path)}' (FORMAT PARQUET)")


def _selected_persons_cte(max_persons: int) -> str:
    return f"""
WITH selected_persons AS (
    SELECT person_id
    FROM person
    WHERE person_id IS NOT NULL
    ORDER BY person_id
    LIMIT {int(max_persons)}
)
"""


def _cehr_table_sql(table: str, max_persons: int) -> str:
    if table == "concept":
        return "SELECT * FROM concept"
    cte = _selected_persons_cte(max_persons)
    if table == "person":
        return cte + "SELECT p.* FROM person p JOIN selected_persons s USING (person_id)"
    if table == "death":
        return cte + "SELECT d.* FROM death d JOIN selected_persons s USING (person_id)"
    return cte + f"SELECT t.* FROM {table} t JOIN selected_persons s USING (person_id)"


def export_cehr_omop_smoke(duckdb_path: Path, output_dir: Path, *, max_persons: int = 512) -> dict[str, Any]:
    import duckdb

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"duckdb_path": str(duckdb_path), "output_dir": str(output_dir), "max_persons": max_persons, "tables": {}}
    with duckdb.connect(str(duckdb_path), read_only=True) as conn:
        available = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        for table in CEHR_TABLES:
            if table not in available:
                summary["tables"][table] = {"status": "missing"}
                continue
            output_path = output_dir / f"{table}.csv"
            _copy_query_to_csv(conn, _cehr_table_sql(table, max_persons), output_path)
            summary["tables"][table] = {"status": "exported", "path": str(output_path), "bytes": output_path.stat().st_size}
        for table in CEHR_PARQUET_TABLES:
            if table not in available:
                summary["tables"].setdefault(table, {"status": "missing"})
                continue
            output_path = output_dir / table / "part-00000.parquet"
            _copy_query_to_parquet(conn, _cehr_table_sql(table, max_persons), output_path)
            summary["tables"].setdefault(table, {"status": "exported"})
            summary["tables"][table]["parquet_path"] = str(output_path)
            summary["tables"][table]["parquet_bytes"] = output_path.stat().st_size
    return summary


def _core_event_sql(table: str, max_persons: int) -> str:
    concept_col = {
        "condition_occurrence": "condition_concept_id",
        "procedure_occurrence": "procedure_concept_id",
        "drug_exposure": "drug_concept_id",
    }[table]
    time_expr = {
        "condition_occurrence": "COALESCE(condition_start_datetime, CAST(condition_start_date AS TIMESTAMP))",
        "procedure_occurrence": "COALESCE(procedure_datetime, CAST(procedure_date AS TIMESTAMP))",
        "drug_exposure": "COALESCE(drug_exposure_start_datetime, CAST(drug_exposure_start_date AS TIMESTAMP))",
    }[table]
    cte = _selected_persons_cte(max_persons)
    return cte + f"""
SELECT
    t.person_id AS PID,
    {time_expr} AS TIMESTAMP,
    t.visit_occurrence_id AS ADMISSION_ID,
    CAST(t.{concept_col} AS VARCHAR) AS CONCEPT
FROM {table} t
JOIN selected_persons s USING (person_id)
WHERE {time_expr} IS NOT NULL
  AND t.{concept_col} IS NOT NULL
ORDER BY PID, TIMESTAMP
"""


def export_corebehrt_flat_smoke(duckdb_path: Path, output_dir: Path, *, max_persons: int = 512) -> dict[str, Any]:
    import duckdb

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"duckdb_path": str(duckdb_path), "output_dir": str(output_dir), "max_persons": max_persons, "tables": {}}
    with duckdb.connect(str(duckdb_path), read_only=True) as conn:
        patient_sql = _selected_persons_cte(max_persons) + """
SELECT
    p.person_id AS PID,
    CAST(COALESCE(p.birth_datetime, make_date(p.year_of_birth, COALESCE(p.month_of_birth, 1), COALESCE(p.day_of_birth, 1))) AS DATE) AS DATE_OF_BIRTH,
    CAST(p.gender_concept_id AS VARCHAR) AS GENDER,
    CAST(p.race_concept_id AS VARCHAR) AS RACE
FROM person p
JOIN selected_persons s USING (person_id)
ORDER BY PID
"""
        patients_path = output_dir / "patients_info.csv"
        _copy_query_to_csv(conn, patient_sql, patients_path)
        summary["tables"]["patients_info"] = {"status": "exported", "path": str(patients_path), "bytes": patients_path.stat().st_size}
        patient_alias_path = output_dir / "patient_format.csv"
        _copy_query_to_csv(conn, patient_sql, patient_alias_path)
        summary["tables"]["patient_format"] = {"status": "exported", "path": str(patient_alias_path), "bytes": patient_alias_path.stat().st_size}
        for table in CORE_EVENT_TABLES:
            output_path = output_dir / table / "part-00000.parquet"
            _copy_query_to_parquet(conn, _core_event_sql(table, max_persons), output_path)
            summary["tables"][table] = {"status": "exported", "path": str(output_path), "bytes": output_path.stat().st_size}
    return summary


def export_all_omop_smoke_layouts(duckdb_path: Path, output_root: Path, *, max_persons: int = 512) -> dict[str, Any]:
    summary = {
        "cehrbert": export_cehr_omop_smoke(duckdb_path, output_root / "cehrbert_omop", max_persons=max_persons),
        "cehrgpt": export_cehr_omop_smoke(duckdb_path, output_root / "cehrgpt_omop", max_persons=max_persons),
        "corebehrt": export_corebehrt_flat_smoke(duckdb_path, output_root / "corebehrt_flat", max_persons=max_persons),
    }
    write_json(output_root / "omop_smoke_export_summary.json", summary)
    return summary

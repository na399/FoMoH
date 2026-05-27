from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterator

import duckdb


@dataclass(frozen=True)
class DuckDBConnectionSpec:
    path: Path
    public_path: str
    schema: str | None = None
    encryption_key: str | None = None
    readonly: bool = True


def duckdb_spec_from_args(
    duckdb_path: Path | None = None,
    *,
    duckdb_env: str | None = None,
    schema: str | None = None,
    schema_env: str | None = None,
    encryption_key_env: str | None = None,
) -> DuckDBConnectionSpec:
    if duckdb_path is None:
        if not duckdb_env:
            raise ValueError("Either duckdb_path or duckdb_env is required")
        value = os.environ.get(duckdb_env)
        if not value:
            raise RuntimeError(f"Missing DuckDB path env var: {duckdb_env}")
        duckdb_path = Path(value)
        public_path = f"<env:{duckdb_env}>"
    else:
        public_path = str(duckdb_path)
    resolved_schema = schema
    if resolved_schema is None and schema_env:
        resolved_schema = os.environ.get(schema_env)
    encryption_key = os.environ.get(encryption_key_env) if encryption_key_env else None
    return DuckDBConnectionSpec(
        path=duckdb_path,
        public_path=public_path,
        schema=resolved_schema,
        encryption_key=encryption_key,
    )


def _escaped(value: str) -> str:
    return value.replace("'", "''")


def _looks_encrypted(exc: Exception) -> bool:
    text = str(exc).lower()
    return "encrypt" in text or "encrypted" in text or "encryption" in text


@contextmanager
def connect_duckdb_source(spec: DuckDBConnectionSpec) -> Iterator[duckdb.DuckDBPyConnection]:
    conn: duckdb.DuckDBPyConnection | None = None
    try:
        try:
            conn = duckdb.connect(str(spec.path), read_only=spec.readonly)
        except duckdb.Error as exc:
            if not _looks_encrypted(exc):
                raise
            if not spec.encryption_key:
                raise RuntimeError("DuckDB source appears encrypted but no encryption key env var was provided") from exc
            conn = duckdb.connect(":memory:")
            options = ["READ_ONLY"] if spec.readonly else []
            options.append(f"ENCRYPTION_KEY '{_escaped(spec.encryption_key)}'")
            conn.execute(f"ATTACH '{_escaped(str(spec.path))}' AS omop ({', '.join(options)});")
            conn.execute("USE omop;")
        if spec.schema:
            conn.execute(f"SET schema '{_escaped(spec.schema)}';")
        yield conn
    finally:
        if conn is not None:
            conn.close()

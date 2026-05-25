from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from ehr_foundation_model_benchmark.fomoh_mimic.labels import TASKS as OUTCOME_TASKS
from ehr_foundation_model_benchmark.fomoh_mimic.phenotypes import PHENOTYPE_NAMES


REPO_TEMP_ROOT = Path("temp/fomoh_mimic")
RUN_ROOT = Path("runs/fomoh_mimic")
DEFAULT_TMPFS_ROOT = Path(os.environ.get("FOMOH_MIMIC_TMPFS_ROOT", "/dev/shm/fomoh_mimic"))
REQUIRED_ATHENA_TABLES = ("concept", "concept_ancestor", "concept_relationship")
ATHENA_TABLES = (
    "concept",
    "concept_ancestor",
    "concept_class",
    "concept_relationship",
    "concept_synonym",
    "domain",
    "drug_strength",
    "relationship",
    "vocabulary",
)
SPLITS = ("train", "tuning", "held_out")


class TaskStatus(StrEnum):
    IDENTIFIABLE = "identifiable"
    NON_IDENTIFIABLE = "non_identifiable"
    MISSING = "missing"
    SINGLE_CLASS = "single_class"


@dataclass(frozen=True)
class ModelEnvManifest:
    model_name: str
    venv_path: Path
    python_path: Path
    python_version: str
    install_command: list[str]
    package_versions: dict[str, str]
    import_smoke: dict[str, Any]
    tmpfs_root: Path
    log_path: Path
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.UTC).isoformat())


@dataclass(frozen=True)
class TaskSummary:
    family: str
    task: str
    status: TaskStatus
    rows: int
    positives: int
    split_rows: dict[str, int]
    split_positives: dict[str, int]


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    log_path: Path
    status: str


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    return str(value)


def materialize_manifest(manifest: ModelEnvManifest, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True, default=_json_default))
    return output_path


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return path


def ensure_tmpfs_root(tmpfs_root: Path = DEFAULT_TMPFS_ROOT) -> Path:
    tmpfs_root.mkdir(parents=True, exist_ok=True)
    return tmpfs_root


def ensure_repo_pointer(pointer_path: Path, target_path: Path) -> None:
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    if pointer_path.exists() or pointer_path.is_symlink():
        if pointer_path.is_symlink() and Path(os.readlink(pointer_path)) == target_path:
            return
        if pointer_path.resolve() == target_path.resolve():
            return
        raise FileExistsError(f"Refusing to replace existing path: {pointer_path}")
    pointer_path.symlink_to(target_path, target_is_directory=True)


def run_logged(command: list[str], log_path: Path, *, cwd: Path | None = None, env: dict[str, str] | None = None) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return CommandResult(
        command=command,
        returncode=completed.returncode,
        log_path=log_path,
        status="passed" if completed.returncode == 0 else "failed",
    )


def package_versions(python_path: Path, packages: Iterable[str]) -> dict[str, str]:
    script = (
        "import importlib.metadata as m, json; "
        "pkgs = " + repr(list(packages)) + "; "
        "print(json.dumps({p: (m.version(p) if p in {d.metadata['Name'] for d in m.distributions()} else 'missing') for p in pkgs}, sort_keys=True))"
    )
    completed = subprocess.run([str(python_path), "-c", script], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {package: "unknown" for package in packages}
    return json.loads(completed.stdout)


def validate_athena_csv_export(export_dir: Path, required_tables: Iterable[str] = REQUIRED_ATHENA_TABLES) -> dict[str, Any]:
    tables: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    empty: list[str] = []
    for table in required_tables:
        path = export_dir / f"{table}.csv"
        if not path.exists():
            missing.append(table)
            continue
        size = path.stat().st_size
        if size == 0:
            empty.append(table)
        tables[table] = {"path": str(path), "bytes": size}
    status = "valid" if not missing and not empty else "invalid"
    return {"status": status, "export_dir": str(export_dir), "tables": tables, "missing": missing, "empty": empty}


def export_athena_vocab_csvs(duckdb_path: Path, output_dir: Path, *, tables: Iterable[str] = ATHENA_TABLES) -> dict[str, Any]:
    import duckdb

    output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, dict[str, Any]] = {}
    with duckdb.connect(str(duckdb_path), read_only=True) as conn:
        available = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        for table in tables:
            if table not in available:
                continue
            output_path = output_dir / f"{table}.csv"
            escaped_path = str(output_path).replace("'", "''")
            conn.execute(f"COPY (SELECT * FROM {table}) TO '{escaped_path}' (HEADER, DELIMITER ',')")
            exported[table] = {"path": str(output_path), "bytes": output_path.stat().st_size}
    validation = validate_athena_csv_export(output_dir)
    return {"duckdb_path": str(duckdb_path), "exported": exported, "validation": validation}


def _empty_task_summary(family: str, task: str) -> TaskSummary:
    return TaskSummary(
        family=family,
        task=task,
        status=TaskStatus.MISSING,
        rows=0,
        positives=0,
        split_rows={split: 0 for split in SPLITS},
        split_positives={split: 0 for split in SPLITS},
    )


def _read_split_counts(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    df = pl.read_parquet(path, columns=["boolean_value"])
    if df.is_empty():
        return 0, 0
    positives = int(df.select(pl.col("boolean_value").cast(pl.Int64).sum()).item())
    return df.height, positives


def summarize_task_bundle(labels_dir: Path) -> dict[str, TaskSummary]:
    summary: dict[str, TaskSummary] = {}
    expected = [("patient_outcomes", task) for task in OUTCOME_TASKS] + [
        ("phenotypes", task) for task in PHENOTYPE_NAMES
    ]
    for family, task in expected:
        task_dir = labels_dir / family / task
        if not task_dir.exists():
            summary[f"{family}/{task}"] = _empty_task_summary(family, task)
            continue
        split_rows: dict[str, int] = {}
        split_positives: dict[str, int] = {}
        for split in SPLITS:
            rows, positives = _read_split_counts(task_dir / f"{split}.parquet")
            split_rows[split] = rows
            split_positives[split] = positives
        rows = sum(split_rows.values())
        positives = sum(split_positives.values())
        if rows == 0:
            status = TaskStatus.MISSING
        elif positives == 0:
            status = TaskStatus.NON_IDENTIFIABLE
        elif positives == rows:
            status = TaskStatus.SINGLE_CLASS
        else:
            status = TaskStatus.IDENTIFIABLE
        summary[f"{family}/{task}"] = TaskSummary(
            family=family,
            task=task,
            status=status,
            rows=rows,
            positives=positives,
            split_rows=split_rows,
            split_positives=split_positives,
        )
    return summary


def write_task_bundle_manifest(labels_dir: Path, output_path: Path) -> dict[str, Any]:
    summary = summarize_task_bundle(labels_dir)
    payload = {key: asdict(value) for key, value in summary.items()}
    write_json(output_path, payload)
    return payload


def create_uv_venv(
    name: str,
    python_version: str,
    *,
    tmpfs_root: Path = DEFAULT_TMPFS_ROOT,
    repo_venv_pointer_root: Path = REPO_TEMP_ROOT / "venvs",
    log_dir: Path = REPO_TEMP_ROOT / "env_logs",
) -> CommandResult:
    tmpfs_root = ensure_tmpfs_root(tmpfs_root)
    venv_path = tmpfs_root / "venvs" / name
    result = run_logged(
        ["uv", "venv", "--python", python_version, str(venv_path)],
        log_dir / f"{name}_venv.log",
    )
    if result.returncode == 0:
        ensure_repo_pointer(repo_venv_pointer_root / name, venv_path)
    return result


def copy_text_files(src_dir: Path, dst_dir: Path, names: Iterable[str]) -> dict[str, str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name in names:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        shutil.copy2(src, dst)
        copied[name] = str(dst)
    return copied

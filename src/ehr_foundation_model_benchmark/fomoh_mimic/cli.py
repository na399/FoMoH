from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ehr_foundation_model_benchmark.fomoh_mimic.diagnostics import write_diagnostics_bundle
from ehr_foundation_model_benchmark.fomoh_mimic.ehrshot_labels import export_task_to_ehrshot_csv
from ehr_foundation_model_benchmark.fomoh_mimic.features import write_count_features_for_labels, write_count_features_for_tasks
from ehr_foundation_model_benchmark.fomoh_mimic.labels import write_labels_by_split
from ehr_foundation_model_benchmark.fomoh_mimic.omop_exports import export_all_omop_smoke_layouts
from ehr_foundation_model_benchmark.fomoh_mimic.phenotypes import write_simple_phenotype_labels
from ehr_foundation_model_benchmark.fomoh_mimic.phase1_summary import write_phase1_summary
from ehr_foundation_model_benchmark.fomoh_mimic.probe import run_probe
from ehr_foundation_model_benchmark.fomoh_mimic.report import write_report
from ehr_foundation_model_benchmark.fomoh_mimic.smoke_infra import (
    create_uv_venv,
    export_athena_vocab_csvs,
    validate_athena_csv_export,
    write_task_bundle_manifest,
)
from ehr_foundation_model_benchmark.fomoh_mimic.resources import (
    DEFAULT_MEDS_DIR,
    DEFAULT_TOKENIZER_CONFIG,
    summarize_resources,
)
from ehr_foundation_model_benchmark.fomoh_mimic.tabular_models import run_tabular_model
from ehr_foundation_model_benchmark.fomoh_mimic.tokenization import summarize_tokenization_coverage


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="FoMoH-on-MIMIC workflow utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-resources")
    validate.add_argument("--output-json", type=Path, required=True)

    labels = subparsers.add_parser("generate-labels")
    labels.add_argument("--meds-dir", type=Path, default=DEFAULT_MEDS_DIR)
    labels.add_argument("--output-dir", type=Path, required=True)
    labels.add_argument("--max-subjects-per-split", type=int, default=None)
    labels.add_argument("--summary-json", type=Path, required=True)

    token = subparsers.add_parser("tokenization-smoke")
    token.add_argument("--meds-dir", type=Path, default=DEFAULT_MEDS_DIR)
    token.add_argument("--tokenizer-config", type=Path, default=DEFAULT_TOKENIZER_CONFIG)
    token.add_argument("--output-json", type=Path, required=True)
    token.add_argument("--min-map-fraction", type=float, default=0.01)

    phenotypes = subparsers.add_parser("generate-phenotype-labels")
    phenotypes.add_argument("--duckdb", type=Path, default=Path("/home/natthawut/MIMIC/data/mimiciv_omop.duckdb"))
    phenotypes.add_argument("--subject-splits", type=Path, default=DEFAULT_MEDS_DIR / "metadata" / "subject_splits.parquet")
    phenotypes.add_argument("--cohort-definitions-dir", type=Path, default=Path("src/ehr_foundation_model_benchmark/phenotypes/cohort_definitions"))
    phenotypes.add_argument("--output-dir", type=Path, required=True)
    phenotypes.add_argument("--max-rows-per-task", type=int, default=None)
    phenotypes.add_argument("--max-negative-rows-per-task", type=int, default=None)
    phenotypes.add_argument("--tasks", nargs="+", default=None)
    phenotypes.add_argument("--summary-json", type=Path, required=True)

    features = subparsers.add_parser("count-features")
    features.add_argument("--meds-dir", type=Path, default=DEFAULT_MEDS_DIR)
    features.add_argument("--labels-dir", type=Path, required=True)
    features.add_argument("--output-dir", type=Path, required=True)
    features.add_argument("--task", default="death")
    features.add_argument("--label-family", default="patient_outcomes")
    features.add_argument("--summary-json", type=Path, required=True)
    features.add_argument("--feature-set", choices=["count", "windowed"], default="count")

    multi_features = subparsers.add_parser("count-features-batch")
    multi_features.add_argument("--meds-dir", type=Path, default=DEFAULT_MEDS_DIR)
    multi_features.add_argument("--labels-dir", type=Path, required=True)
    multi_features.add_argument("--output-dir", type=Path, required=True)
    multi_features.add_argument("--tasks", nargs="+", required=True)
    multi_features.add_argument("--label-family", default="patient_outcomes")
    multi_features.add_argument("--summary-json", type=Path, required=True)
    multi_features.add_argument("--feature-set", choices=["count", "windowed"], default="count")

    probe = subparsers.add_parser("probe")
    probe.add_argument("--features-dir", type=Path, required=True)
    probe.add_argument("--output-dir", type=Path, required=True)
    probe.add_argument("--task", default="death")
    probe.add_argument("--metrics-json", type=Path, required=True)

    tabular = subparsers.add_parser("tabular-model")
    tabular.add_argument("--features-dir", type=Path, required=True)
    tabular.add_argument("--output-dir", type=Path, required=True)
    tabular.add_argument("--task", required=True)
    tabular.add_argument("--model-name", required=True, choices=["femr_count_lr", "sklearn_histgb", "medstab_xgboost", "femr_lightgbm"])
    tabular.add_argument("--metrics-json", type=Path, required=True)
    tabular.add_argument("--seed", type=int, default=123)

    diagnostics = subparsers.add_parser("diagnostics")
    diagnostics.add_argument("--predictions-root", type=Path, required=True)
    diagnostics.add_argument("--features-root", type=Path, required=True)
    diagnostics.add_argument("--duckdb", type=Path, default=Path("/home/natthawut/MIMIC/data/mimiciv_omop.duckdb"))
    diagnostics.add_argument("--meds-dir", type=Path, default=DEFAULT_MEDS_DIR)
    diagnostics.add_argument("--tasks", nargs="+", required=True)
    diagnostics.add_argument("--model-name", default="count_lr")
    diagnostics.add_argument("--max-events-per-split", type=int, default=500_000)
    diagnostics.add_argument("--output-json", type=Path, required=True)

    athena_export = subparsers.add_parser("export-athena-vocab")
    athena_export.add_argument("--duckdb", type=Path, default=Path("/home/natthawut/omop/omop_vocab_v20260227.duckdb"))
    athena_export.add_argument("--output-dir", type=Path, required=True)
    athena_export.add_argument("--summary-json", type=Path, required=True)

    athena_validate = subparsers.add_parser("validate-athena-vocab")
    athena_validate.add_argument("--export-dir", type=Path, required=True)
    athena_validate.add_argument("--output-json", type=Path, required=True)

    omop_export = subparsers.add_parser("export-omop-smoke-layouts")
    omop_export.add_argument("--duckdb", type=Path, default=Path("/home/natthawut/MIMIC/data/mimiciv_omop.duckdb"))
    omop_export.add_argument("--output-root", type=Path, required=True)
    omop_export.add_argument("--max-persons", type=int, default=512)
    omop_export.add_argument("--summary-json", type=Path, required=True)

    task_manifest = subparsers.add_parser("task-bundle-manifest")
    task_manifest.add_argument("--labels-dir", type=Path, required=True)
    task_manifest.add_argument("--output-json", type=Path, required=True)

    create_venv = subparsers.add_parser("create-venv")
    create_venv.add_argument("--name", required=True)
    create_venv.add_argument("--python-version", required=True)
    create_venv.add_argument("--tmpfs-root", type=Path, default=Path("/dev/shm/fomoh_mimic"))
    create_venv.add_argument("--summary-json", type=Path, required=True)

    ehrshot = subparsers.add_parser("export-ehrshot-labels")
    ehrshot.add_argument("--labels-dir", type=Path, required=True)
    ehrshot.add_argument("--task", required=True)
    ehrshot.add_argument("--output-dir", type=Path, required=True)

    report = subparsers.add_parser("report")
    report.add_argument("--output-md", type=Path, required=True)
    report.add_argument("--resource-json", type=Path, required=True)
    report.add_argument("--labels-json", type=Path, required=True)
    report.add_argument("--tokenization-json", type=Path, required=True)
    report.add_argument("--features-json", type=Path, required=True)
    report.add_argument("--metrics-json", type=Path, required=True)
    report.add_argument("--training-json", type=Path, required=True)

    phase1_summary = subparsers.add_parser("phase1-summary")
    phase1_summary.add_argument("--status-json", type=Path, default=Path("runs/fomoh_mimic/benchmark_scale_status.json"))
    phase1_summary.add_argument("--output-md", type=Path, default=Path("runs/fomoh_mimic/mimic_phase1_benchmark_summary.md"))

    args = parser.parse_args()
    if args.command == "validate-resources":
        _write_json(args.output_json, asdict(summarize_resources()))
    elif args.command == "generate-labels":
        _write_json(
            args.summary_json,
            write_labels_by_split(
                args.meds_dir,
                args.output_dir,
                max_subjects_per_split=args.max_subjects_per_split,
            ),
        )
    elif args.command == "tokenization-smoke":
        summary = summarize_tokenization_coverage(args.meds_dir, args.tokenizer_config)
        payload = asdict(summary)
        _write_json(args.output_json, payload)
        if summary.map_fraction < args.min_map_fraction:
            raise RuntimeError(
                f"Tokenization map fraction {summary.map_fraction:.4f} is below {args.min_map_fraction:.4f}"
            )
    elif args.command == "generate-phenotype-labels":
        _write_json(
            args.summary_json,
            write_simple_phenotype_labels(
                args.duckdb,
                args.subject_splits,
                args.cohort_definitions_dir,
                args.output_dir,
                max_rows_per_task=args.max_rows_per_task,
                max_negative_rows_per_task=args.max_negative_rows_per_task,
                tasks=args.tasks,
            ),
        )
    elif args.command == "count-features":
        _write_json(
            args.summary_json,
            write_count_features_for_labels(
                args.meds_dir,
                args.labels_dir,
                args.output_dir,
                task=args.task,
                label_family=args.label_family,
                feature_set=args.feature_set,
            ),
        )
    elif args.command == "count-features-batch":
        _write_json(
            args.summary_json,
            write_count_features_for_tasks(
                args.meds_dir,
                args.labels_dir,
                args.output_dir,
                tasks=args.tasks,
                label_family=args.label_family,
                feature_set=args.feature_set,
            ),
        )
    elif args.command == "probe":
        metrics = run_probe(args.features_dir, args.output_dir, task=args.task)
        _write_json(args.metrics_json, metrics)
    elif args.command == "tabular-model":
        metrics = run_tabular_model(
            args.features_dir,
            args.output_dir,
            task=args.task,
            model_name=args.model_name,
            seed=args.seed,
        )
        _write_json(args.metrics_json, metrics)
    elif args.command == "diagnostics":
        _write_json(
            args.output_json,
            write_diagnostics_bundle(
                args.predictions_root,
                args.features_root,
                args.duckdb,
                args.meds_dir,
                args.output_json,
                tasks=args.tasks,
                model_name=args.model_name,
                max_events_per_split=args.max_events_per_split,
            ),
        )
    elif args.command == "export-athena-vocab":
        _write_json(args.summary_json, export_athena_vocab_csvs(args.duckdb, args.output_dir))
    elif args.command == "validate-athena-vocab":
        _write_json(args.output_json, validate_athena_csv_export(args.export_dir))
    elif args.command == "export-omop-smoke-layouts":
        summary = export_all_omop_smoke_layouts(args.duckdb, args.output_root, max_persons=args.max_persons)
        _write_json(args.summary_json, summary)
    elif args.command == "task-bundle-manifest":
        write_task_bundle_manifest(args.labels_dir, args.output_json)
    elif args.command == "create-venv":
        result = create_uv_venv(args.name, args.python_version, tmpfs_root=args.tmpfs_root)
        _write_json(args.summary_json, asdict(result))
    elif args.command == "export-ehrshot-labels":
        print(export_task_to_ehrshot_csv(args.labels_dir, args.task, args.output_dir))
    elif args.command == "report":
        write_report(
            args.output_md,
            resource_summary=json.loads(args.resource_json.read_text()),
            label_counts=json.loads(args.labels_json.read_text()),
            tokenization_summary=json.loads(args.tokenization_json.read_text()),
            feature_counts=json.loads(args.features_json.read_text()),
            metrics=json.loads(args.metrics_json.read_text()),
            training_summary=json.loads(args.training_json.read_text()),
        )
    elif args.command == "phase1-summary":
        write_phase1_summary(args.status_json, args.output_md)


if __name__ == "__main__":
    main()


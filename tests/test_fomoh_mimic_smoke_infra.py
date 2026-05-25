import datetime as dt
import json
from pathlib import Path

import polars as pl

from ehr_foundation_model_benchmark.fomoh_mimic.model_envs import apply_compat_patches
from ehr_foundation_model_benchmark.fomoh_mimic.smoke_infra import (
    ModelEnvManifest,
    REQUIRED_ATHENA_TABLES,
    TaskStatus,
    materialize_manifest,
    summarize_task_bundle,
    validate_athena_csv_export,
)


def _write_labels(path: Path, positives_by_split: dict[str, int]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for split, positives in positives_by_split.items():
        total = max(2, positives + 1)
        pl.DataFrame(
            {
                "subject_id": list(range(total)),
                "prediction_time": [dt.datetime(2020, 1, 1)] * total,
                "boolean_value": [True] * positives + [False] * (total - positives),
            }
        ).write_parquet(path / f"{split}.parquet")


def test_env_manifest_records_interpreter_packages_and_smoke_status(tmp_path):
    manifest = ModelEnvManifest(
        model_name="cehrbert",
        venv_path=tmp_path / "venvs" / "cehrbert",
        python_path=tmp_path / "venvs" / "cehrbert" / "bin" / "python",
        python_version="3.10.14",
        install_command=["uv", "pip", "install", "--no-deps", "cehrbert==1.4.3"],
        package_versions={"cehrbert": "1.4.3", "numpy": "1.26.4"},
        import_smoke={"status": "passed", "imports": ["cehrbert"]},
        tmpfs_root=tmp_path / "shm" / "fomoh_mimic",
        log_path=tmp_path / "env_logs" / "cehrbert.log",
    )

    output_path = materialize_manifest(manifest, tmp_path / "manifest.json")
    payload = json.loads(output_path.read_text())

    assert payload["model_name"] == "cehrbert"
    assert payload["python_path"].endswith("bin/python")
    assert payload["package_versions"]["numpy"] == "1.26.4"
    assert payload["install_command"][:3] == ["uv", "pip", "install"]
    assert payload["import_smoke"]["status"] == "passed"
    assert payload["tmpfs_root"].endswith("fomoh_mimic")


def test_athena_export_validator_requires_non_empty_core_csvs(tmp_path):
    export_dir = tmp_path / "athena"
    export_dir.mkdir()
    for table in REQUIRED_ATHENA_TABLES:
        (export_dir / f"{table}.csv").write_text("concept_id\n1\n")

    summary = validate_athena_csv_export(export_dir)

    assert summary["status"] == "valid"
    assert set(summary["tables"]) == set(REQUIRED_ATHENA_TABLES)
    assert all(value["bytes"] > 0 for value in summary["tables"].values())


def test_task_bundle_summary_marks_all_tasks_and_zero_positive_non_identifiable(tmp_path):
    labels_dir = tmp_path / "labels"
    _write_labels(labels_dir / "patient_outcomes" / "death", {"train": 1, "tuning": 1, "held_out": 1})
    _write_labels(labels_dir / "patient_outcomes" / "long_los", {"train": 1, "tuning": 0, "held_out": 1})
    _write_labels(labels_dir / "patient_outcomes" / "readmission", {"train": 1, "tuning": 1, "held_out": 1})
    for name in (
        "ami",
        "celiac",
        "cll",
        "htn",
        "masld",
        "osteoporosis",
        "pancreatic_cancer",
        "schizophrenia",
        "sle",
        "t2dm",
    ):
        _write_labels(labels_dir / "phenotypes" / name, {"train": 1, "tuning": 1, "held_out": 1})
    _write_labels(labels_dir / "phenotypes" / "ischemic_stroke", {"train": 0, "tuning": 0, "held_out": 0})

    summary = summarize_task_bundle(labels_dir)

    assert len(summary) == 14
    assert summary["patient_outcomes/death"].status == TaskStatus.IDENTIFIABLE
    assert summary["phenotypes/ischemic_stroke"].status == TaskStatus.NON_IDENTIFIABLE
    assert summary["phenotypes/ischemic_stroke"].positives == 0


def test_cehrgpt_compat_patch_falls_back_to_start_and_end_tokens(tmp_path):
    venv_path = tmp_path / "venv"
    tokenizer_path = venv_path / "lib" / "python3.10" / "site-packages" / "cehrgpt" / "models" / "tokenization_hf_cehrgpt.py"
    tokenizer_path.parent.mkdir(parents=True)
    tokenizer_path.write_text(
        '    @property\n'
        '    def vs_token_id(self):\n'
        '        if "VS" in self._tokenizer.get_vocab():\n'
        '            return self._convert_token_to_id("VS")\n'
        '        elif "[VS]" in self._tokenizer.get_vocab():\n'
        '            return self._convert_token_to_id("[VS]")\n'
        '        else:\n'
        '            raise RuntimeError("The tokenizer does not contain either VS or [VS]")\n'
        '    @property\n'
        '    def ve_token_id(self):\n'
        '        if "VE" in self._tokenizer.get_vocab():\n'
        '            return self._convert_token_to_id("VE")\n'
        '        elif "[VE]" in self._tokenizer.get_vocab():\n'
        '            return self._convert_token_to_id("[VE]")\n'
        '        else:\n'
        '            raise RuntimeError("The tokenizer does not contain either VE or [VE]")\n'
    )

    result = apply_compat_patches("cehrgpt", venv_path)

    patched = tokenizer_path.read_text()
    assert result["cehrgpt_visit_start_token_fallback"] == "applied"
    assert result["cehrgpt_visit_end_token_fallback"] == "applied"
    assert "return self.start_token_id" in patched
    assert "return self.end_token_id" in patched
    assert "does not contain either VS" not in patched
    assert "does not contain either VE" not in patched

def test_medstab_compat_patch_writes_runtime_shims(tmp_path):
    venv_path = tmp_path / "venv"
    site = venv_path / "lib" / "python3.11" / "site-packages"
    (site / "MEDS_transforms" / "mapreduce").mkdir(parents=True)

    result = apply_compat_patches("medstab", venv_path)

    mixin_text = (site / "mixins" / "__init__.py").read_text()
    assert result["medstab_timeable_mixin"] == "applied"
    assert "class TimeableMixin" in mixin_text
    assert "def _register_start" in mixin_text
    assert "def _register_end" in mixin_text
    assert (site / "MEDS_transforms" / "mapreduce" / "utils.py").read_text() == "from .rwlock import rwlock_wrap\n"


def test_corebehrt_compat_patch_handles_transformers_and_smoke_knobs(tmp_path):
    venv_path = tmp_path / "venv"
    site = venv_path / "lib" / "python3.10" / "site-packages"
    model_path = site / "corebehrt" / "model" / "model.py"
    init_path = site / "corebehrt" / "common" / "initialize.py"
    pretrain_path = site / "corebehrt" / "main_pretrain.py"
    model_path.parent.mkdir(parents=True)
    init_path.parent.mkdir(parents=True)
    model_path.write_text(
        'modeling_bert.BERT_SELF_ATTENTION_CLASSES.update({"flash_attention_2": RoFormerSelfFlashAttention})\n'
        '        print(f"is_flash_attn_2_available(): {is_flash_attn_2_available()}")\n'
        '        print(f"config._attn_implementation: {config._attn_implementation}")\n'
        '        for layer in self.encoder.layer:\n'
        '            layer.intermediate.intermediate_act_fn = SwiGLU(config)\n'
        '            if is_flash_attn_2_available() and config._attn_implementation == "flash_attention_2":\n'
    )
    init_path.write_text("        print('Checking flash attention', model.config._attn_implementation)\n")
    pretrain_path.write_text(
        '    dataloader_num_workers = cfg.get("trainer_args").get("dataloader_num_workers", 8)\n'
        '    dataloader_prefetch_factor = cfg.get("trainer_args").get("dataloader_prefetch_factor", 8)\n'
        '\n'
        '    training_args = TrainingArguments(\n'
        '        output_dir=cfg.paths.output_path,\n'
        '        save_strategy="epoch",\n'
        "        eval_strategy='epoch',\n"
        '        warmup_steps=500,\n'
        '        per_device_train_batch_size=per_device_train_batch_size,\n'
        '        gradient_accumulation_steps=gradient_accumulation_steps,\n'
        '        per_device_eval_batch_size=32,\n'
        '        learning_rate=learning_rate,\n'
        '        weight_decay=0.01,\n'
        '        num_train_epochs=20,\n'
        '        data_seed=31,\n'
        '        seed=31,\n'
        '        disable_tqdm=False,\n'
        "        metric_for_best_model='eval_loss',\n"
        '        load_best_model_at_end=True,\n'
        '        logging_steps=logging_steps,\n'
        '        label_names=label_names,\n'
        '        do_train=True,\n'
        '        dataloader_num_workers=dataloader_num_workers,\n'
        '        dataloader_prefetch_factor=dataloader_prefetch_factor,\n'
        '    )\n'
        '    trainer = trainer_class(\n'
        '        model=model,\n'
        '        args=training_args,\n'
        '        train_dataset=prepared_dataset["train"],\n'
        '        eval_dataset=prepared_dataset["validation"],\n'
        '        callbacks=[\n'
        '            CustomEarlyStoppingCallback(1, 0.01),\n'
        '            LossLoggingCallback(log_file=os.path.join(cfg.paths.output_path, "loss_log.json"))\n'
        '        ],\n'
        '        data_collator=data_collator_fn\n'
        '    )\n'
    )

    result = apply_compat_patches("corebehrt", venv_path)

    model_text = model_path.read_text()
    pretrain_text = pretrain_path.read_text()
    assert result["corebehrt_transformers_compat"] == "applied"
    assert result["corebehrt_attn_implementation_compat"] == "applied"
    assert "attn_implementation = getattr(config" in model_text
    assert "config._attn_implementation" not in model_text
    assert "getattr(model.config, '_attn_implementation'" in init_path.read_text()
    assert "evaluation_strategy=evaluation_strategy" in pretrain_text
    assert "max_steps=max_steps" in pretrain_text
    assert "dataloader_prefetch_factor=dataloader_prefetch_factor" not in pretrain_text
    assert "callbacks = [LossLoggingCallback" in pretrain_text

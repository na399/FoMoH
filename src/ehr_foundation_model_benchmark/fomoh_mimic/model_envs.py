from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ehr_foundation_model_benchmark.fomoh_mimic.smoke_infra import (
    DEFAULT_TMPFS_ROOT,
    ModelEnvManifest,
    materialize_manifest,
    package_versions,
    run_logged,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = REPO_ROOT / "src" / "ehr_foundation_model_benchmark" / "evaluations"
ENV_LOG_ROOT = REPO_ROOT / "temp" / "fomoh_mimic" / "env_logs"
MANIFEST_ROOT = REPO_ROOT / "runs" / "fomoh_mimic" / "env_manifests"


@dataclass(frozen=True)
class EnvSpec:
    name: str
    python_version: str
    installs: tuple[tuple[str, ...], ...]
    imports: tuple[str, ...]
    packages: tuple[str, ...]
    help_commands: tuple[tuple[str, ...], ...] = ()


def _wheel(path: Path) -> str:
    return str(path)


ENV_SPECS: dict[str, EnvSpec] = {
    "cehrbert": EnvSpec(
        name="cehrbert",
        python_version="3.10",
        installs=(
            ("uv", "pip", "install", "--python", "{python}", "numpy==1.26.4", "pandas==2.2.0", "pyarrow==15.0.2", "scipy==1.11.4", "scikit-learn==1.4.2", "transformers==4.44.1", "tokenizers==0.19.0", "datasets==2.19.2", "accelerate==0.30.1", "torch==2.3.1", "pyspark==3.5.1", "meds==0.3.3", "meds_reader==0.1.13", "peft==0.10.0", "openai==1.54.3", "optuna==4.0.0", "lightgbm"),
            ("uv", "pip", "install", "--python", "{python}", "--no-deps", "cehrbert_data==0.0.9", "cehrbert==1.4.3", _wheel(EVAL_ROOT / "motor" / "femr-0.2.0-py3-none-any.whl"), _wheel(EVAL_ROOT / "cehrbert" / "meds_evaluation-0.1.dev95+g841c87f-py3-none-any.whl"), "-e", str(REPO_ROOT)),
        ),
        imports=("cehrbert", "cehrbert_data", "meds_evaluation", "ehr_foundation_model_benchmark"),
        packages=("cehrbert", "cehrbert-data", "meds-evaluation", "numpy", "pandas", "pyarrow", "torch", "transformers"),
        help_commands=(("-m", "cehrbert.runners.hf_cehrbert_pretrain_runner", "--help"),),
    ),
    "cehrgpt": EnvSpec(
        name="cehrgpt",
        python_version="3.10",
        installs=(
            ("uv", "pip", "install", "--python", "{python}", "numpy==1.26.4", "pandas==2.2.0", "pyarrow==15.0.2", "scipy==1.11.4", "scikit-learn==1.4.2", "transformers==4.40.2", "datasets==2.19.2", "accelerate==0.30.1", "torch==2.3.1", "pyspark==3.5.1", "meds==0.3.3", "meds_reader==0.1.13"),
            ("uv", "pip", "install", "--python", "{python}", "--no-deps", "cehrbert_data==0.1.1", "cehrbert==1.4.9", "cehrgpt==0.1.6.post4", _wheel(EVAL_ROOT / "motor" / "femr-0.2.0-py3-none-any.whl"), _wheel(EVAL_ROOT / "cehrgpt" / "meds_evaluation-0.1.dev95+g841c87f-py3-none-any.whl"), "-e", str(REPO_ROOT)),
        ),
        imports=("cehrgpt", "cehrbert_data", "meds_evaluation", "ehr_foundation_model_benchmark"),
        packages=("cehrgpt", "cehrbert-data", "meds-evaluation", "numpy", "pandas", "pyarrow", "torch", "transformers"),
        help_commands=(("-m", "cehrgpt.runners.hf_cehrgpt_pretrain_runner", "--help"),),
    ),
    "corebehrt": EnvSpec(
        name="corebehrt",
        python_version="3.10",
        installs=(
            ("uv", "pip", "install", "--python", "{python}", "numpy==1.26.4", "pandas==2.2.0", "pyarrow==12.0.1", "scipy==1.11.4", "scikit-learn==1.4.2", "torch==2.3.1", "transformers==4.35.2", "hydra-core==1.3.2", "datasets==2.19.2", "accelerate==0.30.1"),
            ("uv", "pip", "install", "--python", "{python}", "--no-deps", _wheel(EVAL_ROOT / "corebehrt" / "corebehrt-0.1.0-py3-none-any.whl"), _wheel(EVAL_ROOT / "corebehrt" / "meds_evaluation-0.1.dev95+g841c87f-py3-none-any.whl"), "-e", str(REPO_ROOT)),
        ),
        imports=("corebehrt", "meds_evaluation", "ehr_foundation_model_benchmark"),
        packages=("corebehrt", "meds-evaluation", "numpy", "pandas", "pyarrow", "torch", "transformers", "accelerate"),
        help_commands=(("-m", "corebehrt.main_create_data", "--help"), ("-m", "corebehrt.main_pretrain", "--help")),
    ),
    "medstab": EnvSpec(
        name="medstab",
        python_version="3.11",
        installs=(
            ("uv", "pip", "install", "--python", "{python}", "numpy==1.26.4", "pandas==2.2.2", "polars==1.41.0", "pyarrow==15.0.2", "scipy==1.11.4", "scikit-learn==1.4.2", "xgboost==2.0.3", "hydra-core==1.3.2", "omegaconf==2.3.0", "loguru==0.7.3", "hydra-optuna-sweeper==1.2.0", "hydra-joblib-launcher==1.2.0", "meds==0.4.1", "MEDS-transforms==0.6.7"),
            ("uv", "pip", "install", "--python", "{python}", "--no-deps", _wheel(EVAL_ROOT / "medstab" / "meds_tab-0.1.dev478+g74f80c3-py3-none-any.whl"), _wheel(EVAL_ROOT / "medstab" / "meds_evaluation-0.1.dev95+g841c87f-py3-none-any.whl"), "-e", str(REPO_ROOT)),
        ),
        imports=("MEDS_tabular_automl", "meds_evaluation", "ehr_foundation_model_benchmark"),
        packages=("meds-tab", "meds-evaluation", "MEDS-transforms", "meds", "hydra-core", "numpy", "pandas", "polars", "pyarrow", "scipy", "scikit-learn", "xgboost"),
    ),
    "motor": EnvSpec(
        name="motor",
        python_version="3.10",
        installs=(
            ("uv", "pip", "install", "--python", "{python}", "numpy==1.26.4", "pandas==2.2.0", "polars==0.20.31", "pyarrow==15.0.2", "scipy==1.11.4", "scikit-learn==1.4.2", "meds_reader==0.1.13"),
            ("uv", "pip", "install", "--python", "{python}", _wheel(EVAL_ROOT / "motor" / "femr-0.2.0-py3-none-any.whl"), _wheel(EVAL_ROOT / "motor" / "meds_evaluation-0.1.dev95+g841c87f-py3-none-any.whl"), "-e", str(REPO_ROOT)),
        ),
        imports=("femr", "meds_reader", "meds_evaluation", "ehr_foundation_model_benchmark"),
        packages=("femr", "meds-reader", "meds-evaluation", "numpy", "pandas", "pyarrow"),
        help_commands=(("-m", "femr.omop_meds_tutorial.prepare_motor", "--help"),),
    ),
    "hf_ehr": EnvSpec(
        name="hf_ehr",
        python_version="3.10",
        installs=(
            ("uv", "pip", "install", "--python", "{python}", "numpy==1.26.4", "pandas==2.2.2", "polars==0.20.31", "scipy==1.11.4", "meds_reader==0.1.13", "omegaconf==2.3.0", "hydra-core==1.3.2", "lightning==2.5.0", "transformers==4.48.2", "torch==2.7.1", "wandb==0.17.9", "loguru==0.7.2", "jaxtyping==0.2.36", "tensorboard==2.17.1"),
            ("uv", "pip", "install", "--python", "{python}", "--no-deps", "-e", "/home/natthawut/hf_ehr", _wheel(EVAL_ROOT / "llama_mamba" / "meds_evaluation-0.1.dev95+g841c87f-py3-none-any.whl"), "-e", str(REPO_ROOT)),
        ),
        imports=("hf_ehr", "meds_reader", "lightning", "transformers", "torch"),
        packages=("hf-ehr", "meds-reader", "lightning", "transformers", "torch", "numpy", "pandas", "polars"),
        help_commands=(("-m", "hf_ehr.scripts.run", "--help"),),
    ),
}


def _format_command(command: tuple[str, ...], python_path: Path) -> list[str]:
    return [str(python_path) if part == "{python}" else part for part in command]


def _replace_once(path: Path, old: str, new: str) -> str:
    if not path.exists():
        return "missing"
    text = path.read_text()
    if new in text:
        return "already_applied"
    if old not in text:
        return "old_text_missing"
    path.write_text(text.replace(old, new))
    return "applied"


def _site_packages_dir(venv_path: Path) -> Path:
    candidates = sorted((venv_path / "lib").glob("python*/site-packages"))
    if candidates:
        return candidates[0]
    return venv_path / "lib" / "python3.10" / "site-packages"


def apply_compat_patches(name: str, venv_path: Path) -> dict[str, str]:
    site = _site_packages_dir(venv_path)
    results: dict[str, str] = {}
    if name == "cehrbert":
        results["cehrbert_transformers_compat"] = _replace_once(
            site / "cehrbert" / "models" / "hf_models" / "hf_cehrbert.py",
            'modeling_bert.BERT_SELF_ATTENTION_CLASSES.update({"flash_attention_2": BertSelfFlashAttention})',
            'if hasattr(modeling_bert, "BERT_SELF_ATTENTION_CLASSES"):\n    modeling_bert.BERT_SELF_ATTENTION_CLASSES.update({"flash_attention_2": BertSelfFlashAttention})',
        )
    if name == "cehrgpt":
        tokenizer_path = site / "cehrgpt" / "models" / "tokenization_hf_cehrgpt.py"
        results["cehrgpt_visit_start_token_fallback"] = _replace_once(
            tokenizer_path,
            '        else:\n            raise RuntimeError("The tokenizer does not contain either VS or [VS]")',
            '        else:\n            return self.start_token_id',
        )
        results["cehrgpt_visit_end_token_fallback"] = _replace_once(
            tokenizer_path,
            '        else:\n            raise RuntimeError("The tokenizer does not contain either VE or [VE]")',
            '        else:\n            return self.end_token_id',
        )
    if name == "corebehrt":
        model_path = site / "corebehrt" / "model" / "model.py"
        initialize_path = site / "corebehrt" / "common" / "initialize.py"
        pretrain_path = site / "corebehrt" / "main_pretrain.py"
        results["corebehrt_transformers_compat"] = _replace_once(
            model_path,
            'modeling_bert.BERT_SELF_ATTENTION_CLASSES.update({"flash_attention_2": RoFormerSelfFlashAttention})',
            'if hasattr(modeling_bert, "BERT_SELF_ATTENTION_CLASSES"):\n    modeling_bert.BERT_SELF_ATTENTION_CLASSES.update({"flash_attention_2": RoFormerSelfFlashAttention})',
        )
        results["corebehrt_attn_implementation_compat"] = _replace_once(
            model_path,
            '        print(f"is_flash_attn_2_available(): {is_flash_attn_2_available()}")\n'
            '        print(f"config._attn_implementation: {config._attn_implementation}")\n'
            '        for layer in self.encoder.layer:\n'
            '            layer.intermediate.intermediate_act_fn = SwiGLU(config)\n'
            '            if is_flash_attn_2_available() and config._attn_implementation == "flash_attention_2":',
            '        attn_implementation = getattr(config, "_attn_implementation", getattr(config, "attn_implementation", "eager"))\n'
            '        print(f"is_flash_attn_2_available(): {is_flash_attn_2_available()}")\n'
            '        print(f"config.attn_implementation: {attn_implementation}")\n'
            '        for layer in self.encoder.layer:\n'
            '            layer.intermediate.intermediate_act_fn = SwiGLU(config)\n'
            '            if is_flash_attn_2_available() and attn_implementation == "flash_attention_2":',
        )
        results["corebehrt_initialize_attn_compat"] = _replace_once(
            initialize_path,
            "        print('Checking flash attention', model.config._attn_implementation)",
            "        print('Checking flash attention', getattr(model.config, '_attn_implementation', getattr(model.config, 'attn_implementation', 'eager')))",
        )
        results["corebehrt_training_args_eval_name"] = _replace_once(
            pretrain_path,
            "        eval_strategy='epoch',",
            "        evaluation_strategy='epoch',",
        )
        results["corebehrt_training_args_smoke_knobs"] = _replace_once(
            pretrain_path,
            '    dataloader_num_workers = cfg.get("trainer_args").get("dataloader_num_workers", 8)\n'
            '    dataloader_prefetch_factor = cfg.get("trainer_args").get("dataloader_prefetch_factor", 8)\n'
            '\n'
            '    training_args = TrainingArguments(\n'
            '        output_dir=cfg.paths.output_path,\n'
            '        save_strategy="epoch",\n'
            "        evaluation_strategy='epoch',\n"
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
            '    )',
            '    dataloader_num_workers = cfg.get("trainer_args").get("dataloader_num_workers", 8)\n'
            '    max_steps = cfg.get("trainer_args").get("max_steps", -1)\n'
            '    num_train_epochs = cfg.get("trainer_args").get("num_train_epochs", 20)\n'
            '    evaluation_strategy = cfg.get("trainer_args").get("evaluation_strategy", "epoch")\n'
            '    save_strategy = cfg.get("trainer_args").get("save_strategy", "epoch")\n'
            '    load_best_model_at_end = cfg.get("trainer_args").get("load_best_model_at_end", evaluation_strategy != "no")\n'
            '\n'
            '    training_args = TrainingArguments(\n'
            '        output_dir=cfg.paths.output_path,\n'
            '        save_strategy=save_strategy,\n'
            '        save_steps=cfg.get("trainer_args").get("save_steps", 500),\n'
            '        evaluation_strategy=evaluation_strategy,\n'
            '        warmup_steps=cfg.get("trainer_args").get("warmup_steps", 500),\n'
            '        per_device_train_batch_size=per_device_train_batch_size,\n'
            '        gradient_accumulation_steps=gradient_accumulation_steps,\n'
            '        per_device_eval_batch_size=cfg.get("trainer_args").get("per_device_eval_batch_size", 32),\n'
            '        learning_rate=learning_rate,\n'
            '        weight_decay=0.01,\n'
            '        max_steps=max_steps,\n'
            '        num_train_epochs=num_train_epochs,\n'
            '        data_seed=31,\n'
            '        seed=31,\n'
            '        disable_tqdm=False,\n'
            "        metric_for_best_model='eval_loss',\n"
            '        load_best_model_at_end=load_best_model_at_end,\n'
            '        logging_steps=logging_steps,\n'
            '        label_names=label_names,\n'
            '        do_train=True,\n'
            '        do_eval=evaluation_strategy != "no",\n'
            '        dataloader_num_workers=dataloader_num_workers,\n'
            '    )',
        )
        results["corebehrt_conditional_early_stopping"] = _replace_once(
            pretrain_path,
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
            '    )',
            '    callbacks = [LossLoggingCallback(log_file=os.path.join(cfg.paths.output_path, "loss_log.json"))]\n'
            '    if load_best_model_at_end:\n'
            '        callbacks.insert(0, CustomEarlyStoppingCallback(1, 0.01))\n'
            '\n'
            '    trainer = trainer_class(\n'
            '        model=model,\n'
            '        args=training_args,\n'
            '        train_dataset=prepared_dataset["train"],\n'
            '        eval_dataset=prepared_dataset["validation"],\n'
            '        callbacks=callbacks,\n'
            '        data_collator=data_collator_fn\n'
            '    )',
        )
    if name == "medstab":
        mixins_dir = site / "mixins"
        mixins_dir.mkdir(parents=True, exist_ok=True)
        (mixins_dir / "__init__.py").write_text(
            "class TimeableMixin:\n"
            "    def __init__(self, *args, cache_prefix=None, **kwargs):\n"
            "        self.cache_prefix = cache_prefix\n"
            "        if args or kwargs:\n"
            "            try:\n"
            "                super().__init__(*args, **kwargs)\n"
            "            except TypeError:\n"
            "                super().__init__()\n"
            "        else:\n"
            "            super().__init__()\n"
            "\n"
            "    def _register_start(self, key=None):\n"
            "        return None\n"
            "\n"
            "    def _register_end(self, key=None):\n"
            "        return None\n"
            "\n"
            "    @staticmethod\n"
            "    def TimeAs(func):\n"
            "        def wrapper(*args, **kwargs):\n"
            "            return func(*args, **kwargs)\n"
            "        wrapper.__name__ = getattr(func, \"__name__\", \"wrapper\")\n"
            "        wrapper.__doc__ = getattr(func, \"__doc__\", None)\n"
            "        return wrapper\n"
        )
        results["medstab_timeable_mixin"] = "applied"
        mapreduce_dir = site / "MEDS_transforms" / "mapreduce"
        mapreduce_dir.mkdir(parents=True, exist_ok=True)
        (mapreduce_dir / "utils.py").write_text("from .rwlock import rwlock_wrap\n")
        results["medstab_meds_transforms_utils"] = "applied"
    return results


def _run_import_smoke(python_path: Path, imports: tuple[str, ...], log_path: Path) -> dict[str, Any]:
    code = "\n".join(
        [
            "import importlib, json",
            f"mods = {list(imports)!r}",
            "out = {}",
            "for m in mods:",
            "    try:",
            "        importlib.import_module(m)",
            "        out[m] = 'ok'",
            "    except Exception as e:",
            "        out[m] = type(e).__name__ + ': ' + str(e)",
            "print(json.dumps(out, sort_keys=True))",
            "raise SystemExit(0 if all(v == 'ok' for v in out.values()) else 1)",
        ]
    )
    result = run_logged([str(python_path), "-c", code], log_path)
    payload = {"status": result.status, "returncode": result.returncode, "log_path": str(log_path), "imports": list(imports)}
    try:
        last_line = log_path.read_text().strip().splitlines()[-1]
        payload["results"] = json.loads(last_line)
    except (IndexError, json.JSONDecodeError):
        payload["results"] = {}
    return payload


def prepare_env(spec: EnvSpec, *, tmpfs_root: Path = DEFAULT_TMPFS_ROOT) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(tmpfs_root / "uv_cache"))
    env.setdefault("TMPDIR", str(tmpfs_root / "tmp"))
    (tmpfs_root / "tmp").mkdir(parents=True, exist_ok=True)
    venv_path = tmpfs_root / "venvs" / spec.name
    python_path = venv_path / "bin" / "python"
    log_dir = ENV_LOG_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = MANIFEST_ROOT
    manifest_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    venv_result = run_logged(["uv", "venv", "--python", spec.python_version, str(venv_path)], log_dir / f"{spec.name}_00_venv.log", env=env)
    steps.append(asdict(venv_result))
    if venv_result.returncode == 0:
        pointer = REPO_ROOT / "temp" / "fomoh_mimic" / "venvs" / spec.name
        pointer.parent.mkdir(parents=True, exist_ok=True)
        if not pointer.exists() and not pointer.is_symlink():
            pointer.symlink_to(venv_path, target_is_directory=True)
    for idx, install in enumerate(spec.installs, start=1):
        command = _format_command(install, python_path)
        result = run_logged(command, log_dir / f"{spec.name}_{idx:02d}_install.log", env=env)
        steps.append(asdict(result))
        if result.returncode != 0:
            break

    patch_results = apply_compat_patches(spec.name, venv_path)
    if patch_results:
        write_json(manifest_dir / f"{spec.name}_compat_patches.json", patch_results)

    smoke = {"status": "not_run", "imports": list(spec.imports)}
    if all(step["returncode"] == 0 for step in steps):
        smoke = _run_import_smoke(python_path, spec.imports, log_dir / f"{spec.name}_import_smoke.log")
        for idx, help_command in enumerate(spec.help_commands, start=1):
            result = run_logged([str(python_path), *help_command], log_dir / f"{spec.name}_help_{idx:02d}.log", env=env)
            steps.append(asdict(result))

    py_version = "unknown"
    if python_path.exists():
        proc = subprocess.run([str(python_path), "--version"], capture_output=True, text=True, check=False)
        py_version = (proc.stdout or proc.stderr).strip()
    manifest = ModelEnvManifest(
        model_name=spec.name,
        venv_path=venv_path,
        python_path=python_path,
        python_version=py_version,
        install_command=["; ".join(" ".join(_format_command(command, python_path)) for command in spec.installs)],
        package_versions=package_versions(python_path, spec.packages) if python_path.exists() else {},
        import_smoke=smoke,
        tmpfs_root=tmpfs_root,
        log_path=log_dir / f"{spec.name}_import_smoke.log",
    )
    manifest_path = materialize_manifest(manifest, manifest_dir / f"{spec.name}.json")
    payload = {"manifest_path": str(manifest_path), "steps": steps, "import_smoke": smoke}
    write_json(manifest_dir / f"{spec.name}_steps.json", payload)
    return payload


def prepare_envs(names: list[str] | None = None, *, tmpfs_root: Path = DEFAULT_TMPFS_ROOT) -> dict[str, Any]:
    selected = names or list(ENV_SPECS)
    return {name: prepare_env(ENV_SPECS[name], tmpfs_root=tmpfs_root) for name in selected}


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare isolated FoMoH MIMIC model smoke environments.")
    parser.add_argument("--models", nargs="+", choices=sorted(ENV_SPECS), default=None)
    parser.add_argument("--tmpfs-root", type=Path, default=DEFAULT_TMPFS_ROOT)
    parser.add_argument("--summary-json", type=Path, default=MANIFEST_ROOT / "env_setup_summary.json")
    args = parser.parse_args(argv)
    summary = prepare_envs(args.models, tmpfs_root=args.tmpfs_root)
    write_json(args.summary_json, summary)


if __name__ == "__main__":
    main(sys.argv[1:])

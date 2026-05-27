import importlib.util
import json
from pathlib import Path

import polars as pl

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ehr_foundation_model_benchmark.fomoh_mimic.hydra_app import compose_run_plan, render_commands, write_run_plan


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(*overrides: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def test_hydra_configs_compose_for_every_phase_and_model():
    phases = ["validate", "smoke", "full_active", "report", "non_cuda"]
    models = ["hf_ehr_gpt", "hf_ehr_llama", "hf_ehr_mamba", "medstab", "motor", "cehrbert", "cehrgpt", "corebehrt"]

    for phase in phases:
        for model in models:
            cfg = _compose("experiment=mimic_phase1", f"phase={phase}", f"model={model}")
            assert cfg.dataset.name == "fomoh_mimic"
            assert cfg.phase.name == phase
            assert cfg.model.name == model
            assert cfg.dry_run is True
            assert cfg.execute is False


def test_mimic_phase1_uses_env_backed_paths_not_local_absolute_paths():
    cfg = _compose("experiment=mimic_phase1", "phase=report")
    rendered = OmegaConf.to_yaml(cfg, resolve=False)

    assert "${oc.env:FOMOH_MIMIC_MEDS_DIR" in rendered
    assert "/home/natthawut" not in rendered
    assert cfg.dataset.name == "fomoh_mimic"
    assert str(cfg.paths.tmpfs_root).startswith("/dev/shm/")


def test_command_plan_contains_report_commands_and_writes_deterministic_json(tmp_path):
    cfg = _compose("experiment=mimic_phase1", "phase=report")
    cfg.paths.run_root = str(tmp_path / "runs" / "fomoh_mimic")
    cfg.paths.temp_root = str(tmp_path / "temp" / "fomoh_mimic")
    cfg.paths.tmpfs_root = str(tmp_path / "shm" / "fomoh_mimic")

    plan = compose_run_plan(cfg)
    commands = render_commands(plan)
    output_path = write_run_plan(plan, tmp_path / "plans")

    assert [command.name for command in commands] == ["write_benchmark_status", "write_benchmark_profile", "write_phase1_summary"]
    assert output_path == tmp_path / "plans" / "report_plan.json"
    payload = json.loads(output_path.read_text())
    assert payload["phase"] == "report"
    assert payload["execute"] is False
    assert payload["commands"][0]["name"] == "write_benchmark_status"
    assert payload["commands"][0]["command"][:2] == ["uv", "run"]


def test_full_active_mamba_plan_uses_validated_kernel_and_cpu_safe_dry_run():
    cfg = _compose("experiment=mimic_phase1", "phase=full_active", "model=hf_ehr_mamba")
    plan = compose_run_plan(cfg)
    commands = render_commands(plan)

    assert plan.requires_gpu is True
    assert plan.dry_run is True
    assert plan.execute is False
    assert plan.gpu_min_free_mib == 35000
    assert plan.gpu_max_process_mib == 10000
    assert any("causal-conv1d" in command.command for command in commands)
    assert any("mamba-ssm" in command.command for command in commands)
    run_command = next(command for command in commands if command.name == "hf_ehr_full_active")
    assert run_command.env["FOMOH_HF_EHR_MODELS"] == "mamba"
    assert run_command.env["FOMOH_HF_EHR_BATCH_SIZE"] == "128"
    assert run_command.env["FOMOH_HF_EHR_AUTOCAST"] == "1"
    assert run_command.env["FOMOH_HF_EHR_AUTOCAST_DTYPE"] == "bfloat16"
    assert "/dev/shm/" in run_command.env["FOMOH_HF_EHR_BENCH_TMP"]


def test_validate_and_smoke_plans_emit_cpu_safe_commands():
    validate_cfg = _compose("experiment=mimic_phase1", "phase=validate")
    validate_plan = compose_run_plan(validate_cfg)
    assert validate_plan.requires_gpu is False
    assert [command.name for command in render_commands(validate_plan)] == ["validate_resources", "task_manifest_check"]

    smoke_cfg = _compose("experiment=mimic_phase1", "phase=smoke", "model=medstab")
    smoke_plan = compose_run_plan(smoke_cfg)
    smoke_commands = render_commands(smoke_plan)
    assert smoke_plan.dry_run is True
    assert [command.name for command in smoke_commands] == ["medstab_all_task_smoke"]
    assert "run_medstab_official_all_task_smoke.py" in " ".join(smoke_commands[0].command)
    assert all("nvidia-smi" not in command.command for command in smoke_commands)


def test_smoke_plans_are_model_specific_and_gpu_gated_when_needed():
    mamba_cfg = _compose("experiment=mimic_phase1", "phase=smoke", "model=hf_ehr_mamba")
    mamba_plan = compose_run_plan(mamba_cfg)
    mamba_commands = render_commands(mamba_plan)
    assert mamba_plan.requires_gpu is True
    assert [command.name for command in mamba_commands] == [
        "hf_ehr_mamba_kernel_install",
        "hf_ehr_architecture_smokes",
        "hf_ehr_all_task_probe_smokes",
    ]
    assert any("mamba-ssm" in command.command for command in mamba_commands)
    assert any("run_hf_ehr_smokes.py" in " ".join(command.command) for command in mamba_commands)

    cehrbert_cfg = _compose("experiment=mimic_phase1", "phase=smoke", "model=cehrbert")
    cehrbert_plan = compose_run_plan(cehrbert_cfg)
    cehrbert_commands = render_commands(cehrbert_plan)
    assert cehrbert_plan.requires_gpu is False
    assert cehrbert_commands[0].name == "cehrbert_smoke_evidence"
    assert cehrbert_commands[0].command[-1].endswith("cehrbert_smoke.json")


def test_si_experiment_uses_env_backed_duckdb_without_local_paths():
    cfg = _compose("experiment=si_non_cuda", "phase=validate")
    rendered = OmegaConf.to_yaml(cfg, resolve=False)

    assert cfg.dataset.name == "fomoh_si"
    assert "allow_data_reads" not in cfg.dataset
    assert cfg.dataset.duckdb_env == "BRANCHUS_DUCKDB_PATH"
    assert cfg.dataset.schema_env == "BRANCHUS_DUCKDB_SCHEMA"
    assert cfg.dataset.encryption_key_env == "BRANCHUS_DUCKDB_ENCRYPTION_KEY"
    assert cfg.dataset.branchus_config_env == "BRANCHUS_CONFIG_PATH"
    assert "/home/natthawut" not in rendered
    assert "${oc.env:BRANCHUS_DUCKDB_PATH" not in rendered


def test_si_validate_plan_checks_public_env_contract(tmp_path):
    cfg = _compose("experiment=si_non_cuda", "phase=validate")
    cfg.paths.run_root = str(tmp_path / "runs" / "fomoh_si")
    cfg.paths.temp_root = str(tmp_path / "temp" / "fomoh_si")

    plan = compose_run_plan(cfg)
    commands = render_commands(plan)
    output_path = write_run_plan(plan, tmp_path / "plans")
    payload = json.loads(output_path.read_text())

    assert plan.requires_gpu is False
    assert [command.name for command in commands] == ["si_env_contract_check"]
    assert "validate-resources" not in " ".join(commands[0].command)
    assert "--write-env-contract" in commands[0].command
    assert payload["dataset"] == "fomoh_si"


def test_si_non_cuda_plan_runs_real_cpu_safe_pipeline():
    cfg = _compose("experiment=si_non_cuda", "phase=non_cuda", "model=medstab")
    plan = compose_run_plan(cfg)
    commands = render_commands(plan)

    assert plan.requires_gpu is False
    assert [command.name for command in commands] == ["si_env_contract_check", "si_non_cuda_pipeline"]
    joined = " ".join(part for command in commands for part in command.command)
    assert "run_si_non_branchus_steps.py" in joined
    assert "--run-root" in joined
    assert "--summary-json" in joined
    assert "--write-step-record" not in joined
    assert "run_si_non_cuda_steps.py" not in joined


def test_model_sensitive_plans_use_model_specific_artifact_names(tmp_path):
    cfg = _compose("experiment=mimic_phase1", "phase=smoke", "model=medstab")
    plan = compose_run_plan(cfg)
    output_path = write_run_plan(plan, tmp_path / "plans")
    assert output_path.name == "smoke_medstab_plan.json"
    assert (tmp_path / "plans" / "smoke_medstab_commands.sh").exists()


def test_si_prevalence_baseline_writes_split_metrics(tmp_path):
    module_path = Path(__file__).resolve().parents[1] / "temp" / "fomoh_si" / "scripts" / "run_si_non_cuda_steps.py"
    spec = importlib.util.spec_from_file_location("run_si_non_cuda_steps", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    labels = tmp_path / "labels"
    labels.mkdir()
    pl.DataFrame(
        {
            "task_name": ["toy"] * 8,
            "person_id": list(range(8)),
            "split": ["train", "train", "train", "train", "val", "val", "test", "test"],
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    ).write_parquet(labels / "toy.parquet")

    summary = module.run_prevalence_baselines(labels, tmp_path / "baseline")

    assert summary["status"] == "completed"
    assert summary["tasks"]["toy"]["splits"]["test"]["status"] == "completed"
    assert summary["tasks"]["toy"]["splits"]["test"]["prevalence_train"] == 0.5
    assert (tmp_path / "baseline" / "predictions" / "toy" / "test.parquet").exists()

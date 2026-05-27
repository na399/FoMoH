from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class CommandSpec:
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = "."
    log_path: str | None = None
    requires_gpu: bool = False


@dataclass(frozen=True)
class RunPlan:
    dataset: str
    model: str
    phase: str
    dry_run: bool
    execute: bool
    requires_gpu: bool
    run_root: str
    temp_root: str
    tmpfs_root: str
    commands: list[CommandSpec]
    gpu_min_free_mib: int = 35_000
    gpu_max_process_mib: int = 10_000


def _to_plain(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return cfg


def _str(value: Any) -> str:
    return str(value)


def _command_log(temp_root: str, name: str) -> str:
    return str(Path(temp_root) / "hydra" / f"{name}.log")


def _dataset_config_name(cfg: dict[str, Any]) -> str:
    return _str(cfg.get("dataset", {}).get("config_name", cfg.get("dataset", {}).get("name", "template")))


def _self_overrides(cfg: dict[str, Any]) -> list[str]:
    return [
        f"dataset={_dataset_config_name(cfg)}",
        f"phase={cfg['phase']['name']}",
        f"model={cfg['model']['name']}",
        f"paths.run_root={cfg['paths']['run_root']}",
        f"paths.temp_root={cfg['paths']['temp_root']}",
        f"paths.tmpfs_root={cfg['paths']['tmpfs_root']}",
    ]


def _env_contract_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["temp_root"]) / "hydra" / f"{cfg['dataset']['name']}_env_contract.json"


def _step_record_path(cfg: dict[str, Any], kind: str) -> Path:
    return Path(cfg["paths"]["temp_root"]) / "hydra" / f"{cfg['dataset']['name']}_{cfg['model']['name']}_{kind}.json"


def _env_contract_command(cfg: dict[str, Any]) -> CommandSpec:
    temp_root = cfg["paths"]["temp_root"]
    return CommandSpec(
        name="si_env_contract_check" if cfg["dataset"]["name"] == "fomoh_si" else "env_contract_check",
        command=[
            "uv",
            "run",
            "python",
            "-m",
            "ehr_foundation_model_benchmark.fomoh_mimic.hydra_app",
            "--write-env-contract",
            str(_env_contract_path(cfg)),
            *_self_overrides(cfg),
        ],
        log_path=_command_log(temp_root, "env_contract_check"),
    )


def _si_non_cuda_command(cfg: dict[str, Any], *, kind: str = "non_cuda") -> CommandSpec:
    temp_root = cfg["paths"]["temp_root"]
    run_root = Path(cfg["paths"]["run_root"]) / "non_branchus" / "latest"
    summary_json = Path(cfg["paths"]["run_root"]) / "non_branchus" / "latest_summary.json"
    return CommandSpec(
        name="si_non_cuda_pipeline",
        command=[
            "uv",
            "run",
            "python",
            _str(cfg["scripts"].get("si_non_branchus", cfg["scripts"]["si_non_cuda"])),
            "--run-root",
            str(run_root),
            "--summary-json",
            str(summary_json),
        ],
        env={"WANDB_MODE": "disabled"},
        log_path=_command_log(temp_root, f"si_{kind}_non_cuda_pipeline"),
    )


def build_env_contract(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    plain = _to_plain(cfg)
    dataset = plain["dataset"]
    env_fields = [
        "duckdb_env",
        "schema_env",
        "encryption_key_env",
        "branchus_config_env",
    ]
    env_vars = {field: _str(dataset[field]) for field in env_fields if dataset.get(field)}
    return {
        "dataset": dataset["name"],
        "source_format": dataset.get("source_format"),
        "direct_data_read_by_agent": False,
        "env_vars": {
            field: {"name": name, "present": bool(os.environ.get(name))}
            for field, name in env_vars.items()
        },
    }


def write_env_contract(cfg: DictConfig | dict[str, Any], output_path: Path | str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_env_contract(cfg), indent=2, sort_keys=True) + "\n")
    return output


def write_step_record(cfg: DictConfig | dict[str, Any], output_path: Path | str) -> Path:
    plain = _to_plain(cfg)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": plain["dataset"]["name"],
        "model": plain["model"]["name"],
        "phase": plain["phase"]["name"],
        "status": "plan_only",
        "direct_data_read": False,
        "reason": "Plan-only helper retained for compatibility; SI data access is handled by explicit non-CUDA pipeline commands.",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def _hf_ehr_env(cfg: dict[str, Any], model: dict[str, Any], resource: dict[str, Any]) -> dict[str, str]:
    hf = resource["hf_ehr"].copy()
    if model["name"] == "hf_ehr_mamba":
        hf.update(resource.get("mamba", {}))
    return {
        "FOMOH_HF_EHR_MODELS": _str(model["hf_ehr_arch"]),
        "FOMOH_HF_EHR_PER_CLASS_PER_SPLIT": "0",
        "FOMOH_HF_EHR_BATCH_SIZE": _str(hf["batch_size"]),
        "FOMOH_HF_EHR_TOKENIZE_BATCH_SIZE": _str(hf["tokenize_batch_size"]),
        "FOMOH_HF_EHR_MEDS_READER_THREADS": _str(hf["meds_reader_threads"]),
        "FOMOH_HF_EHR_AUTOCAST": "1" if hf.get("autocast", False) else "0",
        "FOMOH_HF_EHR_AUTOCAST_DTYPE": _str(hf["autocast_dtype"]),
        "FOMOH_HF_EHR_BENCH_TMP": str(Path(cfg["paths"]["tmpfs_root"]) / "hf_ehr_active_task_benchmark"),
        "WANDB_MODE": "disabled",
    }


def _kernel_install_command(cfg: dict[str, Any], model: dict[str, Any], resource: dict[str, Any]) -> CommandSpec:
    kernel_build = resource["kernel_build"]
    env = {
        "MAX_JOBS": _str(kernel_build["max_jobs"]),
        "CUDA_HOME": _str(kernel_build["cuda_home"]),
        "TORCH_CUDA_ARCH_LIST": _str(kernel_build["torch_cuda_arch_list"]),
        "UV_CACHE_DIR": str(Path(cfg["paths"]["tmpfs_root"]) / "uv_cache"),
    }
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        _str(model["python"]),
        "--no-deps",
        "--no-build-isolation",
        *model.get("kernel_packages", []),
    ]
    return CommandSpec(
        name=f"{model['name']}_kernel_install",
        command=command,
        env=env,
        log_path=_command_log(cfg["paths"]["temp_root"], f"{model['name']}_kernel_install"),
        requires_gpu=False,
    )


def _validate_commands(cfg: dict[str, Any]) -> list[CommandSpec]:
    if cfg["dataset"]["name"] == "fomoh_si":
        return [_env_contract_command(cfg)]
    temp_root = cfg["paths"]["temp_root"]
    return [
        CommandSpec(
            name="validate_resources",
            command=[
                "uv",
                "run",
                "python",
                "-m",
                "ehr_foundation_model_benchmark.fomoh_mimic.cli",
                "validate-resources",
                "--output-json",
                str(Path(temp_root) / "resource_summary.json"),
            ],
            log_path=_command_log(temp_root, "validate_resources"),
        ),
        CommandSpec(
            name="task_manifest_check",
            command=[
                "uv",
                "run",
                "python",
                "-m",
                "json.tool",
                _str(cfg["dataset"]["task_manifest"]),
            ],
            log_path=_command_log(temp_root, "task_manifest_check"),
        ),
    ]


def _report_commands(cfg: dict[str, Any]) -> list[CommandSpec]:
    if cfg["dataset"]["name"] == "fomoh_si":
        return [_env_contract_command(cfg), _si_non_cuda_command(cfg, kind="report")]
    run_root = Path(cfg["paths"]["run_root"])
    temp_root = cfg["paths"]["temp_root"]
    scripts = cfg["scripts"]
    return [
        CommandSpec(
            name="write_benchmark_status",
            command=["uv", "run", "python", _str(scripts["status_report"])],
            log_path=_command_log(temp_root, "write_benchmark_status"),
        ),
        CommandSpec(
            name="write_benchmark_profile",
            command=["uv", "run", "python", _str(scripts["benchmark_profile"])],
            log_path=_command_log(temp_root, "write_benchmark_profile"),
        ),
        CommandSpec(
            name="write_phase1_summary",
            command=[
                "uv",
                "run",
                "python",
                "-m",
                _str(scripts["phase1_summary_module"]),
                "phase1-summary",
                "--status-json",
                str(run_root / "benchmark_scale_status.json"),
                "--output-md",
                str(run_root / "phase1_benchmark_summary.md"),
            ],
            log_path=_command_log(temp_root, "write_phase1_summary"),
        ),
    ]


def _full_active_commands(cfg: dict[str, Any]) -> list[CommandSpec]:
    if cfg["dataset"]["name"] == "fomoh_si":
        return [_env_contract_command(cfg), _si_non_cuda_command(cfg, kind="full_active")]
    model = cfg["model"]
    temp_root = cfg["paths"]["temp_root"]
    scripts = cfg["scripts"]
    commands: list[CommandSpec] = []
    if model.get("kernel_packages"):
        commands.append(_kernel_install_command(cfg, model, cfg["resource"]))
    if model["family"] == "hf_ehr":
        commands.append(
            CommandSpec(
                name="hf_ehr_full_active",
                command=[_str(model["python"]), _str(scripts["hf_ehr_benchmark"])],
                env=_hf_ehr_env(cfg, model, cfg["resource"]),
                log_path=_command_log(temp_root, f"{model['name']}_full_active"),
                requires_gpu=True,
            )
        )
    elif model["family"] == "medstab":
        commands.append(
            CommandSpec(
                name="medstab_full_active",
                command=["uv", "run", "python", _str(scripts["medstab_full_active"])],
                log_path=_command_log(temp_root, "medstab_full_active"),
            )
        )
        commands.append(
            CommandSpec(
                name="medstab_full_metrics",
                command=["uv", "run", "python", _str(scripts["medstab_metrics"])],
                log_path=_command_log(temp_root, "medstab_full_metrics"),
            )
        )
    else:
        commands.append(
            CommandSpec(
                name="downstream_active_export",
                command=["uv", "run", "python", _str(scripts["downstream_export"])],
                log_path=_command_log(temp_root, f"{model['name']}_downstream_export"),
            )
        )
        commands.append(
            CommandSpec(
                name="downstream_active_smoke",
                command=["uv", "run", "python", _str(scripts["downstream_smoke"])],
                log_path=_command_log(temp_root, f"{model['name']}_downstream_smoke"),
            )
        )
    return commands


def _json_evidence_command(cfg: dict[str, Any], name: str, path: Path | str) -> CommandSpec:
    temp_root = cfg["paths"]["temp_root"]
    return CommandSpec(
        name=name,
        command=["uv", "run", "python", "-m", "json.tool", _str(path)],
        log_path=_command_log(temp_root, name),
    )


def _smoke_commands(cfg: dict[str, Any]) -> list[CommandSpec]:
    if cfg["dataset"]["name"] == "fomoh_si":
        if bool(cfg["model"].get("requires_gpu", False)):
            raise ValueError(f"SI smoke is restricted to non-CUDA models until CUDA is vacant: {cfg['model']['name']}")
        return [_env_contract_command(cfg), _si_non_cuda_command(cfg, kind="smoke")]
    model = cfg["model"]
    temp_root = cfg["paths"]["temp_root"]
    run_root = Path(cfg["paths"]["run_root"])
    scripts = cfg["scripts"]
    evidence_root = run_root / "model_smokes"
    commands: list[CommandSpec] = []

    if model.get("kernel_packages"):
        commands.append(_kernel_install_command(cfg, model, cfg["resource"]))

    if model["family"] == "hf_ehr":
        commands.append(
            CommandSpec(
                name="hf_ehr_architecture_smokes",
                command=[_str(model["python"]), _str(scripts["hf_ehr_smoke"])],
                env={"WANDB_MODE": "disabled"},
                log_path=_command_log(temp_root, "hf_ehr_architecture_smokes"),
                requires_gpu=True,
            )
        )
        commands.append(
            CommandSpec(
                name="hf_ehr_all_task_probe_smokes",
                command=[_str(model["python"]), _str(scripts["hf_ehr_probe_smoke"])],
                env={"WANDB_MODE": "disabled"},
                log_path=_command_log(temp_root, "hf_ehr_all_task_probe_smokes"),
                requires_gpu=True,
            )
        )
    elif model["family"] == "medstab":
        commands.append(
            CommandSpec(
                name="medstab_all_task_smoke",
                command=["uv", "run", "python", _str(scripts["medstab_smoke"])],
                log_path=_command_log(temp_root, "medstab_all_task_smoke"),
            )
        )
    elif model["family"] == "motor":
        commands.append(_json_evidence_command(cfg, "motor_prepare_smoke_evidence", evidence_root / "motor_prepare_smoke_300_skip_empty.json"))
        commands.append(_json_evidence_command(cfg, "motor_pretrain_smoke_evidence", evidence_root / "motor_pretrain_smoke_retry_xformers.json"))
        commands.append(
            CommandSpec(
                name="motor_all_task_probe_smoke",
                command=[_str(model["python"]), _str(scripts["motor_probe_smoke"])],
                log_path=_command_log(temp_root, "motor_all_task_probe_smoke"),
                requires_gpu=True,
            )
        )
    elif model["name"] == "cehrgpt":
        commands.append(
            CommandSpec(
                name="cehrgpt_pretrain_smoke",
                command=[_str(model["python"]), _str(scripts["cehrgpt_smoke"])],
                log_path=_command_log(temp_root, "cehrgpt_pretrain_smoke"),
            )
        )
        commands.append(_json_evidence_command(cfg, "cehrgpt_smoke_evidence", evidence_root / "cehrgpt_smoke_retry.json"))
    elif model["name"] == "cehrbert":
        commands.append(_json_evidence_command(cfg, "cehrbert_smoke_evidence", evidence_root / "cehrbert_smoke.json"))
    elif model["name"] == "corebehrt":
        commands.append(_json_evidence_command(cfg, "corebehrt_smoke_evidence", evidence_root / "corebehrt_smoke.json"))
    else:
        commands.append(_json_evidence_command(cfg, f"{model['name']}_smoke_evidence", evidence_root / f"{model['name']}_smoke.json"))

    return commands


def compose_run_plan(cfg: DictConfig | dict[str, Any]) -> RunPlan:
    plain = _to_plain(cfg)
    phase = plain["phase"]["name"]
    model = plain["model"]
    if phase == "validate":
        commands = _validate_commands(plain)
    elif phase == "report":
        commands = _report_commands(plain)
    elif phase == "full_active":
        commands = _full_active_commands(plain)
    elif phase == "smoke":
        commands = _smoke_commands(plain)
    elif phase == "non_cuda":
        if plain["dataset"]["name"] != "fomoh_si":
            raise ValueError(f"Unsupported non_cuda dataset: {plain['dataset']['name']}")
        commands = [_env_contract_command(plain), _si_non_cuda_command(plain)]
    else:
        raise ValueError(f"Unsupported phase: {phase}")
    requires_gpu = bool(plain["phase"].get("requires_gpu", False) or any(command.requires_gpu for command in commands))
    return RunPlan(
        dataset=_str(plain["dataset"]["name"]),
        model=_str(model["name"]),
        phase=_str(phase),
        dry_run=bool(plain.get("dry_run", True)),
        execute=bool(plain.get("execute", False)),
        requires_gpu=requires_gpu,
        run_root=_str(plain["paths"]["run_root"]),
        temp_root=_str(plain["paths"]["temp_root"]),
        tmpfs_root=_str(plain["paths"]["tmpfs_root"]),
        commands=commands,
        gpu_min_free_mib=int(plain["resource"]["gpu_gate"]["min_free_mib"]),
        gpu_max_process_mib=int(plain["resource"]["gpu_gate"]["max_unrelated_process_mib"]),
    )


def render_commands(plan: RunPlan) -> list[CommandSpec]:
    return plan.commands


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:=+-")
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _shell_line(command: CommandSpec) -> str:
    prefix = [f"{key}={_shell_quote(value)}" for key, value in sorted(command.env.items())]
    return " ".join(prefix + [_shell_quote(part) for part in command.command])


def write_run_plan(plan: RunPlan, output_dir: Path | str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = asdict(plan)
    stem = f"{plan.phase}_{plan.model}" if plan.phase in {"smoke", "full_active"} else plan.phase
    json_path = output / f"{stem}_plan.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    shell_path = output / f"{stem}_commands.sh"
    shell_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    shell_lines.extend(_shell_line(command) for command in plan.commands)
    shell_path.write_text("\n".join(shell_lines) + "\n")
    return json_path


def check_gpu_gate(plan: RunPlan) -> None:
    if not plan.requires_gpu:
        return
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"GPU gate failed: {completed.stdout.strip()}")
    free_mib = max(int(line.strip()) for line in completed.stdout.splitlines() if line.strip())
    if free_mib < plan.gpu_min_free_mib:
        raise RuntimeError(f"GPU gate failed: only {free_mib} MiB free")
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode == 0:
        usages = [int(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
        if any(value > plan.gpu_max_process_mib for value in usages):
            raise RuntimeError(f"GPU gate failed: process above {plan.gpu_max_process_mib} MiB")


def execute_plan(plan: RunPlan) -> None:
    if not plan.execute:
        return
    check_gpu_gate(plan)
    env_base = os.environ.copy()
    for command in plan.commands:
        env = env_base | command.env
        log_path = Path(command.log_path) if command.log_path else None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as log:
                log.write("$ " + _shell_line(command) + "\n")
                log.flush()
                completed = subprocess.run(command.command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        else:
            completed = subprocess.run(command.command, env=env, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Command failed ({completed.returncode}): {command.name}")


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "configs"


def _pop_path_arg(args: list[str], flag: str) -> Path | None:
    if flag not in args:
        return None
    index = args.index(flag)
    try:
        value = args[index + 1]
    except IndexError as exc:
        raise SystemExit(f"{flag} requires a path") from exc
    del args[index : index + 2]
    return Path(value)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    env_contract_path = _pop_path_arg(args, "--write-env-contract")
    step_record_path = _pop_path_arg(args, "--write-step-record")
    overrides: list[str] = []
    if "--dry-run" in args:
        args.remove("--dry-run")
        overrides.extend(["dry_run=true", "execute=false"])
    if "--execute" in args:
        args.remove("--execute")
        overrides.extend(["dry_run=false", "execute=true"])
    overrides.extend(args)
    with initialize_config_dir(config_dir=str(_config_dir()), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    if env_contract_path is not None:
        print(write_env_contract(cfg, env_contract_path))
        return
    if step_record_path is not None:
        print(write_step_record(cfg, step_record_path))
        return
    plan = compose_run_plan(cfg)
    output_path = write_run_plan(plan, Path(plan.run_root) / "hydra_plans")
    print(f"wrote {output_path}")
    for command in plan.commands:
        print(_shell_line(command))
    if plan.execute:
        execute_plan(plan)


if __name__ == "__main__":
    main()

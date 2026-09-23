#!/usr/bin/env python3
"""Render one Task1 KEY=value contract into a direct launcher."""

from __future__ import annotations

import argparse
import shlex
import shutil
from pathlib import Path


def read_contract(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def output_directory(values: dict[str, str]) -> str:
    if values.get("RUN_OUTPUT_DIR"):
        return values["RUN_OUTPUT_DIR"]
    leaf = "eval" if values["RUN_MODE"] == "eval_only" else "train"
    return str(Path(values["RUNS_DIR"]) / values["RUN_ID"] / leaf)


def milestones(values: dict[str, str]) -> str:
    explicit = values.get("CHECKPOINT_MILESTONE_STEPS", "")
    if explicit:
        return f"[{explicit}]"
    cadence = int(values.get("CHECKPOINT_SAVE_EVERY_N_STEPS", "0") or 0)
    source = int(values.get("EXPECTED_CHECKPOINT_STEP", "0") or 0)
    target = int(values["TRAIN_MAX_STEPS"])
    if not cadence:
        return f"[{target}]"
    steps = list(range(source + cadence, target + 1, cadence))
    if not steps or steps[-1] != target:
        steps.append(target)
    return "[" + ",".join(map(str, steps)) + "]"


def write_env(path: Path, values: dict[str, str]) -> None:
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8")


def legacy_train_values(values: dict[str, str], run_output: str) -> dict[str, str]:
    result = {
        "HARDWARE_PROFILE": values["HARDWARE_PROFILE"],
        "TRAINER_DEVICES": "1",
        "GLOBAL_BATCH_SIZE": values["TRAIN_GLOBAL_BATCH_SIZE"],
        "MICRO_BATCH_SIZE": values["TRAIN_MICRO_BATCH_SIZE"],
        "ACCUMULATE_GRAD_BATCHES": values["TRAIN_GRADIENT_ACCUMULATION"],
        "TRAINING_TIME_SAMPLING": values["TRAIN_TIME_SAMPLING"],
        "RUN_DIR": run_output,
        "MAX_STEPS": values["TRAIN_MAX_STEPS"] if values["RUN_MODE"] == "fresh" else "",
        "EXPERIMENT_RUN_ID": values["RUN_ID"] if values["RUN_MODE"] == "fresh" else "",
        "LEARNING_RATE": values["TRAIN_LEARNING_RATE"],
        "OPTIM_BETA1": values["TRAIN_OPTIM_BETA1"],
        "OPTIM_BETA2": values["TRAIN_OPTIM_BETA2"],
        "OPTIM_EPS": values["TRAIN_OPTIM_EPS"],
        "OPTIM_WEIGHT_DECAY": values["TRAIN_OPTIM_WEIGHT_DECAY"],
        "EMA_DECAY": values["TRAIN_EMA_DECAY"],
        "WARMUP_STEPS": values["TRAIN_WARMUP_STEPS"],
        "TOKEN_BIAS_WARMUP_STEPS": values["TRAIN_TOKEN_BIAS_WARMUP_STEPS"],
        "EVAL_BATCH_SIZE": values["TRAIN_EVAL_BATCH_SIZE"],
        "CHECKPOINT_EVERY_N_STEPS": values["CHECKPOINT_SAVE_EVERY_N_STEPS"],
        "CHECKPOINT_MILESTONE_STEPS": values["CHECKPOINT_MILESTONE_STEPS"],
        "RESUME_CHECKPOINT_PATH": values["CHECKPOINT_PATH"],
        "SOURCE_LEARNING_RATE": values["SOURCE_LEARNING_RATE"],
        "SOURCE_GLOBAL_STEP": values["EXPECTED_CHECKPOINT_STEP"],
        "TASK1_VARIANT": values["TASK1_VARIANT"],
        "TARGET_MAX_STEPS": values["TRAIN_MAX_STEPS"] if values["RUN_MODE"] == "full_state" else "",
        "TASK1_RUN_ID": values["RUN_ID"] if values["RUN_MODE"] == "full_state" else "",
        "RESUME_CONSTANT_LEARNING_RATE": "",
        "RESUME_TARGET_LEARNING_RATE": "",
        "RESUME_LR_TRANSITION_STEPS": "",
        "RESUME_LR_TRANSITION_SCHEDULE": "linear",
    }
    resume_mode = values.get("RESUME_LR_MODE", "preserve")
    if resume_mode == "constant":
        result["RESUME_CONSTANT_LEARNING_RATE"] = values["RESUME_TARGET_LEARNING_RATE"]
    elif resume_mode in {"linear", "cosine"}:
        result["RESUME_TARGET_LEARNING_RATE"] = values["RESUME_TARGET_LEARNING_RATE"]
        result["RESUME_LR_TRANSITION_STEPS"] = values["RESUME_LR_TRANSITION_STEPS"]
        result["RESUME_LR_TRANSITION_SCHEDULE"] = resume_mode
    return result


def runtime_environment(
        values: dict[str, str], run_output: str,
        legacy_path: Path | None) -> dict[str, str]:
    environment = {
        "FLM_PROJECT_DIR": values["PROJECT_DIR"],
        "FLM_STORAGE_DIR": values["PERSISTENT_ROOT"],
        "VIRTUAL_ENV": values["VENV_DIR"],
        "PIP_CACHE_DIR": str(Path(values["CACHE_DIR"]) / "pip"),
        "HF_HOME": str(Path(values["CACHE_DIR"]) / "huggingface"),
        "HF_DATASETS_CACHE": str(Path(values["CACHE_DIR"]) / "huggingface/datasets"),
        "HUGGINGFACE_HUB_CACHE": str(Path(values["CACHE_DIR"]) / "huggingface/hub"),
        "TORCH_HOME": str(Path(values["CACHE_DIR"]) / "torch"),
        "WANDB_DIR": str(Path(values["RUNS_DIR"]) / "wandb"),
        "FLM_PACKED_DATA_DIR": values["DATA_DIR"],
        "FLM_TOKENIZER_PATH": values["TOKENIZER_DIR"],
        "MODEL_CONFIG": values["MODEL_CONFIG"],
        "DATA_CONFIG": values["DATA_CONFIG"],
        "EVAL_MODEL_DIR": values["EVALUATOR_DIR"],
        "RUN_DIR": run_output,
    }
    if values["RUN_MODE"] in {"fresh", "full_state"}:
        environment.update({
            "TASK1_TRAIN_CONFIG": str(legacy_path),
            "LR_SCHEDULER_CONFIG": values["TRAIN_LR_SCHEDULER_CONFIG"],
            "VALIDATION_INTERVAL_STEPS": values["TRAIN_VALIDATION_INTERVAL_STEPS"],
            "LIMIT_VAL_BATCHES": values["TRAIN_LIMIT_VAL_BATCHES"],
            "LOG_EVERY_N_STEPS": values["TRAIN_LOG_EVERY_N_STEPS"],
            "THROUGHPUT_EVERY_N_STEPS": values["TRAIN_THROUGHPUT_EVERY_N_STEPS"],
            "MILESTONE_STEPS": milestones(values),
            "MILESTONE_LAST_EVERY_N_STEPS": values["CHECKPOINT_ROLLING_EVERY_N_STEPS"],
            "MILESTONE_SAVE_WEIGHTS_ONLY": values["CHECKPOINT_SAVE_WEIGHTS_ONLY"],
        })
        if values["RUN_MODE"] == "full_state" and values.get("RESUME_LR_MODE") == "preserve":
            environment["RESUME_PRESERVE_CHECKPOINT_LR_SCHEDULER"] = "1"
    else:
        environment.update({
            "LOSS_VARIANT": "task1_a" if values["TASK1_VARIANT"] == "A" else "task1_c",
            "CHECKPOINT_PATH": values["CHECKPOINT_PATH"],
            "NUM_SAMPLES": values["EVAL_NUM_SAMPLES"],
            "SAMPLING_STEPS": values["EVAL_NFE"],
            "EVAL_BATCH_SIZE": values["EVAL_BATCH_SIZE"],
            "EVAL_SEED": values["EVAL_SEED"],
            "SAMPLING_SOLVER": values["EVAL_SOLVER"],
            "SAMPLING_TEMPERATURE": values["EVAL_TEMPERATURE"],
            "TASK1_TIME_GRID": values["EVAL_TIME_GRID"],
            "DISABLE_EMA": "false" if values["CHECKPOINT_USE_EMA"] == "true" else "true",
            "GEN_PPL_COMPARABLE": values["EVAL_COMPARABLE"],
            "GEN_PPL_PROTOCOL_ID": values["GEN_PPL_PROTOCOL_ID"],
            "COMPUTE_GEN_PPL": "true",
            "EVALUATION_PHASE": f"task1_runpack_{values['RUN_ID']}",
        })
    return environment


def prepare(source: Path) -> Path:
    values = read_contract(source)
    run_output = output_directory(values)
    contracts = Path(values["CONTRACTS_DIR"])
    contract_path = contracts / f"{values['RUN_ID']}.env"
    render_dir = contracts / "rendered" / values["RUN_ID"]
    render_dir.mkdir(parents=True, exist_ok=True)
    if source.resolve() != contract_path.resolve():
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, contract_path)
    resolved = dict(values, RUN_OUTPUT_DIR=run_output)
    write_env(render_dir / "resolved.env", resolved)
    legacy_path = None
    if values["RUN_MODE"] in {"fresh", "full_state"}:
        legacy_path = render_dir / "train.env"
        write_env(legacy_path, legacy_train_values(values, run_output))
    environment = runtime_environment(values, run_output, legacy_path)
    scripts = Path(values["PROJECT_DIR"]) / "scripts"
    if values["RUN_MODE"] == "fresh":
        command = [str(scripts / "train_task1_a.sh"), values["HARDWARE_PROFILE"]]
    elif values["RUN_MODE"] == "full_state":
        command = [str(scripts / "resume_task1_vocab_mse.sh")]
    else:
        command = [str(scripts / "eval_owt_128_langflow_hybrid.sh")]
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.extend(
        f"export {key}={shlex.quote(value)}"
        for key, value in environment.items())
    lines.extend(("", "exec " + " ".join(map(shlex.quote, command)), ""))
    launch = render_dir / "launch.sh"
    launch.write_text("\n".join(lines), encoding="utf-8")
    launch.chmod(0o755)
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare",))
    parser.add_argument("contract", type=Path)
    args = parser.parse_args()
    launch = prepare(args.contract)
    print(f"LAUNCH={launch}")


if __name__ == "__main__":
    main()

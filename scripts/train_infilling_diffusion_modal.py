#!/usr/bin/env python3
"""Run infilling diffusion training on Modal.

This is intentionally a thin wrapper around ``scripts/train_infilling_diffusion.py``.
Modal-specific concerns live here; training arguments are forwarded directly to
the existing training script.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

import modal

APP_NAME = "leandojo-infilling-diffusion-train"
WORKSPACE_DIR = "/workspace"
CACHE_ROOT = "/cache"
OUTPUT_ROOT = "/outputs"
DEFAULT_WANDB_RUN_NAME = "infilling-mdm-dream7b-april"
DEFAULT_GPU_TYPE = "H100"
DEFAULT_GPU_COUNT = 4
SUPPORTED_GPU_TYPES = ("H100", "A100-80GB", "L40S", "A10G")
SUPPORTED_GPU_COUNTS = (1, 2, 4, 8)

app = modal.App(APP_NAME)

cache_volume = modal.Volume.from_name("leandojo-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("leandojo-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "curl",
        "wget",
        "build-essential",
    )
    .uv_pip_install(
        "uv",
        "torch>=2.0.0",
        "transformers>=4.54,<5",
        "peft>=0.17.0",
        "datasets>=4.0.0",
        "trl>=0.25.1",
        "accelerate>=0.34.0",
        "deepspeed>=0.7.0",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
        "wandb>=0.17.0",
        "python-dotenv>=0.19.0",
        "filelock>=3.8.0",
    )
    .workdir(WORKSPACE_DIR)
    .add_local_python_source("lean_dojo_v2")
    .add_local_dir("scripts", remote_path=f"{WORKSPACE_DIR}/scripts")
)

if Path("datasets").exists():
    image = image.add_local_dir("datasets", remote_path=f"{WORKSPACE_DIR}/datasets")


def _safe_dir_component(value: str) -> str:
    normalized = value.strip().replace("/", "-").replace("\\", "-").replace(" ", "-")
    return normalized or "run"


def _get_arg_value(args: list[str], flag: str) -> str | None:
    for index, value in enumerate(args):
        if value == flag and index + 1 < len(args):
            return args[index + 1]
    return None


def _normalize_train_args(train_args: list[str]) -> list[str]:
    normalized = list(train_args)
    if _get_arg_value(normalized, "--output-dir") is not None:
        return normalized

    wandb_run_name = _get_arg_value(normalized, "--wandb-run-name")
    output_dir = (
        f"{OUTPUT_ROOT}/{_safe_dir_component(wandb_run_name or DEFAULT_WANDB_RUN_NAME)}"
    )
    normalized.extend(["--output-dir", output_dir])
    return normalized


def _build_train_command(train_args: list[str], gpu_count: int) -> list[str]:
    if gpu_count > 1:
        return [
            "uv",
            "run",
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={gpu_count}",
            "scripts/train_infilling_diffusion.py",
            *train_args,
        ]
    return ["uv", "run", "python", "scripts/train_infilling_diffusion.py", *train_args]


def _make_training_function(gpu_type: str, gpu_count: int):
    gpu_spec = gpu_type if gpu_count == 1 else f"{gpu_type}:{gpu_count}"
    function_name = (
        f"train_infilling_diffusion_{gpu_type.lower().replace('-', '_')}_{gpu_count}x"
    )

    @app.function(
        name=function_name,
        serialized=True,
        image=image,
        gpu=gpu_spec,
        cpu=32,
        memory=131072,
        timeout=48 * 60 * 60,
        volumes={
            CACHE_ROOT: cache_volume,
            OUTPUT_ROOT: output_volume,
        },
    )
    def _run(train_args: list[str]) -> str:
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["HF_HOME"] = f"{CACHE_ROOT}/huggingface"
        env["HUGGINGFACE_HUB_CACHE"] = f"{CACHE_ROOT}/huggingface/hub"
        env["TRANSFORMERS_CACHE"] = f"{CACHE_ROOT}/huggingface/transformers"
        env["WANDB_ROOT"] = f"{CACHE_ROOT}/wandb"
        env["WANDB_DIR"] = f"{CACHE_ROOT}/wandb/runs"
        env["WANDB_DATA_DIR"] = f"{CACHE_ROOT}/wandb/data"
        env["WANDB_CACHE_DIR"] = f"{CACHE_ROOT}/wandb/cache"
        env["WANDB_ARTIFACT_DIR"] = f"{CACHE_ROOT}/wandb/artifacts"
        env["PYTHONPATH"] = WORKSPACE_DIR + (
            f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
        )

        normalized_train_args = _normalize_train_args(train_args)
        cmd = _build_train_command(normalized_train_args, gpu_count=gpu_count)

        print("Launching training command:")
        print(" ".join(shlex.quote(part) for part in cmd))
        subprocess.run(cmd, cwd=WORKSPACE_DIR, env=env, check=True)

        return _get_arg_value(normalized_train_args, "--output-dir") or OUTPUT_ROOT

    return _run


TRAINING_FUNCTIONS = {
    (gpu_type, gpu_count): _make_training_function(gpu_type, gpu_count)
    for gpu_type in SUPPORTED_GPU_TYPES
    for gpu_count in SUPPORTED_GPU_COUNTS
}


def _select_training_function(gpu_type: str, gpu_count: int):
    key = (gpu_type, gpu_count)
    if key not in TRAINING_FUNCTIONS:
        raise ValueError(
            f"Unsupported Modal GPU config: gpu_type={gpu_type!r}, gpu_count={gpu_count!r}."
        )
    return TRAINING_FUNCTIONS[key]


@app.local_entrypoint()
def main(
    gpu_type: str = DEFAULT_GPU_TYPE,
    gpu_count: int = DEFAULT_GPU_COUNT,
    train_args: str = "",
) -> None:
    forwarded_args = shlex.split(train_args)
    if not forwarded_args:
        raise ValueError("Pass training flags via --train-args.")

    run_training = _select_training_function(gpu_type, gpu_count)
    output_path = run_training.remote(forwarded_args)
    print(f"\nModel saved to {output_path}")
    print(f"Ran on Modal with {gpu_count}x {gpu_type}.")


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run infilling diffusion training on Modal.",
    )
    parser.add_argument(
        "--gpu-type",
        choices=SUPPORTED_GPU_TYPES,
        default=DEFAULT_GPU_TYPE,
        help="Modal GPU type.",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        choices=SUPPORTED_GPU_COUNTS,
        default=DEFAULT_GPU_COUNT,
        help="Number of GPUs in the Modal container.",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to scripts/train_infilling_diffusion.py.",
    )
    args = parser.parse_args()

    forwarded_args = list(args.train_args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]
    if not forwarded_args:
        raise ValueError("Pass training flags after `--`.")

    run_training = _select_training_function(args.gpu_type, args.gpu_count)
    with app.run():
        output_path = run_training.remote(forwarded_args)
    print(f"\nModel saved to {output_path}")
    print(f"Ran on Modal with {args.gpu_count}x {args.gpu_type}.")


if __name__ == "__main__":
    cli_main()

#!/usr/bin/env python3
"""Run diffusion step sweep evaluation on Modal.

Thin wrapper around ``scripts/eval_diffusion_steps_vs_accuracy.py`` that handles
Modal-specific image, volume, and path normalization concerns.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "leandojo-diffusion-steps-eval"
WORKSPACE_DIR = "/workspace"
CACHE_ROOT = "/cache"
OUTPUT_ROOT = "/outputs"
MODEL_FLAGS = {"--model"}
OUTPUT_FLAGS = {"--json-output", "--csv-output"}

app = modal.App(APP_NAME)

cache_volume = modal.Volume.from_name("leandojo-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("leandojo-outputs", create_if_missing=True)
github_access_token = os.getenv("GITHUB_ACCESS_TOKEN", "").strip()
modal_secrets = []
if github_access_token:
    modal_secrets.append(
        modal.Secret.from_dict(
            {
                "GITHUB_ACCESS_TOKEN": github_access_token,
                "GITHUB_TOKEN": github_access_token,
            }
        )
    )

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


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _looks_like_repo_outputs_path(value: str) -> bool:
    if not value:
        return False
    path = Path(value)
    return bool(path.parts) and path.parts[0] == "outputs"


def _normalize_model_or_output_path(value: str) -> str:
    if not _looks_like_repo_outputs_path(value):
        return value
    relative = Path(value).relative_to("outputs")
    return str(Path(OUTPUT_ROOT) / relative)


def _flag_has_value(args: list[str], flag: str) -> bool:
    return any(value == flag for value in args)


def _normalize_eval_args(raw_args: list[str]) -> list[str]:
    normalized = list(raw_args)
    for index, value in enumerate(normalized[:-1]):
        if value in MODEL_FLAGS or value in OUTPUT_FLAGS:
            normalized[index + 1] = _normalize_model_or_output_path(normalized[index + 1])

    if not _flag_has_value(normalized, "--json-output"):
        normalized.extend(
            [
                "--json-output",
                f"{OUTPUT_ROOT}/benchmarks/diffusion_steps_vs_accuracy_{_timestamp()}.json",
            ]
        )
    if not _flag_has_value(normalized, "--csv-output"):
        normalized.extend(
            [
                "--csv-output",
                f"{OUTPUT_ROOT}/benchmarks/diffusion_steps_vs_accuracy_{_timestamp()}.csv",
            ]
        )
    return normalized


def _build_command(forwarded_args: list[str]) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "scripts/eval_diffusion_steps_vs_accuracy.py",
        *forwarded_args,
    ]


@app.function(
    serialized=True,
    image=image,
    gpu="H100",
    cpu=16,
    memory=131072,
    timeout=12 * 60 * 60,
    volumes={
        CACHE_ROOT: cache_volume,
        OUTPUT_ROOT: output_volume,
    },
    secrets=modal_secrets,
)
def run_step_sweep(eval_args: list[str]) -> None:
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HOME"] = f"{CACHE_ROOT}/huggingface"
    env["HF_HUB_CACHE"] = f"{CACHE_ROOT}/huggingface/hub"
    env["TRANSFORMERS_CACHE"] = f"{CACHE_ROOT}/huggingface/transformers"
    env["PYTHONPATH"] = WORKSPACE_DIR + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )

    normalized_args = _normalize_eval_args(eval_args)
    cmd = _build_command(normalized_args)

    print("Running on Modal GPU: 1x H100")
    print("Launching step-sweep command:")
    print(" ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=WORKSPACE_DIR, env=env, check=True)


@app.local_entrypoint()
def main(eval_args: str = "") -> None:
    forwarded_args = shlex.split(eval_args)
    run_step_sweep.remote(forwarded_args)


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diffusion step sweep evaluation on Modal.",
    )
    parser.add_argument(
        "eval_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to scripts/eval_diffusion_steps_vs_accuracy.py.",
    )
    args = parser.parse_args()

    forwarded_args = list(args.eval_args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    with app.run():
        run_step_sweep.remote(forwarded_args)


if __name__ == "__main__":
    cli_main()

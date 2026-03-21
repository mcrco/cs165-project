#!/usr/bin/env python3
"""Recompute eval loss for early infilling checkpoints and merge with train loss."""

from __future__ import annotations

import argparse
import gc
import json
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from peft import PeftConfig
from transformers import AutoTokenizer, Trainer, TrainingArguments

from infilling_eval_utils import (
    effective_dtype,
    load_ar_model,
    load_diffusion_model_bundle,
    resolve_device,
    resolve_eval_dataset_path,
    write_csv,
    write_json,
)
from lean_dojo_v2.diffusion import create_diffusion_training_objective
from lean_dojo_v2.diffusion.loading import load_diffusion_tokenizer, resolve_diffusion_family
from lean_dojo_v2.diffusion.token_utils import resolve_mask_token_id
from lean_dojo_v2.trainer.diffusion_sft_trainer import MdlmTrainer
from lean_dojo_v2.trainer.infilling_autoregressive_trainer import (
    InfillingARCollator,
    InfillingARDataset,
)
from lean_dojo_v2.trainer.infilling_diffusion_trainer import InfillingMDMDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIRS = [
    # REPO_ROOT / "outputs" / "infilling-ar-april-thme-train10k-val1k",
    # REPO_ROOT / "outputs" / "infilling-diffusion-april-thme-train10k-val1k",
    REPO_ROOT / "outputs" / "infilling-diffusion-april-thme-train100k-val10k",
    REPO_ROOT / "outputs" / "infilling-ar-april-thme-train100k-val10k",
]
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis" / "checkpoint_loss_curves"
CHECKPOINT_PREFIX = "checkpoint-"
WEIGHT_FILENAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
)
DEFAULT_DATASET_PATH = "datasets/april/leandojo_infilling/thme_1m_100k.val.jsonl"
EPOCH_TOLERANCE = 1e-4


@dataclass(frozen=True)
class CheckpointInfo:
    experiment_dir: Path
    experiment_name: str
    checkpoint_dir: Path
    checkpoint_name: str
    checkpoint_step: int
    epoch: float
    global_step: int
    model_family: str
    base_model_name_or_path: str
    has_weights: bool
    weight_filename: Optional[str]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute validation loss for early infilling checkpoints and merge it "
            "with locally reconstructed training-loss history."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        action="append",
        default=[],
        help=(
            "Experiment output directory. Repeat to evaluate multiple runs. "
            "Defaults to the four APRIL infilling experiments from the plan."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help="Validation dataset path (JSON or JSONL).",
    )
    parser.add_argument(
        "--max-examples",
        type=_positive_int,
        default=10_000,
        help="Maximum number of usable validation examples to preprocess per checkpoint.",
    )
    parser.add_argument(
        "--min-epoch",
        type=_positive_int,
        default=1,
        help="First integer epoch to target.",
    )
    parser.add_argument(
        "--max-epoch",
        type=_positive_int,
        default=10,
        help="Last integer epoch to target.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code through to HF loading.",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load models in bf16 on CUDA when available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to make diffusion eval masking reproducible.",
    )
    parser.add_argument(
        "--analysis-dir",
        default=str(DEFAULT_ANALYSIS_DIR),
        help="Directory where CSV/JSON analysis outputs are written.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing analysis output files.",
    )
    parser.add_argument(
        "--progress-bars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars during evaluation.",
    )
    parser.add_argument(
        "--ar-max-length",
        type=int,
        default=1024,
        help="AR max sequence length for eval dataset construction.",
    )
    parser.add_argument(
        "--ar-batch-size",
        type=int,
        default=8,
        help="Per-device eval batch size for AR checkpoints.",
    )
    parser.add_argument(
        "--diffusion-max-length",
        type=int,
        default=1024,
        help="Diffusion max sequence length for eval dataset construction.",
    )
    parser.add_argument(
        "--diffusion-mask-span-length",
        type=int,
        default=64,
        help="Masked span length for diffusion eval dataset construction.",
    )
    parser.add_argument(
        "--diffusion-min-mask-ratio",
        type=float,
        default=1.0,
        help=(
            "Minimum diffusion masking ratio used during eval-loss computation. "
            "Defaults to 1.0 so diffusion eval is inference-like with the full "
            "hole masked."
        ),
    )
    parser.add_argument(
        "--diffusion-max-mask-ratio",
        type=float,
        default=1.0,
        help=(
            "Maximum diffusion masking ratio used during eval-loss computation. "
            "Defaults to 1.0 so diffusion eval is inference-like with the full "
            "hole masked."
        ),
    )
    parser.add_argument(
        "--diffusion-batch-size",
        type=int,
        default=8,
        help="Per-device eval batch size for diffusion checkpoints.",
    )
    return parser


def _set_eval_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_step(path: Path) -> int:
    name = path.name
    if not name.startswith(CHECKPOINT_PREFIX):
        return -1
    suffix = name[len(CHECKPOINT_PREFIX) :]
    return int(suffix) if suffix.isdigit() else -1


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _iter_checkpoint_dirs(experiment_dir: Path) -> List[Path]:
    checkpoint_dirs = [
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and path.name.startswith(CHECKPOINT_PREFIX)
    ]
    return sorted(checkpoint_dirs, key=_checkpoint_step)


def _weight_filename(checkpoint_dir: Path) -> Optional[str]:
    for filename in WEIGHT_FILENAMES:
        if (checkpoint_dir / filename).exists():
            return filename
    return None


def _detect_model_family(
    experiment_name: str,
    base_model_name_or_path: str,
    checkpoint_dir: Path,
    trust_remote_code: bool,
) -> str:
    lower_name = experiment_name.lower()
    lower_base = base_model_name_or_path.lower()
    if "diffusion" in lower_name or "dream" in lower_base or "llada" in lower_base:
        return "diffusion"
    if "infilling-ar" in lower_name or "qwen" in lower_base:
        return "ar"
    return resolve_diffusion_family(
        str(checkpoint_dir),
        trust_remote_code=trust_remote_code,
        base_model_name=base_model_name_or_path,
    )


def discover_checkpoints(
    experiment_dir: Path,
    *,
    trust_remote_code: bool,
) -> List[CheckpointInfo]:
    experiment_name = experiment_dir.name
    checkpoints: List[CheckpointInfo] = []
    for checkpoint_dir in _iter_checkpoint_dirs(experiment_dir):
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        adapter_config_path = checkpoint_dir / "adapter_config.json"
        if not trainer_state_path.exists() or not adapter_config_path.exists():
            continue

        trainer_state = _load_json(trainer_state_path)
        adapter_config = _load_json(adapter_config_path)
        base_model_name_or_path = str(
            adapter_config.get("base_model_name_or_path") or ""
        ).strip()
        model_family = _detect_model_family(
            experiment_name,
            base_model_name_or_path,
            checkpoint_dir,
            trust_remote_code,
        )
        weight_filename = _weight_filename(checkpoint_dir)
        checkpoints.append(
            CheckpointInfo(
                experiment_dir=experiment_dir,
                experiment_name=experiment_name,
                checkpoint_dir=checkpoint_dir,
                checkpoint_name=checkpoint_dir.name,
                checkpoint_step=_checkpoint_step(checkpoint_dir),
                epoch=float(trainer_state.get("epoch") or 0.0),
                global_step=int(trainer_state.get("global_step") or 0),
                model_family="diffusion" if model_family != "ar" else "ar",
                base_model_name_or_path=base_model_name_or_path,
                has_weights=weight_filename is not None,
                weight_filename=weight_filename,
            )
        )
    return checkpoints


def select_target_checkpoints(
    checkpoints: Sequence[CheckpointInfo],
    *,
    min_epoch: int,
    max_epoch: int,
) -> List[Dict[str, Any]]:
    selections: List[Dict[str, Any]] = []
    for requested_epoch in range(min_epoch, max_epoch + 1):
        matches = [
            checkpoint
            for checkpoint in checkpoints
            if abs(checkpoint.epoch - float(requested_epoch)) <= EPOCH_TOLERANCE
        ]
        matches.sort(key=lambda checkpoint: checkpoint.checkpoint_step)
        chosen = matches[0] if matches else None
        selections.append(
            {
                "requested_epoch": requested_epoch,
                "checkpoint": chosen,
                "status": "selected" if chosen is not None else "missing_checkpoint",
            }
        )
    return selections


def latest_checkpoint(checkpoints: Sequence[CheckpointInfo]) -> Optional[CheckpointInfo]:
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda checkpoint: checkpoint.checkpoint_step)


def extract_history_rows(
    latest_checkpoint_info: CheckpointInfo,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    trainer_state = _load_json(latest_checkpoint_info.checkpoint_dir / "trainer_state.json")
    log_history = trainer_state.get("log_history") or []
    if not isinstance(log_history, list):
        raise ValueError(
            f"log_history must be a list in {latest_checkpoint_info.checkpoint_dir}"
        )

    train_rows: List[Dict[str, Any]] = []
    logged_eval_rows: List[Dict[str, Any]] = []
    for entry in log_history:
        if not isinstance(entry, dict):
            continue
        common = {
            "experiment_name": latest_checkpoint_info.experiment_name,
            "experiment_dir": str(latest_checkpoint_info.experiment_dir),
            "model_family": latest_checkpoint_info.model_family,
            "source": "trainer_state",
            "epoch": float(entry.get("epoch")) if entry.get("epoch") is not None else None,
            "step": int(entry.get("step")) if entry.get("step") is not None else None,
            "checkpoint_path": str(latest_checkpoint_info.checkpoint_dir),
        }
        if entry.get("loss") is not None:
            train_rows.append(
                {
                    **common,
                    "metric_name": "train_loss",
                    "loss_value": float(entry["loss"]),
                }
            )
        if entry.get("eval_loss") is not None:
            logged_eval_rows.append(
                {
                    **common,
                    "metric_name": "logged_eval_loss",
                    "loss_value": float(entry["eval_loss"]),
                    "eval_runtime": float(entry.get("eval_runtime") or 0.0),
                    "eval_samples_per_second": float(
                        entry.get("eval_samples_per_second") or 0.0
                    ),
                    "eval_steps_per_second": float(
                        entry.get("eval_steps_per_second") or 0.0
                    ),
                }
            )
    return train_rows, logged_eval_rows


def _base_training_args(
    output_dir: Path,
    *,
    batch_size: int,
    disable_tqdm: bool,
    bf16: bool,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=batch_size,
        do_train=False,
        do_eval=True,
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=disable_tqdm,
        bf16=bf16,
        fp16=False,
        dataloader_drop_last=False,
    )


def evaluate_ar_checkpoint(
    checkpoint: CheckpointInfo,
    *,
    dataset_path: Path,
    device: torch.device,
    trust_remote_code: bool,
    prefer_bf16: bool,
    batch_size: int,
    max_length: int,
    max_examples: int,
    analysis_dir: Path,
    disable_tqdm: bool,
    eval_dataset: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Any]:
    tokenizer, model = load_ar_model(
        str(checkpoint.checkpoint_dir),
        device=device,
        trust_remote_code=trust_remote_code,
        use_lora_adapter=True,
        bf16=prefer_bf16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if eval_dataset is None:
        eval_dataset = InfillingARDataset(
            str(dataset_path),
            tokenizer=tokenizer,
            max_length=max_length,
            max_examples=max_examples,
        ).to_hf()
    data_collator = InfillingARCollator(
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
    )
    trainer = Trainer(
        model=model,
        args=_base_training_args(
            analysis_dir / "_hf_eval" / checkpoint.experiment_name / checkpoint.checkpoint_name,
            batch_size=batch_size,
            disable_tqdm=disable_tqdm,
            bf16=device.type == "cuda" and prefer_bf16,
        ),
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    metrics = trainer.evaluate()
    metrics["num_eval_examples"] = len(eval_dataset)
    return metrics, eval_dataset


def evaluate_diffusion_checkpoint(
    checkpoint: CheckpointInfo,
    *,
    dataset_path: Path,
    device: torch.device,
    trust_remote_code: bool,
    prefer_bf16: bool,
    batch_size: int,
    max_length: int,
    max_examples: int,
    mask_span_length: int,
    min_mask_ratio: float,
    max_mask_ratio: float,
    seed: int,
    analysis_dir: Path,
    disable_tqdm: bool,
    eval_dataset: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Any]:
    components = load_diffusion_model_bundle(
        str(checkpoint.checkpoint_dir),
        device=device,
        trust_remote_code=trust_remote_code,
        use_lora_adapter=True,
        bf16=prefer_bf16,
    )
    _set_eval_seed(seed)
    if eval_dataset is None:
        eval_dataset = InfillingMDMDataset(
            str(dataset_path),
            tokenizer=components.tokenizer,
            max_length=max_length,
            mask_span_length=mask_span_length,
            max_examples=max_examples,
        ).to_hf()
    training_objective = create_diffusion_training_objective(
        family=components.family,
        mode="infilling",
        tokenizer=components.tokenizer,
        mask_token_id=components.mask_token_id,
        min_mask_ratio=min_mask_ratio,
        max_mask_ratio=max_mask_ratio,
        pad_token_id=components.tokenizer.pad_token_id,
    )
    trainer = MdlmTrainer(
        model=components.model,
        args=_base_training_args(
            analysis_dir / "_hf_eval" / checkpoint.experiment_name / checkpoint.checkpoint_name,
            batch_size=batch_size,
            disable_tqdm=disable_tqdm,
            bf16=device.type == "cuda" and prefer_bf16,
        ),
        eval_dataset=eval_dataset,
        data_collator=training_objective,
        tokenizer=components.tokenizer,
        training_objective=training_objective,
    )
    metrics = trainer.evaluate()
    metrics["num_eval_examples"] = len(eval_dataset)
    return metrics, eval_dataset


def cleanup_after_eval(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def build_checkpoint_eval_row(
    checkpoint: Optional[CheckpointInfo],
    *,
    experiment_name: str,
    experiment_dir: Path,
    model_family: str,
    requested_epoch: int,
    dataset_path: Path,
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    checkpoint_path = str(checkpoint.checkpoint_dir) if checkpoint is not None else None
    epoch = float(checkpoint.epoch) if checkpoint is not None else float(requested_epoch)
    step = checkpoint.global_step if checkpoint is not None else None
    row: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "model_family": model_family,
        "requested_epoch": requested_epoch,
        "epoch": epoch,
        "step": step,
        "global_step": step,
        "metric_name": "checkpoint_val_loss",
        "loss_value": None,
        "status": status,
        "checkpoint_path": checkpoint_path,
        "checkpoint_name": checkpoint.checkpoint_name if checkpoint is not None else None,
        "checkpoint_step": checkpoint.checkpoint_step if checkpoint is not None else None,
        "dataset_path": str(dataset_path),
        "weight_filename": checkpoint.weight_filename if checkpoint is not None else None,
        "error_message": error_message,
    }
    if metrics is not None:
        row["loss_value"] = float(metrics.get("eval_loss")) if metrics.get("eval_loss") is not None else None
        row["eval_runtime"] = float(metrics.get("eval_runtime") or 0.0)
        row["eval_samples_per_second"] = float(
            metrics.get("eval_samples_per_second") or 0.0
        )
        row["eval_steps_per_second"] = float(metrics.get("eval_steps_per_second") or 0.0)
        row["num_eval_examples"] = int(metrics.get("num_eval_examples") or 0)
    return row


def resolve_experiment_dirs(raw_experiment_dirs: Sequence[str]) -> List[Path]:
    if not raw_experiment_dirs:
        return list(DEFAULT_EXPERIMENT_DIRS)
    resolved: List[Path] = []
    for value in raw_experiment_dirs:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        resolved.append(path)
    return resolved


def _read_base_model_name(checkpoint_dir: Path) -> str:
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        return ""
    payload = _load_json(adapter_config_path)
    return str(payload.get("base_model_name_or_path") or "").strip()


def build_experiment_manifest_row(
    experiment_dir: Path,
    checkpoints: Sequence[CheckpointInfo],
    selections: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    latest = latest_checkpoint(checkpoints)
    available_epochs = [
        int(round(checkpoint.epoch))
        for checkpoint in checkpoints
        if abs(checkpoint.epoch - round(checkpoint.epoch)) <= EPOCH_TOLERANCE
    ]
    return {
        "experiment_name": experiment_dir.name,
        "experiment_dir": str(experiment_dir),
        "model_family": latest.model_family if latest is not None else "unknown",
        "base_model_name_or_path": latest.base_model_name_or_path if latest is not None else "",
        "num_checkpoint_dirs": len(checkpoints),
        "available_integer_epochs": sorted(set(available_epochs)),
        "selected_epochs": [
            item["requested_epoch"]
            for item in selections
            if item.get("checkpoint") is not None
        ],
        "missing_epochs": [
            item["requested_epoch"]
            for item in selections
            if item.get("checkpoint") is None
        ],
    }


def ensure_outputs_writable(analysis_dir: Path, *, overwrite: bool) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return
    output_paths = [
        analysis_dir / "checkpoint_eval_losses.csv",
        analysis_dir / "checkpoint_eval_losses.json",
        analysis_dir / "training_loss_history.csv",
        analysis_dir / "training_loss_history.json",
        analysis_dir / "merged_loss_curves.csv",
        analysis_dir / "merged_loss_curves.json",
        analysis_dir / "run_manifest.json",
    ]
    conflicts = [str(path) for path in output_paths if path.exists()]
    if conflicts:
        raise FileExistsError(
            "Refusing to overwrite existing outputs:\n" + "\n".join(conflicts)
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.min_epoch > args.max_epoch:
        raise ValueError("--min-epoch must be <= --max-epoch.")
    if not (0.0 < args.diffusion_min_mask_ratio <= args.diffusion_max_mask_ratio <= 1.0):
        raise ValueError(
            "Diffusion mask ratios must satisfy 0 < min <= max <= 1."
        )

    experiment_dirs = resolve_experiment_dirs(args.experiment_dir)
    dataset_path = resolve_eval_dataset_path(args.dataset_path)
    device = resolve_device(args.device)
    analysis_dir = Path(args.analysis_dir).expanduser()
    if not analysis_dir.is_absolute():
        analysis_dir = REPO_ROOT / analysis_dir
    ensure_outputs_writable(analysis_dir, overwrite=args.overwrite)

    checkpoint_eval_rows: List[Dict[str, Any]] = []
    training_rows: List[Dict[str, Any]] = []
    logged_eval_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    print(f"Analysis dir: {analysis_dir}")
    print(f"Target epoch window: {args.min_epoch}..{args.max_epoch}")
    print(f"Max eval examples per checkpoint: {args.max_examples:,}")
    print(
        "Diffusion eval mask ratio range: "
        f"{args.diffusion_min_mask_ratio:g}..{args.diffusion_max_mask_ratio:g}"
    )

    for experiment_dir in experiment_dirs:
        print(f"\nInspecting {experiment_dir}")
        if not experiment_dir.exists():
            checkpoint_eval_rows.extend(
                [
                    build_checkpoint_eval_row(
                        None,
                        experiment_name=experiment_dir.name,
                        experiment_dir=experiment_dir,
                        model_family="unknown",
                        requested_epoch=requested_epoch,
                        dataset_path=dataset_path,
                        status="missing_experiment_dir",
                        error_message="Experiment directory does not exist.",
                    )
                    for requested_epoch in range(args.min_epoch, args.max_epoch + 1)
                ]
            )
            manifest_rows.append(
                {
                    "experiment_name": experiment_dir.name,
                    "experiment_dir": str(experiment_dir),
                    "model_family": "unknown",
                    "base_model_name_or_path": "",
                    "num_checkpoint_dirs": 0,
                    "available_integer_epochs": [],
                    "selected_epochs": [],
                    "missing_epochs": list(range(args.min_epoch, args.max_epoch + 1)),
                }
            )
            continue

        checkpoints = discover_checkpoints(
            experiment_dir,
            trust_remote_code=args.trust_remote_code,
        )
        if not checkpoints:
            checkpoint_eval_rows.extend(
                [
                    build_checkpoint_eval_row(
                        None,
                        experiment_name=experiment_dir.name,
                        experiment_dir=experiment_dir,
                        model_family="unknown",
                        requested_epoch=requested_epoch,
                        dataset_path=dataset_path,
                        status="no_checkpoints_found",
                        error_message="No checkpoint directories with trainer_state.json were found.",
                    )
                    for requested_epoch in range(args.min_epoch, args.max_epoch + 1)
                ]
            )
            manifest_rows.append(
                {
                    "experiment_name": experiment_dir.name,
                    "experiment_dir": str(experiment_dir),
                    "model_family": "unknown",
                    "base_model_name_or_path": "",
                    "num_checkpoint_dirs": 0,
                    "available_integer_epochs": [],
                    "selected_epochs": [],
                    "missing_epochs": list(range(args.min_epoch, args.max_epoch + 1)),
                }
            )
            continue

        selections = select_target_checkpoints(
            checkpoints,
            min_epoch=args.min_epoch,
            max_epoch=args.max_epoch,
        )
        cached_eval_dataset: Optional[Any] = None
        manifest_rows.append(build_experiment_manifest_row(experiment_dir, checkpoints, selections))

        latest = latest_checkpoint(checkpoints)
        if latest is not None:
            experiment_train_rows, experiment_logged_eval_rows = extract_history_rows(latest)
            training_rows.extend(experiment_train_rows)
            logged_eval_rows.extend(experiment_logged_eval_rows)

        for selection in selections:
            requested_epoch = int(selection["requested_epoch"])
            checkpoint = selection["checkpoint"]
            if checkpoint is None:
                checkpoint_eval_rows.append(
                    build_checkpoint_eval_row(
                        None,
                        experiment_name=experiment_dir.name,
                        experiment_dir=experiment_dir,
                        model_family=latest.model_family if latest is not None else "unknown",
                        requested_epoch=requested_epoch,
                        dataset_path=dataset_path,
                        status="missing_checkpoint",
                        error_message="No saved checkpoint matched this integer epoch.",
                    )
                )
                continue

            if not checkpoint.has_weights:
                checkpoint_eval_rows.append(
                    build_checkpoint_eval_row(
                        checkpoint,
                        experiment_name=experiment_dir.name,
                        experiment_dir=experiment_dir,
                        model_family=checkpoint.model_family,
                        requested_epoch=requested_epoch,
                        dataset_path=dataset_path,
                        status="missing_weights",
                        error_message="Checkpoint metadata exists, but no weight file was found.",
                    )
                )
                continue

            print(
                f"  Evaluating {checkpoint.checkpoint_name} "
                f"(epoch={checkpoint.epoch:g}, family={checkpoint.model_family})"
            )
            try:
                if checkpoint.model_family == "ar":
                    metrics, cached_eval_dataset = evaluate_ar_checkpoint(
                        checkpoint,
                        dataset_path=dataset_path,
                        device=device,
                        trust_remote_code=args.trust_remote_code,
                        prefer_bf16=args.bf16,
                        batch_size=args.ar_batch_size,
                        max_length=args.ar_max_length,
                        max_examples=args.max_examples,
                        analysis_dir=analysis_dir,
                        disable_tqdm=not args.progress_bars,
                        eval_dataset=cached_eval_dataset,
                    )
                else:
                    metrics, cached_eval_dataset = evaluate_diffusion_checkpoint(
                        checkpoint,
                        dataset_path=dataset_path,
                        device=device,
                        trust_remote_code=args.trust_remote_code,
                        prefer_bf16=args.bf16,
                        batch_size=args.diffusion_batch_size,
                        max_length=args.diffusion_max_length,
                        max_examples=args.max_examples,
                        mask_span_length=args.diffusion_mask_span_length,
                        min_mask_ratio=args.diffusion_min_mask_ratio,
                        max_mask_ratio=args.diffusion_max_mask_ratio,
                        seed=args.seed,
                        analysis_dir=analysis_dir,
                        disable_tqdm=not args.progress_bars,
                        eval_dataset=cached_eval_dataset,
                    )
                checkpoint_eval_rows.append(
                    build_checkpoint_eval_row(
                        checkpoint,
                        experiment_name=experiment_dir.name,
                        experiment_dir=experiment_dir,
                        model_family=checkpoint.model_family,
                        requested_epoch=requested_epoch,
                        dataset_path=dataset_path,
                        status="evaluated",
                        metrics=metrics,
                    )
                )
            except Exception as exc:
                checkpoint_eval_rows.append(
                    build_checkpoint_eval_row(
                        checkpoint,
                        experiment_name=experiment_dir.name,
                        experiment_dir=experiment_dir,
                        model_family=checkpoint.model_family,
                        requested_epoch=requested_epoch,
                        dataset_path=dataset_path,
                        status="eval_error",
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                )
                print(traceback.format_exc())
            finally:
                cleanup_after_eval(device)

    merged_rows = training_rows + logged_eval_rows + checkpoint_eval_rows
    payload = {
        "dataset_path": str(dataset_path),
        "device": str(device),
        "seed": int(args.seed),
        "analysis_dir": str(analysis_dir),
        "max_examples": int(args.max_examples),
        "epoch_window": {"min_epoch": int(args.min_epoch), "max_epoch": int(args.max_epoch)},
        "experiments": manifest_rows,
        "checkpoint_eval_rows": checkpoint_eval_rows,
        "training_loss_rows": training_rows,
        "logged_eval_loss_rows": logged_eval_rows,
        "merged_rows": merged_rows,
    }

    write_csv(str(analysis_dir / "checkpoint_eval_losses.csv"), checkpoint_eval_rows)
    write_json(str(analysis_dir / "checkpoint_eval_losses.json"), {"rows": checkpoint_eval_rows})
    write_csv(str(analysis_dir / "training_loss_history.csv"), training_rows + logged_eval_rows)
    write_json(
        str(analysis_dir / "training_loss_history.json"),
        {"train_loss_rows": training_rows, "logged_eval_loss_rows": logged_eval_rows},
    )
    write_csv(str(analysis_dir / "merged_loss_curves.csv"), merged_rows)
    write_json(str(analysis_dir / "merged_loss_curves.json"), {"rows": merged_rows})
    write_json(str(analysis_dir / "run_manifest.json"), payload)

    print("\nWrote:")
    print(f"  {analysis_dir / 'checkpoint_eval_losses.csv'}")
    print(f"  {analysis_dir / 'checkpoint_eval_losses.json'}")
    print(f"  {analysis_dir / 'training_loss_history.csv'}")
    print(f"  {analysis_dir / 'training_loss_history.json'}")
    print(f"  {analysis_dir / 'merged_loss_curves.csv'}")
    print(f"  {analysis_dir / 'merged_loss_curves.json'}")
    print(f"  {analysis_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()

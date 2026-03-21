"""
Train an autoregressive infilling model on Lean proof data.

Usage:
    python scripts/train_infilling_autoregressive.py --help
"""

import argparse
import os
from pathlib import Path
from typing import Optional

from peft import LoraConfig, TaskType

from lean_dojo_v2.trainer.infilling_autoregressive_trainer import (
    InfillingAutoregressiveTrainer,
)


def _optional_positive_int(value: str) -> Optional[int]:
    normalized = value.strip().lower()
    if normalized in {"none", "all", ""}:
        return None

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive or 'none'.")
    return parsed


def _safe_dir_component(value: str) -> str:
    normalized = value.strip().replace("/", "-").replace("\\", "-").replace(" ", "-")
    return normalized or "run"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an autoregressive infilling model on Lean proof holes.",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    parser.add_argument(
        "--train-path",
        default="datasets/april/leandojo_infilling/thme.train.jsonl",
        help="Path to training dataset (JSON or JSONL).",
    )
    parser.add_argument(
        "--val-path",
        default="datasets/april/leandojo_infilling/thme.val.jsonl",
        help="Path to validation dataset (JSON or JSONL). If missing, validation is disabled.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save checkpoints and final model (default: outputs/<wandb-run-name>).",
    )
    parser.add_argument("--epochs", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", default="epoch")
    parser.add_argument(
        "--qual-num-samples-per-split",
        type=int,
        default=64,
        help="Number of train/val examples to include in qualitative sample tables.",
    )
    parser.add_argument(
        "--subset-eval-num-samples",
        type=int,
        default=512,
        help=(
            "Number of fixed validation examples to score each eval for the cheap "
            "checkpoint-selection exact-match metric."
        ),
    )
    parser.add_argument(
        "--full-exact-match-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the expensive full validation exact-match sweep during eval. "
            "Disabled by default to avoid long rank-0-only work under DDP."
        ),
    )
    parser.add_argument(
        "--max-train-examples",
        type=_optional_positive_int,
        default=None,
        help="Cap train examples; use 'none' to disable cap.",
    )
    parser.add_argument(
        "--max-val-examples",
        type=_optional_positive_int,
        default=None,
        help="Cap val examples; use 'none' to disable cap.",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.getenv("WANDB_PROJECT", "infilling-ar"),
        help="Weights & Biases project name. Use empty string to disable.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default="infilling-ar-qwen2.5-7b-april",
        help="Weights & Biases run name.",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override bf16 usage (default: auto-detect from CUDA availability).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow transformers trust_remote_code when loading model/tokenizer.",
    )
    parser.add_argument(
        "--use-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LoRA fine-tuning.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "v_proj", "k_proj", "o_proj"],
        help="Space-separated LoRA target module names.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = args.output_dir or str(Path("outputs") / _safe_dir_component(args.wandb_run_name))

    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    resolved_val_path = str(val_path) if val_path.exists() else None

    lora_config = None
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    trainer = InfillingAutoregressiveTrainer(
        model_name=args.model_name,
        train_path=str(train_path),
        val_path=resolved_val_path,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        lora_config=lora_config,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        qual_num_samples_per_split=args.qual_num_samples_per_split,
        subset_eval_num_samples=args.subset_eval_num_samples,
        full_exact_match_eval=args.full_exact_match_eval,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
        trust_remote_code=args.trust_remote_code,
    )

    # Keep tqdm progress bars and suppress raw metric dict logs.
    trainer.training_args.disable_tqdm = False
    trainer.train()

    print(f"\nModel saved to {output_dir}")
    print("You can now use this model for autoregressive infilling inference.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the diffusion LLM (LLaDA) with SFT on APRIL LeanDojo-format data.

This script bypasses the full agent/database pipeline and directly trains
on pre-converted APRIL JSON files (the ones under datasets/april/leandojo/).

Usage:
    python scripts/train_diffusion_sft_april.py \
        --data-dir datasets/april/leandojo/test \
        --output-dir /resnick/scratch/tram/cs165-project/outputs-diffusion-sft \
        --epochs 3 --batch-size 1 --lr 2e-5 --use-lora
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments

from lean_dojo_v2.diffusion import resolve_mask_token_id
from lean_dojo_v2.trainer.diffusion_sft_trainer import (
    DiffusionDataCollator,
    DiffusionSFTDataset,
    MdlmTrainer,
)


class LossPrinterCallback(TrainerCallback):
    """Prints loss to stdout so it appears in SLURM .out logs."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        epoch = logs.get("epoch")
        grad_norm = logs.get("grad_norm")
        parts = [f"step={step}"]
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        if grad_norm is not None:
            parts.append(f"grad_norm={grad_norm:.4f}")
        if epoch is not None:
            parts.append(f"epoch={epoch:.2f}")
        print(f"  [TRAIN] {', '.join(parts)}", flush=True)


def load_merged_data(data_dir: Path, data_files: list[str] | None) -> list[dict[str, Any]]:
    """Load and merge all JSON files from *data_dir* into a single list."""
    if data_files:
        paths = [data_dir / f for f in data_files]
    else:
        paths = sorted(data_dir.glob("*.json"))

    if not paths:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    merged: list[dict[str, Any]] = []
    for p in paths:
        items = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(items, list):
            merged.extend(items)
            print(f"  Loaded {p.name}: {len(items)} theorems")
        else:
            print(f"  Skipping {p.name}: not a JSON array", file=sys.stderr)

    return merged


def write_merged_json(data: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/april/leandojo/test"),
        help="Directory containing LeanDojo-format JSON files",
    )
    parser.add_argument(
        "--data-files",
        nargs="*",
        default=None,
        help="Specific JSON files to use (relative to --data-dir). If omitted, uses all *.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/resnick/scratch/tram/cs165-project/outputs-diffusion-sft",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--min-mask-ratio", type=float, default=0.01)
    parser.add_argument("--max-mask-ratio", type=float, default=1.0)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print("Diffusion SFT Training on APRIL Data")
    print("=" * 60)
    print(f"Model:       {args.model_name}")
    print(f"Data dir:    {args.data_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Grad accum:  {args.gradient_accumulation_steps}")
    print(f"LR:          {args.lr}")
    print(f"Use LoRA:    {args.use_lora}")
    print(f"Max length:  {args.max_length}")
    print(f"CUDA avail:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
        print(f"GPU memory:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    # -- Load and merge data --
    print("\n[1/5] Loading data...")
    merged = load_merged_data(args.data_dir, args.data_files)
    total_tactics = sum(len(item.get("traced_tactics", [])) for item in merged)
    print(f"  Total: {len(merged)} theorems, {total_tactics} tactic examples")

    if total_tactics == 0:
        print("ERROR: No tactic examples found. Exiting.", file=sys.stderr)
        sys.exit(1)

    merged_path = Path(args.output_dir) / "merged_train_data.json"
    write_merged_json(merged, merged_path)
    print(f"  Merged data written to {merged_path}")

    # -- Load tokenizer and model --
    print("\n[2/5] Loading tokenizer and model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    mask_token_id = resolve_mask_token_id(tokenizer)

    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    if args.use_lora:
        print("\n[2.5/5] Applying LoRA...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # -- Build dataset --
    print("\n[3/5] Building training dataset...")
    train_dataset = DiffusionSFTDataset(
        data_path=str(merged_path),
        tokenizer=tokenizer,
        max_length=args.max_length,
        gen_length=args.gen_length,
    ).to_hf()
    print(f"  Dataset size: {len(train_dataset)} examples")

    if len(train_dataset) == 0:
        print("ERROR: Dataset is empty after processing. Exiting.", file=sys.stderr)
        sys.exit(1)

    # -- Set up training --
    print("\n[4/5] Configuring training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        bf16=torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=4,
        save_total_limit=3,
    )

    data_collator = DiffusionDataCollator(
        tokenizer=tokenizer,
        mask_token_id=mask_token_id,
        min_mask_ratio=args.min_mask_ratio,
        max_mask_ratio=args.max_mask_ratio,
    )

    trainer = MdlmTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[LossPrinterCallback()],
    )

    # -- Train --
    print("\n[5/5] Starting training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # -- Save final checkpoint --
    print("\nSaving final model...")
    if args.use_lora:
        model.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # -- Print loss summary --
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    log_history = trainer.state.log_history
    loss_entries = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    if loss_entries:
        first_loss = loss_entries[0][1]
        last_loss = loss_entries[-1][1]
        min_loss = min(l for _, l in loss_entries)
        print(f"  First loss (step {loss_entries[0][0]}):  {first_loss:.4f}")
        print(f"  Last loss  (step {loss_entries[-1][0]}):  {last_loss:.4f}")
        print(f"  Best loss:                    {min_loss:.4f}")
        print(f"  Improvement:                  {first_loss - last_loss:.4f} ({(first_loss - last_loss) / first_loss * 100:.1f}%)")
        print(f"  Total logged steps:           {len(loss_entries)}")
    else:
        print("  No loss entries found in training log.")

    print(f"\nModel saved to {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()

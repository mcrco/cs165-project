#!/usr/bin/env python3
"""Sweep diffusion generation steps against validation exact-match accuracy."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

DEFAULT_DIFFUSION_MODEL_NAME = "Dream-org/Dream-v0-Instruct-7B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate how diffusion denoising steps affect validation exact-match "
            "accuracy and throughput."
        )
    )
    parser.add_argument(
        "--dataset-path",
        default="datasets/april/leandojo_infilling/thme_1m_100k.val.jsonl",
        help=(
            "Validation dataset path (JSON or JSONL). If missing, the script tries "
            "common APRIL validation locations."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10_000,
        help="Maximum number of validation examples to evaluate.",
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
        help="Load the diffusion model in bf16 on CUDA when available.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Number of untimed warmup batches to run before each step sweep point.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_DIFFUSION_MODEL_NAME,
        help="Diffusion model name, checkpoint path, or adapter path.",
    )
    parser.add_argument(
        "--use-lora-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interpret --model as a PEFT adapter path.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum context length for diffusion prompt construction.",
    )
    parser.add_argument(
        "--mask-span-length",
        type=int,
        default=64,
        help="Masked span length used for diffusion infilling.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for diffusion generation.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512],
        help="Space-separated denoising step counts to evaluate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Diffusion sampling temperature.",
    )
    parser.add_argument(
        "--remasking",
        default=None,
        help="Diffusion remasking strategy. Defaults to the model family preset.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to save the full sweep payload as JSON.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional path to save the step sweep table as CSV.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from infilling_eval_utils import (
        benchmark_diffusion_examples,
        default_diffusion_remasking,
        load_diffusion_model_bundle,
        load_raw_infilling_examples,
        print_summary_block,
        resolve_device,
        resolve_eval_dataset_path,
        write_csv,
        write_json,
    )
    dataset_path = resolve_eval_dataset_path(args.dataset_path)
    device = resolve_device(args.device)
    examples = load_raw_infilling_examples(
        dataset_path,
        max_examples=args.max_examples,
    )
    if not examples:
        raise ValueError(f"No usable infilling examples found in {dataset_path}.")

    remasking = (
        default_diffusion_remasking(args.model)
        if args.remasking is None
        else args.remasking
    )
    step_values = sorted({max(1, int(step)) for step in args.steps})

    print(f"Dataset: {dataset_path}")
    print(f"Loaded {len(examples)} validation examples")
    print(f"Device: {device}")
    print(f"Step sweep: {step_values}")

    print("\nLoading diffusion model...")
    components = load_diffusion_model_bundle(
        args.model,
        device=device,
        trust_remote_code=args.trust_remote_code,
        use_lora_adapter=args.use_lora_adapter,
        bf16=args.bf16,
    )

    rows: List[Dict[str, Any]] = []
    for step_count in step_values:
        print(f"\nEvaluating {step_count} diffusion steps...")
        metrics = benchmark_diffusion_examples(
            examples,
            model=components.model,
            tokenizer=components.tokenizer,
            mask_token_id=components.mask_token_id,
            device=device,
            max_length=args.max_length,
            mask_span_length=args.mask_span_length,
            batch_size=args.batch_size,
            steps=step_count,
            temperature=args.temperature,
            remasking=remasking,
            warmup_batches=args.warmup_batches,
        )
        row = {
            "steps": step_count,
            "temperature": args.temperature,
            "remasking": remasking,
            **metrics,
        }
        rows.append(row)
        print_summary_block(f"steps={step_count}", metrics)

    payload = {
        "dataset_path": str(dataset_path),
        "device": str(device),
        "num_examples": len(examples),
        "model_name_or_path": args.model,
        "use_lora_adapter": args.use_lora_adapter,
        "results": rows,
    }
    write_json(args.json_output, payload)
    write_csv(args.csv_output, rows)

    if args.json_output:
        print(f"\nWrote JSON results to {args.json_output}")
    if args.csv_output:
        print(f"Wrote CSV results to {args.csv_output}")


if __name__ == "__main__":
    main()

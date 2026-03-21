#!/usr/bin/env python3
"""Benchmark diffusion vs autoregressive Lean infilling on APRIL validation data."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

DEFAULT_DIFFUSION_MODEL_NAME = "Dream-org/Dream-v0-Instruct-7B"
DEFAULT_AR_MODEL_NAME = "Qwen/Qwen2.5-7B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare inference speed and exact-match accuracy for a diffusion infilling "
            "model against an autoregressive infilling model."
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
        help="Maximum number of validation examples to benchmark.",
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
        "--warmup-batches",
        type=int,
        default=1,
        help="Number of untimed warmup batches to run before each benchmark.",
    )

    parser.add_argument(
        "--diffusion-model",
        default=DEFAULT_DIFFUSION_MODEL_NAME,
        help="Diffusion model name, checkpoint path, or adapter path.",
    )
    parser.add_argument(
        "--diffusion-use-lora-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interpret --diffusion-model as a PEFT adapter path.",
    )
    parser.add_argument(
        "--diffusion-max-length",
        type=int,
        default=1024,
        help="Maximum context length for diffusion prompt construction.",
    )
    parser.add_argument(
        "--diffusion-mask-span-length",
        type=int,
        default=64,
        help="Masked span length used for diffusion infilling.",
    )
    parser.add_argument(
        "--diffusion-batch-size",
        type=int,
        default=4,
        help="Batch size for diffusion generation.",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=None,
        help="Diffusion denoising steps. Defaults to the model family preset.",
    )
    parser.add_argument(
        "--diffusion-temperature",
        type=float,
        default=0.0,
        help="Diffusion sampling temperature.",
    )
    parser.add_argument(
        "--diffusion-remasking",
        default=None,
        help="Diffusion remasking strategy. Defaults to the model family preset.",
    )

    parser.add_argument(
        "--ar-model",
        default=DEFAULT_AR_MODEL_NAME,
        help="Autoregressive model name, checkpoint path, or adapter path.",
    )
    parser.add_argument(
        "--ar-use-lora-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interpret --ar-model as a PEFT adapter path.",
    )
    parser.add_argument(
        "--ar-max-length",
        type=int,
        default=1024,
        help="Maximum prompt length for AR generation.",
    )
    parser.add_argument(
        "--ar-max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens for AR generation.",
    )
    parser.add_argument(
        "--ar-batch-size",
        type=int,
        default=8,
        help="Batch size for autoregressive generation.",
    )
    parser.add_argument(
        "--ar-temperature",
        type=float,
        default=0.0,
        help="AR sampling temperature. Use 0.0 for greedy decoding.",
    )
    parser.add_argument(
        "--ar-top-p",
        type=float,
        default=0.9,
        help="AR top-p when sampling is enabled.",
    )

    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to save the full benchmark payload as JSON.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional path to save the two-row comparison table as CSV.",
    )
    return parser


def main() -> None:
    import gc

    args = build_parser().parse_args()
    from infilling_eval_utils import (
        benchmark_ar_examples,
        benchmark_diffusion_examples,
        default_diffusion_remasking,
        default_diffusion_steps,
        load_ar_model,
        load_diffusion_model_bundle,
        load_raw_infilling_examples,
        print_summary_block,
        resolve_device,
        resolve_eval_dataset_path,
        summary_metrics_only,
        write_csv,
        write_json,
    )
    import torch

    dataset_path = resolve_eval_dataset_path(args.dataset_path)
    device = resolve_device(args.device)

    examples = load_raw_infilling_examples(
        dataset_path,
        max_examples=args.max_examples,
    )
    if not examples:
        raise ValueError(f"No usable infilling examples found in {dataset_path}.")

    diffusion_steps = (
        default_diffusion_steps(args.diffusion_model)
        if args.diffusion_steps is None
        else max(1, int(args.diffusion_steps))
    )
    diffusion_remasking = (
        default_diffusion_remasking(args.diffusion_model)
        if args.diffusion_remasking is None
        else args.diffusion_remasking
    )

    print(f"Dataset: {dataset_path}")
    print(f"Loaded {len(examples)} validation examples")
    print(f"Device: {device}")

    print("\nLoading diffusion model...")
    diffusion_components = load_diffusion_model_bundle(
        args.diffusion_model,
        device=device,
        trust_remote_code=args.trust_remote_code,
        use_lora_adapter=args.diffusion_use_lora_adapter,
        bf16=args.bf16,
    )
    diffusion_metrics = benchmark_diffusion_examples(
        examples,
        model=diffusion_components.model,
        tokenizer=diffusion_components.tokenizer,
        mask_token_id=diffusion_components.mask_token_id,
        device=device,
        max_length=args.diffusion_max_length,
        mask_span_length=args.diffusion_mask_span_length,
        batch_size=args.diffusion_batch_size,
        steps=diffusion_steps,
        temperature=args.diffusion_temperature,
        remasking=diffusion_remasking,
        warmup_batches=args.warmup_batches,
    )
    del diffusion_components
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nLoading autoregressive model...")
    ar_tokenizer, ar_model = load_ar_model(
        args.ar_model,
        device=device,
        trust_remote_code=args.trust_remote_code,
        use_lora_adapter=args.ar_use_lora_adapter,
        bf16=args.bf16,
    )
    ar_metrics = benchmark_ar_examples(
        examples,
        model=ar_model,
        tokenizer=ar_tokenizer,
        device=device,
        max_length=args.ar_max_length,
        max_new_tokens=args.ar_max_new_tokens,
        batch_size=args.ar_batch_size,
        temperature=args.ar_temperature,
        top_p=args.ar_top_p,
        warmup_batches=args.warmup_batches,
    )

    print("\nResults")
    print("-------")
    print_summary_block("Diffusion", diffusion_metrics)
    print_summary_block("Autoregressive", ar_metrics)

    comparison_rows: List[Dict[str, Any]] = [
        {
            "model_family": "diffusion",
            "model_name_or_path": args.diffusion_model,
            "use_lora_adapter": args.diffusion_use_lora_adapter,
            "steps": diffusion_steps,
            "temperature": args.diffusion_temperature,
            "remasking": diffusion_remasking,
            **diffusion_metrics,
        },
        {
            "model_family": "autoregressive",
            "model_name_or_path": args.ar_model,
            "use_lora_adapter": args.ar_use_lora_adapter,
            "max_new_tokens": args.ar_max_new_tokens,
            "temperature": args.ar_temperature,
            "top_p": args.ar_top_p,
            **ar_metrics,
        },
    ]
    comparison_csv_rows = [summary_metrics_only(row) for row in comparison_rows]
    payload = {
        "dataset_path": str(dataset_path),
        "device": str(device),
        "num_examples": len(examples),
        "comparison": comparison_rows,
    }
    write_json(args.json_output, payload)
    write_csv(args.csv_output, comparison_csv_rows)

    if args.json_output:
        print(f"\nWrote JSON results to {args.json_output}")
    if args.csv_output:
        print(f"Wrote CSV results to {args.csv_output}")


if __name__ == "__main__":
    main()

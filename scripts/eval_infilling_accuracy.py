#!/usr/bin/env python3
"""Evaluate accuracy for the lowest-eval-loss AR and diffusion checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_LOSS_JSON = (
    REPO_ROOT
    / "outputs"
    / "analysis"
    / "checkpoint_loss_curves"
    / "checkpoint_eval_losses.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read checkpoint eval losses, pick the best AR and diffusion checkpoints, "
            "then measure exact-match accuracy for those checkpoints."
        )
    )
    parser.add_argument(
        "--checkpoint-loss-json",
        default=str(DEFAULT_CHECKPOINT_LOSS_JSON),
        help=(
            "JSON output from scripts/eval_checkpoint_losses.py. The script selects "
            "the lowest eval-loss checkpoint per model family from this file."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        default="datasets/april/leandojo_infilling/thme.test.jsonl",
        help=(
            "Validation dataset path (JSON or JSONL). If missing, the script tries "
            "common APRIL validation locations."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5_000,
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
        help="Load models in bf16 on CUDA when available.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Number of untimed warmup batches to run before each evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for reproducible evaluation sampling behavior.",
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
        default=8,
        help="Batch size for diffusion generation.",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64],
        help="Space-separated denoising step counts to evaluate for the best diffusion checkpoint.",
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
        help="Optional path to save the full accuracy payload as JSON.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional path to save the accuracy result rows as CSV.",
    )
    parser.add_argument(
        "--generations-jsonl-output",
        default=None,
        help=(
            "Optional path to save per-example prompt and generation records as JSONL. "
            "If omitted, the script derives a sibling file from --json-output or "
            "--csv-output when either is provided."
        ),
    )
    return parser


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _resolve_optional_output_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    return _resolve_path(path_str)


def _default_generations_jsonl_output_path(
    explicit_path: str | None,
    json_output: str | None,
    csv_output: str | None,
) -> Path | None:
    explicit = _resolve_optional_output_path(explicit_path)
    if explicit is not None:
        return explicit

    base_output = _resolve_optional_output_path(json_output)
    if base_output is None:
        base_output = _resolve_optional_output_path(csv_output)
    if base_output is None:
        return None
    return base_output.with_name(f"{base_output.stem}_generations.jsonl")


def _normalize_model_family(raw_value: Any) -> str:
    value = str(raw_value or "").strip().lower()
    if value in {"ar", "autoregressive"}:
        return "ar"
    if value == "diffusion":
        return "diffusion"
    return value


def _checkpoint_uses_lora_adapter(checkpoint_path: Path) -> bool:
    return (checkpoint_path / "adapter_config.json").exists()


def _resolve_lora_base_model_name_or_path(checkpoint_path: Path) -> str:
    adapter_config_path = checkpoint_path / "adapter_config.json"
    adapter_config = _load_json(adapter_config_path)
    base_model_name_or_path = str(
        adapter_config.get("base_model_name_or_path") or ""
    ).strip()
    if not base_model_name_or_path:
        raise ValueError(
            f"LoRA adapter config at {adapter_config_path} is missing "
            "'base_model_name_or_path'."
        )
    return base_model_name_or_path


def _set_eval_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _valid_checkpoint_row(row: Mapping[str, Any], *, family: str) -> bool:
    if _normalize_model_family(row.get("model_family")) != family:
        return False
    if row.get("status") != "evaluated":
        return False
    if row.get("metric_name") != "checkpoint_val_loss":
        return False
    if row.get("checkpoint_path") in {None, ""}:
        return False
    try:
        float(row.get("loss_value"))
    except (TypeError, ValueError):
        return False
    return True


def select_best_checkpoint(
    loss_rows: List[Mapping[str, Any]],
    *,
    family: str,
) -> Dict[str, Any]:
    candidates = [dict(row) for row in loss_rows if _valid_checkpoint_row(row, family=family)]
    if not candidates:
        raise ValueError(
            f"No evaluated {family} checkpoints with a valid loss were found in the loss JSON."
        )
    return min(
        candidates,
        key=lambda row: (
            float(row["loss_value"]),
            int(row.get("global_step") or row.get("step") or row.get("checkpoint_step") or 0),
        ),
    )


def cleanup_model_memory(device) -> None:
    gc.collect()
    if device.type == "cuda":
        import torch

        torch.cuda.empty_cache()


def main() -> None:
    args = build_parser().parse_args()

    from infilling_eval_utils import (
        benchmark_ar_examples,
        benchmark_diffusion_examples,
        default_diffusion_remasking,
        load_ar_model,
        load_diffusion_model_bundle,
        load_raw_infilling_examples,
        print_summary_block,
        resolve_device,
        resolve_eval_dataset_path,
        summary_metrics_only,
        write_csv,
        write_json,
        write_jsonl,
    )

    checkpoint_loss_json = _resolve_path(args.checkpoint_loss_json)
    if not checkpoint_loss_json.exists():
        raise FileNotFoundError(
            f"Checkpoint loss JSON not found: {checkpoint_loss_json}. "
            "Run scripts/eval_checkpoint_losses.py first or pass --checkpoint-loss-json."
        )

    loss_payload = _load_json(checkpoint_loss_json)
    loss_rows_raw = loss_payload.get("rows")
    if not isinstance(loss_rows_raw, list):
        raise ValueError(
            f"Expected {checkpoint_loss_json} to contain a top-level 'rows' list."
        )

    _set_eval_seed(int(args.seed))

    best_diffusion = select_best_checkpoint(loss_rows_raw, family="diffusion")
    best_ar = select_best_checkpoint(loss_rows_raw, family="ar")

    dataset_path = resolve_eval_dataset_path(args.dataset_path)
    device = resolve_device(args.device)
    examples = load_raw_infilling_examples(
        dataset_path,
        max_examples=args.max_examples,
    )
    if not examples:
        raise ValueError(f"No usable infilling examples found in {dataset_path}.")

    diffusion_checkpoint_path = Path(str(best_diffusion["checkpoint_path"]))
    ar_checkpoint_path = Path(str(best_ar["checkpoint_path"]))
    diffusion_use_lora_adapter = _checkpoint_uses_lora_adapter(diffusion_checkpoint_path)
    ar_use_lora_adapter = _checkpoint_uses_lora_adapter(ar_checkpoint_path)
    ar_baseline_model_name_or_path = (
        _resolve_lora_base_model_name_or_path(ar_checkpoint_path)
        if ar_use_lora_adapter
        else str(ar_checkpoint_path)
    )
    diffusion_remasking = (
        default_diffusion_remasking(str(diffusion_checkpoint_path))
        if args.diffusion_remasking is None
        else args.diffusion_remasking
    )
    diffusion_step_values = sorted(
        {max(1, int(step)) for step in args.diffusion_steps}
    )
    generations_jsonl_output = _default_generations_jsonl_output_path(
        args.generations_jsonl_output,
        args.json_output,
        args.csv_output,
    )

    print(f"Checkpoint loss JSON: {checkpoint_loss_json}")
    print(f"Dataset: {dataset_path}")
    print(f"Loaded {len(examples)} validation examples")
    print(f"Device: {device}")
    print(f"Seed: {int(args.seed)}")
    print(
        "Best diffusion checkpoint: "
        f"{diffusion_checkpoint_path} (eval_loss={float(best_diffusion['loss_value']):.6f})"
    )
    print(
        "Best AR checkpoint: "
        f"{ar_checkpoint_path} (eval_loss={float(best_ar['loss_value']):.6f})"
    )
    if ar_use_lora_adapter:
        print(
            "AR baseline model: "
            f"{ar_baseline_model_name_or_path} "
            "(evaluated with no LoRA adapter attached)"
        )

    rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    generation_rows: List[Dict[str, Any]] = []

    print("\nLoading best diffusion checkpoint...")
    diffusion_components = load_diffusion_model_bundle(
        str(diffusion_checkpoint_path),
        device=device,
        trust_remote_code=args.trust_remote_code,
        use_lora_adapter=diffusion_use_lora_adapter,
        bf16=args.bf16,
    )
    for step_count in diffusion_step_values:
        print(f"\nEvaluating best diffusion checkpoint at {step_count} steps...")
        metrics, generation_records = benchmark_diffusion_examples(
            examples,
            model=diffusion_components.model,
            tokenizer=diffusion_components.tokenizer,
            mask_token_id=diffusion_components.mask_token_id,
            device=device,
            max_length=args.diffusion_max_length,
            mask_span_length=args.diffusion_mask_span_length,
            batch_size=args.diffusion_batch_size,
            steps=step_count,
            temperature=args.diffusion_temperature,
            remasking=diffusion_remasking,
            warmup_batches=args.warmup_batches,
            progress_desc=f"diffusion {step_count} steps",
            return_generation_records=True,
        )
        row = {
            "model_family": "diffusion",
            "model_name_or_path": str(diffusion_checkpoint_path),
            "use_lora_adapter": diffusion_use_lora_adapter,
            "source_eval_loss": float(best_diffusion["loss_value"]),
            "source_experiment_name": best_diffusion.get("experiment_name"),
            "source_checkpoint_name": best_diffusion.get("checkpoint_name"),
            "source_global_step": best_diffusion.get("global_step"),
            "steps": step_count,
            "temperature": args.diffusion_temperature,
            "remasking": diffusion_remasking,
            **metrics,
        }
        rows.append(row)
        csv_rows.append({**summary_metrics_only(row)})
        generation_rows.extend(
            {
                "model_family": "diffusion",
                "run_name": f"diffusion steps={step_count}",
                "model_name_or_path": str(diffusion_checkpoint_path),
                "use_lora_adapter": diffusion_use_lora_adapter,
                "eval_variant": "best_checkpoint",
                "source_checkpoint_path": str(diffusion_checkpoint_path),
                "source_eval_loss": float(best_diffusion["loss_value"]),
                "source_experiment_name": best_diffusion.get("experiment_name"),
                "source_checkpoint_name": best_diffusion.get("checkpoint_name"),
                "source_global_step": best_diffusion.get("global_step"),
                "steps": step_count,
                "temperature": args.diffusion_temperature,
                "remasking": diffusion_remasking,
                **record,
            }
            for record in generation_records
        )
        print_summary_block(f"diffusion steps={step_count}", metrics)

    del diffusion_components
    cleanup_model_memory(device)

    ar_runs = [
        {
            "name": "autoregressive",
            "progress_desc": "autoregressive",
            "eval_variant": "best_checkpoint",
            "model_name_or_path": str(ar_checkpoint_path),
            "use_lora_adapter": ar_use_lora_adapter,
        }
    ]
    if ar_use_lora_adapter:
        ar_runs.append(
            {
                "name": "autoregressive baseline (no LoRA)",
                "progress_desc": "autoregressive baseline",
                "eval_variant": "base_model_no_lora",
                "model_name_or_path": ar_baseline_model_name_or_path,
                "use_lora_adapter": False,
            }
        )

    for ar_run in ar_runs:
        print(f"\nLoading {ar_run['name']}...")
        ar_tokenizer, ar_model = load_ar_model(
            ar_run["model_name_or_path"],
            device=device,
            trust_remote_code=args.trust_remote_code,
            use_lora_adapter=bool(ar_run["use_lora_adapter"]),
            bf16=args.bf16,
        )
        ar_metrics, ar_generation_records = benchmark_ar_examples(
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
            progress_desc=str(ar_run["progress_desc"]),
            return_generation_records=True,
        )
        ar_row = {
            "model_family": "ar",
            "model_name_or_path": str(ar_run["model_name_or_path"]),
            "use_lora_adapter": bool(ar_run["use_lora_adapter"]),
            "eval_variant": ar_run["eval_variant"],
            "source_checkpoint_path": str(ar_checkpoint_path),
            "source_eval_loss": float(best_ar["loss_value"]),
            "source_experiment_name": best_ar.get("experiment_name"),
            "source_checkpoint_name": best_ar.get("checkpoint_name"),
            "source_global_step": best_ar.get("global_step"),
            "max_new_tokens": args.ar_max_new_tokens,
            "temperature": args.ar_temperature,
            "top_p": args.ar_top_p,
            **ar_metrics,
        }
        rows.append(ar_row)
        csv_rows.append({**summary_metrics_only(ar_row)})
        generation_rows.extend(
            {
                "model_family": "ar",
                "run_name": str(ar_run["name"]),
                "model_name_or_path": str(ar_run["model_name_or_path"]),
                "use_lora_adapter": bool(ar_run["use_lora_adapter"]),
                "eval_variant": ar_run["eval_variant"],
                "source_checkpoint_path": str(ar_checkpoint_path),
                "source_eval_loss": float(best_ar["loss_value"]),
                "source_experiment_name": best_ar.get("experiment_name"),
                "source_checkpoint_name": best_ar.get("checkpoint_name"),
                "source_global_step": best_ar.get("global_step"),
                "max_new_tokens": args.ar_max_new_tokens,
                "temperature": args.ar_temperature,
                "top_p": args.ar_top_p,
                **record,
            }
            for record in ar_generation_records
        )
        print_summary_block(str(ar_run["name"]), ar_metrics)
        del ar_model
        del ar_tokenizer
        cleanup_model_memory(device)

    payload = {
        "checkpoint_loss_json": str(checkpoint_loss_json),
        "dataset_path": str(dataset_path),
        "device": str(device),
        "seed": int(args.seed),
        "num_examples": len(examples),
        "generations_jsonl_output": (
            str(generations_jsonl_output)
            if generations_jsonl_output is not None
            else None
        ),
        "selected_checkpoints": {
            "diffusion": {
                **best_diffusion,
                "checkpoint_path": str(diffusion_checkpoint_path),
                "use_lora_adapter": diffusion_use_lora_adapter,
            },
            "ar": {
                **best_ar,
                "checkpoint_path": str(ar_checkpoint_path),
                "use_lora_adapter": ar_use_lora_adapter,
                "baseline_model_name_or_path": ar_baseline_model_name_or_path,
            },
        },
        "results": rows,
    }
    write_json(args.json_output, payload)
    write_csv(args.csv_output, csv_rows)
    write_jsonl(
        str(generations_jsonl_output) if generations_jsonl_output is not None else None,
        generation_rows,
    )

    if args.json_output:
        print(f"\nWrote JSON results to {args.json_output}")
    if args.csv_output:
        print(f"Wrote CSV results to {args.csv_output}")
    if generations_jsonl_output is not None:
        print(f"Wrote generation records to {generations_jsonl_output}")


if __name__ == "__main__":
    main()

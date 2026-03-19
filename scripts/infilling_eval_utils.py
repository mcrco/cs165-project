from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lean_dojo_v2.diffusion import (
    DEFAULT_DIFFUSION_MODEL_NAME,
    decode_until_stop,
    denoise_masked_sequence,
    get_diffusion_sampling_config,
    load_diffusion_components,
    sanitize_rope_scaling,
)
from lean_dojo_v2.trainer.infilling_autoregressive_trainer import (
    _build_ar_infilling_prompt_prefix,
)
from lean_dojo_v2.trainer.infilling_diffusion_trainer import (
    _build_infilling_prompt_prefix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AR_MODEL_NAME = "Qwen/Qwen2.5-7B"
DEFAULT_DATASET_CANDIDATES = [
    REPO_ROOT / "datasets" / "april" / "leandojo_infilling" / "thme_1m_100k.val.jsonl",
    REPO_ROOT / "datasets" / "april" / "thme_val.json",
    REPO_ROOT / "datasets" / "april" / "thme.val.json",
    REPO_ROOT / "datasets" / "april" / "leandojo_infilling" / "thme.val.jsonl",
    REPO_ROOT / "datasets" / "april" / "leandojo_infilling" / "thme_val.jsonl",
    REPO_ROOT / "datasets" / "april" / "leandojo_infilling" / "thme.val.json",
]


def resolve_eval_dataset_path(dataset_path: Optional[str]) -> Path:
    candidates: List[Path] = []
    if dataset_path:
        user_path = Path(dataset_path)
        candidates.append(user_path)
        if not user_path.is_absolute():
            candidates.append(REPO_ROOT / user_path)
    candidates.extend(DEFAULT_DATASET_CANDIDATES)

    seen: set[Path] = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
        if resolved.exists():
            return resolved

    checked = "\n".join(f"  - {path}" for path in unique_candidates)
    raise FileNotFoundError(
        "Could not find the validation dataset. Checked:\n"
        f"{checked}\n"
        "Pass --dataset-path explicitly if your file lives elsewhere."
    )


def iter_records_json_or_jsonl(data_path: Path) -> Iterable[Dict[str, Any]]:
    if data_path.suffix.lower() == ".jsonl":
        with data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield payload
        return

    with data_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
        return
    if isinstance(payload, dict):
        yield payload
        return
    raise ValueError(
        f"Unsupported dataset payload in {data_path}: {type(payload).__name__}"
    )


def load_raw_infilling_examples(
    data_path: Path,
    *,
    max_examples: Optional[int] = None,
    hole_token: str = "<HOLE>",
) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for item in iter_records_json_or_jsonl(data_path):
        if max_examples is not None and len(examples) >= max_examples:
            break

        infilling = item.get("infilling", {})
        if not isinstance(infilling, dict):
            continue

        proof_with_hole = str(infilling.get("proof_with_hole") or "").strip()
        target_tactic = str(infilling.get("target_tactic") or "").strip()
        if (
            not proof_with_hole
            or hole_token not in proof_with_hole
            or not target_tactic
            or target_tactic == "sorry"
        ):
            continue

        examples.append(
            {
                "full_name": str(item.get("full_name") or "").strip(),
                "theorem_statement": str(item.get("theorem_statement") or "").strip(),
                "proof_with_hole": proof_with_hole,
                "target_tactic": target_tactic,
            }
        )

    return examples


def normalize_tactic_for_exact_match(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "", str(value).strip())


def batched(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_device(device: str) -> torch.device:
    normalized = device.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def effective_dtype(device: torch.device, prefer_bf16: bool) -> torch.dtype:
    if device.type == "cuda" and prefer_bf16:
        return torch.bfloat16
    return torch.float32


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    clipped_q = min(max(float(q), 0.0), 1.0)
    ordered = sorted(float(value) for value in values)
    index = clipped_q * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarise_predictions(
    examples: Sequence[Dict[str, str]],
    predictions: Sequence[str],
    *,
    valid_mask: Optional[Sequence[bool]] = None,
    generation_seconds: float,
    latencies_seconds: Sequence[float],
) -> Dict[str, Any]:
    num_examples = len(examples)
    num_exact_matches = 0
    num_nonempty_predictions = 0
    num_valid_examples = 0
    valid_exact_matches = 0

    if valid_mask is None:
        valid_mask = [True] * num_examples
    if len(valid_mask) != num_examples:
        raise ValueError("valid_mask must align with examples.")

    for example, prediction, is_valid in zip(examples, predictions, valid_mask):
        expected_norm = normalize_tactic_for_exact_match(example.get("target_tactic", ""))
        predicted_norm = normalize_tactic_for_exact_match(prediction)
        if predicted_norm:
            num_nonempty_predictions += 1
        if expected_norm and predicted_norm == expected_norm:
            num_exact_matches += 1
        if is_valid:
            num_valid_examples += 1
            if expected_norm and predicted_norm == expected_norm:
                valid_exact_matches += 1

    accuracy_all = (
        float(num_exact_matches) / float(num_examples) if num_examples > 0 else 0.0
    )
    accuracy_valid = (
        float(valid_exact_matches) / float(num_valid_examples)
        if num_valid_examples > 0
        else 0.0
    )
    nonempty_rate = (
        float(num_nonempty_predictions) / float(num_examples) if num_examples > 0 else 0.0
    )

    return {
        "num_examples": num_examples,
        "num_valid_examples": num_valid_examples,
        "num_invalid_examples": num_examples - num_valid_examples,
        "num_exact_matches": num_exact_matches,
        "num_exact_matches_on_valid": valid_exact_matches,
        "num_nonempty_predictions": num_nonempty_predictions,
        "exact_match_accuracy": accuracy_all,
        "exact_match_accuracy_on_valid": accuracy_valid,
        "nonempty_prediction_rate": nonempty_rate,
        "generation_seconds": float(generation_seconds),
        "examples_per_second": (
            float(num_valid_examples) / float(generation_seconds)
            if generation_seconds > 0.0
            else 0.0
        ),
        "latency_seconds_mean_effective": (
            sum(latencies_seconds) / len(latencies_seconds) if latencies_seconds else 0.0
        ),
        "latency_seconds_p50_effective": percentile(latencies_seconds, 0.50),
        "latency_seconds_p95_effective": percentile(latencies_seconds, 0.95),
    }


def write_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Optional[str], rows: Sequence[Dict[str, Any]]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_ar_model(
    model_name_or_path: str,
    *,
    device: torch.device,
    trust_remote_code: bool = True,
    use_lora_adapter: bool = False,
    bf16: bool = True,
):
    dtype = effective_dtype(device, bf16)
    config = sanitize_rope_scaling(
        AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_lora_adapter:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    model = model.to(device)
    model.eval()
    return tokenizer, model


def load_diffusion_model_bundle(
    model_name_or_path: str,
    *,
    device: torch.device,
    trust_remote_code: bool = True,
    use_lora_adapter: bool = False,
    bf16: bool = True,
):
    dtype = effective_dtype(device, bf16)
    components = load_diffusion_components(
        model_name_or_path,
        torch_dtype=dtype,
        device=device,
        trust_remote_code=trust_remote_code,
        use_lora_adapter=use_lora_adapter,
    )
    components.model.eval()
    return components


def prepare_diffusion_inputs(
    examples: Sequence[Dict[str, str]],
    *,
    tokenizer,
    mask_token_id: int,
    max_length: int,
    mask_span_length: int,
    hole_token: str = "<HOLE>",
) -> List[Optional[Dict[str, Any]]]:
    prepared: List[Optional[Dict[str, Any]]] = []
    for example in examples:
        proof_with_hole = example.get("proof_with_hole", "")
        theorem_statement = example.get("theorem_statement", "")
        hole_pos = proof_with_hole.find(hole_token)
        if hole_pos == -1:
            prepared.append(None)
            continue

        prompt_prefix, _ = _build_infilling_prompt_prefix(
            tokenizer,
            theorem_statement=theorem_statement,
        )
        prompt_ids = tokenizer(
            prompt_prefix,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        if len(prompt_ids) + mask_span_length > max_length:
            prepared.append(None)
            continue

        available_for_proof = max_length - len(prompt_ids)
        left_text = proof_with_hole[:hole_pos]
        right_text = proof_with_hole[hole_pos + len(hole_token) :]

        left_ids = tokenizer(
            left_text,
            add_special_tokens=False,
            truncation=True,
            max_length=available_for_proof - mask_span_length,
        )["input_ids"]
        right_ids = tokenizer(
            right_text,
            add_special_tokens=False,
            truncation=True,
            max_length=available_for_proof - mask_span_length - len(left_ids),
        )["input_ids"]

        total_len = len(left_ids) + mask_span_length + len(right_ids)
        if total_len > available_for_proof:
            available_for_right = (
                available_for_proof - len(left_ids) - mask_span_length
            )
            if available_for_right < 0:
                prepared.append(None)
                continue
            right_ids = right_ids[:available_for_right]

        assistant_start = len(prompt_ids)
        mask_start = assistant_start + len(left_ids)
        mask_end = mask_start + mask_span_length
        input_ids = (
            prompt_ids
            + left_ids
            + ([mask_token_id] * mask_span_length)
            + right_ids
        )
        prepared.append(
            {
                "input_ids": input_ids,
                "mask_start": mask_start,
                "mask_end": mask_end,
            }
        )
    return prepared


def predict_diffusion_batch(
    model,
    tokenizer,
    mask_token_id: int,
    rows: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
    steps: int,
    temperature: float,
    remasking: str,
) -> List[str]:
    if not rows:
        return []

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    max_len = max(len(row["input_ids"]) for row in rows)
    batch_size = len(rows)
    input_tensor = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
        device=device,
    )
    masked_input = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )

    supervised_positions_by_row: List[List[int]] = []
    for batch_idx, row in enumerate(rows):
        ids = list(row["input_ids"])
        seq_len = len(ids)
        input_tensor[batch_idx, :seq_len] = torch.tensor(
            ids,
            dtype=torch.long,
            device=device,
        )
        masked_input[batch_idx, :seq_len] = torch.tensor(
            ids,
            dtype=torch.long,
            device=device,
        )
        attention_mask[batch_idx, :seq_len] = 1

        mask_start = int(row["mask_start"])
        mask_end = int(row["mask_end"])
        positions = list(range(mask_start, min(mask_end, seq_len)))
        supervised_positions_by_row.append(positions)
        for position in positions:
            masked_input[batch_idx, position] = mask_token_id

    with torch.inference_mode():
        sampled = denoise_masked_sequence(
            model=model,
            input_ids=masked_input,
            attention_mask=attention_mask,
            mask_token_id=mask_token_id,
            steps=max(1, int(steps)),
            temperature=float(temperature),
            remasking=remasking,
        )

    predictions: List[str] = []
    for batch_idx, positions in enumerate(supervised_positions_by_row):
        predicted_ids = sampled[batch_idx].tolist()
        predicted_span = [predicted_ids[position] for position in positions]
        predictions.append(decode_until_stop(tokenizer, predicted_span, mask_token_id))
    return predictions


def predict_ar_batch(
    model,
    tokenizer,
    examples: Sequence[Dict[str, str]],
    *,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    if not examples:
        return []

    prompts = [
        _build_ar_infilling_prompt_prefix(
            tokenizer,
            theorem_statement=example.get("theorem_statement", ""),
            proof_with_hole=example.get("proof_with_hole", ""),
        )
        for example in examples
    ]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max(1, int(max_new_tokens)),
        "do_sample": temperature > 0.0,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0.0:
        generation_kwargs["temperature"] = float(temperature)
        generation_kwargs["top_p"] = float(top_p)

    with torch.inference_mode():
        generated = model.generate(**generation_kwargs)

    predictions: List[str] = []
    for batch_idx in range(generated.shape[0]):
        prompt_len = int(attention_mask[batch_idx].sum().item())
        completion_ids = generated[batch_idx].tolist()[prompt_len:]
        decoded = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        if decoded:
            decoded = decoded.splitlines()[0].strip()
        predictions.append(decoded)
    return predictions


def benchmark_diffusion_examples(
    examples: Sequence[Dict[str, str]],
    *,
    model,
    tokenizer,
    mask_token_id: int,
    device: torch.device,
    max_length: int,
    mask_span_length: int,
    batch_size: int,
    steps: int,
    temperature: float,
    remasking: str,
    warmup_batches: int = 1,
) -> Dict[str, Any]:
    prepared = prepare_diffusion_inputs(
        examples,
        tokenizer=tokenizer,
        mask_token_id=mask_token_id,
        max_length=max_length,
        mask_span_length=mask_span_length,
    )
    predictions = [""] * len(examples)
    valid_entries = [
        (idx, row) for idx, row in enumerate(prepared) if row is not None
    ]

    for batch_index, batch in enumerate(batched(valid_entries, batch_size)):
        if batch_index >= max(0, int(warmup_batches)):
            break
        rows = [row for _, row in batch]
        predict_diffusion_batch(
            model,
            tokenizer,
            mask_token_id,
            rows,
            device=device,
            steps=steps,
            temperature=temperature,
            remasking=remasking,
        )
        synchronize_device(device)

    generation_seconds = 0.0
    per_example_latencies: List[float] = []
    for batch in batched(valid_entries, batch_size):
        indices = [idx for idx, _ in batch]
        rows = [row for _, row in batch]
        synchronize_device(device)
        start_time = time.perf_counter()
        batch_predictions = predict_diffusion_batch(
            model,
            tokenizer,
            mask_token_id,
            rows,
            device=device,
            steps=steps,
            temperature=temperature,
            remasking=remasking,
        )
        synchronize_device(device)
        elapsed = time.perf_counter() - start_time
        generation_seconds += elapsed
        if batch_predictions:
            per_example_latencies.extend(
                [elapsed / float(len(batch_predictions))] * len(batch_predictions)
            )
        for idx, prediction in zip(indices, batch_predictions):
            predictions[idx] = prediction

    return summarise_predictions(
        examples,
        predictions,
        valid_mask=[row is not None for row in prepared],
        generation_seconds=generation_seconds,
        latencies_seconds=per_example_latencies,
    )


def benchmark_ar_examples(
    examples: Sequence[Dict[str, str]],
    *,
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    warmup_batches: int = 1,
) -> Dict[str, Any]:
    for batch_index, batch in enumerate(batched(examples, batch_size)):
        if batch_index >= max(0, int(warmup_batches)):
            break
        predict_ar_batch(
            model,
            tokenizer,
            batch,
            device=device,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        synchronize_device(device)

    predictions = [""] * len(examples)
    generation_seconds = 0.0
    per_example_latencies: List[float] = []
    for start_index, batch in enumerate(batched(examples, batch_size)):
        synchronize_device(device)
        start_time = time.perf_counter()
        batch_predictions = predict_ar_batch(
            model,
            tokenizer,
            batch,
            device=device,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        synchronize_device(device)
        elapsed = time.perf_counter() - start_time
        generation_seconds += elapsed
        if batch_predictions:
            per_example_latencies.extend(
                [elapsed / float(len(batch_predictions))] * len(batch_predictions)
            )
        batch_start = start_index * max(1, int(batch_size))
        for offset, prediction in enumerate(batch_predictions):
            predictions[batch_start + offset] = prediction

    return summarise_predictions(
        examples,
        predictions,
        valid_mask=[True] * len(examples),
        generation_seconds=generation_seconds,
        latencies_seconds=per_example_latencies,
    )


def default_diffusion_steps(model_name_or_path: str) -> int:
    return int(get_diffusion_sampling_config(model_name_or_path, mode="infilling").steps)


def default_diffusion_remasking(model_name_or_path: str) -> str:
    return str(
        get_diffusion_sampling_config(model_name_or_path, mode="infilling").remasking
    )


def print_summary_block(name: str, metrics: Dict[str, Any]) -> None:
    print(f"{name}:")
    print(f"  exact-match accuracy:          {metrics['exact_match_accuracy']:.4f}")
    print(
        "  exact-match accuracy (valid): "
        f"{metrics['exact_match_accuracy_on_valid']:.4f}"
    )
    print(
        f"  valid / total examples:        "
        f"{metrics['num_valid_examples']} / {metrics['num_examples']}"
    )
    print(
        f"  non-empty prediction rate:     "
        f"{metrics['nonempty_prediction_rate']:.4f}"
    )
    print(
        f"  generation time (s):           {metrics['generation_seconds']:.2f}"
    )
    print(f"  examples / second:             {metrics['examples_per_second']:.2f}")
    print(
        f"  effective latency mean/p50/p95 (ms): "
        f"{metrics['latency_seconds_mean_effective'] * 1000.0:.1f} / "
        f"{metrics['latency_seconds_p50_effective'] * 1000.0:.1f} / "
        f"{metrics['latency_seconds_p95_effective'] * 1000.0:.1f}"
    )


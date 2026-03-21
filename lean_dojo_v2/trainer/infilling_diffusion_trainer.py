from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from transformers import (
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from datasets import Dataset
from lean_dojo_v2.diffusion import (
    DEFAULT_DIFFUSION_MODEL_NAME,
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_DIFFUSION_TEMPERATURE,
    DEFAULT_REMASKING,
    create_diffusion_training_objective,
    decode_until_stop,
    denoise_masked_sequence,
    get_diffusion_sampling_config,
    load_diffusion_components,
    resolve_mask_token_id,
)

from .diffusion_sft_trainer import (
    MdlmTrainer,
    QuietProgressCallback,
    _ensure_prepare_inputs_for_generation,
)


def _normalize_optional_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_tactic_for_exact_match(value: Any) -> str:
    """Normalize tactic text for strict-but-whitespace-insensitive matching."""
    if value is None:
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", "", text)


def _truncate_text_for_table(value: Any, max_chars: int) -> str:
    text = _normalize_optional_text(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _is_main_process() -> bool:
    rank = os.getenv("RANK")
    if rank is not None:
        try:
            return int(rank) == 0
        except ValueError:
            pass

    local_rank = os.getenv("LOCAL_RANK")
    if local_rank is not None:
        try:
            return int(local_rank) == 0
        except ValueError:
            pass

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    return True


def _distributed_rank() -> int:
    rank = os.getenv("RANK")
    if rank is not None:
        try:
            return int(rank)
        except ValueError:
            pass

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    return 0


def _world_size() -> int:
    world_size = os.getenv("WORLD_SIZE")
    if world_size is not None:
        try:
            return int(world_size)
        except ValueError:
            pass

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()

    return 1


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _resolve_resume_checkpoint(
    resume_from_checkpoint: Optional[str], output_dir: str
) -> Optional[str]:
    if resume_from_checkpoint is None:
        return None

    normalized = str(resume_from_checkpoint).strip()
    if not normalized:
        return None

    def _latest_checkpoint(root: Path) -> Optional[Path]:
        candidates = [path for path in root.glob("checkpoint-*") if path.is_dir()]
        if not candidates:
            return None
        return max(candidates, key=_checkpoint_step)

    if normalized.lower() in {"latest", "last", "true"}:
        latest = _latest_checkpoint(Path(output_dir))
        if latest is None:
            raise FileNotFoundError(
                f"No checkpoint directories found under output_dir={output_dir!r}."
            )
        return str(latest)

    checkpoint_path = Path(normalized)
    if checkpoint_path.is_dir():
        if (checkpoint_path / "trainer_state.json").exists():
            return str(checkpoint_path)
        latest = _latest_checkpoint(checkpoint_path)
        if latest is not None:
            return str(latest)

    raise FileNotFoundError(
        "resume_from_checkpoint must point to a checkpoint directory "
        "or a directory containing checkpoint-* subdirectories: "
        f"{normalized!r}"
    )


def _build_infilling_user_content(theorem_statement: str) -> str:
    chunks: List[str] = [
        "Implement a Lean 4 proof for the given formal statement.",
    ]
    if theorem_statement:
        chunks.append(f"Formal statement:\n{theorem_statement}")
    chunks.append("Output only Lean 4 proof code.")
    return "\n\n".join(chunks)


def _build_infilling_prompt_prefix(
    tokenizer, theorem_statement: str
) -> Tuple[str, str]:
    user_content = _build_infilling_user_content(theorem_statement)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Lean 4 proof generator.\n"
                "You are given a theorem statement and a partial Lean 4 proof with a masked span.\n"
                "Output only the completed proof text, preserving surrounding code and syntax.\n"
                "Never use `sorry` or `admit`."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    prompt_prefix = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return prompt_prefix, user_content


def _iter_records_json_or_jsonl(data_path: str) -> Iterable[Dict[str, Any]]:
    """Yield dataset records from either JSON array/object or JSONL."""
    path = Path(data_path)
    if path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in tqdm(
                f,
                desc=f"Loading {path.name}",
                unit="line",
                leave=False,
            ):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict):
                    yield rec
        return

    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    if isinstance(loaded, list):
        for rec in tqdm(
            loaded,
            desc=f"Loading {path.name}",
            unit="record",
            leave=False,
        ):
            if isinstance(rec, dict):
                yield rec
        return
    if isinstance(loaded, dict):
        yield loaded
        return
    raise ValueError(
        f"Unsupported dataset payload in {data_path}: {type(loaded).__name__}"
    )


class InfillingMDMDataset:
    """Dataset for infilling-style training with MDM mask tokens embedded in proof context."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        mask_span_length: int = 64,
        max_examples: Optional[int] = None,
        hole_token: str = "<HOLE>",
        mask_token: str = "<|mdm_mask|>",
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_span_length = mask_span_length
        self.max_examples = max_examples
        self.hole_token = hole_token
        self.mask_token = mask_token

        self.mask_token_id = resolve_mask_token_id(tokenizer)

        self.data = self._process_data(
            _iter_records_json_or_jsonl(data_path),
            max_examples=max_examples,
        )

    def _process_data(
        self,
        data: Iterable[Dict[str, Any]],
        *,
        max_examples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        eos_id = self.tokenizer.eos_token_id

        for item in data:
            if max_examples is not None and len(processed) >= max_examples:
                break

            infilling = item.get("infilling", {})
            proof_with_hole = infilling.get("proof_with_hole", "")
            target_tactic = infilling.get("target_tactic", "").strip()
            theorem_statement = _normalize_optional_text(
                item.get("theorem_statement", "")
            )

            if not proof_with_hole or not target_tactic or target_tactic == "sorry":
                continue

            hole_pos = proof_with_hole.find(self.hole_token)
            if hole_pos == -1:
                continue

            left_text = proof_with_hole[:hole_pos]
            right_text = proof_with_hole[hole_pos + len(self.hole_token) :]

            prompt_prefix, user_content = _build_infilling_prompt_prefix(
                self.tokenizer, theorem_statement=theorem_statement
            )
            prompt_ids = self.tokenizer(
                prompt_prefix,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            if len(prompt_ids) + self.mask_span_length > self.max_length:
                continue

            available_for_proof = self.max_length - len(prompt_ids)

            left_ids = self.tokenizer(
                left_text,
                add_special_tokens=False,
                truncation=True,
                max_length=available_for_proof - self.mask_span_length,
            )["input_ids"]

            right_ids = self.tokenizer(
                right_text,
                add_special_tokens=False,
                truncation=True,
                max_length=available_for_proof - self.mask_span_length - len(left_ids),
            )["input_ids"]

            total_proof_len = len(left_ids) + self.mask_span_length + len(right_ids)
            if total_proof_len > available_for_proof:
                available_for_right = (
                    available_for_proof - len(left_ids) - self.mask_span_length
                )
                if available_for_right < 0:
                    continue
                right_ids = right_ids[:available_for_right]

            target_ids = self.tokenizer(
                target_tactic,
                add_special_tokens=False,
            )["input_ids"]
            max_target_len = max(0, self.mask_span_length - 1)
            if len(target_ids) > max_target_len:
                target_ids = target_ids[:max_target_len]
            target_plus_eos = target_ids + [eos_id]
            n_mask_pad = max(0, self.mask_span_length - len(target_plus_eos))
            span_tokens = target_plus_eos + ([self.mask_token_id] * n_mask_pad)

            assistant_ids = left_ids + span_tokens + right_ids
            input_ids = prompt_ids + assistant_ids
            assistant_start = len(prompt_ids)
            mask_start = assistant_start + len(left_ids)
            mask_end = mask_start + self.mask_span_length

            processed.append(
                {
                    "input_ids": input_ids,
                    "assistant_start": assistant_start,
                    "mask_start": mask_start,
                    "mask_end": mask_end,
                    "full_name": item.get("full_name", ""),
                    "theorem_statement": theorem_statement,
                    "prompt_user": user_content,
                    "proof_with_hole": proof_with_hole,
                    "target_tactic": target_tactic,
                    "state_before_hole": infilling.get("state_before_hole", ""),
                }
            )

        return processed

    def to_hf(self) -> Dataset:
        return Dataset.from_list(self.data)


class InfillingMDMCollator:
    """Padding collator with random masking over infilling spans."""

    def __init__(
        self,
        tokenizer,
        mask_token_id: int,
        pad_token_id: Optional[int] = None,
        min_mask_ratio: float = 0.01,
        max_mask_ratio: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.pad_token_id = (
            pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
        )
        if not (0.0 < min_mask_ratio <= max_mask_ratio <= 1.0):
            raise ValueError("Mask ratios must satisfy 0 < min <= max <= 1")
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_size = len(features)

        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, feature in enumerate(features):
            seq_len = len(feature["input_ids"])
            input_ids[i, :seq_len] = torch.tensor(
                feature["input_ids"], dtype=torch.long
            )
            attention_mask[i, :seq_len] = 1

            mask_start = int(feature.get("mask_start", -1))
            mask_end = int(feature.get("mask_end", -1))
            if mask_start < 0 or mask_end <= mask_start:
                continue
            mask_end = min(mask_end, seq_len)
            if mask_end <= mask_start:
                continue

            span_ids = input_ids[i, mask_start:mask_end].clone()
            span_positions = torch.arange(mask_start, mask_end, dtype=torch.long)

            pad_positions = span_positions[span_ids == self.mask_token_id]
            real_positions = span_positions[span_ids != self.mask_token_id]

            # Keep pad positions masked so the model learns to emit stop/pad mask tokens.
            if pad_positions.numel() > 0:
                labels[i, pad_positions] = input_ids[i, pad_positions]

            if real_positions.numel() > 0:
                mask_ratio = (
                    torch.empty(1)
                    .uniform_(self.min_mask_ratio, self.max_mask_ratio)
                    .item()
                )
                num_to_mask = max(1, int(round(real_positions.numel() * mask_ratio)))
                chosen = real_positions[
                    torch.randperm(real_positions.numel())[:num_to_mask]
                ]
                labels[i, chosen] = input_ids[i, chosen]
                input_ids[i, chosen] = self.mask_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class WandbQualitativeCallback(TrainerCallback):
    """Periodically log qualitative train/val samples to Weights & Biases."""

    def __init__(
        self,
        *,
        wandb_module=None,
        tokenizer,
        mask_token_id: int,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        num_samples_per_split: int,
        subset_eval_num_samples: int = 512,
        sampling_steps: int = DEFAULT_DIFFUSION_STEPS,
        sampling_temperature: float = DEFAULT_DIFFUSION_TEMPERATURE,
        sampling_remasking: str = DEFAULT_REMASKING,
        full_eval_batch_size: int = 8,
        enable_full_exact_match_eval: bool = False,
        seed: int = 0,
        log_history_table: bool = True,
        log_snapshot_table: bool = False,
        max_cell_chars: int = 4096,
    ):
        self.wandb = wandb_module
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.num_samples_per_split = max(1, int(num_samples_per_split))
        self.sampling_steps = max(1, int(sampling_steps))
        self.sampling_temperature = float(sampling_temperature)
        self.sampling_remasking = sampling_remasking
        self.full_eval_batch_size = max(1, int(full_eval_batch_size))
        self.enable_full_exact_match_eval = bool(enable_full_exact_match_eval)
        self.seed = seed
        self.log_history_table = bool(log_history_table)
        self.log_snapshot_table = bool(log_snapshot_table)
        self.max_cell_chars = max(128, int(max_cell_chars))
        self._last_logged_step_by_split: Dict[str, Optional[int]] = {
            "train": None,
            "val": None,
        }

        self.train_indices = self._sample_indices(train_dataset, seed)
        self.val_indices = (
            self._sample_indices(eval_dataset, seed + 1)
            if eval_dataset is not None
            else []
        )
        self.subset_eval_indices = (
            self._sample_indices(
                eval_dataset,
                seed + 2,
                num_samples=max(1, int(subset_eval_num_samples)),
            )
            if eval_dataset is not None
            else []
        )
        self._qual_columns = [
            "split",
            "sample_id",
            "theorem",
            "prompt",
            "expected_tactic",
            "predicted_tactic",
            "global_step",
            "epoch",
        ]
        self.train_history_rows: List[Tuple[Any, ...]] = []
        self.val_history_rows: List[Tuple[Any, ...]] = []
        self._last_full_eval_step: Optional[int] = None
        self._last_subset_eval_step: Optional[int] = None

    def _rows_to_table(self, rows: List[Tuple[Any, ...]]):
        table = self.wandb.Table(columns=self._qual_columns)
        for record in rows:
            table.add_data(*record)
        return table

    def _sample_indices(
        self,
        dataset: Optional[Dataset],
        seed: int,
        *,
        num_samples: Optional[int] = None,
    ) -> List[int]:
        if dataset is None:
            return []
        n = len(dataset)
        if n == 0:
            return []
        target_samples = self.num_samples_per_split if num_samples is None else num_samples
        k = min(max(1, int(target_samples)), n)
        if k == n:
            return list(range(n))
        rng = random.Random(seed)
        return sorted(rng.sample(range(n), k))

    def _log_payload(self, payload: Dict[str, Any], *, step: int) -> None:
        if self.wandb is None or not payload:
            return
        run = getattr(self.wandb, "run", None)
        current_step = getattr(run, "step", None)
        if current_step is None:
            self.wandb.log(payload, step=step)
        else:
            self.wandb.log(payload, step=max(step, int(current_step)))

    def _decode_predicted_tactic(self, token_ids: List[int]) -> str:
        return decode_until_stop(self.tokenizer, token_ids, self.mask_token_id)

    def _format_display_prompt(self, row: Dict[str, Any]) -> str:
        input_ids = row.get("input_ids")
        if isinstance(input_ids, list) and input_ids:
            display_ids = list(input_ids)
            if "mask_start" in row and "mask_end" in row:
                mask_start = max(0, int(row["mask_start"]))
                mask_end = min(int(row["mask_end"]), len(display_ids))
                for pos in range(mask_start, max(mask_start, mask_end)):
                    display_ids[pos] = self.mask_token_id
            elif "assistant_start" in row:
                assistant_start = max(0, int(row["assistant_start"]))
                for pos in range(assistant_start, len(display_ids)):
                    display_ids[pos] = self.mask_token_id
            return self.tokenizer.decode(
                display_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        prompt = row.get("prompt_user", "")
        proof_with_hole = row.get("proof_with_hole", "")
        if proof_with_hole:
            prompt = (
                f"{prompt}\n\nAssistant proof template:\n{proof_with_hole}"
                if prompt
                else proof_with_hole
            )
        return prompt

    def _predict_tactic(self, model, row: Dict[str, Any]) -> str:
        device = next(model.parameters()).device
        input_ids = torch.tensor(
            row["input_ids"], dtype=torch.long, device=device
        ).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        if "mask_start" in row and "mask_end" in row:
            mask_start = int(row["mask_start"])
            mask_end = int(row["mask_end"])
            supervised_positions = list(
                range(mask_start, min(mask_end, input_ids.shape[1]))
            )
            if not supervised_positions:
                return ""
            masked_input = input_ids.clone()
            masked_input[:, mask_start:mask_end] = self.mask_token_id
        elif "assistant_start" in row:
            assistant_start = int(row["assistant_start"])
            supervised_positions = list(range(assistant_start, input_ids.shape[1]))
            if not supervised_positions:
                return ""
            masked_input = input_ids.clone()
            masked_input[:, assistant_start:] = self.mask_token_id
        else:
            raise ValueError(
                "Expected infilling-chat ('mask_start'/'mask_end') or next-tactic "
                "('assistant_start') schema for qualitative sampling."
            )

        with torch.inference_mode():
            sampled = denoise_masked_sequence(
                model=model,
                input_ids=masked_input,
                attention_mask=attention_mask,
                mask_token_id=self.mask_token_id,
                steps=self.sampling_steps,
                temperature=self.sampling_temperature,
                remasking=self.sampling_remasking,
            )
        predicted_ids = sampled[0].tolist()
        predicted_span = [predicted_ids[pos] for pos in supervised_positions]
        return self._decode_predicted_tactic(predicted_span)

    def _predict_tactics_batch(self, model, rows: List[Dict[str, Any]]) -> List[str]:
        if not rows:
            return []

        device = next(model.parameters()).device
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        predictions: List[str] = [""] * len(rows)
        batch_entries: List[Tuple[int, List[int], List[int]]] = []
        for row_idx, row in enumerate(rows):
            input_ids = list(row["input_ids"])
            if "mask_start" in row and "mask_end" in row:
                mask_start = int(row["mask_start"])
                mask_end = int(row["mask_end"])
                supervised_positions = list(
                    range(mask_start, min(mask_end, len(input_ids)))
                )
            elif "assistant_start" in row:
                assistant_start = int(row["assistant_start"])
                supervised_positions = list(range(assistant_start, len(input_ids)))
            else:
                raise ValueError(
                    "Expected infilling-chat ('mask_start'/'mask_end') or next-tactic "
                    "('assistant_start') schema for qualitative sampling."
                )
            if supervised_positions:
                batch_entries.append((row_idx, input_ids, supervised_positions))

        if not batch_entries:
            return predictions

        max_len = max(len(entry[1]) for entry in batch_entries)
        batch_size = len(batch_entries)
        input_tensor = torch.full(
            (batch_size, max_len),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=device
        )

        for batch_idx, (_row_idx, ids, _positions) in enumerate(batch_entries):
            seq_len = len(ids)
            input_tensor[batch_idx, :seq_len] = torch.tensor(
                ids, dtype=torch.long, device=device
            )
            attention_mask[batch_idx, :seq_len] = 1

        masked_input = input_tensor.clone()
        for batch_idx, (_row_idx, _ids, positions) in enumerate(batch_entries):
            for pos in positions:
                masked_input[batch_idx, pos] = self.mask_token_id

        with torch.inference_mode():
            sampled = denoise_masked_sequence(
                model=model,
                input_ids=masked_input,
                attention_mask=attention_mask,
                mask_token_id=self.mask_token_id,
                steps=self.sampling_steps,
                temperature=self.sampling_temperature,
                remasking=self.sampling_remasking,
            )

        for batch_idx, (row_idx, _ids, positions) in enumerate(batch_entries):
            predicted_ids = sampled[batch_idx].tolist()
            predicted_span = [predicted_ids[pos] for pos in positions]
            predictions[row_idx] = self._decode_predicted_tactic(predicted_span)

        return predictions

    def _compute_exact_match_payload(
        self,
        *,
        model,
        dataset: Optional[Dataset],
        indices: List[int],
        metric_prefix: str,
    ) -> Dict[str, Any]:
        if dataset is None or not indices:
            return {
                f"{metric_prefix}_exact_match_accuracy": 0.0,
                f"{metric_prefix}_num_samples": 0,
                f"{metric_prefix}_num_exact_matches": 0,
            }

        num_examples = 0
        num_exact_matches = 0
        for start in range(0, len(indices), self.full_eval_batch_size):
            batch_indices = indices[start : start + self.full_eval_batch_size]
            rows = [dataset[idx] for idx in batch_indices]
            predicted_tactics = self._predict_tactics_batch(model, rows)
            for row, predicted_tactic in zip(rows, predicted_tactics):
                expected_tactic = row.get("target_tactic", "")
                expected_norm = _normalize_tactic_for_exact_match(expected_tactic)
                predicted_norm = _normalize_tactic_for_exact_match(predicted_tactic)
                num_examples += 1
                if expected_norm and expected_norm == predicted_norm:
                    num_exact_matches += 1

        exact_match_accuracy = (
            float(num_exact_matches) / float(num_examples) if num_examples > 0 else 0.0
        )
        return {
            f"{metric_prefix}_exact_match_accuracy": exact_match_accuracy,
            f"{metric_prefix}_num_samples": num_examples,
            f"{metric_prefix}_num_exact_matches": num_exact_matches,
        }

    def _log_split(
        self,
        *,
        model,
        split: str,
        dataset: Optional[Dataset],
        indices: List[int],
        step: int,
        epoch: Optional[float],
    ):
        if dataset is None or not indices:
            return {}

        history_rows = (
            self.train_history_rows if split == "train" else self.val_history_rows
        )
        snapshot_table = (
            self.wandb.Table(columns=self._qual_columns)
            if self.log_snapshot_table and self.wandb is not None
            else None
        )
        num_examples = 0
        num_exact_matches = 0
        for start in range(0, len(indices), self.full_eval_batch_size):
            batch_indices = indices[start : start + self.full_eval_batch_size]
            rows = [dataset[idx] for idx in batch_indices]
            predicted_tactics = self._predict_tactics_batch(model, rows)
            for idx, row, predicted_tactic in zip(
                batch_indices, rows, predicted_tactics
            ):
                full_name = row.get("full_name", "")
                theorem_statement = row.get("theorem_statement", "")
                theorem = full_name.strip()
                if theorem_statement:
                    theorem = (
                        f"{theorem}: {theorem_statement}"
                        if theorem
                        else theorem_statement
                    )

                prompt = self._format_display_prompt(row)
                expected_tactic = row.get("target_tactic", "")
                expected_norm = _normalize_tactic_for_exact_match(expected_tactic)
                predicted_norm = _normalize_tactic_for_exact_match(predicted_tactic)
                num_examples += 1
                if expected_norm and expected_norm == predicted_norm:
                    num_exact_matches += 1

                record = (
                    split,
                    int(idx),
                    _truncate_text_for_table(theorem, self.max_cell_chars),
                    _truncate_text_for_table(prompt, self.max_cell_chars),
                    _truncate_text_for_table(expected_tactic, self.max_cell_chars),
                    _truncate_text_for_table(predicted_tactic, self.max_cell_chars),
                    step,
                    float(epoch) if epoch is not None else None,
                )
                if self.log_history_table:
                    history_rows.append(record)
                if snapshot_table is not None:
                    snapshot_table.add_data(*record)

        payload: Dict[str, Any] = {
            f"{split}_qualitative_exact_match_accuracy": (
                float(num_exact_matches) / float(num_examples)
                if num_examples > 0
                else 0.0
            ),
            f"{split}_qualitative_num_samples": num_examples,
            f"{split}_qualitative_num_exact_matches": num_exact_matches,
        }
        if self.log_history_table and self.wandb is not None:
            payload[f"{split}_qualitative_samples"] = self._rows_to_table(history_rows)
        if snapshot_table is not None:
            payload[f"{split}_qualitative_samples_snapshot"] = snapshot_table
        self._log_payload(payload, step=step)
        return payload

    def _log_full_eval_exact_match(
        self,
        *,
        model,
        step: int,
        epoch: Optional[float],
    ) -> None:
        if self.eval_dataset is None or len(self.eval_dataset) == 0:
            return
        payload = self._compute_exact_match_payload(
            model=model,
            dataset=self.eval_dataset,
            indices=list(range(len(self.eval_dataset))),
            metric_prefix="val_full",
        )
        if epoch is not None:
            payload["val_full_epoch"] = float(epoch)
        self._log_payload(payload, step=step)

    def _maybe_log_split(
        self,
        *,
        model,
        split: str,
        step: int,
        epoch: Optional[float],
    ):
        last_logged_step = self._last_logged_step_by_split.get(split)
        if last_logged_step == step:
            return

        self._last_logged_step_by_split[split] = step
        if split == "train":
            dataset = self.train_dataset
            indices = self.train_indices
        elif split == "val":
            dataset = self.eval_dataset
            indices = self.val_indices
        else:
            raise ValueError(f"Unknown split for qualitative logging: {split}")

        was_training = model.training
        model.eval()
        try:
            self._log_split(
                model=model,
                split=split,
                dataset=dataset,
                indices=indices,
                step=step,
                epoch=epoch,
            )
        finally:
            if was_training:
                model.train()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        return

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step <= 0 or self.wandb is None:
            return
        try:
            self._maybe_log_split(
                model=model,
                split="train",
                step=state.global_step,
                epoch=float(state.epoch) if state.epoch is not None else None,
            )
        except Exception as exc:
            print(
                f"[wandb] train qualitative logging failed at epoch end "
                f"(step {state.global_step}): {exc}"
            )

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        was_training = model.training
        metrics = kwargs.get("metrics")
        try:
            model.eval()
            if self.wandb is not None:
                self._maybe_log_split(
                    model=model,
                    split="val",
                    step=state.global_step,
                    epoch=float(state.epoch) if state.epoch is not None else None,
                )
            if self._last_subset_eval_step != state.global_step:
                self._last_subset_eval_step = state.global_step
                subset_payload = self._compute_exact_match_payload(
                    model=model,
                    dataset=self.eval_dataset,
                    indices=self.subset_eval_indices,
                    metric_prefix="val_subset",
                )
                self._log_payload(subset_payload, step=state.global_step)
                if isinstance(metrics, dict):
                    metrics["eval_subset_exact_match_accuracy"] = subset_payload[
                        "val_subset_exact_match_accuracy"
                    ]
                    metrics["eval_subset_num_samples"] = subset_payload[
                        "val_subset_num_samples"
                    ]
                    metrics["eval_subset_num_exact_matches"] = subset_payload[
                        "val_subset_num_exact_matches"
                    ]
            if (
                self.wandb is not None
                and self.enable_full_exact_match_eval
                and self._last_full_eval_step != state.global_step
            ):
                self._last_full_eval_step = state.global_step
                self._log_full_eval_exact_match(
                    model=model,
                    step=state.global_step,
                    epoch=float(state.epoch) if state.epoch is not None else None,
                )
        except Exception as exc:
            print(
                f"[wandb] qualitative eval logging failed at step {state.global_step}: {exc}"
            )
        finally:
            if was_training:
                model.train()


class InfillingDiffusionTrainer:
    """Trainer for infilling-style diffusion training on Lean proofs.

    Uses fixed-length mask spans embedded directly in the proof context.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DIFFUSION_MODEL_NAME,
        train_path: str = "",
        val_path: Optional[str] = None,
        output_dir: str = "outputs/infilling-mdm",
        epochs: float = 3.0,
        batch_size: int = 8,
        lr: float = 2e-5,
        max_length: int = 1024,
        mask_span_length: int = 64,
        min_mask_ratio: float = 0.01,
        max_mask_ratio: float = 1.0,
        lora_config: Optional[LoraConfig] = None,
        bf16: Optional[bool] = None,
        logging_steps: int = 10,
        save_strategy: str = "epoch",
        eval_every_n_epochs: int = 1,
        wandb_project: Optional[str] = "infilling",
        wandb_run_name: Optional[str] = None,
        qual_num_samples_per_split: int = 64,
        subset_eval_num_samples: int = 512,
        qual_sampling_steps: Optional[int] = 16,
        full_exact_match_eval: bool = False,
        max_train_examples: Optional[int] = None,
        max_val_examples: Optional[int] = None,
        trust_remote_code: bool = True,
        resume_from_checkpoint: Optional[str] = None,
    ):
        if not train_path:
            raise ValueError("train_path is required")

        self.model_name = model_name
        self.train_path = train_path
        self.val_path = val_path
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.mask_span_length = mask_span_length
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.lora_config = lora_config
        self.use_lora = lora_config is not None
        self.logging_steps = logging_steps
        self.save_strategy = save_strategy
        if eval_every_n_epochs <= 0:
            raise ValueError("eval_every_n_epochs must be positive.")
        self.eval_every_n_epochs = int(eval_every_n_epochs)
        self.sampling_config = get_diffusion_sampling_config(
            model_name, mode="infilling"
        )
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.qual_num_samples_per_split = qual_num_samples_per_split
        self.subset_eval_num_samples = max(1, int(subset_eval_num_samples))
        self.qual_sampling_steps = (
            self.sampling_config.steps
            if qual_sampling_steps is None
            else max(1, int(qual_sampling_steps))
        )
        self.full_exact_match_eval = bool(full_exact_match_eval)
        self.max_train_examples = max_train_examples
        self.max_val_examples = max_val_examples
        self.wandb_enabled = bool(wandb_project)
        self.is_main_process = _is_main_process()
        self.trust_remote_code = trust_remote_code
        self.resume_from_checkpoint = _resolve_resume_checkpoint(
            resume_from_checkpoint, output_dir
        )

        # Auto-detect bf16 if not specified
        if bf16 is None:
            bf16 = torch.cuda.is_available()
        self.bf16 = bf16

        self.components = load_diffusion_components(
            model_name,
            torch_dtype=torch.bfloat16 if self.bf16 else torch.float32,
            trust_remote_code=self.trust_remote_code,
            for_training=True,
        )
        self.family = self.components.family
        self.tokenizer = self.components.tokenizer
        self.model = self.components.model

        if self.use_lora:
            self.model = self._apply_lora()

        # Get mask token ID
        self.mask_token_id = self.components.mask_token_id
        self.training_objective = create_diffusion_training_objective(
            family=self.family,
            mode="infilling",
            tokenizer=self.tokenizer,
            mask_token_id=self.mask_token_id,
            min_mask_ratio=self.min_mask_ratio,
            max_mask_ratio=self.max_mask_ratio,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Setup training arguments
        eval_strategy = "epoch" if val_path else "no"
        load_best_model_at_end = bool(val_path) and self.eval_every_n_epochs == 1
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            bf16=self.bf16,
            fp16=False,
            remove_unused_columns=False,
            report_to=["wandb"]
            if self.wandb_enabled and self.is_main_process
            else "none",
            save_strategy=save_strategy,
            eval_strategy=eval_strategy,
            logging_steps=logging_steps,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=(
                "eval_subset_exact_match_accuracy" if load_best_model_at_end else None
            ),
            greater_is_better=True if load_best_model_at_end else None,
            run_name=wandb_run_name,
        )

    def _apply_lora(self):
        """Apply LoRA to the model."""
        if self.lora_config is None:
            raise ValueError("LoRA config is required when use_lora is True")
        _ensure_prepare_inputs_for_generation(self.model)
        model = get_peft_model(self.model, self.lora_config)
        model.print_trainable_parameters()
        return model

    def _ensure_lora_trainable(self) -> int:
        if not self.use_lora:
            return sum(1 for param in self.model.parameters() if param.requires_grad)

        trainable_tensors = 0
        for name, param in self.model.named_parameters():
            should_train = "lora_" in name
            if param.requires_grad != should_train:
                param.requires_grad = should_train
            if should_train:
                trainable_tensors += 1
        return trainable_tensors

    def _load_dataset(
        self, data_path: str, max_examples: Optional[int] = None
    ) -> Dataset:
        """Load and process dataset."""
        dataset = InfillingMDMDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            mask_span_length=self.mask_span_length,
            max_examples=max_examples,
        )
        hf_dataset = dataset.to_hf()
        if max_examples is not None:
            if max_examples <= 0:
                raise ValueError("max_examples must be positive when provided.")
            hf_dataset = hf_dataset.select(range(min(max_examples, len(hf_dataset))))
        return hf_dataset

    def _build_infilling_input(
        self,
        proof_with_hole: str,
        *,
        theorem_statement: str = "",
        hole_token: str = "<HOLE>",
    ) -> Optional[Tuple[List[int], int, int, int]]:
        hole_pos = proof_with_hole.find(hole_token)
        if hole_pos == -1:
            return None

        theorem_statement = _normalize_optional_text(theorem_statement)
        prompt_prefix, _ = _build_infilling_prompt_prefix(
            self.tokenizer, theorem_statement=theorem_statement
        )
        prompt_ids = self.tokenizer(
            prompt_prefix,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        if len(prompt_ids) + self.mask_span_length > self.max_length:
            return None
        available_for_proof = self.max_length - len(prompt_ids)

        left_text = proof_with_hole[:hole_pos]
        right_text = proof_with_hole[hole_pos + len(hole_token) :]
        left_ids = self.tokenizer(
            left_text,
            add_special_tokens=False,
            truncation=True,
            max_length=available_for_proof - self.mask_span_length,
        )["input_ids"]
        right_ids = self.tokenizer(
            right_text,
            add_special_tokens=False,
            truncation=True,
            max_length=available_for_proof - self.mask_span_length - len(left_ids),
        )["input_ids"]

        total_len = len(left_ids) + self.mask_span_length + len(right_ids)
        if total_len > available_for_proof:
            available_for_right = (
                available_for_proof - len(left_ids) - self.mask_span_length
            )
            if available_for_right < 0:
                return None
            right_ids = right_ids[:available_for_right]

        assistant_start = len(prompt_ids)
        mask_start = assistant_start + len(left_ids)
        mask_end = mask_start + self.mask_span_length
        input_ids = (
            prompt_ids
            + left_ids
            + ([self.mask_token_id] * self.mask_span_length)
            + right_ids
        )
        return input_ids, assistant_start, mask_start, mask_end

    def sample_infilling_tactic(
        self,
        proof_with_hole: str,
        *,
        theorem_statement: str = "",
        num_return_sequences: int = 1,
        steps: Optional[int] = None,
        temperature: Optional[float] = None,
        remasking: Optional[str] = None,
        hole_token: str = "<HOLE>",
    ) -> List[str]:
        """Iteratively denoise the infilling hole and decode predicted tactic text."""
        built = self._build_infilling_input(
            proof_with_hole,
            theorem_statement=theorem_statement,
            hole_token=hole_token,
        )
        if built is None:
            return []
        input_ids, _assistant_start, mask_start, mask_end = built

        x = torch.tensor(
            input_ids, dtype=torch.long, device=self.model.device
        ).unsqueeze(0)
        x = x.repeat(num_return_sequences, 1)
        attention_mask = torch.ones_like(x)
        effective_steps = (
            self.sampling_config.steps if steps is None else max(1, int(steps))
        )
        effective_temperature = (
            self.sampling_config.temperature
            if temperature is None
            else float(temperature)
        )
        effective_remasking = (
            self.sampling_config.remasking if remasking is None else remasking
        )

        sampled = denoise_masked_sequence(
            model=self.model,
            input_ids=x,
            attention_mask=attention_mask,
            mask_token_id=self.mask_token_id,
            steps=effective_steps,
            temperature=effective_temperature,
            remasking=effective_remasking,
        )

        results: List[str] = []
        for seq in sampled.tolist():
            predicted_span = seq[mask_start:mask_end]
            results.append(
                decode_until_stop(self.tokenizer, predicted_span, self.mask_token_id)
            )
        return results

    def train(self):
        """Run training."""
        wandb_module = None
        if self.wandb_enabled and self.is_main_process:
            import wandb

            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config={
                    "model_name": self.model_name,
                    "train_path": self.train_path,
                    "val_path": self.val_path,
                    "output_dir": self.output_dir,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.lr,
                    "max_length": self.max_length,
                    "mask_span_length": self.mask_span_length,
                    "min_mask_ratio": self.min_mask_ratio,
                    "max_mask_ratio": self.max_mask_ratio,
                    "qual_num_samples_per_split": self.qual_num_samples_per_split,
                    "subset_eval_num_samples": self.subset_eval_num_samples,
                    "qual_sampling_steps": self.qual_sampling_steps,
                    "full_exact_match_eval": self.full_exact_match_eval,
                    "max_train_examples": self.max_train_examples,
                    "max_val_examples": self.max_val_examples,
                    "use_lora": self.use_lora,
                },
            )
            wandb_module = wandb

        # Load datasets
        print(f"Loading training data from {self.train_path}")
        train_dataset = self._load_dataset(self.train_path, self.max_train_examples)
        print(f"Loaded {len(train_dataset)} training examples")

        eval_dataset = None
        if self.val_path:
            print(f"Loading validation data from {self.val_path}")
            eval_dataset = self._load_dataset(self.val_path, self.max_val_examples)
            print(f"Loaded {len(eval_dataset)} validation examples")

        # Create collator
        data_collator = self.training_objective

        callbacks = [
            WandbQualitativeCallback(
                wandb_module=wandb_module,
                tokenizer=self.tokenizer,
                mask_token_id=self.mask_token_id,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                num_samples_per_split=self.qual_num_samples_per_split,
                subset_eval_num_samples=self.subset_eval_num_samples,
                sampling_steps=self.qual_sampling_steps,
                full_eval_batch_size=max(1, int(self.batch_size) * 2),
                enable_full_exact_match_eval=self.full_exact_match_eval,
            )
        ]

        # Create trainer
        trainer = MdlmTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
            training_objective=self.training_objective,
        )
        if eval_dataset is not None and self.eval_every_n_epochs > 1:
            steps_per_epoch = max(1, len(trainer.get_train_dataloader()))
            trainer.args.eval_strategy = "steps"
            trainer.args.eval_steps = steps_per_epoch * self.eval_every_n_epochs
            trainer.args.load_best_model_at_end = False
            trainer.args.metric_for_best_model = None
            if self.is_main_process:
                effective_eval_epochs = trainer.args.eval_steps / steps_per_epoch
                print(
                    "Validation schedule: "
                    f"every {trainer.args.eval_steps} optimizer steps "
                    f"(about {effective_eval_epochs:g} epochs)"
                )
        # Keep tqdm with dummy quiet callback but suppress raw metric-dict prints from HF callbacks.
        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(QuietProgressCallback())
        trainable_tensors = self._ensure_lora_trainable()
        if _world_size() > 1:
            print(
                f"[rank {_distributed_rank()}] trainable parameter tensors before DDP wrap: "
                f"{trainable_tensors}"
            )
        try:
            # Train
            print("Starting training...")
            if self.resume_from_checkpoint is not None:
                print(f"Resuming from checkpoint: {self.resume_from_checkpoint}")
                trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
            else:
                trainer.train()

            # Save
            print(f"Saving model to {self.output_dir}")
            if self.use_lora:
                self.model.save_pretrained(self.output_dir)
            else:
                trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)

            # Print summary
            print("\nTraining complete!")
            print(f"  Train examples: {len(train_dataset)}")
            if eval_dataset:
                print(f"  Val examples: {len(eval_dataset)}")
            print(f"  Output: {self.output_dir}")
        finally:
            if wandb_module is not None and wandb_module.run is not None:
                wandb_module.finish()

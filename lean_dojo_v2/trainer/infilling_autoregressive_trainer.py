from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from .diffusion_sft_trainer import (
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


def _build_ar_infilling_prompt_prefix(
    tokenizer,
    theorem_statement: str,
    proof_with_hole: str,
) -> str:
    """Build chat prompt for autoregressive infilling of a single tactic."""
    system_chunks = [
        "You are a Lean 4 proof infilling assistant.",
        "Given a theorem statement and a partial Lean 4 proof with exactly one <HOLE>, output exactly the missing tactic.",
        "Rules:",
        "- Output only the missing tactic text.",
        "- Do not repeat or rewrite surrounding proof context.",
        "- Single line only when possible.",
        "- Never use `sorry` or `admit`.",
    ]
    system_chunks.append(
        "- Do not include reasoning, scratch work, or <think>...</think> content."
    )
    system_prompt = "\n".join(system_chunks)

    user_chunks = [
        "Fill the single <HOLE> in this Lean 4 proof.",
    ]
    theorem_statement = _normalize_optional_text(theorem_statement)
    if theorem_statement:
        user_chunks.append(f"Formal statement:\n{theorem_statement}")
    user_chunks.append(f"Partial proof:\n{proof_with_hole}")
    user_chunks.append("Return only the missing tactic.")
    user_prompt = "\n\n".join(user_chunks)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


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
    raise ValueError(f"Unsupported dataset payload in {data_path}: {type(loaded).__name__}")


class InfillingARDataset:
    """Dataset for autoregressive infilling on APRIL-style infilling examples."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        *,
        max_length: int = 1024,
        max_examples: Optional[int] = None,
        hole_token: str = "<HOLE>",
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_examples = max_examples
        self.hole_token = hole_token

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
        if eos_id is None:
            raise ValueError("Tokenizer must define eos_token_id for AR infilling training.")

        for item in data:
            if max_examples is not None and len(processed) >= max_examples:
                break

            infilling = item.get("infilling", {})
            proof_with_hole = _normalize_optional_text(infilling.get("proof_with_hole", ""))
            target_tactic = _normalize_optional_text(infilling.get("target_tactic", ""))
            theorem_statement = _normalize_optional_text(item.get("theorem_statement", ""))
            full_name = _normalize_optional_text(item.get("full_name", ""))

            if not proof_with_hole or self.hole_token not in proof_with_hole:
                continue
            if not target_tactic or target_tactic == "sorry":
                continue

            prompt_prefix = _build_ar_infilling_prompt_prefix(
                self.tokenizer,
                theorem_statement,
                proof_with_hole,
            )
            prompt_ids = self.tokenizer(
                prompt_prefix,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            if len(prompt_ids) >= self.max_length:
                continue

            target_ids = self.tokenizer(
                target_tactic,
                add_special_tokens=False,
            )["input_ids"]
            if not target_ids:
                continue
            target_plus_eos = target_ids + [eos_id]

            available_for_target = self.max_length - len(prompt_ids)
            if available_for_target <= 0:
                continue
            if len(target_plus_eos) > available_for_target:
                target_plus_eos = target_plus_eos[:available_for_target]

            input_ids = prompt_ids + target_plus_eos
            labels = ([-100] * len(prompt_ids)) + target_plus_eos
            if len(input_ids) != len(labels):
                continue

            processed.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "full_name": full_name,
                    "theorem_statement": theorem_statement,
                    "proof_with_hole": proof_with_hole,
                    "target_tactic": target_tactic,
                }
            )

        return processed

    def to_hf(self) -> Dataset:
        return Dataset.from_list(self.data)


class InfillingARCollator:
    """Padding collator for autoregressive infilling examples."""

    def __init__(self, tokenizer, pad_token_id: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_size = len(features)

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, feature in enumerate(features):
            seq_len = len(feature["input_ids"])
            input_ids[i, :seq_len] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = torch.tensor(feature["labels"], dtype=torch.long)

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
        max_new_tokens: int,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        num_samples_per_split: int,
        subset_eval_num_samples: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        full_eval_batch_size: int = 8,
        enable_full_exact_match_eval: bool = False,
        seed: int = 0,
        log_history_table: bool = True,
        log_snapshot_table: bool = False,
    ):
        self.wandb = wandb_module
        self.tokenizer = tokenizer
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.num_samples_per_split = max(1, int(num_samples_per_split))
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.full_eval_batch_size = max(1, int(full_eval_batch_size))
        self.enable_full_exact_match_eval = bool(enable_full_exact_match_eval)
        self.seed = seed
        self.log_history_table = bool(log_history_table)
        self.log_snapshot_table = bool(log_snapshot_table)
        self._last_logged_step_by_split: Dict[str, Optional[int]] = {
            "train": None,
            "val": None,
        }

        self.train_indices = self._sample_indices(train_dataset, seed)
        self.val_indices = (
            self._sample_indices(eval_dataset, seed + 1) if eval_dataset is not None else []
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

    def _predict_tactic(self, model, row: Dict[str, Any]) -> str:
        theorem_statement = _normalize_optional_text(row.get("theorem_statement", ""))
        proof_with_hole = _normalize_optional_text(row.get("proof_with_hole", ""))
        if not proof_with_hole:
            return ""

        prompt_prefix = _build_ar_infilling_prompt_prefix(
            self.tokenizer,
            theorem_statement=theorem_statement,
            proof_with_hole=proof_with_hole,
        )
        encoded = self.tokenizer(prompt_prefix, return_tensors="pt")
        input_ids = encoded.input_ids.to(model.device)
        attention_mask = encoded.attention_mask.to(model.device)

        do_sample = self.temperature > 0.0
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = input_ids.shape[1]
        completion_ids = generated[0].tolist()[prompt_len:]
        decoded = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        if decoded:
            decoded = decoded.splitlines()[0].strip()
        return decoded

    def _predict_tactics_batch(self, model, rows: List[Dict[str, Any]]) -> List[str]:
        if not rows:
            return []

        prompts: List[str] = []
        for row in rows:
            theorem_statement = _normalize_optional_text(row.get("theorem_statement", ""))
            proof_with_hole = _normalize_optional_text(row.get("proof_with_hole", ""))
            if not proof_with_hole:
                prompts.append("")
                continue
            prompts.append(
                _build_ar_infilling_prompt_prefix(
                    self.tokenizer,
                    theorem_statement=theorem_statement,
                    proof_with_hole=proof_with_hole,
                )
            )

        valid_indices = [i for i, prompt in enumerate(prompts) if prompt]
        predictions: List[str] = [""] * len(rows)
        if not valid_indices:
            return predictions

        valid_prompts = [prompts[i] for i in valid_indices]
        encoded = self.tokenizer(valid_prompts, return_tensors="pt", padding=True)
        input_ids = encoded.input_ids.to(model.device)
        attention_mask = encoded.attention_mask.to(model.device)

        do_sample = self.temperature > 0.0
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        for batch_idx, original_idx in enumerate(valid_indices):
            prompt_len = int(attention_mask[batch_idx].sum().item())
            completion_ids = generated[batch_idx].tolist()[prompt_len:]
            decoded = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            if decoded:
                decoded = decoded.splitlines()[0].strip()
            predictions[original_idx] = decoded

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

        history_rows = self.train_history_rows if split == "train" else self.val_history_rows
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
            for idx, row, predicted_tactic in zip(batch_indices, rows, predicted_tactics):
                full_name = row.get("full_name", "")
                theorem_statement = row.get("theorem_statement", "")
                theorem = full_name.strip()
                if theorem_statement:
                    theorem = f"{theorem}: {theorem_statement}" if theorem else theorem_statement

                prompt = row.get("proof_with_hole", "")
                expected_tactic = row.get("target_tactic", "")
                expected_norm = _normalize_tactic_for_exact_match(expected_tactic)
                predicted_norm = _normalize_tactic_for_exact_match(predicted_tactic)
                num_examples += 1
                if expected_norm and expected_norm == predicted_norm:
                    num_exact_matches += 1

                record = (
                    split,
                    int(idx),
                    theorem,
                    prompt,
                    expected_tactic,
                    predicted_tactic,
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
            print(f"[wandb] qualitative eval logging failed at step {state.global_step}: {exc}")
        finally:
            if was_training:
                model.train()


class InfillingAutoregressiveTrainer:
    """Trainer for autoregressive infilling on Lean proof holes."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B",
        train_path: str = "",
        val_path: Optional[str] = None,
        output_dir: str = "outputs/infilling-ar",
        epochs: float = 3.0,
        batch_size: int = 8,
        lr: float = 2e-5,
        max_length: int = 1024,
        max_new_tokens: int = 128,
        lora_config: Optional[LoraConfig] = None,
        bf16: Optional[bool] = None,
        logging_steps: int = 10,
        save_strategy: str = "epoch",
        wandb_project: Optional[str] = "infilling-ar",
        wandb_run_name: Optional[str] = None,
        qual_num_samples_per_split: int = 64,
        subset_eval_num_samples: int = 512,
        qual_sampling_temperature: float = 0.0,
        qual_sampling_top_p: float = 0.9,
        full_exact_match_eval: bool = False,
        max_train_examples: Optional[int] = None,
        max_val_examples: Optional[int] = None,
        trust_remote_code: bool = True,
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
        self.max_new_tokens = max_new_tokens
        self.lora_config = lora_config
        self.use_lora = lora_config is not None
        self.logging_steps = logging_steps
        self.save_strategy = save_strategy
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.qual_num_samples_per_split = qual_num_samples_per_split
        self.subset_eval_num_samples = max(1, int(subset_eval_num_samples))
        self.qual_sampling_temperature = qual_sampling_temperature
        self.qual_sampling_top_p = qual_sampling_top_p
        self.full_exact_match_eval = bool(full_exact_match_eval)
        self.max_train_examples = max_train_examples
        self.max_val_examples = max_val_examples
        self.wandb_enabled = bool(wandb_project)
        self.is_main_process = _is_main_process()
        self.trust_remote_code = trust_remote_code

        if bf16 is None:
            bf16 = torch.cuda.is_available()
        self.bf16 = bf16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=self.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.bf16 else torch.float32,
            trust_remote_code=self.trust_remote_code,
        )
        if self.use_lora:
            self.model = self._apply_lora()

        eval_strategy = "epoch" if val_path else "no"
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            bf16=self.bf16,
            fp16=False,
            remove_unused_columns=False,
            report_to=["wandb"] if self.wandb_enabled and self.is_main_process else "none",
            save_strategy=save_strategy,
            eval_strategy=eval_strategy,
            logging_steps=logging_steps,
            load_best_model_at_end=True if val_path else False,
            metric_for_best_model=(
                "eval_subset_exact_match_accuracy" if val_path else None
            ),
            greater_is_better=True if val_path else None,
            run_name=wandb_run_name,
        )

    def _apply_lora(self):
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

    def _load_dataset(self, data_path: str, max_examples: Optional[int] = None) -> Dataset:
        dataset = InfillingARDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            max_examples=max_examples,
        )
        hf_dataset = dataset.to_hf()
        if max_examples is not None:
            if max_examples <= 0:
                raise ValueError("max_examples must be positive when provided.")
            hf_dataset = hf_dataset.select(range(min(max_examples, len(hf_dataset))))
        return hf_dataset

    def sample_infilling_tactic(
        self,
        proof_with_hole: str,
        *,
        theorem_statement: str = "",
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        hole_token: str = "<HOLE>",
    ) -> List[str]:
        if not proof_with_hole or hole_token not in proof_with_hole:
            return []

        prompt_prefix = _build_ar_infilling_prompt_prefix(
            self.tokenizer,
            theorem_statement=theorem_statement,
            proof_with_hole=proof_with_hole,
        )
        encoded = self.tokenizer(prompt_prefix, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.model.device)
        attention_mask = encoded.attention_mask.to(self.model.device)

        do_sample = temperature > 0.0
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                num_return_sequences=max(1, int(num_return_sequences)),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = input_ids.shape[1]
        results: List[str] = []
        for seq in generated.tolist():
            completion_ids = seq[prompt_len:]
            decoded = self.tokenizer.decode(
                completion_ids, skip_special_tokens=True
            ).strip()
            if decoded:
                decoded = decoded.splitlines()[0].strip()
            results.append(decoded)
        return results

    def train(self):
        wandb_module = None
        if self.wandb_enabled and self.is_main_process:
            try:
                import wandb  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "wandb logging is enabled but wandb is not installed. "
                    "Install it with `pip install wandb`."
                ) from exc

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
                    "max_new_tokens": self.max_new_tokens,
                    "qual_num_samples_per_split": self.qual_num_samples_per_split,
                    "subset_eval_num_samples": self.subset_eval_num_samples,
                    "qual_sampling_temperature": self.qual_sampling_temperature,
                    "qual_sampling_top_p": self.qual_sampling_top_p,
                    "full_exact_match_eval": self.full_exact_match_eval,
                    "max_train_examples": self.max_train_examples,
                    "max_val_examples": self.max_val_examples,
                    "use_lora": self.use_lora,
                },
            )
            wandb_module = wandb

        print(f"Loading training data from {self.train_path}")
        train_dataset = self._load_dataset(self.train_path, self.max_train_examples)
        print(f"Loaded {len(train_dataset)} training examples")

        eval_dataset = None
        if self.val_path:
            print(f"Loading validation data from {self.val_path}")
            eval_dataset = self._load_dataset(self.val_path, self.max_val_examples)
            print(f"Loaded {len(eval_dataset)} validation examples")

        data_collator = InfillingARCollator(
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        callbacks = [
            WandbQualitativeCallback(
                wandb_module=wandb_module,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                num_samples_per_split=self.qual_num_samples_per_split,
                subset_eval_num_samples=self.subset_eval_num_samples,
                temperature=self.qual_sampling_temperature,
                top_p=self.qual_sampling_top_p,
                full_eval_batch_size=max(1, int(self.batch_size) * 2),
                enable_full_exact_match_eval=self.full_exact_match_eval,
            )
        ]

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )
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
            print("Starting training...")
            trainer.train()

            print(f"Saving model to {self.output_dir}")
            if self.use_lora:
                self.model.save_pretrained(self.output_dir)
            else:
                trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)

            print("\nTraining complete!")
            print(f"  Train examples: {len(train_dataset)}")
            if eval_dataset:
                print(f"  Val examples: {len(eval_dataset)}")
            print(f"  Output: {self.output_dir}")
        finally:
            if wandb_module is not None and wandb_module.run is not None:
                wandb_module.finish()

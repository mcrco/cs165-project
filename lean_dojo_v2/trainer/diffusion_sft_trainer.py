from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from lean_dojo_v2.database import DynamicDatabase
from lean_dojo_v2.diffusion import (
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_DIFFUSION_TEMPERATURE,
    DEFAULT_REMASKING,
    decode_until_stop,
    denoise_masked_sequence,
)
from lean_dojo_v2.lean_dojo.data_extraction.lean import LeanGitRepo
from lean_dojo_v2.utils import remove_marks


def _ensure_prepare_inputs_for_generation(model) -> None:
    """PEFT expects this method for CAUSAL_LM wrappers on some custom models."""
    if hasattr(model, "prepare_inputs_for_generation"):
        return

    def _prepare_inputs_for_generation(input_ids, **kwargs):
        model_inputs = {"input_ids": input_ids}
        model_inputs.update(kwargs)
        return model_inputs

    model.prepare_inputs_for_generation = _prepare_inputs_for_generation  # type: ignore[attr-defined]


def _normalize_optional_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _build_infilling_user_content(theorem_statement: str) -> str:
    chunks: List[str] = [
        "Implement a Lean 4 proof for the given formal statement.",
    ]
    if theorem_statement:
        chunks.append(f"Formal statement:\n{theorem_statement}")
    chunks.append("Output only Lean 4 proof code.")
    return "\n\n".join(chunks)


def _build_infilling_prompt_prefix(tokenizer, theorem_statement: str) -> Tuple[str, str]:
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


class DiffusionSFTDataset:
    """Builds next-tactic SFT examples for diffusion-style training.

    Each example is: [prompt tokens] + [tactic tokens] + [EOS] + [mask padding to gen_length].
    The generation region (from assistant_start) is always exactly gen_length tokens,
    matching the inference-time sequence shape.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024,
                 gen_length: int = 64):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.gen_length = gen_length

        self.mask_token_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        if self.mask_token_id is None or self.mask_token_id < 0:
            self.mask_token_id = 156895

        with open(data_path, encoding="utf-8") as f:
            self.json_data = json.load(f)
        self.data = self._process_data(self.json_data)

    def _make_prompt_prefix(self, goal_state: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Lean 4 tactic generator. Given a goal state, "
                    "output exactly ONE Lean tactic that advances or solves the goal.\n"
                    "Rules:\n"
                    "- Output only the tactic text; no prose or code fences.\n"
                    "- Single line only; no `by` blocks.\n"
                    "- Never use `sorry` or `admit`.\n"
                ),
            },
            {"role": "user", "content": goal_state},
        ]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def _process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        eos_id = self.tokenizer.eos_token_id

        for item in data:
            for tactic in item.get("traced_tactics", []):
                tactic_text = (
                    remove_marks(tactic.get("tactic", "")).splitlines()[0].strip()
                )
                if not tactic_text or tactic_text == "sorry":
                    continue

                goal_state = remove_marks(tactic["state_before"]).strip()
                prompt_prefix = self._make_prompt_prefix(goal_state)

                prompt_ids = self.tokenizer(
                    prompt_prefix,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]

                tactic_ids = self.tokenizer(
                    tactic_text,
                    add_special_tokens=False,
                )["input_ids"]

                tactic_plus_eos = tactic_ids + [eos_id]

                if len(tactic_plus_eos) > self.gen_length:
                    continue

                n_mask_pad = self.gen_length - len(tactic_plus_eos)
                gen_region = tactic_plus_eos + [self.mask_token_id] * n_mask_pad

                full_ids = prompt_ids + gen_region
                if len(full_ids) > self.max_length:
                    continue

                processed.append(
                    {
                        "input_ids": full_ids,
                        "assistant_start": len(prompt_ids),
                    }
                )

        return processed

    def to_hf(self) -> Dataset:
        return Dataset.from_list(self.data)


class DiffusionDataCollator:
    """
    Applies random token masking only on assistant (completion) tokens.
    The model learns to denoise masked target tactic tokens.
    """

    def __init__(
        self,
        tokenizer,
        mask_token_id: int,
        min_mask_ratio: float = 0.01,
        max_mask_ratio: float = 1.0,
    ):
        if not (0.0 < min_mask_ratio <= max_mask_ratio <= 1.0):
            raise ValueError("Mask ratios must satisfy 0 < min <= max <= 1")

        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_size = len(features)

        input_ids = torch.full(
            (batch_size, max_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, feature in enumerate(features):
            ids = torch.tensor(feature["input_ids"], dtype=torch.long)
            seq_len = ids.size(0)
            assistant_start = int(feature["assistant_start"])

            input_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1

            if assistant_start >= seq_len:
                continue

            # The generation region includes tactic + EOS + mask padding.
            # Positions that are already mask_token_id (padding) always get
            # masked and trained; for real tactic+EOS tokens, sample a ratio.
            real_positions = []
            pad_positions = []
            for pos in range(assistant_start, seq_len):
                if ids[pos].item() == self.mask_token_id:
                    pad_positions.append(pos)
                else:
                    real_positions.append(pos)

            # Always mask and train on the pad region so the model learns to
            # produce mask/pad tokens after the tactic ends.
            for pos in pad_positions:
                labels[i, pos] = ids[pos]

            if real_positions:
                real_positions = torch.tensor(real_positions, dtype=torch.long)
                mask_ratio = (
                    torch.empty(1).uniform_(self.min_mask_ratio, self.max_mask_ratio).item()
                )
                num_to_mask = max(1, int(round(real_positions.numel() * mask_ratio)))
                chosen = real_positions[
                    torch.randperm(real_positions.numel())[:num_to_mask]
                ]
                input_ids[i, chosen] = self.mask_token_id
                labels[i, chosen] = ids[chosen]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class MdlmTrainer(Trainer):
    """
    Only thing different between MDLM and ARLM is the compute_loss method
    that computes denoising loss only on masked positions. We can reuse
    everything else for training MDLMs as well.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        vocab_size = logits.shape[-1]
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


class DiffusionSFTTrainer:
    def __init__(
        self,
        model_name: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        output_dir: str = "outputs-diffusion-sft",
        epochs_per_repo: float = 1.0,
        batch_size: int = 1,
        lr: float = 2e-5,
        max_length: int = 1024,
        gen_length: int = 64,
        min_mask_ratio: float = 0.01,
        max_mask_ratio: float = 1.0,
        lora_config: Optional[LoraConfig] = None,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.gen_length = gen_length
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.lora_config = lora_config
        self.use_lora = lora_config is not None
        self.trust_remote_code = trust_remote_code

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=self.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=self.trust_remote_code,
        )

        if self.use_lora:
            self.model = self._apply_lora()

        self.mask_token_id = self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        if self.mask_token_id is None or self.mask_token_id < 0:
            print(f"<|mdm_mask|> token not found in tokenizeer for {model_name}.")
            self.mask_token_id = 156895  # LLaDA MoE default

        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs_per_repo,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            bf16=torch.cuda.is_available(),
            fp16=False,
            remove_unused_columns=False,
            report_to="none",
            save_strategy="epoch",
            logging_steps=10,
        )

        self.data_collator = DiffusionDataCollator(
            tokenizer=self.tokenizer,
            mask_token_id=self.mask_token_id,
            min_mask_ratio=min_mask_ratio,
            max_mask_ratio=max_mask_ratio,
        )

    def _apply_lora(self):
        if self.lora_config is None:
            raise ValueError("LoRA config is required when use_lora is True")
        _ensure_prepare_inputs_for_generation(self.model)
        model = get_peft_model(self.model, self.lora_config)
        model.print_trainable_parameters()
        return model

    def _build_trainer(self, train_dataset: Dataset) -> Trainer:
        return MdlmTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )

    def train(
        self,
        repos: List[LeanGitRepo],
        database: DynamicDatabase,
        data_path: Path,
    ):
        repos_to_process = []

        for repo in repos:
            repos_to_process.append(repo)
            database.export_merged_data(repos_to_process, data_path)

            train_file = os.path.join(data_path, "random", "train.json")
            train_dataset = DiffusionSFTDataset(
                data_path=train_file,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                gen_length=self.gen_length,
            ).to_hf()

            trainer = self._build_trainer(train_dataset)
            trainer.train()

            self.model = trainer.model

            if self.use_lora:
                self.model.save_pretrained(self.output_dir)
            else:
                trainer.save_model(self.output_dir)

        self.tokenizer.save_pretrained(self.output_dir)

    def sample_next_tactic(
        self,
        goal_state: str,
        *,
        num_return_sequences: int = 1,
        steps: int = DEFAULT_DIFFUSION_STEPS,
        temperature: float = DEFAULT_DIFFUSION_TEMPERATURE,
        remasking: str = DEFAULT_REMASKING,
    ) -> List[str]:
        """Iteratively denoise a masked generation span for next-tactic prediction."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Lean 4 tactic generator. Given a goal state, "
                    "output exactly ONE Lean tactic that advances or solves the goal.\n"
                    "Rules:\n"
                    "- Output only the tactic text; no prose or code fences.\n"
                    "- Single line only; no `by` blocks.\n"
                    "- Never use `sorry` or `admit`.\n"
                ),
            },
            {"role": "user", "content": goal_state},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids = encoded.input_ids.to(self.model.device)
        prompt_ids = prompt_ids.repeat(num_return_sequences, 1)

        x = torch.full(
            (num_return_sequences, prompt_ids.shape[1] + self.gen_length),
            self.mask_token_id,
            dtype=torch.long,
            device=self.model.device,
        )
        x[:, : prompt_ids.shape[1]] = prompt_ids
        attention_mask = torch.ones_like(x)

        sampled = denoise_masked_sequence(
            model=self.model,
            input_ids=x,
            attention_mask=attention_mask,
            mask_token_id=self.mask_token_id,
            steps=steps,
            temperature=temperature,
            remasking=remasking,
        )

        generated_only = sampled[:, prompt_ids.shape[1] :].tolist()
        return [
            decode_until_stop(self.tokenizer, seq, self.mask_token_id)
            for seq in generated_only
        ]


class InfillingMDMDataset:
    """Dataset for infilling-style training with MDM mask tokens embedded in proof context.

    Each example has:
    - input_ids: proof text with <HOLE> replaced by fixed-length <|mdm_mask|> span
    - labels: -100 everywhere except mask positions, which have true token IDs
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        mask_span_length: int = 64,
        hole_token: str = "<HOLE>",
        mask_token: str = "<|mdm_mask|>",
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_span_length = mask_span_length
        self.hole_token = hole_token
        self.mask_token = mask_token

        self.mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
        if self.mask_token_id is None or self.mask_token_id < 0:
            self.mask_token_id = 156895  # LLaDA MoE default

        with open(data_path, encoding="utf-8") as f:
            self.json_data = json.load(f)
        self.data = self._process_data(self.json_data)

    def _process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert infilling examples to MDM training format."""
        processed: List[Dict[str, Any]] = []
        eos_id = self.tokenizer.eos_token_id

        for item in data:
            infilling = item.get("infilling", {})
            proof_with_hole = infilling.get("proof_with_hole", "")
            target_tactic = infilling.get("target_tactic", "").strip()
            theorem_statement = _normalize_optional_text(item.get("theorem_statement", ""))

            if not proof_with_hole or not target_tactic or target_tactic == "sorry":
                continue

            # Find hole position
            hole_pos = proof_with_hole.find(self.hole_token)
            if hole_pos == -1:
                continue

            # Split into left/right context
            left_text = proof_with_hole[:hole_pos]
            right_text = proof_with_hole[hole_pos + len(self.hole_token):]

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

            # Tokenize left and right contexts
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

            # Check if combined length fits
            total_proof_len = len(left_ids) + self.mask_span_length + len(right_ids)
            if total_proof_len > available_for_proof:
                # Truncate right context more aggressively
                available_for_right = available_for_proof - len(left_ids) - self.mask_span_length
                if available_for_right < 0:
                    continue  # Left context too long, skip this example
                right_ids = right_ids[:available_for_right]
                total_proof_len = len(left_ids) + self.mask_span_length + len(right_ids)

            # Build assistant ids: left + mask_span + right
            mask_span = [self.mask_token_id] * self.mask_span_length
            assistant_ids = left_ids + mask_span + right_ids
            input_ids = prompt_ids + assistant_ids

            # Tokenize target tactic + EOS
            target_ids = self.tokenizer(
                target_tactic,
                add_special_tokens=False,
            )["input_ids"]
            target_plus_eos = target_ids + [eos_id]

            # Build labels: -100 everywhere, fill only mask span with target
            labels = [-100] * len(input_ids)
            assistant_start = len(prompt_ids)
            mask_start = assistant_start + len(left_ids)
            mask_end = mask_start + self.mask_span_length

            # Fill with target tactic tokens
            for i, tid in enumerate(target_plus_eos):
                if mask_start + i < mask_end:
                    labels[mask_start + i] = tid

            # Fill remaining mask positions with mask_token_id (model learns to keep as padding)
            for i in range(len(target_plus_eos), self.mask_span_length):
                if mask_start + i < mask_end:
                    labels[mask_start + i] = self.mask_token_id

            processed.append({
                "input_ids": input_ids,
                "labels": labels,
                "assistant_start": assistant_start,
                "mask_start": mask_start,
                "mask_end": mask_end,
                "full_name": item.get("full_name", ""),
                "theorem_statement": theorem_statement,
                "prompt_user": user_content,
                "proof_with_hole": proof_with_hole,
                "target_tactic": target_tactic,
                "state_before_hole": infilling.get("state_before_hole", ""),
            })

        return processed

    def to_hf(self) -> Dataset:
        """Convert to HuggingFace Dataset."""
        return Dataset.from_list(self.data)


class InfillingMDMCollator:
    """Simple padding collator for pre-masked infilling examples.

    No random masking - just pads input_ids, attention_mask, and labels.
    """

    def __init__(self, tokenizer, pad_token_id: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Pad and batch features."""
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
        wandb_module,
        tokenizer,
        mask_token_id: int,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        log_every_n_steps: int,
        num_samples_per_split: int,
        sampling_steps: int = DEFAULT_DIFFUSION_STEPS,
        sampling_temperature: float = DEFAULT_DIFFUSION_TEMPERATURE,
        sampling_remasking: str = DEFAULT_REMASKING,
        seed: int = 0,
    ):
        self.wandb = wandb_module
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.num_samples_per_split = max(1, int(num_samples_per_split))
        self.sampling_steps = max(1, int(sampling_steps))
        self.sampling_temperature = float(sampling_temperature)
        self.sampling_remasking = sampling_remasking
        self.seed = seed
        self._last_logged_step: Optional[int] = None

        self.train_indices = self._sample_indices(train_dataset, seed)
        self.val_indices = (
            self._sample_indices(eval_dataset, seed + 1) if eval_dataset is not None else []
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
        # Keep a persistent table per split so we can compare the same
        # examples across training steps/epochs in one view.
        self.train_history_table = self.wandb.Table(columns=self._qual_columns)
        self.val_history_table = (
            self.wandb.Table(columns=self._qual_columns) if eval_dataset is not None else None
        )

    def _sample_indices(self, dataset: Optional[Dataset], seed: int) -> List[int]:
        if dataset is None:
            return []
        n = len(dataset)
        if n == 0:
            return []
        k = min(self.num_samples_per_split, n)
        if k == n:
            return list(range(n))
        rng = random.Random(seed)
        return sorted(rng.sample(range(n), k))

    def _decode_predicted_tactic(self, token_ids: List[int]) -> str:
        return decode_until_stop(self.tokenizer, token_ids, self.mask_token_id)

    def _predict_tactic(self, model, row: Dict[str, Any]) -> str:
        device = next(model.parameters()).device
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        if "mask_start" in row and "mask_end" in row:
            mask_start = int(row["mask_start"])
            mask_end = int(row["mask_end"])
            supervised_positions = list(range(mask_start, min(mask_end, input_ids.shape[1])))
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
            return

        table = self.train_history_table if split == "train" else self.val_history_table
        if table is None:
            return

        for idx in indices:
            row = dataset[idx]
            full_name = row.get("full_name", "")
            theorem_statement = row.get("theorem_statement", "")
            theorem = full_name.strip()
            if theorem_statement:
                theorem = f"{theorem}: {theorem_statement}" if theorem else theorem_statement

            prompt = row.get("prompt_user", "")
            proof_with_hole = row.get("proof_with_hole", "")
            if proof_with_hole:
                prompt = f"{prompt}\n\nAssistant proof template:\n{proof_with_hole}" if prompt else proof_with_hole
            expected_tactic = row.get("target_tactic", "")
            predicted_tactic = self._predict_tactic(model, row)

            table.add_data(
                split,
                int(idx),
                theorem,
                prompt,
                expected_tactic,
                predicted_tactic,
                step,
                float(epoch) if epoch is not None else None,
            )

        self.wandb.log({f"{split}_qualitative_samples": table}, step=step)

    def _maybe_log_qualitative(self, *, model, step: int, epoch: Optional[float]):
        if self._last_logged_step == step:
            return

        self._last_logged_step = step
        was_training = model.training
        model.eval()
        try:
            self._log_split(
                model=model,
                split="train",
                dataset=self.train_dataset,
                indices=self.train_indices,
                step=step,
                epoch=epoch,
            )
            self._log_split(
                model=model,
                split="val",
                dataset=self.eval_dataset,
                indices=self.val_indices,
                step=step,
                epoch=epoch,
            )
        finally:
            if was_training:
                model.train()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step <= 0:
            return
        if state.global_step % self.log_every_n_steps != 0:
            return
        try:
            self._maybe_log_qualitative(
                model=model,
                step=state.global_step,
                epoch=float(state.epoch) if state.epoch is not None else None,
            )
        except Exception as exc:
            print(f"[wandb] qualitative logging failed at step {state.global_step}: {exc}")

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        try:
            self._maybe_log_qualitative(
                model=model,
                step=state.global_step,
                epoch=float(state.epoch) if state.epoch is not None else None,
            )
        except Exception as exc:
            print(f"[wandb] qualitative eval logging failed at step {state.global_step}: {exc}")


class InfillingDiffusionTrainer:
    """Trainer for infilling-style diffusion training on Lean proofs.

    Uses fixed-length mask spans embedded directly in the proof context.
    """

    def __init__(
        self,
        model_name: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        train_path: str = "",
        val_path: Optional[str] = None,
        output_dir: str = "outputs/infilling-mdm",
        epochs: float = 3.0,
        batch_size: int = 1,
        lr: float = 2e-5,
        max_length: int = 1024,
        mask_span_length: int = 64,
        lora_config: Optional[LoraConfig] = None,
        bf16: Optional[bool] = None,
        logging_steps: int = 10,
        save_strategy: str = "epoch",
        wandb_project: Optional[str] = "CS165Proj",
        wandb_run_name: Optional[str] = None,
        qual_log_every_n_steps: int = 200,
        qual_num_samples_per_split: int = 64,
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
        self.mask_span_length = mask_span_length
        self.lora_config = lora_config
        self.use_lora = lora_config is not None
        self.logging_steps = logging_steps
        self.save_strategy = save_strategy
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.qual_log_every_n_steps = qual_log_every_n_steps
        self.qual_num_samples_per_split = qual_num_samples_per_split
        self.wandb_enabled = bool(wandb_project)
        self.trust_remote_code = trust_remote_code

        # Auto-detect bf16 if not specified
        if bf16 is None:
            bf16 = torch.cuda.is_available()
        self.bf16 = bf16

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=self.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.bf16 else torch.float32,
            trust_remote_code=self.trust_remote_code,
        )

        if self.use_lora:
            self.model = self._apply_lora()

        # Get mask token ID
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        if self.mask_token_id is None or self.mask_token_id < 0:
            print(f"<|mdm_mask|> token not found in tokenizer for {model_name}.")
            self.mask_token_id = 156895  # LLaDA MoE default

        # Setup training arguments
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
            report_to=["wandb"] if self.wandb_enabled else "none",
            save_strategy=save_strategy,
            eval_strategy=eval_strategy,
            logging_steps=logging_steps,
            load_best_model_at_end=True if val_path else False,
            metric_for_best_model="eval_loss" if val_path else None,
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

    def _load_dataset(self, data_path: str) -> Dataset:
        """Load and process dataset."""
        dataset = InfillingMDMDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            mask_span_length=self.mask_span_length,
        )
        return dataset.to_hf()

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
        right_text = proof_with_hole[hole_pos + len(hole_token):]
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
            available_for_right = available_for_proof - len(left_ids) - self.mask_span_length
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
        steps: int = DEFAULT_DIFFUSION_STEPS,
        temperature: float = DEFAULT_DIFFUSION_TEMPERATURE,
        remasking: str = DEFAULT_REMASKING,
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

        x = torch.tensor(input_ids, dtype=torch.long, device=self.model.device).unsqueeze(0)
        x = x.repeat(num_return_sequences, 1)
        attention_mask = torch.ones_like(x)

        sampled = denoise_masked_sequence(
            model=self.model,
            input_ids=x,
            attention_mask=attention_mask,
            mask_token_id=self.mask_token_id,
            steps=steps,
            temperature=temperature,
            remasking=remasking,
        )

        results: List[str] = []
        for seq in sampled.tolist():
            predicted_span = seq[mask_start:mask_end]
            results.append(decode_until_stop(self.tokenizer, predicted_span, self.mask_token_id))
        return results

    def train(self):
        """Run training."""
        wandb_module = None
        if self.wandb_enabled:
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
                    "mask_span_length": self.mask_span_length,
                    "qual_log_every_n_steps": self.qual_log_every_n_steps,
                    "qual_num_samples_per_split": self.qual_num_samples_per_split,
                    "use_lora": self.use_lora,
                },
            )
            wandb_module = wandb

        # Load datasets
        print(f"Loading training data from {self.train_path}")
        train_dataset = self._load_dataset(self.train_path)
        print(f"Loaded {len(train_dataset)} training examples")

        eval_dataset = None
        if self.val_path:
            print(f"Loading validation data from {self.val_path}")
            eval_dataset = self._load_dataset(self.val_path)
            print(f"Loaded {len(eval_dataset)} validation examples")

        # Create collator
        data_collator = InfillingMDMCollator(
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        callbacks = []
        if self.wandb_enabled and wandb_module is not None:
            callbacks.append(
                WandbQualitativeCallback(
                    wandb_module=wandb_module,
                    tokenizer=self.tokenizer,
                    mask_token_id=self.mask_token_id,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    log_every_n_steps=self.qual_log_every_n_steps,
                    num_samples_per_split=self.qual_num_samples_per_split,
                )
            )

        # Create trainer
        trainer = MdlmTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )
        try:
            # Train
            print("Starting training...")
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

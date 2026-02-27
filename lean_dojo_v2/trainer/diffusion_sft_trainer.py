from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from lean_dojo_v2.database import DynamicDatabase
from lean_dojo_v2.lean_dojo.data_extraction.lean import LeanGitRepo
from lean_dojo_v2.utils import remove_marks


class DiffusionSFTDataset:
    """Builds next-tactic SFT examples for diffusion-style training."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        for item in data:
            for tactic in item.get("traced_tactics", []):
                tactic_text = (
                    remove_marks(tactic.get("tactic", "")).splitlines()[0].strip()
                )
                if not tactic_text or tactic_text == "sorry":
                    continue

                goal_state = remove_marks(tactic["state_before"]).strip()
                prompt_prefix = self._make_prompt_prefix(goal_state)
                full_text = prompt_prefix + tactic_text

                prompt_ids = self.tokenizer(
                    prompt_prefix,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]
                full_ids = self.tokenizer(
                    full_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]

                assistant_start = min(len(prompt_ids), len(full_ids))
                if assistant_start >= len(full_ids):
                    continue

                processed.append(
                    {
                        "input_ids": full_ids,
                        "assistant_start": assistant_start,
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
        min_mask_ratio: float = 0.15,
        max_mask_ratio: float = 0.60,
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

            candidate_positions = torch.arange(
                assistant_start, seq_len, dtype=torch.long
            )
            if candidate_positions.numel() == 0:
                continue

            mask_ratio = (
                torch.empty(1).uniform_(self.min_mask_ratio, self.max_mask_ratio).item()
            )
            num_to_mask = max(1, int(round(candidate_positions.numel() * mask_ratio)))
            chosen = candidate_positions[
                torch.randperm(candidate_positions.numel())[:num_to_mask]
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
        min_mask_ratio: float = 0.15,
        max_mask_ratio: float = 0.60,
        lora_config: Optional[LoraConfig] = None,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.lora_config = lora_config
        self.use_lora = lora_config is not None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

        if self.use_lora:
            self.model = self._apply_lora()

        self.mask_token_id = self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        if self.mask_token_id is None or self.mask_token_id < 0:
            self.mask_token_id = 156895

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
            ).to_hf()

            trainer = self._build_trainer(train_dataset)
            trainer.train()

            self.model = trainer.model

            if self.use_lora:
                self.model.save_pretrained(self.output_dir)
            else:
                trainer.save_model(self.output_dir)

        self.tokenizer.save_pretrained(self.output_dir)

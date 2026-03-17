from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import ProgressCallback

from lean_dojo_v2.database import DynamicDatabase
from lean_dojo_v2.diffusion import (
    DEFAULT_DIFFUSION_MODEL_NAME,
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_DIFFUSION_TEMPERATURE,
    DEFAULT_REMASKING,
    create_diffusion_training_objective,
    decode_until_stop,
    denoise_masked_sequence,
    is_dream_model,
    load_diffusion_components,
    prepare_diffusion_forward_kwargs,
    resolve_mask_token_id,
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

        self.mask_token_id = resolve_mask_token_id(tokenizer)

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

    def __init__(self, *args, training_objective=None, **kwargs):
        self.training_objective = training_objective
        super().__init__(*args, **kwargs)

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None and is_dream_model(self.model):
            inputs["attention_mask"] = attention_mask.bool()
        return inputs

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        if self.training_objective is None:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, return_outputs=False)

        if isinstance(loss, tuple):
            loss = loss[0]
        loss = loss.detach().mean()
        return (loss, None, None)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.training_objective is not None:
            return self.training_objective.compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
            )
        labels = inputs.pop("labels")
        outputs = model(
            **prepare_diffusion_forward_kwargs(
                model=model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
        )
        logits = outputs.logits
        vocab_size = logits.shape[-1]
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


class QuietProgressCallback(ProgressCallback):
    """Keep tqdm progress bars without printing per-step metric dicts."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        return


class DiffusionSFTTrainer:
    def __init__(
        self,
        model_name: str = DEFAULT_DIFFUSION_MODEL_NAME,
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

        self.components = load_diffusion_components(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=self.trust_remote_code,
            for_training=True,
        )
        self.family = self.components.family
        self.sampling_config = self.components.sampling
        self.tokenizer = self.components.tokenizer
        self.model = self.components.model

        if self.use_lora:
            self.model = self._apply_lora()

        self.mask_token_id = self.components.mask_token_id

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

        self.training_objective = create_diffusion_training_objective(
            family=self.family,
            mode="sft",
            tokenizer=self.tokenizer,
            mask_token_id=self.mask_token_id,
            min_mask_ratio=min_mask_ratio,
            max_mask_ratio=max_mask_ratio,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.data_collator = self.training_objective

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
            training_objective=self.training_objective,
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
        steps: Optional[int] = None,
        temperature: Optional[float] = None,
        remasking: Optional[str] = None,
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
        effective_steps = self.sampling_config.steps if steps is None else max(1, int(steps))
        effective_temperature = (
            self.sampling_config.temperature
            if temperature is None
            else float(temperature)
        )
        effective_remasking = self.sampling_config.remasking if remasking is None else remasking

        sampled = denoise_masked_sequence(
            model=self.model,
            input_ids=x,
            attention_mask=attention_mask,
            mask_token_id=self.mask_token_id,
            steps=effective_steps,
            temperature=effective_temperature,
            remasking=effective_remasking,
        )

        generated_only = sampled[:, prompt_ids.shape[1] :].tolist()
        return [
            decode_until_stop(self.tokenizer, seq, self.mask_token_id)
            for seq in generated_only
        ]



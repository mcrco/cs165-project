from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
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


def _strip_thinking_segments(text: str) -> str:
    """Best-effort removal of explicit thinking tags from model outputs."""
    out = str(text)
    while True:
        start = out.find("<think>")
        if start == -1:
            break
        end = out.find("</think>", start + len("<think>"))
        if end == -1:
            out = out[:start]
            break
        out = out[:start] + out[end + len("</think>") :]
    return out.strip()


def _build_ar_infilling_prompt_prefix(
    tokenizer,
    theorem_statement: str,
    proof_with_hole: str,
    *,
    non_thinking_mode: bool = True,
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
    if non_thinking_mode:
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


class InfillingARDataset:
    """Dataset for autoregressive infilling on APRIL-style infilling examples."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        *,
        max_length: int = 1024,
        hole_token: str = "<HOLE>",
        non_thinking_mode: bool = True,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hole_token = hole_token
        self.non_thinking_mode = non_thinking_mode

        with open(data_path, encoding="utf-8") as f:
            self.json_data = json.load(f)
        self.data = self._process_data(self.json_data)

    def _process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must define eos_token_id for AR infilling training.")

        for item in data:
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
                non_thinking_mode=self.non_thinking_mode,
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


class InfillingAutoregressiveTrainer:
    """Trainer for autoregressive infilling on Lean proof holes."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-7B",
        train_path: str = "",
        val_path: Optional[str] = None,
        output_dir: str = "outputs/infilling-ar",
        epochs: float = 3.0,
        batch_size: int = 8,
        lr: float = 2e-5,
        max_length: int = 1024,
        max_new_tokens: int = 128,
        non_thinking_mode: bool = True,
        lora_config: Optional[LoraConfig] = None,
        bf16: Optional[bool] = None,
        logging_steps: int = 10,
        save_strategy: str = "epoch",
        wandb_project: Optional[str] = "infilling-ar",
        wandb_run_name: Optional[str] = None,
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
        self.non_thinking_mode = non_thinking_mode
        self.lora_config = lora_config
        self.use_lora = lora_config is not None
        self.logging_steps = logging_steps
        self.save_strategy = save_strategy
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_enabled = bool(wandb_project)
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
            report_to=["wandb"] if self.wandb_enabled else "none",
            save_strategy=save_strategy,
            eval_strategy=eval_strategy,
            logging_steps=logging_steps,
            load_best_model_at_end=True if val_path else False,
            metric_for_best_model="eval_loss" if val_path else None,
            run_name=wandb_run_name,
        )

    def _apply_lora(self):
        if self.lora_config is None:
            raise ValueError("LoRA config is required when use_lora is True")
        _ensure_prepare_inputs_for_generation(self.model)
        model = get_peft_model(self.model, self.lora_config)
        model.print_trainable_parameters()
        return model

    def _load_dataset(self, data_path: str) -> Dataset:
        dataset = InfillingARDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            non_thinking_mode=self.non_thinking_mode,
        )
        return dataset.to_hf()

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
            non_thinking_mode=self.non_thinking_mode,
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
            decoded = _strip_thinking_segments(decoded)
            if decoded:
                decoded = decoded.splitlines()[0].strip()
            results.append(decoded)
        return results

    def train(self):
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
                    "max_new_tokens": self.max_new_tokens,
                    "non_thinking_mode": self.non_thinking_mode,
                    "use_lora": self.use_lora,
                },
            )
            wandb_module = wandb

        print(f"Loading training data from {self.train_path}")
        train_dataset = self._load_dataset(self.train_path)
        print(f"Loaded {len(train_dataset)} training examples")

        eval_dataset = None
        if self.val_path:
            print(f"Loading validation data from {self.val_path}")
            eval_dataset = self._load_dataset(self.val_path)
            print(f"Loaded {len(eval_dataset)} validation examples")

        data_collator = InfillingARCollator(
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(QuietProgressCallback())

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

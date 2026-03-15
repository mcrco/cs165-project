from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from lean_dojo_v2.diffusion import (
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_DIFFUSION_TEMPERATURE,
    DEFAULT_REMASKING,
    decode_until_stop,
    denoise_masked_sequence,
)
from .diffusion_sft_trainer import (
    InfillingMDMCollator,
    InfillingMDMDataset,
    MdlmTrainer,
    QuietProgressCallback,
    WandbQualitativeCallback,
    _build_infilling_prompt_prefix,
    _ensure_prepare_inputs_for_generation,
    _normalize_optional_text,
)


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
        batch_size: int = 8,
        lr: float = 2e-5,
        max_length: int = 1024,
        mask_span_length: int = 64,
        lora_config: Optional[LoraConfig] = None,
        bf16: Optional[bool] = None,
        logging_steps: int = 10,
        save_strategy: str = "epoch",
        wandb_project: Optional[str] = "infilling",
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
        # Keep tqdm with dummy quiet callback but suppress raw metric-dict prints from HF callbacks.
        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(QuietProgressCallback())
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

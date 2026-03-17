"""Diffusion LLM-based prover."""

import random
from typing import Optional

import torch
from pantograph.expr import GoalState, Tactic

from lean_dojo_v2.database.models.theorems import Theorem
from lean_dojo_v2.diffusion import (
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_DIFFUSION_TEMPERATURE,
    DEFAULT_LLADA_MODEL_NAME,
    DEFAULT_REMASKING,
    decode_until_stop,
    denoise_masked_sequence,
    generate_llada_blockwise,
    load_diffusion_components,
)
from lean_dojo_v2.prover.base_prover import BaseProver


class DiffusionProver(BaseProver):
    """Loads a diffusion model for Lean theorem proving."""

    def __init__(
        self,
        ckpt_path: str = DEFAULT_LLADA_MODEL_NAME,
        use_lora: bool = False,
        device: str = "auto",
    ):
        super().__init__()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.components = load_diffusion_components(
            ckpt_path,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device=self.device,
            trust_remote_code=True,
            use_lora_adapter=use_lora,
        )
        self.family = self.components.family
        self.sampling_config = self.components.sampling
        self.tokenizer = self.components.tokenizer
        self.model = self.components.model
        self.mask_token_id = self.components.mask_token_id

        self.model.eval()

    def next_tactic(
        self,
        state: GoalState,
        goal_id: int,
    ) -> Optional[Tactic]:
        """Generate the next tactic using the loaded HuggingFace model."""
        if not hasattr(self, "theorem") or self.theorem is None:
            return None

        prompt = self._build_chat_prompt(
            system_prompt=(
                "You are a Lean 4 tactic generator. Given a goal state, "
                "output exactly ONE Lean tactic that advances or solves the goal.\n"
                "Rules:\n"
                "- Output only the tactic text; no prose, quotes, or code fences.\n"
                "- Single line only; no `by` blocks.\n"
                "- Never use `sorry` or `admit`.\n"
            ),
            user_content=str(state),
        )

        with torch.no_grad():
            generated_texts = self._diffusion_sample(
                prompt=prompt,
                gen_length=64,
                num_return_sequences=5,
                steps=self.sampling_config.steps,
                block_length=64,
                temperature=self.sampling_config.temperature,
                cfg_scale=0.0,
                remasking=self.sampling_config.remasking,
            )

        tactics = []
        for text in generated_texts:
            tactic_part = text.strip().split("\n")[0].split("<;>")[0].strip()
            if tactic_part and tactic_part != "sorry":
                tactics.append(tactic_part)

        if not tactics:
            return None

        return tactics[0]

    def generate_whole_proof(self, theorem: Theorem) -> str:
        self.theorem = theorem

        prompt = self._build_chat_prompt(
            system_prompt=(
                "Given a theorem statement, output the complete proof of the theorem "
                "in Lean 4 code.\n"
                "Only output the proof, no explanation, no comments, no theorem, "
                "nothing else."
            ),
            user_content=str(theorem),
        )

        with torch.no_grad():
            proofs = self._diffusion_sample(
                prompt=prompt,
                gen_length=512,
                num_return_sequences=1,
                steps=self.sampling_config.steps,
                block_length=128,
                temperature=self.sampling_config.temperature,
                cfg_scale=0.0,
                remasking=self.sampling_config.remasking,
            )

        if not proofs:
            return ""

        return proofs[0].replace("<;> ", "").strip()

    def _build_chat_prompt(self, system_prompt: str, user_content: str) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def _diffusion_sample(
        self,
        prompt: str,
        gen_length: int,
        num_return_sequences: int,
        steps: int,
        block_length: int,
        temperature: float,
        cfg_scale: float,
        remasking: str,
    ) -> list[str]:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids = encoded.input_ids.to(self.device).repeat(num_return_sequences, 1)
        if self.family == "llada":
            sampled_ids = generate_llada_blockwise(
                model=self.model,
                prompt=prompt_ids,
                attention_mask=encoded.attention_mask.to(self.device).repeat(
                    num_return_sequences, 1
                ),
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=self.mask_token_id,
            )
            generated_only = sampled_ids[:, prompt_ids.shape[1] :]
        else:
            x = torch.full(
                (num_return_sequences, prompt_ids.shape[1] + gen_length),
                self.mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
            x[:, : prompt_ids.shape[1]] = prompt_ids
            attention_mask = torch.ones_like(x)
            sampled_ids = denoise_masked_sequence(
                model=self.model,
                input_ids=x,
                attention_mask=attention_mask,
                mask_token_id=self.mask_token_id,
                steps=steps,
                temperature=temperature,
                remasking=remasking,
            )
            generated_only = sampled_ids[:, prompt_ids.shape[1] :]

        return [
            decode_until_stop(self.tokenizer, seq.tolist(), self.mask_token_id)
            for seq in generated_only
        ]

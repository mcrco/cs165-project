"""Diffusion LLM-based prover."""

import random
from typing import Optional

import torch
from pantograph.expr import GoalState, Tactic
from peft import AutoPeftModelForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lean_dojo_v2.database.models.theorems import Theorem
from lean_dojo_v2.diffusion import (
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_DIFFUSION_TEMPERATURE,
    DEFAULT_REMASKING,
    decode_until_stop,
    generate_llada_blockwise,
)
from lean_dojo_v2.prover.base_prover import BaseProver


def _sanitize_rope_scaling(cfg):
    rope = getattr(cfg, "rope_scaling", None)
    if not isinstance(rope, dict):
        return cfg
    rope = dict(rope)
    for key in ("factor", "beta_fast", "beta_slow"):
        if key in rope and rope[key] is not None:
            try:
                rope[key] = float(rope[key])
            except (TypeError, ValueError):
                pass
    cfg.rope_scaling = rope
    return cfg


class DiffusionProver(BaseProver):
    """Loads a diffusion model for Lean theorem proving."""

    def __init__(
        self,
        ckpt_path: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        use_lora: bool = False,
        device: str = "auto",
    ):
        super().__init__()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        trust_remote_code = ckpt_path == "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
        config = _sanitize_rope_scaling(
            AutoConfig.from_pretrained(ckpt_path, trust_remote_code=trust_remote_code)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path, trust_remote_code=trust_remote_code
        )

        if use_lora:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                ckpt_path, config=config, trust_remote_code=trust_remote_code
            ).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                ckpt_path, config=config, trust_remote_code=trust_remote_code
            ).to(self.device)
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        if self.mask_token_id is None or self.mask_token_id < 0:
            self.mask_token_id = 156895

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
                steps=DEFAULT_DIFFUSION_STEPS,
                block_length=64,
                temperature=DEFAULT_DIFFUSION_TEMPERATURE,
                cfg_scale=0.0,
                remasking=DEFAULT_REMASKING,
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
                steps=DEFAULT_DIFFUSION_STEPS,
                block_length=128,
                temperature=DEFAULT_DIFFUSION_TEMPERATURE,
                cfg_scale=0.0,
                remasking=DEFAULT_REMASKING,
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

        sampled_ids = generate_llada_blockwise(
            model=self.model,
            prompt=prompt_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=self.mask_token_id,
        )
        generated_only = sampled_ids[:, prompt_ids.shape[1] :]

        return [
            decode_until_stop(self.tokenizer, seq.tolist(), self.mask_token_id)
            for seq in generated_only
        ]

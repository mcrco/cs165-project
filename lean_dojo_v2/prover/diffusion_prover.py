"""Diffusion LLM-based prover."""

import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from pantograph.expr import GoalState, Tactic
from transformers import AutoModelForCausalLM, AutoTokenizer

from lean_dojo_v2.database.models.theorems import Theorem
from lean_dojo_v2.prover.base_prover import BaseProver


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

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, 
            trust_remote_code=ckpt_path == "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
        ).to(self.device)
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        if self.mask_token_id is None or self.mask_token_id < 0:
            # LLaDA mask token id (https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct)
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
                steps=128,
                block_length=32,
                temperature=0.7,
                cfg_scale=0.0,
                remasking="low_confidence",
            )

        tactics = []
        for text in generated_texts:
            tactic_part = text.strip().split("\n")[0].split("<;>")[0].strip()
            if tactic_part and tactic_part != "sorry":
                tactics.append(tactic_part)

        if not tactics:
            return None

        return random.choice(tactics)

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
                steps=128,
                block_length=128,
                temperature=0.7,
                cfg_scale=0.0,
                remasking="low_confidence",
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

        sampled_ids = _generate(
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
        return self.tokenizer.batch_decode(generated_only, skip_special_tokens=True)


# Helpers for diffusion language models
# Taken from https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct.


def _add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


def _generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.7,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=156895,
):
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)

    batch_size = prompt.shape[0]
    x = torch.full(
        (batch_size, prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # Since this is from LLaDA MoE, which does block diffusion, we loop over
    # blocks. However, to leverage bidirectionality, intuitively we should keep
    # the entire generation to one block.
    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1] + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(
                logits_with_noise, dim=-1
            )  # b, l (proposed tokens for each index)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # b, l
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # prevent future blocks from being updated.
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            # prevent already unmasked tokens from being updated.
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # only unmask top num_transfer_tokens per block per batch
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if k <= 0:
                    continue
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

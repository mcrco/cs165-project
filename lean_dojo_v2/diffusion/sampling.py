"""Shared diffusion sampling utilities.

This module centralizes the sampling loops used across trainer/prover code.
The block-wise generator is adapted from the official LLaDA inference code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_DIFFUSION_STEPS = 8
DEFAULT_DIFFUSION_TEMPERATURE = 0.0
DEFAULT_REMASKING = "low_confidence"
DEFAULT_LLADA_MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
REFERENCE_LLADA_MOE_7B_A1B_STEPS = 128
REFERENCE_LLADA_MOE_7B_A1B_BLOCK_LENGTH = 32
REFERENCE_LLADA_8B_STEPS = 128
REFERENCE_LLADA_8B_BLOCK_LENGTH = 32


@dataclass(frozen=True)
class DiffusionSamplingConfig:
    steps: int = DEFAULT_DIFFUSION_STEPS
    temperature: float = DEFAULT_DIFFUSION_TEMPERATURE
    remasking: str = DEFAULT_REMASKING
    block_length: int = 128
    cfg_scale: float = 0.0


def get_diffusion_sampling_config(
    model_name: Optional[str],
    *,
    mode: str = "infilling",
) -> DiffusionSamplingConfig:
    """Return model-aware sampling defaults.

    The Hugging Face `inclusionAI/LLaDA-MoE-7B-A1B-*` and
    `GSAI-ML/LLaDA-8B-*` reference samplers both use 128 denoising steps,
    low-confidence remasking, and block length 32 for blockwise generation.
    We reuse those defaults when either model family is selected.
    """
    normalized = (model_name or "").strip().lower()
    if mode not in {"infilling", "blockwise"}:
        raise ValueError(f"Unsupported diffusion sampling mode: {mode}")

    if normalized.startswith("inclusionai/llada-moe-7b-a1b"):
        return DiffusionSamplingConfig(
            steps=REFERENCE_LLADA_MOE_7B_A1B_STEPS,
            temperature=0.0,
            remasking="low_confidence",
            block_length=REFERENCE_LLADA_MOE_7B_A1B_BLOCK_LENGTH,
            cfg_scale=0.0,
        )

    if normalized.startswith("gsai-ml/llada-8b"):
        return DiffusionSamplingConfig(
            steps=REFERENCE_LLADA_8B_STEPS,
            temperature=0.0,
            remasking="low_confidence",
            block_length=REFERENCE_LLADA_8B_BLOCK_LENGTH,
            cfg_scale=0.0,
        )

    return DiffusionSamplingConfig()


def resolve_mask_token_id(tokenizer, fallback_id: int = 156895) -> int:
    """Resolve the diffusion mask token id across tokenizer variants."""
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is not None and int(mask_token_id) >= 0:
        return int(mask_token_id)

    candidates = []
    mask_token = getattr(tokenizer, "mask_token", None)
    if isinstance(mask_token, str) and mask_token:
        candidates.append(mask_token)
    candidates.extend(["<|mask|>", "<|mdm_mask|>"])

    seen = set()
    for token in candidates:
        if token in seen:
            continue
        seen.add(token)
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id >= 0:
            return int(token_id)

    return int(fallback_id)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise used by diffusion token selection."""
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def decode_until_stop(tokenizer, token_ids: list[int], mask_token_id: int) -> str:
    """Decode until hitting EOS, mask token, or pad token."""
    cleaned: list[int] = []
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    for token_id in token_ids:
        if eos_id is not None and token_id == eos_id:
            break
        if token_id == mask_token_id:
            break
        if pad_id is not None and token_id == pad_id:
            break
        cleaned.append(token_id)
    if not cleaned:
        return ""
    return tokenizer.decode(cleaned, skip_special_tokens=True).strip()


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def generate_llada_blockwise(
    *,
    model,
    prompt: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    steps: int = DEFAULT_DIFFUSION_STEPS,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = DEFAULT_DIFFUSION_TEMPERATURE,
    cfg_scale: float = 0.0,
    remasking: str = DEFAULT_REMASKING,
    mask_id: int = 156895,
) -> torch.Tensor:
    """Block-wise LLaDA diffusion generation (ground-truth sampler).

    Adapted from the LLaDA-MoE reference sampling loop.
    """
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
    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, gen_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=-1,
        )
    prompt_index = x != mask_id

    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError("steps must be divisible by number of blocks")
    steps = steps // num_blocks

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
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                    logits = model(x_, attention_mask=attention_mask_).logits
                else:
                    logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if k <= 0:
                    continue
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    return x


def denoise_masked_sequence(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    mask_token_id: int,
    steps: int = DEFAULT_DIFFUSION_STEPS,
    temperature: float = DEFAULT_DIFFUSION_TEMPERATURE,
    remasking: str = DEFAULT_REMASKING,
) -> torch.Tensor:
    """Iteratively denoise arbitrary mask positions inside a sequence.

    Unlike `generate_llada_blockwise`, this function supports masked spans in
    the middle of context (used by infilling).
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    x = input_ids.clone()
    if attention_mask is None:
        attention_mask = torch.ones_like(x)

    steps = max(1, int(steps))

    for step_idx in range(steps):
        mask_index = x.eq(mask_token_id)
        remaining_masks = mask_index.sum(dim=1)
        if int(remaining_masks.max().item()) == 0:
            break

        with torch.no_grad():
            outputs = model(input_ids=x, attention_mask=attention_mask)
            logits = outputs.logits

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        proposed_tokens = torch.argmax(logits_with_noise, dim=-1)
        proposed_tokens = torch.where(mask_index, proposed_tokens, x)

        if remasking == "low_confidence":
            probs = F.softmax(logits, dim=-1)
            confidence = torch.gather(
                probs, dim=-1, index=proposed_tokens.unsqueeze(-1)
            ).squeeze(-1)
        elif remasking == "random":
            confidence = torch.rand(x.shape, device=x.device, dtype=torch.float32)
        else:
            raise NotImplementedError(remasking)

        confidence = torch.where(
            mask_index,
            confidence,
            torch.full_like(confidence, float("-inf")),
        )

        remaining_steps = steps - step_idx
        num_to_unmask = (remaining_masks + remaining_steps - 1) // remaining_steps
        transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)
        for b in range(x.shape[0]):
            k = int(num_to_unmask[b].item())
            if k <= 0:
                continue
            available = int(mask_index[b].sum().item())
            if available <= 0:
                continue
            k = min(k, available)
            _, selected = torch.topk(confidence[b], k=k)
            transfer_index[b, selected] = True

        x[transfer_index] = proposed_tokens[transfer_index]

    return x

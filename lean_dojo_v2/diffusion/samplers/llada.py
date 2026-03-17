"""LLaDA-family diffusion sampling helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..families import (
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_DIFFUSION_TEMPERATURE,
    DEFAULT_REMASKING,
)
from ..model_adapter import forward_diffusion_logits


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise used by diffusion token selection."""
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
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
    """Block-wise LLaDA diffusion generation."""
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
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                attention_mask_ = (
                    torch.cat([attention_mask, attention_mask], dim=0)
                    if attention_mask is not None
                    else None
                )
                logits = forward_diffusion_logits(
                    model=model,
                    input_ids=x_,
                    attention_mask=attention_mask_,
                )
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = forward_diffusion_logits(
                    model=model,
                    input_ids=x,
                    attention_mask=attention_mask,
                )

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


def denoise_llada_masked_sequence(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    mask_token_id: int,
    steps: int = DEFAULT_DIFFUSION_STEPS,
    temperature: float = DEFAULT_DIFFUSION_TEMPERATURE,
    remasking: str = DEFAULT_REMASKING,
) -> torch.Tensor:
    """Iteratively denoise arbitrary mask positions inside a sequence."""
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
            logits = forward_diffusion_logits(
                model=model,
                input_ids=x,
                attention_mask=attention_mask,
            )

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

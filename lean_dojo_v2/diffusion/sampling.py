"""Shared diffusion sampling facade.

This module preserves the historical public API while delegating the actual
implementations to family-specific modules.
"""

from __future__ import annotations

from typing import Optional

import torch

from .families import (
    DEFAULT_DIFFUSION_MODEL_NAME,
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_DIFFUSION_TEMPERATURE,
    DEFAULT_LLADA_MODEL_NAME,
    DEFAULT_REMASKING,
    DiffusionSamplingConfig,
    get_diffusion_sampling_config,
    is_dream_model_name,
)
from .model_adapter import (
    forward_diffusion_logits,
    is_dream_model,
    prepare_diffusion_forward_kwargs,
)
from .samplers.dream import denoise_dream_masked_sequence
from .samplers.llada import (
    add_gumbel_noise,
    denoise_llada_masked_sequence,
    generate_llada_blockwise,
    get_num_transfer_tokens,
)
from .token_utils import decode_until_stop, resolve_mask_token_id


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
    """Iteratively denoise arbitrary mask positions inside a sequence."""
    if remasking.startswith("dream_"):
        return denoise_dream_masked_sequence(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_token_id=mask_token_id,
            steps=steps,
            temperature=temperature,
            remasking=remasking,
        )
    return denoise_llada_masked_sequence(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        mask_token_id=mask_token_id,
        steps=steps,
        temperature=temperature,
        remasking=remasking,
    )

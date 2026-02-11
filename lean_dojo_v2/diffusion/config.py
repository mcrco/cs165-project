"""Configuration dataclasses for diffusion language model inference."""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class DiffusionConfig:
    """Configuration for the discrete masked-diffusion sampler.

    This config is tailored for LLaDA-style absorbing-state diffusion models
    that iteratively denoise masked tokens via Gumbel-noise sampling.

    Attributes:
        model_name: HuggingFace model identifier (e.g. 'inclusionAI/LLaDA-MoE-7B-A1B-Instruct').
        mask_id: Token id used as the mask/absorbing token. Default 156895 for LLaDA.
        steps: Number of denoising steps per block.
        gen_length: Total number of tokens to generate.
        block_length: Block size for semi-autoregressive generation (gen_length must be divisible by this).
        temperature: Gumbel noise temperature (0 = greedy argmax).
        cfg_scale: Classifier-free guidance scale (0 = no guidance).
        remasking: Strategy for remasking during denoising ('low_confidence' or 'random').
        device: Device to run on ('auto', 'cuda', 'cpu').
        dtype: Model dtype. Default bf16 to match repo convention.
        max_tactic_tokens: Max tokens to generate for a single tactic.
        max_proof_tokens: Max tokens to generate for a whole proof.
    """

    model_name: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
    mask_id: int = 156895
    steps: int = 128
    gen_length: int = 128
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    device: str = "auto"
    dtype: torch.dtype = torch.bfloat16
    max_tactic_tokens: int = 64
    max_proof_tokens: int = 512

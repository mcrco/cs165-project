"""Diffusion model families and default sampling presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

DIFFUSION_FAMILY_DREAM = "dream"
DIFFUSION_FAMILY_LLADA = "llada"
DIFFUSION_FAMILY_GENERIC = "generic"

DEFAULT_DIFFUSION_STEPS = 8
DEFAULT_DIFFUSION_TEMPERATURE = 0.0
DEFAULT_REMASKING = "low_confidence"
DEFAULT_DIFFUSION_MODEL_NAME = "Dream-org/Dream-v0-Instruct-7B"
DEFAULT_LLADA_MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"

REFERENCE_DREAM_7B_STEPS = 512
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


def normalize_model_name(model_name: Optional[str]) -> str:
    return (model_name or "").strip().lower()


def is_dream_model_name(model_name: Optional[str]) -> bool:
    return normalize_model_name(model_name).startswith("dream-org/dream-v0")


def is_llada_model_name(model_name: Optional[str]) -> bool:
    normalized = normalize_model_name(model_name)
    return normalized.startswith("inclusionai/llada") or normalized.startswith(
        "gsai-ml/llada"
    )


def detect_diffusion_family(model_name: Optional[str]) -> str:
    if is_dream_model_name(model_name):
        return DIFFUSION_FAMILY_DREAM
    if is_llada_model_name(model_name):
        return DIFFUSION_FAMILY_LLADA
    return DIFFUSION_FAMILY_GENERIC


def get_diffusion_sampling_config(
    model_name: Optional[str],
    *,
    mode: str = "infilling",
) -> DiffusionSamplingConfig:
    """Return model-aware sampling defaults."""
    if mode not in {"infilling", "blockwise"}:
        raise ValueError(f"Unsupported diffusion sampling mode: {mode}")

    family = detect_diffusion_family(model_name)
    normalized = normalize_model_name(model_name)
    if family == DIFFUSION_FAMILY_DREAM:
        return DiffusionSamplingConfig(
            steps=REFERENCE_DREAM_7B_STEPS,
            temperature=0.0,
            remasking="dream_origin",
            block_length=128,
            cfg_scale=0.0,
        )

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

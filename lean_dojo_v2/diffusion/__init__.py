"""Diffusion language model module for discrete masked diffusion inference."""

from .config import DiffusionConfig
from .sampler import DiffusionSampler

__all__ = ["DiffusionConfig", "DiffusionSampler"]

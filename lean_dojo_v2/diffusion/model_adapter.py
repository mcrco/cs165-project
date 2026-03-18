"""Family-aware model calling helpers for diffusion models."""

from __future__ import annotations

from typing import Optional

import torch

from .families import DIFFUSION_FAMILY_DREAM, detect_diffusion_family


def _unwrap_model(model):
    """Peel common training wrappers until we reach the underlying model."""
    current = model
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        config = getattr(current, "config", None)
        model_type = str(getattr(config, "model_type", "")).strip().lower()
        if model_type:
            return current

        for attr in ("module", "model", "base_model"):
            wrapped = getattr(current, attr, None)
            if wrapped is not None and wrapped is not current:
                current = wrapped
                break
        else:
            return current
    return current


def is_dream_model(model) -> bool:
    model = _unwrap_model(model)
    config = getattr(model, "config", None)
    if config is None:
        return False
    model_type = str(getattr(config, "model_type", "")).strip().lower()
    if model_type == DIFFUSION_FAMILY_DREAM:
        return True
    name_or_path = getattr(config, "_name_or_path", None)
    return detect_diffusion_family(name_or_path) == DIFFUSION_FAMILY_DREAM


def get_model_family(model) -> str:
    model = _unwrap_model(model)
    if is_dream_model(model):
        return DIFFUSION_FAMILY_DREAM
    config = getattr(model, "config", None)
    if config is None:
        return detect_diffusion_family(None)
    return detect_diffusion_family(getattr(config, "_name_or_path", None))


def prepare_diffusion_forward_kwargs(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> dict:
    """Normalize forward kwargs across diffusion model families."""
    kwargs = {"input_ids": input_ids}
    if not is_dream_model(model):
        kwargs["attention_mask"] = attention_mask
        return kwargs

    if attention_mask is None:
        return kwargs
    if attention_mask.dim() != 2:
        kwargs["attention_mask"] = attention_mask
        return kwargs
    if not torch.any(attention_mask == 0):
        return kwargs

    # Match Dream's official generation code for padded batches.
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    kwargs["position_ids"] = position_ids
    kwargs["attention_mask"] = torch.logical_and(
        attention_mask.unsqueeze(1).unsqueeze(-2).bool(),
        attention_mask.unsqueeze(1).unsqueeze(-1).bool(),
    )
    return kwargs


def forward_diffusion_logits(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    outputs = model(
        **prepare_diffusion_forward_kwargs(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    )
    logits = outputs.logits
    if is_dream_model(model):
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits

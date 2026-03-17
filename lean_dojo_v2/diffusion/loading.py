"""Shared loading helpers for diffusion models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from .families import (
    DiffusionSamplingConfig,
    detect_diffusion_family,
    get_diffusion_sampling_config,
    is_dream_model_name,
)
from .token_utils import resolve_mask_token_id


def sanitize_rope_scaling(cfg):
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


def _infer_family_from_config(
    model_name_or_path: str,
    *,
    trust_remote_code: bool,
) -> Optional[str]:
    try:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        return None
    model_type = str(getattr(config, "model_type", "")).strip().lower()
    if model_type == "dream":
        return "dream"
    if model_type == "llada":
        return "llada"
    return None


def resolve_diffusion_family(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = True,
    base_model_name: Optional[str] = None,
) -> str:
    for candidate in (base_model_name, model_name_or_path):
        family = detect_diffusion_family(candidate)
        if family != "generic":
            return family
    family = _infer_family_from_config(
        base_model_name or model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    return family or "generic"


@dataclass(frozen=True)
class DiffusionComponents:
    family: str
    tokenizer: Any
    model: Any
    mask_token_id: int
    sampling: DiffusionSamplingConfig


def load_diffusion_tokenizer(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = True,
    family: Optional[str] = None,
):
    if family is None:
        family = resolve_diffusion_family(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
    tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
    if family != "dream":
        tokenizer_kwargs["use_fast"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_diffusion_model(
    model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
    trust_remote_code: bool = True,
    family: Optional[str] = None,
    use_lora_adapter: bool = False,
):
    base_model_name = None
    if use_lora_adapter:
        peft_config = PeftConfig.from_pretrained(model_name_or_path)
        base_model_name = peft_config.base_model_name_or_path
    if family is None:
        family = resolve_diffusion_family(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            base_model_name=base_model_name,
        )

    if use_lora_adapter:
        assert base_model_name is not None
        base_model = load_diffusion_model(
            base_model_name,
            torch_dtype=torch_dtype,
            device=None,
            trust_remote_code=trust_remote_code,
            family=family,
            use_lora_adapter=False,
        )
        model = PeftModel.from_pretrained(base_model, model_name_or_path)
    else:
        source_name = base_model_name or model_name_or_path
        config = sanitize_rope_scaling(
            AutoConfig.from_pretrained(
                source_name,
                trust_remote_code=trust_remote_code,
            )
        )
        loader = AutoModel if family == "dream" else AutoModelForCausalLM
        model = loader.from_pretrained(
            source_name,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

    if device is not None:
        model = model.to(device)
    return model


def load_diffusion_components(
    model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
    trust_remote_code: bool = True,
    use_lora_adapter: bool = False,
    for_training: bool = False,
) -> DiffusionComponents:
    del for_training
    base_model_name = None
    if use_lora_adapter:
        base_model_name = PeftConfig.from_pretrained(
            model_name_or_path
        ).base_model_name_or_path
    family = resolve_diffusion_family(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        base_model_name=base_model_name,
    )
    tokenizer = load_diffusion_tokenizer(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        family=family,
    )
    model = load_diffusion_model(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device=device,
        trust_remote_code=trust_remote_code,
        family=family,
        use_lora_adapter=use_lora_adapter,
    )
    mask_token_id = resolve_mask_token_id(tokenizer)
    sampling = get_diffusion_sampling_config(
        base_model_name or model_name_or_path,
        mode="infilling",
    )
    return DiffusionComponents(
        family=family,
        tokenizer=tokenizer,
        model=model,
        mask_token_id=mask_token_id,
        sampling=sampling,
    )

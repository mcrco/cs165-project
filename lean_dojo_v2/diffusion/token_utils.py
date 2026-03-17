"""Shared token-level helpers for diffusion models."""

from __future__ import annotations


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

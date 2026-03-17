"""Dream-family diffusion sampling helpers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributions as dists
import torch.nn.functional as F

from ..model_adapter import forward_diffusion_logits


def top_p_logits(logits: torch.Tensor, top_p: Optional[float] = None) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def top_k_logits(logits: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
    top_k = min(int(top_k), logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)


def dream_sample_tokens(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        x0 = dists.Categorical(probs=probs).sample()
        confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[:, 0] - sorted_probs[:, 1]

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


def denoise_dream_masked_sequence(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    mask_token_id: int,
    steps: int,
    temperature: float,
    remasking: str,
    eps: float = 1e-3,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    alg_temp: Optional[float] = None,
) -> torch.Tensor:
    """Dream diffusion denoiser adapted for arbitrary masked spans."""
    x = input_ids.clone()
    steps = max(1, int(steps))
    timesteps = torch.linspace(1.0, eps, steps + 1, device=x.device)
    alg = remasking.removeprefix("dream_")

    for step_idx in range(steps):
        mask_index = x.eq(mask_token_id)
        if int(mask_index.sum().item()) == 0:
            break

        with torch.no_grad():
            logits = forward_diffusion_logits(
                model=model,
                input_ids=x,
                attention_mask=attention_mask,
            )

        mask_logits = logits[mask_index]
        if mask_logits.numel() == 0:
            break

        t = timesteps[step_idx]
        s = timesteps[step_idx + 1]
        if alg == "origin":
            p_transfer = 1 - s / t if step_idx < steps - 1 else 1.0
            sampled = torch.full_like(x[mask_index], mask_token_id)
            transfer_now = torch.rand(sampled.shape, device=x.device) < p_transfer
            if int(transfer_now.sum().item()) > 0:
                _, sampled_now = dream_sample_tokens(
                    mask_logits[transfer_now],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                sampled[transfer_now] = sampled_now
            x[mask_index] = sampled
            continue

        if alg == "maskgit_plus":
            confidence, x0 = dream_sample_tokens(
                mask_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        elif alg == "topk_margin":
            confidence, x0 = dream_sample_tokens(
                mask_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                margin_confidence=True,
            )
        elif alg == "entropy":
            confidence, x0 = dream_sample_tokens(
                mask_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                neg_entropy=True,
            )
        else:
            raise NotImplementedError(remasking)

        full_confidence = torch.full_like(x, float("-inf"), dtype=logits.dtype)
        full_confidence[mask_index] = confidence
        proposed = torch.full_like(x, mask_token_id)
        proposed[mask_index] = x0

        for batch_idx in range(x.shape[0]):
            available = int(mask_index[batch_idx].sum().item())
            if available <= 0:
                continue
            if step_idx < steps - 1:
                num_transfer_tokens = int(available * float(1 - s / t))
            else:
                num_transfer_tokens = available
            if num_transfer_tokens <= 0:
                continue
            num_transfer_tokens = min(num_transfer_tokens, available)

            confidence_row = full_confidence[batch_idx]
            if alg_temp is None or alg_temp == 0:
                _, transfer_index = torch.topk(confidence_row, num_transfer_tokens)
            else:
                scaled_confidence = torch.softmax(confidence_row / alg_temp, dim=-1)
                transfer_index = torch.multinomial(
                    scaled_confidence,
                    num_samples=num_transfer_tokens,
                )
            x[batch_idx, transfer_index] = proposed[batch_idx, transfer_index]

    return x

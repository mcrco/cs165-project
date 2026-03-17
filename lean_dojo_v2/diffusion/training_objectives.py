"""Family-aware training objectives for diffusion models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .model_adapter import prepare_diffusion_forward_kwargs


def q_sample(
    input_ids: torch.Tensor,
    *,
    maskable_mask: torch.Tensor,
    mask_token_id: int,
    min_t: float = 0.0,
    max_t: float = 1.0,
    eos_token_id: Optional[int] = None,
    t: Optional[torch.Tensor] = None,
    t_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_0 = input_ids

    if t_mask is None:
        if t is None:
            t = torch.rand((x_0.shape[0],), dtype=torch.float, device=input_ids.device)
            t = min_t + (max_t - min_t) * t
        u = torch.rand_like(x_0, dtype=torch.float)
        t_mask = (u < t[:, None]) & maskable_mask

    x_t = x_0.masked_fill(t_mask, mask_token_id)

    if eos_token_id is not None:
        last_non_eos_token_idx = ((input_ids != eos_token_id) | (~maskable_mask)).sum(
            dim=-1
        ) - 1
        seq_len = x_0.shape[1]
        for i in range(x_0.shape[0]):
            if last_non_eos_token_idx[i] < seq_len - 1:
                t_mask_at_eos = t_mask[i, last_non_eos_token_idx[i] + 1]
                if t_mask_at_eos:
                    x_t[i, last_non_eos_token_idx[i] + 1 :] = mask_token_id
                    t_mask[i, last_non_eos_token_idx[i] + 1 :] = True
                else:
                    x_t[i, last_non_eos_token_idx[i] + 1 :] = eos_token_id
                    t_mask[i, last_non_eos_token_idx[i] + 1 :] = False

    return x_t, t, t_mask


def context_adaptive_reweight(seq_len: int, *, cart_p: float = 0.8) -> torch.Tensor:
    position_ids_l = torch.arange(seq_len).reshape(-1, 1)
    position_ids_r = torch.arange(seq_len).reshape(1, -1)
    distance = position_ids_l - position_ids_r
    if not 0 < cart_p <= 1:
        raise ValueError("cart_p must be between 0 and 1")

    res = (math.log(cart_p) + (distance.abs() - 1) * math.log(1 - cart_p)).exp() * 0.5
    res.masked_fill_(distance == 0, 0)
    return res


def _find_infilling_supervised_end(
    span_ids: torch.Tensor,
    *,
    mask_start: int,
    mask_end: int,
    eos_token_id: Optional[int],
    mask_token_id: int,
) -> int:
    """Supervise only the tactic tokens and the first EOS inside a fixed infilling span."""
    effective_end = mask_end
    if eos_token_id is not None:
        eos_positions = (span_ids == eos_token_id).nonzero(as_tuple=False)
        if eos_positions.numel() > 0:
            return mask_start + int(eos_positions[0].item()) + 1

    mask_positions = (span_ids == mask_token_id).nonzero(as_tuple=False)
    if mask_positions.numel() > 0:
        effective_end = mask_start + int(mask_positions[0].item())
    return effective_end


@dataclass
class DiffusionTrainingObjective:
    family: str
    mode: str
    tokenizer: Any
    mask_token_id: int
    min_mask_ratio: float
    max_mask_ratio: float
    pad_token_id: Optional[int] = None
    time_reweighting: str = "cart"
    token_reweighting: bool = False
    alpha: float = 1.0
    gamma: float = 1.0
    cart_p: float = 0.8

    def __post_init__(self) -> None:
        if not (0.0 < self.min_mask_ratio <= self.max_mask_ratio <= 1.0):
            raise ValueError("Mask ratios must satisfy 0 < min <= max <= 1")
        if self.mode not in {"sft", "infilling"}:
            raise ValueError(f"Unsupported objective mode: {self.mode}")
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if self.family == "dream":
            return self._dream_collate(features)
        return self._llada_collate(features)

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        if self.family == "dream":
            return self._compute_dream_loss(model, inputs, return_outputs=return_outputs)
        return self._compute_llada_loss(model, inputs, return_outputs=return_outputs)

    def _llada_collate(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_size = len(features)

        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, feature in enumerate(features):
            ids = torch.tensor(feature["input_ids"], dtype=torch.long)
            seq_len = ids.size(0)

            input_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1

            if self.mode == "sft":
                assistant_start = int(feature["assistant_start"])
                if assistant_start >= seq_len:
                    continue
                real_positions = []
                pad_positions = []
                for pos in range(assistant_start, seq_len):
                    if ids[pos].item() == self.mask_token_id:
                        pad_positions.append(pos)
                    else:
                        real_positions.append(pos)
                for pos in pad_positions:
                    labels[i, pos] = ids[pos]
            else:
                mask_start = int(feature.get("mask_start", -1))
                mask_end = int(feature.get("mask_end", -1))
                if mask_start < 0 or mask_end <= mask_start:
                    continue
                mask_end = min(mask_end, seq_len)
                span_ids = input_ids[i, mask_start:mask_end].clone()
                supervised_end = _find_infilling_supervised_end(
                    span_ids,
                    mask_start=mask_start,
                    mask_end=mask_end,
                    eos_token_id=self.tokenizer.eos_token_id,
                    mask_token_id=self.mask_token_id,
                )
                if supervised_end <= mask_start:
                    continue
                span_positions = torch.arange(mask_start, supervised_end, dtype=torch.long)
                real_positions = span_positions

            if self.mode == "sft":
                if not real_positions:
                    continue
                real_positions = torch.tensor(real_positions, dtype=torch.long)
            elif real_positions.numel() == 0:
                continue

            mask_ratio = (
                torch.empty(1).uniform_(self.min_mask_ratio, self.max_mask_ratio).item()
            )
            num_to_mask = max(1, int(round(real_positions.numel() * mask_ratio)))
            chosen = real_positions[torch.randperm(real_positions.numel())[:num_to_mask]]
            input_ids[i, chosen] = self.mask_token_id
            labels[i, chosen] = ids[chosen]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _dream_collate(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for i, feature in enumerate(features):
            ids = torch.tensor(feature["input_ids"], dtype=torch.long)
            seq_len = ids.size(0)
            input_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = True

            if self.mode == "sft":
                assistant_start = int(feature["assistant_start"])
                if assistant_start < seq_len:
                    loss_mask[i, assistant_start:seq_len] = True
            else:
                mask_start = int(feature.get("mask_start", -1))
                mask_end = int(feature.get("mask_end", -1))
                if mask_start >= 0 and mask_end > mask_start:
                    mask_end = min(mask_end, seq_len)
                    span_ids = ids[mask_start:mask_end]
                    supervised_end = _find_infilling_supervised_end(
                        span_ids,
                        mask_start=mask_start,
                        mask_end=mask_end,
                        eos_token_id=self.tokenizer.eos_token_id,
                        mask_token_id=self.mask_token_id,
                    )
                    if supervised_end > mask_start:
                        loss_mask[i, mask_start:supervised_end] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }

    def _compute_llada_loss(self, model, inputs, return_outputs: bool = False):
        labels = inputs.pop("labels")
        outputs = model(
            **prepare_diffusion_forward_kwargs(
                model=model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
        )
        logits = outputs.logits
        vocab_size = logits.shape[-1]
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss

    def _compute_dream_loss(self, model, inputs, return_outputs: bool = False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        loss_mask = inputs["loss_mask"].bool()
        labels = input_ids.contiguous()

        masked_input_ids, t, loss_mask_nonflatten = q_sample(
            input_ids,
            maskable_mask=loss_mask,
            mask_token_id=self.mask_token_id,
            eos_token_id=self.pad_token_id,
        )
        outputs = model(
            **prepare_diffusion_forward_kwargs(
                model=model,
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
            )
        )
        logits = outputs.logits
        shift_logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1).contiguous()
        flat_logits = shift_logits.view(-1, shift_logits.shape[-1])
        flat_labels = labels.view(-1).to(flat_logits.device)
        loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")

        flat_loss_mask = loss_mask_nonflatten.reshape(-1).to(loss.device)
        loss = loss.masked_fill(~flat_loss_mask, 0)
        if self.token_reweighting:
            loss = self.alpha * (1 - torch.exp(-loss)) ** self.gamma * loss

        if self.time_reweighting == "original":
            weight = 1 / t[:, None].float().expand(labels.size())
        elif self.time_reweighting == "linear":
            weight = 1 - t[:, None].float().expand(labels.size())
        elif self.time_reweighting == "cart":
            seq_len = input_ids.shape[-1]
            weight_matrix = context_adaptive_reweight(seq_len, cart_p=self.cart_p)
            _weight_matrix = weight_matrix[:seq_len, :seq_len].to(loss.device)
            non_mask = ~loss_mask_nonflatten.to(loss.device)
            weight = (
                non_mask.type_as(_weight_matrix)
                .matmul(_weight_matrix)
                .masked_fill(non_mask, 0)
            )
        else:
            weight = t.new_ones((input_ids.shape[0], 1)).float().expand(labels.size())

        loss = loss * weight.reshape(-1)
        valid = torch.sum(flat_loss_mask)
        loss = torch.sum(loss) / valid
        return (loss, outputs) if return_outputs else loss


def create_diffusion_training_objective(
    *,
    family: str,
    mode: str,
    tokenizer,
    mask_token_id: int,
    min_mask_ratio: float,
    max_mask_ratio: float,
    pad_token_id: Optional[int] = None,
) -> DiffusionTrainingObjective:
    return DiffusionTrainingObjective(
        family=family,
        mode=mode,
        tokenizer=tokenizer,
        mask_token_id=mask_token_id,
        min_mask_ratio=min_mask_ratio,
        max_mask_ratio=max_mask_ratio,
        pad_token_id=pad_token_id,
    )

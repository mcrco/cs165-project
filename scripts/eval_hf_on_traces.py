#!/usr/bin/env python3
"""Pilot evaluation for HF/DeepSeek tactic models on traced tactic datasets.

Input JSON format is the LeanDojo-style theorem list with `traced_tactics`.
Metrics:
- exact@1: first sampled tactic string exactly matches gold tactic
- exact@k: any of k sampled tactics matches gold tactic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from lean_dojo_v2.prover.hf_prover import HFProver
from lean_dojo_v2.utils import remove_marks

REPO_DIR = Path(__file__).resolve().parents[1]
APRIL_DIR = REPO_DIR / "datasets" / "april"
DEFAULT_DATA_JSON = APRIL_DIR / "leandojo" / "val.json"


SYSTEM_PROMPT = (
    "You are a Lean 4 tactic generator. Given a goal state, "
    "output exactly ONE Lean tactic that advances or solves the goal.\n"
    "Rules:\n"
    "- Output only the tactic text; no prose, quotes, or code fences.\n"
    "- Single line only; no `by` blocks.\n"
    "- Never use `sorry` or `admit`.\n"
)


def normalize_tactic(t: str) -> str:
    return remove_marks((t or "")).splitlines()[0].split("<;>")[0].strip()


def iter_examples(data: list[dict[str, Any]]):
    for item in data:
        for tac in item.get("traced_tactics", []):
            goal = remove_marks(tac.get("state_before", "")).strip()
            gold = normalize_tactic(tac.get("tactic", ""))
            if goal and gold and gold != "sorry":
                yield goal, gold


def make_prompt(goal_state: str) -> str:
    return f"### System:\n{SYSTEM_PROMPT}### User:\n{goal_state}\n\n### Assistant:\n"


def sample_tactics(
    prover: HFProver,
    goal_state: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    prompt = make_prompt(goal_state)
    inputs = prover.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(prover.device)

    with torch.no_grad():
        outputs = prover.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=k,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=prover.tokenizer.pad_token_id,
            eos_token_id=prover.tokenizer.eos_token_id,
        )

    decoded = prover.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    preds: list[str] = []
    for text in decoded:
        tactic = text[len(prompt) :].strip()
        tactic = normalize_tactic(tactic)
        if tactic and tactic != "sorry":
            preds.append(tactic)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-json",
        type=Path,
        default=DEFAULT_DATA_JSON,
        help=f"LeanDojo val JSON (default: {DEFAULT_DATA_JSON})",
    )
    parser.add_argument("--ckpt", type=str, default="deepseek-ai/DeepSeek-Prover-V2-7B")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    data = json.loads(args.data_json.read_text(encoding="utf-8"))
    prover = HFProver(ckpt_path=args.ckpt, use_lora=args.use_lora, device=args.device)

    total = 0
    hit1 = 0
    hitk = 0

    for goal_state, gold in iter_examples(data):
        preds = sample_tactics(
            prover=prover,
            goal_state=goal_state,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if not preds:
            continue

        total += 1
        if preds[0] == gold:
            hit1 += 1
        if gold in preds:
            hitk += 1

        if total >= args.max_steps:
            break

    exact1 = (hit1 / total) if total else 0.0
    exactk = (hitk / total) if total else 0.0

    print(f"evaluated_steps={total}")
    print(f"exact@1={exact1:.4f}")
    print(f"exact@{args.k}={exactk:.4f}")


if __name__ == "__main__":
    main()

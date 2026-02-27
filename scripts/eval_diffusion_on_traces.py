#!/usr/bin/env python3
"""Pilot evaluation for diffusion tactic models on traced tactic datasets.

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

from lean_dojo_v2.prover.diffusion_prover import DiffusionProver
from lean_dojo_v2.utils import remove_marks

REPO_DIR = Path(__file__).resolve().parents[1]
APRIL_DIR = REPO_DIR / "datasets" / "april"
DEFAULT_DATA_JSON = APRIL_DIR / "leandojo" / "val.json"


def normalize_tactic(t: str) -> str:
    return remove_marks((t or "")).splitlines()[0].split("<;>")[0].strip()


def iter_examples(data: list[dict[str, Any]]):
    for item in data:
        for tac in item.get("traced_tactics", []):
            goal = remove_marks(tac.get("state_before", "")).strip()
            gold = normalize_tactic(tac.get("tactic", ""))
            if goal and gold and gold != "sorry":
                yield goal, gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-json",
        type=Path,
        default=DEFAULT_DATA_JSON,
        help=f"LeanDojo val JSON (default: {DEFAULT_DATA_JSON})",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    data = json.loads(args.data_json.read_text(encoding="utf-8"))

    prover = DiffusionProver(ckpt_path=args.ckpt, device=args.device)

    total = 0
    hit1 = 0
    hitk = 0

    for goal_state, gold in iter_examples(data):
        prompt = prover._build_chat_prompt(
            system_prompt=(
                "You are a Lean 4 tactic generator. Given a goal state, "
                "output exactly ONE Lean tactic that advances or solves the goal.\n"
                "Rules:\n"
                "- Output only the tactic text; no prose, quotes, or code fences.\n"
                "- Single line only; no `by` blocks.\n"
                "- Never use `sorry` or `admit`.\n"
            ),
            user_content=goal_state,
        )

        preds = prover._diffusion_sample(
            prompt=prompt,
            gen_length=args.gen_length,
            num_return_sequences=args.k,
            steps=args.steps,
            block_length=32,
            temperature=0.7,
            cfg_scale=0.0,
            remasking="low_confidence",
        )
        preds = [normalize_tactic(p) for p in preds if normalize_tactic(p)]
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

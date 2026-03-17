#!/usr/bin/env python3
"""Evaluate the overfit-200 model: generate tactics for the same examples it was
trained on and check if it can reproduce them (memorization test).

Uses the real LLaDA diffusion sampling algorithm (not a simplified version).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from lean_dojo_v2.diffusion import generate_llada_blockwise, load_diffusion_components


def build_prompt(tokenizer, goal_state: str) -> str:
    messages = [
        {"role": "system", "content": (
            "You are a Lean 4 tactic generator. Given a goal state, "
            "output exactly ONE Lean tactic that advances or solves the goal.\n"
            "Rules:\n- Output only the tactic text; no prose or code fences.\n"
            "- Single line only; no `by` blocks.\n"
            "- Never use `sorry` or `admit`.\n"
        )},
        {"role": "user", "content": goal_state},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="inclusionAI/LLaDA-MoE-7B-A1B-Instruct")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--block-length", type=int, default=64)
    args = parser.parse_args()

    print("Loading tokenizer and model...")
    components = load_diffusion_components(
        args.ckpt_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_lora_adapter=True,
    )
    tokenizer = components.tokenizer
    model = components.model
    model.eval()
    mask_id = components.mask_token_id
    print(f"Mask token ID: {mask_id}")

    test_cases = []
    for p in sorted(args.data_dir.glob("*.json")):
        data = json.loads(p.read_text())
        for item in data:
            for t in item.get("traced_tactics", []):
                tactic = t["tactic"].splitlines()[0].strip()
                if tactic and tactic != "sorry":
                    test_cases.append({
                        "goal": t["state_before"],
                        "expected": tactic,
                        "theorem": item.get("full_name", "unknown"),
                    })

    print(f"\nEvaluating on {len(test_cases)} tactic examples", flush=True)
    print("=" * 80, flush=True)

    exact = 0
    partial = 0
    results = []

    for i, tc in enumerate(test_cases):
        prompt_text = build_prompt(tokenizer, tc["goal"])
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(model.device)

        with torch.no_grad():
            out = generate_llada_blockwise(
                model=model,
                prompt=prompt_ids,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=mask_id,
            )

        generated = out[0, prompt_ids.shape[1]:]
        eos_positions = (generated == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if eos_positions.numel() > 0:
            generated = generated[:eos_positions[0]]
        mask_positions = (generated == mask_id).nonzero(as_tuple=True)[0]
        if mask_positions.numel() > 0:
            generated = generated[:mask_positions[0]]
        output = tokenizer.decode(generated, skip_special_tokens=True).strip().split("\n")[0].strip()
        expected = tc["expected"].strip()

        match = ""
        if output == expected:
            exact += 1
            match = " *** EXACT ***"
        elif expected.lower() in output.lower() or output.lower() in expected.lower():
            partial += 1
            match = " ~~ partial ~~"

        results.append({"expected": expected, "output": output, "match": match})

        print(f"\n--- [{i+1}/{len(test_cases)}] {tc['theorem']} ---", flush=True)
        print(f"  Goal:     {tc['goal'][:100]}...", flush=True)
        print(f"  Expected: {expected}", flush=True)
        print(f"  Output:   {output}{match}", flush=True)

    print("\n" + "=" * 80)
    print("OVERFIT TEST RESULTS")
    print("=" * 80)
    print(f"  Total examples:  {len(test_cases)}")
    print(f"  Exact matches:   {exact} ({exact/len(test_cases)*100:.1f}%)")
    print(f"  Partial matches: {partial} ({partial/len(test_cases)*100:.1f}%)")
    print(f"  No match:        {len(test_cases)-exact-partial} ({(len(test_cases)-exact-partial)/len(test_cases)*100:.1f}%)")
    print("=" * 80)

    if exact / len(test_cases) >= 0.5:
        print("\nVERDICT: Pipeline is WORKING - model can memorize training data.")
    elif exact / len(test_cases) >= 0.2:
        print("\nVERDICT: Pipeline PARTIALLY works - model is learning but not memorizing fully.")
    else:
        print("\nVERDICT: Pipeline may have issues - model cannot memorize even 200 examples.")


if __name__ == "__main__":
    main()

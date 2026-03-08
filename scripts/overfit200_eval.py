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

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def generate(model, prompt, steps=128, gen_length=128, block_length=128,
             temperature=0.7, cfg_scale=0.0, remasking="low_confidence", mask_id=156895):
    """Official LLaDA diffusion sampling algorithm."""
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)

    batch_size = prompt.shape[0]
    x = torch.full(
        (batch_size, prompt.shape[1] + gen_length), mask_id,
        dtype=torch.long, device=prompt.device,
    )
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if k <= 0:
                    continue
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


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

    print("Loading tokenizer from checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.ckpt_dir)
    model.eval()

    mask_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
    if mask_id is None or mask_id < 0:
        mask_id = 156895
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
            out = generate(
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

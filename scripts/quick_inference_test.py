"""Quick inference sanity check: does the fine-tuned diffusion model produce anything sensible?"""
import json
import torch

from lean_dojo_v2.diffusion import (
    generate_llada_blockwise,
    load_diffusion_components,
)

CKPT = "/resnick/scratch/tram/cs165-project/outputs-diffusion-sft"
BASE = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"

print("Loading tokenizer...")
components = load_diffusion_components(
    CKPT,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_lora_adapter=True,
)
tokenizer = components.tokenizer

print("Loading base model on GPU...")
model = components.model
model.eval()
print("Model loaded!\n")

test_files = [
    "datasets/april/leandojo/test/thme_test.json",
    "datasets/april/leandojo/test/lme_test.json",
]
test_cases = []
for f in test_files:
    data = json.load(open(f))
    for item in data:
        for t in item.get("traced_tactics", []):
            tactic = t["tactic"].splitlines()[0].strip()
            if tactic and tactic != "sorry" and len(tactic) < 80:
                test_cases.append({
                    "goal": t["state_before"],
                    "expected": tactic,
                    "theorem": item["full_name"]
                })
            if len(test_cases) >= 10:
                break
        if len(test_cases) >= 10:
            break
    if len(test_cases) >= 10:
        break

mask_id = components.mask_token_id

print("=" * 80)
print("INFERENCE TEST: Fine-tuned diffusion model on APRIL test examples")
print(f"Using mask token id: {mask_id}")
print("=" * 80)

exact_matches = 0
partial_matches = 0

for i, tc in enumerate(test_cases):
    messages = [
        {"role": "system", "content": (
            "You are a Lean 4 tactic generator. Given a goal state, "
            "output exactly ONE Lean tactic that advances or solves the goal.\n"
            "Rules:\n- Output only the tactic text; no prose or code fences.\n"
            "- Single line only; no `by` blocks.\n"
            "- Never use `sorry` or `admit`.\n"
        )},
        {"role": "user", "content": tc["goal"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()

    generated = generate_llada_blockwise(
        model=model,
        prompt=prompt_ids,
        steps=32,
        gen_length=64,
        block_length=64,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=mask_id,
    )[:, prompt_ids.shape[1]:]
    output = tokenizer.decode(generated[0], skip_special_tokens=True).strip().split("\n")[0].strip()
    expected = tc["expected"].strip()

    match_str = ""
    if output == expected:
        exact_matches += 1
        match_str = " *** EXACT MATCH ***"
    elif expected.lower() in output.lower() or output.lower() in expected.lower():
        partial_matches += 1
        match_str = " ~~ partial match ~~"

    print(f"\n--- Example {i+1} (theorem: {tc['theorem']}) ---")
    print(f"  Goal:     {tc['goal'][:120]}...")
    print(f"  Expected: {expected}")
    print(f"  Model:    {output}{match_str}")

print("\n" + "=" * 80)
print(f"RESULTS: {exact_matches}/{len(test_cases)} exact matches, "
      f"{partial_matches}/{len(test_cases)} partial matches")
print(f"Total with some match: {exact_matches + partial_matches}/{len(test_cases)}")
print("=" * 80)

# Also compare to base model (no LoRA) for reference
print("\n\nNow testing BASE MODEL (no fine-tuning) for comparison...")
base_only = load_diffusion_components(
    BASE,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).model
base_only.eval()

base_exact = 0
base_partial = 0

for i, tc in enumerate(test_cases[:5]):
    messages = [
        {"role": "system", "content": (
            "You are a Lean 4 tactic generator. Given a goal state, "
            "output exactly ONE Lean tactic that advances or solves the goal.\n"
            "Rules:\n- Output only the tactic text; no prose or code fences.\n"
            "- Single line only; no `by` blocks.\n"
            "- Never use `sorry` or `admit`.\n"
        )},
        {"role": "user", "content": tc["goal"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()

    generated = generate_llada_blockwise(
        model=base_only,
        prompt=prompt_ids,
        steps=32,
        gen_length=64,
        block_length=64,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=mask_id,
    )[:, prompt_ids.shape[1]:]
    output = tokenizer.decode(generated[0], skip_special_tokens=True).strip().split("\n")[0].strip()
    expected = tc["expected"].strip()

    match_str = ""
    if output == expected:
        base_exact += 1
        match_str = " *** EXACT MATCH ***"
    elif expected.lower() in output.lower() or output.lower() in expected.lower():
        base_partial += 1
        match_str = " ~~ partial match ~~"

    print(f"\n--- Base Example {i+1} (theorem: {tc['theorem']}) ---")
    print(f"  Expected: {expected}")
    print(f"  Base:     {output}{match_str}")

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print(f"  Fine-tuned: {exact_matches} exact, {partial_matches} partial out of {len(test_cases)}")
print(f"  Base model: {base_exact} exact, {base_partial} partial out of {min(5, len(test_cases))}")
print("=" * 80)

"""Quick inference sanity check: does the fine-tuned diffusion model produce anything sensible?"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

CKPT = "/resnick/scratch/tram/cs165-project/outputs-diffusion-sft"
BASE = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CKPT)

print("Loading base model on GPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, trust_remote_code=True,
    device_map="auto"
)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, CKPT)
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

mask_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
if mask_id is None or mask_id < 0:
    mask_id = 156895

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

    gen_length = 64
    x = torch.full((1, prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device="cuda")
    x[:, :prompt_ids.shape[1]] = prompt_ids

    # Iterative denoising: 32 steps (lighter version of the full 128-step schedule)
    num_steps = 32
    mask_positions_orig = (x == mask_id)
    num_masked = mask_positions_orig.sum().item()

    for step in range(num_steps):
        with torch.no_grad():
            logits = model(x).logits

        still_masked = (x == mask_id)
        if not still_masked.any():
            break

        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        confidence[~still_masked] = float('inf')

        n_to_unmask = max(1, int(num_masked * (1.0 / num_steps)))
        if step == num_steps - 1:
            n_to_unmask = still_masked.sum().item()

        flat_conf = confidence.view(-1)
        flat_mask = still_masked.view(-1)
        flat_conf[~flat_mask] = float('inf')

        _, indices = flat_conf.topk(min(n_to_unmask, flat_mask.sum().item()), largest=False)
        predicted = torch.argmax(logits.view(-1, logits.size(-1)), dim=-1)
        x.view(-1)[indices] = predicted[indices]

    generated = x[:, prompt_ids.shape[1]:]
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
base_only = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, trust_remote_code=True,
    device_map="auto"
)
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

    gen_length = 64
    x = torch.full((1, prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device="cuda")
    x[:, :prompt_ids.shape[1]] = prompt_ids

    num_steps = 32
    mask_positions_orig = (x == mask_id)
    num_masked = mask_positions_orig.sum().item()

    for step in range(num_steps):
        with torch.no_grad():
            logits = base_only(x).logits

        still_masked = (x == mask_id)
        if not still_masked.any():
            break

        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        confidence[~still_masked] = float('inf')

        n_to_unmask = max(1, int(num_masked * (1.0 / num_steps)))
        if step == num_steps - 1:
            n_to_unmask = still_masked.sum().item()

        flat_conf = confidence.view(-1)
        flat_mask = still_masked.view(-1)
        flat_conf[~flat_mask] = float('inf')

        _, indices = flat_conf.topk(min(n_to_unmask, flat_mask.sum().item()), largest=False)
        predicted = torch.argmax(logits.view(-1, logits.size(-1)), dim=-1)
        x.view(-1)[indices] = predicted[indices]

    generated = x[:, prompt_ids.shape[1]:]
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

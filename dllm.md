
### Diffusion Language Models (DLMs) for LeanDojo-v2 — Implementation Outline

This document is a concrete plan to integrate **diffusion-based language models** into this repo for **automated Lean 4 theorem proving**.

LeanDojo-v2 already supports theorem proving through:
- **Next-tactic generation** (core trainable task today): `GoalState -> tactic` used inside proof search (`BaseProver.search`).
- **Whole-proof generation** (inference mode today): `theorem statement -> proof`.

This plan prioritizes **next-tactic diffusion training + inference** first (because it matches the existing training/data + proof-search interface), then extends to **whole-proof diffusion**.

**Scope note (important): this plan focuses on NON-retrieval proving.**
- We target diffusion models that operate directly in the existing proving APIs:
  - `BaseProver.next_tactic(...)` for proof search
  - `BaseProver.generate_whole_proof(...)` for whole-proof generation
- We intentionally do **not** cover the retrieval subsystem (`LeanAgent`, `RetrievalTrainer`, `RetrievalProver`, premise indexing, R@k eval).

---

### Goals (What "done" looks like)

- **G1: Next-tactic diffusion prover works end-to-end**
  - A diffusion model can sample tactic strings from a Lean goal state prompt.
  - The prover can plug into existing proof search (`BaseProver.search`) and solve at least some `sorry` theorems in a traced repo.
  - We have an evaluation harness for proof success + tactic quality.

- **G2: Next-tactic diffusion training pipeline exists**
  - We can train/fine-tune a diffusion model on LeanDojo-v2 traced tactic data.
  - Training produces checkpoints + a reproducible config, and integrates cleanly with this repo's conventions.

- **G3: Whole-proof diffusion inference is supported**
  - A diffusion model can produce full Lean proofs for a theorem statement (even if initial quality is low).
  - This is integrated as a prover mode similar to `generate_whole_proof`.

- **G4 (optional/longer-term): Whole-proof diffusion training**
  - We can build/curate a dataset of theorem statements paired with proof terms and train diffusion models for whole-proof generation.

---

### Current repo interfaces (integration "contract")

- **Proof search loop**: implemented in [`base_prover.py`](lean_dojo_v2/prover/base_prover.py).
  - [`BaseProver`](lean_dojo_v2/prover/base_prover.py#L17-L277) extends `pantograph.search.Agent` (ABC).
  - The search loop is [`BaseProver.search()`](lean_dojo_v2/prover/base_prover.py#L53-L229): stack-based DFS with `max_steps=100`, `max_trials_per_goal=5`.
  - The only model-dependent method for search is:
    - [`next_tactic(state: GoalState, goal_id: int) -> Optional[Tactic]`](lean_dojo_v2/prover/base_prover.py#L28-L43) (abstract)
  - So "next-tactic diffusion" needs to implement: `next_tactic`.

- **Whole-proof generation**: [`BaseProver.generate_whole_proof(theorem: Theorem) -> str`](lean_dojo_v2/prover/base_prover.py#L231-L244) (abstract).

- **Agent orchestration**: [`base_agent.py`](lean_dojo_v2/agent/base_agent.py).
  - [`BaseAgent`](lean_dojo_v2/agent/base_agent.py#L15-L141) handles: tracing + database + iterating over `sorry` theorems.
  - Key abstract methods to implement:
    - [`_setup_prover()`](lean_dojo_v2/agent/base_agent.py#L99-L102): instantiate `self.prover`.
    - [`_get_build_deps()`](lean_dojo_v2/agent/base_agent.py#L27-L30): whether to build deps for tracing.
  - The prove flow: [`prove()`](lean_dojo_v2/agent/base_agent.py#L104-L114) → [`prove_theorem()`](lean_dojo_v2/agent/base_agent.py#L116-L141) → `self.prover.search()` or `self.prover.generate_whole_proof()`.
  - A new diffusion prover can be dropped in by making a new agent class (like [`ExternalAgent`](lean_dojo_v2/agent/external_agent.py)).

- **Existing training tasks (non-retrieval)**
  - [`SFTTrainer`](lean_dojo_v2/trainer/sft_trainer.py#L72-L148) / [`GRPOTrainer`](lean_dojo_v2/trainer/grpo_trainer.py#L66-L144): train **next-tactic** causal LM policies (baseline to compare against diffusion).
  - [`ProgressTrainer`](lean_dojo_v2/trainer/progress_trainer.py#L53-L187): step-remaining regressor (not a generator; optional auxiliary signal).

- **Existing provers (reference implementations)**
  - [`HFProver`](lean_dojo_v2/prover/hf_prover.py): local HuggingFace causal LM inference — **primary template** for `DiffusionProver`.
    - [`next_tactic()`](lean_dojo_v2/prover/hf_prover.py#L41-L91): prompt construction, `model.generate()`, post-processing (strip `\n`, split `<;>`, skip `sorry`).
    - [`generate_whole_proof()`](lean_dojo_v2/prover/hf_prover.py#L93-L127): longer generation from theorem statement.
  - [`ExternalProver`](lean_dojo_v2/prover/external_prover.py): delegates to [`HFTacticGenerator`](lean_dojo_v2/external_api/python/external_models/hf_runner.py) (HTTP API).

- **Existing datasets (reference implementations)**
  - [`SFTDataset`](lean_dojo_v2/trainer/sft_trainer.py#L22-L69): reads merged JSON, iterates `traced_tactics`, formats as chat messages.
  - [`GRPODataset`](lean_dojo_v2/trainer/grpo_trainer.py#L21-L63): same source, different format (prompt-only for RL).

- **Registration points** (files to update when adding a new prover/agent/trainer)
  - [`prover/__init__.py`](lean_dojo_v2/prover/__init__.py): add `DiffusionProver` to imports and `__all__`.
  - [`agent/__init__.py`](lean_dojo_v2/agent/__init__.py): add `DiffusionAgent` to imports and `__all__`.
  - [`trainer/__init__.py`](lean_dojo_v2/trainer/__init__.py): add `DiffusionTrainer` to imports and `__all__`.

---

### Phase 0 — Decide diffusion formulation + pick a baseline implementation

You need a concrete DLM "shape" before coding.

- **D0.0: Library choice (recommended default: `transformers` + `torch` + `accelerate`)**
  - Many discrete-token diffusion LMs (e.g., LLaDA/DREAM-style masked denoising) are implemented primarily with:
    - `transformers` (model backbone, tokenizer, config)
    - `torch` (custom corruption + denoising loop)
    - `accelerate` (multi-GPU + mixed precision training)
  - **`diffusers` is optional**:
    - use it if you want its training-script conventions/utilities, but don't assume it provides a turnkey discrete-token diffusion scheduler.
  - **Current repo convention**: existing trainers ([`SFTTrainer`](lean_dojo_v2/trainer/sft_trainer.py), [`GRPOTrainer`](lean_dojo_v2/trainer/grpo_trainer.py)) use `transformers` + `trl` + `peft` for model loading, and [`HFProver`](lean_dojo_v2/prover/hf_prover.py) uses `AutoModelForCausalLM` + `AutoTokenizer` for inference. The diffusion stack should align with this convention (use `transformers` for backbone, but replace `trl` training loop with a custom discrete-diffusion loop).

- **D0.1: Choose diffusion formulation**
  - **Discrete diffusion** over tokens (e.g., mask-and-denoise / absorbing diffusion) is simplest to align with text output.
  - Alternative: diffusion in embedding space + decoder (more moving parts).

- **D0.2: Decide inference API**
  - **Local PyTorch inference** (preferred for research iteration + training alignment).
  - Optional: **HTTP inference service** (mirrors current [`ExternalProver`](lean_dojo_v2/prover/external_prover.py) style via [`HFTacticGenerator`](lean_dojo_v2/external_api/python/external_models/hf_runner.py)).

- **D0.3: Define an internal sampler interface**
  - A single class that exposes:
    - `sample_tactic(prompt: str, n: int, **sampling_kwargs) -> list[str]`
    - `sample_proof(prompt: str, n: int, **sampling_kwargs) -> list[str]`

Deliverable: a short design note (in this file or a separate doc) stating the exact DLM algorithm, training loss, and sampling schedule.

- **D0.4: Dependency + environment setup**
  - Minimum deps for discrete-token diffusion text models:
    - `transformers`, `accelerate`, `safetensors`
  - Optional deps:
    - `diffusers` (utilities / conventions; not required)
  - Decide precision/runtime:
    - bf16/fp16 inference & training (current repo uses `torch.bfloat16` — see [`SFTTrainer.__init__`](lean_dojo_v2/trainer/sft_trainer.py#L88-L90) `torch_dtype=torch.bfloat16`)
    - optional xFormers / SDPA depending on your denoiser architecture
  - **[Update `pyproject.toml`](pyproject.toml)** to add any new dependencies.

#### Phase 0 — Design Note (completed)

> **Algorithm**: Discrete masked diffusion (absorbing-state), as implemented by the LLaDA family of models.
>
> **Pivot model**: [`inclusionAI/LLaDA-MoE-7B-A1B-Instruct`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) — 7.03B total params, 1.4B active (MoE), 4096 context, vocabulary 157184, mask token id `156895`.
>
> **Training loss**: Masked denoising objective — corrupt target tokens to mask id, train the model to predict original tokens at all masked positions simultaneously (cross-entropy over masked positions). This is the standard LLaDA pre-training/fine-tuning loss.
>
> **Inference / sampling schedule**:
> - Semi-autoregressive block-based generation: output is divided into blocks of `block_length` tokens; each block is denoised in `steps_per_block` iterations.
> - At each iteration: run full forward pass → apply Gumbel noise (temperature-controlled) → argmax → rank candidates by confidence (softmax probability at predicted token) → unmask the top-k most confident tokens (k = remaining_masks / remaining_steps).
> - Remasking strategy: `low_confidence` (default) keeps high-confidence predictions, remasks low-confidence ones; `random` remasks randomly.
> - Classifier-free guidance (optional): `cfg_scale > 0` interpolates between conditional and unconditional logits.
>
> **Defaults**: `steps=128`, `gen_length=128`, `block_length=32`, `temperature=0.0` (greedy), `cfg_scale=0.0` (no guidance), `remasking=low_confidence`.
>
> **Implementation**: `lean_dojo_v2/diffusion/` package with `DiffusionConfig` (dataclass) and `DiffusionSampler` (model loading + generate loop). Integrated via `DiffusionProver` (subclass of `BaseProver`) and `DiffusionAgent` (subclass of `BaseAgent`).

---

### Phase 1 — Next-tactic diffusion inference integrated into proof search (highest priority)

#### Implementation tasks

- **T1.1: Add diffusion module package**
  - Create `lean_dojo_v2/diffusion/` with:
    - `config.py`: dataclasses for model + sampling params.
    - `sampler.py`: the diffusion sampling loop.
    - `tokenizer.py` (optional): any special tokenization / formatting.
    - `modeling.py`: model definition or wrappers around an external library.

- **T1.2: Implement `DiffusionProver`**
  - New file: `lean_dojo_v2/prover/diffusion_prover.py`
  - Subclass [`BaseProver`](lean_dojo_v2/prover/base_prover.py#L17-L277) and implement:
    - `next_tactic(state, goal_id)`:
      - Format prompt from `GoalState` — **reuse prompt template from [`HFProver.next_tactic()`](lean_dojo_v2/prover/hf_prover.py#L51-L62)**:
        ```
        "### System:\n"
        "You are a Lean 4 tactic generator. Given a goal state, ..."
        "### User:\n{goal_str}\n\n"
        "### Assistant:\n"
        ```
      - Call diffusion sampler to get `k` candidate tactics.
      - Post-process to enforce "single line tactic" constraints — **reuse logic from [`HFProver.next_tactic()` L82-L88](lean_dojo_v2/prover/hf_prover.py#L82-L88)**: strip prompt prefix, split on `\n`, split on `<;>`, strip, skip `sorry`.
      - Return one tactic (or select by score if the DLM provides per-sample likelihood/energy).
    - `generate_whole_proof(theorem)`:
      - implement minimal stub by sampling a longer completion — **mirror [`HFProver.generate_whole_proof()`](lean_dojo_v2/prover/hf_prover.py#L93-L127)**.
  - **Register**: add `DiffusionProver` to [`prover/__init__.py`](lean_dojo_v2/prover/__init__.py).

- **T1.3: Add `DiffusionAgent`**
  - New file: `lean_dojo_v2/agent/diffusion_agent.py`
  - Mirror [`ExternalAgent`](lean_dojo_v2/agent/external_agent.py) (20 lines):
    - `_setup_prover()` instantiates `DiffusionProver` with a checkpoint/config.
    - `_get_build_deps()` likely `False` initially (fast iteration), can become configurable.
  - **Register**: add `DiffusionAgent` to [`agent/__init__.py`](lean_dojo_v2/agent/__init__.py).

- **T1.4: Add example scripts**
  - `examples/diffusion_next_tactic.py`: proof search on a simple goal — **mirror [`generate_tactics.py`](examples/theorem_proving/generate_tactics.py)**.
  - `examples/diffusion_agent.py`: run on a traced GitHub repo and try to solve `sorry` theorems — **mirror [`external_agent.py` main()](lean_dojo_v2/agent/external_agent.py#L23-L32)**.

#### Prompting / formatting decisions (make these explicit)

- Use the same behavioral contract as [`SFTDataset`](lean_dojo_v2/trainer/sft_trainer.py#L22-L69) and [`HFProver`](lean_dojo_v2/prover/hf_prover.py):
  - System prompt is identical across both (compare [`SFTDataset._process_data()` L43-L53](lean_dojo_v2/trainer/sft_trainer.py#L43-L53) ↔ [`HFProver.next_tactic()` L52-L59](lean_dojo_v2/prover/hf_prover.py#L52-L59)).
  - Input: string form of the `GoalState` (via [`remove_marks()`](lean_dojo_v2/utils/constants.py#L259-L261) in training; `str(state)` in inference).
  - Output: **exactly one** Lean tactic line (no `by`, no code fences, no explanation).
- Post-processing: hard filters for:
  - empty output
  - multi-line output
  - outputs containing `sorry` / `admit`
  - optional: strip `<;>` tails or truncation artifacts

> [!IMPORTANT]
> **Prompt consistency gap**: The prompt template in [`HFProver.next_tactic()`](lean_dojo_v2/prover/hf_prover.py#L51-L62) uses inline string formatting (`### System:` / `### User:` / `### Assistant:`) rather than the chat-template `messages` format used by [`SFTDataset._process_data()`](lean_dojo_v2/trainer/sft_trainer.py#L39-L65). This means the format the model sees at training time (TRL chat template) can differ from inference time (raw string). The new `DiffusionProver` should choose **one canonical format** and share it with the training pipeline (ideally extract to a shared `formatting.py`).

#### Bidirectionality notes (what DLMs can actually exploit)

Diffusion LMs are naturally **bidirectional** (good at denoising/infilling with both left + right context).
However, *whether this helps* depends on the proving interface:

- **Next-tactic proving (`GoalState -> tactic`)**
  - The model output is a **single next action**, and there is usually **no fixed "future text"** to condition on during interactive proof search.
  - `GoalState` is already a **semantic, normalized** representation of the proof obligation (locals + hypotheses + target), and is typically the best primary conditioning signal for next-tactic generation.
  - If we add extra context, prefer **stable context** that does not change as the proof evolves:
    - theorem statement / theorem name
    - module path/imports
    - (later/optional) retrieved premises (if we extend beyond the non-retrieval scope)

- **Whole-proof / `sorry`-replacement**
  - Bidirectionality is most useful when we reformulate the task as **source-code infilling**:
    - condition on the Lean file **prefix** up to the `sorry`
    - condition on the Lean file **suffix** after the `sorry` (when present/meaningful)
    - generate the missing proof block (possibly multi-line) consistent with both sides
  - This setting gives a true "right context" (indentation/structure/`·` blocks/`match`/`have` scaffolding), where bidirectional denoising should help more than pure left-to-right completion.

#### Testing (definition of working)

- **Unit tests**
  - `test_diffusion_postprocess_single_line()`
  - `test_diffusion_prover_returns_string_or_none()`
  - Existing test location: [`lean_dojo_v2/tests/`](lean_dojo_v2/tests) (currently has [`test_deepseek_loading.py`](lean_dojo_v2/tests/test_deepseek_loading.py) and [`test_leanprogress_examples.py`](lean_dojo_v2/tests/test_leanprogress_examples.py)).

- **Integration tests**
  - Reuse Pantograph search harness:
    - pick a small goal (`∀ {p q : Prop}, p ∧ q → q ∧ p`) and ensure the prover can run end-to-end — **same goal used in [`generate_tactics.py`](examples/theorem_proving/generate_tactics.py#L15)**.
  - Repo-level: run `DiffusionAgent.prove()` on `lean4-example` and verify at least 1 theorem succeeds (or record baseline success=0 but no crashes) — **same repo used in [`ExternalAgent.main()`](lean_dojo_v2/agent/external_agent.py#L28-L29)** (`https://github.com/durant42040/lean4-example`, commit `005de00d`).

#### Evaluation (what to log)

- **Proof success rate**: solved / attempted sorry theorems.
- **Search efficiency**: steps to solve, wall time, tactic failures.
- **Pass@k tactic validity**: fraction of sampled tactics that parse / don't immediately fail (cheap proxy).
- **Ablations**: schedule steps, guidance scale, sampling temperature, candidate count `k`.

---

### Phase 2 — Next-tactic diffusion training pipeline

This repo's current trainable generator objective is **next-tactic**. We should match that first.

#### Data

- Reuse [`DynamicDatabase.export_merged_data()`](lean_dojo_v2/database/dynamic_database.py#L63-L113) which produces merged JSON with `traced_tactics` entries containing:
  - `state_before` (goal state string)
  - `tactic` (target tactic string)
- Current SFT/GRPO datasets enforce single-line tactic targets; diffusion training should do the same.

- **Where this merged/traced tactic data lives (in this repo)**
  - [`DynamicDatabase.export_merged_data()`](lean_dojo_v2/database/dynamic_database.py#L63-L113) writes a merged dataset directory (default: `RAID_DIR/DATA_DIR/merged/` — see [`BaseAgent.__init__`](lean_dojo_v2/agent/base_agent.py#L22-L25) where `self.data_path = Path(os.path.join(RAID_DIR, DATA_DIR, "merged"))`).
  - Proof data is exported by [`_export_proofs()`](lean_dojo_v2/database/dynamic_database.py#L201-L245) which writes JSON splits under:
    - `merged/random/{train,val,test}.json`
    - `merged/novel_premises/{train,val,test}.json`
  - Each theorem item includes (at least):
    - `theorem_statement`
    - `traced_tactics`: list of dicts with `tactic`, `state_before`, `state_after` (plus `annotated_tactic`)
  - It also exports:
    - `merged/corpus.jsonl` (premises) — via [`_merge_corpus()`](lean_dojo_v2/database/dynamic_database.py#L115-L139)
    - `merged/traced_files.jsonl` — via [`_export_traced_files()`](lean_dojo_v2/database/dynamic_database.py#L247-L252)
    - `merged/metadata.json` — via [`_export_metadata()`](lean_dojo_v2/database/dynamic_database.py#L254-L282)
  - **Data model**: Each theorem → [`Theorem`](lean_dojo_v2/database/models/theorems.py#L12-L95) dataclass with `traced_tactics: List[AnnotatedTactic]`. Each [`AnnotatedTactic`](lean_dojo_v2/database/models/annotations.py#L76-L121) has fields: `tactic`, `annotated_tactic`, `state_before`, `state_after`.
  - **Export filter**: [`_export_proofs()`](lean_dojo_v2/database/dynamic_database.py#L209-L229) filters tactics where `state_before != "no goals"` and `"·" not in tactic`.

> [!NOTE]
> The training flow is: `BaseAgent.train()` → `self.trainer.train(repos, database, data_path)` → trainer calls `database.export_merged_data(repos, data_path)` → then loads `data_path/random/train.json` via the Dataset class. See [`SFTTrainer.train()`](lean_dojo_v2/trainer/sft_trainer.py#L118-L148) for the reference flow.

#### Bidirectional infilling dataset (DLM-catered; recommended addition)

To explicitly leverage DLM **bidirectionality**, add a dataset that turns existing proofs into **infilling / denoising** problems.
One simple starting point is tactic-script infilling derived from `traced_tactics` in `merged/*/{train,val,test}.json`.

- **Core idea**: take a proved theorem's tactic sequence, randomly replace some tactics (or contiguous spans) with a hole marker, and train the DLM to recover the missing original content.
  - Example corruption (span-level): `[t1, t2, t3, t4, t5] -> [t1, <HOLE_1>, t5]` with target = `[t2, t3, t4]`.
  - Optionally use `sorry` as a hole marker, but prefer a sentinel like `<HOLE_1>` to avoid teaching the model that emitting Lean's `sorry` is acceptable.
  - **Practical inference note**: when infilling real Lean files that contain `sorry`, we can simply **replace `sorry` with `<HOLE_1>`** (or the appropriate sentinel format), run the infiller, then splice the generated proof back into the file and verify in the repo context.

- **Suggested export format** (new files, parallel to the existing merged export):
  - `merged_infill/random/{train,val,test}.jsonl` (and optionally `novel_premises/`)
  - Each example:
    - `theorem_statement` (optional but often helpful)
    - `corrupted_tactics`: list of strings (tactics + `<HOLE_i>` markers)
    - `targets`: list of `{hole_id, original_span}` where `original_span` is a list of tactic strings
    - Optional metadata: `url`, `commit`, `file_path`, `full_name`, `start`, `end`

- **Corruption policy knobs to log/ablate**
  - hole rate / expected span length (e.g., geometric)
  - number of holes per proof
  - whether holes are single-tactic or multi-tactic spans
  - whether to condition on the *suffix* (keep remaining tactics after the hole) vs "prefix-only"

- **Implementation hook (where to put it)**
  - Add a new exporter alongside/after [`DynamicDatabase.export_merged_data()`](lean_dojo_v2/database/dynamic_database.py#L63-L113) that reads `merged/*/{train,val,test}.json`, performs corruption, and writes `merged_infill/...`.
  - Alternatively implement an `InfillDataset` reader that corrupts on-the-fly from the existing merged JSON (simpler iteration, but less reproducible unless you fix seeds and persist configs).

#### Implementation tasks

- **T2.1: Implement `DiffusionTacticDataset`**
  - Mirror [`SFTDataset`](lean_dojo_v2/trainer/sft_trainer.py#L22-L69) but output the exact tensors/fields your diffusion objective needs.
  - Key reference: [`SFTDataset._process_data()`](lean_dojo_v2/trainer/sft_trainer.py#L32-L66) iterates `item["traced_tactics"]`, skips `sorry`, uses [`remove_marks()`](lean_dojo_v2/utils/constants.py#L259-L261), and formats as `(system, user=state_before, assistant=tactic)`.
  - Keep the prompt template consistent with inference.

- **T2.2: Add a diffusion training package (trainer + objectives)**
  - Create `lean_dojo_v2/diffusion_training/`:
    - `config.py`: dataclasses for objective + corruption schedule + sampling params.
    - `objectives.py`: loss implementations (next-tactic denoising; infill denoising).
    - `corruption.py`: discrete corruption operators (mask/span corruption; hole placement).
    - `formatting.py`: prompt templates shared with `DiffusionProver`.
  - Rationale: diffusion training is not a drop-in replacement for TRL SFT/GRPO; we want a small, explicit training stack that still uses the repo's export format.

- **T2.3: Implement `DiffusionTrainer` (custom loop)**
  - New file: `lean_dojo_v2/trainer/diffusion_trainer.py`
  - Responsibilities:
    - load base diffusion(-style) text model + tokenizer (local PyTorch)
    - implement training step for a discrete denoising objective (masked/span denoise)
    - support bf16/fp16, gradient accumulation, checkpointing, and periodic sampling
    - save checkpoints in a directory consumable by `DiffusionProver`
  - Suggested minimal interface — **mirror [`SFTTrainer`](lean_dojo_v2/trainer/sft_trainer.py#L72-L148)**:
    - `train(repos, database, data_path, objective: str, ...)`
    - `sample(prompt, n, ...)` (for quick sanity checks)
  - **Register**: add `DiffusionTrainer` to [`trainer/__init__.py`](lean_dojo_v2/trainer/__init__.py).

- **T2.4: Add `DiffusionInfillDataset` (tactic-script infilling)**
  - Source: the existing merged exports `merged/*/{train,val,test}.json`.
  - For each theorem:
    - collect `tactic_script = [t["tactic"] for t in traced_tactics]`
    - apply span corruption to create `corrupted_tactics` with `<HOLE_i>` markers
    - define targets either as:
      - **span prediction**: predict each missing span conditioned on full corrupted script (preferred for bidirectionality), or
      - **full reconstruction**: predict the full original script (simpler, but less "targeted")
  - Emit to `merged_infill/...` (offline) or corrupt on-the-fly (online).

- **T2.5: Define two non-retrieval diffusion objectives (v1)**
  - **O1 (next-tactic denoising, SFT-compatible task)**:
    - input: `GoalState` prompt (same as Phase 1 inference)
    - target: a single-line tactic
    - diffusion formulation: treat the tactic as the denoised sequence; corrupt target tokens and denoise conditioned on the goal prompt
    - why: matches the existing proving/search interface and can be evaluated immediately in proof search
  - **O2 (bidirectional infilling / repair, DLM-native task)**:
    - input: a *corrupted* proof script with `<HOLE_i>` markers (or file prefix/suffix around a hole)
    - target: the missing span(s) (or reconstructed full script)
    - why: directly exploits bidirectionality; aligns with "replace `sorry` with `<HOLE>` then infill"

- **T2.6: Add training examples/scripts**
  - `examples/diffusion_train_next_tactic.py`:
    - trace + `export_merged_data`
    - train objective O1 on `merged/random/train.json`
    - **Mirror [`sft.py` example](examples/sft.py) flow.**
  - `examples/diffusion_train_infill.py`:
    - trace + `export_merged_data`
    - build/use `merged_infill/...`
    - train objective O2
  - Keep scripts non-retrieval: do not use premise corpora or retrievers.

- **T2.7: Add quick sanity tests**
  - Unit tests for:
    - deterministic corruption (seeded span corruption)
    - post-processing constraints (single-line tactic for O1)
    - round-trip: corrupt → denoise (teacher-forced) → exact reconstruction on a tiny synthetic sample

#### Training-time evaluation

- **Exact-match (weak)**: predicted tactic == gold tactic (not a great metric, but simple).
- **Execution-based metric (better)**: sample `k` tactics and measure:
  - "applies without error" rate against Pantograph on the recorded `state_before`.
  - average number of new goals / whether it solves the goal.
 - **Infilling metrics (for O2)**:
   - span exact-match / edit distance against the original missing span
   - "proof checks" rate after replacing `<HOLE_i>` with predicted span(s) and running Lean in the repo context (gold-standard but slower)

---

### Phase 3 — Whole-proof diffusion inference (plug-in)

Even before training for whole-proof, you can implement inference plumbing.

#### Implementation tasks

- **T3.1: Add `generate_whole_proof` path**
  - In `DiffusionProver.generate_whole_proof(theorem)`:
    - prompt with theorem statement (consistent with [`HFProver.generate_whole_proof()`](lean_dojo_v2/prover/hf_prover.py#L93-L127) / [`ExternalProver.generate_whole_proof()`](lean_dojo_v2/prover/external_prover.py#L31-L33) style)
    - sample a long output
    - post-process (strip fences/explanations; keep Lean code)

- **T3.1b (recommended DLM-specific variant): `sorry` block infilling**
  - Implement an alternative whole-proof mode that treats proving as **Lean source infilling** rather than pure completion:
    - input: `(file_prefix, file_suffix)` where the hole corresponds to the `sorry` region
    - output: proof script to replace the hole (multi-line allowed)
  - This is the most direct way to leverage **bidirectionality** (prefix + suffix conditioning).
  - Verification still happens by checking the modified file in the repo context.

> [!WARNING]
> **Interface gap**: The current [`BaseProver.generate_whole_proof()`](lean_dojo_v2/prover/base_prover.py#L231-L244) signature is `generate_whole_proof(theorem: Theorem) -> str`. It takes only a `Theorem` object, which has no field for file prefix/suffix context. For `sorry`-block infilling (T3.1b), you would need to either:
> 1. Add optional `prefix`/`suffix` args to `generate_whole_proof()` (breaking the base class signature), or
> 2. Add a separate method like `infill_sorry(prefix, suffix, theorem) -> str` on `DiffusionProver` only, or
> 3. Extend the `Theorem` dataclass to carry source context.
>
> This is a design decision to resolve before implementing Phase 3.

> [!WARNING]
> **Interface gap**: [`BaseAgent.prove_theorem()`](lean_dojo_v2/agent/base_agent.py#L116-L141) calls `self.prover.generate_whole_proof(theorem)` but does **not verify** the generated proof against Lean. It just `print(proof)`. For whole-proof diffusion to be properly evaluated, `prove_theorem` needs a verification step (e.g., insert proof into file and run Lean/Pantograph to check). This is not specific to diffusion — it's a gap in the current pipeline.

- **T3.2: Add a demo**
  - `examples/diffusion_whole_proof.py`:
    - call `DiffusionProver.generate_whole_proof` on a toy theorem.
    - **Mirror [`generate_whole_proof.py`](examples/theorem_proving/generate_whole_proof.py).**
  - Optional: `examples/diffusion_infill_sorry.py`:
    - load a Lean file containing a `sorry`
    - construct `(prefix, suffix)` around `sorry`
    - infill and then verify in a traced repo project

#### Testing

- "No crash" test + sanity checks:
  - non-empty output
  - doesn't contain the theorem restatement (optional)
  - doesn't contain forbidden tokens (e.g. "Here's the proof:")

---

### Phase 4 — Whole-proof diffusion training (longer-term; requires new data)

Current merged exports are tactic-level; whole-proof needs theorem→proof pairs.

Options for building a whole-proof dataset:

- **Option A: Extract proofs from source files**
  - During tracing, store the full `by ...` proof term associated with each theorem.
  - Requires extending the extraction pipeline (Lean instrumentation) to capture proof text spans reliably.
  - **Current data**: [`Theorem`](lean_dojo_v2/database/models/theorems.py#L12-L95) has `start`/`end` `Pos` and `file_path`, so **the source span is locatable** but not currently extracted as a string. The tracing pipeline would need to also read the source file and extract `file_content[start:end]`.

- **Option B: Replay tactic traces into full proofs**
  - Many traced datasets include sequences of tactics; if you can reconstruct a tactic script, you can build a "proof" target as a `by` block.
  - Caveat: proof scripts may require semicolons, `all_goals`, indentation, etc.
  - **Available data**: [`_export_proofs()`](lean_dojo_v2/database/dynamic_database.py#L201-L245) already assembles `traced_tactics` per theorem. The `tactic` field of each [`AnnotatedTactic`](lean_dojo_v2/database/models/annotations.py#L76-L121) is a single tactic string. Joining them as `by\n  t1\n  t2\n  ...` is a starting point but may not always be valid Lean.

> [!IMPORTANT]
> **Filter gap**: [`_export_proofs()`](lean_dojo_v2/database/dynamic_database.py#L209-L229) currently **filters out** tactics containing `"·"` (focused dots). This means multi-goal proofs that use focusing will have incomplete tactic scripts. For whole-proof reconstruction (Option B), you may need to **relax or modify** this filter, or handle `·`-focused tactics during reconstruction.

- **Option C: Mine existing Lean repos**
  - Collect theorem statements + proof bodies from non-`sorry` theorems.
  - Ensure imports/context are preserved enough for checking.

Training task definition:
- Input: theorem statement (+ optional context/imports)
- Output: Lean code proof term
- Evaluation: "proof checks" via Lean/Pantograph in a controlled environment

---

### Identified gaps and open questions

> [!CAUTION]
> **Summary of gaps found during code review:**
>
> 1. **Prompt format mismatch** (Phase 1/2): `HFProver` uses raw `### System:/User:/Assistant:` string formatting, while `SFTDataset` uses structured `messages` dicts passed through TRL's chat template. The diffusion pipeline must pick one canonical format and share it across training and inference. **Recommended**: extract a shared `formatting.py` module.
>
> 2. **No `sorry` tactic exclusion utility** (Phase 1/2): Post-processing logic (skip `sorry`, split `\n`, split `<;>`) is duplicated between `HFProver.next_tactic()` and would need to be copied again for `DiffusionProver`. **Recommended**: extract to a shared utility function in `lean_dojo_v2/utils/`.
>
> 3. **No whole-proof verification** (Phase 3): `BaseAgent.prove_theorem()` with `whole_proof=True` just prints the proof — no Lean checking. This gap affects evaluation of both existing and new provers. **Recommended**: add Pantograph verification in `prove_theorem()`.
>
> 4. **`generate_whole_proof` lacks context args** (Phase 3): The abstract method takes only `Theorem`, but infilling needs `(prefix, suffix)`. Design decision needed.
>
> 5. **`·`-focused tactics filtered out** (Phase 4): `_export_proofs()` drops tactics containing `·`, making tactic-script reconstruction incomplete for multi-goal proofs.
>
> 6. **No source-level proof text in data model** (Phase 4): `Theorem` has position info (`start`, `end`, `file_path`) but not the actual proof body text. Extracting this would require reading source files during tracing/export.
>
> 7. **`HFAgent` couples trainer + prover** (general): `HFAgent.__init__` takes a `SFTTrainer` and later creates `HFProver` from the trainer's output dir. A `DiffusionAgent` that supports both train-then-prove and standalone inference may want a more flexible constructor (accept either a trainer or a checkpoint path).

---

### Explicit non-goals (to keep the DLM project focused)

- **No premise retrieval / RAG in this plan**
  - Not building premise indices.
  - Not training dense retrievers.
  - Not conditioning diffusion models on retrieved premise blocks (for now).
- **No LeanAgent pipeline work**
  - We will not modify [`LeanAgent`](lean_dojo_v2/agent/lean_agent.py), [`RetrievalTrainer`](lean_dojo_v2/trainer/retrieval_trainer.py), or [`RetrievalProver`](lean_dojo_v2/prover/retrieval_prover.py) as part of the DLM plan.

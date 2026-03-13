#!/usr/bin/env python3
"""Distributed theorem-level proof search evaluation using Modal.

This script evaluates proof search performance in parallel across multiple Modal
GPU workers. Each theorem is evaluated independently, allowing for massive
parallelization of the proof search process.

Usage:
    modal run scripts/eval_proof_search_modal.py --data-json /path/to/theorems.json \
        --model-type diffusion --ckpt /path/to/checkpoint

Or with Python directly:
    python scripts/eval_proof_search_modal.py --data-json /path/to/theorems.json \
        --model-type diffusion --ckpt /path/to/checkpoint
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Modal App Configuration
# ---------------------------------------------------------------------------

app = modal.App("leandojo-proof-search-eval")

# Create a persistent volume for caching models and outputs
cache_volume = modal.Volume.from_name("leandojo-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("leandojo-outputs", create_if_missing=True)

# Define the container image with all dependencies
# Uses the project's pyproject.toml for consistent dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "curl",
        "wget",
        "build-essential",
    )
    # Install Lean toolchain (elan + lean)
    .run_commands(
        "curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y",
        "echo 'export PATH=\"$HOME/.elan/bin:$PATH\"' >> ~/.bashrc",
    )
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.54,<5",
        "peft>=0.17.0",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
    )
    # Install pantograph from GitHub
    .pip_install("pantograph", extra_options="--find-links https://github.com/stanford-centaur/PyPantograph")
    # Install lean-dojo-v2 from local (assumes it's uploaded or available)
    # For production, you'd want to pip install from the repo
    .workdir("/workspace")
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TheoremItem:
    """Serializable theorem item for Modal transport."""
    index: int
    full_name: str
    theorem_statement: str | None
    traced_tactics: list[dict] | None


def extract_goal_expr(theorem_statement: str) -> str:
    """Extract the goal expression from a theorem statement."""
    s = (theorem_statement or "").strip()
    if not s:
        return ""

    if s.startswith("⊢"):
        return s[1:].strip()

    m = re.search(r":\s*(.*?)\s*:=\s*by\s*$", s, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"\b(?:theorem|lemma)\b.*?:\s*(.*)$", s, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    return s


def extract_goal_from_state(state: str) -> str:
    """Extract goal from a proof state string."""
    s = (state or "").strip()
    if not s:
        return ""
    for line in reversed(s.splitlines()):
        line = line.strip()
        if line.startswith("⊢"):
            return line[1:].strip()
    return ""


def extract_goal_from_item(item: TheoremItem) -> str:
    """Extract the goal from a theorem item using multiple fallback strategies."""
    # Primary path: theorem statement text.
    goal = extract_goal_expr(item.theorem_statement or "")
    if goal:
        return goal

    # APRIL-style fallback: first traced tactic usually carries state_before with a goal line.
    if isinstance(item.traced_tactics, list):
        for step in item.traced_tactics:
            if not isinstance(step, dict):
                continue
            goal = extract_goal_from_state(step.get("state_before") or "")
            if goal:
                return goal

    return ""


# ---------------------------------------------------------------------------
# Modal Worker Class
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu="A10G",  # Use A10G for cost-effective inference
    timeout=600,  # 10 minutes per theorem
    volumes={
        "/cache": cache_volume,
        "/outputs": output_volume,
    },
)
class ProofSearchWorker:
    """Modal class that manages prover and Pantograph server lifecycle."""

    model_type: str = modal.parameter()
    ckpt: str = modal.parameter()
    device: str = modal.parameter(default="cuda")
    use_lora: bool = modal.parameter(default=False)
    imports_csv: str = modal.parameter(default="Init,Mathlib")
    project_path: str = modal.parameter(default="")
    server_timeout: int = modal.parameter(default=300)

    @modal.enter()
    def setup(self):
        """Initialize model and Pantograph server once per container.

        This runs when the container starts and caches the loaded model
        and server for all subsequent theorem evaluations.
        """
        import torch
        from pantograph import Server

        # Initialize instance attributes (can't use __init__ with modal.parameter)
        self._prover = None
        self._server = None

        # Set up Lean environment
        os.environ["PATH"] = os.path.expanduser("~/.elan/bin") + ":" + os.environ.get("PATH", "")

        print(f"[{self.model_type}] Loading model from {self.ckpt}...")

        # Import prover classes
        if self.model_type == "diffusion":
            from lean_dojo_v2.prover.diffusion_prover import DiffusionProver
            self._prover = DiffusionProver(
                ckpt_path=self.ckpt,
                use_lora=self.use_lora,
                device=self.device,
            )
        elif self.model_type == "hf":
            from lean_dojo_v2.prover.hf_prover import HFProver
            self._prover = HFProver(
                ckpt_path=self.ckpt,
                use_lora=self.use_lora,
                device=self.device,
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        print(f"[{self.model_type}] Model loaded successfully")

        # Initialize Pantograph server
        self._imports = [s.strip() for s in self.imports_csv.split(",") if s.strip()]
        if not self._imports:
            self._imports = ["Init", "Mathlib"]
        self._project_path = self.project_path or None

        print(f"Starting Pantograph server with imports: {self._imports}")
        self._server = Server(
            imports=self._imports,
            project_path=self._project_path,
            timeout=self.server_timeout,
        )
        print("Pantograph server ready")

    @modal.method()
    def evaluate_theorem(self, item: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a single theorem's proof search.

        Args:
            item: Theorem item with keys like 'index', 'full_name', 'theorem_statement', etc.

        Returns:
            Dict containing evaluation results including success status, steps, tactics, timing.
        """
        theorem_item = TheoremItem(
            index=item.get("index", 0),
            full_name=item.get("full_name", f"theorem_{item.get('index', 0)}"),
            theorem_statement=item.get("theorem_statement"),
            traced_tactics=item.get("traced_tactics"),
        )

        goal = extract_goal_from_item(theorem_item)
        name = theorem_item.full_name

        # Handle parse failure
        if not goal:
            return {
                "index": theorem_item.index,
                "attempted_index": None,
                "name": name,
                "goal": "",
                "status": "parse_failed",
                "success": False,
                "steps": None,
                "used_tactics": [],
                "model_tactics": [],
                "error": None,
                "elapsed_sec": None,
            }

        # Set up tactic logging
        theorem_tactics: list[dict[str, Any]] = []
        original_next_tactic = self._prover.next_tactic

        def logging_next_tactic(state, goal_id):
            tactic = original_next_tactic(state, goal_id)
            theorem_tactics.append({
                "goal_id": goal_id,
                "goal_state": str(state),
                "tactic": str(tactic) if tactic is not None else None,
            })
            return tactic

        self._prover.next_tactic = logging_next_tactic
        theorem_t0 = time.time()

        # Non-None sentinel so next_tactic() runs in current prover implementations.
        theorem_sentinel = object()

        try:
            result, used_tactics = self._prover.search(
                server=self._server,
                goal=goal,
                theorem=theorem_sentinel,
                verbose=False,
            )
            ok = bool(result.success)
            elapsed = round(time.time() - theorem_t0, 6)

            return {
                "index": theorem_item.index,
                "attempted_index": item.get("attempted_index"),
                "name": name,
                "goal": goal,
                "status": "ok",
                "success": ok,
                "steps": result.steps,
                "used_tactics": used_tactics or [],
                "model_tactics": theorem_tactics,
                "error": None,
                "elapsed_sec": elapsed,
            }
        except Exception as e:
            elapsed = round(time.time() - theorem_t0, 6)
            return {
                "index": theorem_item.index,
                "attempted_index": item.get("attempted_index"),
                "name": name,
                "goal": goal,
                "status": "runtime_error",
                "success": False,
                "steps": None,
                "used_tactics": [],
                "model_tactics": theorem_tactics,
                "error": {"type": type(e).__name__, "message": str(e)},
                "elapsed_sec": elapsed,
            }
        finally:
            # Restore original method
            self._prover.next_tactic = original_next_tactic


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    data_json: str = "datasets/april/leandojo/test/thme_test.json",
    model_type: str = "diffusion",
    ckpt: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
    device: str = "auto",
    use_lora: bool = False,
    imports: str = "Init,Mathlib",
    project_path: str = "",
    server_timeout: int = 300,
    max_theorems: int = 0,
    results_jsonl: str = "",
    concurrency: int = 10,
):
    """Run distributed proof search evaluation on Modal.

    Args:
        data_json: Path to LeanDojo-style JSON file with theorems
        model_type: Type of prover ('diffusion' or 'hf')
        ckpt: Path to model checkpoint or HuggingFace model name
        device: Device for inference ('cuda', 'cpu', or 'auto')
        use_lora: Whether to use LoRA weights
        imports: Comma-separated Lean imports for Pantograph server
        project_path: Lean project root for import resolution (empty string = none)
        server_timeout: Pantograph server timeout in seconds
        max_theorems: Maximum number of theorems to evaluate (0 for all)
        results_jsonl: Path to write results JSONL file (empty string to disable)
        concurrency: Number of parallel containers to run
    """
    imports_list = [s.strip() for s in imports.split(",") if s.strip()]
    if not imports_list:
        imports_list = ["Init", "Mathlib"]
    project_path_value = project_path or None
    max_theorems_value = max_theorems if max_theorems > 0 else None
    results_jsonl_value = results_jsonl or None

    data_path = Path(data_json)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_json}")
        raise FileNotFoundError(f"Data file not found: {data_json}")

    print(f"Loading theorems from {data_json}...")
    data: list[dict[str, Any]] = json.loads(data_path.read_text(encoding="utf-8"))

    if max_theorems_value is not None:
        data = data[:max_theorems_value]

    print(f"Loaded {len(data)} theorems")
    print(f"Model type: {model_type}")
    print(f"Checkpoint: {ckpt}")
    print(f"Concurrency: {concurrency} parallel workers")

    # Initialize the worker class
    worker = ProofSearchWorker(
        model_type=model_type,
        ckpt=ckpt,
        device=device,
        use_lora=use_lora,
        imports_csv=",".join(imports_list),
        project_path=project_path_value or "",
        server_timeout=server_timeout,
    )

    # Run distributed evaluation
    t0 = time.time()
    print(f"\nStarting distributed evaluation at {datetime.now(timezone.utc).isoformat()}")

    # Use .map() for parallel execution with controlled concurrency
    results = list(worker.evaluate_theorem.map(
        data,
        order_outputs=True,  # Maintain input order in results
    ))

    elapsed = time.time() - t0

    # Aggregate results
    attempted = 0
    succeeded = 0
    parse_failed = 0
    runtime_failed = 0

    for i, result in enumerate(results):
        result["index"] = i  # Ensure correct indexing
        if result["status"] == "parse_failed":
            parse_failed += 1
        elif result["status"] == "runtime_error":
            runtime_failed += 1
            attempted += 1
        else:
            attempted += 1
            if result["success"]:
                succeeded += 1

    success_rate = (succeeded / attempted) if attempted else 0.0

    # Print summary
    print("\n" + "=" * 50)
    print("Proof Search Summary (Modal Distributed)")
    print("=" * 50)
    print(f"model_type={model_type}")
    print(f"ckpt={ckpt}")
    print(f"total_theorems={len(data)}")
    print(f"attempted={attempted}")
    print(f"succeeded={succeeded}")
    print(f"success_rate={success_rate:.4f}")
    print(f"parse_failed={parse_failed}")
    print(f"runtime_failed={runtime_failed}")
    print(f"elapsed_sec={elapsed:.2f}")
    print(f"avg_time_per_theorem={elapsed/max(attempted, 1):.2f}s")
    print("=" * 50)

    # Save results
    if results_jsonl_value is not None:
        results_path = Path(results_jsonl_value)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        run_info = {
            "run_type": "proof_search_eval_modal",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_type": model_type,
            "ckpt": ckpt,
            "data_json": str(data_json),
            "project_path": project_path_value,
            "imports": imports_list,
            "server_timeout": server_timeout,
            "max_theorems": max_theorems_value,
            "concurrency": concurrency,
            "summary": {
                "total": len(data),
                "attempted": attempted,
                "succeeded": succeeded,
                "success_rate": success_rate,
                "parse_failed": parse_failed,
                "runtime_failed": runtime_failed,
                "elapsed_sec": elapsed,
            },
        }

        with results_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"meta": run_info}, ensure_ascii=False) + "\n")
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"\nResults saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# CLI Entry Point (for local testing without Modal CLI)
# ---------------------------------------------------------------------------

def cli_main():
    """Command-line interface for local execution."""
    parser = argparse.ArgumentParser(
        description="Distributed proof search evaluation using Modal"
    )
    parser.add_argument(
        "--data-json",
        type=Path,
        default=Path("datasets/april/leandojo/test/thme_test.json"),
        help="Path to LeanDojo test JSON file",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["diffusion", "hf"],
        required=True,
        help="Which prover backend to evaluate",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA weights")
    parser.add_argument(
        "--imports",
        nargs="+",
        default=["Init", "Mathlib"],
        help="Imports to initialize Pantograph server",
    )
    parser.add_argument(
        "--project-path",
        type=str,
        default=None,
        help="Lean project root for import resolution",
    )
    parser.add_argument("--server-timeout", type=int, default=300, help="Server timeout")
    parser.add_argument(
        "--max-theorems",
        type=int,
        default=None,
        help="Maximum theorems to evaluate",
    )
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=None,
        help="Path to write results JSONL file",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of parallel Modal workers",
    )
    args = parser.parse_args()

    # When running locally (not via `modal run`), we need to handle differently
    if modal.is_local():
        print("Running in local mode - launching Modal remotely...")
        main(
            data_json=str(args.data_json),
            model_type=args.model_type,
            ckpt=args.ckpt,
            device=args.device,
            use_lora=args.use_lora,
            imports=",".join(args.imports),
            project_path=args.project_path or "",
            server_timeout=args.server_timeout,
            max_theorems=args.max_theorems or 0,
            results_jsonl=str(args.results_jsonl) if args.results_jsonl else "",
            concurrency=args.concurrency,
        )
    else:
        print("Already running in Modal container")


if __name__ == "__main__":
    cli_main()

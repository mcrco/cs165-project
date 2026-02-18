"""
Usage:
python examples/theorem_proving/diffusion_eval.py \
  --url https://github.com/durant42040/lean4-example \
  --commit b14fef0ceca29a65bc3122bf730406b33c7effe5 \
  --ckpt-path inclusionAI/LLaDA-MoE-7B-A1B-Instruct

Evaluate DiffusionProver on `sorry` theorems from a traced Lean repository.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pantograph import Server

from lean_dojo_v2.database import DynamicDatabase
from lean_dojo_v2.lean_dojo.data_extraction.lean import LeanGitRepo
from lean_dojo_v2.lean_dojo.data_extraction.trace import get_traced_repo_path
from lean_dojo_v2.prover.diffusion_prover import DiffusionProver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DiffusionProver on a traced Lean repository."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://github.com/durant42040/lean4-example",
        help="Repository URL.",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default="b14fef0ceca29a65bc3122bf730406b33c7effe5",
        help="Repository commit.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        help="Diffusion model checkpoint path or HF model id.",
    )
    parser.add_argument(
        "--database-path",
        type=str,
        default="diffusion_eval_db.json",
        help="Path to DynamicDatabase json file.",
    )
    parser.add_argument(
        "--build-deps",
        action="store_true",
        help="Whether to build Lean dependencies while tracing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for the model (e.g. auto, cuda, cpu).",
    )
    parser.add_argument(
        "--max-theorems",
        type=int,
        default=0,
        help="Max number of sorry theorems to evaluate (0 = all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for theorem subsampling order.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="",
        help="Optional output path to write per-theorem results as JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    db = DynamicDatabase(json_path=args.database_path)
    repo = db.trace_repository(args.url, args.commit, build_deps=args.build_deps)
    db.add_repository(repo)

    sorry_theorems = list(repo.sorry_theorems_unproved)
    if not sorry_theorems:
        print("No sorry theorems found in this repository.")
        return

    if args.max_theorems > 0:
        random.shuffle(sorry_theorems)
        sorry_theorems = sorry_theorems[: args.max_theorems]

    prover = DiffusionProver(ckpt_path=args.ckpt_path, device=args.device)

    lean_repo = LeanGitRepo(repo.url, repo.commit)
    traced_repo_path = get_traced_repo_path(lean_repo, build_deps=args.build_deps)

    output_path = Path(args.output_jsonl) if args.output_jsonl else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    num_success = 0
    total_duration = 0.0
    total_steps = 0

    for i, theorem in enumerate(sorry_theorems, start=1):
        server = Server(
            imports=["Init", str(theorem.file_path).replace(".lean", "")],
            project_path=traced_repo_path,
        )

        result, used_tactics = prover.search(
            server=server, theorem=theorem, verbose=False
        )
        success = bool(result.success)
        steps = len(used_tactics) if used_tactics else 0

        if success:
            num_success += 1
        total_duration += float(result.duration)
        total_steps += steps

        row = {
            "theorem": theorem.full_name,
            "success": success,
            "duration_sec": float(result.duration),
            "steps": steps,
            "tactics": used_tactics or [],
        }

        print(
            f"[{i}/{len(sorry_theorems)}] "
            f"{theorem.full_name} success={success} "
            f"duration={row['duration_sec']:.2f}s steps={steps}"
        )

        if output_path is not None:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    n = len(sorry_theorems)
    solve_rate = num_success / n
    avg_duration = total_duration / n
    avg_steps = total_steps / n

    print("\nSummary")
    print(f"Theorems: {n}")
    print(f"Solved: {num_success}")
    print(f"Solve rate: {solve_rate:.3f}")
    print(f"Avg duration (s): {avg_duration:.2f}")
    print(f"Avg steps: {avg_steps:.2f}")


if __name__ == "__main__":
    main()

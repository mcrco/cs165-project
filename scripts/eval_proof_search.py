#!/usr/bin/env python3
"""Evaluate theorem-level proof search on LeanDojo-style JSON data.

This runs true proof search (iterative next-tactic generation + tactic execution in
Pantograph) and reports theorem-level success rate.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from pantograph import Server

from lean_dojo_v2.prover.diffusion_prover import DiffusionProver
from lean_dojo_v2.prover.hf_prover import HFProver

REPO_DIR = Path(__file__).resolve().parents[1]
APRIL_DIR = REPO_DIR / "datasets" / "april"
DEFAULT_DATA_JSON = APRIL_DIR / "leandojo" / "val.json"
DEFAULT_PROJECT_PATH = APRIL_DIR / "april_eval_project"


def extract_goal_expr(theorem_statement: str) -> str:
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


def build_prover(model_type: str, ckpt: str, device: str, use_lora: bool):
    if model_type == "diffusion":
        return DiffusionProver(ckpt_path=ckpt, device=device)
    if model_type == "hf":
        return HFProver(ckpt_path=ckpt, use_lora=use_lora, device=device)
    raise ValueError(f"Unsupported model_type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-json",
        type=Path,
        default=DEFAULT_DATA_JSON,
        help=f"LeanDojo val JSON (default: {DEFAULT_DATA_JSON})",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["diffusion", "hf"],
        required=True,
        help="Which prover backend to evaluate",
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument(
        "--imports",
        nargs="+",
        default=["Init", "Mathlib"],
        help="Imports to initialize Pantograph server",
    )
    parser.add_argument(
        "--project-path",
        type=Path,
        default=DEFAULT_PROJECT_PATH,
        help=f"Lean project root for import resolution (default: {DEFAULT_PROJECT_PATH})",
    )
    parser.add_argument("--server-timeout", type=int, default=60)
    parser.add_argument("--max-theorems", type=int, default=None)
    args = parser.parse_args()

    data: list[dict[str, Any]] = json.loads(args.data_json.read_text(encoding="utf-8"))
    if args.max_theorems is not None:
        data = data[: args.max_theorems]

    prover = build_prover(args.model_type, args.ckpt, args.device, args.use_lora)

    server = Server(
        imports=args.imports,
        project_path=str(args.project_path) if args.project_path else None,
        timeout=args.server_timeout,
    )

    attempted = 0
    succeeded = 0
    parse_failed = 0
    runtime_failed = 0

    # Non-None sentinel so next_tactic() runs in current prover implementations.
    theorem_sentinel = object()

    t0 = time.time()
    for i, item in enumerate(data):
        goal = extract_goal_expr(item.get("theorem_statement") or "")
        if not goal:
            parse_failed += 1
            continue

        attempted += 1
        name = item.get("full_name") or f"theorem_{i}"

        try:
            result, used_tactics = prover.search(
                server=server,
                goal=goal,
                theorem=theorem_sentinel,
                verbose=False,
            )
            ok = bool(result.success)
            if ok:
                succeeded += 1
            print(
                f"[{attempted}] {name}: success={ok} "
                f"steps={result.steps} tactics_used={len(used_tactics or [])}"
            )
        except Exception as e:
            runtime_failed += 1
            print(f"[{attempted}] {name}: runtime_error={type(e).__name__}: {e}")

    elapsed = time.time() - t0
    success_rate = (succeeded / attempted) if attempted else 0.0

    print("\n=== proof-search summary ===")
    print(f"model_type={args.model_type}")
    print(f"ckpt={args.ckpt}")
    print(f"attempted={attempted}")
    print(f"succeeded={succeeded}")
    print(f"success_rate={success_rate:.4f}")
    print(f"parse_failed={parse_failed}")
    print(f"runtime_failed={runtime_failed}")
    print(f"elapsed_sec={elapsed:.2f}")


if __name__ == "__main__":
    main()

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pantograph import Server

from lean_dojo_v2.prover.diffusion_prover import DiffusionProver
from lean_dojo_v2.prover.hf_prover import HFProver

REPO_DIR = Path(__file__).resolve().parents[1]
APRIL_DIR = REPO_DIR / "datasets" / "april"
DEFAULT_DATA_JSON = APRIL_DIR / "leandojo" / "test/thme_test.json"
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


def extract_goal_from_state(state: str) -> str:
    s = (state or "").strip()
    if not s:
        return ""
    for line in reversed(s.splitlines()):
        line = line.strip()
        if line.startswith("⊢"):
            return line[1:].strip()
    return ""


def extract_goal_from_item(item: dict[str, Any]) -> str:
    # Primary path: theorem statement text.
    goal = extract_goal_expr(item.get("theorem_statement") or "")
    if goal:
        return goal

    # APRIL-style fallback: first traced tactic usually carries state_before with a goal line.
    traced = item.get("traced_tactics")
    if isinstance(traced, list):
        for step in traced:
            if not isinstance(step, dict):
                continue
            goal = extract_goal_from_state(step.get("state_before") or "")
            if goal:
                return goal

    return ""


def build_prover(model_type: str, ckpt: str, device: str, use_lora: bool):
    if model_type == "diffusion":
        return DiffusionProver(ckpt_path=ckpt, use_lora=use_lora, device=device)
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
    parser.add_argument("--server-timeout", type=int, default=300)
    parser.add_argument("--max-theorems", type=int, default=None)
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=None,
        help="Optional path to write per-theorem proof-search results as JSONL",
    )
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

    results_path: Path | None = args.results_jsonl
    result_rows: list[dict[str, Any]] = []

    # Non-None sentinel so next_tactic() runs in current prover implementations.
    theorem_sentinel = object()

    t0 = time.time()
    for i, item in enumerate(data):
        goal = extract_goal_from_item(item)
        name = item.get("full_name") or f"theorem_{i}"

        if not goal:
            parse_failed += 1
            result_rows.append(
                {
                    "index": i,
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
            )
            continue

        attempted += 1
        theorem_tactics: list[dict[str, Any]] = []
        original_next_tactic = prover.next_tactic

        def logging_next_tactic(state, goal_id):
            tactic = original_next_tactic(state, goal_id)
            theorem_tactics.append(
                {
                    "goal_id": goal_id,
                    "goal_state": str(state),
                    "tactic": str(tactic) if tactic is not None else None,
                }
            )
            return tactic

        prover.next_tactic = logging_next_tactic
        theorem_t0 = time.time()

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
            result_rows.append(
                {
                    "index": i,
                    "attempted_index": attempted,
                    "name": name,
                    "goal": goal,
                    "status": "ok",
                    "success": ok,
                    "steps": result.steps,
                    "used_tactics": used_tactics or [],
                    "model_tactics": theorem_tactics,
                    "error": None,
                    "elapsed_sec": round(time.time() - theorem_t0, 6),
                }
            )
        except Exception as e:
            runtime_failed += 1
            print(f"[{attempted}] {name}: runtime_error={type(e).__name__}: {e}")
            result_rows.append(
                {
                    "index": i,
                    "attempted_index": attempted,
                    "name": name,
                    "goal": goal,
                    "status": "runtime_error",
                    "success": False,
                    "steps": None,
                    "used_tactics": [],
                    "model_tactics": theorem_tactics,
                    "error": {"type": type(e).__name__, "message": str(e)},
                    "elapsed_sec": round(time.time() - theorem_t0, 6),
                }
            )
        finally:
            prover.next_tactic = original_next_tactic

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

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        run_info = {
            "run_type": "proof_search_eval",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_type": args.model_type,
            "ckpt": args.ckpt,
            "data_json": str(args.data_json),
            "project_path": str(args.project_path),
            "imports": args.imports,
            "server_timeout": args.server_timeout,
            "max_theorems": args.max_theorems,
        }
        with results_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"meta": run_info}, ensure_ascii=False) + "\n")
            for row in result_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"results_jsonl={results_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert APRIL rows into LeanDojo-compatible traced-tactics JSON.

Expected input: JSONL where each row has at least:
- path
- theorem
- correct_proof

Output format matches the JSON consumed by DiffusionSFTDataset:
[
  {
    "url": "...",
    "commit": "...",
    "file_path": "...",
    "full_name": "...",
    "theorem_statement": "...",
    "start": [1, 1],
    "end": [1, 1],
    "traced_tactics": [
      {
        "tactic": "...",
        "annotated_tactic": ["...", []],
        "state_before": "...",
        "state_after": "..."
      }
    ]
  }
]
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from pantograph import Server

DEFAULT_URL = "https://huggingface.co/datasets/uw-math-ai/APRIL"
APRIL_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_JSONL = APRIL_DIR / "raw" / "val" / "mlme_val.jsonl"
DEFAULT_OUTPUT_JSON = APRIL_DIR / "leandojo" / "val.json"
DEFAULT_PROJECT_PATH = APRIL_DIR / "april_eval_project"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def infer_full_name(row: dict[str, Any], idx: int) -> str:
    theorem_name = row.get("theorem")
    if theorem_name and isinstance(theorem_name, str):
        return theorem_name

    code = row.get("correct_proof", "")
    if isinstance(code, str):
        m = re.search(r"\\b(?:theorem|lemma)\\s+([A-Za-z0-9_'.]+)", code)
        if m:
            return m.group(1)

    return f"april_{idx}"


def infer_statement(code: str) -> str:
    m = re.search(
        r"\\b(?:theorem|lemma)\\s+[A-Za-z0-9_'.]+\\s*(.*?)\\s*:=\\s*by",
        code,
        flags=re.DOTALL,
    )
    if m:
        return re.sub(r"\\s+", " ", m.group(1)).strip()
    return ""


def trace_proof(server: Server, code: str, file_name: Path) -> list[dict[str, Any]]:
    file_name.write_text(code, encoding="utf-8")
    units = server.tactic_invocations(file_name)

    traced_tactics: list[dict[str, Any]] = []
    for unit in units:
        if not unit.invocations:
            continue
        for inv in unit.invocations:
            tactic = (inv.tactic or "").strip()
            state_before = (inv.before or "").strip()
            state_after = (inv.after or "").strip()

            if not tactic or tactic == "sorry":
                continue
            if state_before == "no goals" or "·" in tactic:
                continue

            traced_tactics.append(
                {
                    "tactic": tactic,
                    "annotated_tactic": [tactic, []],
                    "state_before": state_before,
                    "state_after": state_after,
                }
            )
    return traced_tactics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=DEFAULT_INPUT_JSONL,
        help=f"APRIL source JSONL (default: {DEFAULT_INPUT_JSONL})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"LeanDojo output JSON (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--imports",
        nargs="+",
        default=["Mathlib"],
        help="Pantograph imports for tracing (default: Mathlib)",
    )
    parser.add_argument(
        "--project-path",
        type=Path,
        default=DEFAULT_PROJECT_PATH,
        help=f"Lean project root to resolve imports/dependencies (default: {DEFAULT_PROJECT_PATH})",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap for quick pilot runs",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    server = Server(
        imports=args.imports,
        project_path=str(args.project_path) if args.project_path else None,
    )

    converted: list[dict[str, Any]] = []
    num_failed = 0

    with tempfile.TemporaryDirectory(prefix="april_trace_") as d:
        tmp_dir = Path(d)
        for i, row in enumerate(rows):
            code = row.get("correct_proof")
            if not isinstance(code, str) or not code.strip():
                num_failed += 1
                continue

            tmp_file = tmp_dir / f"ex_{i}.lean"
            try:
                traced_tactics = trace_proof(server, code, tmp_file)
            except Exception:
                num_failed += 1
                continue

            if not traced_tactics:
                num_failed += 1
                continue

            full_name = infer_full_name(row, i)
            theorem_statement = infer_statement(code)
            file_path = str(row.get("path") or f"APRIL/{i}.lean")

            converted.append(
                {
                    "url": DEFAULT_URL,
                    "commit": row.get("src_commit") or "",
                    "file_path": file_path,
                    "full_name": full_name,
                    "theorem_statement": theorem_statement,
                    "start": [1, 1],
                    "end": [1, 1],
                    "traced_tactics": traced_tactics,
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total = len(rows)
    ok = len(converted)
    print(f"Converted {ok}/{total} rows. Failed/no tactics: {num_failed}")
    print(f"Wrote: {args.output_json}")


if __name__ == "__main__":
    main()

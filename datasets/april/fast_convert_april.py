#!/usr/bin/env python3
"""Fast text-based converter: APRIL raw JSONL -> LeanDojo-compatible JSON.

Unlike convert_april_to_leandojo.py (which runs `lake env lean` per example
and takes ~35s each), this script parses the correct_proof text directly to
extract (goal_state, tactic) pairs.  The trade-off is that we approximate
`state_before` from the theorem signature rather than getting exact
type-checked Lean goal states.  For single-tactic proofs (~35% of data)
the approximation is very close; for multi-tactic proofs we only extract
the first step with the initial goal.

Produces output compatible with DiffusionSFTDataset.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_URL = "https://huggingface.co/datasets/uw-math-ai/APRIL"


def parse_theorem_signature(code: str) -> tuple[str, str, list[str]] | None:
    """Extract (theorem_name, goal_state_approx, tactic_lines) from a correct_proof."""
    m = re.search(
        r"\b(?:theorem|lemma)\s+([A-Za-z0-9_'.]+)\s*(.*?)\s*:=\s*by\b",
        code,
        flags=re.DOTALL,
    )
    if not m:
        return None

    theorem_name = m.group(1)
    signature_body = m.group(2).strip()

    by_split = code.split(":= by", 1)
    if len(by_split) < 2:
        return None
    tactic_block = by_split[1]
    tactic_lines = extract_tactic_lines(tactic_block)
    if not tactic_lines:
        return None

    goal_state = build_goal_state(code, signature_body)
    return theorem_name, goal_state, tactic_lines


def extract_tactic_lines(tactic_block: str) -> list[str]:
    """Split a tactic block into individual tactic strings.

    Handles indented sub-blocks by joining continuation lines.
    """
    raw_lines = tactic_block.split("\n")
    tactics: list[str] = []
    current: list[str] = []
    base_indent: int | None = None

    DECL_KEYWORDS = {"theorem", "lemma", "def", "noncomputable", "instance",
                      "class", "structure", "inductive", "namespace", "section",
                      "end", "import", "open", "variable", "set_option", "#"}

    for line in raw_lines:
        stripped = line.rstrip()
        if not stripped:
            continue
        trimmed = stripped.lstrip()

        if trimmed.startswith("--"):
            continue

        indent = len(line) - len(line.lstrip())
        if base_indent is None:
            base_indent = indent

        first_word = trimmed.split()[0] if trimmed.split() else ""
        if indent <= base_indent and first_word in DECL_KEYWORDS:
            break

        if indent <= base_indent and current:
            tactics.append("\n".join(current))
            current = [stripped]
        else:
            current.append(stripped)

    if current:
        tactics.append("\n".join(current))

    return [t.strip() for t in tactics if t.strip()]


def build_goal_state(code: str, signature_body: str) -> str:
    """Build an approximate Lean goal state from the theorem signature.

    Extracts variables/hypotheses from the signature and the goal type.
    """
    variables = extract_variables_from_context(code)
    hyps, goal_type = parse_signature_params(signature_body)
    all_hyps = variables + hyps

    parts = []
    for h in all_hyps:
        parts.append(h)
    if goal_type:
        parts.append(f"⊢ {goal_type}")
    else:
        parts.append("⊢ <unknown>")

    return "\n".join(parts)


def extract_variables_from_context(code: str) -> list[str]:
    """Extract variable declarations from `variable` commands preceding the theorem."""
    hyps: list[str] = []
    for m in re.finditer(r"\bvariable\b\s*(.*?)(?=\n\s*\n|\btheorem\b|\blemma\b|\bdef\b|\bopen\b|\bsection\b|$)", code, re.DOTALL):
        var_block = m.group(1).strip()
        for segment in _extract_bracketed_segments(var_block):
            segment = re.sub(r"\s+", " ", segment).strip()
            if ":" in segment:
                hyps.append(segment)
    return hyps


def _extract_bracketed_segments(text: str) -> list[str]:
    """Extract top-level bracketed segments, handling nesting correctly."""
    segments: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch in "({[":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch in ")}]":
            depth -= 1
            if depth == 0 and start >= 0:
                segments.append(text[start:i])
                start = -1
    return segments


def parse_signature_params(sig: str) -> tuple[list[str], str]:
    """Parse theorem signature into (hypothesis_list, goal_type_string).

    Example input: '{f g : a ⟶ b} (η : f ⟶ g) : η = ...'
    Returns: (['f g : a ⟶ b', 'η : f ⟶ g'], 'η = ...')
    """
    hyps: list[str] = []
    goal_type = ""

    depth = 0
    current_start = 0
    i = 0
    while i < len(sig):
        ch = sig[i]
        if ch in "({[":
            if depth == 0:
                current_start = i + 1
            depth += 1
        elif ch in ")}]":
            depth -= 1
            if depth == 0:
                param = sig[current_start:i].strip()
                param = re.sub(r"\s+", " ", param)
                if param and ":" in param:
                    hyps.append(param)
        elif ch == ":" and depth == 0:
            goal_type = sig[i + 1:].strip()
            goal_type = re.sub(r"\s+", " ", goal_type)
            break
        i += 1

    return hyps, goal_type


def convert_row(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    """Convert a single APRIL row to LeanDojo format."""
    code = row.get("correct_proof")
    if not isinstance(code, str) or not code.strip():
        return None

    result = parse_theorem_signature(code)
    if result is None:
        return None

    theorem_name, goal_state, tactic_lines = result

    if not goal_state or not tactic_lines:
        return None

    JUNK_FIRST_WORDS = {"where", "·", ".", "|"}
    first_tactic = tactic_lines[0]
    first_word = first_tactic.splitlines()[0].strip().split()[0] if first_tactic.strip() else ""
    if first_word in JUNK_FIRST_WORDS:
        if len(tactic_lines) < 2:
            return None
        first_tactic = tactic_lines[1]
        first_word = first_tactic.splitlines()[0].strip().split()[0] if first_tactic.strip() else ""
        if first_word in JUNK_FIRST_WORDS:
            return None

    traced_tactics = [{
        "tactic": first_tactic,
        "annotated_tactic": [first_tactic, []],
        "state_before": goal_state,
        "state_after": "no goals" if len(tactic_lines) == 1 else "",
    }]

    file_path = str(row.get("path") or f"APRIL/{idx}.lean")
    return {
        "url": DEFAULT_URL,
        "commit": row.get("src_commit", row.get("src_hash", "")),
        "file_path": file_path,
        "full_name": theorem_name,
        "theorem_statement": "",
        "start": [1, 1],
        "end": [1, 1],
        "traced_tactics": traced_tactics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("datasets/april/raw/train"),
        help="Directory containing raw APRIL JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/april/leandojo/train"),
        help="Output directory for LeanDojo-format JSON files",
    )
    parser.add_argument(
        "--max-examples-per-file",
        type=int,
        default=None,
        help="Optional cap per input file (for testing)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_files = sorted(args.input_dir.glob("*.jsonl"))
    if not input_files:
        print(f"No JSONL files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    grand_total = 0
    grand_converted = 0
    grand_tactics = 0
    grand_first_tactic_only = 0

    for input_path in input_files:
        print(f"\nProcessing {input_path.name}...")
        out_name = input_path.stem.replace("_train", "") + "_train.json"
        out_path = args.output_dir / out_name

        n_rows = 0
        n_converted = 0
        n_first_only = 0
        n_total_tactics = 0

        with out_path.open("w", encoding="utf-8") as out_f:
            out_f.write("[\n")
            first_written = True

            with input_path.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    n_rows += 1

                    if args.max_examples_per_file is not None and n_rows > args.max_examples_per_file:
                        break

                    rec = convert_row(row, grand_total + n_rows - 1)
                    if rec is not None:
                        if not first_written:
                            out_f.write(",\n")
                        else:
                            first_written = False
                        out_f.write(json.dumps(rec, ensure_ascii=False))
                        n_converted += 1
                        n_tactics = len(rec["traced_tactics"])
                        n_total_tactics += n_tactics
                        has_intermediate = any(
                            "[intermediate state" in t["state_before"]
                            for t in rec["traced_tactics"]
                        )
                        if has_intermediate:
                            n_first_only += 1

                    if n_rows % 25000 == 0:
                        print(f"  ... {n_rows} rows processed, {n_converted} converted", flush=True)

            out_f.write("\n]")

        grand_total += n_rows
        grand_converted += n_converted
        grand_tactics += n_total_tactics
        grand_first_tactic_only += n_first_only

        print(f"  Rows: {n_rows} -> Converted: {n_converted} "
              f"({n_converted/max(n_rows,1)*100:.1f}%)")
        print(f"  Total tactic pairs: {n_total_tactics}")
        print(f"  Multi-step (first-tactic-only with initial goal): {n_first_only}")
        print(f"  Wrote: {out_path}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Total raw rows: {grand_total}")
    print(f"  Successfully converted: {grand_converted} "
          f"({grand_converted/grand_total*100:.1f}%)")
    print(f"  Total tactic training pairs: {grand_tactics}")
    print(f"  Multi-step proofs (approx intermediate states): {grand_first_tactic_only}")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

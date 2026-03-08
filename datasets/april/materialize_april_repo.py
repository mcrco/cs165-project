#!/usr/bin/env python3
"""Materialize APRIL rows into a Lean repo module tree.

This script writes APRIL proofs as real Lean modules under the local eval project,
so you can run the standard LeanDojo repo tracing path over a concrete repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_DATASET_URL = "https://huggingface.co/datasets/uw-math-ai/APRIL"
APRIL_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_JSONL = APRIL_DIR / "raw" / "val" / "mlme_val.jsonl"
DEFAULT_PROJECT_PATH = APRIL_DIR / "april_eval_project"
DEFAULT_MODULE_PREFIX = "AprilEval.Materialized"
DEFAULT_MANIFEST_PATH = APRIL_DIR / "leandojo" / "materialized" / "april_manifest.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def module_prefix_to_path(prefix: str) -> Path:
    parts = [p for p in prefix.split(".") if p]
    if not parts:
        raise ValueError("Module prefix must not be empty")
    return Path(*parts)


def ensure_import(root_module_file: Path, import_line: str) -> None:
    if not root_module_file.is_file():
        raise FileNotFoundError(f"Missing root module file: {root_module_file}")
    lines = root_module_file.read_text(encoding="utf-8").splitlines()
    if import_line not in lines:
        lines.append(import_line)
        root_module_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=DEFAULT_INPUT_JSONL,
        help=f"APRIL source JSONL (default: {DEFAULT_INPUT_JSONL})",
    )
    parser.add_argument(
        "--project-path",
        type=Path,
        default=DEFAULT_PROJECT_PATH,
        help=f"Lean project root to materialize into (default: {DEFAULT_PROJECT_PATH})",
    )
    parser.add_argument(
        "--module-prefix",
        type=str,
        default=DEFAULT_MODULE_PREFIX,
        help=f"Module prefix for generated files (default: {DEFAULT_MODULE_PREFIX})",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"JSONL manifest output path (default: {DEFAULT_MANIFEST_PATH})",
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

    if not args.project_path.is_dir():
        raise FileNotFoundError(f"Project path does not exist: {args.project_path}")

    module_prefix = args.module_prefix.strip()
    module_path = module_prefix_to_path(module_prefix)
    if len(module_path.parts) < 2:
        raise ValueError("Module prefix must be nested under a root library (e.g. X.Y)")

    generated_dir = args.project_path / module_path
    generated_dir.mkdir(parents=True, exist_ok=True)
    for p in generated_dir.glob("Row*.lean"):
        p.unlink()

    manifest_rows: list[dict[str, Any]] = []
    skipped = 0

    for row_idx, row in enumerate(rows):
        code = row.get("correct_proof")
        if not isinstance(code, str) or not code.strip():
            skipped += 1
            continue

        module_name = f"{module_prefix}.Row{row_idx:08d}"
        rel_path = module_path / f"Row{row_idx:08d}.lean"
        abs_path = args.project_path / rel_path
        abs_path.write_text(code.rstrip() + "\n", encoding="utf-8")

        manifest_rows.append(
            {
                "row_idx": row_idx,
                "module": module_name,
                "relative_path": str(rel_path),
                "source_url": DEFAULT_DATASET_URL,
                "dataset_path": row.get("path"),
                "theorem": row.get("theorem"),
            }
        )

    all_module = f"{module_prefix}.All"
    all_rel = module_path / "All.lean"
    all_abs = args.project_path / all_rel
    import_lines = [f"import {rec['module']}" for rec in manifest_rows]
    all_abs.write_text(
        "\n".join(import_lines) + ("\n" if import_lines else ""), encoding="utf-8"
    )

    bridge_rel = module_path.with_suffix(".lean")
    bridge_abs = args.project_path / bridge_rel
    bridge_abs.write_text(f"import {all_module}\n", encoding="utf-8")

    root_module_name = module_path.parts[0]
    root_module_file = args.project_path / f"{root_module_name}.lean"
    ensure_import(root_module_file, f"import {module_prefix}")

    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_path.open("w", encoding="utf-8") as f:
        for rec in manifest_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Materialized rows: {len(manifest_rows)}/{len(rows)}")
    print(f"Skipped rows (missing proof): {skipped}")
    print(f"Generated module dir: {generated_dir}")
    print(f"Manifest: {args.manifest_path}")


if __name__ == "__main__":
    main()

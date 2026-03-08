#!/usr/bin/env python3
"""Materialize NuminaMath-LEAN rows into a Lean repo module tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

DEFAULT_DATASET_URL = "https://huggingface.co/datasets/AI-MO/NuminaMath-LEAN"
NUMINA_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = NUMINA_DIR / "raw"
DEFAULT_PROJECT_PATH = NUMINA_DIR / "numina_math_lean_eval_project"
DEFAULT_MODULE_PREFIX = "NuminaMathLeanEval.Materialized"
DEFAULT_MANIFEST_PATH = (
    NUMINA_DIR / "leandojo" / "materialized" / "numina_manifest.jsonl"
)
DEFAULT_PROOF_FIELDS = ["formal_ground_truth", "formal_proof"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_rows(input_path: Path) -> list[dict[str, Any]]:
    if input_path.is_file():
        if input_path.suffix == ".jsonl":
            return read_jsonl(input_path)
        if input_path.suffix == ".parquet":
            ds = load_dataset("parquet", data_files=str(input_path), split="train")
            return [dict(rec) for rec in ds]
        raise ValueError(f"Unsupported input file type: {input_path}")

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    parquet_files = sorted(input_path.rglob("*.parquet"))
    if parquet_files:
        ds = load_dataset(
            "parquet", data_files=[str(p) for p in parquet_files], split="train"
        )
        return [dict(rec) for rec in ds]

    jsonl_files = sorted(input_path.rglob("*.jsonl"))
    if jsonl_files:
        rows: list[dict[str, Any]] = []
        for file in jsonl_files:
            rows.extend(read_jsonl(file))
        return rows

    raise FileNotFoundError(
        f"No parquet/jsonl files found under input directory: {input_path}"
    )


def parse_proof_fields(raw: str) -> list[str]:
    fields = [part.strip() for part in raw.split(",") if part.strip()]
    if not fields:
        raise ValueError("--proof-fields must contain at least one field")
    return fields


def select_proof(
    row: dict[str, Any], proof_fields: list[str]
) -> tuple[str | None, str]:
    for field in proof_fields:
        val = row.get(field)
        if isinstance(val, str) and val.strip():
            return field, val
    return None, ""


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
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "NuminaMath-LEAN source path (dir or parquet/jsonl file) "
            f"(default: {DEFAULT_INPUT_PATH})"
        ),
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
        "--proof-fields",
        type=str,
        default=",".join(DEFAULT_PROOF_FIELDS),
        help=(
            "Comma-separated proof fields to try in order "
            f"(default: {','.join(DEFAULT_PROOF_FIELDS)})"
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap for quick pilot runs",
    )
    args = parser.parse_args()

    rows = load_rows(args.input_path)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    proof_fields = parse_proof_fields(args.proof_fields)

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
        proof_field, code = select_proof(row, proof_fields)
        if not code:
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
                "uuid": row.get("uuid"),
                "source": row.get("source"),
                "proof_field": proof_field,
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

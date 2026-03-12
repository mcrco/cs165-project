#!/usr/bin/env python3
"""Convert NuminaMath-LEAN raw data into infilling training examples.

This script takes raw parquet files containing Lean proofs and generates infilling
training examples by replacing each tactic with <HOLE>. For each theorem with N
tactics, it creates N training examples where each example has one tactic replaced
by <HOLE> and the original tactic as the target to predict.

Output format matches the LeanDojo theorem JSON structure with additional fields
for infilling tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm
from datasets import load_dataset

DEFAULT_URL = "https://huggingface.co/datasets/AI-MO/NuminaMath-LEAN"
NUMINA_DIR = Path(__file__).resolve().parent
REPO_ROOT = NUMINA_DIR.parents[1]
EXTRACT_DATA_PATH = (
    REPO_ROOT / "lean_dojo_v2" / "lean_dojo" / "data_extraction" / "ExtractData.lean"
)
DEFAULT_INPUT_PATH = NUMINA_DIR / "raw"
DEFAULT_OUTPUT_JSON = NUMINA_DIR / "leandojo_infilling" / "train.json"
DEFAULT_PROJECT_PATH = NUMINA_DIR / "numina_math_lean_eval_project"
DEFAULT_PROOF_FIELDS = ["formal_ground_truth", "formal_proof"]
HOLE_TOKEN = "<HOLE>"


def configure_elan_toolchain(project_path: Path) -> None:
    toolchain_file = project_path / "lean-toolchain"
    if not toolchain_file.is_file():
        return

    toolchain = toolchain_file.read_text(encoding="utf-8").strip()
    if not toolchain:
        return

    os.environ["ELAN_TOOLCHAIN"] = toolchain


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


def infer_full_name(row: dict[str, Any], code: str, idx: int) -> str:
    theorem_name = row.get("theorem")
    if theorem_name and isinstance(theorem_name, str):
        return theorem_name

    m = re.search(r"\b(?:theorem|lemma)\s+([A-Za-z0-9_'.]+)", code)
    if m:
        return m.group(1)

    uid = row.get("uuid")
    if isinstance(uid, str) and uid:
        return f"numina_{uid}"

    return f"numina_{idx}"


def infer_statement(code: str) -> str:
    m = re.search(
        r"\b(?:theorem|lemma)\s+[A-Za-z0-9_'.]+\s*(.*?)\s*:=\s*by",
        code,
        flags=re.DOTALL,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def run_extract_data(
    project_path: Path,
    lean_relative_path: Path,
    timeout_seconds: int,
) -> tuple[int, str, str]:
    cmd = [
        "lake",
        "env",
        "lean",
        "--run",
        str(EXTRACT_DATA_PATH),
        str(lean_relative_path),
    ]
    env = os.environ.copy()
    prev_lean_path = env.get("LEAN_PATH", "")
    env["LEAN_PATH"] = (
        f"{project_path}:{prev_lean_path}" if prev_lean_path else str(project_path)
    )

    proc = subprocess.Popen(
        cmd,
        cwd=project_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as ex:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        stdout, stderr = proc.communicate()
        ex.stdout = stdout
        ex.stderr = stderr
        raise
    return proc.returncode, stdout, stderr


def get_ast_json_path(project_path: Path, lean_relative_path: Path) -> Path | None:
    rel = lean_relative_path.with_suffix(".ast.json")
    candidates = [
        project_path / ".lake" / "build" / "ir" / rel,
        project_path / "build" / "ir" / rel,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def extract_raw_tactic(code_bytes: bytes, start: int, end: int) -> str:
    if start < 0 or end < start:
        return ""
    if start >= len(code_bytes):
        return ""
    end = min(end, len(code_bytes))
    return code_bytes[start:end].decode("utf-8", errors="ignore").strip()


def parse_byte_offset(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        byte_idx = value.get("byteIdx")
        if isinstance(byte_idx, int):
            return byte_idx
    return None


def parse_tactics_from_ast(ast_json_path: Path, code: str) -> list[dict[str, Any]]:
    """Parse tactics from AST JSON, returning detailed position information."""
    payload = json.loads(ast_json_path.read_text(encoding="utf-8"))
    raw_tactics = payload.get("tactics", [])
    code_bytes = code.encode("utf-8")

    tactics: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for tac in raw_tactics:
        if not isinstance(tac, dict):
            continue

        start = parse_byte_offset(tac.get("pos"))
        end = parse_byte_offset(tac.get("endPos"))
        if start is None or end is None:
            continue

        tactic_text = extract_raw_tactic(code_bytes, start, end)
        state_before = str(tac.get("stateBefore") or "").strip()
        state_after = str(tac.get("stateAfter") or "").strip()

        if not tactic_text or tactic_text == "sorry":
            continue
        if state_before == "no goals" or "·" in tactic_text:
            continue

        sig = (state_before, tactic_text, state_after)
        if sig in seen:
            continue
        seen.add(sig)

        tactics.append({
            "tactic": tactic_text,
            "annotated_tactic": [tactic_text, []],
            "state_before": state_before,
            "state_after": state_after,
            "start_byte": start,
            "end_byte": end,
        })

    return tactics


def create_proof_with_hole(code: str, tactic_info: dict[str, Any]) -> str:
    """Replace a tactic at specific byte positions with <HOLE>."""
    code_bytes = code.encode("utf-8")
    start = tactic_info["start_byte"]
    end = tactic_info["end_byte"]

    before = code_bytes[:start].decode("utf-8", errors="ignore")
    after = code_bytes[end:].decode("utf-8", errors="ignore")

    return before + HOLE_TOKEN + after


def create_infilling_examples(
    theorem_record: dict[str, Any],
    tactics: list[dict[str, Any]],
    code: str,
) -> list[dict[str, Any]]:
    """Create infilling examples for each tactic in a theorem.

    For each tactic, creates an example where that tactic is replaced by <HOLE>.
    """
    examples = []

    for i, tactic_info in enumerate(tactics):
        # Create proof with this tactic replaced by <HOLE>
        proof_with_hole = create_proof_with_hole(code, tactic_info)

        example = {
            # Original LeanDojo fields
            "url": theorem_record["url"],
            "commit": theorem_record["commit"],
            "file_path": theorem_record["file_path"],
            "full_name": theorem_record["full_name"],
            "theorem_statement": theorem_record["theorem_statement"],
            "start": theorem_record["start"],
            "end": theorem_record["end"],
            # Infilling-specific fields
            "infilling": {
                "hole_index": i,
                "total_tactics": len(tactics),
                "proof_with_hole": proof_with_hole,
                "target_tactic": tactic_info["tactic"],
                "state_before_hole": tactic_info["state_before"],
                "state_after_hole": tactic_info["state_after"],
            },
            # Original traced tactics for reference
            "traced_tactics": tactics,
        }
        examples.append(example)

    return examples


def process_row(
    row_idx: int,
    row: dict[str, Any],
    project_path: Path,
    tmp_dir: Path,
    extract_timeout: int,
    proof_fields: list[str],
) -> tuple[int, list[dict[str, Any]] | None, dict[str, Any] | None]:
    """Process a single row and return infilling examples or error info."""
    proof_field, code = select_proof(row, proof_fields)
    if not code:
        return (
            row_idx,
            None,
            {
                "row_idx": row_idx,
                "reason": "missing_proof_fields",
                "uuid": row.get("uuid"),
                "proof_fields": proof_fields,
            },
        )

    tmp_file = tmp_dir / f"ex_{row_idx}.lean"
    tmp_rel = tmp_file.relative_to(project_path)
    tmp_file.write_text(code, encoding="utf-8")

    try:
        returncode, stdout, stderr = run_extract_data(
            project_path,
            tmp_rel,
            timeout_seconds=extract_timeout,
        )
    except subprocess.TimeoutExpired as ex:
        timeout_stdout = (ex.stdout or "").strip()
        timeout_stderr = (ex.stderr or "").strip()
        detail_parts: list[str] = [str(ex)]
        if timeout_stderr:
            detail_parts.append(f"stderr:\n{timeout_stderr}")
        if timeout_stdout:
            detail_parts.append(f"stdout:\n{timeout_stdout}")
        detail = "\n\n".join(detail_parts)[:2000]
        return (
            row_idx,
            None,
            {
                "row_idx": row_idx,
                "reason": "extractdata_timeout",
                "uuid": row.get("uuid"),
                "proof_field": proof_field,
                "detail": detail,
            },
        )
    except Exception as ex:
        return (
            row_idx,
            None,
            {
                "row_idx": row_idx,
                "reason": f"extractdata_exception:{type(ex).__name__}",
                "uuid": row.get("uuid"),
                "proof_field": proof_field,
                "detail": str(ex),
            },
        )

    if returncode != 0:
        stderr = (stderr or "").strip()
        stdout = (stdout or "").strip()
        detail_parts: list[str] = []
        if stderr:
            detail_parts.append(f"stderr:\n{stderr}")
        if stdout:
            detail_parts.append(f"stdout:\n{stdout}")
        detail = "\n\n".join(detail_parts)[:2000]
        rec: dict[str, Any] = {
            "row_idx": row_idx,
            "reason": "extractdata_process_error",
            "uuid": row.get("uuid"),
            "proof_field": proof_field,
        }
        if detail:
            rec["detail"] = detail
        return row_idx, None, rec

    ast_json_path = get_ast_json_path(project_path, tmp_rel)
    if ast_json_path is None:
        return (
            row_idx,
            None,
            {
                "row_idx": row_idx,
                "reason": "missing_ast_json",
                "uuid": row.get("uuid"),
                "proof_field": proof_field,
                "detail": str(tmp_rel),
            },
        )

    try:
        tactics = parse_tactics_from_ast(ast_json_path, code)
    except Exception as ex:
        return (
            row_idx,
            None,
            {
                "row_idx": row_idx,
                "reason": f"parse_ast_json_exception:{type(ex).__name__}",
                "uuid": row.get("uuid"),
                "proof_field": proof_field,
                "detail": str(ex),
            },
        )

    if not tactics:
        return (
            row_idx,
            None,
            {
                "row_idx": row_idx,
                "reason": "no_traced_tactics",
                "uuid": row.get("uuid"),
                "proof_field": proof_field,
            },
        )

    # Create the base theorem record
    theorem_name = infer_full_name(row, code, row_idx)
    theorem_statement = infer_statement(code)
    source = str(row.get("source") or "numina_math_lean")
    uid = str(row.get("uuid") or row_idx)
    file_path = f"{source}/{uid}.lean"

    base_record = {
        "url": DEFAULT_URL,
        "commit": "",
        "file_path": file_path,
        "full_name": theorem_name,
        "theorem_statement": theorem_statement,
        "start": [1, 1],
        "end": [1, 1],
    }

    # Create infilling examples
    infilling_examples = create_infilling_examples(base_record, tactics, code)

    return row_idx, infilling_examples, None


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
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON path for infilling examples (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--project-path",
        type=Path,
        default=DEFAULT_PROJECT_PATH,
        help=f"Lean project root for running ExtractData.lean (default: {DEFAULT_PROJECT_PATH})",
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
        "--max-theorems",
        type=int,
        default=None,
        help="Maximum number of theorems to process (default: all)",
    )
    parser.add_argument(
        "--max-examples-per-theorem",
        type=int,
        default=None,
        help="Maximum number of infilling examples per theorem (default: all tactics)",
    )
    parser.add_argument(
        "--extract-timeout",
        type=int,
        default=300,
        help="Per-file timeout (seconds) for `lake env lean --run ExtractData.lean` (default: 300)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of concurrent worker threads for conversion (default: 1)",
    )
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=None,
        help="Optional JSONL path to write per-row failure diagnostics",
    )
    parser.add_argument(
        "--print-failures",
        type=int,
        default=0,
        help="Print the first N failure diagnostics to stdout (default: 0)",
    )
    parser.add_argument(
        "--hole-token",
        type=str,
        default=HOLE_TOKEN,
        help=f"Token to use for holes (default: {HOLE_TOKEN})",
    )
    args = parser.parse_args()

    proof_fields = parse_proof_fields(args.proof_fields)
    rows = load_rows(args.input_path)

    if args.max_theorems is not None:
        rows = rows[: args.max_theorems]

    configure_elan_toolchain(args.project_path)
    if not args.project_path.is_dir():
        raise FileNotFoundError(f"Project path does not exist: {args.project_path}")
    if not EXTRACT_DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing ExtractData.lean at: {EXTRACT_DATA_PATH}")

    all_examples: list[dict[str, Any]] = []
    num_failed = 0
    num_theorems_with_examples = 0
    failure_rows: list[dict[str, Any]] = []
    failure_reasons: Counter[str] = Counter()

    # Keep generated files under a real module root so Lean can resolve module names.
    tmp_dir = args.project_path / "NuminaMathLeanEval" / "Generated"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in tmp_dir.glob("ex_*.lean"):
        p.unlink()

    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    progress = tqdm(
        total=len(rows),
        desc=f"Processing {args.input_path.name}",
        unit="theorem",
    )

    def ingest_result(
        result: tuple[int, list[dict[str, Any]] | None, dict[str, Any] | None],
    ) -> None:
        nonlocal num_failed, num_theorems_with_examples
        row_idx, examples, failure_rec = result
        if examples is not None:
            num_theorems_with_examples += 1
            # Optionally limit examples per theorem
            if args.max_examples_per_theorem is not None:
                examples = examples[: args.max_examples_per_theorem]
            all_examples.extend(examples)
        elif failure_rec is not None:
            num_failed += 1
            failure_rows.append(failure_rec)
            failure_reasons[failure_rec["reason"]] += 1

    if args.jobs == 1:
        for row_idx, row in enumerate(rows):
            ingest_result(
                process_row(
                    row_idx,
                    row,
                    args.project_path,
                    tmp_dir,
                    args.extract_timeout,
                    proof_fields,
                )
            )
            progress.update(1)
            if progress.n % 25 == 0:
                progress.set_postfix(
                    theorems_ok=num_theorems_with_examples,
                    total_examples=len(all_examples),
                    failed=num_failed,
                )
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(
                    process_row,
                    row_idx,
                    row,
                    args.project_path,
                    tmp_dir,
                    args.extract_timeout,
                    proof_fields,
                )
                for row_idx, row in enumerate(rows)
            ]
            for future in as_completed(futures):
                ingest_result(future.result())
                progress.update(1)
                if progress.n % 25 == 0:
                    progress.set_postfix(
                        theorems_ok=num_theorems_with_examples,
                        total_examples=len(all_examples),
                        failed=num_failed,
                    )

    progress.close()

    # Sort examples by theorem index and hole index for consistency
    all_examples.sort(key=lambda ex: (ex["file_path"], ex["infilling"]["hole_index"]))
    failure_rows.sort(key=lambda rec: rec["row_idx"])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(all_examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.failure_log is not None:
        args.failure_log.parent.mkdir(parents=True, exist_ok=True)
        with args.failure_log.open("w", encoding="utf-8") as f:
            for rec in failure_rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(rows)
    total_examples = len(all_examples)

    print(f"Processed {total} theorems")
    print(f"Theorems with infilling examples: {num_theorems_with_examples}")
    print(f"Total infilling examples: {total_examples}")
    print(f"Failed theorems: {num_failed}")
    print(f"Wrote: {args.output_json}")

    if failure_reasons:
        print("Failure reasons:")
        for reason, count in failure_reasons.most_common():
            print(f"  - {reason}: {count}")

    if args.failure_log is not None:
        print(f"Failure log: {args.failure_log}")

    if args.print_failures > 0 and failure_rows:
        print(f"First {min(args.print_failures, len(failure_rows))} failures:")
        for rec in failure_rows[: args.print_failures]:
            print(
                f"  - row={rec['row_idx']} reason={rec['reason']} "
                f"uuid={rec.get('uuid')} proof_field={rec.get('proof_field')}"
            )


if __name__ == "__main__":
    main()

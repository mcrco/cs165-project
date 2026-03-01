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
import os
import re
import signal
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

DEFAULT_URL = "https://huggingface.co/datasets/uw-math-ai/APRIL"
APRIL_DIR = Path(__file__).resolve().parent
REPO_ROOT = APRIL_DIR.parents[1]
EXTRACT_DATA_PATH = (
    REPO_ROOT / "lean_dojo_v2" / "lean_dojo" / "data_extraction" / "ExtractData.lean"
)
DEFAULT_INPUT_JSONL = APRIL_DIR / "raw" / "val" / "mlme_val.jsonl"
DEFAULT_OUTPUT_JSON = APRIL_DIR / "leandojo" / "val.json"
DEFAULT_PROJECT_PATH = APRIL_DIR / "april_eval_project"


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
    proc = subprocess.Popen(
        cmd,
        cwd=project_path,
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


def traced_tactics_from_ast(ast_json_path: Path, code: str) -> list[dict[str, Any]]:
    payload = json.loads(ast_json_path.read_text(encoding="utf-8"))
    raw_tactics = payload.get("tactics", [])
    code_bytes = code.encode("utf-8")

    traced_tactics: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for tac in raw_tactics:
        if not isinstance(tac, dict):
            continue

        start = parse_byte_offset(tac.get("pos"))
        end = parse_byte_offset(tac.get("endPos"))
        if start is None or end is None:
            continue

        tactic = extract_raw_tactic(code_bytes, start, end)
        state_before = str(tac.get("stateBefore") or "").strip()
        state_after = str(tac.get("stateAfter") or "").strip()

        if not tactic or tactic == "sorry":
            continue
        if state_before == "no goals" or "·" in tactic:
            continue

        sig = (state_before, tactic, state_after)
        if sig in seen:
            continue
        seen.add(sig)

        traced_tactics.append(
            {
                "tactic": tactic,
                "annotated_tactic": [tactic, []],
                "state_before": state_before,
                "state_after": state_after,
            }
        )

    return traced_tactics


def process_row(
    row_idx: int,
    row: dict[str, Any],
    project_path: Path,
    tmp_dir: Path,
    extract_timeout: int,
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
    code = row.get("correct_proof")
    if not isinstance(code, str) or not code.strip():
        return (
            row_idx,
            None,
            {
                "row_idx": row_idx,
                "reason": "missing_or_empty_correct_proof",
                "path": row.get("path"),
                "theorem": row.get("theorem"),
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
        return (
            row_idx,
            None,
            {
                "row_idx": row_idx,
                "reason": "extractdata_timeout",
                "path": row.get("path"),
                "theorem": row.get("theorem"),
                "detail": str(ex),
            },
        )
    except Exception as ex:
        return (
            row_idx,
            None,
            {
                "row_idx": row_idx,
                "reason": f"extractdata_exception:{type(ex).__name__}",
                "path": row.get("path"),
                "theorem": row.get("theorem"),
                "detail": str(ex),
            },
        )

    if returncode != 0:
        detail = (stderr or stdout).strip()
        if detail:
            detail = detail[:1000]
        rec = {
            "row_idx": row_idx,
            "reason": "extractdata_process_error",
            "path": row.get("path"),
            "theorem": row.get("theorem"),
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
                "path": row.get("path"),
                "theorem": row.get("theorem"),
                "detail": str(tmp_rel),
            },
        )

    try:
        traced_tactics = traced_tactics_from_ast(ast_json_path, code)
    except Exception as ex:
        return (
            row_idx,
            None,
            {
                "row_idx": row_idx,
                "reason": f"parse_ast_json_exception:{type(ex).__name__}",
                "path": row.get("path"),
                "theorem": row.get("theorem"),
                "detail": str(ex),
            },
        )

    if not traced_tactics:
        return (
            row_idx,
            None,
            {
                "row_idx": row_idx,
                "reason": "no_traced_tactics",
                "path": row.get("path"),
                "theorem": row.get("theorem"),
            },
        )

    theorem_name = infer_full_name(row, row_idx)
    theorem_statement = infer_statement(code)
    file_path = str(row.get("path") or f"APRIL/{row_idx}.lean")
    converted = {
        "url": DEFAULT_URL,
        "commit": row.get("src_commit") or "",
        "file_path": file_path,
        "full_name": theorem_name,
        "theorem_statement": theorem_statement,
        "start": [1, 1],
        "end": [1, 1],
        "traced_tactics": traced_tactics,
    }
    return row_idx, converted, None


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
        "--project-path",
        type=Path,
        default=DEFAULT_PROJECT_PATH,
        help=f"Lean project root for running ExtractData.lean (default: {DEFAULT_PROJECT_PATH})",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap for quick pilot runs",
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
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    configure_elan_toolchain(args.project_path)
    if not args.project_path.is_dir():
        raise FileNotFoundError(f"Project path does not exist: {args.project_path}")
    if not EXTRACT_DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing ExtractData.lean at: {EXTRACT_DATA_PATH}")

    converted_by_idx: dict[int, dict[str, Any]] = {}
    num_failed = 0
    failure_rows: list[dict[str, Any]] = []
    failure_reasons: Counter[str] = Counter()

    # Keep generated files under a real module root so Lean can resolve module names.
    tmp_dir = args.project_path / "AprilEval" / "Generated"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in tmp_dir.glob("ex_*.lean"):
        p.unlink()

    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")

    progress = tqdm(
        total=len(rows),
        desc=f"Converting {args.input_jsonl.name}",
        unit="row",
    )

    def ingest_result(
        result: tuple[int, dict[str, Any] | None, dict[str, Any] | None],
    ) -> None:
        nonlocal num_failed
        row_idx, converted_rec, failure_rec = result
        if converted_rec is not None:
            converted_by_idx[row_idx] = converted_rec
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
                )
            )
            progress.update(1)
            if progress.n % 25 == 0:
                progress.set_postfix(ok=len(converted_by_idx), failed=num_failed)
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
                )
                for row_idx, row in enumerate(rows)
            ]
            for future in as_completed(futures):
                ingest_result(future.result())
                progress.update(1)
                if progress.n % 25 == 0:
                    progress.set_postfix(ok=len(converted_by_idx), failed=num_failed)

    progress.close()

    converted = [converted_by_idx[i] for i in sorted(converted_by_idx)]
    failure_rows.sort(key=lambda rec: rec["row_idx"])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.failure_log is not None:
        args.failure_log.parent.mkdir(parents=True, exist_ok=True)
        with args.failure_log.open("w", encoding="utf-8") as f:
            for rec in failure_rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(rows)
    ok = len(converted)
    print(f"Converted {ok}/{total} rows. Failed/no tactics: {num_failed}")
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
                f"path={rec['path']} theorem={rec['theorem']}"
            )


if __name__ == "__main__":
    main()

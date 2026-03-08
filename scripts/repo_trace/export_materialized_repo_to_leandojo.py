#!/usr/bin/env python3
"""Trace a local materialized Lean repo and export LeanDojo-style theorem JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lean_dojo_v2.lean_dojo.data_extraction.lean import LEAN4_PACKAGES_DIR, LeanGitRepo
from lean_dojo_v2.lean_dojo.data_extraction.trace import trace


def _file_path_for_theorem(traced_repo, theorem) -> str:
    if theorem.repo == traced_repo.repo:
        return str(theorem.theorem.file_path)
    for name, dep in traced_repo.dependencies.items():
        if dep == theorem.repo:
            return f"{LEAN4_PACKAGES_DIR}/{name}/{theorem.theorem.file_path}"
    raise ValueError(f"Unable to map file path for theorem in repo {theorem.repo}")


def _theorem_statement(theorem) -> str:
    if theorem.has_tactic_proof() and theorem.get_tactic_proof() is not None:
        return theorem.get_theorem_statement()
    return ""


def export_materialized(
    project_path: Path,
    module_prefix: str,
    output_json: Path,
    dataset_url: str,
    build_deps: bool,
) -> dict[str, Any]:
    repo = LeanGitRepo.from_path(project_path)
    traced_repo = trace(repo, build_deps=build_deps)

    module_path_prefix = Path(*module_prefix.split("."))
    row_prefix = str(module_path_prefix / "Row")

    out: list[dict[str, Any]] = []
    for thm in traced_repo.get_traced_theorems():
        fp = _file_path_for_theorem(traced_repo, thm)
        if not fp.startswith(row_prefix):
            continue

        tactics = [
            {
                "tactic": tactic.tactic,
                "annotated_tactic": tactic.get_annotated_tactic(),
                "state_before": tactic.state_before,
                "state_after": tactic.state_after,
            }
            for tactic in thm.get_traced_tactics()
            if tactic.state_before != "no goals" and "·" not in tactic.tactic
        ]
        out.append(
            {
                "url": dataset_url,
                "commit": repo.commit,
                "file_path": fp,
                "full_name": thm.theorem.full_name,
                "theorem_statement": _theorem_statement(thm),
                "start": list(thm.start),
                "end": list(thm.end),
                "traced_tactics": tactics,
            }
        )

    def _row_idx_key(rec: dict[str, Any]) -> tuple[int, str]:
        name = Path(rec["file_path"]).name
        # Row00000123.lean -> 123
        row_idx = (
            int(name[3:-5]) if name.startswith("Row") and name.endswith(".lean") else -1
        )
        return row_idx, rec.get("full_name") or ""

    out.sort(key=_row_idx_key)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "repo_url": repo.url,
        "repo_commit": repo.commit,
        "exported_theorems": len(out),
        "output_json": str(output_json),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-path", type=Path, required=True)
    parser.add_argument("--module-prefix", type=str, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--dataset-url", type=str, required=True)
    parser.add_argument("--build-deps", action="store_true")
    args = parser.parse_args()

    summary = export_materialized(
        project_path=args.project_path,
        module_prefix=args.module_prefix,
        output_json=args.output_json,
        dataset_url=args.dataset_url,
        build_deps=args.build_deps,
    )
    for k, v in summary.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()

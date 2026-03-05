#!/usr/bin/env python3
"""Download NuminaMath-LEAN dataset files from Hugging Face.

Defaults:
- repo: AI-MO/NuminaMath-LEAN
- target dir: datasets/numina-math-lean/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_target = repo_root / "datasets" / "numina-math-lean" / "raw"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        type=str,
        default="AI-MO/NuminaMath-LEAN",
        help="HF dataset repo id (default: AI-MO/NuminaMath-LEAN)",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=default_target,
        help=f"Local directory for downloaded files (default: {default_target})",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Dataset revision/branch/tag (default: main)",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help=(
            "Optional allowlist patterns (e.g. 'data/*'). "
            "If omitted, downloads full dataset snapshot."
        ),
    )

    args = parser.parse_args()

    args.target_dir.mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.target_dir),
        local_dir_use_symlinks=False,
        revision=args.revision,
        allow_patterns=args.include,
    )

    print(f"Downloaded dataset snapshot to: {local_path}")
    print("Example next step:")
    print(
        "  .venv/bin/python datasets/numina-math-lean/convert_numina_math_lean_to_leandojo.py "
        "--input-path datasets/numina-math-lean/raw"
    )


if __name__ == "__main__":
    main()

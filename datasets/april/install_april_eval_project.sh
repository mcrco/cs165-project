#!/usr/bin/env bash
set -euo pipefail

# Create a Lean project configured with mathlib for APRIL proof-search evaluation.
# Default install location: datasets/april/april_eval_project

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/april_eval_project"

usage() {
  cat <<'EOF'
Usage: install_april_eval_project.sh [--target-dir DIR]

Options:
  --target-dir DIR   Where to create/use the Lean project.
                     Default: datasets/april/april_eval_project
  -h, --help         Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

if ! need_cmd lean || ! need_cmd lake; then
  cat >&2 <<'EOF'
Lean tooling not found.
Expected: `lean` and `lake` are already installed and on PATH.
EOF
  exit 1
fi

mkdir -p "$TARGET_DIR"

if [[ ! -f "$TARGET_DIR/lakefile.lean" ]]; then
  echo "[setup] Initializing Lean project at: $TARGET_DIR"
  (
    cd "$TARGET_DIR"
    lake init AprilEval
  )
fi

cat > "$TARGET_DIR/lakefile.lean" <<'EOF'
import Lake
open Lake DSL

package «AprilEval»

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
EOF

echo "[setup] Resolving dependencies..."
(
  cd "$TARGET_DIR"
  lake update
)

echo "[setup] Downloading mathlib cache (if available)..."
(
  cd "$TARGET_DIR"
  lake exe cache get
)

echo "[done] APRIL eval project ready."
echo "Use this in eval scripts:"
echo "  --imports Init Mathlib --project-path $TARGET_DIR"

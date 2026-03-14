#!/usr/bin/env bash
set -euo pipefail

# Create a Lean project configured with mathlib for APRIL conversion.
# Default install location: datasets/april/base_dependency_repo

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/base_dependency_repo"
LEAN_TOOLCHAIN="leanprover/lean4:v4.27.0"
MATHLIB_TAG="v4.27.0"

usage() {
  cat <<'EOF'
Usage: install_base_dependency_repo.sh [--target-dir DIR]

Options:
  --target-dir DIR   Where to create/use the Lean project.
                     Default: datasets/april/base_dependency_repo
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

cat > "$TARGET_DIR/lakefile.lean" <<EOF
import Lake
open Lake DSL

package «AprilEval»

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "$MATHLIB_TAG"

@[default_target]
lean_lib «AprilEval»
EOF

# NOTE: APRIL repo metadata references Lean v4.22.0-rc4, but this project uses
# ExtractData.lean from LeanDojo, which depends on newer Lean APIs and fails on v4.22.
# Keep Lean/mathlib at v4.27.0 unless ExtractData.lean is backported.
printf '%s\n' "$LEAN_TOOLCHAIN" > "$TARGET_DIR/lean-toolchain"

echo "[setup] Resolving dependencies..."
(
  cd "$TARGET_DIR"
  echo "[setup] Purging Lake build/cache artifacts to avoid stale .olean/.trace incompatibilities..."
  rm -rf .lake build
  lake update
)

echo "[setup] Downloading mathlib cache (if available)..."
(
  cd "$TARGET_DIR"
  lake exe cache get
)

echo "[setup] Committing template files to git..."
(
  cd "$TARGET_DIR"
  git add -A
  if ! git diff --cached --quiet; then
    git commit -q -m "Initial AprilEval template with mathlib v4.27.0"
  fi
)

echo "[done] AprilEval base repo ready at $TARGET_DIR"

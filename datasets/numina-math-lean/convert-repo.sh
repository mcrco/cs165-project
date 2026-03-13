#!/bin/bash
#SBATCH -p expansion
#SBATCH --time=3:00:00
#SBATCH -c 32
#SBATCH --mem=32G

set -euo pipefail

# Determine the script's location. When running via sbatch, BASH_SOURCE[0]
# points to a copy in /var/spool/slurmd/, so use SLURM_SUBMIT_DIR instead.
# Assume we're running from datasets/numina-math-lean directory.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Cache/tmp directories default to the script directory (override with env vars)
# Otherwise it uses default home dir which has a quota on Caltech HPC.
CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/.cache/lean_dojo}"
TMP_DIR="${TMP_DIR:-${SCRIPT_DIR}/.tmp}"
export CACHE_DIR TMP_DIR
mkdir -p "${CACHE_DIR}" "${TMP_DIR}"
echo "[convert-repo] CACHE_DIR=${CACHE_DIR}"
echo "[convert-repo] TMP_DIR=${TMP_DIR}"

cd "${SCRIPT_DIR}"

WORK_REPO="${SCRIPT_DIR}/numina_math_repo"

if [[ ! -e "${WORK_REPO}/.git" ]]; then
  echo "[convert-repo] ERROR: WORK_REPO is not a git repository: ${WORK_REPO}"
  echo "[convert-repo] Run materialize-repo.sh first to create the materialized repository."
  exit 1
fi

echo "[convert-repo] Starting conversion at $(date -Iseconds)"
echo "[convert-repo] Using WORK_REPO=${WORK_REPO}"

OUTPUT_JSON="${OUTPUT_JSON:-output.json}"
echo "[convert-repo] Output will be written to: ${OUTPUT_JSON}"

mkdir -p "$(dirname "${OUTPUT_JSON}")"

# Useless unless we are doing retrieval tasks.
TRACE_BUILD_DEPS="${TRACE_BUILD_DEPS:-0}"

export_cmd=(
  uv run python "${REPO_DIR}/datasets/export_materialized_repo_to_leandojo.py"
  --project-path "${WORK_REPO}"
  --module-prefix "NuminaMathLeanEval.Materialized"
  --dataset-url "https://huggingface.co/datasets/AI-MO/NuminaMath-LEAN"
  --output-json "${OUTPUT_JSON}"
)
if [[ "${TRACE_BUILD_DEPS}" != "0" ]]; then
  export_cmd+=(--build-deps)
fi

if ! "${export_cmd[@]}"; then
  echo "[convert-repo] ERROR: Conversion failed."
  exit 1
fi

echo "[convert-repo] Done at $(date -Iseconds)"
echo "[convert-repo] Output: ${OUTPUT_JSON}"

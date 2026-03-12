#!/bin/bash
#SBATCH -p expansion
#SBATCH --time=12:00:00
#SBATCH -c 32
#SBATCH --mem=64G

set -euo pipefail

if [[ -n "${NUMINA_MATH_LEAN_DIR:-}" ]]; then
  SCRIPT_DIR="$(cd "${NUMINA_MATH_LEAN_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ ! -d "${SCRIPT_DIR}/numina_math_lean_eval_project" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/numina_math_lean_eval_project" ]]; then
      SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    elif [[ -d "$(pwd)/numina_math_lean_eval_project" ]]; then
      SCRIPT_DIR="$(pwd)"
    else
      echo "[convert-repo] Could not locate dataset directory."
      echo "[convert-repo] Set NUMINA_MATH_LEAN_DIR or submit from datasets/numina-math-lean."
      exit 1
    fi
  fi
fi

REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

echo "[convert-repo] Starting Numina trace conversion at $(date -Iseconds)"

OUT_ROOT="${OUT_ROOT:-leandojo_repo}"
mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/failures"

WORK_REPO="${WORK_REPO:-${OUT_ROOT}/numina_math_lean_eval_project}"
WORK_REPO="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${WORK_REPO}")"
if [[ ! -e "${WORK_REPO}/.git" ]]; then
  echo "[convert-repo] WORK_REPO is not a git repository: ${WORK_REPO}"
  echo "[convert-repo] Run materialize-repo.sh first to set up the materialized repository."
  exit 1
fi
echo "[convert-repo] Using WORK_REPO=${WORK_REPO}"

TRACE_BUILD_DEPS="${TRACE_BUILD_DEPS:-1}"

mapfile -t inputs < <(find raw -type f \( -name '*.parquet' -o -name '*.jsonl' \) | sort)

if [[ ${#inputs[@]} -eq 0 ]]; then
  echo "[convert-repo] No input files found under raw/."
  echo "[convert-repo] Run download_numina_math_lean_dataset.py first."
  exit 1
fi

for input in "${inputs[@]}"; do
  rel="${input#raw/}"
  ext="${input##*.}"
  stem="${rel%.${ext}}"

  output="${OUT_ROOT}/${stem}.json"
  failure_log="${OUT_ROOT}/failures/${stem}.failures.jsonl"

  mkdir -p "$(dirname "${output}")"

  echo "[convert-repo] ${stem} -> ${output}"

  export_cmd=(
    uv run python "${REPO_DIR}/scripts/repo_trace/export_materialized_repo_to_leandojo.py"
    --project-path "${WORK_REPO}"
    --module-prefix "NuminaMathLeanEval.Materialized"
    --dataset-url "https://huggingface.co/datasets/AI-MO/NuminaMath-LEAN"
    --output-json "${output}"
  )
  if [[ "${TRACE_BUILD_DEPS}" != "0" ]]; then
    export_cmd+=(--build-deps)
  fi
  if ! "${export_cmd[@]}"; then
    echo "{\"reason\":\"trace_or_export_failed\",\"input\":\"${input}\"}" > "${failure_log}"
  fi
done

echo "[convert-repo] Done at $(date -Iseconds)"

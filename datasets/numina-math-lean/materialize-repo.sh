#!/bin/bash
#SBATCH -p expansion
#SBATCH --time=12:00:00
#SBATCH -c 16
#SBATCH --mem=64G

# Clones the numina math lean eval project into a new repo, then adds each of the proofs
# in numina math as its own file in the new repo.
# Defaults to datasets/numina-math-lean/leandojo_repo/numina_math_lean_eval_project.

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
      echo "[materialize-repo] Could not locate dataset directory."
      echo "[materialize-repo] Set NUMINA_MATH_LEAN_DIR or submit from datasets/numina-math-lean."
      exit 1
    fi
  fi
fi

cd "${SCRIPT_DIR}"

echo "[materialize-repo] Starting Numina materialization at $(date -Iseconds)"

OUT_ROOT="${OUT_ROOT:-leandojo_repo}"
mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/failures"

WORK_REPO="${WORK_REPO:-${OUT_ROOT}/numina_math_lean_eval_project}"
WORK_REPO="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${WORK_REPO}")"
if [[ ! -e "${WORK_REPO}/.git" ]]; then
  mkdir -p "$(dirname "${WORK_REPO}")"
  rm -rf "${WORK_REPO}"
  git clone -q "${SCRIPT_DIR}/numina_math_lean_eval_project" "${WORK_REPO}"
fi
if [[ ! -e "${WORK_REPO}/.git" ]]; then
  echo "[materialize-repo] WORK_REPO is not a git repository: ${WORK_REPO}"
  exit 1
fi
git -C "${WORK_REPO}" config user.email "materializer@local"
git -C "${WORK_REPO}" config user.name "Materializer"
echo "[materialize-repo] Using WORK_REPO=${WORK_REPO}"

mapfile -t inputs < <(find raw -type f \( -name '*.parquet' -o -name '*.jsonl' \) | sort)

if [[ ${#inputs[@]} -eq 0 ]]; then
  echo "[materialize-repo] No input files found under raw/."
  echo "[materialize-repo] Run download_numina_math_lean_dataset.py first."
  exit 1
fi

for input in "${inputs[@]}"; do
  rel="${input#raw/}"
  ext="${input##*.}"
  stem="${rel%.${ext}}"

  manifest="${OUT_ROOT}/materialized/${stem}.manifest.jsonl"
  failure_log="${OUT_ROOT}/failures/${stem}.failures.jsonl"

  mkdir -p "$(dirname "${manifest}")" "$(dirname "${failure_log}")"

  echo "[materialize-repo] ${input} -> ${manifest}"

  git -C "${WORK_REPO}" reset --hard -q
  git -C "${WORK_REPO}" clean -fdq

  materialize_cmd=(
    uv run python materialize_numina_math_lean_repo.py
    --input-path "${input}"
    --project-path "${WORK_REPO}"
    --manifest-path "${manifest}"
    --proof-fields "${PROOF_FIELDS:-formal_ground_truth,formal_proof}"
  )
  if [[ -n "${MAX_EXAMPLES:-}" ]]; then
    materialize_cmd+=(--max-examples "${MAX_EXAMPLES}")
  fi
  if ! "${materialize_cmd[@]}"; then
    echo "{\"reason\":\"materialize_failed\",\"input\":\"${input}\"}" > "${failure_log}"
    continue
  fi

  git -C "${WORK_REPO}" add -A
  if git -C "${WORK_REPO}" diff --cached --quiet; then
    echo "[materialize-repo] no materialized changes for ${input}, skipping commit"
    echo '{"reason":"no_materialized_changes"}' > "${failure_log}"
    continue
  fi
  git -C "${WORK_REPO}" commit -q -m "materialize ${stem}"
  echo "[materialize-repo] Committed materialized changes for ${stem}"
done

echo "[materialize-repo] Done at $(date -Iseconds)"
#!/bin/bash
#SBATCH -p expansion
#SBATCH --time=24:00:00
#SBATCH -c 16
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

echo "[convert-repo] Starting Numina materialize+trace conversion at $(date -Iseconds)"

OUT_ROOT="${OUT_ROOT:-leandojo_repo}"
mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/failures"

TMP_BASE="${TMP_BASE:-/tmp}"
RUN_DIR="$(mktemp -d "${TMP_BASE%/}/numina_repo_trace.XXXXXX")"
cleanup() {
  rm -rf "${RUN_DIR}"
}
trap cleanup EXIT

WORK_REPO="${RUN_DIR}/numina_math_lean_eval_project"
git clone -q "${SCRIPT_DIR}/numina_math_lean_eval_project" "${WORK_REPO}"
git -C "${WORK_REPO}" config user.email "materializer@local"
git -C "${WORK_REPO}" config user.name "Materializer"

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
  manifest="${OUT_ROOT}/materialized/${stem}.manifest.jsonl"
  failure_log="${OUT_ROOT}/failures/${stem}.failures.jsonl"

  mkdir -p "$(dirname "${output}")" "$(dirname "${manifest}")" "$(dirname "${failure_log}")"

  echo "[convert-repo] ${input} -> ${output}"

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
  "${materialize_cmd[@]}"

  git -C "${WORK_REPO}" add -A
  if git -C "${WORK_REPO}" diff --cached --quiet; then
    echo "[convert-repo] no materialized changes for ${input}, skipping trace"
    echo '{"reason":"no_materialized_changes"}' > "${failure_log}"
    continue
  fi
  git -C "${WORK_REPO}" commit -q -m "materialize ${stem}"

  if ! uv run python "${REPO_DIR}/scripts/repo_trace/export_materialized_repo_to_leandojo.py" \
    --project-path "${WORK_REPO}" \
    --module-prefix "NuminaMathLeanEval.Materialized" \
    --dataset-url "https://huggingface.co/datasets/AI-MO/NuminaMath-LEAN" \
    --output-json "${output}" ; then
    echo "{\"reason\":\"trace_or_export_failed\",\"input\":\"${input}\"}" > "${failure_log}"
  fi
done

echo "[convert-repo] Done at $(date -Iseconds)"

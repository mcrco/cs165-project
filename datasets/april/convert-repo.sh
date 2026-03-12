#!/bin/bash
#SBATCH -p expansion
#SBATCH --time=24:00:00
#SBATCH -c 8
#SBATCH --mem=32G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

echo "[convert-repo] Starting APRIL materialize+trace conversion at $(date -Iseconds)"

OUT_ROOT="${OUT_ROOT:-leandojo_repo}"
mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/failures"

TMP_BASE="${TMP_BASE:-/tmp}"
RUN_DIR="$(mktemp -d "${TMP_BASE%/}/april_repo_trace.XXXXXX")"
cleanup() {
  rm -rf "${RUN_DIR}"
}
trap cleanup EXIT

WORK_REPO="${RUN_DIR}/april_eval_project"
git clone -q "${SCRIPT_DIR}/april_eval_project" "${WORK_REPO}"
git -C "${WORK_REPO}" config user.email "materializer@local"
git -C "${WORK_REPO}" config user.name "Materializer"

for input in raw/*/*.jsonl; do
  split="$(basename "$(dirname "${input}")")"
  stem="$(basename "${input}" .jsonl)"
  output="${OUT_ROOT}/${split}/${stem}.json"
  manifest="${OUT_ROOT}/materialized/${split}/${stem}.manifest.jsonl"
  failure_log="${OUT_ROOT}/failures/${split}/${stem}.failures.jsonl"

  mkdir -p "$(dirname "${output}")" "$(dirname "${manifest}")" "$(dirname "${failure_log}")"

  echo "[convert-repo] ${input} -> ${output}"

  git -C "${WORK_REPO}" reset --hard -q
  git -C "${WORK_REPO}" clean -fdq

  materialize_cmd=(
    uv run python materialize_april_repo.py
    --input-jsonl "${input}"
    --project-path "${WORK_REPO}"
    --manifest-path "${manifest}"
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
  git -C "${WORK_REPO}" commit -q -m "materialize ${split}/${stem}"

  if ! uv run python "${REPO_DIR}/datasets/export_materialized_repo_to_leandojo.py" \
    --project-path "${WORK_REPO}" \
    --module-prefix "AprilEval.Materialized" \
    --dataset-url "https://huggingface.co/datasets/uw-math-ai/APRIL" \
    --output-json "${output}" ; then
    echo "{\"reason\":\"trace_or_export_failed\",\"input\":\"${input}\"}" > "${failure_log}"
  fi
done

echo "[convert-repo] Done at $(date -Iseconds)"

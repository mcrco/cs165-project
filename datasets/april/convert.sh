#!/bin/bash
#SBATCH -p expansion
#SBATCH --time=12:00:00
#SBATCH -c 4
#SBATCH --mem=16G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "[convert] Starting APRIL -> LeanDojo conversion at $(date -Iseconds)"

# Ensure the Lean project is configured as a real module root.
cd "${SCRIPT_DIR}/april_eval_project"
lake build AprilEval
cd "${SCRIPT_DIR}"

mkdir -p leandojo

for input in raw/*/*.jsonl; do
  split="$(basename "$(dirname "${input}")")"
  stem="$(basename "${input}" .jsonl)"
  output="leandojo/${split}/${stem}.json"
  failure_log="leandojo/failures/${split}/${stem}.failures.jsonl"

  mkdir -p "$(dirname "${output}")" "$(dirname "${failure_log}")"

  echo "[convert] ${input} -> ${output}"
  if [[ "${MATERIALIZE_REPO:-0}" == "1" ]]; then
    manifest="leandojo/materialized/${split}/${stem}.manifest.jsonl"
    mkdir -p "$(dirname "${manifest}")"
    echo "[materialize] ${input} -> ${manifest}"
    uv run python materialize_april_repo.py \
      --input-jsonl "${input}" \
      --manifest-path "${manifest}"
  fi
  uv run python convert_april_to_leandojo.py \
    --input-jsonl "${input}" \
    --output-json "${output}" \
    --failure-log "${failure_log}"
done

echo "[convert] Done at $(date -Iseconds)"

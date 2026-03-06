#!/bin/bash
#SBATCH -p expansion
#SBATCH --time=24:00:00
#SBATCH -c 16
#SBATCH --mem=32g

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "[convert] Starting NuminaMath-LEAN -> LeanDojo conversion at $(date -Iseconds)"

# Ensure the Lean project is configured as a real module root.
cd "${SCRIPT_DIR}/numina_math_lean_eval_project"
lake build NuminaMathLeanEval
cd "${SCRIPT_DIR}"

mkdir -p leandojo/failures

inputs=( $(find raw -type f \( -name '*.parquet' -o -name '*.jsonl' \) | sort) )

if [[ ${#inputs[@]} -eq 0 ]]; then
  echo "[convert] No input files found under raw/."
  echo "[convert] Run download_numina_math_lean_dataset.py first."
  exit 1
fi

for input in "${inputs[@]}"; do
  rel="${input#raw/}"
  ext="${input##*.}"
  stem="${rel%.${ext}}"

  output="leandojo/${stem}.json"
  failure_log="leandojo/failures/${stem}.failures.jsonl"

  mkdir -p "$(dirname "${output}")" "$(dirname "${failure_log}")"

  echo "[convert] ${input} -> ${output}"
  uv run python convert_numina_math_lean_to_leandojo.py \
    --input-path "${input}" \
    --output-json "${output}" \
    --failure-log "${failure_log}"
done

echo "[convert] Done at $(date -Iseconds)"

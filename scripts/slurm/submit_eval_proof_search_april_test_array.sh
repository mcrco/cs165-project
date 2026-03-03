#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_DIR"

TEST_DIR="${TEST_DIR:-datasets/april/leandojo/test}"
RUNS_DIR="${RUNS_DIR:-logs/slurm/proof_search_runs}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"

if [[ -z "${MODEL_TYPE:-}" ]]; then
  echo "ERROR: MODEL_TYPE must be set (diffusion|hf)." >&2
  exit 1
fi
mkdir -p "$RUNS_DIR" logs/slurm
run_id="$(date +%Y%m%d_%H%M%S)"
run_dir="${RUNS_DIR}/${run_id}"
mkdir -p "$run_dir"

filelist="${run_dir}/test_files.txt"
if [[ -d "$TEST_DIR" ]]; then
  find "$TEST_DIR" -maxdepth 1 -type f -name '*.json' | sort > "$filelist"
else
  echo "ERROR: TEST_DIR=$TEST_DIR does not exist." >&2
  exit 1
fi

num_files="$(wc -l < "$filelist" | tr -d ' ')"
if [[ "$num_files" -eq 0 ]]; then
  echo "ERROR: No JSON files found to evaluate." >&2
  exit 1
fi

array_spec="0-$((num_files - 1))%${MAX_PARALLEL}"
sbatch_script="${SCRIPT_DIR}/eval_proof_search_april_test_array.sbatch"

echo "Submitting ${num_files} files with array=${array_spec}"
echo "File list: ${filelist}"

sbatch \
  --array="$array_spec" \
  --export=ALL,REPO_DIR="$REPO_DIR",FILELIST="$filelist",MODEL_TYPE="$MODEL_TYPE" \
  "$sbatch_script"

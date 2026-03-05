#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_DIR"

TEST_DIR="${TEST_DIR:-datasets/april/leandojo/test}"
RUNS_DIR="${RUNS_DIR:-logs/slurm/proof_search_runs}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
MODEL_TYPE="${MODEL_TYPE:-hf}"

if [[ "$MODEL_TYPE" != "diffusion" && "$MODEL_TYPE" != "hf" ]]; then
  echo "ERROR: MODEL_TYPE must be one of: diffusion|hf (got: $MODEL_TYPE)." >&2
  exit 1
fi
mkdir -p "$RUNS_DIR" logs/slurm
run_id="$(date +%Y%m%d_%H%M%S)"
run_dir="${RUNS_DIR}/${run_id}"
mkdir -p "$run_dir"

if [[ -d "$TEST_DIR" ]]; then
  num_files="$(find "$TEST_DIR" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')"
else
  echo "ERROR: TEST_DIR=$TEST_DIR does not exist." >&2
  exit 1
fi

if [[ "$num_files" -eq 0 ]]; then
  echo "ERROR: No JSON files found to evaluate." >&2
  exit 1
fi

array_spec="0-$((num_files - 1))%${MAX_PARALLEL}"
sbatch_script="${SCRIPT_DIR}/eval_proof_search_april_test_array.sbatch"

echo "Submitting ${num_files} files with array=${array_spec}"
echo "Test dir: ${TEST_DIR}"
echo "Run dir: ${run_dir}"

sbatch \
  --array="$array_spec" \
  --export=ALL,REPO_DIR="$REPO_DIR",TEST_DIR="$TEST_DIR",RUN_ID="$run_id",MODEL_TYPE="$MODEL_TYPE" \
  "$sbatch_script"

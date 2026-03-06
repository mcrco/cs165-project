#!/usr/bin/env bash
#SBATCH --job-name=april-proofsearch
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

if [[ -n "${MODULES_TO_LOAD:-}" ]]; then
  module purge
  # Space-separated list, e.g. "python/3.11.6-gcc-13.2.0 cuda/12.2"
  # shellcheck disable=SC2206
  modules=(${MODULES_TO_LOAD})
  module load "${modules[@]}"
fi

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"

if [[ -f "$REPO_DIR/hf_var_fixes.sh" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_DIR/hf_var_fixes.sh"
fi
export UV_NATIVE_TLS="${UV_NATIVE_TLS:-1}"

if [[ -f "${VENV_PATH:-.venv/bin/activate}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH:-.venv/bin/activate}"
fi

TEST_DIR="${TEST_DIR:-datasets/april/leandojo/test}"
MODEL_TYPE="${MODEL_TYPE:-hf}"
if [[ ! -d "$TEST_DIR" ]]; then
  echo "ERROR: TEST_DIR does not exist: $TEST_DIR" >&2
  exit 1
fi
if [[ "$MODEL_TYPE" != "diffusion" && "$MODEL_TYPE" != "hf" ]]; then
  echo "ERROR: Unsupported MODEL_TYPE=$MODEL_TYPE (expected hf|diffusion)." >&2
  exit 1
fi

# Default checkpoints per model type (override by setting CKPT explicitly).
if [[ -z "${CKPT:-}" ]]; then
  case "$MODEL_TYPE" in
    hf)
      CKPT="deepseek-ai/DeepSeek-Prover-V2-7B"
      ;;
    diffusion)
      CKPT="inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
      ;;
    *)
      echo "ERROR: Unsupported MODEL_TYPE=$MODEL_TYPE (expected hf|diffusion)." >&2
      exit 1
      ;;
  esac
fi

# Build deterministic ordering per task at runtime to avoid stale filelist paths.
mapfile -t data_files < <(find "$TEST_DIR" -maxdepth 1 -type f -name '*.json' | sort)
if [[ "${#data_files[@]}" -eq 0 ]]; then
  echo "ERROR: No JSON files found in TEST_DIR=$TEST_DIR" >&2
  exit 1
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#data_files[@]} )); then
  echo "ERROR: No data file for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi
DATA_JSON="${data_files[$SLURM_ARRAY_TASK_ID]}"

basename_no_ext="$(basename "${DATA_JSON%.*}")"
OUT_DIR="${OUT_DIR:-logs/proof_search/${SLURM_ARRAY_JOB_ID}}"
mkdir -p "$OUT_DIR"
out_file="${OUT_DIR}/${basename_no_ext}.summary.txt"

IFS=' ' read -r -a imports_arr <<< "${IMPORTS:-Init Mathlib}"
cmd=(
  uv run python scripts/eval_proof_search.py
  --data-json "$DATA_JSON"
  --model-type "$MODEL_TYPE"
  --ckpt "$CKPT"
  --device "${DEVICE:-auto}"
  --project-path "${PROJECT_PATH:-datasets/april/april_eval_project}"
  --server-timeout "${SERVER_TIMEOUT:-60}"
  --imports "${imports_arr[@]}"
)

if [[ "${USE_LORA:-0}" == "1" ]]; then
  cmd+=(--use-lora)
fi
if [[ -n "${MAX_THEOREMS:-}" ]]; then
  cmd+=(--max-theorems "$MAX_THEOREMS")
fi

{
  echo "[$(date -Is)] Starting proof search"
  echo "repo_dir=$REPO_DIR"
  echo "test_dir=$TEST_DIR"
  echo "data_json=$DATA_JSON"
  echo "model_type=$MODEL_TYPE"
  echo "ckpt=$CKPT"
  echo "task_id=${SLURM_ARRAY_TASK_ID}"
  echo "job_id=${SLURM_ARRAY_JOB_ID}"
  echo "----------------------------------------"
  srun "${cmd[@]}"
} | tee "$out_file"

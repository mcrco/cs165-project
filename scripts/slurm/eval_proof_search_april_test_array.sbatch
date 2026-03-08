#!/usr/bin/env bash
#SBATCH --job-name=april-proofsearch
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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

if [[ -f "${VENV_PATH:-.venv/bin/activate}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH:-.venv/bin/activate}"
fi

# Load project secrets/tokens when available.
if [[ -f "${ENV_FILE:-.env}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE:-.env}"
  set +a
fi

# Reduce certificate-related HF failures on shared HPC environments.
if command -v python >/dev/null 2>&1; then
  CERT_PATH="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
  if [[ -n "${CERT_PATH}" ]]; then
    export SSL_CERT_FILE="${SSL_CERT_FILE:-$CERT_PATH}"
    export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$CERT_PATH}"
    export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-$CERT_PATH}"
  fi
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ -z "${FILELIST:-}" ]]; then
  echo "ERROR: FILELIST env var is required." >&2
  exit 1
fi
if [[ -z "${MODEL_TYPE:-}" ]]; then
  echo "ERROR: MODEL_TYPE env var is required (diffusion|hf)." >&2
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

line_no=$((SLURM_ARRAY_TASK_ID + 1))
DATA_JSON="$(sed -n "${line_no}p" "$FILELIST")"
if [[ -z "${DATA_JSON}" ]]; then
  echo "ERROR: No data file for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

basename_no_ext="$(basename "${DATA_JSON%.*}")"
OUT_DIR="${OUT_DIR:-logs/proof_search/${SLURM_ARRAY_JOB_ID}}"
mkdir -p "$OUT_DIR"
out_file="${OUT_DIR}/${basename_no_ext}.summary.txt"

IFS=' ' read -r -a imports_arr <<< "${IMPORTS:-Init Mathlib}"
if command -v uv >/dev/null 2>&1; then
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
else
  cmd=(
    python scripts/eval_proof_search.py
    --data-json "$DATA_JSON"
    --model-type "$MODEL_TYPE"
    --ckpt "$CKPT"
    --device "${DEVICE:-auto}"
    --project-path "${PROJECT_PATH:-datasets/april/april_eval_project}"
    --server-timeout "${SERVER_TIMEOUT:-60}"
    --imports "${imports_arr[@]}"
  )
fi

if [[ "${USE_LORA:-0}" == "1" ]]; then
  cmd+=(--use-lora)
fi
if [[ -n "${MAX_THEOREMS:-}" ]]; then
  cmd+=(--max-theorems "$MAX_THEOREMS")
fi

{
  echo "[$(date -Is)] Starting proof search"
  echo "data_json=$DATA_JSON"
  echo "model_type=$MODEL_TYPE"
  echo "ckpt=$CKPT"
  echo "task_id=${SLURM_ARRAY_TASK_ID}"
  echo "job_id=${SLURM_ARRAY_JOB_ID}"
  echo "----------------------------------------"
  srun "${cmd[@]}"
} | tee "$out_file"

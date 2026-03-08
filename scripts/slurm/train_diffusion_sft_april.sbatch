#!/usr/bin/env bash
#SBATCH --job-name=diffusion-sft
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
trap 'echo "ERROR at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

PROJECT_DIR="${REPO_DIR:-/home/tram/cs165-project}"
SCRATCH_BASE="/resnick/scratch/tram"

cd "$PROJECT_DIR"

# -- Environment setup --
module purge
module load cuda/12.2.1-gcc-11.3.1-sdqrj2e git/2.39.3-gcc-11.3.1-zfr3sti

source "$PROJECT_DIR/.venv/bin/activate"
export PATH="$HOME/.elan/bin:$PATH"

# Load project secrets.
if [[ -f "${ENV_FILE:-.env}" ]]; then
  set -a
  source "${ENV_FILE:-.env}"
  set +a
fi

# Scratch directories for caching.
mkdir -p "$SCRATCH_BASE/cache" "$SCRATCH_BASE/tmp" "$SCRATCH_BASE/hf"
export CACHE_DIR="$SCRATCH_BASE/cache"
export TMP_DIR="$SCRATCH_BASE/tmp"
export TMPDIR="$SCRATCH_BASE/tmp"
export RAY_TMPDIR="$SCRATCH_BASE/tmp"
export HF_HOME="$SCRATCH_BASE/hf"

# SSL cert fix for HPC.
CERT_PATH="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
if [[ -n "${CERT_PATH}" ]]; then
  export SSL_CERT_FILE="${SSL_CERT_FILE:-$CERT_PATH}"
  export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$CERT_PATH}"
  export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-$CERT_PATH}"
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Performance tuning.
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# -- Diagnostics --
echo "=============================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Date:       $(date -Iseconds)"
echo "Project:    $PROJECT_DIR"
echo "Python:     $(which python)"
echo "=============================================="
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
nvidia-smi

# -- Training parameters (override via environment) --
MODEL_NAME="${MODEL_NAME:-inclusionAI/LLaDA-MoE-7B-A1B-Instruct}"
DATA_DIR="${DATA_DIR:-datasets/april/leandojo/test}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_BASE/cs165-project/outputs-diffusion-sft}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

mkdir -p "$OUTPUT_DIR" logs/slurm

cmd=(
  python scripts/train_diffusion_sft_april.py
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --model-name "$MODEL_NAME"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRAD_ACCUM"
  --lr "$LR"
  --max-length "$MAX_LENGTH"
  --gradient-checkpointing
  --logging-steps 5
  --warmup-steps 20
)

if [[ "$USE_LORA" == "1" ]]; then
  cmd+=(--use-lora --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA")
fi

echo ""
echo "Running: ${cmd[*]}"
echo ""

srun "${cmd[@]}"

echo ""
echo "Training finished at $(date -Iseconds)"
echo "Output saved to $OUTPUT_DIR"

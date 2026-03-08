#!/usr/bin/env bash
#SBATCH --job-name=overfit200
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
trap 'echo "ERROR at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

PROJECT_DIR="${REPO_DIR:-/home/tram/cs165-project}"
SCRATCH_BASE="/resnick/scratch/tram"

cd "$PROJECT_DIR"

module purge
module load cuda/12.2.1-gcc-11.3.1-sdqrj2e git/2.39.3-gcc-11.3.1-zfr3sti

source "$PROJECT_DIR/.venv/bin/activate"
export PATH="$HOME/.elan/bin:$PATH"

if [[ -f "${ENV_FILE:-.env}" ]]; then
  set -a
  source "${ENV_FILE:-.env}"
  set +a
fi

mkdir -p "$SCRATCH_BASE/cache" "$SCRATCH_BASE/tmp" "$SCRATCH_BASE/hf"
export CACHE_DIR="$SCRATCH_BASE/cache"
export TMP_DIR="$SCRATCH_BASE/tmp"
export TMPDIR="$SCRATCH_BASE/tmp"
export RAY_TMPDIR="$SCRATCH_BASE/tmp"
export HF_HOME="$SCRATCH_BASE/hf"

CERT_PATH="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
if [[ -n "${CERT_PATH}" ]]; then
  export SSL_CERT_FILE="${SSL_CERT_FILE:-$CERT_PATH}"
  export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$CERT_PATH}"
  export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-$CERT_PATH}"
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

echo "=============================================="
echo "OVERFIT TEST: 200 examples, train=test"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Date:       $(date -Iseconds)"
echo "=============================================="
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
nvidia-smi

OUTPUT_DIR="$SCRATCH_BASE/cs165-project/outputs-overfit200"
mkdir -p "$OUTPUT_DIR" logs/slurm

echo ""
echo "=== PHASE 1: TRAINING (100 epochs, fixed gen_length=64, EOS delimiter) ==="
echo ""

python scripts/train_diffusion_sft_april.py \
  --data-dir datasets/april/leandojo/overfit200 \
  --output-dir "$OUTPUT_DIR" \
  --model-name inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --epochs 100 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --lr 5e-5 \
  --max-length 1024 \
  --min-mask-ratio 0.01 \
  --max-mask-ratio 1.0 \
  --gen-length 64 \
  --gradient-checkpointing \
  --logging-steps 5 \
  --warmup-steps 10 \
  --save-steps 500 \
  --use-lora --lora-r 64 --lora-alpha 128

echo ""
echo "=== PHASE 2: DETERMINISTIC INFERENCE (temp=0, block=gen_length, stop at EOS) ==="
echo ""

python scripts/overfit200_eval.py \
  --data-dir datasets/april/leandojo/overfit200 \
  --ckpt-dir "$OUTPUT_DIR" \
  --base-model inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --steps 128 \
  --gen-length 64 \
  --block-length 64 \
  --temperature 0.0

echo ""
echo "=== ALL DONE: $(date -Iseconds) ==="

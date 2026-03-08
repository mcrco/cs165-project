#!/usr/bin/env bash
#SBATCH --job-name=overfit-eval
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

PROJECT_DIR="/home/tram/cs165-project"
SCRATCH_BASE="/resnick/scratch/tram"

cd "$PROJECT_DIR"

module purge
module load cuda/12.2.1-gcc-11.3.1-sdqrj2e git/2.39.3-gcc-11.3.1-zfr3sti

source "$PROJECT_DIR/.venv/bin/activate"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

mkdir -p "$SCRATCH_BASE/cache" "$SCRATCH_BASE/tmp" "$SCRATCH_BASE/hf"
export CACHE_DIR="$SCRATCH_BASE/cache" TMPDIR="$SCRATCH_BASE/tmp" HF_HOME="$SCRATCH_BASE/hf"
CERT_PATH="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
[[ -n "$CERT_PATH" ]] && export SSL_CERT_FILE="$CERT_PATH" REQUESTS_CA_BUNDLE="$CERT_PATH"
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false

echo "=== RE-EVAL with correct inference params ==="
echo "Job: $SLURM_JOB_ID, Node: $(hostname)"
nvidia-smi

echo ""
echo "=== Test 1: temperature=0.7, gen_length=64, block_length=32 (matching DiffusionProver) ==="
python scripts/overfit200_eval.py \
  --data-dir datasets/april/leandojo/overfit200 \
  --ckpt-dir "$SCRATCH_BASE/cs165-project/outputs-overfit200" \
  --base-model inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --steps 128 \
  --gen-length 64 \
  --temperature 0.7

echo ""
echo "=== DONE: $(date -Iseconds) ==="

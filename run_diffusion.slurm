#!/bin/bash
#SBATCH --job-name=lean_diffusion
#SBATCH --output=/resnick/scratch/tram/cs165-project/logs/%j.out
#SBATCH --error=/resnick/scratch/tram/cs165-project/logs/%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu

set -Eeuo pipefail
set -x
trap 'echo "ERROR at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

PROJECT_DIR="/home/tram/cs165-project"
SCRATCH_BASE="/resnick/scratch/tram"
LOG_DIR="$SCRATCH_BASE/cs165-project/logs"

mkdir -p "$LOG_DIR" "$SCRATCH_BASE/cache" "$SCRATCH_BASE/tmp" "$SCRATCH_BASE/hf" "$SCRATCH_BASE/cs165-project/raid"

cd "$PROJECT_DIR"
module load cuda/12.1 python/3.11 git
source "$PROJECT_DIR/.venv/bin/activate"
export PATH="$HOME/.elan/bin:$PATH"
elan toolchain install leanprover/lean4:v4.26.0 || true

if [[ -z "${GITHUB_ACCESS_TOKEN:-}" ]]; then
  echo "GITHUB_ACCESS_TOKEN is not set."
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set."
  exit 1
fi

export CACHE_DIR="$SCRATCH_BASE/cache"
export TMP_DIR="$SCRATCH_BASE/tmp"
export RAY_TMPDIR="$SCRATCH_BASE/tmp"
export RAID_DIR="$SCRATCH_BASE/cs165-project/raid"
export HF_HOME="$SCRATCH_BASE/hf"
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=4
export PANTOGRAPH_TOOLCHAIN_BIN="$HOME/.elan/toolchains/leanprover--lean4---v4.26.0/bin"
export PANTOGRAPH_STDLIB_LEAN_PATH="$HOME/.elan/toolchains/leanprover--lean4---v4.26.0/lib/lean"
export PROOF_ATTEMPTS_LOG="$LOG_DIR/${SLURM_JOB_ID}_proof_attempts.log"

CERT_PATH="$(python -c 'import certifi; print(certifi.where())')"
export SSL_CERT_FILE="$CERT_PATH"
export REQUESTS_CA_BUNDLE="$CERT_PATH"
export CURL_CA_BUNDLE="$CERT_PATH"

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"
nvidia-smi
echo "Proof attempts log: $PROOF_ATTEMPTS_LOG"

python -m lean_dojo_v2.agent.diffusion_agent

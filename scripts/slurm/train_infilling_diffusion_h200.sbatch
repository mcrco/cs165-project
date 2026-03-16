#!/usr/bin/env bash
#SBATCH --job-name=infilling-diffusion
#SBATCH --output=/resnick/groups/tensorlab/myang2/cs165-project/logs/slurm/%x-%j.out
#SBATCH --error=/resnick/groups/tensorlab/myang2/cs165-project/logs/slurm/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
trap 'echo "ERROR at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# auto detect project directory
PROJECT_DIR=""
for candidate in \
  "${REPO_DIR:-}" \
  "${SLURM_SUBMIT_DIR:-}" \
  "/resnick/groups/tensorlab/myang2/cs165-project" \
  "/groups/tensorlab/myang2/cs165-project"
do
  if [[ -n "$candidate" && -f "$candidate/scripts/train_infilling_diffusion.py" ]]; then
    PROJECT_DIR="$candidate"
    break
  fi
done

if [[ -z "$PROJECT_DIR" ]]; then
  echo "Could not locate repository directory. Set REPO_DIR explicitly." >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs/slurm"
cd "$PROJECT_DIR"

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-infilling}"
MAX_DATASET_SIZE="${MAX_DATASET_SIZE:-}"

echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Date:          $(date -Iseconds)"
echo "Project dir:   $PROJECT_DIR"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "MAX_DATASET_SIZE: ${MAX_DATASET_SIZE:-<script-default>}"
echo "=============================================="

nvidia-smi

CMD=(uv run scripts/train_infilling_diffusion.py)
if [[ -n "$MAX_DATASET_SIZE" ]]; then
  CMD+=(--max-train-examples "$MAX_DATASET_SIZE" --max-val-examples "$MAX_DATASET_SIZE")
fi

# Avoid inheriting restrictive CPU bind masks when this batch job is submitted
# from inside an existing salloc/srun shell.
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE
srun --cpu-bind=none "${CMD[@]}"

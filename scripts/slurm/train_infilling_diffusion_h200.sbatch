#!/usr/bin/env bash
#SBATCH --job-name=infilling-diffusion
#SBATCH --output=/resnick/groups/tensorlab/myang2/cs165-project/logs/slurm/%x-%j.out
#SBATCH --error=/resnick/groups/tensorlab/myang2/cs165-project/logs/slurm/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
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
export WANDB_ROOT="${WANDB_ROOT:-/groups/tensorlab/myang2/.wandb}"
export WANDB_DIR="${WANDB_DIR:-$WANDB_ROOT/runs}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$WANDB_ROOT/data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_ROOT/cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$WANDB_ROOT/artifacts}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/groups/tensorlab/myang2/.cache/uv}"
# Ensure local package imports (e.g., lean_dojo_v2.*) resolve in torchrun workers.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p \
  "$WANDB_DIR" \
  "$WANDB_DATA_DIR" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_ARTIFACT_DIR" \
  "$UV_CACHE_DIR"

echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Date:          $(date -Iseconds)"
echo "Project dir:   $PROJECT_DIR"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "WANDB_ROOT:    $WANDB_ROOT"
echo "UV_CACHE_DIR:  $UV_CACHE_DIR"
echo "=============================================="

nvidia-smi

CMD=(uv run scripts/train_infilling_diffusion.py)
CMD+=("$@")

# Avoid inheriting restrictive CPU bind masks when this batch job is submitted
# from inside an existing salloc/srun shell.
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE

# Infer process count from allocated GPUs on this node.
NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-1}"
if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || [[ "$NPROC_PER_NODE" -lt 1 ]]; then
  echo "Invalid NPROC_PER_NODE='$NPROC_PER_NODE' (must be a positive integer)." >&2
  exit 1
fi
echo "Launching $NPROC_PER_NODE process(es) on this node."

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  srun --cpu-bind=none \
    uv run torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    scripts/train_infilling_diffusion.py "$@"
else
  srun --cpu-bind=none "${CMD[@]}"
fi

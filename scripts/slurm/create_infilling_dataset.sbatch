#!/usr/bin/env bash
#SBATCH --job-name=create-infilling
#SBATCH --output=/resnick/groups/tensorlab/myang2/cs165-project/logs/slurm/%x-%j.out
#SBATCH --error=/resnick/groups/tensorlab/myang2/cs165-project/logs/slurm/%x-%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
trap 'echo "ERROR at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

PROJECT_DIR=""
for candidate in \
  "${REPO_DIR:-}" \
  "${SLURM_SUBMIT_DIR:-}" \
  "/resnick/groups/tensorlab/myang2/cs165-project" \
  "/groups/tensorlab/myang2/cs165-project"
do
  if [[ -n "$candidate" && -f "$candidate/datasets/april/create_infilling_dataset.py" ]]; then
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

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "=============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $(hostname)"
echo "Date:        $(date -Iseconds)"
echo "Project dir: $PROJECT_DIR"
echo "=============================================="

srun --cpu-bind=none uv run datasets/april/create_infilling_dataset.py "$@"

#!/usr/bin/env bash
#SBATCH --job-name=diffusion-steps-eval
#SBATCH --output=/resnick/groups/tensorlab/myang2/cs165-project/logs/slurm/%x-%j.out
#SBATCH --error=/resnick/groups/tensorlab/myang2/cs165-project/logs/slurm/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
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
  if [[ -n "$candidate" && -f "$candidate/scripts/eval_diffusion_steps_vs_accuracy.py" ]]; then
    PROJECT_DIR="$candidate"
    break
  fi
done

if [[ -z "$PROJECT_DIR" ]]; then
  echo "Could not locate repository directory. Set REPO_DIR explicitly." >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs/slurm" "$PROJECT_DIR/outputs/benchmarks"
cd "$PROJECT_DIR"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

JSON_OUTPUT="${JSON_OUTPUT:-outputs/benchmarks/diffusion_steps_vs_accuracy_${SLURM_JOB_ID}.json}"
CSV_OUTPUT="${CSV_OUTPUT:-outputs/benchmarks/diffusion_steps_vs_accuracy_${SLURM_JOB_ID}.csv}"

echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Date:          $(date -Iseconds)"
echo "Project dir:   $PROJECT_DIR"
echo "JSON output:   $JSON_OUTPUT"
echo "CSV output:    $CSV_OUTPUT"
echo "=============================================="

nvidia-smi

CMD=(
  uv run scripts/eval_diffusion_steps_vs_accuracy.py
  --json-output "$JSON_OUTPUT"
  --csv-output "$CSV_OUTPUT"
)
CMD+=("$@")

unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE

srun --cpu-bind=none "${CMD[@]}"

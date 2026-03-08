#!/bin/bash
#SBATCH --job-name=lc-small
#SBATCH --output=/home/tram/cs165-project/logs/slurm/lean-convert-small-%j.out
#SBATCH --error=/home/tram/cs165-project/logs/slurm/lean-convert-small-%j.err
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2

export PATH="$HOME/.elan/bin:$PATH"
export ELAN_TOOLCHAIN="leanprover/lean4:v4.27.0"
source /home/tram/cs165-project/.venv/bin/activate
cd /home/tram/cs165-project

PROJECT=/resnick/scratch/tram/cs165-project/april_eval_project
OUTDIR=/resnick/scratch/tram/cs165-project/leandojo_train
FAILDIR=/resnick/scratch/tram/cs165-project/leandojo_failures
mkdir -p "$OUTDIR" "$FAILDIR"

SPLITS=(lme mlme tme)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

echo "=== ${SPLIT^^} conversion (array task $SLURM_ARRAY_TASK_ID, 28 workers) ==="
echo "Node: $(hostname), CPUs: $(nproc)"
echo "Started: $(date)"

python datasets/april/convert_april_to_leandojo.py \
    --input-jsonl "datasets/april/raw/train/${SPLIT}_train.jsonl" \
    --output-json "${OUTDIR}/${SPLIT}_train.json" \
    --project-path "$PROJECT" \
    --jobs 28 \
    --extract-timeout 120 \
    --failure-log "${FAILDIR}/${SPLIT}_failures.jsonl" \
    --print-failures 10

echo "Finished: $(date)"
ls -lh "${OUTDIR}/${SPLIT}_train.json"

#!/bin/bash
#SBATCH --job-name=lc-thme
#SBATCH --output=/home/tram/cs165-project/logs/slurm/lean-convert-thme-%j.out
#SBATCH --error=/home/tram/cs165-project/logs/slurm/lean-convert-thme-%j.err
#SBATCH --partition=beta
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00

export PATH="$HOME/.elan/bin:$PATH"
export ELAN_TOOLCHAIN="leanprover/lean4:v4.27.0"
source /home/tram/cs165-project/.venv/bin/activate
cd /home/tram/cs165-project

PROJECT=/resnick/scratch/tram/cs165-project/april_eval_project
OUTDIR=/resnick/scratch/tram/cs165-project/leandojo_train
FAILDIR=/resnick/scratch/tram/cs165-project/leandojo_failures
mkdir -p "$OUTDIR" "$FAILDIR"

echo "=== THME conversion (148K examples, 56 workers) ==="
echo "Node: $(hostname), CPUs: $(nproc)"
echo "Started: $(date)"

python datasets/april/convert_april_to_leandojo.py \
    --input-jsonl datasets/april/raw/train/thme_train.jsonl \
    --output-json "${OUTDIR}/thme_train.json" \
    --project-path "$PROJECT" \
    --jobs 56 \
    --extract-timeout 120 \
    --failure-log "${FAILDIR}/thme_failures.jsonl" \
    --print-failures 10

echo "Finished: $(date)"
ls -lh "${OUTDIR}/thme_train.json"

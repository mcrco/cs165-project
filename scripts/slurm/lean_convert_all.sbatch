#!/bin/bash
#SBATCH --job-name=lean-convert
#SBATCH --output=/home/tram/cs165-project/logs/slurm/lean-convert-%j.out
#SBATCH --error=/home/tram/cs165-project/logs/slurm/lean-convert-%j.err
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

export PATH="$HOME/.elan/bin:$PATH"
export ELAN_TOOLCHAIN="leanprover/lean4:v4.27.0"
source /home/tram/cs165-project/.venv/bin/activate
cd /home/tram/cs165-project

PROJECT=/resnick/scratch/tram/cs165-project/april_eval_project
OUTDIR=/resnick/scratch/tram/cs165-project/leandojo_train
FAILDIR=/resnick/scratch/tram/cs165-project/leandojo_failures
JOBS=28

mkdir -p "$OUTDIR" "$FAILDIR"

echo "=== Full Lean-based APRIL conversion ==="
echo "Node: $(hostname), CPUs: $(nproc), Workers: $JOBS"
echo "Lean: $(lean --version 2>&1 | head -1)"
echo "Started: $(date)"
echo ""

for SPLIT in lme mlme tme thme; do
    INPUT="datasets/april/raw/train/${SPLIT}_train.jsonl"
    OUTPUT="${OUTDIR}/${SPLIT}_train.json"
    FAILURES="${FAILDIR}/${SPLIT}_failures.jsonl"

    if [ -f "$OUTPUT" ]; then
        echo "=== SKIPPING $SPLIT (output already exists) ==="
        continue
    fi

    echo "=== Processing $SPLIT ==="
    echo "  Input: $INPUT"
    echo "  Started: $(date)"

    python datasets/april/convert_april_to_leandojo.py \
        --input-jsonl "$INPUT" \
        --output-json "$OUTPUT" \
        --project-path "$PROJECT" \
        --jobs "$JOBS" \
        --extract-timeout 120 \
        --failure-log "$FAILURES" \
        --print-failures 10

    echo "  Finished: $(date)"
    ls -lh "$OUTPUT"
    echo ""
done

echo "=== ALL DONE: $(date) ==="
ls -lh "$OUTDIR/"

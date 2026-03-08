#!/bin/bash
#SBATCH --job-name=lean-convert
#SBATCH --output=/home/tram/cs165-project/logs/slurm/lean-convert-%j.out
#SBATCH --error=/home/tram/cs165-project/logs/slurm/lean-convert-%j.err
#SBATCH --partition=beta
#SBATCH --cpus-per-task=96
#SBATCH --mem=192G
#SBATCH --time=1-12:00:00

export PATH="$HOME/.elan/bin:$PATH"
export ELAN_TOOLCHAIN="leanprover/lean4:v4.27.0"

source /home/tram/cs165-project/.venv/bin/activate

cd /home/tram/cs165-project

PROJECT=/resnick/scratch/tram/cs165-project/april_eval_project
OUTDIR=/resnick/scratch/tram/cs165-project/leandojo_train
FAILDIR=/resnick/scratch/tram/cs165-project/leandojo_failures
JOBS=88

mkdir -p "$OUTDIR" "$FAILDIR"

echo "Starting full Lean-based APRIL conversion"
echo "Node: $(hostname), CPUs: $(nproc), Jobs: $JOBS"
echo "Project: $PROJECT"
echo "Toolchain: $(lean --version 2>/dev/null || echo 'lean not in PATH')"
echo "Lake: $(which lake)"
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
    echo "  Output: $OUTPUT"
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
    echo ""
done

echo "=== ALL DONE ==="
echo "Output files:"
ls -lh "$OUTDIR/"
echo ""
echo "Failure logs:"
ls -lh "$FAILDIR/"

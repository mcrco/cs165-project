#!/bin/bash
#SBATCH --job-name=lc-test
#SBATCH --output=/home/tram/cs165-project/logs/slurm/lean-convert-test-%j.out
#SBATCH --error=/home/tram/cs165-project/logs/slurm/lean-convert-test-%j.err
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00

export PATH="$HOME/.elan/bin:$PATH"
export ELAN_TOOLCHAIN="leanprover/lean4:v4.27.0"
source /home/tram/cs165-project/.venv/bin/activate
cd /home/tram/cs165-project

PROJECT=/resnick/scratch/tram/cs165-project/april_eval_project
OUTDIR=/resnick/scratch/tram/cs165-project/leandojo_test_run
mkdir -p "$OUTDIR"

echo "=== TEST RUN: 100 examples, 28 workers ==="
echo "Node: $(hostname), CPUs: $(nproc)"
echo "Lean: $(lean --version 2>&1 | head -1)"
echo "Lake: $(which lake)"
echo "Started: $(date)"
echo ""

python datasets/april/convert_april_to_leandojo.py \
    --input-jsonl datasets/april/raw/train/lme_train.jsonl \
    --output-json "${OUTDIR}/lme_test_100.json" \
    --project-path "$PROJECT" \
    --max-examples 100 \
    --jobs 28 \
    --extract-timeout 120 \
    --failure-log "${OUTDIR}/lme_test_100_failures.jsonl" \
    --print-failures 10

echo ""
echo "Finished: $(date)"
echo ""

# Validate the output
python -c "
import json
data = json.load(open('${OUTDIR}/lme_test_100.json'))
print(f'Converted: {len(data)} theorems')
total_tactics = sum(len(d['traced_tactics']) for d in data)
print(f'Total tactic pairs: {total_tactics}')

# Check quality
for i, item in enumerate(data[:3]):
    tt = item['traced_tactics'][0]
    print(f'')
    print(f'Example {i+1}: {item[\"full_name\"]}')
    print(f'  state_before: {tt[\"state_before\"][:150]}...')
    print(f'  tactic: {tt[\"tactic\"].splitlines()[0][:80]}')
    has_inst = 'inst' in tt['state_before']
    has_turnstile = '⊢' in tt['state_before']
    print(f'  has inst✝: {has_inst}, has ⊢: {has_turnstile}')

# Check for any corrupted entries
errors = 0
for item in data:
    for tt in item['traced_tactics']:
        if not tt.get('state_before') or '⊢' not in tt['state_before']:
            errors += 1
        if not tt.get('tactic') or not tt['tactic'].strip():
            errors += 1
print(f'')
print(f'Validation errors: {errors}')
print(f'SUCCESS' if errors == 0 else 'ISSUES FOUND')
"

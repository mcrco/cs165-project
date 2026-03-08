#!/bin/bash
#SBATCH --job-name=convert-april
#SBATCH --output=/home/tram/cs165-project/logs/slurm/convert-april-%j.out
#SBATCH --error=/home/tram/cs165-project/logs/slurm/convert-april-%j.err
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00

source /home/tram/cs165-project/.venv/bin/activate

cd /home/tram/cs165-project
echo "Starting fast APRIL data conversion..."
echo ""

python datasets/april/fast_convert_april.py \
    --input-dir datasets/april/raw/train \
    --output-dir datasets/april/leandojo/train

echo ""
echo "Output files:"
ls -lh datasets/april/leandojo/train/
echo ""
echo "Done!"

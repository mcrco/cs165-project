#!/bin/bash
#SBATCH --job-name=diffusion-inference-test
#SBATCH --output=/home/tram/cs165-project/logs/slurm/inference-test-%j.out
#SBATCH --error=/home/tram/cs165-project/logs/slurm/inference-test-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

module purge
module load cuda/12.6.3-gcc-13.3.0
module load git/2.47.1-gcc-13.3.0

source /home/tram/cs165-project/.venv/bin/activate
source /home/tram/cs165-project/.env

export HF_HOME=/resnick/scratch/tram/hf
export HF_HUB_CACHE=/resnick/scratch/tram/hf/hub
export TRANSFORMERS_CACHE=/resnick/scratch/tram/hf/hub
export TMPDIR=/resnick/scratch/tram/tmp
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Running inference test..."
echo ""

cd /home/tram/cs165-project
python scripts/quick_inference_test.py

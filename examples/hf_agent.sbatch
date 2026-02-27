#!/usr/bin/env bash
#SBATCH --job-name=leandojo-hf-agent
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --account=myang2

set -euo pipefail

module purge
module load python/3.11.6-gcc-13.2.0

cd /home/marco/cs165-project
source .venv/bin/activate

# HF/HPC TLS + Xet workaround
source ./hf_var_fixes.sh

srun python -m lean_dojo_v2.agent.hf_agent

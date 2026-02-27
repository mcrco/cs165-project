#!/usr/bin/env bash
#SBATCH --job-name=leandojo-hf-agent
#SBATCH --output=$HOME/cs165-project/logs/%x-%j.out
#SBATCH --error=$HOME/cs165-project/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

module purge
module load python/3.11.6-gcc-13.2.0

cd $HOME/cs165-project
mkdir -p logs/
source .venv/bin/activate

# HF/HPC TLS + Xet workaround
source ./hf_var_fixes.sh

srun python -m lean_dojo_v2.agent.hf_agent

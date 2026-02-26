#!/bin/bash
#SBATCH --job-name=lean_baseline
#SBATCH --output=/resnick/scratch/tram/cs165-project/logs/%j.out
#SBATCH --error=/resnick/scratch/tram/cs165-project/logs/%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu

# --- Commands below run ON THE COMPUTE NODE ---

# 1. Load Modules & Activate Environment
module load cuda/12.1 python/3.11 git
source .venv/bin/activate
elan toolchain install leanprover/lean4:v4.27.0
elan default leanprover/lean4:v4.27.0


# 2. Set Environment Variables (CRITICAL for data management and GitHub access)
export GITHUB_ACCESS_TOKEN="GITHUB TOKEN"
export HF_TOKEN="HF TOKEN"
export CACHE_DIR="/resnick/scratch/tram/cache"      # Redirect cache to scratch
export RAY_TMPDIR="/resnick/scratch/tram/tmp"       # Redirect Ray temp files

# 3. Ensure scratch directories exist (important if your job starts before you run Step 1 manually)
mkdir -p /resnick/scratch/tram/cs165-project/logs
mkdir -p /resnick/scratch/tram/tmp

# 4. Navigate to your project directory on the compute node
cd /home/tram/cs165-project

# 5. Run the Baseline
# This runs the main entry point we found in the code
python -m lean_dojo_v2.lean_agent

# 6. Log cloned repo toolchain (after LeanDojo-v2 has potentially cloned)
echo "--- Finding lean-toolchain in RAID directory after LeanAgent run ---"
find /resnick/scratch/tram/cs165-project/raid -name "lean-toolchain"

#!/bin/bash
#SBATCH -p expansion
#SBATCH --time=12:00:00
#SBATCH -c 16
#SBATCH --mem=64G

# Copies base_dependency_repo into april_eval_project, then adds each of the proofs
# in APRIL dataset as its own file in the new repo.

set -euo pipefail

# Determine the script's location.
# When running via sbatch, BASH_SOURCE[0] points to a copy in /var/spool/slurmd/,
# so we detect that case and fall back to the current working directory (where
# the script was submitted from).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${BASH_SOURCE[0]}" == /var/spool/slurmd/* && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
fi

cd "${SCRIPT_DIR}"

echo "[materialize-repo] Starting APRIL materialization at $(date -Iseconds)"

WORK_REPO="${SCRIPT_DIR}/april_eval_project"
mkdir -p "${SCRIPT_DIR}/failures"

if [[ -e "${WORK_REPO}" ]]; then
  rm -rf "${WORK_REPO}"
fi
git clone -q "${SCRIPT_DIR}/base_dependency_repo" "${WORK_REPO}"
if [[ ! -e "${WORK_REPO}/.git" ]]; then
  echo "[materialize-repo] WORK_REPO is not a git repository: ${WORK_REPO}"
  exit 1
fi

git -C "${WORK_REPO}" config user.email "materializer@local"
git -C "${WORK_REPO}" config user.name "Materializer"
echo "[materialize-repo] Using WORK_REPO=${WORK_REPO}"

mapfile -t inputs < <(find raw -type f -name '*.jsonl' | sort)

if [[ ${#inputs[@]} -eq 0 ]]; then
  echo "[materialize-repo] No input files found under raw/."
  echo "[materialize-repo] Run download_april_dataset.py first."
  exit 1
fi

for input in "${inputs[@]}"; do
  rel="${input#raw/}"
  stem="${rel%.jsonl}"

  manifest="${SCRIPT_DIR}/materialized/${stem}.manifest.jsonl"
  failure_log="${SCRIPT_DIR}/failures/${stem}.failures.jsonl"

  mkdir -p "$(dirname "${manifest}")" "$(dirname "${failure_log}")"

  echo "[materialize-repo] ${input} -> ${manifest}"

  materialize_base_cmd="uv run python materialize_april_repo.py --project-path ${WORK_REPO} --module-prefix AprilEval.Materialized"
  if [[ -n "${MAX_EXAMPLES:-}" ]]; then
    materialize_base_cmd="${materialize_base_cmd} --max-examples ${MAX_EXAMPLES}"
  fi

  materialize_cmd=(
    uv run python materialize_april_repo.py
    --input-jsonl "${input}"
    --project-path "${WORK_REPO}"
    --manifest-path "${manifest}"
    --module-prefix "AprilEval.Materialized"
  )
  if [[ -n "${MAX_EXAMPLES:-}" ]]; then
    materialize_cmd+=(--max-examples "${MAX_EXAMPLES}")
  fi
  if ! "${materialize_cmd[@]}"; then
    echo "{\"reason\":\"materialize_failed\",\"input\":\"${input}\"}" > "${failure_log}"
    continue
  fi
done

git -C "${WORK_REPO}" add -A
if git -C "${WORK_REPO}" diff --cached --quiet; then
  echo "[materialize-repo] no materialized changes to commit"
else
  git -C "${WORK_REPO}" commit -q -m "${materialize_base_cmd}"
  echo "[materialize-repo] Committed all materialized changes"
fi

echo "[materialize-repo] Done at $(date -Iseconds)"

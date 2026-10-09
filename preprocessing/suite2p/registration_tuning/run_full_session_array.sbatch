#!/bin/bash
#SBATCH --job-name=s2p-full
#SBATCH --output=logs/s2p_full_%A_%a.out
#SBATCH --error=logs/s2p_full_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: run_full_session_array.sbatch MANIFEST REPO_ROOT PYTHON_EXECUTABLE" >&2
  exit 2
fi

manifest=$1
repo_root=$2
python_executable=$3

export PYTHONNOUSERSITE=1
"${python_executable}" "${repo_root}/preprocessing/suite2p/registration_tuning/run_candidate.py" \
  --manifest "${manifest}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --device cuda

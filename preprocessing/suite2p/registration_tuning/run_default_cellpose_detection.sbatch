#!/bin/bash
#SBATCH --job-name=s2p-cpsam
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: run_default_cellpose_detection.sbatch RUN_DIR REPO_ROOT PYTHON_EXECUTABLE" >&2
  exit 2
fi

run_dir=$1
repo_root=$2
python_executable=$3
export PYTHONNOUSERSITE=1

"${python_executable}" \
  "${repo_root}/preprocessing/suite2p/registration_tuning/run_default_cellpose_detection.py" \
  --run-dir "${run_dir}" \
  --device cuda

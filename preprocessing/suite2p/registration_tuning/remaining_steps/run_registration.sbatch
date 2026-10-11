#!/bin/bash
#SBATCH --job-name=s2p-one-reg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: run_registration.sbatch MANIFEST WORKFLOW_DIR PYTHON_EXECUTABLE" >&2
  exit 2
fi

manifest=$1
workflow_dir=$2
python_executable=$3
export PYTHONNOUSERSITE=1

"${python_executable}" "${workflow_dir}/run_candidate.py" \
  --manifest "${manifest}" \
  --task-id 0 \
  --device cuda

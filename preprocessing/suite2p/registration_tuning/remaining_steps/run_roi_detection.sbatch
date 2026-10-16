#!/bin/bash
#SBATCH --job-name=s2p-one-roi
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: run_roi_detection.sbatch RUN_DIR SCRIPT_DIR PYTHON_EXECUTABLE" >&2
  exit 2
fi

run_dir=$1
script_dir=$2
python_executable=$3
suite2p_dir=$(cd "${script_dir}/../.." && pwd)
pipeline_dir="${suite2p_dir}/full_session_pipeline"
export PYTHONNOUSERSITE=1

"${python_executable}" "${pipeline_dir}/run_roi_detection.py" \
  --run-dir "${run_dir}" \
  --device cuda

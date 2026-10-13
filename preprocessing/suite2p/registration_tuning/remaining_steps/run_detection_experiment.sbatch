#!/bin/bash
#SBATCH --job-name=s2p-detect
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: run_detection_experiment.sbatch RUN_DIR SCRIPT_DIR PYTHON_EXECUTABLE EXPERIMENT_NAME [OPTIONS...]" >&2
  exit 2
fi

run_dir=$1
script_dir=$2
python_executable=$3
experiment_name=$4
shift 4
export PYTHONNOUSERSITE=1

"${python_executable}" "${script_dir}/run_detection_experiment.py" \
  --run-dir "${run_dir}" \
  --experiment-name "${experiment_name}" \
  --device cuda \
  "$@"

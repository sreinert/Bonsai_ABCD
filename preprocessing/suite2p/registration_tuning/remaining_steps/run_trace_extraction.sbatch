#!/bin/bash
#SBATCH --job-name=s2p-one-traces
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: run_trace_extraction.sbatch RUN_DIR SCRIPT_DIR PYTHON_EXECUTABLE [DETECTION_DIR] [OUTPUT_DIR]" >&2
  exit 2
fi

run_dir=$1
script_dir=$2
python_executable=$3
suite2p_dir=$(cd "${script_dir}/../.." && pwd)
pipeline_dir="${suite2p_dir}/full_session_pipeline"
detection_dir=${4:-}
output_dir=${5:-}
export PYTHONNOUSERSITE=1

arguments=(--run-dir "${run_dir}" --device cuda)
if [[ -n "${detection_dir}" ]]; then
  arguments+=(--detection-dir "${detection_dir}")
fi
if [[ -n "${output_dir}" ]]; then
  arguments+=(--output-dir "${output_dir}")
fi

"${python_executable}" "${pipeline_dir}/run_trace_extraction.py" "${arguments[@]}"

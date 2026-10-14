#!/bin/bash
#SBATCH --job-name=s2p-full-session
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 MANIFEST RUN_DIR SCRIPT_DIR PYTHON_EXECUTABLE START_STAGE" >&2
  exit 2
fi

manifest=$1
run_dir=$2
script_dir=$3
python_executable=$4
start_stage=$5
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
scratch_root=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
export MPLCONFIGDIR="${scratch_root}/matplotlib-full-session-${SLURM_JOB_ID:-$$}"
mkdir -p "${MPLCONFIGDIR}"

"${python_executable}" "${script_dir}/run_full_session_pipeline.py" \
  --manifest "${manifest}" \
  --run-dir "${run_dir}" \
  --script-dir "${script_dir}" \
  --python-executable "${python_executable}" \
  --start-stage "${start_stage}"

#!/bin/bash
#SBATCH --job-name=nwb-align
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=64G

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 MANIFEST REPO_ROOT PROJECT_ROOT PYTHON_EXECUTABLE" >&2
  exit 2
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set; submit this script as a Slurm array." >&2
  exit 2
fi

manifest=$1
repo_root=$2
project_root=$3
python_executable=$4

cd "${repo_root}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

scratch_root=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
export MPLCONFIGDIR="${scratch_root}/matplotlib-nwb-${SLURM_JOB_ID:-$$}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"

options=()
if [[ "${FORCE_REBUILD:-false}" == true ]]; then
  options+=(--force)
fi

"${python_executable}" -m nwb_barcoding.batch_sequence_compression \
  --project-root "${project_root}" \
  --manifest "${manifest}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  "${options[@]}"

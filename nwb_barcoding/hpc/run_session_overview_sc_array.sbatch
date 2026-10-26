#!/bin/bash
#SBATCH --job-name=session-overview
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
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
export MPLCONFIGDIR="${scratch_root}/matplotlib-overview-${SLURM_JOB_ID:-$$}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"

"${python_executable}" -m analysis.sequence_compression.batch_session_overview_sc \
  --project-root "${project_root}" \
  --manifest "${manifest}" \
  --task-id "${SLURM_ARRAY_TASK_ID}"

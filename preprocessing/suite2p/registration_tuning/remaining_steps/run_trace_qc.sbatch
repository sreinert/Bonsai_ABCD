#!/bin/bash
#SBATCH --job-name=s2p-trace-qc
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=16G

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: run_trace_qc.sbatch TRACE_DIR SCRIPT_DIR PYTHON_EXECUTABLE [QC OPTIONS...]" >&2
  exit 2
fi

trace_dir=$1
script_dir=$2
python_executable=$3
shift 3

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
scratch_root=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
export MPLCONFIGDIR="${scratch_root}/matplotlib-trace-qc-${SLURM_JOB_ID:-$$}"
mkdir -p "${MPLCONFIGDIR}"

"${python_executable}" "${script_dir}/generate_trace_qc.py" \
  --trace-dir "${trace_dir}" \
  "$@"

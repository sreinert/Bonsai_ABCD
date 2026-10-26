#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 PROJECT_ROOT SUBJECT [PYTHON_EXECUTABLE]" >&2
  echo "SUBJECT may be 07, sub-07, or a comma-separated list such as sub-02,sub-07." >&2
  echo "The default conda environment is suite2p-reg-1.1.0." >&2
  exit 2
fi

project_root=$(cd "$1" && pwd)
subjects_csv=$2
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
overview_environment=${SESSION_OVERVIEW_ENV:-suite2p-reg-1.1.0}
if [[ $# -eq 3 ]]; then
  python_executable=$3
elif [[ -n "${PYTHON_EXECUTABLE:-}" ]]; then
  python_executable=${PYTHON_EXECUTABLE}
else
  if [[ "${CONDA_DEFAULT_ENV:-}" != "${overview_environment}" ]]; then
    if type module >/dev/null 2>&1; then
      module load mamba
    fi
    source activate "${overview_environment}"
  fi
  python_executable=$(command -v python)
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required; run this launcher on a Slurm submission host." >&2
  exit 2
fi
if [[ ! -x "${python_executable}" ]]; then
  echo "Python executable is not executable: ${python_executable}" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
"${python_executable}" - <<'PY'
import importlib
import importlib.metadata
import sys

if sys.version_info[:2] != (3, 11):
    raise RuntimeError(
        f"suite2p-reg-1.1.0 requires Python 3.11, but {sys.executable} is "
        f"Python {sys.version.split()[0]}"
    )
suite2p_version = importlib.metadata.version("suite2p")
if suite2p_version != "1.1.0":
    raise RuntimeError(
        f"session_overview_sc requires suite2p==1.1.0, found {suite2p_version} "
        f"using {sys.executable}"
    )
required = (
    "aeon", "h5py", "matplotlib", "numpy", "pandas", "pynwb", "scipy",
    "seaborn", "suite2p", "torch",
)
for name in required:
    importlib.import_module(name)
print("Validated overview Python:", sys.executable)
print("Validated Suite2p version:", suite2p_version)
PY

manifest_root=${project_root}/analysis/cohort2/.session_overview_manifests
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
manifest=${manifest_root}/overview_sessions_${timestamp}.csv
log_dir=${manifest_root}/logs
mkdir -p "${log_dir}"

IFS=',' read -r -a selected_subjects <<< "${subjects_csv}"
manifest_options=()
for subject in "${selected_subjects[@]}"; do
  subject=${subject//[[:space:]]/}
  if [[ -n "${subject}" ]]; then
    manifest_options+=(--include-subject "${subject}")
  fi
done
if [[ ${#manifest_options[@]} -eq 0 ]]; then
  echo "SUBJECT did not contain a subject identifier." >&2
  exit 2
fi

"${python_executable}" -m analysis.sequence_compression.batch_session_overview_sc \
  --project-root "${project_root}" \
  --write-manifest "${manifest}" \
  "${manifest_options[@]}"

task_count=$(($(wc -l < "${manifest}") - 1))
if [[ ${task_count} -eq 0 ]]; then
  echo "No complete aligned sessions were found for: ${subjects_csv}"
  exit 0
fi

last_task=$((task_count - 1))
max_concurrent=${MAX_CONCURRENT:-4}
array_spec=${TASK_RANGE:-0-${last_task}%${max_concurrent}}
partition=${SLURM_PARTITION:-cpu}

sbatch_options=(
  --partition="${partition}"
  --array="${array_spec}"
  --chdir="${repo_root}"
  --output="${log_dir}/session_overview_%A_%a.out"
  --error="${log_dir}/session_overview_%A_%a.err"
)
if [[ -n "${SLURM_MAIL_USER:-}" ]]; then
  sbatch_options+=(--mail-user="${SLURM_MAIL_USER}" --mail-type=BEGIN,END,FAIL)
fi

submission=$(sbatch --parsable \
  "${sbatch_options[@]}" \
  "${repo_root}/nwb_barcoding/hpc/run_session_overview_sc_array.sbatch" \
  "${manifest}" "${repo_root}" "${project_root}" "${python_executable}")
job_id=${submission%%;*}

echo "Submitted ${task_count} overview session(s) as Slurm array job ${job_id}."
echo "Subjects: ${subjects_csv}"
echo "Environment: ${overview_environment}"
echo "Python: ${python_executable}"
echo "Manifest: ${manifest}"
echo "Array: ${array_spec}"
echo "Logs: ${log_dir}/session_overview_${job_id}_<task>.{out,err}"
echo "Overview output: ${project_root}/analysis/cohort2"

#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 PROJECT_ROOT SUBJECT [PYTHON_EXECUTABLE]" >&2
  echo "SUBJECT may be 07, sub-07, or a comma-separated list such as sub-02,sub-07." >&2
  echo "The default environment is /ceph/python_envs/\$USER/nwb_barcoding." >&2
  exit 2
fi

project_root=$(cd "$1" && pwd)
subjects_csv=$2
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
environment_user=${USER:-$(id -un)}
default_environment=/ceph/python_envs/${environment_user}/nwb_barcoding
python_executable=${3:-${PYTHON_EXECUTABLE:-${default_environment}/bin/python}}

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
import sys

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(
        f"nwb_barcoding requires Python 3.12, but {sys.executable} is "
        f"Python {sys.version.split()[0]}"
    )
required = ("aeon", "h5py", "matplotlib", "neuroconv", "numpy", "pynwb", "scipy", "suite2p")
for name in required:
    importlib.import_module(name)
print("Validated batch Python:", sys.executable)
PY

manifest_root=${project_root}/analysis/cohort2/.nwb_barcoding_manifests
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
manifest=${manifest_root}/alignment_sessions_${timestamp}.csv
log_dir=${manifest_root}/alignment_logs
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

"${python_executable}" -m nwb_barcoding.batch_sequence_compression \
  --project-root "${project_root}" \
  --write-manifest "${manifest}" \
  "${manifest_options[@]}"

task_count=$(($(wc -l < "${manifest}") - 1))
if [[ ${task_count} -eq 0 ]]; then
  echo "No completed Suite2p sessions were found for: ${subjects_csv}"
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
  --output="${log_dir}/nwb_align_%A_%a.out"
  --error="${log_dir}/nwb_align_%A_%a.err"
)
if [[ -n "${SLURM_MAIL_USER:-}" ]]; then
  sbatch_options+=(--mail-user="${SLURM_MAIL_USER}" --mail-type=BEGIN,END,FAIL)
fi

submission=$(sbatch --parsable \
  "${sbatch_options[@]}" \
  "${script_dir}/run_sequence_compression_array.sbatch" \
  "${manifest}" "${repo_root}" "${project_root}" "${python_executable}")
job_id=${submission%%;*}

echo "Submitted ${task_count} alignment session(s) as Slurm array job ${job_id}."
echo "Subjects: ${subjects_csv}"
echo "Manifest: ${manifest}"
echo "Array: ${array_spec}"
echo "Logs: ${log_dir}/nwb_align_${job_id}_<task>.{out,err}"
echo "Aligned output: ${project_root}/BehaviourAligned_robust/cohort2"

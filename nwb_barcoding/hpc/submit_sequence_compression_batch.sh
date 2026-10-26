#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 PROJECT_ROOT [PYTHON_EXECUTABLE]" >&2
  echo "Activate the NWB/preprocess environment first, or pass its Python executable." >&2
  exit 2
fi

project_root=$(cd "$1" && pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
python_executable=${2:-${PYTHON_EXECUTABLE:-$(command -v python)}}

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

required = ("aeon", "h5py", "matplotlib", "neuroconv", "numpy", "pynwb", "scipy", "suite2p")
for name in required:
    importlib.import_module(name)
print("Validated batch Python:", sys.executable)
PY

manifest_root=${project_root}/analysis/cohort2/.nwb_barcoding_manifests
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
manifest=${manifest_root}/sessions_${timestamp}.csv
log_dir=${manifest_root}/logs
mkdir -p "${log_dir}"

subjects_csv=${SUBJECTS:-sub-07}
IFS=',' read -r -a selected_subjects <<< "${subjects_csv}"
manifest_options=()
for subject in "${selected_subjects[@]}"; do
  subject=${subject//[[:space:]]/}
  if [[ -n "${subject}" ]]; then
    manifest_options+=(--include-subject "${subject}")
  fi
done
if [[ ${#manifest_options[@]} -eq 0 ]]; then
  echo "SUBJECTS did not contain a subject identifier." >&2
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

echo "Submitted ${task_count} session(s) as Slurm array job ${job_id}."
echo "Subjects: ${subjects_csv}"
echo "Manifest: ${manifest}"
echo "Array: ${array_spec}"
echo "Logs: ${log_dir}/nwb_align_${job_id}_<task>.{out,err}"
echo "Aligned output: ${project_root}/BehaviourAligned_robust/cohort2"
echo "Overview output: ${project_root}/analysis/cohort2"

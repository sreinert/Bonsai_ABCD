#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
project_root=${PROJECT_ROOT:-AtAp_20260119_SequenceCompression}

if [[ -n "${PYTHON_EXECUTABLE:-}" ]]; then
  python_executable=${PYTHON_EXECUTABLE}
else
  python_executable=$(command -v python)
fi

projects_root=${MRSIC_PROJECTS_ROOT:-}
if [[ -z "${projects_root}" ]]; then
  if [[ -d /ceph/mrsic_flogel/public/projects ]]; then
    projects_root=/ceph/mrsic_flogel/public/projects
  else
    projects_root=/Volumes/mrsic_flogel/public/projects
  fi
fi
if [[ "${project_root}" = /* ]]; then
  resolved_project=${project_root}
else
  resolved_project=${projects_root}/${project_root}
fi

manifest_dir=${resolved_project}/processed/cohort2/.suite2p_manifests
mkdir -p "${manifest_dir}/logs"
manifest=${manifest_dir}/full_sessions_$(date +%Y%m%dT%H%M%S).csv

"${python_executable}" "${script_dir}/make_full_session_tasks.py" \
  --project-root "${project_root}" \
  --manifest "${manifest}"

if [[ ! -s "${manifest}" ]]; then
  echo "No manifest was created; there is nothing to submit."
  exit 0
fi

task_count=$(($(wc -l < "${manifest}") - 1))
last_task=$((task_count - 1))
max_concurrent=${MAX_CONCURRENT:-4}
array_spec=${TASK_RANGE:-0-${last_task}%${max_concurrent}}

export PYTHONNOUSERSITE=1
"${python_executable}" - <<'PY'
import importlib.metadata
import sys

version = importlib.metadata.version("suite2p")
if version != "1.1.0":
    raise RuntimeError(
        f"Submission requires suite2p==1.1.0, found {version} using {sys.executable}"
    )
print("Validated Suite2p version:", version)
PY

echo "Manifest: ${manifest}"
echo "Logs: ${manifest_dir}/logs/s2p_full_<job>_<task>.{out,err}"

sbatch \
  --array="${array_spec}" \
  --chdir="${manifest_dir}" \
  --output="${manifest_dir}/logs/s2p_full_%A_%a.out" \
  --error="${manifest_dir}/logs/s2p_full_%A_%a.err" \
  "${script_dir}/run_full_session_array.sbatch" \
  "${manifest}" \
  "${repo_root}" \
  "${python_executable}"

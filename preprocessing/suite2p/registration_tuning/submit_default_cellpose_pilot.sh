#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
project_root=${PROJECT_ROOT:-AtAp_20260119_SequenceCompression}
source_task_id=${SOURCE_TASK_ID:-0}
python_executable=${PYTHON_EXECUTABLE:-$(command -v python)}

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
if [[ -n "${SOURCE_MANIFEST:-}" ]]; then
  source_manifest=${SOURCE_MANIFEST}
else
  mapfile -t full_manifests < <(ls -1t "${manifest_dir}"/full_sessions_*.csv 2>/dev/null || true)
  source_manifest=${full_manifests[0]:-}
fi
if [[ -z "${source_manifest}" || ! -f "${source_manifest}" ]]; then
  echo "No full-session source manifest found under ${manifest_dir}" >&2
  exit 1
fi

pilot_manifest=${manifest_dir}/default_cpsam_pilot_$(date +%Y%m%dT%H%M%S).csv
"${python_executable}" "${script_dir}/make_default_cellpose_pilot_task.py" \
  --source-manifest "${source_manifest}" \
  --source-task-id "${source_task_id}" \
  --manifest "${pilot_manifest}"

run_dir=$("${python_executable}" - "${pilot_manifest}" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    print(next(csv.DictReader(handle))["run_dir"])
PY
)

"${python_executable}" - <<'PY'
import importlib.metadata
for package in ("suite2p", "cellpose", "torch"):
    print(f"{package}=={importlib.metadata.version(package)}")
if importlib.metadata.version("suite2p") != "1.1.0":
    raise RuntimeError("The default Cellpose pilot requires suite2p==1.1.0")
PY

registration_submission=$(sbatch --parsable \
  --array=0 \
  --chdir="${manifest_dir}" \
  --output="${manifest_dir}/logs/default_cpsam_registration_%A_%a.out" \
  --error="${manifest_dir}/logs/default_cpsam_registration_%A_%a.err" \
  "${script_dir}/run_full_session_array.sbatch" \
  "${pilot_manifest}" "${repo_root}" "${python_executable}")
registration_job=${registration_submission%%;*}

detection_submission=$(sbatch --parsable \
  --dependency="afterok:${registration_job}" \
  --chdir="${manifest_dir}" \
  --output="${manifest_dir}/logs/default_cpsam_detection_%j.out" \
  --error="${manifest_dir}/logs/default_cpsam_detection_%j.err" \
  "${script_dir}/run_default_cellpose_detection.sbatch" \
  "${run_dir}" "${repo_root}" "${python_executable}")
detection_job=${detection_submission%%;*}

echo "Source manifest: ${source_manifest}"
echo "Pilot manifest:  ${pilot_manifest}"
echo "Pilot output:    ${run_dir}"
echo "Registration job: ${registration_job}"
echo "Detection job:    ${detection_job} (afterok:${registration_job})"

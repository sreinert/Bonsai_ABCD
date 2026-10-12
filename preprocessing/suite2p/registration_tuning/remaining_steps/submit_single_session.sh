#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 SOURCE_MANIFEST SESSION_INDEX [OUTPUT_ROOT]" >&2
  echo "Run: python prepare_session.py --source-manifest SOURCE_MANIFEST --list" >&2
  exit 2
fi

source_manifest=$1
session_index=$2
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workflow_dir=$(cd "${script_dir}/.." && pwd)
python_executable=${PYTHON_EXECUTABLE:-$(command -v python)}

if [[ ! -f "${source_manifest}" ]]; then
  echo "Source manifest not found: ${source_manifest}" >&2
  exit 2
fi
source_manifest=$(cd "$(dirname "${source_manifest}")" && pwd)/$(basename "${source_manifest}")
output_root=${3:-$(dirname "${source_manifest}")/remaining_steps}
if [[ ! "${session_index}" =~ ^[0-9]+$ ]]; then
  echo "SESSION_INDEX must be a non-negative integer: ${session_index}" >&2
  exit 2
fi

mkdir -p "${output_root}"
output_root=$(cd "${output_root}" && pwd)
manifest_dir=${output_root}/.workflow_manifests
log_dir=${manifest_dir}/logs
mkdir -p "${log_dir}"
manifest=${manifest_dir}/single_session_$(date +%Y%m%dT%H%M%S).csv

"${python_executable}" "${script_dir}/prepare_session.py" \
  --source-manifest "${source_manifest}" \
  --session-index "${session_index}" \
  --output-root "${output_root}" \
  --manifest "${manifest}"

run_dir=$("${python_executable}" - "${manifest}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    print(next(csv.DictReader(handle))["run_dir"])
PY
)

export PYTHONNOUSERSITE=1
"${python_executable}" - <<'PY'
import importlib.metadata
import sys

for package in ("suite2p", "cellpose", "torch"):
    print(f"{package}=={importlib.metadata.version(package)}")
if importlib.metadata.version("suite2p") != "1.1.0":
    raise RuntimeError(
        f"This workflow requires suite2p==1.1.0 using {sys.executable}"
    )
PY

registration_submission=$(sbatch --parsable \
  --chdir="${manifest_dir}" \
  --output="${log_dir}/registration_%j.out" \
  --error="${log_dir}/registration_%j.err" \
  "${script_dir}/run_registration.sbatch" \
  "${manifest}" "${workflow_dir}" "${python_executable}")
registration_job=${registration_submission%%;*}

detection_submission=$(sbatch --parsable \
  --dependency="afterok:${registration_job}" \
  --chdir="${manifest_dir}" \
  --output="${log_dir}/roi_detection_%j.out" \
  --error="${log_dir}/roi_detection_%j.err" \
  "${script_dir}/run_roi_detection.sbatch" \
  "${run_dir}" "${script_dir}" "${python_executable}")
detection_job=${detection_submission%%;*}

echo "Manifest:         ${manifest}"
echo "Selected output:  ${run_dir}"
echo "Registration job: ${registration_job}"
echo "ROI job:          ${detection_job} (afterok:${registration_job})"
echo "Visual report:    ${run_dir}/roi_detection/cpsam/index.html"

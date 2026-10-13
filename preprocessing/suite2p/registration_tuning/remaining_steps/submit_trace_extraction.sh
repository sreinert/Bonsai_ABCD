#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 RUN_DIR [DETECTION_DIR] [OUTPUT_DIR]" >&2
  exit 2
fi

run_dir=$1
detection_dir=${2:-}
output_dir=${3:-}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_executable=${PYTHON_EXECUTABLE:-$(command -v python)}
exclude_nodes=${EXCLUDE_NODES-gpu-350-02,gpu-350-04}

if [[ ! -d "${run_dir}" ]]; then
  echo "Run directory not found: ${run_dir}" >&2
  exit 2
fi
run_dir=$(cd "${run_dir}" && pwd)
if [[ -n "${detection_dir}" ]]; then
  detection_dir=$(cd "${detection_dir}" && pwd)
fi

export PYTHONNOUSERSITE=1
"${python_executable}" - <<'PY'
import importlib.metadata
import sys

for package in ("suite2p", "torch"):
    print(f"{package}=={importlib.metadata.version(package)}")
if importlib.metadata.version("suite2p") != "1.1.0":
    raise RuntimeError(f"Trace extraction requires suite2p==1.1.0 using {sys.executable}")
PY

log_dir=${run_dir}/trace_extraction/logs
mkdir -p "${log_dir}"
sbatch_options=()
if [[ -n "${exclude_nodes}" ]]; then
  sbatch_options+=(--exclude="${exclude_nodes}")
fi

arguments=("${run_dir}" "${script_dir}" "${python_executable}")
if [[ -n "${detection_dir}" || -n "${output_dir}" ]]; then
  arguments+=("${detection_dir}")
fi
if [[ -n "${output_dir}" ]]; then
  arguments+=("${output_dir}")
fi

submission=$(sbatch --parsable \
  "${sbatch_options[@]}" \
  --chdir="${log_dir}" \
  --output="${log_dir}/trace_extraction_%j.out" \
  --error="${log_dir}/trace_extraction_%j.err" \
  "${script_dir}/run_trace_extraction.sbatch" \
  "${arguments[@]}")
job_id=${submission%%;*}

echo "Run directory:    ${run_dir}"
echo "Detection input:  ${detection_dir:-${run_dir}/roi_detection/cpsam}"
echo "Trace output:     ${output_dir:-${run_dir}/trace_extraction/cpsam}"
echo "Trace job:        ${job_id}"
echo "Stdout:           ${log_dir}/trace_extraction_${job_id}.out"
echo "Stderr:           ${log_dir}/trace_extraction_${job_id}.err"
echo "Excluded nodes:   ${exclude_nodes:-<none>}"

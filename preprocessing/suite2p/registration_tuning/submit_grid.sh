#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/tasks.csv" >&2
  exit 2
fi

manifest_arg=$1
if [[ ! -f "${manifest_arg}" ]]; then
  echo "Manifest not found: ${manifest_arg}" >&2
  exit 2
fi

manifest_dir=$(cd "$(dirname "${manifest_arg}")" && pwd)
manifest="${manifest_dir}/$(basename "${manifest_arg}")"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
task_count=$(($(wc -l < "${manifest}") - 1))
if (( task_count < 1 )); then
  echo "Manifest contains no tasks" >&2
  exit 2
fi

last_task=$((task_count - 1))
max_concurrent=${MAX_CONCURRENT:-6}
array_spec=${TASK_RANGE:-0-${last_task}%${max_concurrent}}
output_root=${manifest_dir}
exclude_nodes=${EXCLUDE_NODES:-}
if [[ -n "${PYTHON_EXECUTABLE:-}" ]]; then
  python_executable=${PYTHON_EXECUTABLE}
else
  python_executable=$(command -v python)
fi
if [[ ! -x "${python_executable}" ]]; then
  echo "Python executable is not runnable: ${python_executable}" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
"${python_executable}" - <<'PY'
import importlib.metadata
import sys

version = importlib.metadata.version("suite2p")
if version != "1.1.0":
    raise RuntimeError(
        f"Submission requires suite2p==1.1.0, found {version} using {sys.executable}"
    )
print("Validated submission Python:", sys.executable)
print("Validated Suite2p version:", version)
PY

mkdir -p "${output_root}/logs"

echo "Manifest: ${manifest}"
echo "Working directory: ${output_root}"
echo "Logs: ${output_root}/logs/s2p_reg_<job>_<task>.{out,err}"
echo "Python executable: ${python_executable}"
echo "Excluded nodes: ${exclude_nodes:-<none>}"

sbatch_options=(
  --array="${array_spec}"
  --chdir="${output_root}"
  --output="${output_root}/logs/s2p_reg_%A_%a.out"
  --error="${output_root}/logs/s2p_reg_%A_%a.err"
)
if [[ -n "${exclude_nodes}" ]]; then
  sbatch_options+=(--exclude="${exclude_nodes}")
fi

sbatch "${sbatch_options[@]}" \
  "${script_dir}/run_grid_array.sbatch" \
  "${manifest}" \
  "${repo_root}" \
  "${python_executable}"

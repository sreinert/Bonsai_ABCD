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
conda_env=${SUITE2P_ENV:-suite2p-reg-1.1.0}
array_spec=${TASK_RANGE:-0-${last_task}%${max_concurrent}}
output_root=${manifest_dir}
mkdir -p "${output_root}/logs"

cd "${output_root}"
sbatch --array="${array_spec}" \
  "${script_dir}/run_grid_array.sbatch" \
  "${manifest}" \
  "${repo_root}" \
  "${conda_env}"

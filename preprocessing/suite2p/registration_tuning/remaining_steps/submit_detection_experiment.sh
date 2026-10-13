#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 RUN_DIR EXPERIMENT_NAME [DETECTION OPTIONS...]" >&2
  echo "Example: $0 /path/to/chosen-registration-cpsam cpsam-mean --model cpsam --input-image meanImg" >&2
  exit 2
fi

run_dir=$1
experiment_name=$2
shift 2
if [[ ! "${experiment_name}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "EXPERIMENT_NAME may contain only letters, numbers, dots, underscores, and hyphens" >&2
  exit 2
fi
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_executable=${PYTHON_EXECUTABLE:-$(command -v python)}
exclude_nodes=${EXCLUDE_NODES-gpu-350-02,gpu-350-04}

if [[ ! -d "${run_dir}" ]]; then
  echo "Registered run directory not found: ${run_dir}" >&2
  exit 2
fi
run_dir=$(cd "${run_dir}" && pwd)
for required in db.npy reg_outputs.npy data.bin data_chan2.bin; do
  if [[ ! -f "${run_dir}/suite2p/plane0/${required}" ]]; then
    echo "Required registered input is missing: ${run_dir}/suite2p/plane0/${required}" >&2
    exit 2
  fi
done

export PYTHONNOUSERSITE=1
"${python_executable}" - <<'PY'
import importlib.metadata
import sys

for package in ("suite2p", "cellpose", "torch"):
    print(f"{package}=={importlib.metadata.version(package)}")
if importlib.metadata.version("suite2p") != "1.1.0":
    raise RuntimeError(f"This workflow requires suite2p==1.1.0 using {sys.executable}")
PY

log_dir=${run_dir}/roi_detection/logs
mkdir -p "${log_dir}"
sbatch_options=()
if [[ -n "${exclude_nodes}" ]]; then
  sbatch_options+=(--exclude="${exclude_nodes}")
fi

submission=$(sbatch --parsable \
  "${sbatch_options[@]}" \
  --chdir="${run_dir}" \
  --output="${log_dir}/${experiment_name}_%j.out" \
  --error="${log_dir}/${experiment_name}_%j.err" \
  "${script_dir}/run_detection_experiment.sbatch" \
  "${run_dir}" "${script_dir}" "${python_executable}" "${experiment_name}" "$@")
job_id=${submission%%;*}

echo "Detection job:   ${job_id}"
echo "Registered data: ${run_dir}/suite2p/plane0/data.bin"
echo "Experiment:      ${run_dir}/roi_detection/experiments/${experiment_name}"
echo "Excluded nodes:  ${exclude_nodes:-<none>}"
echo "No registration job was submitted."

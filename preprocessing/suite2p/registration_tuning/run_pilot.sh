#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1

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
task_id=${PILOT_TASK_ID:-0}
conda_env=${SUITE2P_ENV:-suite2p-reg-1.1.0}
device=${PILOT_DEVICE:-cuda}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
log_dir="${manifest_dir}/logs"
log_prefix="${log_dir}/pilot_task${task_id}_${timestamp}_$(hostname -s)"
stdout_log="${log_prefix}.out"
stderr_log="${log_prefix}.err"

mkdir -p "${log_dir}"
exec > >(tee -a "${stdout_log}") 2> >(tee -a "${stderr_log}" >&2)

report_exit() {
  exit_code=$?
  if (( exit_code == 0 )); then
    echo "Pilot finished successfully."
  else
    echo "Pilot failed with exit code ${exit_code}." >&2
  fi
  echo "stdout log: ${stdout_log}"
  echo "stderr log: ${stderr_log}"
}
trap report_exit EXIT

echo "Pilot started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Repository: ${repo_root}"
echo "Manifest: ${manifest}"
echo "Task: ${task_id}"
echo "Device: ${device}"
echo "Conda environment: ${conda_env}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${conda_env}" ]]; then
  if type module >/dev/null 2>&1; then
    module load mamba
  fi
  source activate "${conda_env}"
fi

echo "Active environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python: $(command -v python)"

python - "${device}" <<'PY'
import importlib.metadata
import sys

import torch

device = sys.argv[1]
torch_version = importlib.metadata.version("torch")
torch_module = getattr(torch, "__file__", None)
print("Python version:", sys.version.replace("\n", " "))
print("Suite2p version:", importlib.metadata.version("suite2p"))
print("Torch distribution version:", torch_version)
print("Torch module path:", torch_module)
if not hasattr(torch, "cuda"):
    raise RuntimeError(
        "The imported torch module does not expose torch.cuda. "
        f"Imported from {torch_module!r}; torch distribution is {torch_version}."
    )
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
if cuda_available:
    print("CUDA device:", torch.cuda.get_device_name(0))
if device == "cuda" and not cuda_available:
    raise RuntimeError(
        "CUDA is unavailable. Run this script inside an allocated GPU node, "
        "or set PILOT_DEVICE=cpu for a CPU test."
    )
PY

PYTHONPATH="${script_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
python - "${manifest}" "${task_id}" <<'PY'
import sys
from pathlib import Path

from run_candidate import read_task, validate_paths

manifest = Path(sys.argv[1])
task_id = int(sys.argv[2])
row = read_task(manifest, task_id)
input_path, run_dir, files = validate_paths(row)

print("Resolved mouse/session:", row["mouse_id"], row["session_id"])
print("Candidate:", row["candidate_name"])
print("Channels:", row["nchannels"])
print("Resolved input directory:", input_path)
print("Resolved output directory:", run_dir)
for path in files:
    print(f"Readable TIFF ({path.stat().st_size} bytes): {path}")
PY

python "${script_dir}/run_candidate.py" \
  --manifest "${manifest}" \
  --task-id "${task_id}" \
  --device "${device}"

echo "Pilot task ${task_id} completed and passed output validation."

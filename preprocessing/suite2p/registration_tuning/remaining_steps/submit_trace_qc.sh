#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 TRACE_DIR [QC OPTIONS...]" >&2
  echo "Example: $0 /path/to/trace_extraction/cpsam-meanimg --n-cells 8" >&2
  exit 2
fi

trace_dir=$1
shift
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_executable=${PYTHON_EXECUTABLE:-$(command -v python)}

if [[ ! -d "${trace_dir}" ]]; then
  echo "Trace directory not found: ${trace_dir}" >&2
  exit 2
fi
trace_dir=$(cd "${trace_dir}" && pwd)
for required in F.npy Fneu.npy F_chan2.npy Fneu_chan2.npy iscell.npy ops.npy stat.npy status.json; do
  if [[ ! -f "${trace_dir}/${required}" ]]; then
    echo "Required trace output is missing: ${trace_dir}/${required}" >&2
    exit 2
  fi
done

export PYTHONNOUSERSITE=1
"${python_executable}" - <<'PY'
import importlib.metadata
import sys

for package in ("numpy", "scipy", "matplotlib"):
    print(f"{package}=={importlib.metadata.version(package)}")
print(f"python={sys.executable}")
PY

"${python_executable}" - "${trace_dir}/status.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
state = json.loads(path.read_text(encoding="utf-8")).get("state")
if state != "complete":
    raise RuntimeError(f"Trace extraction is not complete in {path}: state={state!r}")
PY

log_dir=${trace_dir}/logs
mkdir -p "${log_dir}"
submission=$(sbatch --parsable \
  --chdir="${log_dir}" \
  --output="${log_dir}/trace_qc_%j.out" \
  --error="${log_dir}/trace_qc_%j.err" \
  "${script_dir}/run_trace_qc.sbatch" \
  "${trace_dir}" "${script_dir}" "${python_executable}" "$@")
job_id=${submission%%;*}

echo "Trace input: ${trace_dir}"
echo "QC job:      ${job_id}"
echo "Stdout:      ${log_dir}/trace_qc_${job_id}.out"
echo "Stderr:      ${log_dir}/trace_qc_${job_id}.err"
echo "Default QC:  ${trace_dir}/qc/index.html"

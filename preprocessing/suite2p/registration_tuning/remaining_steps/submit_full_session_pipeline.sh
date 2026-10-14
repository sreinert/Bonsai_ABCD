#!/bin/bash
set -euo pipefail

overwrite=false
positional=()
for argument in "$@"; do
  case "${argument}" in
    --overwrite)
      overwrite=true
      ;;
    --*)
      echo "Unknown option: ${argument}" >&2
      exit 2
      ;;
    *)
      positional+=("${argument}")
      ;;
  esac
done

if [[ ${#positional[@]} -ne 2 ]]; then
  echo "Usage: $0 [--overwrite] MOUSE_ID FULL_RAW_SESSION_ID" >&2
  exit 2
fi

mouse_id=${positional[0]}
session_id=${positional[1]}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_executable=${PYTHON_EXECUTABLE:-$(command -v python)}
project_root=${PROJECT_ROOT:-AtAp_20260119_SequenceCompression}
exclude_nodes=${EXCLUDE_NODES-gpu-350-02,gpu-350-04}
mail_user=${SLURM_MAIL_USER:-athina.apostolelli.24@ucl.ac.uk}

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required; run this launcher on a Slurm login or submission host." >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
"${python_executable}" - <<'PY'
import importlib.metadata
import sys

for package in ("suite2p", "cellpose", "torch", "numpy", "scipy", "matplotlib"):
    print(f"{package}=={importlib.metadata.version(package)}")
if importlib.metadata.version("suite2p") != "1.1.0":
    raise RuntimeError(f"Pipeline requires suite2p==1.1.0 using {sys.executable}")
PY

temporary_dir=$(mktemp -d)
paths_json=${temporary_dir}/paths.json
trap 'rm -rf "${temporary_dir}"' EXIT

prepare_options=()
if [[ "${overwrite}" == true ]]; then
  prepare_options+=(--overwrite)
fi

"${python_executable}" "${script_dir}/prepare_full_session_pipeline.py" \
  "${mouse_id}" "${session_id}" \
  --project-root "${project_root}" \
  --paths-json "${paths_json}" \
  "${prepare_options[@]}"

json_field() {
  "${python_executable}" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' \
    "${paths_json}" "$1"
}

already_complete=$(json_field already_complete)
run_dir=$(json_field run_dir)
suite2p_dir=$(json_field suite2p_dir)
plane_dir=$(json_field plane_dir)
qc_dir=$(json_field qc_dir)
workflow_dir=$(json_field workflow_dir)
report=$(json_field report)

if [[ "${already_complete}" == "True" ]]; then
  echo "Suite2p pipeline is already complete."
  echo "Output: ${plane_dir}"
  echo "Report: ${report}"
  exit 0
fi

manifest=$(json_field manifest)
log_dir=${workflow_dir}/logs
mkdir -p "${log_dir}" "${qc_dir}/registration" "${workflow_dir}/trace_extraction"

sbatch_options=(--mail-user="${mail_user}" --mail-type=BEGIN,END,FAIL)
if [[ -n "${exclude_nodes}" ]]; then
  sbatch_options+=(--exclude="${exclude_nodes}")
fi

submission=$(sbatch --parsable \
  "${sbatch_options[@]}" \
  --chdir="${workflow_dir}" \
  --output="${log_dir}/full_pipeline_%j.out" \
  --error="${log_dir}/full_pipeline_%j.err" \
  "${script_dir}/run_full_session_pipeline.sbatch" \
  "${manifest}" "${run_dir}" "${script_dir}" "${python_executable}")
job_id=${submission%%;*}

"${python_executable}" - \
  "${workflow_dir}/submission.json" "${workflow_dir}/status.json" \
  "${mouse_id}" "${session_id}" "${run_dir}" "${mail_user}" "${exclude_nodes}" \
  "${job_id}" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

(
    submission_path, status_path, mouse_id, session_id, run_dir, mail_user,
    exclude_nodes, job_id,
) = sys.argv[1:]
payload = {
    "state": "submitted",
    "submitted_unix": time.time(),
    "submission_hostname": os.uname().nodename,
    "mouse_id": mouse_id,
    "session_id": session_id,
    "run_dir": run_dir,
    "mail_user": mail_user,
    "mail_types": ["BEGIN", "END", "FAIL"],
    "excluded_gpu_nodes": [value for value in exclude_nodes.split(",") if value],
    "job": {
        "job_id": job_id,
        "partition": "gpu",
        "gpus": 1,
        "cpus": 4,
        "memory_gb": 64,
        "hours": 24,
        "stages": ["registration", "detection", "trace_extraction", "qc"],
    },
}
for filename in (submission_path, status_path):
    path = Path(filename)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
PY

echo "Mouse/session:    ${mouse_id} / ${session_id}"
echo "Overwrite:        ${overwrite}"
echo "Suite2p output:   ${plane_dir}"
echo "Pipeline job:     ${job_id} (registration -> detection -> traces -> QC)"
echo "Excluded GPUs:    ${exclude_nodes:-<none>}"
echo "Email:            ${mail_user} (BEGIN,END,FAIL)"
echo "Logs:             ${log_dir}"
echo "Final report:     ${report}"

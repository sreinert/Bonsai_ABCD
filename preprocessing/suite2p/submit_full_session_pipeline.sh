#!/bin/bash
set -euo pipefail

overwrite=false
resume=false
current_node=false
positional=()
for argument in "$@"; do
  case "${argument}" in
    --overwrite)
      overwrite=true
      ;;
    --resume)
      resume=true
      ;;
    --current-node)
      current_node=true
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

if [[ ${#positional[@]} -lt 2 ]]; then
  echo "Usage: $0 [--overwrite|--resume] [--current-node] MOUSE_ID FULL_RAW_SESSION_ID [FULL_RAW_SESSION_ID ...]" >&2
  exit 2
fi
if [[ "${overwrite}" == true && "${resume}" == true ]]; then
  echo "--overwrite and --resume are mutually exclusive" >&2
  exit 2
fi

mouse_id=${positional[0]}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ ${#positional[@]} -gt 2 ]]; then
  session_ids=("${positional[@]:1}")
  if [[ "${current_node}" == true ]]; then
    echo "--current-node: sessions will run sequentially on the current GPU."
  else
    echo "Submitting one independent Slurm job per session."
  fi

  seen_sessions=()
  for session_id in "${session_ids[@]}"; do
    if [[ ${#seen_sessions[@]} -gt 0 ]]; then
      for seen_session in "${seen_sessions[@]}"; do
        if [[ "${session_id}" == "${seen_session}" ]]; then
          echo "Duplicate session in batch: ${session_id}" >&2
          exit 2
        fi
      done
    fi
    seen_sessions+=("${session_id}")
  done

  failed_sessions=()
  completed_count=0
  for session_id in "${session_ids[@]}"; do
    echo
    echo "=== ${mouse_id} / ${session_id} ==="
    submission_command=(bash "${script_dir}/submit_full_session_pipeline.sh")
    if [[ "${overwrite}" == true ]]; then
      submission_command+=(--overwrite)
    fi
    if [[ "${resume}" == true ]]; then
      submission_command+=(--resume)
    fi
    if [[ "${current_node}" == true ]]; then
      submission_command+=(--current-node)
    fi
    submission_command+=("${mouse_id}" "${session_id}")
    if "${submission_command[@]}"; then
      completed_count=$((completed_count + 1))
    else
      failed_sessions+=("${session_id}")
    fi
  done

  echo
  echo "Batch summary: ${completed_count}/${#session_ids[@]} session submissions succeeded."
  if [[ ${#failed_sessions[@]} -gt 0 ]]; then
    echo "Failed sessions:" >&2
    printf '  %s\n' "${failed_sessions[@]}" >&2
    exit 1
  fi
  exit 0
fi

session_id=${positional[1]}
pipeline_dir="${script_dir}/full_session_pipeline"
python_executable=${PYTHON_EXECUTABLE:-$(command -v python)}
project_root=${PROJECT_ROOT:-AtAp_20260119_SequenceCompression}
exclude_nodes=${EXCLUDE_NODES-gpu-350-02,gpu-350-04}
mail_user=${SLURM_MAIL_USER:-athina.apostolelli.24@ucl.ac.uk}

if [[ "${current_node}" == false ]] && ! command -v sbatch >/dev/null 2>&1; then
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

if [[ "${current_node}" == true ]]; then
  current_hostname=$(hostname -s)
  IFS=',' read -r -a excluded_hostnames <<< "${exclude_nodes}"
  for excluded_hostname in "${excluded_hostnames[@]}"; do
    if [[ -n "${excluded_hostname}" && "${current_hostname}" == "${excluded_hostname}" ]]; then
      echo "Refusing current-node analysis on excluded host ${current_hostname}." >&2
      exit 2
    fi
  done
  "${python_executable}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("--current-node requires a CUDA GPU visible to PyTorch")
print(f"Current-node GPU: {torch.cuda.get_device_name(0)}")
PY
fi

temporary_dir=$(mktemp -d)
paths_json=${temporary_dir}/paths.json
trap 'rm -rf "${temporary_dir}"' EXIT

prepare_options=()
if [[ "${overwrite}" == true ]]; then
  prepare_options+=(--overwrite)
fi
if [[ "${resume}" == true ]]; then
  prepare_options+=(--resume)
fi

"${python_executable}" "${pipeline_dir}/prepare_full_session_pipeline.py" \
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
start_stage=$(json_field start_stage)

if [[ "${already_complete}" == "True" ]]; then
  echo "Suite2p pipeline is already complete."
  echo "Output: ${plane_dir}"
  echo "Report: ${report}"
  exit 0
fi

manifest=$(json_field manifest)
log_dir=${workflow_dir}/logs
mkdir -p "${log_dir}" "${qc_dir}/registration" "${workflow_dir}/trace_extraction"

execution_mode=slurm
job_id=""
stdout_log=""
stderr_log=""
if [[ "${current_node}" == true ]]; then
  execution_mode=current_node
  run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
  stdout_log=${log_dir}/full_pipeline_current_node_${run_stamp}.log
  stderr_log=${stdout_log}
else
  sbatch_options=(--mail-user="${mail_user}" --mail-type=BEGIN,END,FAIL)
  if [[ -n "${exclude_nodes}" ]]; then
    sbatch_options+=(--exclude="${exclude_nodes}")
  fi
  submission=$(sbatch --parsable \
    "${sbatch_options[@]}" \
    --chdir="${workflow_dir}" \
    --output="${log_dir}/full_pipeline_%j.out" \
    --error="${log_dir}/full_pipeline_%j.err" \
    "${pipeline_dir}/run_full_session_pipeline.sbatch" \
    "${manifest}" "${run_dir}" "${pipeline_dir}" "${python_executable}" "${start_stage}")
  job_id=${submission%%;*}
  stdout_log=${log_dir}/full_pipeline_${job_id}.out
  stderr_log=${log_dir}/full_pipeline_${job_id}.err
fi

"${python_executable}" - \
  "${workflow_dir}/submission.json" "${workflow_dir}/status.json" \
  "${mouse_id}" "${session_id}" "${run_dir}" "${mail_user}" "${exclude_nodes}" \
  "${job_id}" "${start_stage}" "${execution_mode}" "${stdout_log}" "${stderr_log}" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

(
    submission_path, status_path, mouse_id, session_id, run_dir, mail_user,
    exclude_nodes, job_id, start_stage, execution_mode, stdout_log, stderr_log,
) = sys.argv[1:]
payload = {
    "state": "submitted" if execution_mode == "slurm" else "starting_on_current_node",
    "submitted_unix": time.time(),
    "submission_hostname": os.uname().nodename,
    "mouse_id": mouse_id,
    "session_id": session_id,
    "run_dir": run_dir,
    "execution_mode": execution_mode,
    "mail_user": mail_user if execution_mode == "slurm" else None,
    "mail_types": ["BEGIN", "END", "FAIL"] if execution_mode == "slurm" else [],
    "excluded_gpu_nodes": [value for value in exclude_nodes.split(",") if value],
    "stdout_log": stdout_log,
    "stderr_log": stderr_log,
    "job": {
        "job_id": job_id or None,
        "partition": "gpu" if execution_mode == "slurm" else None,
        "gpus": 1 if execution_mode == "slurm" else None,
        "cpus": 4 if execution_mode == "slurm" else None,
        "memory_gb": 64 if execution_mode == "slurm" else None,
        "hours": 24 if execution_mode == "slurm" else None,
        "start_stage": start_stage,
        "stages": (
            ["detection", "trace_extraction", "qc"]
            if start_stage == "detection"
            else ["registration", "detection", "trace_extraction", "qc"]
        ),
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
echo "Resume:           ${resume} (start stage: ${start_stage})"
echo "Execution mode:   ${execution_mode}"
echo "Suite2p output:   ${plane_dir}"
if [[ "${execution_mode}" == current_node ]]; then
  echo "Current host:     $(hostname -s)"
elif [[ "${start_stage}" == detection ]]; then
  echo "Pipeline job:     ${job_id} (reuse registration -> detection -> traces -> QC)"
else
  echo "Pipeline job:     ${job_id} (registration -> detection -> traces -> QC)"
fi
echo "Excluded GPUs:    ${exclude_nodes:-<none>}"
if [[ "${execution_mode}" == slurm ]]; then
  echo "Email:            ${mail_user} (BEGIN,END,FAIL)"
else
  echo "Email:            disabled (no Slurm job)"
fi
echo "Stdout log:       ${stdout_log}"
echo "Stderr log:       ${stderr_log}"
echo "Final report:     ${report}"

if [[ "${execution_mode}" == current_node ]]; then
  echo "Running pipeline in the foreground on the current node..."
  bash "${pipeline_dir}/run_full_session_pipeline.sbatch" \
    "${manifest}" "${run_dir}" "${pipeline_dir}" "${python_executable}" "${start_stage}" \
    2>&1 | tee "${stdout_log}"
  echo "Pipeline complete. Final report: ${report}"
fi

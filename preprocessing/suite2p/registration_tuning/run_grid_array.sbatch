#!/bin/bash
#SBATCH --job-name=s2p-reg-grid
#SBATCH --output=logs/s2p_reg_%A_%a.out
#SBATCH --error=logs/s2p_reg_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: run_grid_array.sbatch MANIFEST REPO_ROOT CONDA_ENV" >&2
  exit 2
fi

manifest=$1
repo_root=$2
conda_env=$3

module load mamba
source activate "${conda_env}"
export PYTHONNOUSERSITE=1

python - <<'PY'
import importlib.metadata
import sys
import suite2p

print("Python executable:", sys.executable)
print("Suite2p version:", importlib.metadata.version("suite2p"))
print("Suite2p module:", suite2p.__file__)
PY

python "${repo_root}/preprocessing/suite2p/registration_tuning/run_candidate.py" \
  --manifest "${manifest}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --device cuda

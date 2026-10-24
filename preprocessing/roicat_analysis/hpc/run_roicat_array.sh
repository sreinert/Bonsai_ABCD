#!/bin/bash
#SBATCH --job-name=roicat
#SBATCH --output=logs/roicat_%A_%a.out
#SBATCH --error=logs/roicat_%A_%a.out
#
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres gpu:1
# The P5000 nodes are excluded: torch 2.14.1+cu130 has no kernels for Pascal (sm_61) and
# its cuDNN refuses SM < 7.5. Every other GPU type in the partition is sm_75 or newer.
#SBATCH --exclude=gpu-380-[10,12,14]
#SBATCH -t 3:00:00
#SBATCH --mem=64G
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user s.reinert@ucl.ac.uk

# ROICaT tracking across sessions, one animal per array task (roicat_tracking_holography.py).
#
# Submit from this folder (hpc/), one task per animal line:
#   sbatch --array=1-$(grep -cv '^\s*\(#\|$\)' mice_holography.txt) run_roicat_array.sh mice_holography.txt
# Output: <cohort>/analysis/<mouse>/ROICaT/. See README.md for the one-time uv setup
# (venv in /ceph/python_envs/$USER/roicat, per SWC HPC guidelines).

set -euo pipefail

MICE_FILE=${1:-mice_holography.txt}
ROOT=${ROOT:-/ceph/mrsic_flogel/public/projects/SaReSuKuMaRa_20260911_holography/cohort1}

# Submitted without --array: run the first animal
TASK=${SLURM_ARRAY_TASK_ID:-1}

# N-th non-empty, non-comment line of the mice file -> "mouse tag1 tag2 ..."
read -r mouse tags < <(grep -v '^\s*\(#\|$\)' "$SLURM_SUBMIT_DIR/$MICE_FILE" | sed -n "${TASK}p")
if [ -z "${mouse:-}" ] || [ -z "${tags:-}" ]; then
    echo "No animal on line $TASK of $MICE_FILE" >&2
    exit 1
fi

# venv built once with `uv sync --frozen` (README.md); jobs only activate it
ENV=${UV_PROJECT_ENVIRONMENT:-/ceph/python_envs/$USER/roicat}
source "$ENV/bin/activate"
export MPLBACKEND=Agg
export ROICAT_MODEL_DIR=${ROICAT_MODEL_DIR:-/ceph/python_envs/$USER/roicat_models}  # not the shared /tmp
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE                           # h5py on ceph
umask 002                                                    # outputs group-writable

echo "task $TASK on $(hostname): $mouse $tags"
echo "root: $ROOT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
cat "$SLURM_SUBMIT_DIR/../../../../CODE_VERSION_roicat.txt" 2>/dev/null || true

# $tags unquoted on purpose: one argument per session tag
python "$SLURM_SUBMIT_DIR/../roicat_tracking_holography.py" --animal "$mouse" --sessions $tags --root "$ROOT"

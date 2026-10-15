#!/bin/bash
#SBATCH --job-name=nwb_build
#SBATCH --output=logs/nwb_%A_%a.out
#SBATCH --error=logs/nwb_%A_%a.out
#
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH -t 3:00:00
#SBATCH --mem=32G
#SBATCH --mail-type ALL
#SBATCH --mail-user s.reinert@ucl.ac.uk

# Build behav/funcimg/merged NWB + behaviour_data.npz for one session per array task
# (python -m nwb_barcoding.build_session_nwb, single-tree layout: rawdata/ + preprocessed/).
#
# Submit from this folder (hpc/), one task per session line, at most 4 running at once:
#   sbatch --array=1-$(grep -cv '^\s*\(#\|$\)' sessions_holography.txt)%4 build_nwb_array.sh sessions_holography.txt
# Steps whose output already exists are skipped, so resubmitting only redoes what is missing.
# See README.md for the one-time uv setup (venv in /ceph/python_envs/$USER, per SWC HPC guidelines).

set -euo pipefail

SESSIONS_FILE=${1:-sessions_holography.txt}
BASE=${BASE:-/ceph/mrsic_flogel/public/projects/SaReSuKuMaRa_20260911_holography/cohort1}
RAW=$BASE/rawdata
PREPROCESSED=$BASE/preprocessed
OUT=$BASE/BehaviourAligned_robust

# N-th non-empty, non-comment line of the sessions file -> "mouse tag"
read -r mouse tag < <(grep -v '^\s*\(#\|$\)' "$SLURM_SUBMIT_DIR/$SESSIONS_FILE" | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "${mouse:-}" ] || [ -z "${tag:-}" ]; then
    echo "No session on line $SLURM_ARRAY_TASK_ID of $SESSIONS_FILE" >&2
    exit 1
fi

# venv built once on the login node with `uv sync --frozen` (README.md); jobs only use it,
# so parallel array tasks never install into it at the same time
ENV=${UV_PROJECT_ENVIRONMENT:-/ceph/python_envs/$USER/nwb_barcoding}
source "$ENV/bin/activate"
export HDF5_USE_FILE_LOCKING=FALSE                           # h5py on ceph
export PYTHONUNBUFFERED=1
umask 002                                                    # outputs group-writable

echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $mouse $tag"
echo "out: $OUT"
cat "$SLURM_SUBMIT_DIR/../../../CODE_VERSION.txt" 2>/dev/null || true

cd "$SLURM_SUBMIT_DIR/../.."   # Bonsai_ABCD, so `-m nwb_barcoding...` resolves
python -m nwb_barcoding.build_session_nwb "$mouse" "$tag" "$RAW" "$PREPROCESSED" "$OUT"

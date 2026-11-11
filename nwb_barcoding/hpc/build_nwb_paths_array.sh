#!/bin/bash
#SBATCH --job-name=nwb_build_paths
#SBATCH --output=logs/nwb_paths_%A_%a.out
#SBATCH --error=logs/nwb_paths_%A_%a.out
#
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH -t 3:00:00
#SBATCH --mem=32G
#SBATCH --mail-type ALL
#SBATCH --mail-user s.reinert@ucl.ac.uk

# Build behav/funcimg/merged NWB + behaviour_data.npz for one session per array task, from explicit
# paths (python -m nwb_barcoding.build_session_nwb_paths), e.g. for the per-session splits of a
# suite2p run over concatenated sessions (preprocessing/suite2p/split_concat_suite2p.py).
#
# Each sessions-file line is "<raw session> <suite2p folder> <output root>", relative to $BASE
# (or absolute). Submit from this folder (hpc/), at most 4 running at once:
#   sbatch --array=1-$(grep -cv '^\s*\(#\|$\)' sessions_concat007009.txt)%4 build_nwb_paths_array.sh sessions_concat007009.txt
# Steps whose output already exists are skipped, so resubmitting only redoes what is missing.
# See README.md for the one-time uv setup (venv in /ceph/python_envs/$USER, per SWC HPC guidelines).

set -euo pipefail

SESSIONS_FILE=${1:?usage: sbatch --array=... build_nwb_paths_array.sh <sessions file>}
BASE=${BASE:-/ceph/mrsic_flogel/public/projects/SaReSuKuMaRa_20260911_holography/cohort1}

# N-th non-empty, non-comment line of the sessions file -> "raw suite2p out"
read -r raw suite2p out < <(grep -v '^\s*\(#\|$\)' "$SLURM_SUBMIT_DIR/$SESSIONS_FILE" | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "${raw:-}" ] || [ -z "${suite2p:-}" ] || [ -z "${out:-}" ]; then
    echo "No complete '<raw> <suite2p> <out>' entry on line $SLURM_ARRAY_TASK_ID of $SESSIONS_FILE" >&2
    exit 1
fi
[[ $raw == /* ]] || raw=$BASE/$raw
[[ $suite2p == /* ]] || suite2p=$BASE/$suite2p
[[ $out == /* ]] || out=$BASE/$out

# venv built once with `uv sync --frozen` (README.md); jobs only use it,
# so parallel array tasks never install into it at the same time
ENV=${UV_PROJECT_ENVIRONMENT:-/ceph/python_envs/$USER/nwb_barcoding}
source "$ENV/bin/activate"
export HDF5_USE_FILE_LOCKING=FALSE                           # h5py on ceph
export PYTHONUNBUFFERED=1
umask 002                                                    # outputs group-writable

echo "task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "raw:     $raw"
echo "suite2p: $suite2p"
echo "out:     $out"
cat "$SLURM_SUBMIT_DIR/../../../CODE_VERSION.txt" 2>/dev/null || true

cd "$SLURM_SUBMIT_DIR/../.."   # Bonsai_ABCD, so `-m nwb_barcoding...` resolves
python -m nwb_barcoding.build_session_nwb_paths "$raw" "$suite2p" "$out"

#!/bin/bash
#SBATCH --job-name=holo_projected
#SBATCH --output=logs/holo_projected_%j.out
#SBATCH --error=logs/holo_projected_%j.out
#
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 4:00:00
#SBATCH --mem=32G
#SBATCH --mail-type ALL
#SBATCH --mail-user s.reinert@ucl.ac.uk

# Extract traces of the projected source masks (match_rois.py -> stat_projected.npy) from the
# registered holo movie, one session per job (musicboxes/holography/extract_projected.py).
# data.bin is read straight from /ceph, which is why this runs here and not over SMB on the Mac.
#
# Submit from this folder (musicboxes/holography/), after match_rois.py has run:
#   mkdir -p logs && sbatch extract_projected_hpc.sh 001 holo-001
#
# Uses the bonsai_abcd uv venv (pyproject.toml at the repo root, suite2p 0.14.6), built once on
# the login node with `module load uv; UV_PROJECT_ENVIRONMENT=/ceph/python_envs/$USER/bonsai_abcd uv sync --frozen`.

set -euo pipefail

mouse=${1:?usage: sbatch extract_projected_hpc.sh <mouse> <holo_tag>}
tag=${2:?usage: sbatch extract_projected_hpc.sh <mouse> <holo_tag>}

ENV=${UV_PROJECT_ENVIRONMENT:-/ceph/python_envs/$USER/bonsai_abcd}
source "$ENV/bin/activate"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
umask 002

echo "$(hostname): extract_projected $mouse $tag"
cd "$SLURM_SUBMIT_DIR/../.."   # Bonsai_ABCD, so `-m musicboxes...` resolves
python -m musicboxes.holography.extract_projected "$mouse" "$tag"

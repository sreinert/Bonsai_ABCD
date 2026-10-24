#!/bin/bash
# Copy preprocessing/roicat_analysis/ onto the share, for the HPC jobs (run_roicat_array.sh).
# Run on the laptop, with the share mounted:   bash preprocessing/roicat_analysis/hpc/sync_to_cluster.sh
#
# Lands next to the nwb_barcoding code (nwb_barcoding/hpc/sync_to_cluster.sh):
#   <DEST>/Bonsai_ABCD/preprocessing/roicat_analysis/...   (incl. hpc/pyproject.toml, uv.lock)
# Copies the working tree as it is (uncommitted edits included), and records which commit +
# local changes were copied in <DEST>/CODE_VERSION_roicat.txt.
# Never deletes anything at the destination.

set -euo pipefail

DEST=${1:-/Volumes/mrsic_flogel/public/projects/SaRe_20240219_hfs/hpc_files/code}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROICAT="$(cd "$HERE/.." && pwd)"
BONSAI="$(cd "$ROICAT/../.." && pwd)"

if [ ! -d "$(dirname "$DEST")" ]; then
    echo "$(dirname "$DEST") not found -- is the share mounted?" >&2
    exit 1
fi
mkdir -p "$DEST/Bonsai_ABCD/preprocessing" "$DEST/Bonsai_ABCD/preprocessing/roicat_analysis/hpc/logs"

# -rt rather than -a: the SMB mount can't take owners/permissions
rsync -rt --itemize-changes --exclude __pycache__ --exclude '.venv*' --exclude '*.ipynb' \
      --exclude .DS_Store --exclude 'hpc/logs/' \
      "$ROICAT" "$DEST/Bonsai_ABCD/preprocessing/"

{
    echo "synced $(date '+%Y-%m-%d %H:%M:%S') from $(hostname)"
    echo
    echo "== $(basename "$BONSAI"): $(git -C "$BONSAI" rev-parse --abbrev-ref HEAD) @ $(git -C "$BONSAI" rev-parse HEAD)"
    git -C "$BONSAI" status --short -- preprocessing/roicat_analysis 2>/dev/null || true
} > "$DEST/CODE_VERSION_roicat.txt"

echo
echo "Synced to $DEST"
cat "$DEST/CODE_VERSION_roicat.txt"

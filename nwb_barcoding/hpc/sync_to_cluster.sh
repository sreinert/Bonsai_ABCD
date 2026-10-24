#!/bin/bash
# Copy the code nwb_barcoding needs onto the share, for the HPC jobs (build_nwb_array.sh).
# Run on the laptop, with the share mounted:   bash nwb_barcoding/hpc/sync_to_cluster.sh
#
# Once <DEST>/Bonsai_ABCD is a git clone (README.md), Bonsai_ABCD is updated with `git pull`
# on the cluster instead, and this script only copies Aimbot_barcoding: rsyncing into a git
# checkout would leave it full of modified/untracked files.
#
# Keeps the sibling layout _aimbot_path.py expects:
#   <DEST>/Bonsai_ABCD/nwb_barcoding/...          (incl. pyproject.toml, uv.lock, hpc/)
#   <DEST>/Aimbot_barcoding/preprocess_alignment/...
# Copies working trees as they are (uncommitted edits included), and records which
# commits + local changes were copied in <DEST>/CODE_VERSION.txt.
# Never deletes anything at the destination.

set -euo pipefail

DEST=${1:-/Volumes/mrsic_flogel/public/projects/SaRe_20240219_hfs/hpc_files/code}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BONSAI="$(cd "$HERE/../.." && pwd)"
AIMBOT="$(cd "$BONSAI/../Aimbot_barcoding" && pwd)"

if [ ! -d "$(dirname "$DEST")" ]; then
    echo "$(dirname "$DEST") not found -- is the share mounted?" >&2
    exit 1
fi
if [ -d "$DEST/Bonsai_ABCD/.git" ]; then
    BONSAI_IS_CLONE=1
    echo "$DEST/Bonsai_ABCD is a git clone: skipping it (git pull on the cluster), copying Aimbot_barcoding only"
else
    BONSAI_IS_CLONE=0
fi
mkdir -p "$DEST/Aimbot_barcoding"
[ "$BONSAI_IS_CLONE" = 1 ] || mkdir -p "$DEST/Bonsai_ABCD/nwb_barcoding/hpc/logs"

# -rt rather than -a: the SMB mount can't take owners/permissions
RSYNC=(rsync -rt --itemize-changes --exclude __pycache__ --exclude .venv --exclude '*.ipynb'
       --exclude .git --exclude .DS_Store --exclude 'hpc/logs/')

if [ "$BONSAI_IS_CLONE" = 1 ]; then
    REPOS=("$AIMBOT")
else
    "${RSYNC[@]}" "$BONSAI/nwb_barcoding" "$DEST/Bonsai_ABCD/"
    REPOS=("$BONSAI" "$AIMBOT")
fi
"${RSYNC[@]}" "$AIMBOT/preprocess_alignment" "$DEST/Aimbot_barcoding/"

{
    echo "synced $(date '+%Y-%m-%d %H:%M:%S') from $(hostname)"
    [ "$BONSAI_IS_CLONE" = 1 ] && echo "(Bonsai_ABCD is a git clone on the share: see git log there)"
    for repo in "${REPOS[@]}"; do
        echo
        echo "== $(basename "$repo"): $(git -C "$repo" rev-parse --abbrev-ref HEAD) @ $(git -C "$repo" rev-parse HEAD)"
        git -C "$repo" status --short -- nwb_barcoding preprocess_alignment 2>/dev/null || true
    done
} > "$DEST/CODE_VERSION.txt"

echo
echo "Synced to $DEST"
cat "$DEST/CODE_VERSION.txt"

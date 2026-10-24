"""
Save Suite2p's mean images (from ops.npy) as TIFFs for inspection (e.g. in Fiji).

Loads `meanImg` (and `meanImg_chan2` if present) from the session's
suite2p/plane0/ops.npy instead of recomputing them from the binaries.

Usage: python check_mean_img.py <basepath> <mouse> <session> [--out_dir DIR]
  basepath: project root containing the mouse folders
  mouse:    mouse folder name, with or without the 'sub-' prefix (e.g. '004', 'TAA0000066')
  session:  session folder name, or any unique part of it (e.g. 't2', 'full030', a date)
  out_dir:  where to save the TIFFs (default: the suite2p plane0 folder). Use this when
            suite2p sits inside rawdata/, which should not be written to.
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile

parser = argparse.ArgumentParser(description="Save Suite2p mean images as TIFFs.")
parser.add_argument("basepath", type=str, help="Project root containing the mouse folders")
parser.add_argument("mouse", type=str, help="Mouse folder name, e.g. '004', 'sub-004' or 'TAA0000066'")
parser.add_argument("session", type=str, help="Session folder name or a unique part of it")
parser.add_argument("--out_dir", type=str, default=None, help="Output folder (default: suite2p plane0 folder)")
args = parser.parse_args()

# Mouse folder: as given (e.g. cohort2 'TAA0000066'), otherwise with the 'sub-' prefix
mouse_path = Path(args.basepath) / args.mouse
if not mouse_path.is_dir():
    mouse_path = Path(args.basepath) / f"sub-{args.mouse}"

# Session folder: exact name match first, otherwise a unique substring match
session_path = mouse_path / args.session
if not session_path.is_dir():
    matches = [f for f in mouse_path.iterdir() if f.is_dir() and args.session in f.name]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one session folder matching '{args.session}' in {mouse_path}, found {[m.name for m in matches]}")
    session_path = matches[0]
print(f"Session: {session_path}")

# suite2p output can sit at funcimg/suite2p or funcimg/Session/suite2p depending on cohort
ops_files = sorted(session_path.glob("**/suite2p/plane0/ops.npy"))
if not ops_files:
    raise FileNotFoundError(f"No suite2p/plane0/ops.npy found under {session_path}")
ops_path = ops_files[0]
plane_dir = ops_path.parent
print(f"Loading {ops_path}")
out_dir = Path(args.out_dir) if args.out_dir else plane_dir
out_dir.mkdir(parents=True, exist_ok=True)

ops = np.load(ops_path, allow_pickle=True).item()

for key in ["meanImg", "meanImg_chan2"]:
    if key not in ops:
        print(f"{key}: not found in ops")
        continue
    mean_img = np.asarray(ops[key], dtype=np.float32)
    out = out_dir / f"{key}.tif"
    tifffile.imwrite(out, mean_img)
    print(f"  saved {out}  (shape {mean_img.shape}, min {mean_img.min():.1f}, max {mean_img.max():.1f})")

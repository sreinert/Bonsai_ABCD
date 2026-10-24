"""Save suite2p reference / mean images (as tiff + png) for cellpose input and registration QC.

Usage:
    python save_suite2p_meanImg.py <base_path> <mouse> <session_tag>
    e.g. python save_suite2p_meanImg.py SaReSuKuMaRa_20260911_holography/cohort1/preprocessed 004 t0

base_path may be absolute or relative to the lab projects root (ROOT below). mouse can be given
as '004' or 'sub-004'. The session folder is found by matching session_tag as a '-'/'_'-delimited
token in the folder name (so 't1' does not match 't10'). Run without arguments to use the
hardcoded basepath/sessions below. suite2p_path and n_chan are still set in the script.
"""
import argparse
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
import os
from pathlib import Path
from align_images import adjust_intensity
import sys
import tifffile

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root, for ceph_paths
from ceph_paths import PROJECTS_ROOT as ROOT  # HPC, Mac or Windows mount
basepath = Path(ROOT) / "SaReSuKuMaRa_20260911_holography/cohort1/preprocessed" # "SaReMaRa_20260519_NeuralAnalysis/Cohort3" or ./Cohort2 ; "SaReMaRa_20260629_GoalProgress" ; SuKuSaRe_20250923_HFScohort3" ; "AtApSuKuSaRe_20250129_HFScohort2" ; 

sessions = ['sub-004/ses-2LM001_date-20261006T170659-t0']


def find_sessions(basepath, mouse, tag):
    """Return session paths (relative to basepath) under sub-<mouse> whose folder name contains tag as a token."""
    mouse_dir = f"sub-{mouse}" if not str(mouse).startswith("sub-") else str(mouse)
    mouse_path = basepath / mouse_dir
    if not mouse_path.is_dir():
        sys.exit(f"Mouse folder not found: {mouse_path}")
    pattern = re.compile(rf"(^|[-_]){re.escape(tag)}($|[-_])")
    matches = sorted(f"{mouse_dir}/{f.name}" for f in mouse_path.iterdir() if f.is_dir() and pattern.search(f.name))
    if not matches:
        sys.exit(f"No session matching '{tag}' in {mouse_path}")
    if len(matches) > 1:
        print(f"Multiple sessions match '{tag}', processing all: {matches}")
    return matches


parser = argparse.ArgumentParser(description="Save suite2p mean images for one session.")
parser.add_argument("base_path", nargs="?", help="preprocessed root (absolute, or relative to ROOT)")
parser.add_argument("mouse", nargs="?", help="mouse ID, e.g. 004 or sub-004")
parser.add_argument("session_tag", nargs="?", help="session tag, e.g. t0")
args = parser.parse_args()

if any([args.base_path, args.mouse, args.session_tag]):
    if not all([args.base_path, args.mouse, args.session_tag]):
        parser.error("give all of base_path, mouse and session_tag (or none to use the hardcoded values)")
    basepath = Path(args.base_path)
    if not basepath.is_absolute():
        basepath = Path(ROOT) / basepath
    sessions = find_sessions(basepath, args.mouse, args.session_tag)

suite2p_path = 'funcimg/suite2p/plane0' # 'funcimg/suite2p/plane0' for a typical session; some old data e.g. Cohort2 recordings are stored as 'funcimg/Session/suite2p/plane0'
n_chan = 2 # 1 / 2

for session in sessions:
    reg_path = basepath / session / suite2p_path / 'reg_outputs.npy'
    settings_path = basepath / session / suite2p_path / 'settings.npy'
    cellpose_out_path = basepath / session / suite2p_path / 'meanImg_seg.npy'

    # check suite2p registration
    if not os.path.exists(reg_path):
        print(f"Session {session} suite2p registration has not been run.")
        continue

    # check cellpose detection
    if not os.path.exists(cellpose_out_path):
        print(f"Session {session} cellpose detection has not been run.")

    # check if mean image already exists
    if os.path.exists(basepath / session / suite2p_path / 'meanImg.tiff'):
        continue

    reg_outputs = np.load(reg_path, allow_pickle=True).item()
    settings = np.load(settings_path, allow_pickle=True).item()

    # save reference image (useful as a reg quality check)
    refImg = reg_outputs['refImg']
    plt.imshow(refImg, cmap='gray')
    plt.title('Reference Image')
    plt.savefig(basepath / session / suite2p_path / 'refImg.png', dpi=300, bbox_inches='tight')
    plt.close()

    # print registration settings 
    print(f"Session: {session}")
    print(f"  smooth_sigma_time: {settings['registration']['smooth_sigma_time']}")
    print(f"  smooth_sigma: {settings['registration']['smooth_sigma']}")
    print(f"  nimg_init: {settings['registration']['nimg_init']}")
    print(f"  nonrigid: {settings['registration']['nonrigid']}")
    print(f"  block_size: {settings['registration']['block_size']}")
    print(f"  spatial_taper: {settings['registration']['spatial_taper']}")
    print(f"  maxregshift: {settings['registration']['maxregshift']}")

    # save mean images
    img1 = reg_outputs['meanImg']
    img1 = img_as_ubyte(adjust_intensity(img_as_ubyte(rescale_intensity(img1, in_range='image', out_range=(0, 1)))))

    if n_chan > 1:
        img2 = reg_outputs['meanImg_chan2']
        img2 = img_as_ubyte(adjust_intensity(img_as_ubyte(rescale_intensity(img2, in_range='image', out_range=(0, 1)))))

        # Save individual single-channel tiffs and pngs for cellpose input
        tifffile.imwrite(basepath / session / suite2p_path / 'meanImg_chan1.tiff', img1)
        tifffile.imwrite(basepath / session / suite2p_path / 'meanImg_chan2.tiff', img2)
        plt.imsave(str(basepath / session / suite2p_path / 'meanImg_chan1.png'), img1, cmap='gray')
        plt.imsave(str(basepath / session / suite2p_path / 'meanImg_chan2.png'), img2, cmap='gray')

        # Save side-by-side comparison for visual inspection
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        ax[0].imshow(img1, cmap='gray')
        ax[0].set_axis_off()
        ax[0].set_title('Chan1')
        ax[1].imshow(img2, cmap='gray')
        ax[1].set_axis_off()
        ax[1].set_title('Chan2')
        plt.savefig(basepath / session / suite2p_path / 'meanImg_chan1_chan2.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    else:
        plt.imsave(str(basepath / session / suite2p_path / 'meanImg.png'), img1, cmap='gray')
        tifffile.imwrite(basepath / session / suite2p_path / 'meanImg.tiff', img1)

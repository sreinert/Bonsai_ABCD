"""Save suite2p reference / mean images (as tiff + png) for cellpose input and registration QC.

Copy of save_suite2p_meanImg.py that skips the base_path/mouse/session_tag lookup: it takes the
suite2p folder directly, so it works for any folder layout (cohort2 'funcimg/Session/suite2p',
holography, test runs outside the session tree, ...).

Usage:
    python save_suite2p_meanImg_from_path.py <suite2p_path> [<suite2p_path> ...] [--n_chan 1|2] [--force]
    e.g. python save_suite2p_meanImg_from_path.py /ceph/mrsic_flogel/public/projects/.../funcimg/suite2p/plane0

suite2p_path can be the plane folder (containing reg_outputs.npy) or the suite2p folder above it,
in which case every plane* subfolder is processed. Paths go through ceph_path, so a share path
written for Mac, HPC or Windows works on any machine. n_chan is read from reg_outputs
(presence of 'meanImg_chan2') unless --n_chan is given. Existing outputs are skipped unless --force.
"""
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from pathlib import Path
from align_images import adjust_intensity
import tifffile

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root, for ceph_paths
from ceph_paths import ceph_path


def resolve_plane_dirs(path):
    """Return plane folders (containing reg_outputs.npy) for a plane folder or a suite2p folder."""
    path = Path(ceph_path(str(path)))
    if not path.is_dir():
        sys.exit(f"Folder not found: {path}")
    if (path / 'reg_outputs.npy').exists():
        return [path]
    planes = sorted(p for p in path.glob('plane*') if p.is_dir())
    if not planes:
        sys.exit(f"No reg_outputs.npy or plane* folders in {path}: suite2p registration has not been run?")
    return planes


def to_ubyte(img):
    return img_as_ubyte(adjust_intensity(img_as_ubyte(rescale_intensity(img, in_range='image', out_range=(0, 1)))))


def save_mean_images(plane_dir, n_chan=None, force=False):
    reg_path = plane_dir / 'reg_outputs.npy'
    settings_path = plane_dir / 'settings.npy'
    cellpose_out_path = plane_dir / 'meanImg_seg.npy'

    # check suite2p registration
    if not reg_path.exists():
        print(f"{plane_dir}: suite2p registration has not been run.")
        return

    # check cellpose detection
    if not cellpose_out_path.exists():
        print(f"{plane_dir}: cellpose detection has not been run.")

    reg_outputs = np.load(reg_path, allow_pickle=True).item()
    if n_chan is None:
        n_chan = 2 if 'meanImg_chan2' in reg_outputs else 1

    # check if mean image already exists
    done_file = 'meanImg_chan1_chan2.png' if n_chan > 1 else 'meanImg.tiff'
    if (plane_dir / done_file).exists() and not force:
        print(f"{plane_dir}: mean images already saved, skipping (use --force to overwrite).")
        return

    # save reference image (useful as a reg quality check)
    refImg = reg_outputs['refImg']
    plt.imshow(refImg, cmap='gray')
    plt.title('Reference Image')
    plt.savefig(plane_dir / 'refImg.png', dpi=300, bbox_inches='tight')
    plt.close()

    # print registration settings
    print(f"Folder: {plane_dir} (n_chan={n_chan})")
    if settings_path.exists():
        reg_settings = np.load(settings_path, allow_pickle=True).item()['registration']
        for key in ['smooth_sigma_time', 'smooth_sigma', 'nimg_init', 'nonrigid', 'block_size',
                    'spatial_taper', 'maxregshift']:
            print(f"  {key}: {reg_settings.get(key)}")
    else:
        print("  settings.npy not found, registration settings not printed.")

    # save mean images
    img1 = to_ubyte(reg_outputs['meanImg'])

    if n_chan > 1:
        img2 = to_ubyte(reg_outputs['meanImg_chan2'])

        # Save individual single-channel tiffs and pngs for cellpose input
        tifffile.imwrite(plane_dir / 'meanImg_chan1.tiff', img1)
        tifffile.imwrite(plane_dir / 'meanImg_chan2.tiff', img2)
        plt.imsave(str(plane_dir / 'meanImg_chan1.png'), img1, cmap='gray')
        plt.imsave(str(plane_dir / 'meanImg_chan2.png'), img2, cmap='gray')

        # Save side-by-side comparison for visual inspection
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        ax[0].imshow(img1, cmap='gray')
        ax[0].set_axis_off()
        ax[0].set_title('Chan1')
        ax[1].imshow(img2, cmap='gray')
        ax[1].set_axis_off()
        ax[1].set_title('Chan2')
        plt.savefig(plane_dir / 'meanImg_chan1_chan2.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    else:
        plt.imsave(str(plane_dir / 'meanImg.png'), img1, cmap='gray')
        tifffile.imwrite(plane_dir / 'meanImg.tiff', img1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save suite2p mean images for one or more suite2p folders.")
    parser.add_argument("suite2p_paths", nargs="+", help="suite2p folder(s) or plane folder(s)")
    parser.add_argument("--n_chan", type=int, choices=[1, 2], default=None,
                        help="number of channels (default: detect from reg_outputs)")
    parser.add_argument("--force", action="store_true", help="overwrite existing mean images")
    args = parser.parse_args()

    for path in args.suite2p_paths:
        for plane_dir in resolve_plane_dirs(path):
            save_mean_images(plane_dir, n_chan=args.n_chan, force=args.force)

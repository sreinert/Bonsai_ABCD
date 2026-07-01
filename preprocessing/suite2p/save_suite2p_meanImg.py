import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
import os
from pathlib import Path
from align_images import adjust_intensity
import tifffile

if Path("/ceph").exists():
    ROOT = "/ceph/mrsic_flogel/public/projects"
elif Path("/Volumes/mrsic_flogel").exists():
    ROOT = "/Volumes/mrsic_flogel/public/projects"
else:
    ROOT = "Y:/public/projects"
basepath = Path(ROOT) / "SaReMaRa_20260519_NeuralAnalysis/GoalProgressCohort" # "SaReMaRa_20260519_NeuralAnalysis/Cohort3" or ./Cohort2 ; "SaReMaRa_20260629_GoalProgress" ; SuKuSaRe_20250923_HFScohort3" ; "AtApSuKuSaRe_20250129_HFScohort2" ; 

sessions = ['sub-006/ses-001-screening']

suite2p_path = 'funcimg/rec2/suite2p/plane0' # 'funcimg/suite2p/plane0' for a typical session; some old data e.g. Cohort2 recordings are stored as 'funcimg/Session/suite2p/plane0'
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

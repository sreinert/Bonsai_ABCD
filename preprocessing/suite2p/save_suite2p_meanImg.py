import numpy as np
import matplotlib.pyplot as plt

from skimage import io, img_as_float, img_as_ubyte
from skimage.exposure import rescale_intensity
import os, re
from pathlib import Path
from align_images import adjust_intensity

basepath = Path('/Volumes/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/')

sessions = ['TAA0000066/ses-023_date-20250516_protocol-t17']

suite2p_path = 'funcimg/Session/suite2p/plane0'
n_chan = 1 # 1 / 2

for session in sessions:
    img_path = basepath / session / suite2p_path / 'ops.npy'
    cellpose_path = basepath / session / suite2p_path / 'meanImg_seg.npy'

    # check suite2p registration
    if not os.path.exists(img_path):
        print(f"Session {session} has not been registered with suite2p.")
        continue

    # check cellpose detection 
    if not os.path.exists(cellpose_path):
        print(f"Session {session} has not been analysed with cellpose.")

    # check if mean image already exists
    if os.path.exists(os.path.join(basepath, session, suite2p_path, 'meanImg.tiff')):
        continue

    ops = np.load(img_path, allow_pickle=True).item()
    img1 = ops['meanImg']
    img1 = img_as_ubyte(adjust_intensity(img_as_ubyte(rescale_intensity(img1, in_range='image', out_range=(0, 1)))))

    if n_chan > 1:  
        img2 = ops['meanImg_chan2']
        img2 = adjust_intensity(img_as_ubyte(rescale_intensity(img2, in_range='image', out_range=(0, 1))))

        fig, ax = plt.subplots(1,2, figsize=(12,8))
        ax = ax.ravel()

        ax[0].imshow(img1, cmap='gray')
        ax[0].set_axis_off()
        ax[0].set_title('Chan1')

        ax[1].imshow(img2, cmap='gray')
        ax[1].set_axis_off()
        ax[1].set_title('Chan2')

        # plt.show()
        plt.savefig(os.path.join(basepath, session, suite2p_path, 'meanImg_chan1_chan2.png'), dpi=300, bbox_inches='tight')

    else: 
        plt.imsave(os.path.join(basepath, session, suite2p_path, 'meanImg.png'), img1, cmap='gray')
        plt.imsave(os.path.join(basepath, session, suite2p_path, 'meanImg.tiff'), img1, cmap='gray')


# Check reference image
# for session in sessions: 
#     img_path = basepath / session / suite2p_path / 'ops.npy'
#     cellpose_path = basepath / session / suite2p_path / 'meanImg_seg.npy'

#     ops = np.load(img_path, allow_pickle=True).item()
#     refImg = ops['refImg']

#     print(ops['smooth_sigma_time'], ops['smooth_sigma'], ops['nimg_init'])
#     plt.imshow(refImg, cmap='gray')
#     plt.title('Reference Image')
#     plt.show()

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.segmentation import find_boundaries

# import h5py
# from scipy.signal import find_peaks
# from sklearn.preprocessing import normalize

# import pickle
# import parse_session_functions
# from suite2p.extraction import dcnv
# import scipy.stats as stats
# from scipy.ndimage import gaussian_filter1d
# import matplotlib.patches as patches
# import scipy.cluster.hierarchy as sch
# import pandas as pd


# new version

# new version
def concat_masks(image_mask: pd.Series):
    masks = np.zeros(image_mask.loc[0].shape)
    for n in range(len(image_mask)):
        tmp = image_mask.loc[n]

        mask_bool = find_boundaries((tmp > 0).astype(int), mode='inner').astype(bool)
        write_here = mask_bool & (masks == 0)
        masks[write_here] = n + 1
    return masks

def show_cell_fov(cell:int, meanImg: np.array, mask: np.array):
    """
    Show the cell's field of view, as a zoom in and the full field of view.
    """
    mask_br = np.where(
        mask == cell, 1.0,          # exact match
        np.where(mask != 0, -1, 0) # nonzero but not match → 0.5, else 0
    )
    mask_bool = (mask_br != 0).astype(float)

    #create a crop box around the cell
    x, y = np.where(mask == cell)
    x_min = max(0, x.min() - 10)
    x_max = min(mask.shape[0], x.max() + 10)
    y_min = max(0, y.min() - 10)
    y_max = min(mask.shape[1], y.max() + 10)
    crop_mask = meanImg[x_min:x_max, y_min:y_max]

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(crop_mask, cmap='gray')
    ax[0].imshow(mask_br[x_min:x_max, y_min:y_max], alpha=mask_bool[x_min:x_max, y_min:y_max], cmap='bwr')
    ax[0].set_title(f'Cell {cell} zoomed in')

    ax[1].imshow(meanImg, cmap='gray')
    ax[1].imshow(mask_br,alpha=mask_bool,cmap='bwr')
    ax[1].set_title(f'Cell {cell} in full field of view')
    plt.tight_layout()
    plt.show()

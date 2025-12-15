import numpy as np
import matplotlib.pyplot as plt

from skimage import io, img_as_float, img_as_ubyte
from skimage.exposure import rescale_intensity, match_histograms
from skimage.registration import phase_cross_correlation
from skimage.restoration import denoise_bilateral
from scipy.ndimage import shift, gaussian_filter
import cv2
import os, re
from pathlib import Path

def align_image(img1, img2):
    """Align img2 to img1 using phase cross-correlation (translation only)."""

    # Create a mask of the central region to avoid noise artefacts on the edges
    h, w = img1.shape

    mask = np.zeros((h, w), dtype=bool)
    mask[h//4 : 3*h//4, w//4:3*w//4] = True

    shift_vals, _, _ = phase_cross_correlation(img1, img2, upsample_factor=10, reference_mask=mask)

    aligned_img2 = shift(img2, shift=shift_vals, mode='constant', cval=0)

    return aligned_img2


def overlay_images(img1, img2):
    """Creates an RGB overlay: img1 in red, img2 in green."""

    # Convert grayscale images to 3-channel BGR
    img1_color = cv2.merge([img1, np.zeros_like(img1), np.zeros_like(img1)])  # Red
    img2_color = cv2.merge([np.zeros_like(img2), img2, np.zeros_like(img2)])  # Green

    # Overlay images
    overlay = cv2.addWeighted(img1_color, 0.5, img2_color, 0.5, 0)
    
    return overlay.astype(np.uint8)


def adjust_intensity(img):
    p2, p98 = np.percentile(img, (2, 98))  # Get min/max intensities
    img = rescale_intensity(img, in_range=(p2, p98), out_range=(0, 255))

    return img.astype(np.uint8)

def main():
    # Load grayscale images
    # '/Volumes/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/TAA0000066/ses-007_date-20250304_ptrotocol-t2/funcimg/Session/suite2p/plane0'
    suite2p_reg = True  # Matlab or suite2p registration
    basepath = Path('/Volumes/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/')
    animal = 'TAA0000061'
    session2 = 'ses-007_date-20250307_protocol-t2'  
    session1 = 'ses-005_date-20250228_protocol-t1' 
    suite2p_path = 'funcimg/Session/suite2p/plane0'
    matlab_path = 'funcimg/Session'

    if suite2p_reg:
        img_path1 = basepath / animal / session1 / suite2p_path / 'ops.npy'
        img_path2 = basepath / animal / session2 / suite2p_path / 'ops.npy'

        img1 = np.load(img_path1, allow_pickle=True).item()['meanImg']
        img2 = np.load(img_path2, allow_pickle=True).item()['meanImg']

        # Convert to 8-bit grayscale (if needed)
        img1 = img_as_ubyte(rescale_intensity(img1, in_range='image', out_range=(0, 1)))
        img2 = img_as_ubyte(rescale_intensity(img2, in_range='image', out_range=(0, 1)))

    else:
        average_registered_file = 'RegisteredAverageCh2L1.tiff'
        img_path1 = basepath / animal / session1 / matlab_path / average_registered_file
        img_path2 = basepath / animal / session2 / matlab_path / average_registered_file
        img1 = io.imread(img_path1, as_gray=True)
        img2 = io.imread(img_path2, as_gray=True)

        img1 = img_as_ubyte(img_as_float(img1))
        img2 = img_as_ubyte(img_as_float(img2))


    # Adjust intensity 
    img1 = adjust_intensity(img1)
    img2 = adjust_intensity(img2)

    # Align images
    aligned_img2 = align_image(img1, img2)

    # Create overlay
    overlay = adjust_intensity(overlay_images(img1, aligned_img2))

    # Plot the original and transformed images 
    fig, ax = plt.subplots(2,2, figsize=(12,8))
    ax = ax.ravel()

    ax[0].imshow(img1, cmap='gray')
    ax[0].set_axis_off()
    ax[0].set_title('Reference image')

    ax[1].imshow(img2, cmap='gray')
    ax[1].set_axis_off()
    ax[1].set_title('Offset image')

    ax[2].imshow(aligned_img2, cmap='gray')
    ax[2].set_axis_off()
    ax[2].set_title('Aligned img2')

    ax[3].imshow(overlay)
    ax[3].set_axis_off()
    ax[3].set_title("Aligned and overlaid")

    plt.tight_layout()

    # Get the training timepoints
    match_t0 = re.search(r'protocol-(t\d+)', str(img_path1))
    match_t1 = re.search(r'protocol-(t\d+)', str(img_path2))

    t0 = match_t0.group(1) if match_t0 else "unknown"
    t1 = match_t1.group(1) if match_t1 else "unknown"

    # Save image 
    plt.savefig(os.path.join(basepath, animal, f"aligned_overlay_{t0}_vs_{t1}.png"), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
"""
Adapted from https://github.com/MouseLand/suite2p/blob/main/jupyter/Run%20Suite2p.ipynb 
"""
import argparse
import os
import suite2p
from pathlib import Path
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.util import img_as_ubyte
from align_images import adjust_intensity

print("First sys.path entry:", sys.path[0])
print("Using suite2p from:", suite2p.__file__)

parser = argparse.ArgumentParser(description="Run Suite2p with animal and session inputs.")
parser.add_argument('--animal', type=str, default='TAA0000061', help="The animal ID (e.g. 'TAA0000066')")
parser.add_argument('--session', type=str, default='ses-007_date-20250307_protocol-t2', help="The session ID (e.g. 'ses-007_date-20250304_protocol-t2')")
parser.add_argument('--basepath', type=str, default='AtApSuKuSaRe_20250129_HFScohort2', help='The higher-level path where data are stored for this cohort')
args = parser.parse_args()

# Data directory
if Path("/ceph").exists():
    ROOT = "/ceph/mrsic_flogel/public/projects"
else:
    ROOT = "/Volumes/mrsic_flogel/public/projects"
animal =  args.animal 
session = args.session 
basepath = args.basepath
basepath = f"/{ROOT}/{basepath}"  

if 'AtApSuKuSaRe_20250129_HFScohort2' in basepath:
    tiff_path = 'funcimg/Session'
elif 'AtAp_20260119_SequenceCompression/rawdata' in basepath:
    tiff_path = 'funcimg'
elif 'AtAp_20260119_SequenceCompression/funcimg_screening' in basepath:
    tiff_path = ''

data_path = os.path.join(basepath, animal, session, tiff_path)
save_path = data_path 
print(f'Running suite2p in {data_path}')

# ---------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------

parameter_grid = {
    'smooth_sigma': [1.15, 1.5, 2],
    'maxregshift': [0.2, 0.3, 0.4],
    'maxregshiftNR': [5, 10, 15]
}

parameter_names = list(parameter_grid.keys())
parameter_values = list(parameter_grid.values())

combinations = list(itertools.product(*parameter_values))

print(f"Number of parameter combinations: {len(combinations)}")

# ---------------------------------------------------------
# Sweep output directory
# ---------------------------------------------------------

sweep_path = os.path.join(
    data_path,
    'suite2p_registration_sweep'
)

os.makedirs(sweep_path, exist_ok=True)


# ---------------------------------------------------------
# Run sweep
# ---------------------------------------------------------

for run_idx, combination in enumerate(combinations):

    # Convert the tuple into a dictionary
    params = dict(zip(parameter_names, combination))

    print("\n" + "=" * 70)
    print(f"RUN {run_idx + 1}/{len(combinations)}")

    for name, value in params.items():
        print(f"{name}={value}")

    print("=" * 70)


    # -----------------------------------------------------
    # Unique output directory
    # -----------------------------------------------------

    run_name = "_".join(
        f"{name}_{value}"
        for name, value in params.items()
    )

    save_path = os.path.join(
        sweep_path,
        run_name
    )

    # -----------------------------------------------------
    # Check whether meanImg already exists
    # -----------------------------------------------------

    meanimg_path = os.path.join(save_path, 'suite2p', 'plane0')

    if os.path.exists(os.path.join(meanimg_path, 'meanImg.png')):
        print(f"Mean image already exists, skipping: {run_name}")
        continue

    # -----------------------------------------------------
    # Check whether Suite2p has already been run
    # -----------------------------------------------------

    ops_path = os.path.join(
        save_path, 'suite2p', 'plane0', 'ops.npy')
    print(ops_path)

    if os.path.exists(ops_path):

        print(f"Suite2p output already exists.")
        print(f"Loading: {ops_path}")

        output_ops = np.load(ops_path, allow_pickle=True).item()

        img1 = output_ops['meanImg']
        img1 = img_as_ubyte(adjust_intensity(img_as_ubyte(rescale_intensity(img1, in_range='image', out_range=(0, 1)))))
        plt.imsave(os.path.join(meanimg_path, 'meanImg.png'), img1, cmap='gray')
        plt.imsave(os.path.join(meanimg_path, 'meanImg.tiff'), img1, cmap='gray')
        plt.close()
        print('Saved meanImg')
    
        if output_ops['nchannels'] == 2:  
            img2 = output_ops['meanImg_chan2']
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
            plt.savefig(os.path.join(meanimg_path, 'meanImg_chan1_chan2.png'), dpi=300, bbox_inches='tight')
            plt.imsave(os.path.join(meanimg_path, 'meanImg2.png'), img2, cmap='gray')
            plt.imsave(os.path.join(meanimg_path, 'meanImg2.tiff'), img2, cmap='gray')
            plt.close()
            print('Saved meanImg2')

        continue

    else:

        print("Suite2p output not found.")
        print("Running Suite2p...")

        os.makedirs(save_path, exist_ok=True)

        # -----------------------------------------------------
        # Suite2p settings
        # -----------------------------------------------------

        ops = suite2p.default_settings()
        db = suite2p.default_db()


        # General
        ops['tau'] = 0.4
        ops['fs'] = 45


        # Run control
        ops['run']['do_detection'] = False


        # Registration
        ops['registration']['nonrigid'] = True
        # ops['registration']['smooth_sigma'] = 1.15
        ops['registration']['smooth_sigma_time'] = 0
        ops['registration']['nimg_init'] = 500
        ops['registration']['two_step_registration'] = True
        ops['registration']['do_bidiphase'] = False
        ops['registration']['batch_size'] = 200


        # Parameters being swept
        for name, value in params.items():
            ops['registration'][name] = value

        # Detection
        ops['detection']['algorithm'] = 'sparsery'
        ops['detection']['sparsery_settings']['spatial_scale'] = 0


        # Database
        db['data_path'] = [data_path]
        db['save_path0'] = save_path
        db['nchannels'] = 2
        db['keep_movie_raw'] = False


        # -----------------------------------------------------
        # Run Suite2p
        # -----------------------------------------------------

        output_ops = suite2p.run_s2p(settings=ops, db=db)

        # Save meanImg
        img1 = output_ops['meanImg']
        img1 = img_as_ubyte(adjust_intensity(img_as_ubyte(rescale_intensity(img1, in_range='image', out_range=(0, 1)))))
        plt.imsave(os.path.join(meanimg_path, 'meanImg.png'), img1, cmap='gray')
        plt.imsave(os.path.join(meanimg_path, 'meanImg.tiff'), img1, cmap='gray')
        plt.close()
        print('Saved meanImg')
    
        if output_ops['nchannels'] == 2:  
            img2 = output_ops['meanImg_chan2']
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
            plt.savefig(os.path.join(meanimg_path, 'meanImg_chan1_chan2.png'), dpi=300, bbox_inches='tight')
            plt.imsave(os.path.join(meanimg_path, 'meanImg2.png'), img2, cmap='gray')
            plt.imsave(os.path.join(meanimg_path, 'meanImg2.tiff'), img2, cmap='gray')
            plt.close()
            print('Saved meanImg2')
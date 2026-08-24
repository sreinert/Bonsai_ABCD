"""
Adapted from https://github.com/MouseLand/suite2p/blob/main/jupyter/Run%20Suite2p.ipynb 
"""
import argparse
import os
import suite2p
from pathlib import Path
import sys
import itertools

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
    'th_badframes': [0.5, 1.0, 1.5],
    'spatial_taper': [2.0, 3.45, 5.0],
    'maxregshift': [0.05, 0.1, 0.15],
}

combinations = list(itertools.product(
    parameter_grid['th_badframes'],
    parameter_grid['spatial_taper'],
    parameter_grid['maxregshift'],
))

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

for run_idx, (th_badframes, spatial_taper, maxregshift) in enumerate(combinations):

    print("\n" + "=" * 70)
    print(f"RUN {run_idx + 1}/{len(combinations)}")
    print(
        f"th_badframes={th_badframes}, "
        f"spatial_taper={spatial_taper}, "
        f"maxregshift={maxregshift}"
    )
    print("=" * 70)


    # -----------------------------------------------------
    # Unique output directory
    # -----------------------------------------------------

    run_name = (
        f"badframes_{th_badframes}"
        f"_taper_{spatial_taper}"
        f"_maxshift_{maxregshift}"
    )

    save_path = os.path.join(
        sweep_path,
        run_name
    )

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
    ops['registration']['smooth_sigma'] = 1.15
    ops['registration']['smooth_sigma_time'] = 0
    ops['registration']['nimg_init'] = 500
    ops['registration']['two_step_registration'] = True
    ops['registration']['do_bidiphase'] = False
    ops['registration']['batch_size'] = 200


    # Parameters being swept
    ops['registration']['th_badframes'] = th_badframes
    ops['registration']['spatial_taper'] = spatial_taper
    ops['registration']['maxregshift'] = maxregshift


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
# snr_thres, norm_frames, smooth_sigma


# # db = {
# #     "data_path": ['.'], # Directory where your input files are located
# #     "save_path0": '/content/suite2p_output', # Directory where you want suite2p to write output files.
# #     "file_list": [fname], # Specify files you'd like to specifically use in the data_path
# #     "input_format": "tif",
# #     "nplanes": 1, # each tiff has these many planes in sequence
# #     "nchannels": 1, # each tiff has these many channels per plane
# #     "keep_movie_raw": True,
# #     "batch_size": 200, # we will decrease the batch_size in case low RAM on computer
# # }

# # ops['detection']['spatial_scale'] = 0
# ops['tau'] = 0.4
# # ops['diameter'] = 0
# ops['fs'] = 45
# ops['torch_device'] = ['cpu']

# # ops['registration']['norm_frames'] = True
# ops['registration']['nonrigid'] = False
# ops['registration']['smooth_sigma'] = 1.15 # adjust for low SNR (default --1.15)
# ops['registration']['nimg_init'] = 1000 # adjust for low SNR (default --1000)
# ops['registration']['snr_thresh'] = 0.5
# ops['registration']['two_step_registration'] = False # (default --False)
# ops['registration']['smooth_sigma_time'] = 0 # adjust for low SNR (default --0)
# ops['registration']['reg_tif'] = True
# ops['registration']['reg_tif_chan2'] = True
# ops['registration']['align_by_chan2'] = True
# ops['registration']['save_path'] = [data_path]
# ops['do_optimize_motion_params'] = True
# ops['functional_chan'] = 1

# ops['keep_movie_raw'] = False # (default --False)
# # ops['batch_size'] = 500

# ops['do_detection'] = False
# # ops['roidetect'] = False
# ops['sparse_mode'] = True
# ops['connected'] = True
# ops['anatomical_only'] = 2
# # ops['pretrained_model'] = 'cyto3'
# # ops['detection']['algorithm'] = 'cellpose'
# # ops['detection']['cellpose_settings']['cellpose_model'] = 'cyto3'
# # ops['detection']['cellpose_settings']['img'] = 'meanImg'
# # ops['detection']['spatial_scale'] = 1

# db['data_path'] = [data_path] # db instead of ops for new suite2p version
# db['save_path'] = [data_path]
# db['nchannels'] = 2
# db['batch_size'] = 500

# print("cellpose_settings:", ops["detection"]["cellpose_settings"])
# print("registration_settings:", ops["registration"])
# # Run suite2p 
# output_ops = suite2p.run_s2p(settings=ops, db=db)


# These params work - but no meanImg2
# # ops['detection']['spatial_scale'] = 0
# ops['tau'] = 0.4
# # ops['diameter'] = 0
# ops['fs'] = 45
# ops['nchannels'] = 2
# ops['nonrigid'] = False
# ops['smooth_sigma'] = 1.15 # adjust for low SNR (default --1.15)
# ops['smooth_sigma_time'] = 1 # adjust for low SNR (default --1)
# ops['nimg_init'] = 100 # adjust for low SNR (default --1000)
# ops['two_step_registration'] = False # (default --False)

# ops['reg_tif'] = True
# ops['reg_tif_chan2'] = True
# ops['do_optimize_motion_params'] = True
# ops['registration']['align_by_chan2'] = True

# ops['keep_movie_raw'] = False # (default --False)
# ops['batch_size'] = 500

# ops['do_detection'] = False
# # ops['roidetect'] = True
# # ops['sparse_mode'] = True
# ops['connected'] = True
# ops['anatomical_only'] = 2
# # ops['pretrained_model'] = 'cyto3'
# # ops['detection']['algorithm'] = 'cellpose'
# # ops['detection']['cellpose_settings']['cellpose_model'] = 'cyto3'
# # ops['detection']['spatial_scale'] = 1
# db['data_path'] = [data_path] # db instead of ops for new suite2p version
# db['save_path'] = [data_path]
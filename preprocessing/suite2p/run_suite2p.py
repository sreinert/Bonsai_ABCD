"""
Adapted from https://github.com/MouseLand/suite2p/blob/main/jupyter/Run%20Suite2p.ipynb 
"""
import argparse
import os
import suite2p
from pathlib import Path
import sys

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

# Settings 
ops = suite2p.default_settings() # default_settings instead of default_ops for new suite2p version
db = suite2p.default_db()

# Martina's params

## General
ops['tau'] = 0.4
ops['fs'] = 45

## Run control
ops['run']['do_detection'] = False

## Registration
ops['registration']['nonrigid'] = True
ops['registration']['smooth_sigma'] = 1.15
ops['registration']['smooth_sigma_time'] = 0
ops['registration']['nimg_init'] = 400
ops['registration']['two_step_registration'] = False
ops['registration']['batch_size'] = 200  # this is reg batch_size, separate from extraction

## Detection (native sparsery, no Cellpose)
ops['detection']['algorithm'] = 'sparsery'  # default
ops['detection']['sparsery_settings']['spatial_scale'] = 0

## Anatomical ROI Detection (suite2p with Cellpose) 
# ops['detection']['algorithm'] = 'cellpose'
# ops['detection']['cellpose_settings']['cellpose_model'] = 'cyto2'  # or 'cyto3', 'cpsam'
# ops['diameter'] = [0., 0.]  # 0 = auto-estimate cell size

## db
db['data_path'] = [data_path]
db['save_path0'] = data_path  
db['nchannels'] = 2
db['keep_movie_raw'] = False

# Run suite2p 
output_ops = suite2p.run_s2p(settings=ops, db=db)

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
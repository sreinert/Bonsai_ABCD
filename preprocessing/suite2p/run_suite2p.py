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
args = parser.parse_args()

# Data directory
if Path("/ceph").exists():
    ROOT = "/ceph/mrsic_flogel/public/projects" # cluster
elif Path("/Volumes/mrsic_flogel").exists():
    ROOT = "/Volumes/mrsic_flogel/public/projects" # Mac OS
else:
    ROOT = "Y:/public/projects" # Win OS - change drive letter if needed
basepath = f"{ROOT}/AtApSuKuSaRe_20250129_HFScohort2"  
animal =  args.animal 
session = args.session 
tiff_path = 'funcimg/Session'

data_path = os.path.join(basepath, animal, session, tiff_path)

## Settings 
ops = suite2p.default_settings() # default_settings instead of default_ops for new suite2p version
db = suite2p.default_db()

## General
ops['tau'] = 0.4
ops['fs'] = 45

## Run control
ops['run']['do_detection'] = True

## Registration
ops['registration']['nonrigid'] = False
ops['registration']['smooth_sigma'] = 1.15
ops['registration']['smooth_sigma_time'] = 1
ops['registration']['nimg_init'] = 1000
ops['registration']['two_step_registration'] = False
ops['registration']['batch_size'] = 500  # this is reg batch_size, separate from extraction

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
db['nchannels'] = 1
db['keep_movie_raw'] = False

# Run suite2p 
output_ops = suite2p.run_s2p(settings=ops, db=db)



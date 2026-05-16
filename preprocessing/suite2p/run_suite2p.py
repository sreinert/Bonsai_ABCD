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
    ROOT = "/ceph/mrsic_flogel/public/projects"
else:
    ROOT = "/Volumes/mrsic_flogel/public/projects"
basepath = f"/{ROOT}/AtApSuKuSaRe_20250129_HFScohort2"  
animal =  args.animal 
session = args.session 
tiff_path = 'funcimg/Session'

data_path = os.path.join(basepath, animal, session, tiff_path)

# Settings 
ops = suite2p.default_settings() # default_settings instead of default_ops for new suite2p version
db = suite2p.default_db()

ops['detection']['spatial_scale'] = 0
ops['tau'] = 0.4
ops['fs'] = 45
ops['nchannels'] = 1
ops['nonrigid'] = False
ops['smooth_sigma'] = 1.15 # adjust for low SNR (default --1.15)
ops['smooth_sigma_time'] = 1 # adjust for low SNR (default --1)
ops['nimg_init'] = 1000 # adjust for low SNR (default --1000)
ops['two_step_registration'] = False # (default --False)
ops['keep_movie_raw'] = False # (default --False)
ops['batch_size'] = 500
ops['roidetect'] = True
ops['sparse_mode'] = True
ops['connected'] = True
ops['anatomical_only'] = 2
ops['pretrained_model'] = 'cyto2'
ops['detection']['algorithm'] = 'cellpose'
ops['detection']['cellpose_settings']['cellpose_model'] = 'cyto2'
ops['detection']['spatial_scale'] = 0
db['data_path'] = [data_path] # db instead of ops for new suite2p version
db['save_path'] = [data_path]

# Run suite2p 
output_ops = suite2p.run_s2p(settings=ops, db=db)



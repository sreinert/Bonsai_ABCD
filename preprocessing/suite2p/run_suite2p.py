"""
Adapted from https://github.com/MouseLand/suite2p/blob/main/jupyter/Run%20Suite2p.ipynb 
"""
import argparse
import os
import suite2p

parser = argparse.ArgumentParser(description="Run Suite2p with animal and session inputs.")
parser.add_argument('--animal', type=str, default='TAA0000061', help="The animal ID (e.g. 'TAA0000066')")
parser.add_argument('--session', type=str, default='ses-007_date-20250307_protocol-t2', help="The session ID (e.g. 'ses-007_date-20250304_protocol-t2')")
args = parser.parse_args()

# Data directory
# basepath = "/Volumes/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/"  # PC
# basepath = "Z:/public/projects/AtApSuKuSaRe_20250129_HFScohort2"  # 313
basepath = "/ceph/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2"  # HPC
animal =  args.animal 
session = args.session 
tiff_path = 'funcimg/Session'

data_path = os.path.join(basepath, animal, session, tiff_path)

# Settings 
ops = suite2p.default_ops()

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
ops['data_path'] = [data_path]
ops['save_path'] = [data_path]

# Run suite2p 
output_ops = suite2p.run_s2p(ops=ops)



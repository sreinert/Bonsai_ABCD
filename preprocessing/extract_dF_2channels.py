
import numpy as np
from pathlib import Path
import importlib
import sys, os
import argparse

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

import cellTV.cellTV_functions_cohort3 as celltv
importlib.reload(celltv)

parser = argparse.ArgumentParser(description="Extract dG/R with animal and session inputs.")
parser.add_argument('--animal', type=str, default='010', help="The animal ID (e.g. '010')")
parser.add_argument('--session', type=str, default='full010', help="The session ID (e.g. 'full010')")
args = parser.parse_args()

animal =  args.animal 
session = args.session 

# Data directory
# funcimg_root = Path(f"/Volumes/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/preprocessed_funcimg_Nov2025/derivatives") # PC
# behav_root = Path(f"/Volumes/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/preprocessed_behav_Nov2025/derivatives") # PC
funcimg_root = Path(f"/ceph/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/preprocessed_funcimg_Nov2025/derivatives") # HPC
behav_root = Path(f"/ceph/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/preprocessed_behav_Nov2025/derivatives") # HPC

mouse_path = Path(behav_root) / f"sub-{animal}" 
for folder in mouse_path.iterdir():
    if folder.is_dir() and session in folder.name:
        print(f"Found folder: {folder}")
        behav_path = folder
        save_path = behav_path / 'funcimg' 
        if not os.path.exists(save_path):
            os.makedirs(save_path)

# Load dF from green and red channels 
funcimg_data, _, dFred, dF_GR = celltv.load_dF_data(animal, session, funcimg_root, behav_root, save_path, reload=True)
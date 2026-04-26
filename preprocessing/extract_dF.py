from pathlib import Path
import importlib
import sys, os
import argparse

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

parser = argparse.ArgumentParser(description="Extract dF/F with mouse and session inputs.")
parser.add_argument('--mouse', type=str, default='TAA0000059', help="The mouse ID (e.g. 'TAA0000059')")
parser.add_argument('--session', type=str, default='t3', help="The session ID (e.g. 't3')")
parser.add_argument('--cohort', type=str, default='2', help="The mouse cohort (e.g. '2')")
args = parser.parse_args()

mouse =  args.mouse 
session = args.session 
cohort = args.cohort

# Load functions according to cohort 
if cohort == '2':
    import cellTV.cellTV_functions_cohort2 as celltv

elif cohort == '3':
    import cellTV.cellTV_functions_cohort3 as celltv

importlib.reload(celltv)

# Calculate dF and find valid neurons
if cohort == '2':
    base_path = Path("/ceph/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/")

    _ = celltv.load_dF(base_path, mouse, session, reload=True)

# Calculate dF from green and red channels and find valid neurons
elif cohort == '3':
    funcimg_root = Path(f"/ceph/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/preprocessed_funcimg_Nov2025/derivatives") 
    behav_root = Path(f"/ceph/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/preprocessed_behav_Nov2025/derivatives") 

    mouse_path = Path(behav_root) / f"sub-{mouse}" 
    for folder in mouse_path.iterdir():
        if folder.is_dir() and session in folder.name:
            print(f"Found folder: {folder}")
            behav_path = folder
            save_path = behav_path / 'funcimg' 
            if not os.path.exists(save_path):
                os.makedirs(save_path)

    _ = celltv.load_dF(mouse, session, funcimg_root, behav_root, save_path, reload=True)


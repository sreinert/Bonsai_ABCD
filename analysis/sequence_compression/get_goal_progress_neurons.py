import numpy as np
from pathlib import Path
import importlib
import argparse
import sys, os

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import neural_analysis_helpers

parser = argparse.ArgumentParser(description="Get goal-progress tuned neurons.")
parser.add_argument('--mouse', type=str, default='010', help="The mouse ID (e.g. '010')")
parser.add_argument('--session', type=str, default='full010', help="The session ID (e.g. 'full010')")
parser.add_argument('--stage', type=str, default='t5', help="The imaging timepoint (e.g. t5)")
parser.add_argument('--cohort', type=str, default='2', help="Behavioural cohort the mouse belongs to")
args = parser.parse_args()

mouse =  args.mouse 
session_id = args.session 
stage = args.stage
cohort = args.cohort

# Load functions according to cohort 
if cohort == '2':
    import preprocessing.parse_session_functions_cohort2 as parse_session_functions
    import cellTV.cellTV_functions_cohort2 as cellTV
elif cohort == '3':
    import preprocessing.parse_session_functions_cohort3 as parse_session_functions
    import cellTV.cellTV_functions_cohort3 as cellTV

importlib.reload(parse_session_functions)
importlib.reload(neural_analysis_helpers)
importlib.reload(cellTV)

if int(cohort) == 3:
    funcimg_root = Path(f"/ceph/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/preprocessed_funcimg_Nov2025/derivatives") 
    behav_root = Path(f"/ceph/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/preprocessed_behav_Nov2025/derivatives") 

    mouse_path = Path(behav_root) / f"sub-{mouse}" 
    for folder in mouse_path.iterdir():
        if folder.is_dir() and session_id in folder.name:
            print(f"Found folder: {folder}")
            save_path = folder / 'funcimg' 
            if not os.path.exists(save_path):
                os.makedirs(save_path)

    # Load dF and valid neurons - NOTE dF selected here is dG/R
    _, _, dF, neurons = cellTV.load_dF(mouse, session_id, funcimg_root, behav_root, save_path)

    # Create session struct
    if stage == 't3':
        world = 'random'
    else:
        world = 'stable'
    session = parse_session_functions.analyse_npz_pre7(mouse, session_id, behav_root, stage, world)

elif int(cohort) == 2:
    root = f"/ceph/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2" 
    _, _, _, _, date = parse_session_functions.get_session_folders(root, mouse, stage)

    # Load dF and valid neurons
    dF, neurons = cellTV.load_dF(root, mouse, stage)

    # Create session struct
    session = parse_session_functions.analyse_npz_pre7(mouse, date, stage)

event_idx = np.sort(np.concatenate([session['rewards'], session['miss_rew_idx'], session['test_rew_idx']])).astype(int)

# Get goal progress neurons
goal_progress_tuned = neural_analysis_helpers.get_goal_progress_cells(dF, neurons, session, \
                                              event_frames=event_idx, save_path=session['save_path'], \
                                                ngoals=5, bins=90, plot=False, \
                                                shuffle=True, reload=True)
import numpy as np
from pathlib import Path
import importlib
import argparse
import sys

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
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
elif cohort == '3':
    import preprocessing.parse_session_functions_cohort3 as parse_session_functions

importlib.reload(parse_session_functions)
importlib.reload(neural_analysis_helpers)

# Load data 
root = f"/ceph/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/preprocessed_behav_Nov2025/derivatives" 
session_path = parse_session_functions.find_base_path(mouse, session_id, root)

# Load dF and valid neurons
if int(cohort) == 3:
    dF, neurons = parse_session_functions.load_dF(session_path, red_chan=True)

# Create session struct
if stage == 't3':
    world = 'random'
else:
    world = 'stable'
session = parse_session_functions.analyse_npz_pre7(mouse, session_id, root, stage, world)
event_idx = np.sort(np.concatenate([session['rewards'], session['miss_rew_idx'], session['test_rew_idx']])).astype(int)

# Get goal progress neurons
goal_progress_tuned = neural_analysis_helpers.get_goal_progress_cells(dF, neurons, session, \
                                              event_frames=event_idx, \
                                                save_path=session['save_path'], ngoals=5, bins=90, plot=False, \
                                                    shuffle=True, reload=True)
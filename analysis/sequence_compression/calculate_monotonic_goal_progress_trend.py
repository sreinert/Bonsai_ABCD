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

# 1. Load data 
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

    event_idx = np.sort(np.concatenate([session['rewards'], session['miss_rew_idx'], session['test_rew_idx']])).astype(int)

elif int(cohort) == 2:
    root = f"/ceph/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2" 
    _, _, _, _, date = parse_session_functions.get_session_folders(root, mouse, stage)

    # Load dF and valid neurons
    dF, neurons = cellTV.load_dF(root, mouse, stage)

    # Create session struct
    session = parse_session_functions.analyse_npz_pre7(mouse, date, stage)

    event_idx = np.sort(np.concatenate([session['reward_idx'], session['miss_rew_idx'], session['test_rew_idx']])).astype(int)

# 2. Define saving directory 
save_dir = os.path.join(session['save_path'], 'progress_monotonic_trend')
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# 3. Get goal progress neurons
goal_progress_tuned, _, _ = neural_analysis_helpers.get_goal_progress_cells(dF, neurons, session,
                                            event_frames=event_idx, save_path=session['save_path'],
                                            ngoals=5, bins=90, plot=False,
                                            shuffle=True, reload=False)

# 4. Get max activity in a window around neuron's preferred phase for each trial 
max_window_activity = {}
for cell in goal_progress_tuned:
    max_window_activity[cell] = neural_analysis_helpers.get_max_phase_pref_goal_activity(dF, cell, session, 
                                                           event_frames=event_idx, 
                                                           ngoals=5, bins=90, period='goal', stage=None, 
                                                           plot=False, shuffle=False)

# 5. Calculate monotonic trend score
monotonic_trend_results = neural_analysis_helpers.calc_monotonic_trend_score(neurons, max_window_activity, 
                                                            ngoals=5, shuffle=True, nreps=1000, 
                                                            print_results=True, reload=False)

# 6. Plot the analysis results for each cell 
for cell in goal_progress_tuned:
    _ = neural_analysis_helpers.plot_progress_with_monotonic_trend(
                dF, cell,
                event_frames=event_idx,
                ngoals=5,
                bins=90,
                stage=stage,
                session=session,
                activity_by_cell=max_window_activity,
                trend_results=monotonic_trend_results,
                show_permutation=True,
                save_plot=True,
                save_dir=save_dir,
            )
import numpy as np
from pathlib import Path
import importlib
import argparse
import sys, os

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import alternation_analysis_helpers_v2 as alternation 

parser = argparse.ArgumentParser(description="Get goal-progress tuned neurons.")
parser.add_argument('--mouse', type=str, default='014', help="The mouse ID (e.g. '010')")
parser.add_argument('--session', type=str, default='2LM011', help="The session ID (e.g. 'full010')")
parser.add_argument('--stage', type=str, default='t3', help="The imaging timepoint (e.g. t5)")
parser.add_argument('--cohort', type=str, default='3', help="Behavioural cohort the mouse belongs to")
args = parser.parse_args()

mouse =  args.mouse 
session_id = args.session 
stage = args.stage
cohort = args.cohort

if Path("/ceph").exists():
    ROOT = "/ceph/mrsic_flogel/public/projects"
else:
    ROOT = "/Volumes/mrsic_flogel/public/projects"

# Load functions according to cohort 
if cohort == '2':
    import preprocessing.parse_session_functions_cohort2 as parse_session_functions
    import cellTV.cellTV_functions_cohort2 as cellTV
    base_path = Path(f"/{ROOT}/AtApSuKuSaRe_20250129_HFScohort2")
elif cohort == '3':
    import preprocessing.parse_session_functions_cohort3 as parse_session_functions
    import cellTV.cellTV_functions_cohort3 as cellTV
    base_path = Path(f"/{ROOT}/SuKuSaRe_20250923_HFScohort3/preprocessed_behav_Nov2025/derivatives")
    funcimg_root = Path(f"{ROOT}/SuKuSaRe_20250923_HFScohort3/preprocessed_funcimg_Nov2025/derivatives") 

importlib.reload(parse_session_functions)
importlib.reload(alternation)
importlib.reload(cellTV)
alternation.set_parse_session_functions(parse_session_functions)

#%% Load data 

if cohort == '2':
    # Load dF and valid neurons
    dF, neurons = cellTV.load_dF(base_path, mouse, stage)
    
    # Create session struct
    _, _, _, _, date = parse_session_functions.get_session_folders(base_path, mouse, stage)
    session = parse_session_functions.analyse_npz_pre7(mouse, date, stage, base_path, plot=False)
    session['stim_order'] = 'pseudorandom'

    # Define save path
    data_path = parse_session_functions.find_base_path_npz(mouse, date, base_path)
    t = parse_session_functions.extract_int(session['stage'])
    save_dir = os.path.join(data_path, 'analysis', f't{t}_linear_regression_Y2_ramp_XYrepeats')

elif cohort == '3':
    mouse_path = Path(base_path) / f"sub-{mouse}" 
    for folder in mouse_path.iterdir():
        if folder.is_dir() and session_id in folder.name:
            print(f"Found folder: {folder}")
            save_path = folder / 'funcimg' 
            analysis_path = folder / 'analysis'

    # Load dF and valid neurons - NOTE we are using dG/R
    _, _, dF, neurons = cellTV.load_dF(mouse, session_id, funcimg_root, base_path, save_path)

    # Create session struct
    if stage == 't3' or stage == 't4':
        world = 'random'
    else:
        world = 'stable'

    session = parse_session_functions.analyse_npz_pre7(mouse, session_id, base_path, stage, world, plot=False)
    session['stim_order'] = 'random'

    # Define save path
    t = parse_session_functions.extract_int(session['stage'])
    save_dir = os.path.join(analysis_path, f't{t}_linear_regression_Y2_ramp_XYrepeats')

print(f'Successfully loaded dF data for {mouse} {session_id}')

# Collect all events
event_idx = np.sort(np.concatenate([session['reward_idx'], session['miss_rew_idx'], session['nongoal_rew_idx'], session['test_rew_idx']])).astype(int)
if (mouse == 'TAA0000066' and stage == 't3') or (mouse == 'TAA0000059' and stage == 't3'):
    lick_end_idx = 160
    event_idx = event_idx[:lick_end_idx]
session['event_idx'] = event_idx

# Create save path 
print(f'Saving results and plots in {save_dir}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

#%% Bin YY data 

# Define patches
if session['stim_order'] == 'random':
    ABB_patches, BAA_patches, ABB_patches_idx, BAA_patches_idx = alternation.get_XYY_patches(session, precede_XY=True)
elif session['stim_order'] == 'pseudorandom':
    ABB_patches, BAA_patches, ABB_patches_idx, BAA_patches_idx = alternation.get_XYY_patches(session, precede_XY=False)

# Define bins based on min distance between landmarks
frames_around = alternation.get_min_frames_between_lms(session)
bins = frames_around

zscoring = False # whether to z-score dF/F inside each patch (across two YYs)
if BAA_patches:
    print('\tBAA patches found')
    A2_activity = alternation.get_Y2_activity(neurons, dF, session, BAA_patches)

    A2_ramp_regression_results_cpa = alternation.fit_linear_regression_XYlen(neurons, A2_activity, dF, session, condition='BA', data_type='Y2_ramp', 
                                                                                    bins=bins, shuffle=True, nreps=1000, zscoring=False, plot=True, sort_heatmap=True, 
                                                                                    cluster_repeats=True, save_plot=True, save_dir=save_dir, plot_dir=save_dir, reload=True)
    
if ABB_patches:
    print('\tABB patches found')
    B2_activity = alternation.get_Y2_activity(neurons, dF, session, ABB_patches)

    B2_ramp_regression_results_cpa = alternation.fit_linear_regression_XYlen(neurons, B2_activity, dF, session, condition='AB', data_type='Y2_ramp', 
                                                                                    bins=bins, shuffle=True, nreps=1000, zscoring=False, plot=True, sort_heatmap=True, 
                                                                                    cluster_repeats=True, save_plot=True, save_dir=save_dir, plot_dir=save_dir, reload=True)

   
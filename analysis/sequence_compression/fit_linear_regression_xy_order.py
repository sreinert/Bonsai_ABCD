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
    save_dir = os.path.join(data_path, 'analysis', f't{t}_linear_regression_XY_order')

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
    save_dir = os.path.join(analysis_path, f't{t}_linear_regression_XY_order')

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

#%% Define analyses

analyses = {
    'AB_A': {'condition': 'AB', 'chosen_lm': 'A', 'slice': slice(0, None, 2)},
    'AB_B': {'condition': 'AB', 'chosen_lm': 'B', 'slice': slice(1, None, 2)},
    'BA_A': {'condition': 'BA', 'chosen_lm': 'A', 'slice': slice(1, None, 2)},
    'BA_B': {'condition': 'BA', 'chosen_lm': 'B', 'slice': slice(0, None, 2)}
}

# 1. Identify XY repeats and assign an ordinal number to each landmark 
_, AB_patches, BA_patches, _, _, _ = alternation.get_repeating_XY_patches(session, min_length=4, return_list=True)
AB_ordered_patches, BA_ordered_patches = alternation.get_XY_repeat_ordering(session, min_length=4, return_list=True)

patches = {
    'AB': AB_patches,
    'BA': BA_patches
}
ordered_patches = {
    'AB': AB_ordered_patches,
    'BA': BA_ordered_patches
}

# 2. Get the mean response inside all landmarks (split by AB/BA and each lm type)
activity = {}

for name, params in analyses.items():
    activity[name] = alternation.get_mean_lm_activity(
        session,
        neurons,
        dF,
        patches[params['condition']][params['slice']]
    )

# 3. Fit linear regression 
regression_results = {}

for name, params in analyses.items():

    condition = params['condition']
    lm_slice = params['slice']

    regression_results[name] = alternation.fit_linear_regression_XYlen(
        neurons,
        activity[name],
        dF,
        session,
        x_data=ordered_patches[condition][lm_slice],
        heatmap_lms=patches[condition][lm_slice],
        condition=condition,
        chosen_lm=params['chosen_lm'],
        data_type='XY_order',
        bins=30,
        shuffle=True,
        nreps=1000,
        plot=True,
        zscoring=False,
        sort_heatmap=True,
        cluster_repeats=False,
        save_plot=True,
        save_dir=save_dir,
        plot_dir=save_dir,
        reload=False
    )

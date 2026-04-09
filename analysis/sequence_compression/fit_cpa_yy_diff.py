import numpy as np
from pathlib import Path
import importlib
import argparse
import sys, os

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import neural_analysis_helpers
import alternation_analysis_helpers as alternation

parser = argparse.ArgumentParser(description="Get goal-progress tuned neurons.")
parser.add_argument('--mouse', type=str, default='TAA0000059', help="The mouse ID (e.g. '010')")
parser.add_argument('--session', type=str, default='t3', help="The session ID (e.g. 'full010')")
parser.add_argument('--stage', type=str, default='t3', help="The imaging timepoint (e.g. t5)")
parser.add_argument('--cohort', type=str, default='2', help="Behavioural cohort the mouse belongs to")
args = parser.parse_args()

mouse =  args.mouse 
session_id = args.session 
stage = args.stage
cohort = args.cohort

# Load functions according to cohort 
if cohort == '2':
    import preprocessing.parse_session_functions_cohort2 as parse_session_functions
    base_path = Path("/Volumes/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/")
elif cohort == '3':
    import preprocessing.parse_session_functions_cohort3 as parse_session_functions
    base_path = Path("/Volumes/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/preprocessed_behav_Nov2025/derivatives")

importlib.reload(parse_session_functions)
importlib.reload(neural_analysis_helpers)
importlib.reload(alternation)
alternation.set_parse_session_functions(parse_session_functions)

#%% Load data 

if cohort == '2':
    # Load dF and valid neurons
    dF, neurons = parse_session_functions.load_dF(base_path, mouse, stage)
    
    # Create session struct
    _, _, _, _, date = parse_session_functions.get_session_folders(base_path, mouse, stage)
    session = parse_session_functions.analyse_npz_pre7(mouse, date, stage, plot=False)
    session['stim_order'] = 'pseudorandom'

    # Define save path
    data_path = parse_session_functions.find_base_path_npz(mouse, date)
    t = parse_session_functions.extract_int(session['stage'])
    save_dir = os.path.join(data_path, 'analysis', f't{t}_linear_regression_YY_diff_rew_aligned_XYrepeats_cpa')

elif cohort == '3':
    # Load dF and valid neurons
    session_path = parse_session_functions.find_base_path(mouse, session_id, base_path)
    dF, neurons = parse_session_functions.load_dF(session_path, red_chan=True)

    # Create session struct
    if stage == 't3' or stage == 't4':
        world = 'random'
    else:
        world = 'stable'

    behav_path = parse_session_functions.find_base_path(mouse, session_id, '/Volumes/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/rawdata')
    session = parse_session_functions.analyse_npz_pre7(mouse, session_id, base_path, stage, world, plot=False)
    session['stim_order'] = 'random'

    # Define save path
    t = parse_session_functions.extract_int(session['stage'])
    save_dir = os.path.join(session_path, 'analysis', f't{t}_linear_regression_YY_diff_rew_aligned_XYrepeats_cpa')

# Collect all events
event_idx = np.sort(np.concatenate([session['reward_idx'], session['miss_rew_idx'], session['nongoal_rew_idx'], session['test_rew_idx']])).astype(int)
if (mouse == 'TAA0000066' and stage == 't3') or (mouse == 'TAA0000059' and stage == 't3'):
    lick_end_idx = 160
    event_idx = event_idx[:lick_end_idx]
session['event_idx'] = event_idx

# Create save path 
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

#%% Bin YY data 

# Define patches
if session['stim_order'] == 'random':
    ABB_patches, BAA_patches, ABB_patches_idx, BAA_patches_idx = alternation.get_XYY_patches(session, include_next=False, precede_XY=True)
elif session['stim_order'] == 'pseudorandom':
    ABB_patches, BAA_patches, ABB_patches_idx, BAA_patches_idx = alternation.get_XYY_patches(session, include_next=False, precede_XY=False)

# Define bins based on min distance between landmarks
frames_around = alternation.get_min_frames_between_lms(session)
bins = frames_around

zscoring = True # whether to z-score dF/F inside each patch (across two YYs)

if BAA_patches:
    # Find start, reward and end timepoints inside YY events 
    events_AA = alternation.get_YY_events(session, BAA_patches)

    # Temporal binning within XYY patch
    binned_AA_phase_activity = alternation.get_reward_aligned_temporal_phase_binning_per_lm(neurons, dF, BAA_patches, events_AA, bins, condition='AA', zscoring=zscoring, plot=True)

    # Cluster-based permutation analysis (CPA) 
    AA_diff_regression_results_cpa = alternation.fit_linear_regression_XYlen_cpa(neurons, binned_AA_phase_activity, session, condition='BA', data_type='YY_diff', 
                                                                                bins=bins, shuffle=True, nreps=1000, cluster_thres=0.1, zscored=zscoring, 
                                                                                plot=True, sort_heatmap=True, save_plot=False, save_dir='', plot_dir='', 
                                                                                reload=True)

if ABB_patches:
    # Find start, reward and end timepoints inside YY events 
    events_BB = alternation.get_YY_events(session, ABB_patches)

    # Temporal binning within XYY patch
    binned_BB_phase_activity = alternation.get_reward_aligned_temporal_phase_binning_per_lm(neurons, dF, ABB_patches, events_BB, bins, condition='BB', zscoring=zscoring, plot=False)

    # Cluster-based permutation analysis (CPA) 
    BB_diff_regression_results_cpa = alternation.fit_linear_regression_XYlen_cpa(neurons, binned_BB_phase_activity, session, condition='AB', data_type='YY_diff', 
                                                                                bins=bins, shuffle=True, nreps=1000, cluster_thres=0.1, zscored=zscoring, 
                                                                                plot=True, sort_heatmap=True, save_plot=True, save_dir=save_dir, plot_dir=save_dir, 
                                                                                reload=True)
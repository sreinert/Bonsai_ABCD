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
    base_path = Path('/ceph/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/')
elif cohort == '3':
    raise ValueError('This has not been implemented for cohort 3 yet.')
    
importlib.reload(parse_session_functions)
importlib.reload(neural_analysis_helpers)
importlib.reload(alternation)

#%% Load data 
# Load dF and valid neurons
dF, neurons = parse_session_functions.load_dF(base_path, mouse, stage)
            
# Load session data 
_, _, _, _, date = parse_session_functions.get_session_folders(base_path, mouse, stage)
session = parse_session_functions.analyse_npz_pre7(mouse, date, stage)

if mouse == 'TAA0000066' and stage == 't3':
    lick_end_idx = 160
    event_idx = np.sort(np.concatenate([session['reward_idx'], session['miss_rew_idx'], session['nongoal_rew_idx']])).astype(int)[:lick_end_idx]
else:
    event_idx = np.sort(np.concatenate([session['reward_idx'], session['miss_rew_idx'], session['nongoal_rew_idx']])).astype(int)
session['event_idx'] = event_idx

#%% Bin YY data 
bins = 20 

# Define patches again - with include_next requirement the last one might be included (XYYX)
ABB_patches, BAA_patches, ABB_patches_idx, BAA_patches_idx = alternation.get_XYY_patches(session, include_next=False)

# Use lm entry and exit idx
lm_entry_idx, lm_exit_idx = parse_session_functions.get_lm_entry_exit(session)

if mouse == 'TAA0000066' and stage == 't3':
    lm_entry_idx = lm_entry_idx[:lick_end_idx]
    lm_exit_idx = lm_exit_idx[:lick_end_idx]

# Find the midpoint between two YY events
BB_midpoint = [np.around(lm_exit_idx[lm[1]] + ((lm_entry_idx[lm[2]] - lm_exit_idx[lm[1]]) / 2)).astype(int) for lm in ABB_patches]
events_BB = {}
for i, lm in enumerate(ABB_patches):
    B1 = lm[1]
    B2 = lm[2]
    events_BB[B1] = { "start": lm_entry_idx[B1], "reward": event_idx[B1], "end": lm_exit_idx[B1]}
    events_BB[B2] = { "start": lm_entry_idx[B2], "reward": event_idx[B2], "end": lm_exit_idx[B2]}

AA_midpoint = [np.around(lm_exit_idx[lm[1]] + ((lm_entry_idx[lm[2]] - lm_exit_idx[lm[1]]) / 2)).astype(int) for lm in BAA_patches]
events_AA = {}
for i, lm in enumerate(BAA_patches):
    A1 = lm[1]
    A2 = lm[2]
    events_AA[A1] = {"start": lm_entry_idx[A1], "reward": event_idx[A1], "end": lm_exit_idx[A1]}
    events_AA[A2] = {"start": lm_entry_idx[A2], "reward": event_idx[A2], "end": lm_exit_idx[A2]}

# Temporal binning within XYY patch
binned_BB_phase_activity = alternation.get_reward_aligned_temporal_phase_binning_per_lm(neurons, dF, ABB_patches, events_BB, bins, condition='BB', plot=True)
binned_AA_phase_activity = alternation.get_reward_aligned_temporal_phase_binning_per_lm(neurons, dF, BAA_patches, events_AA, bins, condition='AA', plot=True)

# Get the difference between two YYs 
BB_diff = {}
for cell in neurons:
    BB_diff[cell] = binned_BB_phase_activity['temporal_ABB_firing'][cell][:, bins:] - binned_BB_phase_activity['temporal_ABB_firing'][cell][:, :bins]
AA_diff = {}
for cell in neurons:
    AA_diff[cell] = binned_AA_phase_activity['temporal_ABB_firing'][cell][:, bins:] - binned_AA_phase_activity['temporal_ABB_firing'][cell][:, :bins]

# Define save path
t = parse_session_functions.extract_int(session['stage'])
save_dir = os.path.join(base_path, mouse, 't3_t4', 'analysis', f't{t}_linear_regression_YY_diff_rew_aligned_XYrepeats_cpa')
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# Cluster-based permutation analysis (CPA) 
BB_diff_regression_results_cpa = alternation.fit_linear_regression_XYlen_cpa(neurons, BB_diff, session, condition='AB', 
                                                                            shuffle=True, nreps=10000, cluster_thres=0.1, plot=True, 
                                                                            sort_heatmap=True, save_dir=save_dir, save_plot=True, 
                                                                            plot_dir=save_dir)
AA_diff_regression_results_cpa = alternation.fit_linear_regression_XYlen_cpa(neurons, AA_diff, session, condition='BA', 
                                                                            shuffle=True, nreps=10000, cluster_thres=0.1, plot=True, 
                                                                            sort_heatmap=True, save_dir=save_dir, save_plot=True, 
                                                                            plot_dir=save_dir)


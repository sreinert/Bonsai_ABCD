import matplotlib.pyplot as plt
import numpy as np
import palettes 
import os, sys
import scipy.stats as stats
from scipy.stats import friedmanchisquare, wilcoxon, norm, kruskal, mannwhitneyu
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import neural_analysis_helpers

parse_session_functions = None
def set_parse_session_functions(psf):
    global parse_session_functions
    parse_session_functions = psf

def get_XYY_patches(session, include_next=True, precede_XY=False):
    '''
    Find ABB and BAA patches
    
    arguments:
    - include_next: whether the landmark after the XYY patch should be included i.e., XYYX (default = True) 
    - precede_XY: whether the two landmarks preceding the patch should be XY (relevant for truly random sequences) (default = False)
    '''     
    event_idx = session['event_idx']
    non_goals = session['non_goals_idx'][session['non_goals_idx'] < len(event_idx)]
    goals = session['goals_idx'][session['goals_idx'] < len(event_idx)]

    # Combine and label: 0 for non-goal, 1 for goal
    combined = np.concatenate([non_goals, goals])
    labels = np.concatenate([np.zeros(len(non_goals), dtype=int), np.ones(len(goals), dtype=int)])

    # Sort by index
    sorted_indices = np.argsort(combined)
    combined_sorted = combined[sorted_indices]
    labels_sorted = labels[sorted_indices]

    # Find ABB and BAA patches
    ABB_patches = []
    BAA_patches = []

    if include_next: # XYYX
        if precede_XY: 
            for i in range(2, len(labels_sorted)-3):
                if labels_sorted[i-2] == 1 and labels_sorted[i-1] == 0 and labels_sorted[i] == 1 and labels_sorted[i+1] == 0 and labels_sorted[i+2] == 0 and labels_sorted[i+3] == 1: # ABBA
                    ABB_patches.append(combined_sorted[i:i+4])
                if labels_sorted[i-2] == 0 and labels_sorted[i-1] == 1 and labels_sorted[i] == 0 and labels_sorted[i+1] == 1 and labels_sorted[i+2] == 1 and labels_sorted[i+3] == 0: # BAAB
                    BAA_patches.append(combined_sorted[i:i+4])
        else:
            for i in range(0, len(labels_sorted)-3):
                if labels_sorted[i] == 1 and labels_sorted[i+1] == 0 and labels_sorted[i+2] == 0 and labels_sorted[i+3] == 1: # ABBA
                    ABB_patches.append(combined_sorted[i:i+4])
                if labels_sorted[i] == 0 and labels_sorted[i+1] == 1 and labels_sorted[i+2] == 1 and labels_sorted[i+3] == 0: # BAAB
                    BAA_patches.append(combined_sorted[i:i+4])
    else: # XYY
        if precede_XY: 
            for i in range(2, len(labels_sorted)-2):
                if labels_sorted[i-2] == 1 and labels_sorted[i-1] == 0 and labels_sorted[i] == 1 and labels_sorted[i+1] == 0 and labels_sorted[i+2] == 0: # ABB
                    ABB_patches.append(combined_sorted[i:i+3])
                if labels_sorted[i-2] == 0 and labels_sorted[i-1] == 1 and labels_sorted[i] == 0 and labels_sorted[i+1] == 1 and labels_sorted[i+2] == 1: # BBA
                    BAA_patches.append(combined_sorted[i:i+3])
        else:
            for i in range(0, len(labels_sorted)-2):
                if labels_sorted[i] == 1 and labels_sorted[i+1] == 0 and labels_sorted[i+2] == 0: # ABB
                    ABB_patches.append(combined_sorted[i:i+3])
                if labels_sorted[i] == 0 and labels_sorted[i+1] == 1 and labels_sorted[i+2] == 1: # BBA
                    BAA_patches.append(combined_sorted[i:i+3])

    # Convert patches to entry/exit indices
    ABB_patches_idx = [(event_idx[patch[0]], event_idx[patch[-1]]) for patch in ABB_patches]
    BAA_patches_idx = [(event_idx[patch[0]], event_idx[patch[-1]]) for patch in BAA_patches]

    return ABB_patches, BAA_patches, ABB_patches_idx, BAA_patches_idx


def get_repeating_XY_patches(session, min_length=2):
    # Find patches of alternating AB/BA 
    non_goals = session['non_goals_idx'][session['non_goals_idx'] < len(session['event_idx'])]
    goals = session['goals_idx'][session['goals_idx'] < len(session['event_idx'])]

    # Combine and label: 0 for non-goal, 1 for goal
    combined = np.concatenate([non_goals, goals])
    labels = np.concatenate([np.zeros(len(non_goals), dtype=int), np.ones(len(goals), dtype=int)])

    # Sort by index
    sorted_indices = np.argsort(combined)
    combined_sorted = combined[sorted_indices]
    labels_sorted = labels[sorted_indices]

    # Find alternating patches
    patches = []
    start = 0
    for i in range(1, len(labels_sorted)):
        if labels_sorted[i] != labels_sorted[i-1]:
            continue
        if labels_sorted[i] == labels_sorted[i-1]:
            # End of an alternating patch
            if i - start > min_length:  
                patches.append(combined_sorted[start:i])
            start = i

    # Check last patch 
    if len(labels_sorted) - start >= 2:
        patches.append(combined_sorted[start:])

    # Filter patches based on A or B start
    BA_patches = [patch for patch in patches if np.isin(patch[0], non_goals)]
    AB_patches = [patch for patch in patches if np.isin(patch[0], goals)]

    # Find the corresponding indices in the data 
    # lm_entry_idx, lm_exit_idx = parse_session_functions.get_lm_entry_exit(session)

    # # Convert patches to entry/exit indices
    # patches_idx = [(lm_entry_idx[patch[0]], lm_exit_idx[patch[-1]]) for patch in patches]
    # BA_patches_idx = [(lm_entry_idx[patch[0]], lm_exit_idx[patch[-1]]) for patch in BA_patches]
    # AB_patches_idx = [(lm_entry_idx[patch[0]], lm_exit_idx[patch[-1]]) for patch in AB_patches]
    patches_idx = None
    AB_patches_idx = None
    BA_patches_idx = None

    return patches, AB_patches, BA_patches, patches_idx, AB_patches_idx, BA_patches_idx

def get_lm_data(session, neurons, patches, AB_patches, BA_patches, time_around,
                lm_entry_idx, lm_exit_idx, dF, condition='next', n_bins=31, funcimg_frame_rate=45,
                zscoring=False, plot=True):
    """
    Extract dF/F aligned to a chosen landmark for each patch, supporting multiple neurons.
    
    Parameters
    ----------
    session : dict
        Session dictionary with 'reward_idx', 'miss_rew_idx', 'nongoal_rew_idx'.
    neurons : dict or list
        Indexed by session; contains neuron IDs.
    patches : list of arrays
        Patch trial indices.
    AB_patches, BA_patches : list of arrays
        Patches categorized by goal or non-goal.
    lm_entry_idx, lm_exit_idx : array-like
        Entry and exit indices for landmarks.
    dF : np.ndarray
        Fluorescence data, shape (n_neurons, n_timepoints).
    condition : str
        Landmark to extract neural data for
        - next: first after patch end
        - last: last in patch 
        - prev: second to last in patch
    n_bins : int
        Total number of bins (split equally pre/post).
    plot : bool
        Whether to plot or not. 
    
    Returns:
        - Dictionary with goal/non-goal windows and patches_by_length
        - Matplotlib axes for goal and non-goal plots
    """
    event_idx = np.sort(np.concatenate([session['reward_idx'], session['miss_rew_idx'], session['nongoal_rew_idx']])).astype(int)

    neurons = np.atleast_1d(neurons)
    n_neurons = len(neurons)
    
    # Slice dF for the neurons we are analyzing
    dF_sel = dF[neurons, :]
    
    # Flatten patches
    AB_patches_flat = np.unique(np.concatenate([np.ravel(p) for p in AB_patches]))
    BA_patches_flat = np.unique(np.concatenate([np.ravel(p) for p in BA_patches]))
    
    AB_patches_by_length = {}
    BA_patches_by_length = {}

    # Labels
    if condition == 'next':
        label_goal, label_non_goal = 'B test', 'A test'
    elif condition == 'last':
        label_goal, label_non_goal = 'B', 'A'
    elif condition == 'prev':
        label_goal, label_non_goal = 'A', 'B'
    else:
        raise ValueError("condition must be 'prev', 'last', or 'next'")
    
    # Loop over patches
    for p, patch in enumerate(patches):
        patch_len = len(patch)
        
        # Determine landmark index
        if condition == 'next':
            lm = patch[-1] + 1
        elif condition == 'last':
            lm = patch[-1]
        else:  # prev
            lm = patch[-1] - 1
        
        # Skip invalid
        if lm < 0 or lm >= len(lm_entry_idx):
            continue
        
        start, end = lm_entry_idx[lm], lm_exit_idx[lm]
        r = event_idx[lm]
        if end == r:
            continue
        
        # Skip missed goal landmarks
        if np.isin(r, session['miss_rew_idx']): 
            # print('missed')
            continue
         
        if isinstance(time_around, (int, float)):
            start_time = -time_around
            end_time = time_around
        elif isinstance(time_around, (tuple, list)) and len(time_around) == 2:
            start_time, end_time = time_around
        start_frames = int(np.floor(start_time * funcimg_frame_rate))
        end_frames = int(np.ceil(end_time * funcimg_frame_rate))

        # Get indices for each event
        window = np.arange(start_frames, end_frames)
        window_indices = np.add.outer(r, window).astype(int)

        # Remove last events if close to session end 
        valid_mask = window_indices[-1] < dF.shape[1]
        valid_window_indices = window_indices[valid_mask]
        
        # Create binned array (n_neurons x n_bins)
        binned = np.full((n_neurons, len(window)), np.nan)
        binned[:,:] = dF_sel[:, np.squeeze(valid_window_indices)]
        
        # z-score data 
        if zscoring:
            binned = stats.zscore(np.array(binned), axis=1)

        # Append to goal or non-goal dictionary
        if patch[-1] in AB_patches_flat:
            AB_patches_by_length.setdefault(patch_len, []).append(binned)
        elif patch[-1] in BA_patches_flat:
            BA_patches_by_length.setdefault(patch_len, []).append(binned)
    
    # Stack lists into arrays: shape (n_patches, n_neurons, n_bins)
    for length in AB_patches_by_length:
        AB_patches_by_length[length] = np.stack(AB_patches_by_length[length], axis=0)
    for length in BA_patches_by_length:
        BA_patches_by_length[length] = np.stack(BA_patches_by_length[length], axis=0)
    
    if plot:
        # --- Plot Goal Patches ---
        fig_goal, goal_ax = plt.subplots(1, len(AB_patches_by_length), figsize=(3*len(AB_patches_by_length), 3), sharey=True, squeeze=False)
        for i, (length, arr) in enumerate(sorted(AB_patches_by_length.items())):
            if n_neurons == 1:
                mean_data = np.nanmean(np.squeeze(arr, axis=1), axis=0)
            else:
                mean_data = np.nanmean(arr, axis=(0,1))
            goal_ax[0,i].plot(mean_data, color='blue', label=label_goal)
            goal_ax[0,i].axvline(x=n_bins // 2, color='gray', linestyle='--')
            goal_ax[0,i].set_title(f'Goal: Patch length {length}')
            goal_ax[0,i].set_xlabel('Normalized time')
            if i == 0:
                goal_ax[0,i].set_ylabel('dF/F')
        plt.tight_layout()
    
        # --- Plot Non-Goal Patches ---
        fig_non_goal, non_goal_ax = plt.subplots(1, len(BA_patches_by_length), figsize=(3*len(BA_patches_by_length), 3), sharey=True, squeeze=False)
        for i, (length, arr) in enumerate(sorted(BA_patches_by_length.items())):
            if n_neurons == 1:
                mean_data = np.nanmean(np.squeeze(arr, axis=1), axis=0)
            else:
                mean_data = np.nanmean(arr, axis=(0,1))
            non_goal_ax[0,i].plot(mean_data, color='orange', label=label_non_goal)
            non_goal_ax[0,i].axvline(x=n_bins // 2, color='gray', linestyle='--')
            non_goal_ax[0,i].set_title(f'Non-Goal: Patch length {length}')
            non_goal_ax[0,i].set_xlabel('Normalized time')
            if i == 0:
                non_goal_ax[0,i].set_ylabel('dF/F')
        plt.tight_layout()
    
        return {
            "AB_patches_by_length": AB_patches_by_length,
            "BA_patches_by_length": BA_patches_by_length
        }, goal_ax, non_goal_ax
    
    else:
        return {
            "AB_patches_by_length": AB_patches_by_length,
            "BA_patches_by_length": BA_patches_by_length
        }, None, None


def compare_lms_in_AB_patches(neurons, session, patches, AB_patches, BA_patches, dF, time_around, n_bins=10, zscoring=False, plot=True, plot_neurons=None):
    
    lm_entry_idx, lm_exit_idx = parse_session_functions.get_lm_entry_exit(session)

    next_lm_data, _, _ = get_lm_data(session, neurons, patches, AB_patches, BA_patches, time_around,
                lm_entry_idx, lm_exit_idx, dF, condition='next', n_bins=n_bins, zscoring=zscoring, plot=False)

    lm_data, _, _ = get_lm_data(session, neurons, patches, AB_patches, BA_patches, time_around,
                    lm_entry_idx, lm_exit_idx, dF, condition='last', n_bins=n_bins, zscoring=zscoring, plot=False)

    prev_lm_data, _, _ = get_lm_data(session, neurons, patches, AB_patches, BA_patches, time_around,
                    lm_entry_idx, lm_exit_idx, dF, condition='prev', n_bins=n_bins, zscoring=zscoring, plot=False)

    # Extract goal/non-goal by length for all three conditions
    goal_data = {'prev': prev_lm_data['AB_patches_by_length'],
                'last': lm_data['AB_patches_by_length'],
                'next': next_lm_data['AB_patches_by_length']}

    non_goal_data = {'prev': prev_lm_data['BA_patches_by_length'],
                    'last': lm_data['BA_patches_by_length'],
                    'next': next_lm_data['BA_patches_by_length']}

    line_styles = {'prev': ':', 'last': '-', 'next': '--'}
    colors = {'goal': 'blue', 'non_goal': 'orange'}

    if plot:
        # --- Plot Goal Patches ---
        lengths = sorted(goal_data['last'].keys())
        fig, goal_ax = plt.subplots(1, len(lengths), figsize=(4*len(lengths), 3), sharey=True)
        if len(lengths) == 1:
            goal_ax = [goal_ax]

        for i, length in enumerate(lengths):
            ax = goal_ax[i]
            for id, cond in zip([' A', ' B', ' B-test'], ['prev', 'last', 'next']):
                if length not in goal_data[cond]:
                    continue 
                patches_array = goal_data[cond][length]  # shape: n_patches x n_bins
                if patches_array.shape[1] == 1:
                    mean_data = np.nanmean(np.squeeze(patches_array, axis=1), axis=0)
                else:
                    mean_data = np.nanmean(patches_array, axis=(0,1))
                ax.plot(mean_data, color=colors['goal'], linestyle=line_styles[cond], label=[cond+id])
                ax.axvline(x=mean_data.shape[-1] // 2, color='gray', linestyle='--')
            ax.set_title(f'Goal: Patch length {length}')
            ax.set_xlabel('Normalized time')
            if i == 0:
                ax.set_ylabel('dF/F')
            ax.legend()
        plt.tight_layout()

        # --- Plot Non-Goal Patches ---
        lengths = sorted(non_goal_data['last'].keys())
        fig, non_goal_ax = plt.subplots(1, len(lengths), figsize=(4*len(lengths), 3), sharey=True)
        if len(lengths) == 1:
            non_goal_ax = [non_goal_ax]

        for i, length in enumerate(lengths):
            ax = non_goal_ax[i]
            for id, cond in zip([' B', ' A', ' A-test'], ['prev', 'last', 'next']):
                if length not in non_goal_data[cond]:
                    continue 
                patches_array = non_goal_data[cond][length]
                if patches_array.shape[1] == 1:
                    mean_data = np.nanmean(np.squeeze(patches_array, axis=1), axis=0)
                else:
                    mean_data = np.nanmean(patches_array, axis=(0,1))
                ax.plot(mean_data, color=colors['non_goal'], linestyle=line_styles[cond], label=[cond+id])
                ax.axvline(x=mean_data.shape[-1] // 2, color='gray', linestyle='--')
            ax.set_title(f'Non-Goal: Patch length {length}')
            ax.set_xlabel('Normalized time')
            if i == 0:
                ax.set_ylabel('dF/F')
            ax.legend()
        plt.tight_layout()

    if plot_neurons is not None:
        # --- Plot Goal Patches ---
        lengths = sorted(goal_data['last'].keys())
        for n, neuron in enumerate(neurons[:plot_neurons]):
            _, ax = plt.subplots(1, len(lengths), figsize=(4*len(lengths), 3), sharey=False)
            ax = ax.ravel()
            for i, length in enumerate(lengths):
                for id, cond in zip([' A', ' B', ' B-test'], ['prev', 'last', 'next']):
                    if length not in goal_data[cond]:
                        continue 
                    patches_array = goal_data[cond][length]  # shape: n_patches x n_bins
                    mean_data = np.nanmean(patches_array[:,n,:], axis=0)
                    ax[i].plot(mean_data, color=colors['goal'], linestyle=line_styles[cond], label=[cond+id])
                    ax[i].axvline(x=mean_data.shape[-1] // 2, color='gray', linestyle='--')
                ax[i].set_title(f'Goal: Patch length {length}')
                ax[i].set_xlabel('Normalized time')
                if i == 0:
                    ax[i].set_ylabel('dF/F')
                ax[i].legend()
            plt.suptitle(f'Neuron {n}: {neuron}')
            plt.tight_layout()

        # --- Plot Non-Goal Patches ---
        lengths = sorted(non_goal_data['last'].keys())
        for n, neuron in enumerate(neurons[:plot_neurons]):
            _, ax = plt.subplots(1, len(lengths), figsize=(4*len(lengths), 3), sharey=False)
            ax = ax.ravel()
            for i, length in enumerate(lengths):
                for id, cond in zip([' B', ' A', ' A-test'], ['prev', 'last', 'next']):
                    if length not in non_goal_data[cond]:
                        continue 
                    patches_array = non_goal_data[cond][length]  # shape: n_patches x n_bins
                    mean_data = np.nanmean(patches_array[:,n,:], axis=0)
                    ax[i].plot(mean_data, color=colors['non_goal'], linestyle=line_styles[cond], label=[cond+id])
                    ax[i].axvline(x=mean_data.shape[-1] // 2, color='gray', linestyle='--')
                ax[i].set_title(f'Non-Goal: Patch length {length}')
                ax[i].set_xlabel('Normalized time')
                if i == 0:
                    ax[i].set_ylabel('dF/F')
                ax[i].legend()
            plt.suptitle(f'Neuron {n}: {neuron}')
            plt.tight_layout()

    return next_lm_data, lm_data, prev_lm_data


def compute_lm_mean(next_lm_data, lm_data, prev_lm_data):
    # Collapse all bins into per-trial means for each condition and goal type

    all_data = {
        'next': next_lm_data,
        'last': lm_data,
        'prev': prev_lm_data
    }

    goal_means = {}
    non_goal_means = {}

    for cond, data in all_data.items():
        goal_means[cond] = {}
        non_goal_means[cond] = {}

        # Goal patches
        for length, arr in data['AB_patches_by_length'].items():
            # arr shape: (n_patches, n_neurons, n_bins)
            goal_means[cond][length] = np.nanmean(arr, axis=(0,2))  # → (n_neurons,)
        # Non-goal patches
        for length, arr in data['BA_patches_by_length'].items():
            non_goal_means[cond][length] = np.nanmean(arr, axis=(0,2))

    return goal_means, non_goal_means


def compute_neuron_lm_mean(next_lm_data, lm_data, prev_lm_data):
    all_data = {
        'next': next_lm_data,
        'last': lm_data,
        'prev': prev_lm_data
    }

    neuron_goal_means = {}
    neuron_non_goal_means = {}

    for cond, data in all_data.items():

        # collect all length arrays
        goal_arrays = list(data['AB_patches_by_length'].values())
        non_goal_arrays = list(data['BA_patches_by_length'].values())

        # result containers
        goal_list = []
        non_goal_list = []

        # --- GOAL ---
        for arr in goal_arrays:
            # arr shape: (n_patches, n_neurons, n_bins)
            # mean over bins → (n_patches, n_neurons)
            patch_means = np.nanmean(arr, axis=2)
            goal_list.append(patch_means)

        # concatenate across lengths
        neuron_goal_means[cond] = np.concatenate(goal_list, axis=0) if goal_list else None

        # --- NON-GOAL ---
        for arr in non_goal_arrays:
            patch_means = np.nanmean(arr, axis=2)
            non_goal_list.append(patch_means)

        neuron_non_goal_means[cond] = np.concatenate(non_goal_list, axis=0) if non_goal_list else None

    return neuron_goal_means, neuron_non_goal_means


def get_responsive_neurons(psth, event='reward', time_around=1, funcimg_frame_rate=45, plot_neurons=True):
    """
    Mann-Whitney U test comparing firing before and after event.
    
    Parameters
    ----------
    psth : array, shape (neurons, trials, timebins)
    event : str
        Event name
    time_around : int/float or tuple
        If int/float: window in seconds, symmetric around 0 (e.g. 1 → -1 to +1).
        If tuple: (start, end) in seconds (e.g. (-1, 2)).
    funcimg_frame_rate : int
        Imaging frame rate (Hz).
    plot_neurons : bool
        Whether to plot significant neurons.
    """
    # TODO: bootstrapping / permutation test instead? 

    # Handle time window
    if isinstance(time_around, (int, float)):
        start_time = -time_around
        end_time = time_around
    elif isinstance(time_around, (tuple, list)) and len(time_around) == 2:
        start_time, end_time = time_around
    else:
        raise ValueError("time_around must be a number or a tuple/list (start, end)")

    start_frames = int(np.floor(start_time * funcimg_frame_rate))
    end_frames = int(np.ceil(end_time * funcimg_frame_rate))
    num_neurons = psth.shape[0]

    event_frame = -start_frames  # event aligned at t=0
    before_idx = slice(0, event_frame)          # frames before event
    after_idx  = slice(event_frame, event_frame + end_frames)  # frames after event

    # Average across timebins for each trial
    before_event_firing = np.mean(psth[:, :, before_idx], axis=2)
    after_event_firing  = np.mean(psth[:, :, after_idx], axis=2)
    # print(before_event_firing.shape, after_event_firing.shape)

    # Perform the test using all trials for each neuron
    wilcoxon_stat = np.zeros((num_neurons, 1))
    wilcoxon_pval = np.zeros((num_neurons, 1))
    for n in range(num_neurons):
        wilcoxon_stat[n], wilcoxon_pval[n] = stats.wilcoxon(before_event_firing[n, :], after_event_firing[n, :]) #, method=stats.PermutationMethod(n_resamples=1000))

    # Criteria to define tuned neurons
    # 1. p-value
    criterion1 = np.where(wilcoxon_pval < 0.01)[0]   

    # 2. peak in the 1s after event > mean + 2*std of the 1s before the event
    average_psth = np.mean(psth, axis=1)
    before_event_avg_firing = average_psth[:, before_idx]
    after_event_avg_firing = average_psth[:, after_idx]
    criterion2_high = np.where(np.max(after_event_avg_firing, axis=1) > (np.mean(before_event_avg_firing, axis=1) + 2 * np.std(before_event_avg_firing, axis=1)))[0]
    criterion2_low = np.where(np.max(after_event_avg_firing, axis=1) < (np.mean(before_event_avg_firing, axis=1)))[0] #- 2 * np.std(before_event_avg_firing, axis=1)))[0]
    
    tuned_neurons = criterion1
    tuned_neurons_high = np.intersect1d(criterion1, criterion2_high)
    tuned_neurons_low = np.setdiff1d(tuned_neurons, tuned_neurons_high)
    # tuned_neurons = np.intersect1d(criterion1, criterion2)
    print(f'{len(tuned_neurons)} neurons are tuned to {event}.')

    # Plot firing for a few significant neurons
    if plot_neurons:
        for n in tuned_neurons[0:20]:
            fig, ax = plt.subplots(1, 1, figsize=(2,2), sharey=True)
            ax.plot(average_psth[n, :])      
            event_frame = -start_frames  
            # ax.plot(average_psth[n, :])      
            ax.axvspan(event_frame, event_frame + end_frames, color='gray', alpha=0.5)
            ax.set_xlabel('Time')
            ax.set_xticks([event_frame + start_frames, event_frame, event_frame + end_frames])
            ax.set_xticklabels([start_time, 0, end_time])
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_ylabel('DF/F')
            ax.set_title(f'Neuron {n}, p-value {wilcoxon_pval[n]}')

    return tuned_neurons, tuned_neurons_high, tuned_neurons_low, wilcoxon_stat, wilcoxon_pval


def temporal_bin_ABB_firing(ABB_patches_idx, cell, dF, bins=90, plot=True):
    binned_phase_firing = np.zeros((len(ABB_patches_idx), bins))

    for i in range(len(ABB_patches_idx)):
        phase_frames = np.arange(ABB_patches_idx[i][0], ABB_patches_idx[i][1])
        bin_edges = np.linspace(ABB_patches_idx[i][0], ABB_patches_idx[i][1], bins+1)
        phase_firing = dF[cell, phase_frames]
        
        bin_ix = np.digitize(phase_frames, bin_edges)
        for j in range(bins):
            binned_phase_firing[i,j] = np.mean(phase_firing[bin_ix == j+1])

    if plot:
        fig = plt.figure(figsize=(3,3))
        ax1 = fig.add_subplot(111)
        cax = ax1.imshow(binned_phase_firing, aspect='auto', cmap='viridis', interpolation='none')
        ax1.set_title(f'Cell {cell} - Binned Firing Rates')
        plt.colorbar(cax, ax=ax1, label='dF/F')
        plt.tight_layout()

    return binned_phase_firing


def spatial_bin_ABB_firing(ABB_patches_idx, cell, dF, session, bins=90, plot=True):
    positions = session['position']
    binned_phase_firing = np.zeros((len(ABB_patches_idx), bins))

    for i in range(len(ABB_patches_idx)):
        phase_frames = np.arange(ABB_patches_idx[i][0], ABB_patches_idx[i][1])
        phase_positions = positions[phase_frames]
        bin_edges = np.linspace(phase_positions.min(), phase_positions.max(), bins+1)
        phase_firing = dF[cell, phase_frames]

        bin_ix = np.digitize(phase_positions, bin_edges)
        for j in range(bins):
            binned_phase_firing[i,j] = np.mean(phase_firing[bin_ix == j+1])

    if plot:
        fig = plt.figure(figsize=(3,3))
        ax1 = fig.add_subplot(111)
        cax = ax1.imshow(binned_phase_firing, aspect='auto', cmap='viridis', interpolation='none')
        ax1.set_title(f'Cell {cell} - Binned Firing Rates')
        plt.colorbar(cax, ax=ax1, label='dF/F')
        plt.tight_layout()

    return binned_phase_firing


def get_spatial_and_temporal_ABB_binning(ABB_patches_idx, neurons, dF, session, bins=90):
    '''Binning of neural activity inside a XYY patch from the beginning to the end of the patch.'''
    temporal_ABB_firing = {}
    spatial_ABB_firing = {}
    for cell in neurons:
        temporal_ABB_firing[cell] = temporal_bin_ABB_firing(ABB_patches_idx, cell, dF, bins, plot=False)
        spatial_ABB_firing[cell] = spatial_bin_ABB_firing(ABB_patches_idx, cell, dF, session, bins, plot=False)

    # Get the mean across patches for all neurons
    avg_temporal_ABB_firing = np.empty((len(neurons), bins))
    avg_spatial_ABB_firing = np.empty((len(neurons), bins))
    for n, cell in enumerate(neurons):
        avg_temporal_ABB_firing[n] = np.nanmean(temporal_ABB_firing[cell], axis=0)
        avg_spatial_ABB_firing[n] = np.nanmean(spatial_ABB_firing[cell], axis=0)

    # Z-score per neuron
    zscored_avg_temporal_ABB_firing = stats.zscore(avg_temporal_ABB_firing, axis=1)
    zscored_avg_spatial_ABB_firing = stats.zscore(avg_spatial_ABB_firing, axis=1)

    # Sort according to max firing 
    peak_bins = np.argmax(zscored_avg_temporal_ABB_firing, axis=1)
    sort_order = np.argsort(peak_bins)
    zscored_sorted_temporal = zscored_avg_temporal_ABB_firing[sort_order]

    peak_bins = np.argmax(zscored_avg_spatial_ABB_firing, axis=1)
    sort_order = np.argsort(peak_bins)
    zscored_sorted_spatial = zscored_avg_spatial_ABB_firing[sort_order]

    # Plot
    fig = plt.figure(figsize=(6,3))
    ax1 = fig.add_subplot(121)
    cax1 = ax1.imshow(zscored_sorted_temporal, aspect='auto', cmap='viridis', interpolation='none')
    ax1.set_title(f'Binned Firing Rates (Temporal)')
    ax1.set_xlabel('Time bins')
    cb1 = fig.colorbar(cax1, ax=ax1, label='dF/F')  

    ax2 = fig.add_subplot(122)
    cax2 = ax2.imshow(zscored_sorted_spatial, aspect='auto', cmap='viridis', interpolation='none')
    ax2.set_title(f'Binned Firing Rates (Spatial)')
    ax2.set_xlabel('Position bins')
    cb2 = fig.colorbar(cax2, ax=ax2, label='dF/F')  

    for ax in [ax1, ax2]:
        ax.set_yticks([0, len(neurons)-1])
        ax.set_yticklabels([0, len(neurons)])
        ax.set_xticks([0, bins-1])
        ax.set_xticklabels([0, bins])
        ax.set_ylabel('Neurons', labelpad=-5)
        
    plt.tight_layout()

    # Collect all data into a dict
    binned_ABB_firing_rates = {}
    binned_ABB_firing_rates['temporal_ABB_firing'] = temporal_ABB_firing
    binned_ABB_firing_rates['spatial_ABB_firing'] = spatial_ABB_firing
    binned_ABB_firing_rates['avg_temporal_ABB_firing'] = avg_temporal_ABB_firing
    binned_ABB_firing_rates['avg_spatial_ABB_firing'] = avg_spatial_ABB_firing
    binned_ABB_firing_rates['zscored_sorted_temporal'] = zscored_sorted_temporal
    binned_ABB_firing_rates['zscored_sorted_spatial'] = zscored_sorted_spatial

    return binned_ABB_firing_rates

def temporal_bin_lm_firing(lm, cell, dF, bins=90):
    '''Temporal binning within two specific events e.g. between consecutive landmarks.'''
    binned_phase_firing = np.zeros(bins)

    phase_frames = np.arange(lm[0], lm[1])
    bin_edges = np.linspace(lm[0], lm[1], bins+1)
    phase_firing = dF[cell, phase_frames]

    bin_ix = np.digitize(phase_frames, bin_edges)
    for j in range(bins):
        binned_phase_firing[j] = np.mean(phase_firing[bin_ix == j+1])

    return binned_phase_firing

def get_temporal_phase_binning_per_lm(neurons, dF, XYY_patches, event_idx, bins=30, condition='ABB', plot=True):
    '''Binning of neural activity inside a XYY patch from the beginning to the end of each landmark in the the patch.'''
    
    # Collect all landmark pair binnings for all patches
    binned_XYY_phase_firing = {cell: [] for cell in neurons}
 
    n_lms = len(XYY_patches[0]) - 1

    for n, cell in enumerate(neurons):
        for patch in XYY_patches:
            if condition == 'BB' or condition == 'AA':
                # Assume that lm entry and exit, and YY midpoint indices are provided
                assert n_lms == 2, 'Each patch should have 3 landmarks - XYY'
                patch_bin_list = [temporal_bin_lm_firing([event_idx[lm][0], event_idx[lm][1]], cell, dF, bins=bins) for lm in patch[1:]]
            else:
                assert n_lms == 3, 'Each patch should have 4 landmarks - XYYX'
                patch_bin_list = [temporal_bin_lm_firing([event_idx[lm], event_idx[lm+1]], cell, dF, bins=bins) for lm in patch[:-1]]

            linear_patch_binned = np.concatenate(patch_bin_list)  # convert list to array and flatten
            binned_XYY_phase_firing[cell].append(linear_patch_binned)

    for cell in neurons:
        binned_XYY_phase_firing[cell] = np.array(binned_XYY_phase_firing[cell])

    # Average across patches
    avg_binned_XYY_phase_firing = np.empty((len(neurons), bins * n_lms))
    for n, cell in enumerate(neurons):
        avg_binned_XYY_phase_firing[n] = np.nanmean(binned_XYY_phase_firing[cell], axis=0)

    # Z-score
    zscored_avg_binned_XYY_phase_firing = stats.zscore(avg_binned_XYY_phase_firing, axis=1)

    # Sort according to max firing 
    peak_bins = np.argmax(zscored_avg_binned_XYY_phase_firing, axis=1)
    sort_order = np.argsort(peak_bins)
    sorted_zscored_avg_binned_XYY = zscored_avg_binned_XYY_phase_firing[sort_order]

    # Plotting
    if plot:
        fig = plt.figure(figsize=(3,3))
        ax1 = fig.add_subplot(111)
        cax1 = ax1.imshow(sorted_zscored_avg_binned_XYY, aspect='auto', cmap='viridis', interpolation='none')
        if n_lms > 2:
            ax1.vlines(x=bins-1, ymin=0, ymax=len(neurons)-1, linestyles='--', colors='white')
            ax1.vlines(x=2*bins-1, ymin=0, ymax=len(neurons)-1, linestyles='--', colors='white')
            ax1.set_xticks([0, bins-1, 2*bins-1, 3*bins-1])
            if condition == 'ABB':
                ax1.set_xticklabels(['A', 'B', 'B', 'A'])
            elif condition == 'BAA':
                ax1.set_xticklabels(['B', 'A', 'A', 'B'])
            else:
                ax1.set_xticklabels(['X', 'Y', 'Y', 'X'])
        else:
            ax1.vlines(x=bins-1, ymin=0, ymax=len(neurons)-1, linestyles='--', colors='white')
            ax1.set_xticks([0, bins-1, 2*bins-1])
            if condition == 'BB':
                ax1.set_xticklabels(['B1 entry', 'BB mid', 'B2 exit'])
            elif condition == 'AA':
                ax1.set_xticklabels(['A1 entry', 'AA mid', 'A2 exit'])
            else:
                ax1.set_xticklabels(['Y1 entry', 'YY mid', 'Y2 exit'])

        cb1 = fig.colorbar(cax1, ax=ax1, label='dF/F')  
        
        ax1.set_xlabel('Time bins')
        ax1.set_yticks([0, len(neurons)-1])
        ax1.set_yticklabels([0, len(neurons)])
        ax1.set_ylabel('Neurons', labelpad=-5)

    # Collect all data into a dict - maintain similar structure to get_spatial_and_temporal_ABB_binning
    binned_XYY_phase_activity = {}
    binned_XYY_phase_activity['temporal_ABB_firing'] = binned_XYY_phase_firing
    binned_XYY_phase_activity['avg_temporal_ABB_firing'] = avg_binned_XYY_phase_firing
    binned_XYY_phase_activity['zscored_sorted_temporal'] = sorted_zscored_avg_binned_XYY

    return binned_XYY_phase_activity


def get_binning_by_XY_patch_length(neurons, session, dF, condition='AB', bins=90, plot=True, last_lm=False):
    '''
    Bin neural activity according to the length of the patch. 
    If last_lm is False, the neural activity is binned across the entire patch. Otherwise, binning 
    is done only for the last Y in the patch. 
    '''
    # Find patches of alternating AB/BA 
    _, AB_patches, BA_patches, _, _, _ = get_repeating_XY_patches(session, min_length=0)
    
    if condition == 'AB':
        patches = AB_patches
    elif condition == 'BA':
        patches = BA_patches
    
    XY_repeats = np.array([len(patch) / 2 for patch in patches]).astype(int)

    # Bin from the beginning to the end of the patch
    lm_entry_idx, lm_exit_idx = parse_session_functions.get_lm_entry_exit(session)
    entry_exit_events = np.sort(np.concatenate([lm_entry_idx, lm_exit_idx]))
    
    binned_XY_patch_activity = {cell: [] for cell in neurons}
    avg_XY_patch_length_activity = {cell: {} for cell in neurons}

    print(patches)
    for cell in neurons: 
        # Bin activity across each patch
        for i, patch in enumerate(patches):
            if last_lm:
                events = [entry_exit_events[2*patch[-1]], entry_exit_events[2*patch[-1]+1]]
            else:
                events = [entry_exit_events[2*patch[0]], entry_exit_events[2*patch[-1]+1]]
            patch_binned_activity = temporal_bin_lm_firing(events, cell, dF, bins=bins)
            binned_XY_patch_activity[cell].append(patch_binned_activity)

        binned_XY_patch_activity[cell] = np.array(binned_XY_patch_activity[cell]) # (n_patches x n_bins)

        # Average by patch length 
        for length in np.unique(XY_repeats):
            avg_XY_patch_length_activity[cell][length] = np.mean(binned_XY_patch_activity[cell][XY_repeats == length, :], axis=0)

        if plot:
            _, ax = plt.subplots(1, len(np.unique(XY_repeats)), sharey=True, sharex=True, figsize=(12,3))
            ax = ax.ravel()
            for i, length in enumerate(np.unique(XY_repeats)):
                ax[i].plot(avg_XY_patch_length_activity[cell][length])
                ax[i].set_xticks([0, avg_XY_patch_length_activity[cell][length].shape[0]])
                if last_lm:
                    ax[i].set_xticklabels([f'last {condition[1]}\nstart', f'last {condition[1]}\nend'])
                else:
                    ax[i].set_xticklabels([f'first {condition[0]}', f'last {condition[1]}'])
                ax[i].set_title(f'#{condition} = {length}')
            ax[0].set_ylabel('dF/F')
            plt.suptitle(f'Neuron {cell}')
            plt.tight_layout()

    return binned_XY_patch_activity, avg_XY_patch_length_activity


def find_cells_with_ABB_peaks(neurons, binned_activity, condition='temporal', plot=True):
    # ABB_patch_length_cm = session4['position'][ABB_patches_idx[0][1]] - session4['position'][ABB_patches_idx[0][0]]
    # bin_size_cm = ABB_patch_length_cm / nbins
    # place_field_size_cm = 10    # ~ half a lm 
    # place_field_bins = np.ceil(place_field_size_cm / bin_size_cm)

    peak_cells = []
    for n, cell in enumerate(neurons):
        if condition == 'temporal':
            smoothed = gaussian_filter1d(binned_activity['avg_temporal_ABB_firing'][n], sigma=2, mode='nearest')
        elif condition == 'spatial':
            smoothed = gaussian_filter1d(binned_activity['avg_spatial_ABB_firing'][n], sigma=2, mode='nearest')
        data = smoothed
        
        baseline = np.median(data)
        height = 1.2 * baseline

        peaks, props = find_peaks(data, distance=20, height=height)
        edge_peaks = []
        if data[0] > data[1] and (height is None or data[0] >= height):
            edge_peaks.append(0)
        if data[-1] > data[-2] and (height is None or data[-1] >= height):
            edge_peaks.append(len(data) - 1)
        all_peaks = np.sort(np.concatenate([peaks, edge_peaks])).astype(int)

        if len(peaks) > 0:
            peak_cells.append(cell)

        if plot:
            plt.figure(figsize=(4,3))
            plt.plot(data)
            plt.hlines(baseline, xmin=0, xmax=len(data), colors='k', linestyles='--')
            plt.hlines(height, xmin=0, xmax=len(data), colors='g', linestyles='--')
            plt.scatter(all_peaks, data[all_peaks], color='r')
            plt.title(f'Neuron {cell}')

    return peak_cells

def get_YY_diff_cells(neurons, binned_YY_phase_activity, condition='BB', plot=True):
    '''Find neurons with significantly different responses during Y1 vs Y2'''
    wilcoxon_stat = np.zeros((len(neurons), 1))
    wilcoxon_pval = np.zeros((len(neurons), 1))
    
    bins = int(binned_YY_phase_activity['temporal_ABB_firing'][neurons[0]].shape[1] / 2)

    for n, cell in enumerate(neurons):
        mean_Y1 = np.mean(binned_YY_phase_activity['temporal_ABB_firing'][cell][:, :bins], axis=1)
        mean_Y2 = np.mean(binned_YY_phase_activity['temporal_ABB_firing'][cell][:, bins:], axis=1)
        
        wilcoxon_stat[n], wilcoxon_pval[n] = stats.wilcoxon(mean_Y1, mean_Y2)

    YY_diff_cells = np.array(neurons)[np.where(wilcoxon_pval < 0.01)[0]]

    if plot:
        for cell in neurons:
            if np.isin(cell, YY_diff_cells):
                data = binned_YY_phase_activity['temporal_ABB_firing'][cell]

                # Calculate difference between Y1 and Y2
                YY_diff = data[:, bins:] - data[:, :bins] # Y2-Y1
                
                # Plot
                fig = plt.figure(figsize=(6,4))
                gs = plt.GridSpec(1, 2, width_ratios=[2, 1])  
                ax1 = fig.add_subplot(gs[0,0])
                ax2 = fig.add_subplot(gs[0,1], sharey=ax1)
                
                vmin1, vmax1 = np.nanmin(data), np.nanmax(data)
                nbins = data.shape[1]
                n_trials = data.shape[0]

                cax1 = ax1.imshow(data, aspect='auto', cmap='viridis', interpolation='none')
                ax1.set_title(f'Binned Firing Rates (Temporal)')
                ax1.set_xlabel('Time bins')
                cb1 = fig.colorbar(cax1, ax=ax1, label='dF/F', ticks=[vmin1, vmax1])
                cb1.ax.set_yticklabels([f"{vmin1:.1f}", f"{vmax1:.1f}"]) 
                cb1.ax.yaxis.labelpad = -10
                ax1.set_yticks([0, n_trials-1])
                ax1.set_yticklabels([0, n_trials])
                ax1.vlines(x=bins-1, ymin=0, ymax=len(neurons)-1, linestyles='--', colors='white')
                ax1.set_xticks([0, bins-1, 2*bins-1])
                if condition == 'BB':
                    ax1.set_xticklabels(['B1 entry', 'BB mid', 'B2 exit'])
                elif condition == 'AA':
                    ax1.set_xticklabels(['A1 entry', 'AA mid', 'A2 exit'])
                else:
                    ax1.set_xticklabels(['Y1 entry', 'YY mid', 'Y2 exit'])
                ax1.set_ylabel('Trial (patch)', labelpad=-5)
                ax1.vlines(x=bins-1, ymin=0, ymax=n_trials-1, linestyles='--', colors='white')
                
                vmax = np.max(np.abs(YY_diff))
                vmin = -vmax
                cax2 = ax2.imshow(YY_diff, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
                cb2 = fig.colorbar(cax2, ax=ax2, label='YY diff dF/F', ticks=[vmin, vmax])
                cb2.ax.set_yticklabels([f"{vmin:.1f}", f"{vmax:.1f}"])
                cb2.ax.yaxis.labelpad = -10
                ax2.set_yticks([0, n_trials-1])
                ax2.set_yticklabels([0, n_trials])
                ax2.set_xticks([0, bins-1])
                ax2.set_xticklabels([0, bins])
                ax2.set_xlabel('Time bins')
                ax2.set_title(f'Y2-Y1')
                
                plt.suptitle(f'Neuron {cell}') 

    return YY_diff_cells


def get_Y_psth(neurons, session, dF, events, condition='AB', lm='last_Y', time_around=0.5, plot=True):
    '''
    Get the PSTH around a Y event according to patch type. 
    If last_Y, this is the last Y inside an XY patch. Otherwise, if next_Y, this is the first Y following
    an alternation violation i.e., the next Y after the end of a patch.
    '''
    assert lm in ('last_Y', 'next_Y'), "Valid values for lm are 'last_Y and 'next_Y'."

    # Handle time window input
    if isinstance(time_around, (int, float)):
        start_time = -time_around
        end_time = time_around
    elif isinstance(time_around, (tuple, list)) and len(time_around) == 2:
        start_time, end_time = time_around
    else:
        raise ValueError("time_around must be a single number or a tuple/list of (start, end)")

    # Find patches of alternating AB/BA 
    _, AB_patches, BA_patches, _, _, _ = get_repeating_XY_patches(session, min_length=0)

    if condition == 'AB':
        patches = AB_patches
    elif condition == 'BA':
        patches = BA_patches

    XY_repeats = np.array([len(patch) / 2 for patch in patches]).astype(int)

    # PSTH for last/next Y in each patch
    if lm == 'last_Y':
        event_idx = [events[patch[-1]] for patch in patches]
    elif lm == 'next_Y':
        event_idx = [events[patch[-1]+1] for patch in patches if patch[-1]+1 < len(events)]

    psth_patch, _ = neural_analysis_helpers.get_psth(dF, neurons, event_idx, time_around=time_around) # (n_neurons x n_patches x n_timebins)

    # Average PSTH by patch length 
    avg_psth_patch = {}
    for length in np.unique(XY_repeats):
        patch_mask = XY_repeats == length
        avg_psth_patch[length] = np.mean(psth_patch[:, patch_mask[:psth_patch.shape[1]], :], axis=1) # (n_neurons x n_timebins)

    # Plotting
    if plot:
        palette = palettes.met_brew('Johnson', n=len(np.unique(XY_repeats)), brew_type="continuous")
        num_timebins = psth_patch.shape[-1]

        for n, cell in enumerate(neurons):
            _, ax = plt.subplots(1, 1, figsize=(3,3))
            
            for i, length in enumerate(np.unique(XY_repeats).astype(int)):
                patch_mask = XY_repeats == length
                ax.plot(avg_psth_patch[length][n], color=palette[i], linewidth=2, label=f'{length} ({len(np.where(XY_repeats == length)[0])})')
                ax.fill_between(np.arange(num_timebins),
                                avg_psth_patch[length][n] - stats.sem(psth_patch[n, patch_mask[:psth_patch.shape[1]], :], axis=0),
                                avg_psth_patch[length][n] + stats.sem(psth_patch[n, patch_mask[:psth_patch.shape[1]], :], axis=0),
                                color=palette[i], alpha=0.3) # standard error per patch lengthf
            
            ax.set_ylabel('dF/F')
            ax.set_xlabel('Time (s)')
            zero_bin = int(round(-start_time / (end_time - start_time) * num_timebins))
            ax.set_xticks([0, zero_bin, num_timebins - 1])
            ax.set_xticklabels([round(start_time, 2), 0, round(end_time, 2)])
            ax.axvspan(zero_bin, num_timebins, color='gray', alpha=0.5)
            ax.spines[['right', 'top']].set_visible(False)

            plt.figlegend(loc='upper right', title=f'# {condition}s', bbox_to_anchor=(1.25, 0.8), borderaxespad=0)
            if lm == 'last_Y':
                plt.suptitle(f'last {condition[-1]}: neuron {cell}')
            if lm == 'next_Y':
                plt.suptitle(f'next {condition[-1]}: neuron {cell}')
            plt.tight_layout()
    
    return psth_patch, avg_psth_patch

def get_YY_psth(neurons, session, dF, events, condition='AB', time_around=0.5, plot=True):
    '''
    Get the PSTH around the two YY events according to patch type. 
    If last_Y, this is the last Y inside an XY patch. Otherwise, if next_Y, this is the first Y following
    an alternation violation i.e., the next Y after the end of a patch.
    '''
    # Handle time window input
    if isinstance(time_around, (int, float)):
        start_time = -time_around
        end_time = time_around
    elif isinstance(time_around, (tuple, list)) and len(time_around) == 2:
        start_time, end_time = time_around
    else:
        raise ValueError("time_around must be a single number or a tuple/list of (start, end)")

    # Get PSTHs for each type of landmark
    psth_patch_last, avg_psth_patch_last = get_Y_psth(neurons, session, dF, events, condition=condition, lm='last_Y', time_around=time_around, plot=False)
    psth_patch_next, avg_psth_patch_next = get_Y_psth(neurons, session, dF, events, condition=condition, lm='next_Y', time_around=time_around, plot=False)

    # Find patches of alternating AB/BA 
    _, AB_patches, BA_patches, _, _, _ = get_repeating_XY_patches(session, min_length=0)

    if condition == 'AB':
        patches = AB_patches
    elif condition == 'BA':
        patches = BA_patches

    XY_repeats = np.array([len(patch) / 2 for patch in patches]).astype(int)

    # Plotting
    if plot:
        palette = palettes.met_brew('Johnson', n=len(np.unique(XY_repeats)), brew_type="continuous")
        num_timebins = psth_patch_last.shape[-1]

        for n, cell in enumerate(neurons):
            _, axs = plt.subplots(1, 2, figsize=(5,3), sharex=True, sharey=True)
            axs = axs.ravel()
            
            for i, length in enumerate(np.unique(XY_repeats).astype(int)):
                patch_mask = XY_repeats == length

                # Last Y
                num_patches = psth_patch_last.shape[1]
                axs[0].plot(avg_psth_patch_last[length][n], color=palette[i], linewidth=2, label=f'{length} ({len(np.where(XY_repeats == length)[0])})')
                axs[0].fill_between(np.arange(num_timebins),
                                avg_psth_patch_last[length][n] - stats.sem(psth_patch_last[n, patch_mask[:num_patches], :], axis=0),
                                avg_psth_patch_last[length][n] + stats.sem(psth_patch_last[n, patch_mask[:num_patches], :], axis=0),
                                color=palette[i], alpha=0.3) # standard error per patch length
                axs[0].set_title(f'last {condition[-1]}')

                # Next Y
                num_patches = psth_patch_next.shape[1]
                axs[1].plot(avg_psth_patch_next[length][n], color=palette[i], linewidth=2)
                axs[1].fill_between(np.arange(num_timebins),
                                avg_psth_patch_next[length][n] - stats.sem(psth_patch_next[n, patch_mask[:num_patches], :], axis=0),
                                avg_psth_patch_next[length][n] + stats.sem(psth_patch_next[n, patch_mask[:num_patches], :], axis=0),
                                color=palette[i], alpha=0.3) # standard error per patch length
                axs[1].set_title(f'next {condition[-1]}')

            axs[0].set_ylabel('dF/F')
            for ax in axs:
                ax.set_xlabel('Time (s)')
                zero_bin = int(round(-start_time / (end_time - start_time) * num_timebins))
                ax.set_xticks([0, zero_bin, num_timebins - 1])
                ax.set_xticklabels([round(start_time, 2), 0, round(end_time, 2)])
                ax.axvspan(zero_bin, num_timebins, color='gray', alpha=0.5)
                ax.spines[['right', 'top']].set_visible(False)

            plt.figlegend(loc='upper right', title=f'# {condition}s', bbox_to_anchor=(1.15, 0.8), borderaxespad=0)
            plt.suptitle(f'neuron {cell}')
            plt.tight_layout()
    
    return psth_patch_last, avg_psth_patch_last, psth_patch_next, avg_psth_patch_next


# --------- STATISTICS --------- #
def kendalls_W(chi2, N, k):
    """Effect size for Friedman test."""
    return chi2 / (N * (k - 1))

def rank_biserial_from_wilcoxon(z, N):
    """Rank-biserial correlation effect size for Wilcoxon."""
    return z / np.sqrt(N)

def compute_population_stats(goal_means, conditions, idx):
    stats = {}

    # Loop over available patch lengths
    all_lengths = sorted(set().union(*[goal_means[c].keys() for c in conditions]))

    for length in all_lengths:

        # Make sure all conditions contain this patch length
        if not all(length in goal_means[c] for c in conditions):
            print(f'Skipping {length}-length patches, because data for one or more conditions are missing.')
            continue

        # Build matrix (n_neurons, 3)
        M = np.vstack([
            goal_means['prev'][length],
            goal_means['last'][length],
            goal_means['next'][length]
        ]).T

        # # Remove neurons with NaNs
        # good = ~np.isnan(M).any(axis=1)
        # M = M[good]

        if M.shape[0] < 3:    # need at least 3 neurons for Friedman
            continue

        n_neurons = M.shape[0]

        # ----------------------------
        # 1. Friedman (omnibus test)
        # ----------------------------
        chi2, p_friedman = friedmanchisquare(M[:,0], M[:,1], M[:,2])

        # effect size: Kendall's W
        W = kendalls_W(chi2, N=n_neurons, k=3)

        # ----------------------------
        # 2. Pairwise Wilcoxon tests
        # ----------------------------
        pairwise = {}
        pairs = [('prev','last'), ('last','next'), ('prev','next')]

        for a, b in pairs:
            # Wilcoxon returns a statistic but not Z, so compute Z manually
            stat, p_w = wilcoxon(M[:, idx[a]], M[:, idx[b]])
            
            # Compute Z-score from p-value (two-sided)
            # Wilcoxon in scipy uses two-sided p-values by default.
            if p_w == 0:
                # Avoid inf z-score
                z = np.sign(stat - (n_neurons*(n_neurons+1)/4)) * 8.0  
            else:
                z = norm.ppf(p_w / 2) * -1  # two-sided → divide by 2

            r_rb = rank_biserial_from_wilcoxon(z, n_neurons)

            pairwise[f"{a}_vs_{b}"] = {
                "p": p_w,
                "rank_biserial_r": r_rb
            }

        # Store results
        stats[length] = {
            "friedman_p": p_friedman,
            "kendalls_W": W,
            "pairwise": pairwise,
            "n_neurons": n_neurons
        }

    return stats


def compute_per_neuron_stats(neuron_means):
    stats = {}

    # Assume same neuron count across conditions
    n_neurons = neuron_means['prev'].shape[1]

    for neuron in range(n_neurons):

        prev_vals = neuron_means['prev'][:, neuron]
        last_vals = neuron_means['last'][:, neuron]
        next_vals = neuron_means['next'][:, neuron]

        # Remove NaNs
        prev_vals = prev_vals[~np.isnan(prev_vals)]
        last_vals = last_vals[~np.isnan(last_vals)]
        next_vals = next_vals[~np.isnan(next_vals)]

        # Kruskal omnibus test
        kw_stat, kw_p = kruskal(prev_vals, last_vals, next_vals)

        # Pairwise U tests
        pw = {}

        for a, b in [('prev','last'), ('last','next'), ('prev','next')]:

            a_vals = neuron_means[a][:, neuron]
            b_vals = neuron_means[b][:, neuron]

            a_vals = a_vals[~np.isnan(a_vals)]
            b_vals = b_vals[~np.isnan(b_vals)]

            u_stat, p_val = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
            pw[f"{a}_vs_{b}"] = p_val

        stats[neuron] = {
            'kruskal_p': kw_p,
            'pairwise': pw
        }

    return stats


def fit_linear_regression_XYlen(neurons, Y_data, session, condition='AB', data_type='YY_diff', plot=True):
    '''
    Fit linear regression per time bin to determine if the number of preceding XYs predicts:
    (a) the difference between two consecutive Ys ['YY_diff'], or 
    (b) if the activity in the last Y in the patch ['last_Y'] 
    '''
    # Define patches
    _, AB_patches, BA_patches, _, _, _ = get_repeating_XY_patches(session, min_length=0)

    # Find preceding XY length for each patch
    if condition == 'AB':
        patches = AB_patches
    elif condition == 'BA':
        patches = BA_patches

    XY_repeats = np.array([len(patch) / 2 for patch in patches]).astype(int)

    # Perform linear regression per time bin
    x = XY_repeats

    linear_regression_result = {}
    for cell in neurons:
        linear_regression_result[cell] = {}
        for t in range(Y_data[cell].shape[1]):
            y = Y_data[cell][:,t]
            linear_regression_result[cell][t] = stats.linregress(x, y, alternative='two-sided')

    slopes = {cell: [res.slope for t, res in linear_regression_result[cell].items()] for cell in neurons}
    rvalues = {cell: [res.rvalue for t, res in linear_regression_result[cell].items()] for cell in neurons}

    # Plotting
    if plot: 
        max_slope = min(min(v) for v in slopes.values()) # keep y-axis the same for all cells
        min_slope = max(max(v) for v in slopes.values())

        for cell in neurons:
            fig = plt.figure(figsize=(7,4))
            gs = plt.GridSpec(1, 2, width_ratios=[2, 1])  
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1])
            
            n_bins = Y_data[cell].shape[1]
            n_trials = Y_data[cell].shape[0]

            cax1 = ax1.plot(slopes[cell], label='slope')
            ax1.set_title(f'Linear Regression results')
            ax1.set_xlabel('Time bins')
            ax1.set_ylim([max_slope, min_slope])
            ax1.hlines(y=0, xmin=0, xmax=n_bins, linestyles='--', colors='grey')
            # ax1.set_yticks([0, n_trials-1])
            # ax1.set_yticklabels([0, n_trials])
            ax1.set_xticks([0, n_bins])
            ax1.set_ylabel('Beta coefficients (slopes)', labelpad=0)

            axr = ax1.twinx()
            axr.set_ylim(ax1.get_ylim())
            axr.plot(rvalues[cell], color='orange', alpha=0.7, label="r-value")
            axr.set_ylabel("Pearson Correlation (r)", color='orange')
            axr.tick_params(axis='y', labelcolor='orange')
            lines_left, labels_left = ax1.get_legend_handles_labels()
            lines_right, labels_right = axr.get_legend_handles_labels()
            ax1.legend(lines_left + lines_right, labels_left + labels_right, loc="upper right")
            
            if data_type == 'YY_diff':
                vmax = np.max(np.abs(Y_data[cell]))
                vmin = -vmax
                cax2 = ax2.imshow(Y_data[cell], aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
                if condition == 'AB':
                    ax2.set_title(f'B2-B1')
                elif condition == 'BA':
                    ax2.set_title(f'A2-A1')
                cb2 = fig.colorbar(cax2, ax=ax2, label='YY diff dF/F', ticks=[vmin, vmax])
            elif data_type == 'last_Y':
                vmax = np.max(Y_data[cell])
                vmin = np.min(Y_data[cell])
                cax2 = ax2.imshow(Y_data[cell], aspect='auto', cmap='viridis')
                if condition == 'AB':
                    ax2.set_title(f'last B')
                elif condition == 'BA':
                    ax2.set_title(f'last A')
                cb2 = fig.colorbar(cax2, ax=ax2, label='dF/F', ticks=[vmin, vmax])
            cb2.ax.set_yticklabels([f"{vmin:.1f}", f"{vmax:.1f}"])
            cb2.ax.yaxis.labelpad = -10
            ax2.set_yticks([0, n_trials-1])
            ax2.set_yticklabels([0, n_trials])
            ax2.set_xticks([0, n_bins-1])
            ax2.set_xticklabels([0, n_bins])
            ax2.set_xlabel('Time bins')
            
            plt.suptitle(f'{condition}: neuron {cell}') 
            plt.tight_layout()

    return slopes, rvalues


def fit_linear_regression_XYlen_shuffle(neurons, Y_data, session, condition='AB', data_type='YY_diff', 
                                        shuffle=True, nreps=1000, plot=True):
    '''
    Fit linear regression per time bin to determine if the number of preceding XYs predicts:
    (a) the difference between two consecutive Ys ['YY_diff'], or 
    (b) if the activity in the last Y in the patch ['last_Y'] 
    If shuffle is True, a permutation test is performed nreps times. 
    '''

    # Define patches
    _, AB_patches, BA_patches, _, _, _ = get_repeating_XY_patches(session, min_length=0)

    # Find preceding XY length for each patch
    if condition == 'AB':
        patches = AB_patches
    elif condition == 'BA':
        patches = BA_patches

    XY_repeats = np.array([len(patch) / 2 for patch in patches]).astype(int)

    # Perform linear regression per time bin
    x = XY_repeats

    linear_regression_result = {}
    for cell in neurons:
        x = x[:Y_data[cell].shape[0]]
        nbins = Y_data[cell].shape[1]
        linear_regression_result[cell] = {}
        for t in range(nbins):
            y = Y_data[cell][:,t]
            linear_regression_result[cell][t] = stats.linregress(x, y, alternative='two-sided')

    slopes = {cell: np.array([res.slope for t, res in linear_regression_result[cell].items()]) for cell in neurons}
    rvalues = {cell: np.array([res.rvalue for t, res in linear_regression_result[cell].items()]) for cell in neurons}

    # Permutation test to test against null hypothesis
    slopes_shuffled = {}
    rvalues_shuffled = {}
    if shuffle:
        for cell in neurons:
            nbins = Y_data[cell].shape[1]
            slopes_shuffled[cell] = np.empty((nreps, nbins))
            rvalues_shuffled[cell] = np.empty((nreps, nbins))
            for i in range(nreps):
                np.random.shuffle(x)
                for t in range(nbins):
                    y = Y_data[cell][:,t]
                    result = stats.linregress(x, y, alternative='two-sided')
                    slopes_shuffled[cell][i,t] = result.slope
                    rvalues_shuffled[cell][i,t] = result.rvalue

    # Two-sided p-value (for each time bin)
    pvalue = {}
    for cell in neurons:
        null_dist = np.abs(slopes_shuffled[cell])
        obs = np.abs(slopes[cell])
        pvalue[cell] = np.mean(null_dist >= obs, axis=0) # pvalues = % null slopes >= observed slope

    # Compute percentiles 
    low_percentile = {}
    high_percentile = {}
    median_percentile = {}

    for cell in neurons:
        low_percentile[cell] = np.percentile(slopes_shuffled[cell], 2.5, axis=0)
        high_percentile[cell] = np.percentile(slopes_shuffled[cell], 97.5, axis=0)
        median_percentile[cell] = np.median(slopes_shuffled[cell], axis=0)

    # Plotting
    if plot: 
        max_null = max(max(v) for v in high_percentile.values())
        min_null = min(min(v) for v in low_percentile.values())
        max_slope = max(np.max(slopes[cell]) for cell in neurons)
        min_slope = min(np.min(slopes[cell]) for cell in neurons)

        global_ymax = max(max_null, max_slope) + 0.1
        global_ymin = min(min_null, min_slope) - 0.1

        for cell in neurons:
            fig = plt.figure(figsize=(7,4))
            gs = plt.GridSpec(1, 2, width_ratios=[2, 1])  
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1])
            
            n_bins = Y_data[cell].shape[1]
            n_trials = Y_data[cell].shape[0]

            cax1 = ax1.plot(slopes[cell], label='slope')
            
            if shuffle:
                # Plot percentiles of null distribution
                ax1.plot(median_percentile[cell], color='k', label='shuffle median')
                ax1.fill_between(np.arange(nbins), low_percentile[cell], high_percentile[cell], color='k', alpha=0.3)
            
                # Plot p-values
                cell_max_slope = max(np.abs(slopes[cell]))
                sig_bins = np.where(pvalue[cell] < 0.05)[0]
                ax1.scatter(sig_bins, np.ones(len(sig_bins)) * (-cell_max_slope - 0.05), s=10, color='red', marker='*')
            
            ax1.set_title(f'Linear Regression results')
            ax1.set_xlabel('Time bins')
            ax1.set_ylim([global_ymin, global_ymax])
            ax1.hlines(y=0, xmin=0, xmax=n_bins, linestyles='--', colors='grey')
            # ax1.set_yticks([0, n_trials-1])
            # ax1.set_yticklabels([0, n_trials])
            ax1.set_xticks([0, n_bins])
            ax1.set_ylabel('Beta coefficients (slopes)', labelpad=0)

            axr = ax1.twinx()
            axr.set_ylim(ax1.get_ylim())
            axr.plot(rvalues[cell], color='orange', alpha=0.7, label="r-value")
            axr.set_ylabel("Pearson Correlation (r)", color='orange')
            axr.tick_params(axis='y', labelcolor='orange')
            lines_left, labels_left = ax1.get_legend_handles_labels()
            lines_right, labels_right = axr.get_legend_handles_labels()
            ax1.legend(lines_left + lines_right, labels_left + labels_right, loc="upper right")
            
            if data_type == 'YY_diff':
                vmax = np.max(np.abs(Y_data[cell]))
                vmin = -vmax
                cax2 = ax2.imshow(Y_data[cell], aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
                if condition == 'AB':
                    ax2.set_title(f'B2-B1')
                elif condition == 'BA':
                    ax2.set_title(f'A2-A1')
                cb2 = fig.colorbar(cax2, ax=ax2, label='YY diff dF/F', ticks=[vmin, vmax])
            elif data_type == 'last_Y':
                vmax = np.max(Y_data[cell])
                vmin = np.min(Y_data[cell])
                cax2 = ax2.imshow(Y_data[cell], aspect='auto', cmap='viridis')
                if condition == 'AB':
                    ax2.set_title(f'last B')
                elif condition == 'BA':
                    ax2.set_title(f'last A')
                cb2 = fig.colorbar(cax2, ax=ax2, label='dF/F', ticks=[vmin, vmax])
            cb2.ax.set_yticklabels([f"{vmin:.1f}", f"{vmax:.1f}"])
            cb2.ax.yaxis.labelpad = -10
            ax2.set_yticks([0, n_trials-1])
            ax2.set_yticklabels([0, n_trials])
            ax2.set_xticks([0, n_bins-1])
            ax2.set_xticklabels([0, n_bins])
            ax2.set_xlabel('Time bins')
            
            plt.suptitle(f'{condition}: neuron {cell}') 
            plt.tight_layout()

    results = {}
    results['slopes'] = slopes
    results['rvalues'] = rvalues
    if shuffle:
        results['slopes_shuffled'] = slopes_shuffled
        results['rvalues_shuffled'] = rvalues_shuffled
        results['pvalue'] = pvalue
        
    return results
    

def fit_linear_regression_XYlen_cpa(neurons, Y_data, session, condition='AB', data_type='YY_diff', 
                                    shuffle=True, nreps=1000, cluster_thres=0.05, plot=True, 
                                    sort_heatmap=False, save_plot=False, save_dir='', plot_dir='', 
                                    reload=False):
    '''
    Fit linear regression per time bin to determine if the number of preceding XYs predicts:
    (a) the difference between two consecutive Ys ['YY_diff'], or 
    (b) if the activity in the last Y in the patch ['last_Y'] 
    Cluster permutation analysis is also performed to test for the significance of time clusters. 
    '''
    results_file = os.path.join(save_dir, f"{condition}_{data_type}_linear_regression_results.npz")

    # Define patches
    _, AB_patches, BA_patches, _, _, _ = get_repeating_XY_patches(session, min_length=0)

    # Find preceding XY length for each patch
    if condition == 'AB':
        patches = AB_patches
    elif condition == 'BA':
        patches = BA_patches

    XY_repeats = np.array([len(patch) / 2 for patch in patches]).astype(int)
    XY_repeats = XY_repeats[:Y_data[neurons[0]].shape[0]] # exclude last repeat if it doesn't end with an XYY patch

    # Perform linear regression per time bin
    if os.path.exists(results_file) and not reload:
            print('Linear regression with CPA file found. Loading...')
            results = np.load(results_file, allow_pickle=True)
            slopes = results['slopes'].item() 
            rvalues = results['rvalues'].item() 
            pvalues = results['pvalues'].item() 
            clusters = results['clusters'].item() 
            cluster_mass_stat = results['cluster_mass_stat'].item() 
            if 'slopes_shuffled' in results:
                slopes_shuffled = results['slopes_shuffled'].item() 
                rvalues_shuffled = results['rvalues_shuffled'].item() 
                pvalues_shuffled = results['pvalues_shuffled'].item() 
                clusters_shuffled = results['clusters_shuffled'].item() 
                cluster_mass_stat_shuffled = results['cluster_mass_stat_shuffled'].item() 
                pvalue = results['pvalue'].item() 
                cluster_pvalue = results['cluster_pvalue'].item() 
    else:
        print('Fitting linear regression with CPA')
        x = XY_repeats.copy()

        nbins = Y_data[neurons[0]].shape[1]
        linear_regression_result = {}
        for cell in neurons:
            linear_regression_result[cell] = {}
            for t in range(nbins):
                y = Y_data[cell][:,t]
                linear_regression_result[cell][t] = stats.linregress(x, y, alternative='two-sided')

        slopes = {cell: np.array([res.slope for t, res in linear_regression_result[cell].items()]) for cell in neurons}
        rvalues = {cell: np.array([res.rvalue for t, res in linear_regression_result[cell].items()]) for cell in neurons}
        pvalues = {cell: np.array([res.pvalue for t, res in linear_regression_result[cell].items()]) for cell in neurons}
        
        # Compute clusters and cluster-mass slope (statistic of interest here)
        clusters = {} # cluster = continuous span of timepoints when pvalue < threshold
        cluster_mass_stat = {cell: {} for cell in neurons}
        for cell in neurons:
            sig_bins = np.where(pvalues[cell] < cluster_thres)[0]
            
            if len(sig_bins) == 0:
                clusters[cell] = []
                continue

            # Split clusters by whether they have high or low slopes to avoid fluctuations around 0 
            sig_bins_high = sig_bins[slopes[cell][sig_bins] > 0]
            cluster_change_idx = np.where(np.diff(sig_bins_high) > 1)[0] + 1
            split_clusters_high = [c for c in np.split(sig_bins_high, cluster_change_idx) if len(c) > 0]

            sig_bins_low = sig_bins[slopes[cell][sig_bins] < 0]
            cluster_change_idx = np.where(np.diff(sig_bins_low) > 1)[0] + 1
            split_clusters_low = [c for c in np.split(sig_bins_low, cluster_change_idx) if len(c) > 0]

            # Combine all clusters
            split_clusters = split_clusters_high + split_clusters_low
            clusters[cell] = split_clusters

            for c, cluster in enumerate(split_clusters):
                cluster_mass_stat[cell][c] = np.sum(np.abs(slopes[cell][cluster]))  

        # Permutation test to test against null hypothesis
        # The null distribution is a collection of the largest cluster-mass statistic from each simulated data. 
        # If no clusters are detected in a simulation, it contributes a cluster-mass of zero to the null.
        if shuffle:
            slopes_shuffled = {}
            rvalues_shuffled = {}
            pvalues_shuffled = {}
            clusters_shuffled = {cell: {} for cell in neurons}
            cluster_mass_stat_shuffled = {cell: {} for cell in neurons}

            nbins = Y_data[neurons[0]].shape[1]
            for cell in neurons:
                slopes_shuffled[cell] = np.empty((nreps, nbins))
                rvalues_shuffled[cell] = np.empty((nreps, nbins))
                pvalues_shuffled[cell] = np.empty((nreps, nbins))

                for i in range(nreps):
                    np.random.shuffle(x)
                    for t in range(nbins):
                        y = Y_data[cell][:,t]
                        result = stats.linregress(x, y, alternative='two-sided')
                        slopes_shuffled[cell][i,t] = result.slope
                        rvalues_shuffled[cell][i,t] = result.rvalue
                        pvalues_shuffled[cell][i,t] = result.pvalue

                    # Compute clusters and cluster stats for each shuffle
                    sig_bins = np.where(pvalues_shuffled[cell][i,:] < cluster_thres)[0]
                    
                    # Split clusters by whether they have high or low slopes to avoid fluctuations around 0 
                    sig_bins_high = sig_bins[slopes[cell][sig_bins] > 0]
                    cluster_change_idx = np.where(np.diff(sig_bins_high) > 1)[0] + 1
                    split_clusters_high = [c for c in np.split(sig_bins_high, cluster_change_idx) if len(c) > 0]

                    sig_bins_low = sig_bins[slopes[cell][sig_bins] < 0]
                    cluster_change_idx = np.where(np.diff(sig_bins_low) > 1)[0] + 1
                    split_clusters_low = [c for c in np.split(sig_bins_low, cluster_change_idx) if len(c) > 0]

                    # Combine all clusters
                    split_clusters = split_clusters_high + split_clusters_low
                    clusters_shuffled[cell][i] = split_clusters

                    # Find the largest cluster-mass statistic for this shuffle
                    all_cluster_masses = []
                    if len(split_clusters) > 0:
                        for c, cluster in enumerate(split_clusters):
                            all_cluster_masses.append(np.sum(np.abs(slopes_shuffled[cell][i,cluster]))) 
                        max_cluster_mass = np.max(all_cluster_masses) # per shuffle
                        cluster_mass_stat_shuffled[cell][i] = max_cluster_mass
                    else:
                        cluster_mass_stat_shuffled[cell][i] = 0

            # Two-sided p-value (for each time bin) against null hypothesis
            pvalue = {}
            cluster_pvalue = {cell: {} for cell in neurons}
            for cell in neurons:
                null_dist = np.abs(slopes_shuffled[cell])
                obs = np.abs(slopes[cell])
                pvalue[cell] = np.mean(null_dist >= obs, axis=0) # pvalues = % null slopes >= observed slope

                null_cluster_dist = np.array(list(cluster_mass_stat_shuffled[cell].values()))
                
                for c in range(len(clusters[cell])):
                    cluster_obs = np.abs(cluster_mass_stat[cell][c])
                    cluster_pvalue[cell][c] = np.mean(null_cluster_dist >= cluster_obs)

        # Save results
        results = {}
        results['slopes'] = slopes
        results['rvalues'] = rvalues
        results['pvalues'] = pvalues
        results['clusters'] = clusters
        results['cluster_mass_stat'] = cluster_mass_stat
        if shuffle:
            results['slopes_shuffled'] = slopes_shuffled
            results['rvalues_shuffled'] = rvalues_shuffled
            results['pvalues_shuffled'] = pvalues_shuffled
            results['clusters_shuffled'] = clusters_shuffled
            results['cluster_mass_stat_shuffled'] = cluster_mass_stat_shuffled
            results['pvalue'] = pvalue
            results['cluster_pvalue'] = cluster_pvalue
            
        if save_dir:
            np_results = {key: np.array(value, dtype=object) for key, value in results.items()}
            np.savez(results_file, **np_results)
            print(f"Saved results to: {results_file}")
            
    # Compute percentiles 
    low_percentile = {cell: np.percentile(slopes_shuffled[cell], 2.5, axis=0) for cell in neurons}
    high_percentile = {cell: np.percentile(slopes_shuffled[cell], 97.5, axis=0) for cell in neurons}
    median_percentile = {cell: np.median(slopes_shuffled[cell], axis=0) for cell in neurons}

    # Plotting
    if plot: 
        max_null = max(max(v) for v in high_percentile.values())
        min_null = min(min(v) for v in low_percentile.values())
        max_slope = max(np.max(slopes[cell]) for cell in neurons)
        min_slope = min(np.min(slopes[cell]) for cell in neurons)
        max_rvalue = max(np.max(rvalues[cell]) for cell in neurons)
        min_rvalue = min(np.min(rvalues[cell]) for cell in neurons)

        global_ymax = max(max_null, max_slope, max_rvalue) + 0.1
        global_ymin = min(min_null, min_slope, min_rvalue) - 0.8

        for cell in neurons:
            fig = plt.figure(figsize=(8,4))
            gs = plt.GridSpec(1, 2, width_ratios=[5, 3])  
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1])
            
            n_trials = Y_data[cell].shape[0]
            nbins = Y_data[cell].shape[1]

            # Regression results
            ax1.plot(slopes[cell], label='slope')
            
            if shuffle:
                # Plot percentiles of null distribution
                ax1.plot(median_percentile[cell], color='k', label='shuffle median')
                ax1.fill_between(np.arange(nbins), low_percentile[cell], high_percentile[cell], color='k', alpha=0.3)
            
                # Plot p-values
                cell_max_slope = max(np.abs(slopes[cell]))
                cell_min_slope = min(slopes[cell])
                sig_bins = np.where(pvalue[cell] < 0.05)[0]
                ax1.scatter(sig_bins, np.ones(len(sig_bins)) * (cell_min_slope - 0.2), s=10, color='red', marker='*')
            
                # Plot significant clusters from CPA and annotate p-value
                y_pos = cell_min_slope - 0.4
                for c, seg in enumerate(clusters[cell]):
                    if cluster_pvalue[cell][c] < 0.05:
                        ax1.hlines(y_pos, seg[0], seg[-1], color='green', linewidth=3)
                        text_y = y_pos - 0.10  
                        text_x = (seg[0] + seg[-1]) / 2  
                        label = f"p={cluster_pvalue[cell][c]:.3f}"
                        ax1.annotate(label, xy=(text_x, text_y), ha='center', va='top', fontsize=8)
            
            ax1.set_title(f'Linear Regression results')
            ax1.set_xlabel('Time bins')
            ax1.set_ylim([global_ymin - 0.5, global_ymax])
            ax1.hlines(y=0, xmin=0, xmax=nbins-1, linestyles='--', colors='grey')
            # ax1.set_yticks([0, n_trials-1])
            # ax1.set_yticklabels([0, n_trials])
            ax1.set_xticks([0, nbins-1])
            ax1.set_ylabel('Beta coefficients (slopes)', labelpad=0)

            axr = ax1.twinx()
            axr.set_ylim(ax1.get_ylim())
            axr.plot(rvalues[cell], color='orange', alpha=0.7, label="r-value")
            axr.set_ylabel("Pearson Correlation (r)", color='orange')
            axr.tick_params(axis='y', labelcolor='orange')
            lines_left, labels_left = ax1.get_legend_handles_labels()
            lines_right, labels_right = axr.get_legend_handles_labels()
            ax1.legend(lines_left + lines_right, labels_left + labels_right, loc="upper right")
            
            # Heatmaps
            XY_repeat_sorting_idx = np.argsort(XY_repeats, kind='stable')
            sorted_repeats = XY_repeats[XY_repeat_sorting_idx]
            if sort_heatmap:
                heatmap_data = Y_data[cell][XY_repeat_sorting_idx]
                change_rows = np.where(np.diff(sorted_repeats) != 0)[0] + 1

                block_starts = np.concatenate(([0], change_rows))
                block_ends   = np.concatenate((change_rows, [len(sorted_repeats)]))
                block_centers = (block_starts + block_ends) / 2 - 0.5
                block_values  = [sorted_repeats[start] for start in block_starts]

            else:
                heatmap_data = Y_data[cell]

            if data_type == 'YY_diff':
                vmax = np.max(np.abs(heatmap_data))
                vmin = -vmax
                cax2 = ax2.imshow(heatmap_data, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
                if sort_heatmap:
                    for r in change_rows:
                        ax2.axhline(r - 0.5, color='black', linewidth=0.8, linestyle='--')
                    # Indicate number of XY repeats  per block
                    right_ax = ax2.secondary_yaxis('right')
                    right_ax.set_yticks(block_centers)
                    right_ax.set_yticklabels(block_values, fontsize=6)
                    right_ax.set_ylabel('XY repeats', fontsize=8)

                if condition == 'AB':
                    ax2.set_title(f'B2-B1')
                elif condition == 'BA':
                    ax2.set_title(f'A2-A1')
                cb2 = fig.colorbar(cax2, ax=ax2, label='YY diff dF/F', ticks=[vmin, vmax], pad=0.3)
            elif data_type == 'last_Y':
                vmax = np.max(heatmap_data)
                vmin = np.min(heatmap_data)
                cax2 = ax2.imshow(heatmap_data, aspect='auto', cmap='viridis')
                if condition == 'AB':
                    ax2.set_title(f'last B')
                elif condition == 'BA':
                    ax2.set_title(f'last A')
                cb2 = fig.colorbar(cax2, ax=ax2, label='dF/F', ticks=[vmin, vmax], pad=0.3)
            cb2.ax.set_yticklabels([f"{vmin:.1f}", f"{vmax:.1f}"])
            cb2.ax.yaxis.labelpad = -10
            ax2.set_yticks([0, n_trials-1])
            ax2.set_yticklabels([0, n_trials-1])
            ax2.set_xticks([0, nbins-1])
            ax2.set_xticklabels([0, nbins])
            ax2.set_xlabel('Time bins')
            
            plt.suptitle(f'{condition}: neuron {cell}') 
            plt.tight_layout()

            if save_plot:
                condition_save_path = os.path.join(plot_dir, condition)
                os.makedirs(condition_save_path, exist_ok=True)
                plt.savefig(condition_save_path + f'/{data_type}_neuron{cell}.png', dpi=300)

            plt.close(fig)
            
    return results
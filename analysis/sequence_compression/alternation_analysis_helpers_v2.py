import matplotlib.pyplot as plt
import numpy as np
import os, sys
import scipy.stats as stats
from collections import Counter

parse_session_functions = None
def set_parse_session_functions(psf):
    global parse_session_functions
    parse_session_functions = psf

def get_XYY_patches(session, precede_XY=False):
    '''
    Find ABB and BAA patches
    
    arguments:
    - precede_XY: whether the two landmarks preceding the patch should be XY (relevant for truly random sequences) (default = False)
    '''     
    event_idx = session['event_idx'] # includes non-goals
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

    # XYY
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

def get_valid_patches(session, dF, example_cell, condition, bins):
    '''Exclude patches if session ended before enough frames were collected for a landmark (the last one)'''

    if session['stim_order'] == 'random':
        ABB_patches, BAA_patches, _, _ = get_XYY_patches(session, precede_XY=True)
    elif session['stim_order'] == 'pseudorandom':
        ABB_patches, BAA_patches, _, _ = get_XYY_patches(session, precede_XY=False)

    if condition == 'AB':
        XYY_patches = ABB_patches
    elif condition == 'BA':
        XYY_patches = BAA_patches

    events_YY = get_YY_events(session, XYY_patches)

    valid_patch_indices = []
        
    for p_idx, patch in enumerate(XYY_patches):
        valid_patch = True

        for lm in patch[1:]:
            binned = temporal_bin_lm_firing_reward_aligned(example_cell, dF, events_YY[lm], frames_around=bins/2, bins=bins) # using the first neuron as reference

            if binned is None or np.isnan(binned).any():
                valid_patch = False
                break

        if valid_patch:
            valid_patch_indices.append(p_idx)

    return valid_patch_indices

def get_YY_events(session, XYY_patches):
    '''Find the start, reward (or imaginary reward) and end for each YY event in a patch'''

    lm_entry_idx, lm_exit_idx = parse_session_functions.get_lm_entry_exit(session)
    lm_entry_idx = lm_entry_idx[:len(session['event_idx'])]
    lm_exit_idx = lm_exit_idx[:len(session['event_idx'])]

    events_YY = {}
    for lm in XYY_patches:
        Y1 = lm[1]
        Y2 = lm[2]
        events_YY[Y1] = { "start": lm_entry_idx[Y1], "reward": session['event_idx'][Y1], "end": lm_exit_idx[Y1]}
        events_YY[Y2] = { "start": lm_entry_idx[Y2], "reward": session['event_idx'][Y2], "end": lm_exit_idx[Y2]}

    return events_YY

def get_XY_repeats(patches, cluster=False):
    '''Define number of XY repeats inside each XY patch'''

    XY_repeats = np.array([len(patch) / 2 for patch in patches]).astype(int)

    num_repeats = Counter(XY_repeats)

    if cluster and 5 in num_repeats:
        print('Clustering XY repeats of 3-4 and > 5 together to avoid sampling issues.')

        clustered_XY_repeats = []
        for r, rep in enumerate(XY_repeats):
            if rep == 1 or rep == 2:
                clustered_XY_repeats.append(rep)
            elif 3 <= rep <= 4:
                clustered_XY_repeats.append(3)
            else:
                clustered_XY_repeats.append(4)

        return np.array(clustered_XY_repeats), cluster

    else:
        return XY_repeats, False
    
def get_min_frames_between_lms(session):
    '''Find the minimum number of frames between two landmarks'''

    lm_entry_idx, lm_exit_idx = parse_session_functions.get_lm_entry_exit(session)
    d_idx = lm_entry_idx[1:] - lm_exit_idx[:-1]
    frames_around = np.min(d_idx)
    frames_around = int(np.round(frames_around / 10) * 10)

    if frames_around > 30:
        print(f'The min distance in frames between two landmarks is {frames_around}, equivalent to {np.round(frames_around/45,2)} s, but capping to 30 frames.')
        frames_around = 30
    elif frames_around < 20:
        print(f'The min distance in frames between two landmarks is {frames_around}, equivalent to {np.round(frames_around/45,2)} s, but capping to 20 frames.')
        frames_around = 20 
    else:
        print(f'The min distance in frames between two landmarks is {frames_around}, equivalent to {np.round(frames_around/45,2)} s.')

    return frames_around

def get_repeating_XY_patches(session, min_length=2):
    ''' Find patches of alternating AB/BA '''
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
            patch = combined_sorted[start:i]
            if i - start >= min_length:
                # enforce even length (pairs)
                if len(patch) % 2 != 0:
                    patch = patch[:-1]  # drop last element

                if len(patch) >= min_length:  # still valid after trimming
                    patches.append(patch)

            start = i

    # Check last patch 
    if len(labels_sorted) - start >= min_length:
        patch = combined_sorted[start:]
        if len(patch) % 2 != 0:
            patch = patch[:-1] 
        patches.append(patch)

    # Remove patch if it is not followed by another lm 
    patches = [patch for patch in patches if not np.any(patch == len(combined) - 1)]

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

def temporal_bin_lm_firing_reward_aligned(cell, dF, event, frames_around, bins=90):
    '''
    Temporal binning around reward:
    [reward - frames_around → reward + frames_around]
    For non-rewarded landmarks, the event corresponds to where reward would be on average, 
    or the midpoint of the landmark if average reward exceeds landmark boundaries
    '''
    event = event["reward"]
    start = int(event - frames_around)
    end   = int(event + frames_around)

    # Safety check (avoid out-of-bounds)
    if start < 0 or end >= dF.shape[1]:
        return None  # or np.full(bins, np.nan)

    # Extract window
    frames = np.arange(start, end)
    firing = dF[cell, frames]

    # Define bins over fixed window
    bin_edges = np.linspace(0, len(frames), bins + 1)
    bin_ix = np.digitize(np.arange(len(frames)), bin_edges)

    binned_phase_firing = np.zeros(bins)

    for j in range(bins):
        values = firing[bin_ix == j + 1]
        if len(values) > 0:
            binned_phase_firing[j] = np.mean(values)
        else:
            binned_phase_firing[j] = np.nan  # safer than 0

    return binned_phase_firing

def get_reward_aligned_temporal_phase_binning_per_lm(neurons, dF, XYY_patches, event_idx, session, bins=30, condition='AA', zscoring=False, plot=True):
    '''Binning of neural activity inside a XYY patch from the beginning to the end of each landmark in the the patch.'''
    
    n_lms = len(XYY_patches[0]) - 1
    if condition in {"AA", "BB"}:
        assert n_lms == 2, "Each patch should have 3 landmarks - XYY"

    # Compute valid patches i.e. making sure that there are enough frames within all landmarks considered
    valid_patch_indices = get_valid_patches(session, dF, neurons[0], condition, bins)

    # Collect all landmark pair (YY) binnings for all valid patches
    binned_XYY_phase_firing = {cell: [] for cell in neurons}
     
    for n, cell in enumerate(neurons):
        cell_patches = []

        for p_idx in valid_patch_indices:
            patch_bin_list = [temporal_bin_lm_firing_reward_aligned(cell, dF, event_idx[lm], frames_around=bins/2, bins=bins) for lm in XYY_patches[p_idx][1:]]

            # combine data from two landmarks
            linear_patch_binned = np.concatenate(patch_bin_list)

            # z-score within patch
            if zscoring:
                if np.std(linear_patch_binned) > 0:
                    linear_patch_binned = stats.zscore(linear_patch_binned)
                else:
                    linear_patch_binned = np.zeros_like(linear_patch_binned)  # avoid NaNs

            cell_patches.append(linear_patch_binned)

        binned_XYY_phase_firing[cell] = np.asarray(cell_patches)

    # Average across patches [n_neurons x n_bins * n_lms))]
    avg_binned_XYY_phase_firing = np.array([np.nanmean(binned_XYY_phase_firing[cell], axis=0) for cell in neurons])

    # Z-score
    zscored_avg_binned_XYY_phase_firing = stats.zscore(avg_binned_XYY_phase_firing, axis=1)
    vmax = np.nanmax(np.abs(zscored_avg_binned_XYY_phase_firing))
    vmin = -vmax

    # Sort according to max firing 
    peak_bins = np.argmax(zscored_avg_binned_XYY_phase_firing, axis=1)
    sort_order = np.argsort(peak_bins)
    sorted_zscored_avg_binned_XYY = zscored_avg_binned_XYY_phase_firing[sort_order]

    # Plotting
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(4, 3), sharey=True)

        lm_data = [
            sorted_zscored_avg_binned_XYY[:, :bins],
            sorted_zscored_avg_binned_XYY[:, bins:]
        ]

        reward_bin = bins // 2

        if condition == 'AA':
            titles = ['A1', 'A2']
        elif condition == 'BB':
            titles = ['B1', 'B2']
        else:
            titles = ['Y1', 'Y2']

        for ax, data, title in zip(axes, lm_data, titles):
            im = ax.imshow(data, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, interpolation='none')
            ax.axvline(reward_bin, linestyle='--', color='white', linewidth=1)

            ax.set_xticks([0, bins - 1])
            ax.set_xticklabels(['entry', 'exit'], rotation=0)
            ax.set_title(title)
            # ax.set_xlabel('Time bins')

        # Y axis (shared)
        axes[0].set_ylabel('Neurons', labelpad=-5)
        axes[0].set_yticks([0, len(neurons) - 1])
        axes[0].set_yticklabels([0, len(neurons)])

        # Single colorbar
        cax = fig.add_axes([0.97, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('z-scored dF/F')
        cb.set_ticks([vmin, 0, vmax])
        cb.set_ticklabels([np.round(vmin,1), 0, np.round(vmax,1)])
        fig.tight_layout()

    # Collect all data into a dict - maintain similar structure to get_spatial_and_temporal_ABB_binning
    binned_XYY_phase_activity = {}
    binned_XYY_phase_activity['temporal_XYY_firing'] = binned_XYY_phase_firing # this is used for CPA
    binned_XYY_phase_activity['avg_temporal_XYY_firing'] = avg_binned_XYY_phase_firing
    binned_XYY_phase_activity['zscored_sorted_temporal'] = sorted_zscored_avg_binned_XYY
    binned_XYY_phase_activity['valid_patch_indices'] = valid_patch_indices

    return binned_XYY_phase_activity

def fit_linear_regression_XYlen_cpa(neurons, YY_data, session, condition='AB', data_type='YY_diff', 
                                    bins=30, shuffle=True, nreps=1000, cluster_thres=0.05, zscored=False, plot=True, 
                                    sort_heatmap=False, cluster_repeats=False, save_plot=False, save_dir='', plot_dir='', 
                                    reload=False):
    '''
    Fit linear regression per time bin to determine if the number of preceding XYs predicts:
    (a) the difference between two consecutive Ys ['YY_diff'], or 
    (b) if the activity in the last Y in the patch ['last_Y'] 
    Cluster permutation analysis is also performed to test for the significance of time clusters. 
    '''
    results_file = os.path.join(save_dir, f"{condition}_{data_type}_linear_regression_results.npz")

    # Define patches
    _, AB_patches, BA_patches, _, _, _ = get_repeating_XY_patches(session, min_length=2)

    # Find preceding XY length for each patch
    if condition == 'AB':
        patches = AB_patches
    elif condition == 'BA':
        patches = BA_patches

    XY_repeats, _ = get_XY_repeats(patches, cluster=cluster_repeats)
    XY_repeats = XY_repeats[YY_data['valid_patch_indices']]
    
    # Get the difference between two YYs 
    YY_diff = {}
    for cell in neurons:
        YY_diff[cell] = YY_data['temporal_XYY_firing'][cell][:, bins:] - YY_data['temporal_XYY_firing'][cell][:, :bins]
    Y_data = YY_diff

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

        return results

    else:
        print('\tFitting linear regression with CPA')
        x = XY_repeats.copy()

        # Ensure there are more than 1 x values (repeats)
        if len(x) < 2 or len(np.unique(x)) < 2:
            print("Not enough variability in XY repeats — skipping regression.")

            # Return empty / NaN structure
            results = {
                'slopes': {cell: np.full(bins, np.nan) for cell in neurons},
                'rvalues': {cell: np.full(bins, np.nan) for cell in neurons},
                'pvalues': {cell: np.full(bins, np.nan) for cell in neurons},
                'clusters': {cell: [] for cell in neurons},
                'cluster_mass_stat': {cell: {} for cell in neurons},
            }

            return results
        
        else:
            linear_regression_result = {}
            for cell in neurons:
                linear_regression_result[cell] = {}

                # Run the regression for each timebin separately
                for t in range(bins):
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
            
                for cell in neurons:
                    slopes_shuffled[cell] = np.empty((nreps, bins))
                    rvalues_shuffled[cell] = np.empty((nreps, bins))
                    pvalues_shuffled[cell] = np.empty((nreps, bins))

                    for i in range(nreps):
                        x_shuffled = x.copy()
                        np.random.shuffle(x_shuffled)
                        for t in range(bins):
                            y = Y_data[cell][:,t]
                            result = stats.linregress(x_shuffled, y, alternative='two-sided')
                            slopes_shuffled[cell][i,t] = result.slope
                            rvalues_shuffled[cell][i,t] = result.rvalue
                            pvalues_shuffled[cell][i,t] = result.pvalue

                        # Compute clusters and cluster stats for each shuffle
                        sig_bins = np.where(pvalues_shuffled[cell][i,:] < cluster_thres)[0]
                        
                        # Split clusters by whether they have high or low slopes to avoid fluctuations around 0 
                        sig_bins_high = sig_bins[slopes_shuffled[cell][i, sig_bins] > 0]
                        cluster_change_idx = np.where(np.diff(sig_bins_high) > 1)[0] + 1
                        split_clusters_high = [c for c in np.split(sig_bins_high, cluster_change_idx) if len(c) > 0]

                        sig_bins_low = sig_bins[slopes_shuffled[cell][i, sig_bins] < 0]
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
                results['cluster_pvalue'] = cluster_pvalue # this is used to select candidate neurons
                
            if save_dir:
                np_results = {key: np.array(value, dtype=object) for key, value in results.items()}
                np.savez(results_file, **np_results)
                print(f"\tSaved results in: {results_file}")
                
        # Plotting
        if plot: 
            plot_cpa_results(results, neurons, YY_data, session, Y_data, XY_repeats, condition, data_type, bins, sort_heatmap, zscored, save_plot, plot_dir)
                
        return results
    
def plot_cpa_results(cpa_results, neurons, YY_data, session, Y_data=None, XY_repeats=None, 
                     condition='AB', data_type='YY_diff', bins=30, sort_heatmap=True, 
                     cluster_repeats=False, zscored=True, save_plot=False, plot_dir='', axes=None):

    # Unwrap CPA results
    cpa_results = {
        k: v.item() if isinstance(v, np.ndarray) and v.shape == () else v
        for k, v in cpa_results.items()
    }
    
    # Get patches of XY repeats if not provided
    if XY_repeats is None:
        _, AB_patches, BA_patches, _, _, _ = get_repeating_XY_patches(session, min_length=2)

        # Find preceding XY length for each patch
        if condition == 'AB':
            patches = AB_patches
        elif condition == 'BA':
            patches = BA_patches

        XY_repeats, _ = get_XY_repeats(patches, cluster=cluster_repeats)
        XY_repeats = XY_repeats[YY_data['valid_patch_indices']]
    
    # Define Y data for CPA if not provided
    Y_data = {}
    for cell in neurons:
        Y_data[cell] = YY_data['temporal_XYY_firing'][cell][:, bins:] - YY_data['temporal_XYY_firing'][cell][:, :bins]

    # Compute percentiles 
    low_percentile = {cell: np.percentile(cpa_results['slopes_shuffled'][cell], 2.5, axis=0) for cell in neurons}
    high_percentile = {cell: np.percentile(cpa_results['slopes_shuffled'][cell], 97.5, axis=0) for cell in neurons}
    median_percentile = {cell: np.median(cpa_results['slopes_shuffled'][cell], axis=0) for cell in neurons}

    # Plotting
    max_null = max(max(v) for v in high_percentile.values())
    min_null = min(min(v) for v in low_percentile.values())
    max_slope = max(np.max(cpa_results['slopes'][cell]) for cell in neurons)
    min_slope = min(np.min(cpa_results['slopes'][cell]) for cell in neurons)
    max_rvalue = max(np.max(cpa_results['rvalues'][cell]) for cell in neurons)
    min_rvalue = min(np.min(cpa_results['rvalues'][cell]) for cell in neurons)

    global_ymax = max(max_null, max_slope, max_rvalue) + 0.1
    global_ymin = min(min_null, min_slope, min_rvalue) - 0.8

    for cell in neurons:
        if axes is None:
            fig = plt.figure(figsize=(8,4))
            gs = plt.GridSpec(1, 2, width_ratios=[5, 3])  
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1])
        else:
            ax1, ax2 = axes
            fig = ax1.figure 
        
        n_trials = Y_data[cell].shape[0]

        # Regression results
        ax1.plot(cpa_results['slopes'][cell], label='slope')
        
        if 'slopes_shuffled' in cpa_results:
            # Plot percentiles of null distribution
            ax1.plot(median_percentile[cell], color='k', label='shuffle median')
            ax1.fill_between(np.arange(bins), low_percentile[cell], high_percentile[cell], color='k', alpha=0.3)
        
            # Plot p-values
            cell_max_slope = max(np.abs(cpa_results['slopes'][cell]))
            cell_min_slope = min(cpa_results['slopes'][cell])
            sig_bins = np.where(cpa_results['pvalue'][cell] < 0.05)[0]
            ax1.scatter(sig_bins, np.ones(len(sig_bins)) * (cell_min_slope - 0.2), s=10, color='red', marker='*')
        
            # Plot significant clusters from CPA and annotate p-value
            y_pos = cell_min_slope - 0.4
            for c, seg in enumerate(cpa_results['clusters'][cell]):
                if cpa_results['cluster_pvalue'][cell][c] < 0.05:
                    ax1.hlines(y_pos, seg[0], seg[-1], color='green', linewidth=3)
                    text_y = y_pos - 0.10  
                    text_x = (seg[0] + seg[-1]) / 2  
                    label = f"p={cpa_results['cluster_pvalue'][cell][c]:.3f}"
                    ax1.annotate(label, xy=(text_x, text_y), ha='center', va='top', fontsize=8)
        
        ax1.set_title(f'Linear Regression results')
        ax1.set_xlabel('Time bins')
        ax1.set_ylim([global_ymin - 0.5, global_ymax])
        ax1.hlines(y=0, xmin=0, xmax=bins-1, linestyles='--', colors='grey')
        ax1.set_xticks([0, bins-1])
        ax1.set_ylabel('Beta coefficients (slopes)', labelpad=0)

        axr = ax1.twinx()
        axr.set_ylim(ax1.get_ylim())
        axr.plot(cpa_results['rvalues'][cell], color='orange', alpha=0.7, label="r-value")
        axr.set_ylabel("Pearson Correlation (r)", color='orange')
        axr.tick_params(axis='y', labelcolor='orange')
        
        # legend
        handles = []
        labels = []
        for ax in [ax1, axr]:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        leg = ax1.legend(handles, labels, loc="lower right", frameon=False, handlelength=0, handletextpad=0)

        for txt, h in zip(leg.get_texts(), handles):   # color text to match lines + remove handles
            txt.set_color(h.get_color())
        for item in leg.legend_handles:
            item.set_visible(False)
         
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
            if zscored:
                cb2 = fig.colorbar(cax2, ax=ax2, label='z-scored Y-Y dF/F', ticks=[vmin, vmax], pad=0.3)
            else:
                cb2 = fig.colorbar(cax2, ax=ax2, label='Y-Y dF/F', ticks=[vmin, vmax], pad=0.3)
        
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
        ax2.set_xticks([0, bins-1])
        ax2.set_xticklabels([0, bins])
        ax2.set_xlabel('Time bins')
        
        plt.suptitle(f'{condition}: neuron {cell}') 
        plt.tight_layout()

        if save_plot:
            if plot_dir == '':
                plot_dir = session['save_dir']
            condition_save_path = os.path.join(plot_dir, condition)
            os.makedirs(condition_save_path, exist_ok=True)
            plt.savefig(condition_save_path + f'/{data_type}_neuron{cell}.png', dpi=300)

        if len(neurons) > 100:
            plt.close(fig)

def get_binned_Y2_activity(neurons, dF, session, XYY_patches, bins=30, zscoring=False):
    '''Bin neural acitvity in Y2 after XY repeats, reward-aligned'''

    binned_Y2_activity = {cell: [] for cell in neurons}

    # Find start, reward and end timepoints inside YY events 
    events_YY = get_YY_events(session, XYY_patches)

    for patch in XYY_patches:
        Y2 = patch[-1]  

        for cell in neurons:
            binned_Y2 = temporal_bin_lm_firing_reward_aligned(cell, dF, events_YY[Y2], frames_around=bins/2, bins=bins)

            # z-score across bins for this trial
            if zscoring:
                if np.std(binned_Y2) > 0:
                    binned_Y2 = stats.zscore(binned_Y2)
                else:
                    binned_Y2 = np.zeros_like(binned_Y2)  # avoid NaNs

            binned_Y2_activity[cell].append(binned_Y2)

    # Convert each neuron's list of trials into a 2D array
    for cell in neurons:
        binned_Y2_activity[cell] = np.asarray(binned_Y2_activity[cell])

    return binned_Y2_activity

def get_Y2_activity(neurons, dF, session, XYY_patches):
    ''' Compute the average activity inside the Y2 lm (w/o z-scoring) '''

    # Find start, reward and end timepoints inside YY events 
    events_YY = get_YY_events(session, XYY_patches)
    
    Y2_activity = {cell: [] for cell in neurons}

    for patch in XYY_patches:
        Y2 = patch[-1]          # keep the 2nd Y event (first after violation)
        frames = np.arange(events_YY[Y2]['start'], events_YY[Y2]['end'])

        for n, cell in enumerate(neurons):
            Y2_activity[cell].append(np.mean(dF[cell, frames]))

    return Y2_activity

def fit_linear_regression_XYlen(neurons, y_data, dF, session, condition='AB', data_type='Y2_ramp', 
                                    bins=30, shuffle=True, nreps=1000, plot=True, zscoring=False,
                                    sort_heatmap=False, cluster_repeats=False, save_plot=False, save_dir='', plot_dir='', 
                                    reload=False):
    '''
    Fit linear regression per cell to determine if the number of preceding XYs predicts
    the activity in the last Y in the patch ['Y2_ramp'] 
    '''
    results_file = os.path.join(save_dir, f"{condition}_{data_type}_linear_regression_results.npz")

    # Define patches
    _, AB_patches, BA_patches, _, _, _ = get_repeating_XY_patches(session, min_length=2)

    # Find preceding XY length for each patch
    if condition == 'AB':
        patches = AB_patches
    elif condition == 'BA':
        patches = BA_patches

    XY_repeats, _ = get_XY_repeats(patches, cluster=cluster_repeats)
    valid_patch_indices = get_valid_patches(session, dF, neurons[0], condition, bins)
    XY_repeats = XY_repeats[valid_patch_indices]
    
    # Perform linear regression 
    if os.path.exists(results_file) and not reload:
        print('Linear regression file found. Loading...')
        results = np.load(results_file, allow_pickle=True)
        slopes = results['slopes'].item() 
        rvalues = results['rvalues'].item() 
        pvalues = results['pvalues'].item() 
        intercepts = results['intercepts'].item()
        if 'slopes_shuffled' in results:
            slopes_shuffled = results['slopes_shuffled'].item() 
            rvalues_shuffled = results['rvalues_shuffled'].item() 
            pvalues_shuffled = results['pvalues_shuffled'].item() 
            intercepts_shuffled = results['intercepts_shuffled'].item()
            pvalue = results['pvalue'].item() 

        return results

    else:
        print('\tFitting linear regression')
        x = XY_repeats.copy()

        # Ensure there are more than 1 x values (repeats)
        if len(x) < 2 or len(np.unique(x)) < 2:
            print("Not enough variability in XY repeats — skipping regression.")

            # Return empty / NaN structure
            results = {
                'slopes': {cell: np.nan for cell in neurons},
                'rvalues': {cell: np.nan for cell in neurons},
                'pvalues': {cell: np.nan for cell in neurons},
                'intercepts': {cell: np.nan for cell in neurons}
            }

            return results
        
        else:
            linear_regression_result = {}

            # Run the regression for each cell
            for cell in neurons:               
                y = y_data[cell]
                linear_regression_result[cell] = stats.linregress(x, y, alternative='two-sided')
                
            slopes = {cell: linear_regression_result[cell].slope for cell in neurons}
            rvalues = {cell: linear_regression_result[cell].rvalue for cell in neurons}
            pvalues = {cell: linear_regression_result[cell].pvalue for cell in neurons}
            intercepts = {cell: linear_regression_result[cell].intercept for cell in neurons}
            
            # Permutation test to test against null hypothesis
            if shuffle:    
                slopes_shuffled = {}
                rvalues_shuffled = {}
                pvalues_shuffled = {}
                intercepts_shuffled = {}
                
                for cell in neurons:
                    slopes_shuffled[cell] = np.empty((nreps))
                    rvalues_shuffled[cell] = np.empty((nreps))
                    pvalues_shuffled[cell] = np.empty((nreps))
                    intercepts_shuffled[cell] = np.empty((nreps))

                    for i in range(nreps):
                        x_shuffled = x.copy()
                        np.random.shuffle(x_shuffled)
                        y = y_data[cell]
                        result = stats.linregress(x_shuffled, y, alternative='two-sided')
                        slopes_shuffled[cell][i] = result.slope
                        rvalues_shuffled[cell][i] = result.rvalue
                        pvalues_shuffled[cell][i] = result.pvalue
                        intercepts_shuffled[cell][i] = result.intercept

                # Two-sided p-value against null hypothesis
                pvalue = {}
                for cell in neurons:
                    null_dist = np.abs(slopes_shuffled[cell])
                    obs = np.abs(slopes[cell])
                    pvalue[cell] = np.mean(null_dist >= obs, axis=0) # pvalues = % null slopes >= observed slope

            # Save results
            results = {}
            results['slopes'] = slopes
            results['rvalues'] = rvalues
            results['pvalues'] = pvalues
            results['intercepts'] = intercepts
            if shuffle:
                results['slopes_shuffled'] = slopes_shuffled
                results['rvalues_shuffled'] = rvalues_shuffled
                results['pvalues_shuffled'] = pvalues_shuffled
                results['intercepts_shuffled'] = intercepts_shuffled
                results['pvalue'] = pvalue
                
            if save_dir:
                np_results = {key: np.array(value, dtype=object) for key, value in results.items()}
                np.savez(results_file, **np_results)
                print(f"\tSaved results in: {results_file}")
                
        # Plotting
        if plot: 
            plot_linear_regression_results(results, neurons, dF, session, y_data, condition, data_type, bins, sort_heatmap, cluster_repeats, zscoring, save_plot, plot_dir)
                
        return results


def plot_linear_regression_results(results, neurons, dF, session, y_data, 
                                   condition='AB', data_type='Y2_ramp', bins=30, sort_heatmap=True, 
                                   cluster_repeats=False, zscoring=False, save_plot=False, plot_dir='', axes=None):

    # Unwrap linear regression results
    results = {
        k: v.item() if isinstance(v, np.ndarray) and v.shape == () else v
        for k, v in results.items()
    }
    
    # Get patches of XY repeats 
    _, AB_patches, BA_patches, _, _, _ = get_repeating_XY_patches(session, min_length=2)

    # Find preceding XY length for each patch
    if condition == 'AB':
        patches = AB_patches
    elif condition == 'BA':
        patches = BA_patches

    XY_repeats, clustering_done = get_XY_repeats(patches, cluster=cluster_repeats)
    
    # Get binned Y2 activity 
    if session['stim_order'] == 'random':
        ABB_patches, BAA_patches, _, _ = get_XYY_patches(session, precede_XY=True)
    elif session['stim_order'] == 'pseudorandom':
        ABB_patches, BAA_patches, _, _ = get_XYY_patches(session, precede_XY=False)

    if condition == 'AB':
        XYY_patches = ABB_patches
    elif condition == 'BA':
        XYY_patches = BAA_patches

    binned_Y2_activity = get_binned_Y2_activity(neurons, dF, session, XYY_patches, bins=bins, zscoring=zscoring)
    
    # Plotting
    x = XY_repeats

    for cell in neurons:
        if axes is None:
            fig = plt.figure(figsize=(10,4))
            gs = plt.GridSpec(1, 3, width_ratios=[5, 3, 3])  
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1])
            ax3 = fig.add_subplot(gs[0,2])
        else:
            ax1, ax2, ax3 = axes
            fig = ax1.figure 

        # 1. Plot activity vs XY repeats
        ax1.scatter(x, y_data[cell], alpha=0.7, s=40, color='darkblue')

        # Regression line
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = results['intercepts'][cell] + results['slopes'][cell] * x_fit

        ax1.plot(x_fit, y_fit, linewidth=2, color='darkblue')

        ax1.set_xlabel('Number of preceding XY repeats')
        ax1.set_ylabel(f'mean {condition[-1]}2 activity')
        ax1.set_xticks(x)

        ax1.set_title(
            f'slope = {results['slopes'][cell]:.3g}, '
            f'r = {results['rvalues'][cell]:.3g}, '
            f'p = {results['pvalues'][cell]:.3g}'
        )

        # 2. Plot permutation slopes distribution
        ax2.hist(
            results['slopes_shuffled'][cell],
            bins=30,
            density=True,
            alpha=0.7,
            color='darkblue'
        )

        # Observed slope
        ax2.axvline(
            results['slopes'][cell],
            linestyle='--',
            linewidth=2,
            color='black'
        )

        # Zero line
        ax2.axvline(0, linestyle=':', linewidth=1, color='darkblue')

        ax2.set_xlabel('Regression slope')
        ax2.set_ylabel('Density')

        ax2.set_title(
            f'Permutation distribution\n'
            f'p = {results["pvalue"][cell]:.3g}'
        )

        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # 3. Heatmap of binned activity
        n_trials = len(binned_Y2_activity[cell])

        XY_repeat_sorting_idx = np.argsort(x, kind='stable')
        sorted_repeats = x[XY_repeat_sorting_idx]
        if sort_heatmap:
            heatmap_data = binned_Y2_activity[cell][XY_repeat_sorting_idx]
            change_rows = np.where(np.diff(sorted_repeats) != 0)[0] + 1

            block_starts = np.concatenate(([0], change_rows))
            block_ends   = np.concatenate((change_rows, [len(sorted_repeats)]))
            block_centers = (block_starts + block_ends) / 2 - 0.5
            block_values  = [int(sorted_repeats[start]) for start in block_starts]

            for r in change_rows:
                ax3.axhline(r - 0.5, color='black', linewidth=0.8, linestyle='--')
            # Indicate number of XY repeats  per block
            right_ax = ax3.secondary_yaxis('right')
            right_ax.set_yticks(block_centers)
            if cluster_repeats and clustering_done:
                right_ax.set_yticklabels(['1', '2', '3-4', '5+'], fontsize=6)
            else:
                right_ax.set_yticklabels(block_values, fontsize=6)
            right_ax.set_ylabel('XY repeats', fontsize=8)

        else:
            heatmap_data = binned_Y2_activity[cell]

        vmax = np.max(heatmap_data)
        vmin = np.min(heatmap_data)
        cax = ax3.imshow(heatmap_data, aspect='auto', cmap='viridis')
        cb = fig.colorbar(cax, ax=ax3, label='dF/F', ticks=[vmin, vmax], pad=0.3)       
        cb.ax.set_yticklabels([f"{vmin:.1f}", f"{vmax:.1f}"])
        cb.ax.yaxis.labelpad = -10

        if condition == 'AB':
            ax3.set_title(f'B2')
        elif condition == 'BA':
            ax3.set_title(f'A2')
        ax3.set_yticks([0, n_trials-1])
        ax3.set_yticklabels([1, n_trials])
        ax3.set_xticks([0, bins-1])
        ax3.set_xticklabels([0, bins])
        ax3.set_xlabel('Time bins')
        
        plt.suptitle(f'{condition}: neuron {cell}') 
        plt.tight_layout()

        if save_plot:
            if plot_dir == '':
                plot_dir = session['save_dir']
            condition_save_path = os.path.join(plot_dir, condition)
            os.makedirs(condition_save_path, exist_ok=True)
            plt.savefig(condition_save_path + f'/{data_type}_neuron{cell}.png', dpi=300)

        if len(neurons) > 100:
            plt.close(fig)

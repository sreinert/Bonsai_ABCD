import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
import importlib
from pathlib import Path
import argparse
from collections import Counter
from joblib import Parallel, delayed

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import preprocessing.parse_session_functions_cohort1_sc as parse_session_functions

parser = argparse.ArgumentParser(description="Fit behaviour model for sequence compression cohort1.")
parser.add_argument('--mouse', type=str, default='03', help="The mouse ID (e.g. '03')")
parser.add_argument('--num_Ys', type=int, default='3', help="The number of Ys in the rule (e.g. '3')")
parser.add_argument('--grid_size', type=int, default='10', help="The number of values to sample for each param to be fit (e.g. '10')")

args = parser.parse_args()

mouse = args.mouse 
num_Ys = args.num_Ys
grid_size = args.grid_size

if Path("/ceph").exists():
    ROOT = "/ceph/mrsic_flogel/public/projects"
else:
    ROOT = "/Volumes/mrsic_flogel/public/projects"

save_root = f"{ROOT}/AtAp_20260119_SequenceCompression/behaviour_modelling"
if not os.path.exists(save_root):
    os.makedirs(save_root, exist_ok=True)

def find_X_Y_distance_and_positions(x_entry_idx, y_entry_idx, num_Ys, reward_positions=None, rewarded_Xs=True):
    
    if rewarded_Xs:
        if reward_positions is None:
            raise ValueError('Please provide the rewarded positions')
        
        XX_diff = np.zeros((len(reward_positions) - 1), dtype=int)
        XY_diff = np.zeros((len(reward_positions) - 1, num_Ys), dtype=int)
        X_positions = np.zeros((len(reward_positions) - 1), dtype=int)
        Y_positions = np.zeros((len(reward_positions) - 1, num_Ys), dtype=int)

        for i, pos in enumerate(reward_positions[:-1]):
            mask = (np.round(x_entry_idx, 1) > np.round(pos, 1)) & (np.round(x_entry_idx, 1) <= np.round(reward_positions[i + 1], 1))
            following_X = x_entry_idx[mask][0]
            X_positions[i] = following_X
            XX_diff[i] = np.round(following_X - pos)

            # Keep Ys from current X (or reward) up to the next X
            following_Ys = y_entry_idx[(y_entry_idx > pos) & (y_entry_idx < following_X)]
            for j in range(num_Ys):
                Y_positions[i, j] = following_Ys[j]
                XY_diff[i, j] = np.round(Y_positions[i, j] - pos)

    else:
        XX_diff = np.zeros((len(x_entry_idx) - 1), dtype=int)
        XY_diff = np.zeros((len(x_entry_idx) - 1, num_Ys), dtype=int)
        X_positions = np.zeros((len(x_entry_idx) - 1), dtype=int)
        Y_positions = np.zeros((len(x_entry_idx) - 1, num_Ys), dtype=int)

        for i, pos in enumerate(x_entry_idx[:-1]):
            mask = (np.round(x_entry_idx, 1) > np.round(pos, 1)) & (np.round(x_entry_idx, 1) <= np.round(x_entry_idx[i + 1], 1))
            following_X = x_entry_idx[mask][0]
            X_positions[i] = following_X
            XX_diff[i] = np.round(following_X - pos)

            # Keep Ys from current X (or reward) up to the next X
            following_Ys = y_entry_idx[(y_entry_idx > pos) & (y_entry_idx < following_X)]
            for j in range(num_Ys):
                Y_positions[i, j] = following_Ys[j]
                XY_diff[i, j] = np.round(Y_positions[i, j] - pos)

    return XX_diff, XY_diff, X_positions, Y_positions

def create_corridor(num_Ys, LM_SIZE=3, distances=[3,16], n_laps=2000, plot=False):
    D = np.arange(distances[0], distances[1])

    n_laps = 2000   # pattern repetitions
    num_Xs = 1
    # num_Ys = 3
    pattern = np.array([1] * num_Xs + [0] * num_Ys)

    num_lms = num_Xs + num_Ys
    X_idx = np.where(pattern == 1)[0]
    Y_idx = np.where(pattern == 0)[0]

    # Store the positions of the landmarks 
    X = np.zeros((n_laps, 2, num_Xs))
    Y = np.zeros((n_laps, 2, num_Ys))

    # Pre-select distances randomly
    d_list = [np.random.choice(D, size=n_laps) for _ in range(num_lms)]

    # Pre-define corridor
    total_length = np.sum(np.concatenate([d_list[i] for i in range(num_lms)]) + num_lms * LM_SIZE)
    corridor = np.zeros(total_length, dtype=int)

    # Position landmarks and compute the corridor 
    start = 0
    for i in range(num_lms):
        for j in range(n_laps):
            d = d_list[i][j]
            corridor[start + d : start + d + LM_SIZE] = 1
            start += d + LM_SIZE

    # Assign landmark types 
    lm_entries = np.where(np.diff(corridor) > 0)[0] + 1

    x_counter = 0 
    y_counter = 0
    for i in range(num_lms):
        entries = lm_entries[i::num_lms]

        if i in X_idx:
            for j, entry in enumerate(entries):
                lm = entries[j]
                X[j, 0, x_counter] = entry
                X[j, 1, x_counter] = entry + LM_SIZE

            x_counter += 1

        elif i in Y_idx:
            for j, entry in enumerate(entries):
                lm = entries[j]
                Y[j, 0, y_counter] = entry
                Y[j, 1, y_counter] = entry + LM_SIZE

            y_counter += 1

    X = np.array(X, dtype=int)
    Y = np.array(Y, dtype=int)

    # Find landmark indices
    x_idx = []
    for i in range(num_Xs):
        for entry, exit in X[:,:,i]:
            x_idx.extend(np.arange(entry, exit).astype(int))

    y_idx = []
    for i in range(num_Ys):
        for entry, exit in Y[:,:,i]:
            y_idx.extend(np.arange(entry, exit+1))

    x_idx = np.array(x_idx)
    y_idx = np.array(y_idx)

    x_entry_idx = np.sort(np.concatenate([lm_entries[i::num_lms] for i in X_idx]))
    y_entry_idx = np.sort(np.concatenate([lm_entries[i::num_lms] for i in Y_idx]))

    lm_idx = np.sort(np.unique(np.concatenate([x_idx, y_idx]))).astype(int)

    # Measure the mean distance between XX and each XY
    if num_Xs == 1:
        XX_dist, XY_dist, _, _ = find_X_Y_distance_and_positions(x_entry_idx, y_entry_idx, num_Ys, rewarded_Xs=False)

        # ---- Plot histograms ----
        if plot:
            fig, ax = plt.subplots(1, num_lms, figsize=(10,3))

            for i in range(num_Ys):
                XY_hist = Counter(XY_dist[:,i])

                values = np.array(list(XY_hist.keys()))
                counts = np.array(list(XY_hist.values()))
                ax[i].bar(values, counts, width=0.8)
                ax[i].set_xticks([values.min(), values.max()])
                ax[i].set_yticks([counts.min(), counts.max()])
                ax[i].tick_params(labelsize=7)
                ax[i].spines[['top', 'right']].set_visible(False)
                ax[i].set_xlabel(f'XY{i+1} distances')

            XX_hist = Counter(XX_dist)

            values = np.array(list(XX_hist.keys()))
            counts = np.array(list(XX_hist.values()))
            ax[num_Ys].bar(values, counts, width=0.8)
            ax[num_Ys].set_xticks([values.min(), values.max()])
            ax[num_Ys].set_yticks([counts.min(), counts.max()])
            ax[num_Ys].tick_params(labelsize=7)
            ax[num_Ys].spines[['top', 'right']].set_visible(False)
            ax[num_Ys].set_xlabel('XX distances')

            plt.suptitle(f'n_lms={num_lms * n_laps}')
        
    print("Mean reward-reward distance", np.round(np.mean(np.diff(x_entry_idx))))

    return corridor, pattern, lm_idx, x_idx, y_idx, x_entry_idx, y_entry_idx, XX_dist, XY_dist

def calc_distance_from_rew_p_lick(x_entry_idx, y_entry_idx, lick_position, reward_positions, lm_size=3, num_Ys=1, plot=True, ax1=None, ax2=None):
    '''
    1. Probability of first lick at each landmark type per distance bin from X
       (probabilities sum to 1 within each distance bin)
    2. Number of first licks per landmark type per distance bin
    3. Number of trials per landmark type per distance bin
    4. Number of available trials (where no other lm was licked before) per landmark type per distance bin
    '''

    XX_diff, XY_diff, X_positions, Y_positions = find_X_Y_distance_and_positions(x_entry_idx, y_entry_idx, num_Ys, reward_positions, rewarded_Xs=True)

    following_positions = np.array([
        np.sort(np.concatenate([np.atleast_1d(X_positions[i]), np.atleast_1d(Y_positions[i])]))
        for i in range(len(X_positions))
    ])
    
    # Find where first lick after a reward occurred
    lm_licked = np.zeros_like(following_positions)
    for i, positions in enumerate(following_positions):
        for j, pos in enumerate(positions):
            if np.any((lick_position >= pos) & (lick_position <= (pos + lm_size))):
                lm_licked[i, j] = 1
                break
    
    # Bin distances
    all_distances = np.concatenate([XX_diff, XY_diff.flatten()])
    bins = np.linspace(np.min(all_distances), np.max(all_distances), 20)

    bin_idx = []
    for i in range(num_Ys):
        bin_idx.append(np.digitize(XY_diff[:, i], bins))
    bin_idx.append(np.digitize(XX_diff, bins))
    
    num_lms = len(bin_idx)

    # Count how many trials occurred at each distance bin 
    distance_bin_counts = np.array([
        [np.sum(bin_idx[i] == b) for b in range(1, len(bins))] for i in range(num_lms)
    ], dtype=float)

    valid_bins = distance_bin_counts > 0

    distance_bin_counts[distance_bin_counts == 0] = np.nan

    # Count trials unattempted until each lm 
    avail_distance_bin_counts = np.array([
        [np.sum((bin_idx[i] == b) & (~np.any(lm_licked[:, :i], axis=1))) 
            for b in range(1, len(bins))] for i in range(num_lms)], dtype=int)
    
    # Count first licks per lm type per distance bin
    lm_lick_counts = []
    for i in range(num_lms):
        lm_lick_counts.append(np.array([np.sum(lm_licked[:,i][bin_idx[i] == b])
            for b in range(1, len(bins))], dtype=int)) 
    lm_lick_counts = np.array(lm_lick_counts)

    # Total licks per bin across all landmark types
    total_lick_counts = np.sum(lm_lick_counts, axis=0)

    # Per distance bin, count probability of first lick at each landmark type
    lm_lick_prob = np.array([
        np.divide(
            lm_lick_counts[i],
            total_lick_counts,
            out=np.full_like(total_lick_counts, np.nan, dtype=float),
            where=total_lick_counts > 0
        )
        for i in range(num_lms)
    ])

    avail_lm_lick_prob = np.divide(
        lm_lick_counts,
        avail_distance_bin_counts,
        out=np.full_like(avail_distance_bin_counts, np.nan, dtype=float),
        where=avail_distance_bin_counts > 0
    )

    avail_lm_lick_prob = avail_lm_lick_prob.copy()
    avail_lm_lick_prob[valid_bins & np.isnan(avail_lm_lick_prob)] = 0
    
    ## Plotting
    if plot:
        with mpl.rc_context({
            'axes.labelsize': 18,      # x and y axis labels
            'xtick.labelsize': 14,     # x tick labels
            'ytick.labelsize': 14,     # y tick labels
            'legend.fontsize': 18,     # legend text
        }):
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # 1. Plot probability of first lick per landmark type per bin
            if ax1 is None:
                fig, ax = plt.subplots(figsize=(6,4))
            else:
                fig = ax1.figure

            if num_Ys == 1:
                y_labels = ["Y"]
            else:
                y_labels = [f"Y{i+1}" for i in range(num_Ys)]

            labels = y_labels + ["X"]

            color_map = {
                "Y": "orange",
                "Y1": "orange",
                "Y2": "gold",
                "Y3": "brown",
                "X": "darkblue",
            }

            plot_order = [num_Ys] + list(range(num_Ys)) 

            for i in plot_order:
                label = labels[i]

                ax1.plot(
                    bin_centers,
                    avail_lm_lick_prob[i],
                    marker='o',
                    label=label,
                    color=color_map[label], 
                    alpha=0.5,
                    linewidth=3,
                )

                ax1.set_ylim([0,1.1])
                ax1.set_yticks([0,0.5,1])
                ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
                ax1.set_ylabel('Prob first lick')

            # 2. Number of trials per distance bin
            if ax2 is not None:
                for i in plot_order:
                    label = labels[i]

                    ax2.plot(
                        bin_centers,
                        distance_bin_counts[i],
                        marker='o',
                        label=label,
                        color=color_map[label],
                        alpha=0.5,
                        linewidth=3,
                    )
                ax2.set_ylabel('Number of trials')
                y_min = np.nanmin(distance_bin_counts).astype(int)
                y_max = np.nanmax(distance_bin_counts).astype(int)
                y_mid = np.round((y_min + y_max) / 2).astype(int)

                ax2.set_yticks([y_min, y_mid, y_max])
                
            for ax in (a for a in (ax1, ax2) if a is not None):
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                xmin = np.round(np.min(all_distances))
                xmax = np.round(np.max(all_distances))
                ax.set_xticks([xmin, xmax])

                ax.set_xlabel('Distance X → ')
                legend = ax.legend(frameon=False, handlelength=0, handletextpad=0, markerscale=0)
                for handle in legend.legend_handles:
                    handle.set_visible(False)

                # color legend text to match plotted lines
                for text, color in zip(
                    legend.get_texts(),
                    ['darkblue'] + ['orange', 'gold', 'brown'][:num_Ys]
                ):
                    text.set_color(color)
                    # text.set_fontweight('bold')

            return XX_diff, XY_diff, avail_lm_lick_prob, fig, ax1, ax2
    
    else:
        return XX_diff, XY_diff, avail_lm_lick_prob, None, None, None

def simulate_strategy4(x_entry_idx, y_entry_idx, x_idx, lm_idx, corridor, LM_SIZE, num_Ys, 
                       d, beta_switch, beta_lick, sigma_d=0.05, sigma_period=0.5, beta_period=0.01, 
                       p0=0.05, n_runs=20):
    """
    Run one complete simulation. The data returned are exactly the same as the mouse data. 

    Returns
    -------
    bin_centers
    lick_prob
    """

    probs = [] 

    for _ in range(n_runs):
        effective_sigma = sigma_period

        periods = []
        licks = np.zeros_like(corridor)
        rewards = np.zeros_like(corridor)

        # (A) Discover period (determined with 2 samples)
        reward_found = False
        end_reached = False

        rew_counter = 0
        non_rew_counter = 0
        rew_idx = []

        current_lm = lm_idx[0]
        licks[current_lm] = 1

        while reward_found == False: 
            licks[current_lm] = 1  # lick everywhere until 2 rewards are found

            if current_lm in x_idx:
                rew_idx.append(current_lm)
                rew_counter += 1
                rewards[current_lm] = 1
            else:
                non_rew_counter += 1
            next_lm = lm_idx[lm_idx > (current_lm + LM_SIZE)]
            
            if rew_counter == 2:
                reward_found = True
            
            # advance 
            current_lm = next_lm[0]

        # Get the period
        period = non_rew_counter + 1
        sampled_period = max(1, int(np.round(np.random.normal(period, effective_sigma))))
        periods.append(sampled_period)

        # (B) If below preferred distance, follow periodic licking pattern 
        current_lm = rew_idx[-1]
        next_lm = lm_idx[np.where(np.diff(lm_idx) > 1)[0] + 1]
        next_lm = next_lm[next_lm > current_lm]

        dist_since_reward = 0
        skip = 0
        dist_thres = False
        periodic_regime = False

        # distance threshold estimate - reset once reward is found
        # d_hat = d * np.exp(np.random.normal(0, sigma_d))

        while len(next_lm) > 0:
            # distance threshold estimate
            d_hat = d * np.exp(np.random.normal(0, sigma_d))

            # Entering each regime is probabilistic and depends on distance from last reward
            p_distance = 1 / (1 + np.exp(-beta_switch * (dist_since_reward - d_hat)))
            p_periodic = 1 - p_distance

            ### --- Periodic regime --- ###
            if np.random.rand() < p_periodic:
                # periodic_regime = True

                # while periodic_regime:
                skip += 1 

                if skip >= sampled_period:
                    # on the way to the current estimated period 
                    while len(next_lm) > 0:
                        if len(next_lm) <= sampled_period - 1:
                            # periodic_regime = False
                            end_reached = True
                            break

                        target_lm = next_lm[sampled_period - 1]

                        # check distance from last reward
                        while dist_since_reward < d_hat:
                            step = next_lm[0] - current_lm
                            dist_since_reward += step

                            # advance
                            current_lm = next_lm[0]
                            next_lm = next_lm[next_lm > current_lm]

                            # lick at target lm  
                            if current_lm == target_lm:
                                next_lick = current_lm
                                licks[next_lick] = 1
                                break

                        if dist_since_reward >= d_hat:
                            dist_thres = True
                            # periodic_regime = False
                            break 

                        # stochastically lick until reward is eventually found
                        while next_lick not in x_idx:
                            next_lm = next_lm[next_lm > current_lm]

                            if len(next_lm) <= sampled_period - 1:
                                # periodic_regime = False
                                end_reached = True
                                break

                            step = next_lm[0] - current_lm
                            dist_since_reward += step
                            current_lm = next_lm[0]

                            p_visit = p0 + (1 - p0) * (1 - np.exp(-beta_lick * dist_since_reward / d_hat))
                            if np.random.rand() < p_visit:
                                next_lick = current_lm
                                licks[next_lick] = 1
                                
                        if end_reached:
                            # periodic_regime = False
                            break 

                        # reward found
                        rewards[next_lick] = 1

                        # sample new period and distance threshold
                        if dist_since_reward == 0:
                            effective_sigma = sigma_period
                        else:
                            effective_sigma = beta_period * dist_since_reward
                            
                        sampled_period = max(1, int(np.round(np.random.normal(period, effective_sigma))))
                        periods.append(sampled_period)
                        d_hat = d * np.exp(np.random.normal(0, sigma_d))

                        # reset values
                        dist_since_reward = 0
                        skip = 0     
                        # periodic_regime = False                   

                        # update future landmarks
                        next_lm = next_lm[next_lm > current_lm]
                    
                    if end_reached:
                        # periodic_regime = False
                        break

                # advance
                if not dist_thres:
                    next_lm = next_lm[next_lm > current_lm]
                    if len(next_lm) > 0:
                        step = next_lm[0] - current_lm
                        dist_since_reward += step
                        current_lm = next_lm[0]

            ### --- Distance-triggered regime: stochastically lick everything until reward --- ###
            else:
                while len(next_lm) > 0:
            
                    # stochastically choose whether to lick or not until reward is found
                    p_visit = p0 + (1 - p0) * (1 - np.exp(-beta_lick * dist_since_reward / d_hat))
                    # p_visit = np.clip(p_visit, 0.05, 0.99)

                    if np.random.rand() < p_visit:
                        next_lick = current_lm
                        licks[next_lick] = 1

                        # reward found
                        if next_lick in x_idx:
                            rewards[next_lick] = 1

                            # sample new period and distance threshold
                            if dist_since_reward == 0:
                                effective_sigma = sigma_period
                            else:
                                effective_sigma = beta_period * dist_since_reward
                            sampled_period = max(1, int(np.round(np.random.normal(period, effective_sigma))))
                            periods.append(sampled_period)
                            d_hat = d * np.exp(np.random.normal(0, sigma_d))

                            current_lm = next_lick
                            dist_since_reward = 0
                            skip = 0
                            next_lm = next_lm[next_lm > next_lick]
                            break

                    # advance 
                    next_lm = next_lm[next_lm > current_lm]
                    if len(next_lm) > 0:
                        step = next_lm[0] - current_lm
                        dist_since_reward += step
                        current_lm = next_lm[0]
                
        licks_idx = np.where(licks == 1)[0]
        for i in licks_idx:
            if i in x_idx:
                rewards[i] = 1
        rewards_idx = np.where(rewards == 1)[0]

        # Calculate the probability of licking
        XX_diff, XY_diff, lick_prob, _, _, _ = calc_distance_from_rew_p_lick(x_entry_idx, y_entry_idx, licks_idx, rewards_idx, lm_size=LM_SIZE, num_Ys=num_Ys, plot=False)
        probs.append(lick_prob) 

    # Calculate bin centers - using values from last run
    all_distances = np.concatenate([XX_diff, XY_diff.flatten()])
    bins = np.linspace(np.min(all_distances), np.max(all_distances), 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    return bin_centers, np.mean(probs, axis=0)

def loss(x_entry_idx, y_entry_idx, x_idx, lm_idx, corridor, LM_SIZE, num_Ys, 
         d, beta_switch, beta_lick, beta_period, p_mouse, p0=0.05, n_runs=20):

    _, p_model = simulate_strategy4(
        x_entry_idx=x_entry_idx,
        y_entry_idx=y_entry_idx,
        x_idx=x_idx,
        lm_idx=lm_idx,
        corridor=corridor,
        LM_SIZE=LM_SIZE,
        num_Ys=num_Ys,
        d=d,
        beta_switch=beta_switch,
        beta_lick=beta_lick,
        beta_period=beta_period,
        p0=p0,
        n_runs=n_runs,
    )

    assert p_model.shape == p_mouse.shape

    mse = np.nanmean([
        np.nanmean((pm - pd)**2)
        for pm, pd in zip(p_model, p_mouse)
    ])

    return mse

def evaluate_params(
    params,
    x_entry_idx,
    y_entry_idx,
    x_idx,
    lm_idx,
    corridor,
    LM_SIZE,
    num_Ys,
    p_mouse,
):
    d, beta_switch, beta_lick, beta_period = params

    L = loss(
        x_entry_idx,
        y_entry_idx,
        x_idx,
        lm_idx,
        corridor,
        LM_SIZE,
        num_Ys,
        d,
        beta_switch,
        beta_lick,
        beta_period,
        p_mouse,
        n_runs=10,
    )

    return {
        "d": d,
        "beta_switch": beta_switch,
        "beta_lick": beta_lick,
        "beta_period": beta_period,
        "loss": L,
    }

def main():
    #%% Create corridor 
    LM_SIZE = 3
    corridor, pattern, lm_idx, x_idx, y_idx, x_entry_idx, y_entry_idx, XX_dist, XY_dist = create_corridor(num_Ys, LM_SIZE=LM_SIZE, distances=[3,16], n_laps=2000)

    # Find all distances from rewards
    d_values = np.unique(np.concatenate([XX_dist, XY_dist.flatten()])).astype(int)
    d_values = np.arange(d_values[0], d_values[-1]+1)
    first_overlap = np.where(d_values == np.intersect1d(np.unique(XY_dist.flatten()), np.unique(XX_dist))[0])[0][0]

    #%% Load behaviour data 
    p = pattern.tolist()
    if p == [1,0,0,0]:
        cond = 'abbb'
    elif p == [1,0,0]:
        cond = 'abb'
    elif p == [1,0]:
        cond = 'abab'

    data_path = f'{ROOT}/AtAp_20260119_SequenceCompression/processed/sub-{mouse}/analysis/{cond}_sessions'
    behav_data = np.load(os.path.join(data_path, "first_lick_data.npz"), allow_pickle=True)

    # Calculate p(lick) for the mouse in this rule 
    licks = behav_data['lm_lick_counts']
    opportunities = behav_data['avail_distance_bin_counts']

    p_mouse = np.divide(
        licks,
        opportunities,
        out=np.zeros_like(licks, dtype=float),
        where=opportunities > 0
    )

    #%% Perform grid search and optimization 
    d_grid = d_values[::5]
    beta_switch_grid = np.linspace(0.01, 10, grid_size)
    beta_lick_grid = np.linspace(0.05, 10, grid_size)
    beta_period_grid = np.linspace(0.01, 10, grid_size)

    param_grid = [
        (d, beta_switch, beta_lick, beta_period)
        for d in d_grid
        for beta_switch in beta_switch_grid
        for beta_lick in beta_lick_grid
        for beta_period in beta_period_grid
    ]

    results = Parallel(
        n_jobs=4,
        verbose=10,
    )(
        delayed(evaluate_params)(
            params,
            x_entry_idx,
            y_entry_idx,
            x_idx,
            lm_idx,
            corridor,
            LM_SIZE,
            num_Ys,
            p_mouse,
        )
        for params in param_grid
    )

    best_result = min(
        results,
        key=lambda x: x["loss"]
    )

    print("Best parameters:")
    print(best_result)

    #%% Run simulation using best fit parameters
    # best_d, best_beta_switch, best_beta_lick, best_beta_period = best_params

    x_model, p_model = simulate_strategy4(
        x_entry_idx,
        y_entry_idx,
        x_idx,
        lm_idx,
        corridor,
        LM_SIZE,
        num_Ys,
        d=best_result["d"],
        beta_switch=best_result["beta_switch"],
        beta_lick=best_result["beta_lick"],
        beta_period=best_result["beta_period"],
        n_runs=10,
    )

    # Plot mouse vs model 
    with mpl.rc_context({
        'axes.labelsize': 18,      # x and y axis labels
        'xtick.labelsize': 14,     # x tick labels
        'ytick.labelsize': 14,     # y tick labels
        'legend.fontsize': 18,     # legend text
        'axes.titlesize': 18,
        'figure.titlesize': 18,
    }):

        fig, axs = plt.subplots(1, 2, figsize=(10,5), sharey=True)
        axs = axs.ravel()

        if num_Ys == 1:
            b_labels = ["B"]
            y_labels = ["Y"]
        else:
            b_labels = [f"B{i+1}" for i in range(num_Ys)]
            y_labels = [f"Y{i+1}" for i in range(num_Ys)]

        mouse_labels = b_labels + ["A"]
        model_labels = y_labels + ["X"]

        color_map = {
            "Y": "orange",
            "Y1": "orange",
            "Y2": "gold",
            "Y3": "brown",
            "X": "darkblue",
            "B": "orange",
            "B1": "orange",
            "B2": "gold",
            "B3": "brown",
            "A": "darkblue",
        }

        plot_order = [num_Ys] + list(range(num_Ys)) 

        for i in plot_order:
            mouse_label = mouse_labels[i]
            model_label = model_labels[i]

            valid_mask = opportunities[i] != 0
            axs[0].plot(
                behav_data['bin_centers'][valid_mask],
                p_mouse[i][valid_mask],
                marker='o',
                label=mouse_label,
                color=color_map[mouse_label], 
                alpha=1,
                linewidth=3,
            )
            axs[0].set_ylabel('Prob first lick')
            axs[0].set_title('Mouse')
            axs[0].set_xlabel('Distance A → ')
            all_distances = np.concatenate([behav_data["A_A_diff"], np.concatenate(behav_data['A_B_diff'].flatten())])
            xmin = np.round(np.min(all_distances))
            xmax = np.round(np.max(all_distances))
            axs[0].set_xticks([xmin, xmax])

            axs[1].plot(
                x_model,
                p_model[i],
                marker='o',
                label=model_label,
                color=color_map[model_label], 
                alpha=0.5,
                linewidth=3,
            )
            axs[1].set_title('Model')
            axs[1].set_xlabel('Distance X → ')
            all_distances = np.concatenate([XX_dist, XY_dist.flatten()])
            xmin = np.round(np.min(all_distances))
            xmax = np.round(np.max(all_distances))
            axs[1].set_xticks([xmin, xmax])

        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim([0,1.1])
            ax.set_yticks([0,0.5,1])
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            
            legend = ax.legend(frameon=False, handlelength=0, handletextpad=0, markerscale=0)
            for handle in legend.legend_handles:
                handle.set_visible(False)

            # color legend text to match plotted lines
            for text, color in zip(
                legend.get_texts(),
                ['darkblue'] + ['orange', 'gold', 'brown'][:num_Ys]
            ):
                text.set_color(color)

        plt.suptitle(f"{parse_session_functions.format_condition_label(cond=cond)} sub-{mouse}\n\
                d={best_result["d"]}, beta_switch={np.round(best_result["beta_switch"],2)}, beta_lick={np.round(best_result["beta_lick"],2)}, beta_period={np.round(best_result["beta_period"],2)}")
        plt.tight_layout()

        mouse_save_path = os.path.join(save_root, f'strategy4', 'mouse_fits', f'sub-{mouse}')
        if not os.path.exists(mouse_save_path):
            os.makedirs(mouse_save_path, exist_ok=True)

        plt.savefig(mouse_save_path + f'/{parse_session_functions.format_condition_label(pattern=pattern)}.png')
        plt.savefig(mouse_save_path + f'/{parse_session_functions.format_condition_label(pattern=pattern)}.svg', format="svg")


if __name__ == "__main__":
    main()
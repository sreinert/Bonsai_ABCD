import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import re, os, sys
from session_functions.utils import *

np.set_printoptions(suppress=True, precision=2)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))

def plot_ethogram(session):
    sess_dataframe = session.dataframe 

    lick_position = sess_dataframe['Position'].values[sess_dataframe['Licks'].values > 0]
    lick_times = sess_dataframe.index[sess_dataframe['Licks'].values > 0]
    reward_times = sess_dataframe.index[sess_dataframe['Rewards'].notna()]
    reward_positions = sess_dataframe['Position'].values[sess_dataframe['Rewards'].notna()]
    if 'LM_Count' in sess_dataframe.columns:
        release_df = session.estimate_lm_events()
    else:
        release_df = session.estimate_release_events()
    release_times = release_df.index.tolist() # time
    release_times = release_times[1:]  # remove first release for plotting because sometimes the timestamp is NaN
    release_positions = release_df["Position"].tolist()
    release_positions = release_positions[1:]  # remove first release for plotting because sometimes the timestamp is NaN   

    plt.figure(figsize=(12, 6))
    plt.plot(sess_dataframe.index, sess_dataframe['Treadmill']/np.max(sess_dataframe['Treadmill']), label='Treadmill Speed', color='purple')
    plt.plot(sess_dataframe.index, sess_dataframe['Position']/np.max(sess_dataframe['Position']), label='Position', color='blue')
    plt.plot(lick_times, lick_position/np.max(sess_dataframe['Position']), marker='o', linestyle='', label='Licks', color='orange')
    plt.plot(release_times, release_positions/np.max(sess_dataframe['Position']), marker='o', linestyle='', label='Releases', color='red')
    plt.plot(reward_times, reward_positions/np.max(sess_dataframe['Position']), marker='o', linestyle='', label='Rewards', color='green')
    plt.plot(sess_dataframe.index, sess_dataframe['Buffer']/np.max(sess_dataframe['Buffer']), label='Analog Buffer', color='black')
    # plt.plot(sess_dataframe.index, sess_dataframe['Lap']/num_laps, label='Laps', color='brown')

    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Session Data Overview')
    plt.legend()
    plt.show()

def plot_psth(ax, mean, sem, color, label):
    x = np.arange(len(mean))
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean + sem, mean - sem, color=color, alpha=0.3)

def compute_psth_pair(behaviour, events, bins):
    speed = behaviour.get_speed_psth(events=events, bins=bins)
    licks = behaviour.get_lick_rate_psth(events=events, bins=bins)
    return speed, licks

def plot_speed_lick_rate_psth(session, behaviour, bins=None):

    if 'LM_Count' in session.dataframe.columns:
        release_df = session.estimate_lm_events()
    else:
        release_df = session.estimate_release_events()

    dt_idx = np.diff(release_df['Index'])
    dt_seconds = release_df.index.to_series().diff().dt.total_seconds().to_numpy()
    
    if bins is None:
        min_dt_idx = np.min(dt_idx)
        min_dt_seconds = np.nanmin(dt_seconds)
        window_seconds = np.round(min_dt_seconds * 2, 1)
        bins = int(min_dt_idx * 2)
    else:
        window_seconds = np.round(dt_seconds[1:] / dt_idx * bins, 1)
        window_seconds = window_seconds[~np.isnan(window_seconds)][0]

    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    ax_speed, ax_lick = axes

    # --- Get event indices ---
    if session.sequence == 'full':
        landmarks, lm_idx = session.get_landmarks()

    else:
        # Binary sequence
        A_landmarks, B_landmarks, A_idx, B_idx = session.get_A_B_landmarks()

    # --- Define groups dynamically ---
    groups = {}

    if session.sequence == 'ABAB':
        groups = {
            'A': (A_idx, 'darkblue'),
            'B': (B_idx, 'orange')
        }

    elif session.sequence == 'AABB':
        groups = {
            'A1': (A_idx[::2], 'darkblue'),
            'A2': (A_idx[1::2], 'blue'),
            'B1': (B_idx[::2], 'orange'),
            'B2': (B_idx[1::2], 'gold')
        }

    elif session.sequence == 'ABB':
        groups = {
            'A': (A_idx, 'darkblue'),
            'B1': (B_idx[::2], 'orange'),
            'B2': (B_idx[1::2], 'gold')
        }

    elif session.sequence == 'ABBB':
        groups = {
            'A': (A_idx, 'darkblue'),
            'B1': (B_idx[::3], 'orange'),
            'B2': (B_idx[1::3], 'gold'),
            'B3': (B_idx[2::3], 'brown')
        }

    elif session.sequence == 'AAB':
        groups = {
            'A1': (A_idx[::2], 'darkblue'),
            'A2': (A_idx[1::2], 'blue'),
            'B1': (B_idx, 'orange'),
        }

    else:
        groups = {
            'lm': (lm_idx, 'black')
        }

    # --- Compute + plot ---
    for label, (events, color) in groups.items():

        (mean_s, sem_s), (mean_l, sem_l) = compute_psth_pair(behaviour, events, bins)

        plot_psth(ax_speed, mean_s, sem_s, color, label)
        plot_psth(ax_lick, mean_l, sem_l, color, label)

    ax_speed.axhline(session.settings['velocityThreshold'], linestyle='--', color='grey')

    # --- Styling ---
    for ax in axes:
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.axvspan(bins/2, bins, color='grey', alpha=0.3)
        ax.set_xticks([0, bins/2, bins], labels=[f'{-window_seconds/2:.1f}', 0, f'{window_seconds/2:.1f}'])

    ax_speed.set_title('Speed')
    ax_lick.set_title('Lick rate')

    plt.tight_layout()
    return fig

def plot_speed_psth_distance_groups(session, psth_A, psth_B, distance_groups, bins=300):
    """
    Plot PSTHs for each distance group as subplots.
    Each subplot shows A + all B landmarks.
    """
    if 'LM_Count' in session.dataframe.columns:
        release_df = session.estimate_lm_events()
    else:
        release_df = session.estimate_release_events()

    dt_idx = np.diff(release_df['Index'])
    dt_seconds = release_df.index.to_series().diff().dt.total_seconds().to_numpy()
    window_seconds = np.round(dt_seconds[1:] / dt_idx * bins, 1)
    window_seconds = window_seconds[~np.isnan(window_seconds)][0]

    group_names = list(distance_groups.keys())
    num_groups = len(group_names)

    fig, axes = plt.subplots(1, num_groups, figsize=(5 * num_groups, 4), sharey=True)

    if num_groups == 1:
        axes = [axes]

    # --- colors ---
    colors = {
        'A': 'darkblue',
        'B1': 'orange',
        'B2': 'gold',
        'B3': 'brown'
    }

    x = np.arange(bins)

    # =========================
    # --- LOOP OVER GROUPS ---
    # =========================
    for ax, group in zip(axes, group_names):

        # --- A ---
        mean_A, sem_A = psth_A[group]

        ax.plot(x, mean_A, color=colors['A'], label='A')
        ax.fill_between(x, mean_A + sem_A, mean_A - sem_A,
                        color=colors['A'], alpha=0.3)

        # --- Bs ---
        for i, psth_B_j in enumerate(psth_B):
            label = f'B{i+1}'
            mean_B, sem_B = psth_B_j[group]

            ax.plot(x, mean_B, color=colors[label], label=label)
            ax.fill_between(x, mean_B + sem_B, mean_B - sem_B,
                            color=colors[label], alpha=0.3)

        # --- formatting ---
        dists = distance_groups[group]

        if len(dists) > 0:
            d_min = int(np.min(dists))
            d_max = int(np.max(dists))
            title = f"{group} ({d_min}-{d_max})"
        else:
            title = f"{group}\n(no data)"

        ax.set_title(title)
        ax.axvline(bins // 2, linestyle='--', color='grey', alpha=0.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.axvspan(bins/2, bins, color='grey', alpha=0.3)
        ax.set_xticks([0, bins/2, bins], labels=[f'{-window_seconds/2:.1f}', 0, f'{window_seconds/2:.1f}'])

    axes[0].set_ylabel('Speed')
    for ax in axes:
        ax.axhline(session.settings['velocityThreshold'], linestyle='--', color='grey')
        ax.set_xlabel('Time (bins)')
        ax.legend(frameon=False)

    # plt.tight_layout()
    return fig

def plot_transition_matrix(session, performance):
    from matplotlib.colors import Normalize

    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = session.find_targets_distractors()
    
    # Decide if matrix permuation is needed for plotting (currently only for AAB)
    perm = None

    if len(session.reward_seq) == 3:
        A_landmarks = list(np.where(session.reward_seq == 0)[0])
        if len(A_landmarks) == 2:   # AAB
            perm = np.array([target_id[0], target_id[1], distractor_id[0]])

    transition_matrix, lick_tm, ideal_tm = performance.calc_transition_matrix()
    
    if perm is not None:
        transition_matrix = transition_matrix[np.ix_(perm, perm)]
        lick_tm = lick_tm[np.ix_(perm, perm)]
        ideal_tm = ideal_tm[np.ix_(perm, perm)]

    label_map = {}
    for i, tid in enumerate(target_id, start=1):
        label_map[tid] = f"A{i}"
    for i, did in enumerate(distractor_id, start=1):
        label_map[did] = f"B{i}"
    labels = [label_map[i] for i in lm_ids]
    labels = [labels[i] for i in perm] if perm is not None else labels
    
    global_max = max(np.max(lick_tm), np.max(ideal_tm))

    with mpl.rc_context({
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 10,
    }):
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))

        ims = []
        ims.append(axes[0].imshow(transition_matrix, cmap='viridis', interpolation='none',
                                vmin=0, vmax=np.max(transition_matrix)))
        axes[0].set_title('Stimulus Transition Matrix')

        ims.append(axes[1].imshow(lick_tm, cmap='viridis', interpolation='none',
                                vmin=0, vmax=global_max))
        axes[1].set_title('Lick Transition Matrix')

        ims.append(axes[2].imshow(ideal_tm, cmap='viridis', interpolation='none',
                                vmin=0, vmax=global_max))
        axes[2].set_title('Ideal Transition Matrix')

        for ax in axes:
            ax.set_xlabel('Next Landmark ID')
            # ax.set_ylabel('Current Landmark ID')
            ax.set_xticks(range(len(lm_ids)))
            ax.set_yticks(range(len(lm_ids)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
        axes[0].set_ylabel('Current Landmark ID')

        for ax, im in zip(axes, ims):
            cbar = fig.colorbar(im, ax=ax, shrink=0.5, aspect=20, pad=0.05)
            cbar.set_ticks([im.norm.vmin, im.norm.vmax])

        plt.tight_layout()

    return fig

def plot_distance_transition_matrix(session, performance):

    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = session.find_targets_distractors()
    transition_matrix, lick_tm, ideal_tm = performance.calc_distance_transition_matrix()
    
    label_map = {}
    for i, tid in enumerate(target_id, start=1):
        label_map[tid] = f"A{i}"
    for i, did in enumerate(distractor_id, start=1):
        label_map[did] = f"B{i}"
    
    labels = [label_map[i] for i in lm_ids]

    with mpl.rc_context({
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 10,
    }):
        for d in transition_matrix.keys():
            fig, axes = plt.subplots(1, 3, figsize=(12, 5))

            ims = []
            ims.append(axes[0].imshow(transition_matrix[d], cmap='viridis', interpolation='none',
                                        vmin=0, vmax=np.max(transition_matrix[d])))
            axes[0].set_title('Stimulus Transition Matrix')

            ims.append(axes[1].imshow(lick_tm[d], cmap='viridis', interpolation='none',
                                        vmin=0, vmax=np.max(lick_tm[d])))
            axes[1].set_title('Lick Transition Matrix')

            ims.append(axes[2].imshow(ideal_tm[d], cmap='viridis', interpolation='none',
                                        vmin=0, vmax=np.max(ideal_tm[d])))
            axes[2].set_title('Ideal Transition Matrix')

            for ax in axes:
                ax.set_xlabel('Next Landmark ID')
                ax.set_ylabel('Current Landmark ID')
                ax.set_xticks(range(len(lm_ids)))
                ax.set_yticks(range(len(lm_ids)))
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)

            for ax, im in zip(axes, ims):
                cbar = fig.colorbar(im, ax=ax, shrink=0.5, aspect=20, pad=0.05)
                vmin, vmax = im.get_clim()
                cbar.set_ticks([vmin, vmax])
                cbar.set_ticklabels([f'{vmin:.0f}', f'{vmax:.0f}'])
                
            plt.tight_layout()
            if isinstance(d, float):
                fig.suptitle(f'Distance between current and next landmark = {int(d)}')
            else:
                fig.suptitle(f'Distance between current and next landmark = {d}')

    return 

def plot_conditional_matrix(session, performance, n_steps=1):

    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = session.find_targets_distractors()
    transition_prob, control_prob, ideal_prob = performance.calc_conditional_matrix(n_steps)
    
    max_val = max(np.max(transition_prob), np.max(control_prob), np.max(ideal_prob))

    label_map = {}
    for i, tid in enumerate(target_id, start=1):
        label_map[tid] = f"A{i}"
    for i, did in enumerate(distractor_id, start=1):
        label_map[did] = f"B{i}"
    
    xlabels = [label_map[i] for i in lm_ids]
    ylabels = [label_map[i] for i in target_id]

    with mpl.rc_context({
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 10,
    }):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        ims = []
        ims.append(axes[0].imshow(transition_prob, cmap='viridis', interpolation='none',
                                    vmin=0, vmax=max_val))
        axes[0].set_title(f'Transition Probability Matrix') #\n(Licked at {n_steps} lms ahead)')

        ims.append(axes[1].imshow(control_prob, cmap='viridis', interpolation='none',
                                    vmin=0, vmax=max_val))
        axes[1].set_title('Control Probability Matrix\n(Next landmark)')

        ims.append(axes[2].imshow(ideal_prob, cmap='viridis', interpolation='none',
                                    vmin=0, vmax=max_val))
        axes[2].set_title('Ideal Probability Matrix\n(Next A)')

        for ax in axes:
            ax.set_xlabel('Next Landmark ID')
            ax.set_xticks(range(len(lm_ids)))
            ax.set_xticklabels(xlabels)
            ax.set_yticks(range(len(target_id)))
            ax.set_yticklabels(ylabels)
        axes[0].set_ylabel('Landmark ID')
        
        for ax, im in zip(axes, ims):
            cbar = fig.colorbar(im, ax=ax, shrink=0.3, aspect=20, pad=0.05)
            ticks = cbar.get_ticks()
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{i:.1f}" for i in ticks])

        plt.tight_layout()

    return fig

def plot_lick_lm(session, performance):
    target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = session.find_targets_distractors()
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = performance.calc_hit_fa()

    A_landmarks, _, _, _ = session.get_A_B_landmarks()

    was_target = np.zeros(len(lm_id_sequence))
    was_target[A_landmarks] = 1
    was_target = was_target[:,np.newaxis]

    licked_all = licked_all[:,np.newaxis]
    lm_id_sequence = lm_id_sequence[:,np.newaxis]
    fig = plt.figure(figsize=(10,4))
    plt.subplot(3, 1, 1)
    plt.imshow(was_target.T, aspect='auto', cmap='viridis')
    plt.clim(0, 1)
    plt.title('Was Target')

    #invert color map for better visibility
    plt.subplot(3, 1, 2)
    plt.imshow(lm_id_sequence.T, aspect='auto', cmap='viridis_r')
    plt.clim(0, np.max(lm_id_sequence))
    plt.title('Landmark ID')

    plt.subplot(3, 1, 3)
    plt.imshow(licked_all.T, aspect='auto', cmap='viridis')
    plt.clim(0, 1)
    plt.title('Licked All')
    plt.tight_layout()

    return fig


def plot_sw_hit_fa(session, performance, window=10):

    target_positions, distractor_positions, target_id, distractor_id, was_target, lm_id = session.find_targets_distractors()
    hit_rate, fa_rate,d_prime, licked_target, licked_distractor, licked_all, rewarded_all = performance.calc_hit_fa()

    hit_rate_window = np.zeros(len(licked_all)-window)
    false_alarm_rate_window = np.zeros(len(licked_all)-window)
    for i in range(len(licked_all)-window):
        all_window_goals = sum(was_target[i:i+window])
        all_window_distractors = window - all_window_goals
        hit_rate_window[i] = safe_divide(np.sum(licked_all[i:i+window][was_target[i:i+window]==1]), all_window_goals)
        false_alarm_rate_window[i] = safe_divide(np.sum(licked_all[i:i+window][was_target[i:i+window]==0]), all_window_distractors)

    plt.figure(figsize=(10,2))
    plt.plot(hit_rate_window, label='Hit Rate', color='g')
    plt.plot(false_alarm_rate_window, label='False Alarm Rate', color='r')
    plt.xlabel('Landmark')
    plt.ylabel('Rate')
    plt.legend()
    plt.title('Sliding window Hit and False Alarm rates')
    plt.show()

def plot_data(x, y, all_distances, ylabel):
    xmin = np.round(np.min(all_distances))
    xmax = np.round(np.max(all_distances))

    num_Bs = len(y) - 1   # TODO adapt for AAB
    if num_Bs == 1:
        b_labels = ["B"]
    else:
        b_labels = [f"B{i+1}" for i in range(num_Bs)]
    labels = b_labels + ["A"]
    color_map = {
        "B": "orange",      
        "B1": "orange",
        "B2": "gold",
        "B3": "brown",
        "A": "darkblue",
    }

    with mpl.rc_context({
        'axes.labelsize': 18,      # x and y axis labels
        'xtick.labelsize': 14,     # x tick labels
        'ytick.labelsize': 14,     # y tick labels
        'legend.fontsize': 18,     # legend text
    }):
        fig = plt.figure(figsize=(6, 4))

        plot_order = [len(y) - 1] + list(range(len(y) - 1))

        for i in plot_order:
            label = labels[i]
            plt.plot(
                x,
                y[i],
                marker='o',
                label=label,
                color=color_map[label]
            )

        plt.xlabel('Distance from A (rewarded)')
        plt.ylabel(ylabel)

        ax = plt.gca()

        if 'probability' in ylabel.lower():
            plt.ylim(0, 1.05)
            plt.yticks([0, 0.5, 1.0])
            ax.yaxis.set_major_formatter(
                mticker.PercentFormatter(xmax=1.0)
            )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([xmin, xmax])

        legend = ax.legend(frameon=False, loc='best', handlelength=0, handletextpad=0, markerscale=0)
        for handle in legend.legend_handles:
            handle.set_visible(False)
        for text in legend.get_texts():
            label = text.get_text()  
            text.set_color(color_map.get(label, color_map[label]))
            text.set_fontsize(24)

    return fig
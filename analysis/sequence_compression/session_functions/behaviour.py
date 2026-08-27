import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from session_functions.utils import *

np.set_printoptions(suppress=True, precision=2)

class Behaviour():
    def __init__(self, session, performance):
        self.session = session
        self.performance = performance

    def get_lick_persistence(self, plot=False, include_A=False, remove_disengagement=False, plot_disengagement=False):
        '''Get lick persistence (lick counts) per landmark type'''

        lick_position, lick_times, reward_times, reward_positions, _ = self.session.get_event_parsed()
        A_landmarks, B_landmarks, _, _ = self.session.get_A_B_landmarks()
        _, _, target_positions, distractor_positions, _, _ = self.session.find_targets_distractors()
        release_positions = np.sort(np.concatenate([target_positions, distractor_positions]))

        # Find total number of licks inside landmark - TODO redundant? 
        lick_counter = np.zeros(len(release_positions), dtype=int)

        for i, pos in enumerate(release_positions):
            # Licks within landmark boundaries
            mask = (
                (np.round(lick_position, 1) >= np.round(pos, 1)) &
                (np.round(lick_position, 1) < np.round(pos, 1) + self.session.lm_size)
            )

            # If this is an A landmark, and it is rewarded, only count licks before reward delivery
            if i in A_landmarks:
                rew_idx = np.where((reward_positions > pos) & (reward_positions <= pos + self.session.lm_size))[0]

                if len(rew_idx) > 0:
                    reward_time = reward_times[rew_idx[0]]
                    mask &= (lick_times <= reward_time)

            lick_counter[i] = np.sum(mask)

        # Counts licks for each type of landmark
        num_As, num_Bs = self.session.get_num_A_B()

        # Find distances between A and the following Bs
        A_A_diff, A_B_diff, A_positions, B_positions = self.session.find_A_B_distance_and_positions(rewarded_As=True)

        # Licks in rewarded As
        rewarded_lms = []
        licked_As = np.zeros_like(A_positions)
        for i, pos in enumerate(A_positions): # NOTE the first A is not considered 
            mask = (
                (np.round(lick_position, 1) >= np.round(pos, 1)) &
                (np.round(lick_position, 1) <= np.round(pos, 1) + self.session.lm_size)
            )
            rew_idx = np.where((reward_positions > pos) & (reward_positions <= pos + self.session.lm_size))[0]
            if len(rew_idx) > 0:
                target_idx = np.where(target_positions == pos)[0][0]
                rewarded_lms.append(A_landmarks[target_idx])
                reward_time = reward_times[rew_idx[0]]
                mask &= (lick_times <= reward_time)
            licked_As[i] = np.sum(mask)

        # Licks in Bs
        licked_Bs = np.zeros_like(B_positions)
        for i, pos in enumerate(B_positions):
            for j in range(num_Bs):
                mask = (
                    (np.round(lick_position, 1) >= np.round(pos[j], 1)) &
                    (np.round(lick_position, 1) <= np.round(pos[j], 1) + self.session.lm_size)
                )
                licked_Bs[i, j] = np.sum(mask)

        if remove_disengagement:
            # We want to disregard trials where the mouse got a reward and disengaged until the next A, so all intermediate Bs are excluded based on valid As
            _, _, _, _, [valid_mask_A, valid_mask_B], _ = self.performance.calc_time_from_rew_hit_fa(plot=False, remove_disengagement=remove_disengagement, plot_disengagement=False)
            # _, _, _, _, [valid_mask_A, valid_mask_B], _ = calc_time_from_A_hit_fa(sess_dataframe, ses_settings, plot=False, remove_disengagement=remove_disengagement, plot_disengagement=True)
            
            # Apply the masks for A->A
            A_A_diff = A_A_diff[valid_mask_A]
            licked_As = licked_As[valid_mask_A]
            
        # Apply the masks for A->Bs
        A_B_diff_list = [A_B_diff[valid_mask_A, i] if remove_disengagement else A_B_diff[:, i] for i in range(num_Bs)]
        licked_Bs = [licked_Bs[valid_mask_A, i] if remove_disengagement else licked_Bs[:, i] for i in range(num_Bs)]

        # Plot lick counts as a function of distance from A 
        if plot:
            all_B = np.concatenate(A_B_diff_list) if len(A_B_diff_list) > 0 else np.array([])
            if include_A:
                all_distances = np.concatenate([A_A_diff, all_B.flatten()])
                bins = np.linspace(np.min(all_distances), np.max(all_distances), 20)
            else:
                bins = np.linspace(np.min(all_B), np.max(all_B), 20)

            fig = plt.figure(figsize=(6,4))

            if include_A:
                cA, mA, sA = compute_binned_lick_rate(A_A_diff, licked_As, bins)
                plt.errorbar(cA, mA, yerr=sA, label='A', marker='o', color='darkblue')

            for i in range(num_Bs):
                if i == 0:
                    color = 'orange'
                elif i == 1:
                    color = 'gold'
                elif i == 2:
                    color = 'brown'
                
                c, m, s = compute_binned_lick_rate(A_B_diff_list[i], licked_Bs[i], bins)
                plt.errorbar(c, m, yerr=s, label=f'B{i+1}', marker='o', color=color)

            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ymin, ymax = ax.get_ylim()
            plt.yticks([0, np.round(ymax)])
            plt.legend(frameon=False, loc='upper right')
            plt.xlabel('Distance A (rewarded) → ')
            plt.ylabel('Lick counts')

            ax.set_xticks([bins[0], bins[-1]])

            return A_A_diff, A_B_diff_list, licked_As, licked_Bs, fig
        
        else:
            if include_A:
                return A_A_diff, A_B_diff_list, licked_As, licked_Bs, None
            else:
                return None, A_B_diff_list, None, licked_Bs, None

    def get_lick_counts(self, plot=False, threshold=True, misses=False, omissions=False):
        '''Get lick persistence (lick counts) per landmark type'''

        lick_position, lick_times, reward_times, reward_positions, _ = self.session.get_event_parsed(threshold=threshold)
        A_landmarks, B_landmarks, _, _ = self.session.get_A_B_landmarks()
        target_id, distractor_id, target_positions, distractor_positions, lm_ids, _ = self.session.find_targets_distractors()
        release_positions = np.sort(np.concatenate([target_positions, distractor_positions]))

        num_As, num_Bs = self.session.get_num_A_B()

        # Get misses and omissions
        trial = self.session.settings['trial']
        if len(trial) > 1:
            omitted_lms, omitted_pos = self.get_omissions()
        else:
            omitted_lms, omitted_pos = None, None
        missed_lms, missed_pos = self.get_misses()

        # Find total number of licks inside each landmark 
        lick_counter = np.zeros(len(release_positions))

        for i, pos in enumerate(release_positions):
            # Licks within landmark boundaries
            mask = (
                (np.round(lick_position, 1) >= np.round(pos, 1)) &
                (np.round(lick_position, 1) < np.round(pos, 1) + self.session.lm_size)
            )

            # Count licks if this is an A landmark and it was rewarded up to the reward timepoint
            if i in A_landmarks:
                if misses and i not in missed_lms:
                    continue
                rew_idx = np.where((reward_positions > pos) & (reward_positions <= pos + self.session.lm_size))[0]
                if len(rew_idx) > 0:
                    reward_time = reward_times[rew_idx[0]]
                    mask &= (lick_times <= reward_time)
            else:
                prev_pos = release_positions[release_positions < pos][-num_As:]
                if np.any(np.isin(prev_pos, omitted_pos)) or np.any(np.isin(prev_pos, missed_pos)):
                    # Do not consider this landmark if it was preceded by an omission or a miss
                    continue
                
            lick_counter[i] = np.sum(mask)

        # if misses:
        #     lick_counter[lick_counter == 0] = np.nan

        # Split by landmark type
        A1, A2, B1, B2, B3 = self.session.get_A_B_splits()

        A1_licks = lick_counter[A1]
        A2_licks = lick_counter[A2] if len(A2) else np.array([])
        if omissions:
            A1_licks = lick_counter[[i for i in A1 if i in omitted_lms]]
            A2_licks = lick_counter[[i for i in A2 if i in omitted_lms]]
        if misses:
            A1_licks = lick_counter[[i for i in A1 if i in missed_lms]]
            A2_licks = lick_counter[[i for i in A2 if i in missed_lms]]
        B1_licks = lick_counter[B1]
        B2_licks = lick_counter[B2] if len(B2) else np.array([])
        B3_licks = lick_counter[B3] if len(B3) else np.array([])

        lick_counter_groups = {
            ("A" if k == "A1" and num_As == 1 else
            "B" if k == "B1" and num_Bs == 1 else
            k): np.asarray(v)
            for k, v in {
                "A1": A1_licks,
                "A2": A2_licks,
                "B1": B1_licks,
                "B2": B2_licks,
                "B3": B3_licks,
            }.items()
        }

        # Compute distances 
        A_A_diff, A_B_diff, A_positions, B_positions = self.session.find_all_A_B_distance_and_positions()
        
        # Count licks as a function of distance from A1 
        # Licks in As
        licked_As = np.zeros_like(A_positions)
        for i, pos in enumerate(A_positions): # NOTE the first A is not considered 
            for j in range(num_As):
                curr_pos = pos[j]
                if omissions and curr_pos not in omitted_pos:
                    continue
                if misses and curr_pos not in missed_pos:
                    continue
                mask = (
                    (np.round(lick_position, 1) >= np.round(curr_pos, 1)) &
                    (np.round(lick_position, 1) <= np.round(curr_pos, 1) + self.session.lm_size)
                )
                rew_idx = np.where((reward_positions > curr_pos) & (reward_positions <= curr_pos + self.session.lm_size))[0]
                if len(rew_idx) > 0:
                    reward_time = reward_times[rew_idx[0]]
                    mask &= (lick_times <= reward_time)
                licked_As[i, j] = np.sum(mask)
        
        # Licks in Bs
        licked_Bs = np.zeros_like(B_positions)
        for i, pos in enumerate(B_positions):
            for j in range(num_Bs):
                curr_pos = pos[j]
                prev_pos = release_positions[release_positions < curr_pos][-num_As:]
                if np.any(np.isin(prev_pos, omitted_pos)) or np.any(np.isin(prev_pos, missed_pos)):
                    # Do not consider this landmark if it was preceded by an omission or a miss
                    continue
                mask = (
                    (np.round(lick_position, 1) >= np.round(curr_pos, 1)) &
                    (np.round(lick_position, 1) <= np.round(curr_pos, 1) + self.session.lm_size)
                )
                licked_Bs[i, j] = np.sum(mask)
        
        A_A_diff_list = [A_A_diff[:, i] for i in range(num_As)]
        A_B_diff_list = [A_B_diff[:, i] for i in range(num_Bs)]
        licked_As = [licked_As[:, i] for i in range(num_As)]
        licked_Bs = [licked_Bs[:, i] for i in range(num_Bs)]

        # Plot the data 
        if plot:
            with mpl.rc_context({
                'axes.titlesize': 18,
                'axes.labelsize': 18,
                'xtick.labelsize': 14,
                'ytick.labelsize': 14,
                'legend.fontsize': 18,
            }):
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax = ax.ravel()
                
                # 1. Bar plot of number of licks per lm type
                labels = [k for k in lick_counter_groups if len(lick_counter_groups[k]) > 0]
                values = [lick_counter_groups[k] for k in labels]

                colors = {
                    "A": ["darkblue", "mediumblue"],
                    "A1": ["darkblue", "mediumblue"],
                    "A2": ["blue", "dodgerblue"],
                    "B": ["orange", "gold"],
                    "B1": ["orange", "gold"],
                    "B2": ["gold", "yellow"],
                    "B3": ["brown", "sandybrown"]
                }
                
                bar_colors = [colors[l][0] for l in labels]
                dot_colors = [colors[l][1] for l in labels]

                means = [np.nanmean(v) for v in values]

                x = np.arange(len(labels))
                ax[0].bar(x, means, color=bar_colors)

                # Individual data points
                for i, v in enumerate(values):
                    jitter = np.random.uniform(-0.08, 0.08, size=len(v))
                    ax[0].scatter(
                        np.full(len(v), x[i]) + jitter,
                        v,
                        color=dot_colors[i],
                        alpha=0.5
                    )

                ax[0].set_xticks(x)
                if misses:
                    ax[0].set_xticklabels(f"{label} (miss)" if label.startswith("A") else label for label in labels)
                if omissions:
                    ax[0].set_xticklabels(f"{label}\n(omitted)" if label.startswith("A") else label for label in labels)
                if not misses and not omissions:
                    ax[0].set_xticklabels(labels)
                ax[0].set_ylabel("Licks per landmark")
                ax[0].spines['top'].set_visible(False)
                ax[0].spines['right'].set_visible(False)


                # 2. Lick counts as a function of distance from rewarded A1s
                all_distances = np.concatenate([A_A_diff_list, A_B_diff_list])
                bins = np.linspace(np.min(all_distances), np.max(all_distances), 20)
                
                A_order = [1, 0] if num_As == 2 else range(num_As)
                for i, A_val in enumerate(A_order):
                    label = "A" if (num_As == 1 and i == 0) else f"A{i+1}"
                    display_label = (f"{label} (miss)" if misses else f"{label} (omitted)" if omissions else label)
                    c, m, s = compute_binned_lick_rate(A_A_diff_list[A_val], licked_As[A_val], bins)
                    ax[1].errorbar(c, m, yerr=s, label=display_label, marker='o', color=colors[label][0])

                for i in range(num_Bs):
                    label = "B" if (num_Bs == 1 and i == 0) else f"B{i+1}"
                    c, m, s = compute_binned_lick_rate(A_B_diff_list[i], licked_Bs[i], bins)
                    ax[1].errorbar(c, m, yerr=s, label=label, marker='o', color=colors[label][0])

                ymin, ymax = ax[1].get_ylim()
                ax[1].set_yticks([0, np.round(ymax)])
                ax[1].set_xticks([bins[0], bins[-1]])

                legend = ax[1].legend(frameon=False, loc='upper left', handlelength=0, handletextpad=0, markerscale=0)
                for handle, text in zip(legend.legend_handles, legend.get_texts()):
                    text.set_color(handle.get_color())
                    handle.set_visible(False)
                    # text.set_fontsize(24)

                if num_As == 1:
                    ax[1].set_xlabel('Distance A (rewarded) → ')
                else:
                    ax[1].set_xlabel('Distance A1 (rewarded) → ')
                ax[1].set_ylabel('Lick counts')

                for a in ax:
                    a.spines['top'].set_visible(False)
                    a.spines['right'].set_visible(False)

                # plt.tight_layout()

            return lick_counter_groups, licked_As, licked_Bs, A_A_diff_list, A_B_diff_list, fig
        
        else:
            return lick_counter_groups, licked_As, licked_Bs, A_A_diff_list, A_B_diff_list, None

    def get_omissions(self):
        '''Find which As were omitted'''
        # TODO incorporate into estimation of target positions if there are multiple types of trials

        A_landmarks, B_landmarks, A_idx, B_idx = self.session.get_A_B_landmarks()

        all_events = self.session.dataframe['Sequence_Position'].dropna().to_numpy().astype(int)
        all_positions = self.session.dataframe['Position'].values[self.session.dataframe['Sequence_Position'].notna()]

        target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = self.session.find_targets_distractors()
        seq_start = np.where(target_positions[0] == all_positions)[0][0]
        all_events = all_events[seq_start:]
        all_positions = all_positions[seq_start:]

        A_binary = np.zeros(len(all_events), dtype=bool)
        A_binary[A_landmarks] = 1

        omissions = np.where(A_binary & (all_events == -1))[0]
        omission_positions = all_positions[omissions]

        return omissions, omission_positions

    def get_misses(self):
        '''Find which As were missed'''

        A_landmarks, B_landmarks, A_idx, B_idx = self.session.get_A_B_landmarks()
        lick_position, lick_times, reward_times, reward_positions, release_df = self.session.get_event_parsed()

        all_events = self.session.dataframe['Sequence_Position'].dropna().to_numpy().astype(int)
        all_positions = self.session.dataframe['Position'].values[self.session.dataframe['Sequence_Position'].notna()]

        target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = self.session.find_targets_distractors()

        if self.session.cohort == 1:
            seq_start = np.where(target_positions[0] == all_positions)[0][0]
        elif self.session.cohort == 2:
            seq_start = 0
        all_events = all_events[seq_start:]
        all_positions = all_positions[seq_start:]

        misses = []
        miss_positions = []
        for A in A_landmarks:
            curr_pos = all_positions[A]
            rew_idx = np.where((reward_positions > curr_pos) & (reward_positions <= curr_pos + self.session.lm_size))[0]

            if len(rew_idx) > 0:
                continue
            misses.append(A)
            miss_positions.append(curr_pos)  
            
        return misses, miss_positions
  
    def get_speed_psth(self, events=None, bins=300):
        '''Get speed around landmark entry'''

        # Get session data
        position = self.session.sess['position']
        licks_idx = np.where(self.session.sess['licks'] > 0)[0]

        if 'LM_Count' in self.session.dataframe.columns:
            release_df = self.session.estimate_lm_events()
        else:
            release_df = self.session.estimate_release_events()

        if events is None:
            events = release_df['Index']

        # # Remove non-responsive runs for this analysis
        # no_response_trials = []
        # for i, lm_idx in enumerate(events):  
        #     start_idx = lm_idx - bins / 2
        #     end_idx = lm_idx + bins / 2
        #     # only consider lm until the last reward - ignore lms that the mouse just ran through 
        #     if not np.any(licks_idx[(licks_idx > start_idx) & (licks_idx < (end_idx))]):
        #         no_response_trials.append(i)

        # large_gaps = np.where(np.diff(no_response_trials) > 20)[0]
        # if len(large_gaps) > 0:
        #     cutoff_idx = large_gaps[-1] + 1
        #     cutoff_trial = no_response_trials[cutoff_idx]
        #     events = events[:cutoff_trial]
        #     print("Exclude from trial:", cutoff_trial)
            
        # Bin speed
        binned_speed = np.zeros((len(events), bins))

        for i, lm_idx in enumerate(events):  
            start_idx = lm_idx - bins / 2
            end_idx = lm_idx + bins / 2

            if start_idx < 0:
                continue
            if end_idx > len(position):
                break
            
            event_idx = np.arange(start_idx, end_idx).astype(int)
            binned_speed[i] = self.session.sess['speed'][event_idx]
            # bin_edges = np.linspace(start_idx, end_idx, bins + 1).astype(int)
            # binned_speed[i], _, _ = stats.binned_statistic(event_idx, session['speed'][event_idx], statistic='mean', bins=bin_edges)

            # event_pos = position[np.where(release_df['Index'] == lm_idx)[0][0]]
            # lm_exit_idx = np.argmin(np.abs(position - (event_pos + session['lm_size'])))

        mean_binned_speed = np.mean(binned_speed, axis=0)
        sem_binned_speed = stats.sem(binned_speed, axis=0)

        return mean_binned_speed, sem_binned_speed

    def get_lick_rate_psth(self, events=None, bins=300):
        '''Get lick rate around landmark entry'''

        # Threshold licks 
        licks = threshold_lick_events(self.session.sess, self.session.dataframe)

        if 'LM_Count' in self.session.dataframe.columns:
            release_df = self.session.estimate_lm_events()
        else:
            release_df = self.session.estimate_release_events()

        if events is None:
            events = release_df['Index']
        
        # Bin licks
        binned_licks = np.zeros((len(events), bins))

        for i, lm_idx in enumerate(events):    
            start_idx = lm_idx - bins / 2
            end_idx = lm_idx + bins / 2
            if start_idx < 0:
                continue
            if end_idx > len(self.session.sess['position']):
                break
            
            event_idx = np.arange(start_idx, end_idx).astype(int)
            bin_edges = np.linspace(start_idx, end_idx, bins + 1).astype(int)
            
            binned_licks[i], _, _ = stats.binned_statistic(event_idx, licks[event_idx], statistic='mean', bins=bin_edges)

        mean_binned_licks = np.mean(binned_licks, axis=0)
        sem_binned_licks = stats.sem(binned_licks, axis=0)

        return mean_binned_licks, sem_binned_licks

    def get_speed_psth_by_distance(self, bins=300, binning=True, distance_groups=None, rewarded_As=False):
        '''Compute speed PSTH split by landmark type and distance groups'''

        # --- session data ---
        speed = self.session.sess['speed']

        if 'LM_Count' in self.session.dataframe.columns:
            release_df = self.session.estimate_lm_events()
        else:
            release_df = self.session.estimate_release_events()

        # --- landmark structure ---
        target_id, distractor_id, target_positions, distractor_positions, _, _ = \
            self.session.find_targets_distractors()
        lick_position, lick_times, reward_times, reward_positions, release_df = self.session.get_event_parsed()

        # --- number of Bs ---
        num_Bs = len(
            distractor_positions[
                (distractor_positions > target_positions[0]) &
                (distractor_positions < target_positions[1])
            ]
        )

        # =========================
        # --- COMPUTE DISTANCES ---
        # =========================
        A_A_diff, A_B_diff, A_positions, B_positions = self.session.find_A_B_distance_and_positions(rewarded_As)
        
        events_A = []
        events_B_list = [[] for _ in range(num_Bs)]

        if rewarded_As:
            for i, pos in enumerate(reward_positions[:-1]):
                idx_A = np.argmin(np.abs(release_df['Position'] - pos))
                events_A.append(release_df['Index'].iloc[idx_A])

                for j in range(num_Bs):
                    B_pos = B_positions[i, j]
                    idx_B = np.argmin(np.abs(release_df['Position'] - B_pos))
                    events_B_list[j].append(release_df['Index'].iloc[idx_B])

        else:
            for i, pos in enumerate(target_positions[:-1]):
                idx_A = np.argmin(np.abs(release_df['Position'] - pos))
                events_A.append(release_df['Index'].iloc[idx_A])

                for j in range(num_Bs):
                    B_pos = B_positions[i, j]
                    idx_B = np.argmin(np.abs(release_df['Position'] - B_pos))
                    events_B_list[j].append(release_df['Index'].iloc[idx_B])

        events_A = np.array(events_A)
        events_B_list = [np.array(e) for e in events_B_list]

        # =========================
        # --- DISTANCE GROUPING ---
        # =========================

        # Find common distances (where different performance could only be explained by a knowledge of lm type)
        if distance_groups is None:
            AA_distances = np.unique(np.round(A_A_diff))
            AB_distances = np.concatenate([np.unique(np.round(A_B_diff[:, j])) for j in range(A_B_diff.shape[1])])

            distance_range = np.intersect1d(AA_distances, AB_distances)
            
            if binning:
                n = len(distance_range)
                base = n // 3
                distance_groups = {
                    'small':  distance_range[:base],
                    'medium': distance_range[base:2 * base],
                    'large':  distance_range[2 * base:]
                }
            else:
                distance_groups = {d: np.array([d]) for d in distance_range}

        # =========================
        # --- PSTH FUNCTION ---
        # =========================

        def compute_psth(events):
            if len(events) == 0:
                return np.full(bins, np.nan), np.full(bins, np.nan)

            binned_speed = np.zeros((len(events), bins))

            for i, lm_idx in enumerate(events):
                start_idx = int(lm_idx - bins / 2)
                end_idx = int(lm_idx + bins / 2)

                if start_idx < 0 or end_idx > len(speed):
                    continue

                idx_range = np.arange(start_idx, end_idx)
                binned_speed[i] = speed[idx_range]

            return np.nanmean(binned_speed, axis=0), stats.sem(binned_speed, axis=0, nan_policy='omit')

        # =========================
        # --- COMPUTE PSTHs ---
        # =========================
        
        psth_A = {}
        psth_B = []

        # --- A ---
        for group_name, group_distances in distance_groups.items():
            mask = np.isin(A_A_diff, group_distances)
            events_group = events_A[mask]

            psth_A[group_name] = compute_psth(events_group)

        # --- B ---
        for j in range(num_Bs):
            psth_B_j = {}

            distances = A_B_diff[:, j]
            events = events_B_list[j]

            for group_name, group_distances in distance_groups.items():
                mask = np.isin(distances, group_distances)
                events_group = events[mask]

                psth_B_j[group_name] = compute_psth(events_group)

            psth_B.append(psth_B_j)

        return psth_A, psth_B, distance_groups

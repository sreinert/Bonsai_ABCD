import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from session_functions.utils import *
from session_functions.plotting import *

np.set_printoptions(suppress=True, precision=2)

class Performance():
    def __init__(self, session):
        self.session = session

    def calc_hit_fa(self):
        '''Calculate average hit and false alarm rate across a session'''

        target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = self.session.find_targets_distractors()

        lick_position, lick_times, reward_times, reward_positions, release_df = self.session.get_event_parsed()
        release_positions = np.sort(np.concatenate([target_positions, distractor_positions]))

        licked_target = np.zeros(len(target_positions))
        for idx, pos in enumerate(target_positions):
            if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                licked_target[idx] = 1

        licked_distractor = np.zeros(len(distractor_positions))
        for idx, pos in enumerate(distractor_positions):
            if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                licked_distractor[idx] = 1

        licked_all = np.zeros(len(release_df), dtype=int)
        rewarded_all = np.zeros(len(release_df), dtype=int)
        for idx, pos in enumerate(release_positions):
            # only take into account licks/rewards that came later than the release
            licks = lick_position[lick_times >= release_df.index[idx]]
            rewards = reward_positions[reward_times >= release_df.index[idx]]
            # compare licks/rewards to position window (the LM position and logged position are offset by 3)
            if np.any((licks > (pos)) & (licks < (pos + self.session.lm_size))):
                licked_all[idx] = 1
            if np.any((rewards > (pos)) & (rewards < (pos + self.session.lm_size))):
                rewarded_all[idx] = 1

        hit_rate = np.sum(licked_target) / len(licked_target) 
        fa_rate = np.sum(licked_distractor) / len(licked_distractor) 
        # adjust hit rate and fa rate to avoid infinity in d-prime calculation
        if hit_rate == 1:
            hit_rate = 0.99
        if hit_rate == 0:
            hit_rate = 0.01
        if fa_rate == 1:
            fa_rate = 0.99
        if fa_rate == 0:
            fa_rate = 0.01

        d_prime = np.log10(hit_rate/(1-hit_rate)) - np.log10(fa_rate/(1-fa_rate))

        return hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all

    def get_hit_fa_events_split(self):
        target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = self.session.find_targets_distractors()
        A_landmarks, B_landmarks, A_idx, B_idx = self.session.get_A_B_landmarks()
        lick_position, *_ = self.session.get_event_parsed()

        release_positions = np.sort(np.concatenate([target_positions, distractor_positions]))

        # --- split landmark groups exactly like your function ---
        A1, A2, B1, B2, B3 = self.session.get_A_B_splits()

        # map to positions
        A1_pos = release_positions[A1]
        A2_pos = release_positions[A2] if len(A2) else np.array([])
        B1_pos = release_positions[B1]
        B2_pos = release_positions[B2] if len(B2) else np.array([])
        B3_pos = release_positions[B3] if len(B3) else np.array([])

        # helper to compute binary events
        def compute_events(positions):
            events = []
            for pos in positions:
                events.append(
                    int(np.any((lick_position > pos) & (lick_position < pos + self.session.lm_size)))
                )
            return np.array(events)

        events = {
            "A1": compute_events(A1_pos),
            "B1": compute_events(B1_pos),
        }

        if len(A2_pos):
            events["A2"] = compute_events(A2_pos)
        if len(B2_pos):
            events["B2"] = compute_events(B2_pos)
        if len(B3_pos):
            events["B3"] = compute_events(B3_pos)

        # Rename keys if only A1 and B1 exist
        if set(events.keys()) == {"A1", "B1"}:
            events = {
                "A": events["A1"],
                "B": events["B1"]
            }
        return events

    def calc_sw_hit_fa(self, window=12, split_lms=False, plot=True):
        '''Calculate hit and false alarm rates as a sliding window across the session'''

        target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = self.session.find_targets_distractors()
        A_landmarks, B_landmarks, A_idx, B_idx = self.session.get_A_B_landmarks()
        lick_position, lick_times, reward_times, reward_positions, release_df = self.session.get_event_parsed()
        release_positions = np.sort(np.concatenate([target_positions, distractor_positions]))
        
        if not split_lms:
            hit_rate_sw = np.zeros(len(release_positions[:-window]))
            fa_rate_sw = np.zeros(len(release_positions[:-window]))

            for idx, pos in enumerate(release_positions[:-window]):

                # Find landmark events within the specified window
                positions_range = release_positions[idx:idx + window]
                
                lick_pos_range = lick_position[(lick_position >= positions_range[0]) & (lick_position <= positions_range[-1] + self.session.lm_size)]
                target_pos_range = target_positions[(target_positions >= positions_range[0]) & (target_positions <= positions_range[-1])]
                distractor_pos_range = distractor_positions[(distractor_positions >= positions_range[0]) & (distractor_positions <= positions_range[-1])]
                
                # Find responses to targets and distractors inside the lms
                licked_target = np.zeros(len(target_pos_range))
                for a, a_pos in enumerate(target_pos_range):
                    if np.any((lick_pos_range > a_pos) & (lick_pos_range < (a_pos + self.session.lm_size))):
                        licked_target[a] = 1
                
                licked_distractor = np.zeros(len(distractor_pos_range))
                for b, b_pos in enumerate(distractor_pos_range):
                    if np.any((lick_pos_range > b_pos) & (lick_pos_range < (b_pos + self.session.lm_size))):
                        licked_distractor[b] = 1

                # Calculate hit and false alarm rates for each window 
                hit_rate_sw[idx] = np.sum(licked_target) / len(licked_target) 
                fa_rate_sw[idx] = np.sum(licked_distractor) / len(licked_distractor) 
                # adjust hit rate and fa rate to avoid infinity in d-prime calculation
                if hit_rate_sw[idx] == 1:
                    hit_rate_sw[idx] = 0.99
                if hit_rate_sw[idx] == 0:
                    hit_rate_sw[idx] = 0.01
                if fa_rate_sw[idx] == 1:
                    fa_rate_sw[idx] = 0.99
                if fa_rate_sw[idx] == 0:
                    fa_rate_sw[idx] = 0.01

        else:       
            A1, A2, B1, B2, B3 = self.session.get_A_B_splits()

            A1_positions = release_positions[A1]
            A2_positions = release_positions[A2] if len(A2) > 0 else np.array([])

            B1_positions = release_positions[B1]
            B2_positions = release_positions[B2] if len(B2) > 0 else np.array([])
            B3_positions = release_positions[B3] if len(B3) > 0 else np.array([])


            hit_rate_sw = {"A1": np.zeros(len(release_positions[:-window]))}
            if len(A2) > 0:
                hit_rate_sw["A2"] = np.zeros(len(release_positions[:-window]))

            fa_rate_sw  = {"B1": np.zeros(len(release_positions[:-window]))}
            if len(B2) > 0:
                fa_rate_sw["B2"] = np.zeros(len(release_positions[:-window]))
            if len(B3) > 0:
                fa_rate_sw["B3"] = np.zeros(len(release_positions[:-window]))

            for idx, pos in enumerate(release_positions[:-window]):
                
                # Find landmark events within the specified window
                positions_range = release_positions[idx:idx + window]
                
                lick_pos_range = lick_position[(lick_position >= positions_range[0]) & (lick_position <= positions_range[-1] + self.session.lm_size)]
                
                A1_pos_range = A1_positions[(A1_positions >= positions_range[0]) & (A1_positions <= positions_range[-1])]
                A2_pos_range = A2_positions[(A2_positions >= positions_range[0]) & (A2_positions <= positions_range[-1])] if len(A2_positions) else np.array([])
                
                B1_pos_range = B1_positions[(B1_positions >= positions_range[0]) & (B1_positions <= positions_range[-1])]
                B2_pos_range = B2_positions[(B2_positions >= positions_range[0]) & (B2_positions <= positions_range[-1])]
                B3_pos_range = B3_positions[(B3_positions >= positions_range[0]) & (B3_positions <= positions_range[-1])] if len(B3_positions) else np.array([])

                # Find responses to targets and distractors inside the lms
                licked_A1 = np.zeros(len(A1_pos_range))
                for a, a_pos in enumerate(A1_pos_range):
                    if np.any((lick_pos_range > a_pos) & (lick_pos_range < (a_pos + self.session.lm_size))):
                        licked_A1[a] = 1
                
                if len(A2_positions):
                    licked_A2 = np.zeros(len(A2_pos_range))
                    for a, a_pos in enumerate(A2_pos_range):
                        if np.any((lick_pos_range > a_pos) & (lick_pos_range < (a_pos + self.session.lm_size))):
                            licked_A2[a] = 1

                licked_B1 = np.zeros(len(B1_pos_range))
                for b, b_pos in enumerate(B1_pos_range):
                    if np.any((lick_pos_range > b_pos) & (lick_pos_range < (b_pos + self.session.lm_size))):
                        licked_B1[b] = 1

                if len(B2_positions):
                    licked_B2 = np.zeros(len(B2_pos_range))
                    for b, b_pos in enumerate(B2_pos_range):
                        if np.any((lick_pos_range > b_pos) & (lick_pos_range < (b_pos + self.session.lm_size))):
                            licked_B2[b] = 1

                if len(B3_positions):
                    licked_B3 = np.zeros(len(B3_pos_range))
                    for a, a_pos in enumerate(B3_pos_range):
                        if np.any((lick_pos_range > a_pos) & (lick_pos_range < (a_pos + self.session.lm_size))):
                            licked_B3[a] = 1

                # Calculate hit and false alarm rates for each window 
                hit_rate_sw["A1"][idx] = np.sum(licked_A1) / len(licked_A1) 
                if len(A2_positions):
                    hit_rate_sw["A2"][idx] = np.clip(np.sum(licked_A2) / len(licked_A2), 0.01, 0.99)
                fa_rate_sw["B1"][idx] = np.sum(licked_B1) / len(licked_B1) 
                if len(B2_positions):
                    fa_rate_sw["B2"][idx] = np.clip(np.sum(licked_B2) / len(licked_B2), 0.01, 0.99)
                if len(B3_positions):
                    fa_rate_sw["B3"][idx] = np.clip(np.sum(licked_B3) / len(licked_B3), 0.01, 0.99)

                # adjust hit rate and fa rate to avoid infinity in d-prime calculation
                hit_rate_sw["A1"][idx] = np.clip(hit_rate_sw["A1"][idx], 0.01, 0.99)
                fa_rate_sw["B1"][idx] = np.clip(fa_rate_sw["B1"][idx], 0.01, 0.99)

        if plot:
            fig = plt.figure(figsize=(6,3))

            if not split_lms:
                plt.plot(hit_rate_sw, c='darkblue', linewidth=2, label='hit rate')
                plt.plot(fa_rate_sw, c='orange', linewidth=2, label='fa rate')

            else:
                plt.plot(hit_rate_sw["A1"], c='darkblue', linewidth=2, label='Hit A1')
                if "A2" in hit_rate_sw:
                    plt.plot(hit_rate_sw["A2"], c='blue', linewidth=2, label='Hit A2')
                plt.plot(fa_rate_sw["B1"], c='orange', linewidth=2, label='FA B1')
                if "B2" in fa_rate_sw:
                    plt.plot(fa_rate_sw["B2"], c='gold', linewidth=2, label='FA B2')
                if "B3" in fa_rate_sw:
                    plt.plot(fa_rate_sw["B3"], c='brown', linewidth=2, label='FA B3')

            plt.ylim([0,1.1])
            plt.yticks([0,0.5,1])
            plt.xlabel(f'Landmark window (n={window} lms)')
            ax = plt.gca()
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.legend(frameon=False)

            return hit_rate_sw, fa_rate_sw, fig
        
        else:
            return hit_rate_sw, fa_rate_sw, None

    def calc_distance_hit_fa(self, split_lms=False, plot=True):
        '''Calculate hit and fa rates for each distance'''

        lick_position, lick_times, reward_times, reward_positions, release_df = self.session.get_event_parsed()
        target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = self.session.find_targets_distractors()
        A_landmarks, B_landmarks, A_idx, B_idx = self.session.get_A_B_landmarks()

        hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = self.calc_hit_fa()

        distances = np.diff(release_df['Position']) - self.session.lm_size
        
        release_positions = np.sort(np.concatenate([target_positions, distractor_positions]))
        
        if not split_lms:
            hit_rate = {"A": {}}
            fa_rate = {"B": {}}

            for d in np.unique(distances):
                lms_considered = np.where(distances == d)[0] + 1

                target_pos_considered = [pos for pos in target_positions if pos in release_positions[lms_considered]]
                distractor_pos_considered = [pos for pos in distractor_positions if pos in release_positions[lms_considered]]
                
                licked_target = np.zeros(len(target_pos_considered))
                for idx, pos in enumerate(target_pos_considered):
                    if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                        licked_target[idx] = 1

                licked_distractor = np.zeros(len(distractor_pos_considered))
                for idx, pos in enumerate(distractor_pos_considered):
                    if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                        licked_distractor[idx] = 1

                hit_rate["A"][d] = (np.sum(licked_target) / len(licked_target)
                    if len(licked_target) > 0 else np.nan)

                fa_rate["B"][d] = (np.sum(licked_distractor) / len(licked_distractor)
                    if len(licked_distractor) > 0 else np.nan)
        
        else:
            A1, A2, B1, B2, B3 = self.session.get_A_B_splits()

            A1_positions = release_positions[A1]
            A2_positions = release_positions[A2] if len(A2) > 0 else np.array([])

            B1_positions = release_positions[B1]
            B2_positions = release_positions[B2]
            B3_positions = release_positions[B3] if len(B3) > 0 else np.array([])

            hit_rate = {"A1": {}}
            if len(A2) > 0:
                hit_rate["A2"] = {}
            fa_rate = {"B1": {}, "B2": {}}
            if len(B3) > 0:
                fa_rate["B3"] = {}

            for d in np.unique(distances):
                lms_considered = np.where(distances == d)[0] + 1
                pos_considered = release_positions[lms_considered]

                A1_pos_considered = [p for p in A1_positions if p in pos_considered]
                A2_pos_considered = [p for p in A2_positions if p in pos_considered]
                B1_pos_considered = [p for p in B1_positions if p in pos_considered]
                B2_pos_considered = [p for p in B2_positions if p in pos_considered]
                B3_pos_considered = [p for p in B3_positions if p in pos_considered]

                licked_A1 = np.zeros(len(A1_pos_considered))
                for i, pos in enumerate(A1_pos_considered):
                    if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                        licked_A1[i] = 1

                licked_A2 = np.zeros(len(A2_pos_considered))
                for i, pos in enumerate(A2_pos_considered):
                    if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                        licked_A2[i] = 1
                
                licked_B1 = np.zeros(len(B1_pos_considered))
                for i, pos in enumerate(B1_pos_considered):
                    if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                        licked_B1[i] = 1
                
                licked_B2 = np.zeros(len(B2_pos_considered))
                for i, pos in enumerate(B2_pos_considered):
                    if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                        licked_B2[i] = 1

                licked_B3 = np.zeros(len(B3_pos_considered))
                for i, pos in enumerate(B3_pos_considered):
                    if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                        licked_B3[i] = 1

                hit_rate["A1"][d] = (np.sum(licked_A1) / len(licked_A1)
                    if len(licked_A1) > 0 else np.nan)

                if len(A2) > 0:
                    hit_rate["A2"][d] = (np.sum(licked_A2) / len(licked_A2)
                        if len(licked_A2) > 0 else np.nan)

                fa_rate["B1"][d] = (np.sum(licked_B1) / len(licked_B1)
                    if len(licked_B1) > 0 else np.nan)

                fa_rate["B2"][d] = (np.sum(licked_B2) / len(licked_B2)
                    if len(licked_B2) > 0 else np.nan)
                
                if len(B3) > 0:
                    fa_rate["B3"][d] = (np.sum(licked_B3) / len(licked_B3)
                    if len(licked_B3) > 0 else np.nan)
        
        if plot:
            with mpl.rc_context({
                'axes.titlesize': 10,
                'axes.labelsize': 10,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
            }):

                fig = plt.figure(figsize=(6,3))

                if not split_lms:
                    d_sorted = sorted(hit_rate["A"].keys())
                    plt.plot(d_sorted, [hit_rate["A"][d] for d in d_sorted],
                            c='darkblue', linewidth=2, label='hit rate')
                    plt.plot(d_sorted, [fa_rate["B"][d] for d in d_sorted],
                            c='orange', linewidth=2, label='fa rate')

                else:
                    d_sorted = sorted(fa_rate["B1"].keys())
                    plt.plot(d_sorted, [hit_rate["A1"][d] for d in d_sorted],
                            c='darkblue', linewidth=2, label='hit A1')
                    if "A2" in hit_rate:
                        plt.plot(d_sorted, [hit_rate["A2"][d] for d in d_sorted],
                                c='blue', linewidth=2, label='hit A2')
                    plt.plot(d_sorted, [fa_rate["B1"][d] for d in d_sorted],
                            c='orange', linewidth=2, label='fa B1')
                    plt.plot(d_sorted, [fa_rate["B2"][d] for d in d_sorted],
                            c='gold', linewidth=2, label='fa B2')
                    if "B3" in fa_rate:
                        plt.plot(d_sorted, [fa_rate["B3"][d] for d in d_sorted],
                            c='brown', linewidth=2, label='fa B3')

                plt.xlabel("Landmark distance")
                ax = plt.gca()
                ax.set_ylim([0,1.1])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.yticks([0,0.5,1])
                plt.xticks([np.min(distances), np.max(distances)])
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
                plt.legend(frameon=False, loc='lower left')

            return hit_rate, fa_rate, fig
        
        else:
            return hit_rate, fa_rate, None

    def calc_distance_from_A_hit_fa(self, plot=True, remove_disengagement=False, plot_disengagement=False):
        '''Calculate hit and fa rates based on distance from preceding A'''
        
        lick_position, lick_times, reward_times, reward_positions, release_df = self.session.get_event_parsed()
        target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = self.session.find_targets_distractors()
        A_landmarks, B_landmarks, A_idx, B_idx = self.session.get_A_B_landmarks()

        hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = self.calc_hit_fa()

        # Find number of landmarks between two consecutive As
        num_Bs = len(distractor_positions[(distractor_positions > target_positions[0]) & (distractor_positions < target_positions[1])])

        # Find distances between A and the following Bs
        A_A_diff, A_B_diff, A_positions, B_positions = self.session.find_A_B_distance_and_positions(rewarded_As=False)

        # Calculate hit rate
        licked_As = np.zeros((len(target_positions) - 1))
        for i, pos in enumerate(target_positions[:-1]):
            if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                licked_As[i] = 1

        # Calculate false alarm rates
        licked_Bs = np.zeros((len(target_positions) - 1, num_Bs))
        for i, pos in enumerate(B_positions):
            for j in range(num_Bs):
                if np.any((lick_position > pos[j]) & (lick_position < (pos[j] + self.session.lm_size))):
                    licked_Bs[i, j] = 1

        # Filter out disengaged trials 
        if remove_disengagement:
            _, _, _, _, [valid_mask_A, valid_mask_B], _ = self.calc_time_from_A_hit_fa(plot=False, remove_disengagement=True, plot_disengagement=plot_disengagement)
            
            # Apply the masks for A->A
            A_A_diff = A_A_diff[valid_mask_A]
            licked_As = licked_As[valid_mask_A]

        # Apply the masks for A->Bs
        A_B_diff_list = [A_B_diff[valid_mask_A, i] if remove_disengagement else A_B_diff[:, i] for i in range(num_Bs)]
        licked_Bs_list = [licked_Bs[valid_mask_A, i] if remove_disengagement else licked_Bs[:, i] for i in range(num_Bs)]
        
        # A_B_diff_list = [A_B_diff[valid_mask_B[:, i], i] if remove_disengagement else A_B_diff[:, i] for i in range(num_Bs)]
        # licked_Bs_list = [licked_Bs[valid_mask_B[:, i], i] if remove_disengagement else licked_Bs[:, i] for i in range(num_Bs)]
        
        if plot:
            all_B = np.concatenate(A_B_diff) if len(A_B_diff) > 0 else np.array([])
            all_distances = np.concatenate([A_A_diff, all_B.flatten()])
            bins = np.linspace(np.min(all_distances), np.max(all_distances), 20)

            fig = plt.figure(figsize=(6,4))

            cA, mA, sA = compute_binned_lick_rate(A_A_diff, licked_As, bins)
            plt.errorbar(cA, mA, yerr=sA, label='A', marker='o', color='darkblue')

            for i in range(num_Bs):
                if i == 0:
                    color = 'orange'
                elif i == 1:
                    color = 'gold'
                elif i == 2:
                    color = 'brown'
                c, m, s = compute_binned_lick_rate(A_B_diff_list[i], licked_Bs_list[i], bins)
                plt.errorbar(c, m, yerr=s, label=f'B{i+1}', marker='o', color=color)

            plt.ylim([0,1.1])
            plt.yticks([0,0.5,1])
            ax = plt.gca()
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.legend(frameon=False, loc='lower right')
            plt.xlabel('Distance A → ')
            plt.ylabel('Lick rate')

            return A_A_diff, A_B_diff_list, licked_As, licked_Bs_list, fig
        
        else:
            return A_A_diff, A_B_diff_list, licked_As, licked_Bs_list, None

    def calc_time_hit_fa(self, bins=10, plot=True):
        '''Calculate hit and fa rates based on time spent between landmarks'''

        lick_position, lick_times, reward_times, reward_positions, release_df = self.session.get_event_parsed()
        target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = self.session.find_targets_distractors()
        hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = self.calc_hit_fa()

        release_positions = np.sort(np.concatenate([target_positions, distractor_positions]))

        # Bin time 
        dt, _ = self.get_time_between_landmarks(bins, plot=False)
        time_bins = np.linspace(np.floor(np.nanmin(dt)), np.ceil(np.nanmax(dt)), bins+1, dtype=int)
        bin_idx = np.digitize(dt, time_bins) - 1
        
        # Calculate hit and fa rates 
        hit_rate = {}
        fa_rate = {}

        for t in np.unique(bin_idx):
            lms_considered = np.where(bin_idx == t)[0]

            target_pos_considered = [pos for pos in target_positions if pos in release_positions[lms_considered]]
            distractor_pos_considered = [pos for pos in distractor_positions if pos in release_positions[lms_considered]]
            
            licked_target = np.zeros(len(target_pos_considered))
            for idx, pos in enumerate(target_pos_considered):
                if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                    licked_target[idx] = 1

            licked_distractor = np.zeros(len(distractor_pos_considered))
            for idx, pos in enumerate(distractor_pos_considered):
                if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                    licked_distractor[idx] = 1

            hit_rate[t] = (
                np.sum(licked_target) / len(licked_target)
                if len(licked_target) > 0 else np.nan
            )

            fa_rate[t] = (
                np.sum(licked_distractor) / len(licked_distractor)
                if len(licked_distractor) > 0 else np.nan
            )

        if plot:
            with mpl.rc_context({
                'axes.titlesize': 10,
                'axes.labelsize': 10,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
            }):
                fig = plt.figure(figsize=(6,3))
                plt.plot(hit_rate.keys(), hit_rate.values(), c='darkblue', linewidth=2, label='hit rate')
                plt.plot(fa_rate.keys(), fa_rate.values(), c='orange', linewidth=2, label='fa rate')
                plt.ylim([0,1.1])
                plt.yticks([0,0.5,1])
                plt.xticks([np.min(bin_idx), np.max(bin_idx)], labels=[f'{time_bins[0]}-{time_bins[1]}', f'{time_bins[-2]}-{time_bins[-1]}'])
                plt.xlabel(f'Time between landmarks (s)')
                ax = plt.gca()
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.legend(frameon=False)

            return hit_rate, fa_rate, fig
        
        else:
            return hit_rate, fa_rate, None

    def calc_distance_from_rew_hit_fa(self, plot=True, remove_disengagement=False, plot_disengagement=False):
        '''Calculate hit and fa rates based on distance from preceding **rewarded** A'''
        
        lick_position, lick_times, reward_times, reward_positions, release_df = self.session.get_event_parsed()
        target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = self.session.find_targets_distractors()
        A_landmarks, B_landmarks, A_idx, B_idx = self.session.get_A_B_landmarks()

        hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = self.calc_hit_fa()

        # Find number of landmarks between two consecutive As
        num_Bs = len(distractor_positions[(distractor_positions > target_positions[0]) & (distractor_positions < target_positions[1])])

        # Find distances between A and the following Bs
        A_A_diff, A_B_diff, A_positions, B_positions = self.session.find_A_B_distance_and_positions(rewarded_As=True)

        # Calculate hit rate
        licked_As = np.zeros((len(reward_positions) - 1))
        for i, pos in enumerate(A_positions):
            if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                licked_As[i] = 1

        # Calculate false alarm rates
        licked_Bs = np.zeros((len(reward_positions) - 1, num_Bs))
        for i, pos in enumerate(B_positions):
            for j in range(num_Bs):
                if np.any((lick_position > pos[j]) & (lick_position < (pos[j] + self.session.lm_size))):
                    licked_Bs[i, j] = 1
        
        # Filter out disengaged trials 
        if remove_disengagement:
            _, _, _, _, [valid_mask_A, valid_mask_B], _ = self.calc_time_from_rew_hit_fa(plot=False, remove_disengagement=remove_disengagement, plot_disengagement=plot_disengagement)
            
            # Apply the masks for A->A
            A_A_diff = A_A_diff[valid_mask_A]
            licked_As = licked_As[valid_mask_A]

        # Apply the masks for A->Bs
        A_B_diff_list = [A_B_diff[valid_mask_A, i] if remove_disengagement else A_B_diff[:, i] for i in range(num_Bs)]
        licked_Bs_list = [licked_Bs[valid_mask_A, i] if remove_disengagement else licked_Bs[:, i] for i in range(num_Bs)]
        
        # A_B_diff_list = [A_B_diff[valid_mask_B[:, i], i] if remove_disengagement else A_B_diff[:, i] for i in range(num_Bs)]
        # licked_Bs_list = [licked_Bs[valid_mask_B[:, i], i] if remove_disengagement else licked_Bs[:, i] for i in range(num_Bs)]
        
        if plot:
            all_B = np.concatenate(A_B_diff) if len(A_B_diff) > 0 else np.array([])
            all_distances = np.concatenate([A_A_diff, all_B.flatten()])
            bins = np.linspace(np.min(all_distances), np.max(all_distances), 20)

            fig = plt.figure(figsize=(6,4))

            cA, mA, sA = compute_binned_lick_rate(A_A_diff, licked_As, bins)
            plt.errorbar(cA, mA, yerr=sA, label='A', marker='o', color='darkblue')

            for i in range(num_Bs):
                if i == 0:
                    color = 'orange'
                elif i == 1:
                    color = 'gold'
                elif i == 2:
                    color = 'brown'
                c, m, s = compute_binned_lick_rate(A_B_diff_list[i], licked_Bs_list[i], bins)
                plt.errorbar(c, m, yerr=s, label=f'B{i+1}', marker='o', color=color)

            plt.ylim([0,1.1])
            plt.yticks([0,0.5,1])
            ax = plt.gca()
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.legend(frameon=False, loc='lower right')
            plt.xlabel('Distance A (rewarded) → ')
            plt.ylabel('Lick rate')

            return A_A_diff, A_B_diff_list, licked_As, licked_Bs_list, fig
        
        else:
            return A_A_diff, A_B_diff_list, licked_As, licked_Bs_list, None

    def calc_distance_from_A1rew_lick_rate(self, plot=True, remove_disengagement=False, plot_disengagement=False):
        '''Calculate hit and fa rates based on distance from preceding **rewarded** A1'''
        
        lick_position, lick_times, reward_times, reward_positions, release_df = self.session.get_event_parsed()
        
        num_As, num_Bs = self.session.get_num_A_B()

        # Find distances between A and the following Bs
        A_A_diff, A_B_diff, A_positions, B_positions = self.session.find_all_A_B_distance_and_positions()

        num_rew_A1 = len(A_positions)
        
        # Calculate hit rate
        licked_As = np.zeros((num_rew_A1, num_As))
        for i, pos in enumerate(A_positions):
            for j in range(num_As):
                if np.any((lick_position > pos[j]) & (lick_position < (pos[j] + self.session.lm_size))):
                    licked_As[i, j] = 1

        # Calculate false alarm rates
        licked_Bs = np.zeros((num_rew_A1, num_Bs))
        for i, pos in enumerate(B_positions):
            for j in range(num_Bs):
                if np.any((lick_position > pos[j]) & (lick_position < (pos[j] + self.session.lm_size))):
                    licked_Bs[i, j] = 1
        
        # Filter out disengaged trials TODO
        # if remove_disengagement:
        #     _, _, _, _, [valid_mask_A, valid_mask_B], _ = calc_time_from_rew_hit_fa(sess_dataframe, ses_settings, plot=False, remove_disengagement=remove_disengagement, plot_disengagement=plot_disengagement)
            
        #     # Apply the masks for A->A
        #     A_A_diff = A_A_diff[valid_mask_A]
        #     licked_As = licked_As[valid_mask_A]

        # # Apply the masks for A->Bs
        # A_B_diff_list = [A_B_diff[valid_mask_A, i] if remove_disengagement else A_B_diff[:, i] for i in range(num_Bs)]
        # licked_Bs_list = [licked_Bs[valid_mask_A, i] if remove_disengagement else licked_Bs[:, i] for i in range(num_Bs)]
        
        A_A_diff_list = [A_A_diff[:, i] for i in range(num_As)]
        A_B_diff_list = [A_B_diff[:, i] for i in range(num_Bs)]
        licked_As = [licked_As[:, i] for i in range(num_As)]
        licked_Bs = [licked_Bs[:, i] for i in range(num_Bs)]
        
        if plot:
            all_distances = np.concatenate([A_A_diff_list, A_B_diff_list])
            bins = np.linspace(np.min(all_distances), np.max(all_distances), 20)

            colors = {
                "A": ["darkblue", "mediumblue"],
                "A1": ["darkblue", "mediumblue"],
                "A2": ["blue", "dodgerblue"],
                "B": ["orange", "gold"],
                "B1": ["orange", "gold"],
                "B2": ["gold", "yellow"],
                "B3": ["brown", "sandybrown"]
            }

            fig = plt.figure(figsize=(6,4))
            
            A_order = [1, 0] if num_As == 2 else range(num_As)
            for i, A_val in enumerate(A_order):
                label = "A" if (num_As == 1 and i == 0) else f"A{i+1}"
                c, m, s = compute_binned_lick_rate(A_A_diff_list[A_val], licked_As[A_val], bins)
                plt.errorbar(c, m, yerr=s, label=label, marker='o', color=colors[label][0])

            for i in range(num_Bs):
                label = "B" if (num_Bs == 1 and i == 0) else f"B{i+1}"
                c, m, s = compute_binned_lick_rate(A_B_diff_list[i], licked_Bs[i], bins)
                plt.errorbar(c, m, yerr=s, label=label, marker='o', color=colors[label][0])

            plt.ylim([0,1.1])
            plt.yticks([0,0.5,1])
            ax = plt.gca()
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.legend(frameon=False, loc='lower right')
            plt.xlabel('Distance A (rewarded) → ')
            plt.ylabel('Lick rate')

            return A_A_diff_list, A_B_diff_list, licked_As, licked_Bs, fig
        
        else:
            return A_A_diff_list, A_B_diff_list, licked_As, licked_Bs, None
        
    def calc_distance_from_rew_lick_frac(self, plot=True):
        '''Calculate fraction of trials per distance from **rewarded** A that were the mouse's first lick'''
        
        lick_position, lick_times, reward_times, reward_positions, release_df = self.session.get_event_parsed()
        target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = self.session.find_targets_distractors()

        # Find number of landmarks between two consecutive As
        num_Bs = len(distractor_positions[(distractor_positions > target_positions[0]) & (distractor_positions < target_positions[1])])

        # Find distances between A and the following Bs
        A_A_diff, A_B_diff, A_positions, B_positions = self.session.find_A_B_distance_and_positions(rewarded_As=True)
        
        following_positions = np.array([
            np.sort(np.concatenate([np.atleast_1d(A_positions[i]), B_positions[i]]))
            for i in range(len(A_positions))
        ])

        # Find where first lick after a reward occured
        lm_licked = np.zeros_like(following_positions)
        for i, positions in enumerate(following_positions):
            for j, pos in enumerate(positions):
                if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                    lm_licked[i, j] = 1
                    break

        # Apply the masks for A->Bs
        A_B_diff_list = [A_B_diff[:, i] for i in range(num_Bs)]

        if plot:
            all_B = np.concatenate(A_B_diff) if len(A_B_diff) > 0 else np.array([])
            all_distances = np.concatenate([A_A_diff, all_B.flatten()])
            bins = np.linspace(np.min(all_distances), np.max(all_distances), 20)

            fig = plt.figure(figsize=(6,4))

            cA, mA, sA = compute_binned_lick_rate(A_A_diff, lm_licked[:,3], bins)
            plt.errorbar(cA, mA, yerr=sA, label='A', marker='o', color='darkblue')

            for i in range(num_Bs):
                if i == 0:
                    color = 'orange'
                elif i == 1:
                    color = 'gold'
                elif i == 2:
                    color = 'brown'
                c, m, s = compute_binned_lick_rate(A_B_diff_list[i], lm_licked[:,i], bins)
                plt.errorbar(c, m, yerr=s, label=f'B{i+1}', marker='o', color=color)

            plt.ylim([0,1.1])
            plt.yticks([0,0.5,1])
            ax = plt.gca()
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.legend(frameon=False, loc='lower right')
            plt.xlabel('Distance A (rewarded) → ')
            plt.ylabel('Fraction of trials first licked')

            return A_A_diff, A_B_diff_list, lm_licked, fig
        
        else:
            return A_A_diff, A_B_diff_list, lm_licked, None

    def get_time_between_landmarks(self, bins=20, plot=True):
        '''Calculate time spent between different landmark types (AA, BB or AB)'''

        if 'LM_Count' in self.dataframe.columns:
            release_df = self.session.estimate_lm_events()
        else:
            release_df = self.session.estimate_release_events()

        _, _, A_idx, B_idx = self.session.get_A_B_landmarks()

        A_dt = release_df.index[release_df['Index'].isin(A_idx)].to_series().diff().dt.total_seconds().to_numpy()
        B_dt = release_df.index[release_df['Index'].isin(B_idx)].to_series().diff().dt.total_seconds().to_numpy()
        dt = release_df.index.to_series().diff().dt.total_seconds().to_numpy()
        dt = dt[~np.isnan(dt)]

        # Calculate and plot histograms of time between landmarks 
        time_bins = np.linspace(np.floor(np.nanmin(dt)), np.ceil(np.nanmax(dt)), bins+1, dtype=int)
        time_bins_A = np.linspace(np.floor(np.nanmin(A_dt)), np.ceil(np.nanmax(A_dt)), bins+1, dtype=int)
        time_bins_B = np.linspace(np.floor(np.nanmin(B_dt)), np.ceil(np.nanmax(B_dt)), bins+1, dtype=int)

        if plot:
            fig = plt.figure(figsize=(3,3))
            _ = plt.hist(dt, bins=time_bins, alpha=0.5, color='grey', label='AB')
            _ = plt.hist(A_dt, bins=time_bins_A, alpha=0.5, color='darkblue', label='AA')
            _ = plt.hist(B_dt, bins=time_bins_B, alpha=0.5, color='orange', label='BB')
            plt.xlabel('Time between landmarks (s)')
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.legend()

            print(f'min time between landmarks {np.nanmin(dt):.2f}\nmax time between landmarks: {np.nanmax(dt):.2f}')
            print(f'\nmin time between A {np.nanmin(A_dt):.2f}\nmax time between A: {np.nanmax(A_dt):.2f}')
            print(f'\nmin time between B {np.nanmin(B_dt):.2f}\nmax time between B: {np.nanmax(B_dt):.2f}')

            return dt, fig
        
        else:
            return dt, None

    def calc_time_from_A_hit_fa(self, plot=True, remove_disengagement=False, plot_disengagement=False):
        '''Calculate hit and fa rates for each time difference'''

        lick_position, lick_times, reward_times, reward_positions, release_df = self.session.get_event_parsed()
        target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = self.session.find_targets_distractors()
        A_landmarks, B_landmarks, A_idx, B_idx = self.session.get_A_B_landmarks()

        # Find number of landmarks between two consecutive As
        num_Bs = len(distractor_positions[(distractor_positions > target_positions[0]) & (distractor_positions < target_positions[1])])

        # Find time difference between As and between an A and the following Bs
        A_times = release_df.loc[release_df['Index'].isin(A_idx)].index.sort_values().to_numpy()
        B_times = release_df.loc[release_df['Index'].isin(B_idx)].index.sort_values().to_numpy()
        
        A_A_dt = np.zeros((len(A_times) - 1))
        A_B_dt = np.zeros((len(A_times) - 1, num_Bs))
        for i in range(len(A_times) - 1):
            A_A_dt[i] = (A_times[i + 1] - A_times[i]) / np.timedelta64(1, 's')

            # Bs between this A and next A
            mask = (B_times > A_times[i]) & (B_times < A_times[i + 1])
            Bs_between = B_times[mask]
            
            for j in range(min(len(Bs_between), num_Bs)):
                A_B_dt[i, j] = (Bs_between[j] - A_times[i]) / np.timedelta64(1, 's')

        # Calculate hit rate
        licked_As = np.zeros((len(target_positions) - 1))
        for i, pos in enumerate(target_positions[:-1]):
            if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                licked_As[i] = 1
        
        # Define B positions
        B_positions = np.zeros((len(target_positions) - 1, num_Bs))
        for i, pos in enumerate(target_positions[:-1]):
            following_Bs = distractor_positions[(distractor_positions > pos) & (distractor_positions < target_positions[i + 1])]
            for j in range(num_Bs):
                B_positions[i, j] = following_Bs[j]
        
        # Calculate false alarm rates
        licked_Bs = np.zeros((len(target_positions) - 1, num_Bs))
        for i, pos in enumerate(B_positions):
            for j in range(num_Bs):
                if np.any((lick_position > pos[j]) & (lick_position < (pos[j] + self.session.lm_size))):
                    licked_Bs[i, j] = 1

        # Determine if the mouse stopped engaging with the task at some point
        if remove_disengagement:
            # 1. Filter out trials where A->lm took too long
            disengagement_idx_A, disengagement_idx_B = get_disengagement_periods(A_A_dt, A_B_dt, plot=plot_disengagement)
            valid_mask_A = np.ones(len(A_A_dt), dtype=bool)
            if disengagement_idx_A is not None and len(disengagement_idx_A) > 0:
                valid_mask_A[disengagement_idx_A] = False
                
            valid_mask_B = np.ones((len(A_A_dt), A_B_dt.shape[1]), dtype=bool)
            for i, ix in enumerate(disengagement_idx_B):
                if ix is not None and len(ix) > 0:
                    valid_mask_B[ix, i] = False
            
            # 2. Filter out trials after which the mouse was not licking
            cutoff_event = get_response_end(licked_As)
            if cutoff_event is not None:
                valid_mask_A[cutoff_event:] = False
                valid_mask_B[cutoff_event:, :] = False

            # Apply the masks for A->A
            A_A_dt = A_A_dt[valid_mask_A]
            licked_As = licked_As[valid_mask_A]    
        
        # Apply the masks for A->Bs
        # A_B_dt_list = [A_B_dt[valid_mask_B[:, i], i] if remove_disengagement else A_B_dt[:, i] for i in range(num_Bs)]
        # licked_Bs_list = [licked_Bs[valid_mask_B[:, i], i] if remove_disengagement else licked_Bs[:, i] for i in range(num_Bs)]
        
        A_B_dt_list = [A_B_dt[valid_mask_A, i] if remove_disengagement else A_B_dt[:, i] for i in range(num_Bs)]
        licked_Bs_list = [licked_Bs[valid_mask_A, i] if remove_disengagement else licked_Bs[:, i] for i in range(num_Bs)]
        
        if plot:
            all_B = np.concatenate(A_B_dt_list) if len(A_B_dt_list) > 0 else np.array([])
            all_dts = np.concatenate([A_A_dt, all_B.flatten()])
            bins = np.linspace(np.min(all_dts), np.max(all_dts), 20)

            fig = plt.figure(figsize=(6,4))

            cA, mA, sA = compute_binned_lick_rate(A_A_dt, licked_As, bins)
            plt.errorbar(cA, mA, yerr=sA, label='A', marker='o', color='darkblue')

            for i in range(num_Bs):
                if i == 0:
                    color = 'orange'
                elif i == 1:
                    color = 'gold'
                elif i == 2:
                    color = 'brown'
                c, m, s = compute_binned_lick_rate(A_B_dt_list[i], licked_Bs_list[i], bins)
                plt.errorbar(c, m, yerr=s, label=f'B{i+1}', marker='o', color=color)

            plt.ylim([0,1.1])
            plt.yticks([0,0.5,1])
            ax = plt.gca()
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.legend(frameon=False, loc='lower right')
            plt.xlabel('Time A → ')
            plt.ylabel('Lick rate')

            if remove_disengagement:
                return A_A_dt, A_B_dt_list, licked_As, licked_Bs_list, [valid_mask_A, valid_mask_B], fig
            else:
                return A_A_dt, A_B_dt_list, licked_As, licked_Bs, None, fig
        
        else:
            if remove_disengagement:
                return A_A_dt, A_B_dt_list, licked_As, licked_Bs_list, [valid_mask_A, valid_mask_B], None
            else:
                return A_A_dt, A_B_dt_list, licked_As, licked_Bs_list, None, None

    def calc_time_from_rew_hit_fa(self, plot=True, remove_disengagement=False, plot_disengagement=False):
        '''Calculate hit and fa rates for each time difference from a **rewarded** A'''

        lick_position, lick_times, reward_times, reward_positions, release_df = self.session.get_event_parsed()
        target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = self.session.find_targets_distractors()
        A_landmarks, B_landmarks, A_idx, B_idx = self.session.get_A_B_landmarks()

        # Find number of landmarks between two consecutive As
        num_Bs = len(distractor_positions[(distractor_positions > target_positions[0]) & (distractor_positions < target_positions[1])])

        # Find distances between A and the following Bs
        A_A_diff, A_B_diff, A_positions, B_positions = self.session.find_A_B_distance_and_positions(rewarded_As=True)
        
        # Find time difference between As and between an A and the following Bs
        A_times = release_df.loc[release_df['Index'].isin(A_idx)].index.sort_values().to_numpy()
        B_times = release_df.loc[release_df['Index'].isin(B_idx)].index.sort_values().to_numpy()
        
        A_A_dt = np.zeros((len(reward_times) - 1))
        A_B_dt = np.zeros((len(reward_times) - 1, num_Bs))
        for i, time in enumerate(reward_times[:-1]):
            following_A = A_times[(A_times > time)][0]
            A_A_dt[i] = (following_A - time) / np.timedelta64(1, 's')

            # Bs between this A and next A
            mask = (B_times > time) & (B_times < following_A)
            following_Bs = B_times[mask]

            for j in range(min(len(following_Bs), num_Bs)):
                A_B_dt[i, j] = (following_Bs[j] - time) / np.timedelta64(1, 's')

        # Calculate hit rate
        licked_As = np.zeros((len(reward_positions) - 1))
        for i, pos in enumerate(A_positions):
            if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                licked_As[i] = 1

        # Calculate false alarm rates
        licked_Bs = np.zeros((len(reward_positions) - 1, num_Bs))
        for i, pos in enumerate(B_positions):
            for j in range(num_Bs):
                if np.any((lick_position > pos[j]) & (lick_position < (pos[j] + self.session.lm_size))):
                    licked_Bs[i, j] = 1
        
        # Determine if the mouse stopped engaging with the task at some point
        if remove_disengagement:
            # 1. Filter out trials where A->lm took too long
            disengagement_idx_A, disengagement_idx_B = get_disengagement_periods(A_A_dt, A_B_dt, plot=plot_disengagement)
            valid_mask_A = np.ones(len(A_A_dt), dtype=bool)
            if disengagement_idx_A is not None and len(disengagement_idx_A) > 0:
                valid_mask_A[disengagement_idx_A] = False
                
            valid_mask_B = np.ones((len(A_A_dt), A_B_dt.shape[1]), dtype=bool)
            for i, ix in enumerate(disengagement_idx_B):
                if ix is not None and len(ix) > 0:
                    valid_mask_B[ix, i] = False
            
            # 2. Filter out trials after which the mouse was not licking
            cutoff_event = get_response_end(licked_As)
            if cutoff_event is not None:
                valid_mask_A[cutoff_event:] = False
                valid_mask_B[cutoff_event:, :] = False

            # Apply the masks for A->A
            A_A_dt = A_A_dt[valid_mask_A]
            licked_As = licked_As[valid_mask_A]    
        
        # Apply the masks for A->Bs
        # A_B_dt_list = [A_B_dt[valid_mask_B[:, i], i] if remove_disengagement else A_B_dt[:, i] for i in range(num_Bs)]
        # licked_Bs_list = [licked_Bs[valid_mask_B[:, i], i] if remove_disengagement else licked_Bs[:, i] for i in range(num_Bs)]
        
        A_B_dt_list = [A_B_dt[valid_mask_A, i] if remove_disengagement else A_B_dt[:, i] for i in range(num_Bs)]
        licked_Bs_list = [licked_Bs[valid_mask_A, i] if remove_disengagement else licked_Bs[:, i] for i in range(num_Bs)]
        
        if plot:
            all_B = np.concatenate(A_B_dt_list) if len(A_B_dt_list) > 0 else np.array([])
            all_dts = np.concatenate([A_A_dt, all_B.flatten()])
            bins = np.linspace(np.min(all_dts), np.max(all_dts), 20)

            fig = plt.figure(figsize=(6,4))

            cA, mA, sA = compute_binned_lick_rate(A_A_dt, licked_As, bins)
            plt.errorbar(cA, mA, yerr=sA, label='A', marker='o', color='darkblue')

            for i in range(num_Bs):
                if i == 0:
                    color = 'orange'
                elif i == 1:
                    color = 'gold'
                elif i == 2:
                    color = 'brown'
                c, m, s = compute_binned_lick_rate(A_B_dt_list[i], licked_Bs_list[i], bins)
                plt.errorbar(c, m, yerr=s, label=f'B{i+1}', marker='o', color=color)

            plt.ylim([0,1.1])
            plt.yticks([0,0.5,1])
            ax = plt.gca()
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.legend(frameon=False, loc='lower right')
            plt.xlabel('Time A → ')
            plt.ylabel('Lick rate')

            if remove_disengagement:
                return A_A_dt, A_B_dt_list, licked_As, licked_Bs_list, [valid_mask_A, valid_mask_B], fig
            else:
                return A_A_dt, A_B_dt_list, licked_As, licked_Bs, None, fig
        
        else:
            if remove_disengagement:
                return A_A_dt, A_B_dt_list, licked_As, licked_Bs_list, [valid_mask_A, valid_mask_B], None
            else:
                return A_A_dt, A_B_dt_list, licked_As, licked_Bs_list, None, None

    def calc_distance_from_rew_p_lick(self, plot=True, remove_disengagement=False):
        '''
        1. Probability of first lick at each landmark type per distance bin from **rewarded** A
        (probabilities sum to 1 within each distance bin)
        2. Number of first licks per landmark type per distance bin
        3. Number of trials per landmark type per distance bin
        4. Number of available trials (where no other lm was licked before) per landmark type per distance bin
        '''
        
        lick_position, lick_times, reward_times, reward_positions, release_df = self.session.get_event_parsed()
        target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = self.session.find_targets_distractors()
        num_lms = len(lm_ids)

        # Find number of landmarks between two consecutive As
        num_Bs = len(distractor_positions[(distractor_positions > target_positions[0]) & (distractor_positions < target_positions[1])])
        
        if num_Bs == 1:
            num_lms = 2 # abab

        # Find distances between A and the following Bs
        A_A_diff, A_B_diff, A_positions, B_positions = self.session.find_A_B_distance_and_positions(rewarded_As=True)
        
        if remove_disengagement:
            # We want to disregard trials where the mouse got a reward and disengaged until the next A, so all intermediate Bs are excluded based on valid As
            _, _, _, _, [valid_mask_A, valid_mask_B], _ = self.calc_time_from_rew_hit_fa(plot=False, remove_disengagement=remove_disengagement, plot_disengagement=False)
            
            # Apply the masks for A->A
            A_A_diff = A_A_diff[valid_mask_A]
            A_positions = A_positions[valid_mask_A]
            B_positions = B_positions[valid_mask_A, :]

        # Apply the masks for A->Bs
        A_B_diff_list = [A_B_diff[valid_mask_A, i] if remove_disengagement else A_B_diff[:, i] for i in range(num_Bs)]
        
        following_positions = np.array([
            np.sort(np.concatenate([np.atleast_1d(A_positions[i]), np.atleast_1d(B_positions[i])]))
            for i in range(len(A_positions))
        ])

        # Find where first lick after a reward occurred
        lm_licked = np.zeros_like(following_positions)
        for i, positions in enumerate(following_positions):
            for j, pos in enumerate(positions):
                if np.any((lick_position > pos) & (lick_position < (pos + self.session.lm_size))):
                    lm_licked[i, j] = 1
                    break
        
        # Move the last column (A) to the first column
        # lm_licked = np.hstack([
        #     lm_licked[:, -1:],   # A (last column)
        #     lm_licked[:, :-1]    # all B columns
        # ])

        # Bin distances
        all_B = np.concatenate(A_B_diff) if len(A_B_diff) > 0 else np.array([])
        all_distances = np.concatenate([A_A_diff, all_B.flatten()])
        bins = np.linspace(np.min(all_distances), np.max(all_distances), 20)

        bin_idx = []
        for i in range(num_Bs):
            bin_idx.append(np.digitize(A_B_diff_list[i], bins))
        bin_idx.append(np.digitize(A_A_diff, bins))
        
        # Count how many trials occurred at each distance bin 
        distance_bin_counts = np.array([
            [np.sum(bin_idx[i] == b) for b in range(1, len(bins))] for i in range(num_lms)
        ])

        # Count trials unattempted until each lm 
        avail_distance_bin_counts = np.array([
            [np.sum((bin_idx[i] == b) & (~np.any(lm_licked[:, :i], axis=1))) 
                for b in range(1, len(bins))] for i in range(num_lms)], dtype=int)
        # avail_distance_bin_counts = np.array([[
        #         np.sum((bin_idx[i] == b) & (np.sum(np.delete(lm_licked, i, axis=1), axis=1) == 0)) 
        #         for b in range(1, len(bins))] for i in range(num_lms)], dtype=int)
        
        # Count first licks per lm type per distance bin
        lm_lick_counts = []
        for i in range(num_lms):
            lm_lick_counts.append(np.array([np.sum(lm_licked[:,i][bin_idx[i] == b])
                for b in range(1, len(bins))], dtype=int)) 
        lm_lick_counts = np.array(lm_lick_counts)

        # Total licks per bin across all landmark types
        total_lick_counts = np.sum(lm_lick_counts, axis=0)

        # Per distance bin, count probability of first lick at each landmark type
        lm_lick_prob = []
        for i in range(num_lms):
            lm_lick_prob.append(lm_lick_counts[i] / total_lick_counts)
        lm_lick_prob = np.array(lm_lick_prob)
        
        ## Plotting
        if plot:
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # 1. Plot probability of lick per landmark type per bin
            fig1 = plot_data(x=bin_centers, y=lm_lick_prob, all_distances=all_distances, ylabel='Probability of first lick')

            # 2. Plot total lick counts per landmark type per distance bin
            fig2 = plot_data(x=bin_centers, y=lm_lick_counts, all_distances=all_distances, ylabel='Number of licks')
            
            # 3. Number of trials per distance bin
            fig3 = plot_data(x=bin_centers, y=distance_bin_counts, all_distances=all_distances, ylabel='Number of trials')
        
            # 4. Number of trials first licked per distance bin
            fig4 = plot_data(x=bin_centers, y=avail_distance_bin_counts, all_distances=all_distances, ylabel='Number of available trials')
        
            return A_A_diff, A_B_diff_list, lm_licked, lm_lick_counts, lm_lick_prob, distance_bin_counts, avail_distance_bin_counts, fig1, fig2, fig3, fig4
        
        else:
            return A_A_diff, A_B_diff_list, lm_licked, lm_lick_counts, lm_lick_prob, distance_bin_counts, avail_distance_bin_counts, None, None, None, None

    def calc_transition_matrix(self):
    
        target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = self.session.find_targets_distractors()
        hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = self.calc_hit_fa()
        ideal_licks = self.get_ideal_performance()

        num_landmarks = len(self.session.trial['landmarks'])

        lick_sequence = lm_id_sequence[licked_all==1]
        ideal_sequence = lm_id_sequence[ideal_licks==1]

        # stimulus transition matrix
        transition_matrix = np.zeros((num_landmarks, num_landmarks))
        for i in range(len(lm_id_sequence)-1):
            current_lm = int(lm_id_sequence[i])
            next_lm = int(lm_id_sequence[i+1])
            transition_matrix[current_lm, next_lm] += 1

        # lick transition matrix
        lick_tm = np.zeros((num_landmarks, num_landmarks))
        for i in range(len(lick_sequence)-1):
            current_lm = int(lick_sequence[i])
            next_lm = int(lick_sequence[i+1])
            lick_tm[current_lm, next_lm] += 1

        # ideal transition matrix
        ideal_tm = np.zeros((num_landmarks, num_landmarks))
        for i in range(len(ideal_sequence)-1):
            current_lm = int(ideal_sequence[i])
            next_lm = int(ideal_sequence[i+1])
            ideal_tm[current_lm, next_lm] += 1

        # print(f'target ids {target_id} and distractor ids {distractor_id}')

        return transition_matrix, lick_tm, ideal_tm

    def calc_distance_transition_matrix(self, binning=True, distance_groups=None):
        '''
        Create a lick transition matrix based on distance between current and next licked landmark.
        If binning, distances will be grouped into small, medium, large.
        '''

        num_landmarks = len(self.session.trial['landmarks'])

        A_landmarks, B_landmarks, A_idx, B_idx = self.session.get_A_B_landmarks()
        target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = self.session.find_targets_distractors()
        hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = self.calc_hit_fa()
        ideal_licks = self.get_ideal_performance()

        # =========================
        # --- DISTANCE GROUPING ---
        # =========================
        A_A_diff, A_B_diff_list, licked_As, licked_Bs_list, _ = self.calc_distance_from_A_hit_fa(plot=False)

        # Reshape A_B_diff in the order in which Bs appear
        A_B_diff_all = [
            int(A_B_diff_list[j][i])
            for i in range(len(A_B_diff_list[0]))
            for j in range(len(A_B_diff_list))
        ]

        # Find common distances (where different performance could only be explained by a knowledge of lm type)
        if distance_groups is None:
            distance_range = np.intersect1d(np.unique(A_A_diff), np.unique(A_B_diff_all))
            
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
        # --- COMPUTE TRANSITION MATRICES ---
        # =========================
        transition_matrix = {k: np.zeros((num_landmarks, num_landmarks)) for k in distance_groups}
        lick_tm = {k: np.zeros((num_landmarks, num_landmarks)) for k in distance_groups}
        ideal_tm = {k: np.zeros((num_landmarks, num_landmarks)) for k in distance_groups}

        lick_idx = np.where(licked_all)[0]
        ideal_idx = np.where(ideal_licks == 1)[0]

        for group, dist_vals in distance_groups.items():
            mask = np.isin(A_A_diff, dist_vals)
            A_indices = np.where(mask)[0]   # indices in A-space

            for i in A_indices:
                A_lm = A_landmarks[i]
                current_lm = lm_id_sequence[A_lm]
                next_lm = lm_id_sequence[A_lm + 1]
                transition_matrix[group][current_lm, next_lm] += 1

                if A_lm in lick_idx:
                    lick_seq_idx = np.where(lick_idx == A_lm)[0]
                    if len(lick_seq_idx) > 0 and (lick_seq_idx + 1 != len(lick_idx)):
                        next_lick_idx = lick_idx[lick_seq_idx + 1][0]                    
                        next_lm = lm_id_sequence[next_lick_idx]
                        lick_tm[group][current_lm, next_lm] += 1

                if A_lm in ideal_idx:
                    ideal_seq_idx = np.where(ideal_idx == A_lm)[0]
                    if (len(ideal_seq_idx) > 0) and (ideal_seq_idx + 1 != len(ideal_idx)):
                        next_ideal_idx = ideal_idx[ideal_seq_idx + 1][0]
                        next_lm = lm_id_sequence[next_ideal_idx]
                        ideal_tm[group][current_lm, next_lm] += 1

            
            mask = np.isin(A_B_diff_all, dist_vals)
            B_indices = np.where(mask)[0] + A_landmarks[0]  # shift indices from B-space to lm-space

            for i in B_indices:
                B_lm = B_landmarks[i]
                current_lm = lm_id_sequence[B_lm]
                next_lm = lm_id_sequence[B_lm + 1]
                transition_matrix[group][current_lm, next_lm] += 1

                if B_lm in lick_idx:
                    lick_seq_idx = np.where(lick_idx == B_lm)[0]
                    if (len(lick_seq_idx) > 0) and (lick_seq_idx + 1 != len(lick_idx)):
                        next_lick_idx = lick_idx[lick_seq_idx + 1][0]
                        next_lm = lm_id_sequence[next_lick_idx]
                        lick_tm[group][current_lm, next_lm] += 1

                if B_lm in ideal_idx:
                    ideal_seq_idx = np.where(ideal_idx == B_lm)[0]
                    if (len(ideal_seq_idx) > 0) and (ideal_seq_idx + 1 != len(ideal_idx)):
                        next_ideal_idx = ideal_idx[ideal_seq_idx + 1][0]
                        next_lm = lm_id_sequence[next_ideal_idx]
                        ideal_tm[group][current_lm, next_lm] += 1

        return transition_matrix, lick_tm, ideal_tm

    def get_ideal_performance(self):

        target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = self.session.find_targets_distractors()
        
        targets = np.unique(target_id)
        ideal_licks = np.zeros_like(lm_id_sequence, dtype=int)
        target_counter = 0
        for i in range(len(lm_id_sequence)):

            if lm_id_sequence[i] == targets[target_counter]:
                ideal_licks[i] = 1  # ideal lick on target
                if target_counter < len(targets) - 1:
                    target_counter += 1  # switch to the next target
                else:
                    target_counter = 0  # reset to the first target
            else:
                ideal_licks[i] = 0  # no lick on distractor

        return ideal_licks

    def calc_conditional_matrix(self, n_steps=1):
        '''Calculate the transition probabilities given reward n_steps ahead of the reward'''

        hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = self.calc_hit_fa()
        target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = self.session.find_targets_distractors()

        goals = list(target_id)
        all_lms = lm_id_sequence

        licked_lm_ix = np.where(licked_all == 1)[0]

        transition_licks = np.zeros((np.unique(goals).shape[0], np.unique(lm_ids).shape[0]))
        transition_prob = np.zeros((np.unique(goals).shape[0], np.unique(lm_ids).shape[0]))
        
        control_licks = np.zeros((np.unique(goals).shape[0], np.unique(lm_ids).shape[0]))
        control_prob = np.zeros((np.unique(goals).shape[0], np.unique(lm_ids).shape[0]))

        ideal_prob = np.zeros((np.unique(goals).shape[0], np.unique(lm_ids).shape[0]))

        for g in range(np.unique(goals).shape[0]):
            goal = goals[g]

            rewards = np.intersect1d(np.where(rewarded_all == 1)[0], np.where(all_lms == goal)[0])

            for i, reward in enumerate(rewards[:-1]):
                # 1. Transition probability
                if len(licked_lm_ix[licked_lm_ix > reward]) >= n_steps:
                    lick_index = licked_lm_ix[licked_lm_ix > reward][n_steps-1]
                lm = all_lms[lick_index].astype(int)
                
                # position in matrix according to order in AB sequence
                lm_pos = np.where(lm_ids == lm)[0]
                transition_licks[g, lm_pos] += 1
                
                # convert to probability
                transition_prob[g] = transition_licks[g] / np.sum(transition_licks[g], axis=0)
                
                # 2. Control probability - lick at next 
                next_control_index = reward + 1
                lm = all_lms[next_control_index].astype(int)
                
                # position in matrix according to order in AB sequence
                lm_pos = np.where(lm_ids == lm)[0]
                control_licks[g, lm_pos] += 1

                # convert to probability
                control_prob[g] = control_licks[g] / np.sum(control_licks[g], axis=0)
                
        # 3. Ideal probabilities         
        for g in range(np.unique(goals).shape[0]):
            next_goal = goals[g+1] if g+1 < len(goals) else goals[0]

            # position in matrix according to order in AB sequence
            next_goal_pos = np.where(lm_ids == next_goal)[0] 
            ideal_prob[g, next_goal_pos] += 1

        return transition_prob, control_prob, ideal_prob

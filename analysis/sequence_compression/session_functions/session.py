import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from functools import cached_property
from session_functions.utils import *

np.set_printoptions(suppress=True, precision=2)

class Session():
    def __init__(self, ses_settings, sess_dataframe, session_path, animal, cohort, session_id):
        self.settings = ses_settings
        self.dataframe = sess_dataframe
        self.session_path = session_path
        self.animal = animal 
        self.cohort = cohort
        self.session_id = session_id

    @cached_property
    def trial(self):
        trial = self.settings['trial']
        if isinstance(trial, list):
            trial = trial[0]['trial']
        return trial

    @cached_property
    def lm_size(self):
        return self.trial['landmarks'][0][0]['size']

    @cached_property
    def reward_seq(self):
        seq = np.array([lm[0]['rewardSequencePosition'] for lm in self.trial['landmarks']])
        if 'full' in self.session_id:
            return seq
        else:
            return seq[:4] if len(seq) > 4 else seq

    @cached_property
    def sequence(self):
        reward_seq = self.reward_seq
        if len(reward_seq) == 4:
            if np.diff(reward_seq)[0] == 0:
                return 'AABB'
            elif len(np.where(reward_seq == -1)[0]) > 2:
                return 'ABBB'
            return 'ABAB'
        elif len(reward_seq) == 3:
            As = np.where(reward_seq == 0)[0]
            return 'ABB' if len(As) == 1 else 'AAB'
        else:
            return 'full'
    
    @cached_property
    def sess(self):

        # Use the Buffer as datapoint idx
        position = np.nan_to_num(self.dataframe['Position'].values, nan=0.0)
        speed = np.nan_to_num(self.dataframe['Treadmill'].values, nan=0.0)
        licks = self.dataframe['Licks'].values.astype(int)
        rewards = self.dataframe['Buffer'][self.dataframe['Rewards'].notna()].values    
        lick_threshold = self.settings['velocityThreshold']

        sess = {'position': position,
                'speed': speed,
                'licks': licks, 
                'rewards': rewards,
                'lick_threshold': lick_threshold,
                }
        
        return sess

    def get_event_parsed(self, threshold='below'):

        if threshold == 'below':
            licks = threshold_lick_events(self.sess, self.dataframe, below=True)
        elif threshold == 'above':
            licks = threshold_lick_events(self.sess, self.dataframe, below=False)
        elif threshold == 'all':
            licks = self.dataframe['Licks'].values
        lick_position = self.dataframe['Position'].values[licks > 0]
        lick_times = self.dataframe.index[licks > 0]
        reward_times = self.dataframe.index[self.dataframe['Rewards'].notna()]
        reward_positions = self.dataframe['Position'].values[self.dataframe['Rewards'].notna()]

        if 'LM_Count' in self.dataframe.columns:
            release_df = self.estimate_lm_events()
        else:
            release_df = self.estimate_release_events()

        # Fix the order of the first events
        lm_idx = np.asarray(release_df['Index'].to_numpy(), dtype=int)
        position = np.nan_to_num(self.dataframe['Position'].values, nan=0.0)
        release_positions = position[lm_idx]
        
        if len(self.reward_seq) == 4:
            if np.diff(self.reward_seq)[0] == 0:    
                # AABB: re-order AB so that A is always first
                release_df = release_df[2:]
                
            elif len(np.where(self.reward_seq == -1)[0]) > 2:   
                # ABBB: get rid of first event if needed otherwise keep the order the same
                if release_positions[0] < self.session.lm_size:
                    release_df = release_df[1:]
            else:    
                # ABAB: re-order AB so that A is always first
                release_df = release_df[1:]

        if len(self.reward_seq) == 3:
            # ABB: get rid of first event if needed otherwise keep the order the same
            if release_positions[0] < 2:
                release_df = release_df[1:]
        
        return lick_position, lick_times, reward_times, reward_positions, release_df

    def sanity_check_parsing(self):

        lick_position, lick_times, reward_times, reward_positions, release_df = self.get_event_parsed()
        event_ids = release_df["Odour"].to_numpy(dtype=int)
        n_ids = len(event_ids) - (len(event_ids) % 10)
        event_ids = event_ids[:n_ids]
        #reshape ids to have 10 columns (one for each target)
        event_ids_reshaped = event_ids.reshape(-1, 10)
        event_ids_reshaped

        plt.figure(figsize=(10,4))
        plt.imshow(event_ids_reshaped, aspect='auto', cmap='viridis_r', interpolation='none')
        plt.clim(0, np.max(event_ids))
        plt.colorbar()
        plt.title('Released Odour IDs')
        plt.xlabel('Landmark Index')
        plt.ylabel('Lap')
        plt.show()

    def find_targets_distractors(self):
        '''Give an id to each type of landmark'''

        # Give ids to each type of landmark 
        # distractor_id = np.arange(0, len(np.where(reward_seq == -1)[0])) #[0,1]
        # target_id = np.arange(distractor_id[-1] + 1, len(np.where(reward_seq != -1)[0]) + distractor_id[-1] + 1)

        # Define order of landmark ids
        lm_id = np.arange(len(self.reward_seq))
        target_idx = np.where(self.reward_seq == 0)[0] 
        distractor_idx = np.where(self.reward_seq == -1)[0]

        if self.sequence == 'AABB':
            if self.reward_seq[0] == -1:
                distractor_id = lm_id[distractor_idx] + 2
                target_id = lm_id[target_idx] - 2
            else:
                distractor_id = lm_id[distractor_idx]
                target_id = lm_id[target_idx]

        elif self.sequence == 'ABBB':
            distractor_id = np.atleast_1d(lm_id[1:])
            target_id = np.atleast_1d(lm_id[0])

        elif self.sequence == 'ABAB':
            if self.reward_seq[0] == -1:
                distractor_id = lm_id[distractor_idx] + 1
                target_id = lm_id[target_idx] - 1
            else:
                distractor_id = lm_id[distractor_idx]
                target_id = lm_id[target_idx]

        elif self.sequence == 'ABB':
            distractor_id = np.atleast_1d(lm_id[len(target_idx):])
            target_id = np.atleast_1d(lm_id[0:len(target_idx)])

        elif self.sequence == 'AAB':
            distractor_id = np.atleast_1d(lm_id[-1] - 1)
            target_id = [lm_id[target_idx[0]], lm_id[target_idx[1]] + 1]
        
        A_landmarks, B_landmarks, A_idx, B_idx = self.get_A_B_landmarks()

        # Get sequence of landmark ids 
        lm_id_sequence = np.zeros(len(A_landmarks) + len(B_landmarks), dtype=int)
        if self.sequence == 'ABBB':
            lm_id_sequence[A_landmarks] = np.tile(target_id, len(A_landmarks))
            lm_id_sequence[B_landmarks] = np.tile(distractor_id, int(np.ceil(len(B_landmarks)/2)))[:len(B_landmarks)]
        elif self.sequence == 'AABB' or self.sequence == 'ABAB':
            lm_id_sequence[A_landmarks] = np.tile(target_id, int(np.ceil(len(A_landmarks)/2)))[:len(A_landmarks)]
            lm_id_sequence[B_landmarks] = np.tile(distractor_id, int(np.ceil(len(B_landmarks)/2)))[:len(B_landmarks)]
        elif self.sequence == 'ABB':
            lm_id_sequence[A_landmarks] = np.tile(target_id, len(A_landmarks))
            lm_id_sequence[B_landmarks] = np.tile(distractor_id, int(np.ceil(len(B_landmarks)/2)))[:len(B_landmarks)]
        elif self.sequence == 'AAB':
            lm_id_sequence[A_landmarks] = np.tile(target_id, int(np.ceil(len(A_landmarks)/2)))[:len(A_landmarks)]
            lm_id_sequence[B_landmarks] = np.tile(distractor_id, len(B_landmarks))

        # Get landmark visits
        lick_position, lick_times, reward_times, reward_positions, release_df = self.get_event_parsed()
        lm_idx = np.asarray(release_df['Index'].to_numpy(), dtype=int)
        
        # Get positions of targets and distractors
        position = np.nan_to_num(self.dataframe['Position'].values, nan=0.0)

        release_positions = position[lm_idx]
        # release_positions = release_df['Position'].to_numpy()     # less accurate

        target_positions = release_positions[A_landmarks]
        distractor_positions = release_positions[B_landmarks]

        return target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence

    def get_num_A_B(self):
        '''Count the number of As and Bs in the binary pattern'''
        target_id, distractor_id, target_positions, distractor_positions, lm_ids, _ = self.find_targets_distractors()

        num_lms = len(lm_ids)
        num_As = len(target_id)

        # special case: ABAB
        if (
            len(target_id) == 2
            and num_lms % 2 == 0
            and (target_id[1] - target_id[0]) % num_lms == num_lms // 2
        ):
            num_As = 1
            num_Bs = 1
        else:
            num_Bs = num_lms - num_As

        return num_As, num_Bs
    
    def get_A_B_landmarks(self):
        '''Find which landmarks are rewarded (A) or non-rewarded (B)'''
        from itertools import zip_longest

        # Get landmark visits
        lick_position, lick_times, reward_times, reward_positions, release_df = self.get_event_parsed()
        lm_idx = np.asarray(release_df['Index'].to_numpy(), dtype=int) # TODO rename because it conflicts with another definition
   
        sorting = True
        A_landmarks = list(np.where(self.reward_seq == 0)[0])

        # Split As and Bs into subtypes
        if self.cohort == 1:
            if self.sequence == 'AABB':
                A_landmarks = [i - 2 for i, r in enumerate(self.reward_seq) if r == 0]
                B_landmarks = [i + 2 for i, r in enumerate(self.reward_seq) if r == -1]

            elif self.sequence == 'ABBB':
                if A_landmarks[0] == 0:
                    A_landmarks[0] = 3 
                B_landmarks = [i for i in range(len(self.reward_seq)) if (i not in A_landmarks)]

            elif self.sequence == 'ABAB':
                A_landmarks = [i - 1 for i, r in enumerate(self.reward_seq) if r == 0]
                B_landmarks = [i + 1 for i, r in enumerate(self.reward_seq) if r == -1]
            
            elif self.sequence == 'ABB':
                A_landmarks[0] = 2
                B_landmarks = [i for i in range(len(self.reward_seq)) if (i not in A_landmarks)]

            elif self.sequence == 'AAB':   
                sorting = False
                seq1 = list(range(2, len(lm_idx), len(self.reward_seq)))
                seq2 = list(range(0, len(lm_idx), len(self.reward_seq)))
                
                A_landmarks = []
                for a, b in zip_longest(seq1, seq2):
                    if a is not None:
                        A_landmarks.append(a)
                    if b is not None:
                        A_landmarks.append(b)
                A_landmarks = np.sort(A_landmarks)

                B_landmarks = [i for i in range(len(self.reward_seq)) if (i not in A_landmarks)]

        elif self.cohort == 2:
            # there should be an offset, so the first lm can be considered
            if self.sequence == 'AABB':
                A_landmarks = [i - 2 for i, r in enumerate(self.reward_seq) if r == 0]
                B_landmarks = [i + 2 for i, r in enumerate(self.reward_seq) if r == -1]

            elif self.sequence == 'ABBB':
                if A_landmarks[0] == 0:
                    A_landmarks[0] = 3 
                B_landmarks = [i for i in range(len(self.reward_seq)) if (i not in A_landmarks)]

            elif self.sequence == 'ABAB': 
                if 'initialCorridorOffset' in self.settings and self.settings['initialCorridorOffset'] > 0:
                    A_landmarks = [i for i, r in enumerate(self.reward_seq) if r == 0]
                    B_landmarks = [i for i, r in enumerate(self.reward_seq) if r == -1]
                else:
                    A_landmarks = [i - 1 for i, r in enumerate(self.reward_seq) if r == 0]
                    B_landmarks = [i + 1 for i, r in enumerate(self.reward_seq) if r == -1]
                                
            elif self.sequence == 'ABB':
                A_landmarks[0] = 2
                B_landmarks = [i for i in range(len(self.reward_seq)) if (i not in A_landmarks)]

            elif self.sequence == 'AAB':   
                sorting = False
                seq1 = list(range(2, len(lm_idx), len(self.reward_seq)))
                seq2 = list(range(0, len(lm_idx), len(self.reward_seq)))
                
                A_landmarks = []
                for a, b in zip_longest(seq1, seq2):
                    if a is not None:
                        A_landmarks.append(a)
                    if b is not None:
                        A_landmarks.append(b)
                A_landmarks = np.sort(A_landmarks)

                B_landmarks = [i for i in range(len(self.reward_seq)) if (i not in A_landmarks)]

        if sorting:     
            for a in range(len(np.where(self.reward_seq == 0)[0])):
                A_landmarks.extend([i for i in range(A_landmarks[a]+len(self.reward_seq), len(lm_idx), len(self.reward_seq)) if i < len(lm_idx)])
        for b in range(len(np.where(self.reward_seq == -1)[0])):
            B_landmarks.extend([i for i in range(B_landmarks[b]+len(self.reward_seq), len(lm_idx), len(self.reward_seq)) if i < len(lm_idx)])
        
        if sorting:
            A_landmarks = np.sort(A_landmarks)
        B_landmarks = np.sort(B_landmarks)

        # Split the data indices into subtypes
        A_idx = [lm_idx[i] for i in A_landmarks]
        B_idx = [lm_idx[i] for i in B_landmarks]

        assert len(lm_idx) == (len(A_landmarks) + len(B_landmarks)), 'Some landmarks are missing!'

        return A_landmarks, B_landmarks, A_idx, B_idx

    def get_A_B_splits(self):

        A_landmarks, B_landmarks, A_idx, B_idx = self.get_A_B_landmarks()

        # TODO adapt for omissions
        reward_seq = np.array([lm[0]['rewardSequencePosition'] for lm in self.trial['landmarks']])

        A1 = A_landmarks
        A2 = []
        B2 = []
        B3 = []

        if self.sequence == 'ABB':
            B1 = B_landmarks[::2]
            B2 = B_landmarks[1::2]
        elif self.sequence == 'AAB':
            # note for AAB the first lm is A2
            A1 = A_landmarks[1::2]
            A2 = A_landmarks[::2]
            B1 = B_landmarks
        elif self.sequence == 'ABBB':  
            B1 = B_landmarks[::3]
            B2 = B_landmarks[1::3]
            B3 = B_landmarks[2::3]
        elif self.sequence == 'AABB': 
            A1 = A_landmarks[::2]
            A2 = A_landmarks[1::2]
            B1 = B_landmarks[::2]
            B2 = B_landmarks[1::2]
        elif self.sequence == 'ABAB':  
            A1 = A_landmarks
            B1 = B_landmarks
        
        return A1, A2, B1, B2, B3

    def find_A_B_distance_and_positions(self, rewarded_As=False):
        '''Find the positions of As (either rewarded or not) and the consecutive Bs and the distance between them'''
        
        lick_position, lick_times, reward_times, reward_positions, release_df = self.get_event_parsed()
        target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = self.find_targets_distractors()
        
        num_As, num_Bs = self.get_num_A_B()
        # num_Bs = len(distractor_positions[(distractor_positions > target_positions[0]) & (distractor_positions < target_positions[1])])

        # Find distances between A and the following Bs
        if rewarded_As == True:
            A_A_diff = np.zeros((len(reward_positions) - 1))
            A_B_diff = np.zeros((len(reward_positions) - 1, num_Bs))
            A_positions = np.zeros((len(reward_positions) - 1))
            B_positions = np.zeros((len(reward_positions) - 1, num_Bs))

            for i, pos in enumerate(reward_positions[:-1]):
                mask = (np.round(target_positions, 1) > np.round(pos, 1)) & (np.round(target_positions, 1) <= np.round(reward_positions[i + 1], 1))
                following_A = target_positions[mask][0]
                A_positions[i] = following_A
                A_A_diff[i] = np.round(following_A - pos)

                # Keep Bs from current A (or reward) up to the next A
                following_Bs = distractor_positions[(distractor_positions > pos) & (distractor_positions < following_A)]
                for j in range(num_Bs):
                    B_positions[i, j] = following_Bs[j]
                    A_B_diff[i, j] = np.round(B_positions[i, j] - pos)

        else:
            A_A_diff = np.zeros((len(target_positions) - 1))
            A_B_diff = np.zeros((len(target_positions) - 1, num_Bs))
            A_positions = np.zeros((len(target_positions) - 1))
            B_positions = np.zeros((len(target_positions) - 1, num_Bs))

            for i, pos in enumerate(target_positions[:-1]):
                following_A = target_positions[i + 1]
                A_positions[i] = following_A
                A_A_diff[i] = np.round(following_A - pos)

                # Keep Bs from current A (or reward) up to the next A
                following_Bs = distractor_positions[(distractor_positions > pos) & (distractor_positions < following_A)]
                for j in range(num_Bs):
                    B_positions[i, j] = following_Bs[j]
                    A_B_diff[i, j] = np.round(B_positions[i, j] - pos)
                
        return A_A_diff, A_B_diff, A_positions, B_positions

    def find_all_A_B_distance_and_positions(self):
        '''Find the positions of rewarded A1s and the consecutive As and Bs and the distance between them'''
        
        lick_position, lick_times, reward_times, reward_positions, release_df = self.get_event_parsed()
        target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = self.find_targets_distractors()
        A_landmarks, B_landmarks, _, _ = self.get_A_B_landmarks()
        release_positions = np.sort(np.concatenate([target_positions, distractor_positions]))

        num_lms = len(lm_ids)
        num_As = len(target_id)

        # special case: ABAB
        if (
            len(target_id) == 2
            and num_lms % 2 == 0
            and (target_id[1] - target_id[0]) % num_lms == num_lms // 2
        ):
            num_As = 1
            num_Bs = 1
        else:
            num_Bs = num_lms - num_As

        # Consider A1 rewards only 
        A1, A2, B1, B2, B3 = self.get_A_B_splits()

        # determine which A1s were rewarded
        rewarded_A1_positions = []
        for pos in release_positions[A1]:
            if np.any((reward_positions > pos) & (reward_positions <= pos + self.lm_size)):
                rewarded_A1_positions.append(pos)
            
        # Find distances between A1 and the following As and Bs 
        A_A_diff = np.zeros((len(rewarded_A1_positions) - 1, num_As))
        A_B_diff = np.zeros((len(rewarded_A1_positions) - 1, num_Bs))
        A_positions = np.zeros((len(rewarded_A1_positions) - 1, num_As))
        B_positions = np.zeros((len(rewarded_A1_positions) - 1, num_Bs))
        
        for i, pos in enumerate(rewarded_A1_positions[:-1]):
            # Keep As from current A1 (or reward) up to the next A1
            mask = (np.round(target_positions, 1) > np.round(pos, 1)) & (np.round(target_positions, 1) <= np.round(rewarded_A1_positions[i + 1], 1))
            following_As = np.sort(target_positions[mask][:num_As])
            A_positions[i] = following_As
            A_A_diff[i] = np.round(following_As - pos)
            
            # Keep Bs from current A (or reward) up to the next A
            following_Bs = distractor_positions[(distractor_positions > pos) & (distractor_positions < following_As[-1])]
            B_positions[i] = following_Bs
            A_B_diff[i] = np.round(B_positions[i] - pos)

        return A_A_diff, A_B_diff, A_positions, B_positions

    def parse_rew_lms(self):
        rew_odour = []
        rew_texture = []
        non_rew_odour = []
        non_rew_texture = []
        index = []

        for i in self.ses_settings['trial']['landmarks']:
            for j in i:
                if j['rewardSequencePosition'] > -1:
                    if not np.isin(j['rewardSequencePosition'], index): # avoid double counting of odours
                        rew_odour.append(j['odour'])
                        rew_texture.append(j['texture'])
                        index.append(j['rewardSequencePosition'])
                else:
                    non_rew_odour.append(j['odour'])
                    non_rew_texture.append(j['texture'])

        rew_odour = np.array(rew_odour)[np.argsort(index)]
        rew_texture = np.array(rew_texture)[np.argsort(index)]
        non_rew_odour = np.unique(non_rew_odour)
        non_rew_texture = np.unique(non_rew_texture)
        non_rew_odour = non_rew_odour[non_rew_odour != 'odour0']
        non_rew_texture = non_rew_texture[non_rew_texture != 'grey']

        return rew_odour, rew_texture, non_rew_odour, non_rew_texture

    def estimate_lm_events(self):

        lm_position = self.dataframe['LM_Position'].values[self.dataframe['LM_Count'].values >= 0]

        lm_time = self.dataframe.index[self.dataframe['LM_Count'].values >= 0]

        lm_odour = self.dataframe['LM_Odour'].values[self.dataframe['LM_Count'].values >= 0]
        lm_odour = [extract_int(odour) for odour in lm_odour]

        lm_index = self.dataframe['Buffer'].values[self.dataframe['LM_Count'].values >= 0]

        lm_df = pd.DataFrame({
            'time': lm_time,
            'Position': lm_position,
            'Index': lm_index,
            'Odour': lm_odour
        }).set_index('time')

        if lm_df['Position'].iloc[0] != 0:
            # Add initial landmark at position 0 if not present
            initial_lm = pd.DataFrame({
                'time': [pd.NaT],
                'Position': [0],
                'Index': [0], #'Index': [-1],
                'Odour': [0]  # Assume first odour is the initial one
            }).set_index('time')
            lm_df = pd.concat([initial_lm, lm_df]).reset_index().set_index('time')

        return lm_df

    def estimate_release_events(self):
        
        offset = self.trial['offsets'][0]

        lm_gap = self.session.lm_size + offset 

        tmp = self.dataframe.reset_index(drop=False, inplace=False)
        release_subset = tmp[tmp['Events'].str.contains('release', na=False) & ~tmp['Events'].str.contains('odour0', na=False)][['Events', 'Position']]
        release_subset = release_subset.dropna(subset='Events', how='all')

        release_subset_pos = release_subset['Position'].to_numpy()

        # Step 1: Make empty df to store results
        df = pd.DataFrame(np.nan, index=range(1000), columns=["pos", "idx", "released_odour"])
        last_val = release_subset_pos[-1]
        # Fill positions from the bottom upwards
        # This because there are less drifts as sessions progress
        for i in range(len(df)):
            df.loc[len(df)-1 - i, "pos"] = last_val - lm_gap * i

        # Step 2: Find release from idx match (strongest crteria, but it works!)
        for i in reversed(df.index):
            pos_val = df.at[i, "pos"]
            if np.isnan(pos_val):
                continue  # skip rows where pos is NaN

            # find index of closest-position row in events_df
            idx_closest = (tmp["Position"] - pos_val).abs().idxmin()
            event_closest = tmp.loc[idx_closest, "Events"]
            pos_closest = tmp.loc[idx_closest, "Position"]

            # ONLY fill df if this event is a release event
            if isinstance(event_closest, str) and event_closest.startswith("release"):
                df.at[i, "idx"] = idx_closest
                df.at[i, "released_odour"] = extract_int(event_closest)
                df.at[i, "pos"] = pos_closest
            else:
                df.at[i, "idx"] = idx_closest # Only store possible candidates

        # Step 3: Clean df by removing neagtive pos rows
        last_negative_idx = df[df["pos"] < 0].index.max() -1 # keep the last one, just in case
        df = df.loc[last_negative_idx+1:].reset_index(drop=True)

        # Step 4: Find closest release events. If there are multiple release, use earliest
        for i in reversed(df.index):
            if ~np.isnan(df.at[i, "released_odour"]):
                continue # we have already identified odour
            else:
                closed_idx = int(df.at[i, "idx"])
                chosen_idx, _, odour, chosen_pos = find_closest_events(tmp, closed_idx, pos_window = self.lm_size /2, event_priority=["release"], choose = "earliest")
                if odour is not None:
                    df.at[i, "idx"] = chosen_idx
                    df.at[i, "released_odour"] = odour
                    df.at[i, "pos"] = chosen_pos

        # Step 5: Find closest prepare and flush events.
        for i in reversed(df.index):
            if ~np.isnan(df.at[i, "released_odour"]):
                continue # we have already identified odour
            else:
                closed_idx = int(df.at[i, "idx"])
                chosen_idx, _, odour, chosen_pos = find_closest_events(tmp, closed_idx, pos_window = self.lm_size /2, event_priority=["prepare", "flush"], choose = "average")
                if odour is not None:
                    df.at[i, "idx"] = chosen_idx
                    df.at[i, "released_odour"] = odour
                    df.at[i, "pos"] = chosen_pos

        # Step 6: Clean the output format
        result = []
        for i, row in df.iterrows():
            if pd.isna(row["released_odour"]) or int(row["released_odour"]) == 0:
                continue  # no odour released → skip
            idx = int(row["idx"])
            if i == 0 and np.isnan(row["released_odour"]):
                continue # this means nothing was released. We check this at Step 7
            # get timestamp from summary dataframe
            ts = tmp.loc[int(idx), "time"]

            entry = [ts, float(row["pos"]), int(idx), int(row["released_odour"])]
            result.append(entry)

        # Step 7: Add the first odour stimulus that VR ABCD forgot
        # sometimes the VR drops the first release event, check for that and add first element if needed
        first_release = extract_int(self.trial['landmarks'][0][0]['odour'])
        if first_release != 0 and (len(result) == 0 or first_release != result[0][3]):
            result = [[pd.NaT, 0, -1, first_release]] + result

        result_df = pd.DataFrame(result,
                                columns=["time", "Position", "Index", "Odour"]
                                ).set_index("time")

        return result_df
    
    def get_landmarks(self):
        # Get landmark visits (full corridor)
        _, _, _, _, release_df = self.get_event_parsed()
        lm_idx = np.asarray(release_df['Index'].to_numpy(), dtype=int) # TODO rename because it conflicts with another definition
        landmarks = np.arange(len(lm_idx))
        
        return landmarks, lm_idx

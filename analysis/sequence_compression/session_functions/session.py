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
        
        if self.sequence == 'AABB':
            # AABB re-order AB so that A is always first
            release_df = release_df[2:]
                
        elif self.sequence == 'ABBB':   
            # ABBB: get rid of first event if needed otherwise keep the order the same
            if release_positions[0] < self.session.lm_size:
                release_df = release_df[1:]

        elif self.sequence == 'ABAB':    
            release_df = release_df[1:]

        elif self.sequence == 'ABB':
            # ABB: get rid of first event if needed otherwise keep the order the same
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
            if self.cohort == 1:
                if self.reward_seq[0] == -1:
                    distractor_id = lm_id[distractor_idx] + 1
                    target_id = lm_id[target_idx] - 1
                else:
                    distractor_id = lm_id[distractor_idx]
                    target_id = lm_id[target_idx]
            elif self.cohort == 2: 
                if 'initialCorridorOffset' in self.settings and self.settings['initialCorridorOffset'] > 0:
                    distractor_id = lm_id[distractor_idx]
                    target_id = lm_id[target_idx]
                else:
                    distractor_id = lm_id[distractor_idx] + 1
                    target_id = lm_id[target_idx] - 1
            
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
                    A_landmarks = [i - 1 for i, r in enumerate(self.reward_seq) if r == 0]
                    B_landmarks = [i + 1 for i, r in enumerate(self.reward_seq) if r == -1]
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

        if self.cohort == 1:
            # Add initial landmark at position 0 if not present
            if lm_df['Position'].iloc[0] != 0:
                initial_lm = pd.DataFrame({
                    'time': [pd.NaT],
                    'Position': [0],
                    'Index': [0], #'Index': [-1],
                    'Odour': [0]  # Assume first odour is the initial one
                }).set_index('time')
                lm_df = pd.concat([initial_lm, lm_df]).reset_index().set_index('time')
        elif self.cohort == 2:
            if 'initialCorridorOffset' in self.settings and lm_df['Position'].iloc[0] != self.settings['initialCorridorOffset']:
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

class ABCD_Session():
    def __init__(self, ses_settings, ses_rig_settings, sess_dataframe, session_path, animal, stage, session_id):
        self.settings = ses_settings
        self.rig_settings = ses_rig_settings
        self.dataframe = sess_dataframe
        self.session_path = session_path
        self.animal = animal 
        self.stage = stage
        self.session_id = session_id

    @cached_property
    def trial(self):
        trial = self.settings['trial']
        if isinstance(trial, list):
            trial = trial[0]['trial']
        return trial

    @cached_property
    def world(self):
        if self.trial['randomiseSequence'] == True:
            world = 'random'
        else:
            world = 'stable'
        return world
    
    @cached_property
    def lm_size(self):
        return self.trial['landmarks'][0][0]['size']

    @cached_property
    def tunnel_length(self):
        landmarks = self.trial['landmarks']
        if len(self.trial['offsets']) == 1:
            offset = self.trial['offsets'][0]
        else:
            print("Cannot compute corridor length when offsets are randomised")
            return None
        
        tunnel_length = 0
        for lm in landmarks:
            tunnel_length += lm[0]['size']
            tunnel_length += offset
        return tunnel_length
    
    @cached_property
    def position_offset(self):
        if 'positionOffset' in self.rig_settings:
            return int(self.rig_settings['positionOffset'])
        else:
            return 0
    
    def create_session_struct(self):

        # Use the Buffer as datapoint idx
        position = np.nan_to_num(self.dataframe['Position'].values, nan=0.0)
        speed = np.nan_to_num(self.dataframe['Treadmill'].values, nan=0.0)
        licks = self.dataframe['Licks'].values.astype(int)
        rewards = self.dataframe['Buffer'][self.dataframe['Rewards'].notna()].values

        if self.world == 'stable':
            goal_ids, lm_ids = self.parse_stable_goal_ids()
        elif self.world == 'random':
            goal_ids, lm_ids = self.parse_random_goal_ids()
        num_landmarks = len(lm_ids) # unique number of lm ids

        lick_threshold = self.settings['velocityThreshold']

        sess = {'position': position,
                'licks': licks, 
                'rewards': rewards, 
                'goal_ids': goal_ids, 
                'lm_ids': lm_ids,
                'num_landmarks': num_landmarks,
                'lick_threshold': lick_threshold,
                'speed': speed}
        
        return sess

    def threshold_licks(self, session):
        # Threshold licks based on speed 
        speed_ok = session['speed'] < session['lick_threshold']
        licked = session['licks'] > 0
        threshold_mask = speed_ok & licked

        licks_idx = np.where(threshold_mask)[0]
        thresholded_licks = np.zeros(len(session['licks']))
        thresholded_licks[licks_idx] = session['licks'][licks_idx]
        # thresholded_licks = session['licks'][licks_idx]

        session['thresholded_licks'] = thresholded_licks
        session['licks_idx'] = licks_idx

        return session
    
    def get_lm_entry_exit(self, session):
        '''Find data idx closest to landmark entry and exit. The results should be similar to estimate_pd_entry_exit.'''

        positions = session['position']

        lm_entry_idx = []
        lm_exit_idx = []

        if np.abs(positions[0] - session['landmarks'][-1,1]) < np.abs(positions[0] - session['landmarks'][0,0]):
            search_start = np.where(positions <= session['all_landmarks'][0,0])[0][-1]  # the mouse accidentally moved backwards first
        else: 
            search_start = 0
        
        for lm_start in session['all_landmarks'][:,0]:
            lm_entry_idx.append(np.where(positions[search_start:] >= lm_start)[0][0] + search_start)
        for lm_end in session['all_landmarks'][:,1]:
            lm_exit_idx.append(np.where(positions[search_start:] <= lm_end)[0][-1] + search_start)

        return np.array(lm_entry_idx), np.array(lm_exit_idx)
        
    def find_targets_distractors(self):
        
        lick_position, lick_times, reward_times, reward_positions, release_df = self.get_event_parsed()
        rew_odour, rew_texture, non_rew_odour, non_rew_texture = self.parse_rew_lms()

        target_id = []
        target_positions = []
        for i in range(len(rew_odour)):
            test_int = extract_int(rew_odour[i])
            matches = release_df[release_df["Odour"] == test_int] # does released odour match with test_int
            pos = matches["Position"].tolist()

            target_id.extend([i] * len(pos))
            target_positions.extend(pos)

        distractor_id = []
        distractor_positions = []
        for i in range(len(non_rew_odour)):
            test_int = extract_int(non_rew_odour[i])
            matches = release_df[release_df["Odour"] == test_int] # does released odour match with test_int
            pos = matches["Position"].tolist()

            distractor_id.extend([i + len(rew_odour)] * len(pos)) # offset distractor IDs
            distractor_positions.extend(pos)
        
        all_release_positions = release_df["Position"].tolist()
        was_target = np.zeros(len(all_release_positions))
        lm_id = np.zeros(len(all_release_positions))
        for idx, pos in enumerate(all_release_positions):
            if pos in target_positions:
                was_target[idx] = 1
                lm_id[idx] = target_id[np.where(np.isclose(target_positions, pos))[0][0]]
            elif pos in distractor_positions:
                was_target[idx] = 0
                lm_id[idx] = distractor_id[np.where(np.isclose(distractor_positions, pos))[0][0]] 
        
        return target_positions, distractor_positions, target_id, distractor_id, was_target, lm_id

    def get_event_parsed(self):

        lick_position = self.dataframe['Position'].values[self.dataframe['Licks'].values > 0]
        lick_times = self.dataframe.index[self.dataframe['Licks'].values > 0]
        reward_times = self.dataframe.index[self.dataframe['Rewards'].notna()]
        reward_positions = self.dataframe['Position'].values[self.dataframe['Rewards'].notna()]

        # if 'LM_Count' in self.dataframe.columns:
        #     release_df = self.estimate_lm_events()
        # else:
        release_df = self.estimate_release_events()

        return lick_position, lick_times, reward_times, reward_positions, release_df

    def parse_rew_lms(self):

        rew_odour = []
        rew_texture = []
        non_rew_odour = []
        non_rew_texture = []
        index = []

        for i in self.trial['landmarks']:
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
        
    def parse_stable_goal_ids(self):
        '''Identify the number of landmarks and goals for stable world sequences'''

        num_lms = len(self.trial['landmarks'])
        num_goals = self.settings['availableRewardPositions']
        lm_ids = np.arange(num_lms)
        goal_counter = 0
        goals = []
        while goal_counter < num_goals:
            for i in range(num_lms):
                for j in self.trial['landmarks'][i]:
                    if j['rewardSequencePosition'] == goal_counter:
                        goals.append(i)
                        goal_counter += 1
                        if goal_counter >= num_goals:
                            break
                        
        return goals, lm_ids

    def parse_random_goal_ids(self):
        '''Identify the number of landmarks and goals for random world sequences'''
        rew_odour, _, non_rew_odour, _ = self.parse_rew_lms()

        num_lms = len(rew_odour) + len(non_rew_odour)
        num_goals = self.settings['availableRewardPositions']
        lm_ids = np.arange(num_lms)

        goal_counter = 0
        goals = []
        while goal_counter < num_goals:
            for i in range(num_lms):
                for j in self.trial['landmarks'][i]:
                    if j['rewardSequencePosition'] == goal_counter:
                        goals.append(i)
                        goal_counter += 1
                        if goal_counter >= num_goals:
                            break

        return goals, lm_ids

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

        if 'initialCorridorOffset' in self.settings and lm_df['Position'].iloc[0] != self.settings['initialCorridorOffset']:     
        # if lm_df['Position'][0] != 0:
            # Add initial landmark at position 0 if not present
            initial_lm = pd.DataFrame({
                'time': [pd.NaT],
                'Position': [0],
                'Index': [-1],
                'Odour': [0]  # Assume first odour is the initial one
            }).set_index('time')
            lm_df = pd.concat([initial_lm, lm_df]).reset_index().set_index('time')

        return lm_df

    def estimate_release_events(self):

        tmp = self.dataframe.reset_index(drop=False, inplace=False)
        release_subset = tmp[tmp['Events'].str.contains('release', na=False) & ~tmp['Events'].str.contains('odour0', na=False)][['Events', 'Position']]
        release_subset = release_subset.dropna(subset='Events', how='all')                
        release_subset_pos = release_subset['Position'].to_numpy() 

        # Remove duplicates 
        release_subset_pos = np.sort(release_subset_pos)
        filtered_positions = [release_subset_pos[0]]

        for pos in release_subset_pos[1:]:
            if pos - filtered_positions[-1] >= 1:
                filtered_positions.append(pos)

        release_subset_pos = np.array(filtered_positions)

        if len(self.trial['offsets']) > 1:
            offsets = [0]
            offsets.extend(np.diff(release_subset_pos))
            offsets = np.round(offsets)

            # Step 1: Make empty df to store results
            df = pd.DataFrame({
                "pos": release_subset_pos,
                "idx": np.nan,
                "released_odour": np.nan,
            })

        else:
            offset = self.trial['offsets'][0] 
            lm_gap = self.lm_size + offset 

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

        # Step 3: Clean df by removing negative pos rows
        negative_rows = df[df["pos"] < 0]

        if not negative_rows.empty:
            last_negative_idx = negative_rows.index.max()
            df = df.loc[last_negative_idx + 1:].reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        # last_negative_idx = df[df["pos"] < 0].index.max() -1 # keep the last one, just in case
        # df = df.loc[last_negative_idx+1:].reset_index(drop=True)

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

        if 'initialCorridorOffset' in self.settings and np.round(result[0][1] + self.lm_size / 2) != self.settings['initialCorridorOffset']:
            if first_release != 0 and (len(result) == 0 or first_release != result[0][3]):
                result = [[pd.NaT, 0, -1, first_release]] + result

        result_df = pd.DataFrame(result,
                                columns=["time", "Position", "Index", "Odour"]
                                ).set_index("time")
        
        return result_df

    def estimate_pd_entry_exit(self, session, pd='pd1'):
        '''Estimate lm entry and exit indices using photodiode data'''
        binary_pd = (session[pd] >= 100).astype(int)

        all_lm_entry_idx = np.where(np.diff(binary_pd) == 1)[0] + 1
        all_lm_exit_idx = np.where(np.diff(binary_pd) == -1)[0] + 1
        if binary_pd[0] == 1:
            all_lm_entry_idx = np.insert(all_lm_entry_idx, 0, 0)

        offset = self.trial['offsets'][0]

        # Filter out repeated lm visits
        entry_pos = session['position'][all_lm_entry_idx]
        exit_pos  = session['position'][all_lm_exit_idx]

        grey_landmarks_included = any(lm['texture'] == 'grey' for group in self.trial['landmarks'] for lm in group)
        if grey_landmarks_included:
            tol = 0
        else:
            if offset == self.lm_size: # TODO bad fix 
                tol = 0
            else:
                tol = 1

        # Filter out re-entries - use earliest idx
        consecutive_diff = np.where(np.diff(entry_pos) < self.lm_size + tol)[0] + 1
        removed = []
        for i, idx in enumerate(all_lm_entry_idx):
            if i in consecutive_diff:
                if session['position'][idx] < offset:
                    continue
                removed.append(i)
        lm_entry_idx = np.delete(all_lm_entry_idx, removed)

        # Filter out re-exits - use latest idx
        if grey_landmarks_included:
            consecutive_diff = np.where(np.diff(exit_pos) < self.lm_size + tol)[0] + 1
        else:
            consecutive_diff = np.where(np.diff(exit_pos) < self.lm_size + tol)[0] 
        removed = []
        for i, idx in enumerate(all_lm_exit_idx):
            if i in consecutive_diff:
                if session['position'][idx] < offset:
                    continue
                removed.append(i)
        lm_exit_idx = np.delete(all_lm_exit_idx, removed)

        return lm_entry_idx, lm_exit_idx
        
    def get_landmark_positions(self, session, data='pd'):
        '''Get the start and end of each landmark'''

        grey_landmarks_included = any(lm['texture'] == 'grey' for group in self.trial['landmarks'] for lm in group)

        # if 'LM_Count' in self.dataframe.columns:
        #     release_df = self.estimate_lm_events()
        # else:
        release_df = self.estimate_release_events()
        
        release_positions = release_df['Position'].values
        
        if data == 'odour':
            landmarks = np.zeros((len(release_positions), 2))
            for lm, pos in enumerate(release_positions):
                landmarks[lm,0] = pos
                landmarks[lm,1] = pos + self.lm_size

        elif data == 'pd':
            lm_entry_idx1, lm_exit_idx1 = self.estimate_pd_entry_exit(session, pd='pd1')
            lm_entry_idx2, lm_exit_idx2 = self.estimate_pd_entry_exit(session, pd='pd2')
            
            entry_pos1 = session['position'][lm_entry_idx1]
            entry_pos2 = session['position'][lm_entry_idx2]
            exit_pos1  = session['position'][lm_exit_idx1]
            exit_pos2  = session['position'][lm_exit_idx2]

            offset = self.trial['offsets'][0] # TODO
            tol = self.lm_size * 0.5

            # Merge with "keep single" logic
            entry_pos = merge_positions_keep_single(entry_pos1, entry_pos2, tol, offset)
            exit_pos  = merge_positions_keep_single(exit_pos1,  exit_pos2,  tol, offset)

            entry_idx = np.array([np.argmin(np.abs(session['position'] - p)) for p in entry_pos])
            exit_idx = np.array([np.argmin(np.abs(session['position'] - p)) for p in exit_pos])

            # --- Pair entries with the NEXT valid exit ---
            min_gap = np.median(np.diff(entry_pos)) - 1
            if grey_landmarks_included:
                # the zone between landmarks is another landmark (grey), which might give rise to strange estimates for landmark entry and exit events
                filtered_exit_idx = []
                last_kept = None

                for i in range(len(exit_pos)):
                    curr_exit = exit_pos[i]
                    if curr_exit < 2:
                        # special case for weird first landmark 
                        filtered_exit_idx.append(exit_idx[i])
                        last_kept = curr_exit
                        continue
                    
                    # --- condition 1: too close to previous valid exit ---
                    if last_kept is not None:
                        if (curr_exit - last_kept) < min_gap:
                            continue

                    # --- condition 2: too close to any entry ---
                    if np.any(np.abs(entry_pos - curr_exit) < 1):
                        continue

                    # --- keep it ---
                    filtered_exit_idx.append(exit_idx[i])
                    last_kept = curr_exit

                all_lm_entry = session['position'][entry_idx]
                all_lm_exit = session['position'][np.array(filtered_exit_idx)]

            else:
                # Fix exits
                filtered_exit_idx = []
                last_kept = None

                for i in range(len(exit_pos)):
                    curr_exit = exit_pos[i]
                    if curr_exit < 2:
                        continue
                    
                    # too close to previous valid event
                    if last_kept is not None:
                        if (curr_exit - last_kept) < min_gap:
                            continue

                    filtered_exit_idx.append(exit_idx[i])
                    last_kept = curr_exit

                # Fix entries
                filtered_entry_idx = []
                last_kept = None

                for i in range(len(entry_pos)):
                    curr_entry = entry_pos[i]
                    if curr_entry < 2:
                        continue
                    
                    # too close to previous valid event
                    if last_kept is not None:
                        if (curr_entry - last_kept) < min_gap:
                            continue

                    filtered_entry_idx.append(entry_idx[i])
                    last_kept = curr_entry

                # special case for weird first landmark 
                early_entry_idx = entry_idx[entry_pos < 2]
                early_exit_idx  = exit_idx[exit_pos < 2]

                filtered_entry_idx = list(early_entry_idx) + filtered_entry_idx
                filtered_exit_idx  = list(early_exit_idx)  + filtered_exit_idx
                all_lm_entry = session['position'][np.array(filtered_entry_idx)]
                all_lm_exit = session['position'][np.array(filtered_exit_idx)]

                
            # Fix last lm - ignore it if odour was delivered and mouse did not exit 
            num_releases = len(release_positions)
            if release_positions[-1] > all_lm_exit[-1]:
                num_releases -= 1
                all_lm_entry = all_lm_entry[:-1]

            # Fix first lm - only consider first entry and last exit idx
            first_entries = all_lm_entry < offset
            first_exits  = all_lm_exit  < offset
            first_entry = all_lm_entry[first_entries][0] 
            first_exit = all_lm_exit[first_exits][-1]
            
            # Concatenate all landmarks 
            lm_entry = np.concatenate([[first_entry], all_lm_entry[~first_entries]])
            lm_exit = np.concatenate([[first_exit], all_lm_exit[~first_exits]])

            # Confirm number of PD events matches odour releases
            release_diff = np.diff(release_positions)
            release_outlier = np.where(release_diff < self.lm_size + offset)[0]

            first_release_idx = 0
            first_release_considered = release_positions[first_release_idx]

            if grey_landmarks_included:
                # the zone between landmarks is another landmark (grey), which might give rise to strange estimates for landmark entry and exit events
                print('Excluding grey landmarks...')

                entry_indices = np.array([np.argmin(np.abs(session['position'] - lm)) for lm in lm_entry])
                exit_indices = np.array([np.argmin(np.abs(session['position'] - lm)) for lm in lm_exit])
                
                # Deal with the first landmark outlier (distance between release events is not consistent)
                if len(release_outlier) > 0:
                    first_release_idx = release_outlier[0] + 1
                    first_release_considered = release_positions[first_release_idx]

                    correct_entry_indices = []
                    if release_outlier[0] == 0:
                        first_pd = np.argmin(np.abs(lm_entry - first_release_considered))
                        correct_entry_indices.append(entry_indices[0])
                        correct_entry_indices.extend(entry_indices[first_pd::2])
                    else:
                        raise ValueError('Not sure how to deal with this for now...')
                else:
                    first_pd = np.argmin(np.abs(lm_entry - release_positions[0]))
                    correct_entry_indices = entry_indices[first_pd::2]

                # Find all exits after valid entries
                correct_exit_indices = []
                for e, entry in enumerate(correct_entry_indices):
                    valid_exits = exit_indices[exit_indices > entry] 
                    if len(valid_exits) == 0:
                        continue
                    correct_exit_indices.append(valid_exits[0]) # take the earliest one

                lm_entry = np.unique(session['position'][correct_entry_indices])
                lm_exit = np.unique(session['position'][correct_exit_indices])

            assert len(lm_entry) == num_releases, 'Something is wrong with the landmark entries'
            assert np.abs(lm_entry[first_release_idx] - first_release_considered) < 1, 'There is a large offset in the landmark entry and odour release positions'

            if len(lm_entry) != len(lm_exit):
                if len(lm_entry) - len(lm_exit) == 1:
                    print('Session ended before the mouse exited the last landmark. Removing...')
                    n = len(lm_exit)
                    lm_entry = lm_entry[:n]
                else:
                    raise ValueError(f'Something is wrong with landmark parsing using the photodiode data in {session['mouse']} {session['stage']}')

            # Store landmarks 
            landmarks = np.column_stack([lm_entry, lm_exit])

        session['landmarks'] = landmarks

        return session

    def get_goal_positions(self, session):
        '''Get the start and end of each goal landmark using odour release events to find targets'''
        target_positions, _, _, _, _, _ = self.find_targets_distractors()

        offset = self.trial['offsets'][0]

        valid_goals = []
        for pos in np.sort(target_positions):
            distance = np.abs(session['landmarks'][:, 0] - pos)
            if np.all(distance > self.lm_size + offset - 1):  # skip if too far from all landmarks
                continue

            closest_lm = np.argmin(distance)
            valid_goals.append(session['landmarks'][closest_lm])

        # Convert to numpy array
        session['goals'] = np.array(valid_goals)
        
        return session
    
    def get_lap_idx(self, session):
        # Divide the session dataframe into laps based on the position and corridor length
        if self.world == 'stable':
            if self.tunnel_length is not None:
                session['num_laps'] = int(np.ceil(session['position'].max() / self.tunnel_length))
            else:
                session['num_laps'] = len(session['landmarks']) // session['num_landmarks']

        elif self.world == 'random':
            session['num_laps'] = 1

        # For each position, determine which lap it belongs to
        if self.tunnel_length is not None:
            session['lap_idx'] = (session['position'] // self.tunnel_length).astype(int)
        else:
            session['lap_idx'] = np.zeros_like(session['position'])

        return session

    def get_lm_idx(self, session):
        # Get landmark idx for each datapoint
        lm_entry, lm_exit = self.get_lm_entry_exit(session)

        lm_idx = np.zeros(len(session['position']))
        for i, (entry, exit) in enumerate(zip(lm_entry, lm_exit)):
            lm_occupancy = np.arange(entry, exit + 1)
            lm_idx[lm_occupancy] = i + 1

        session['lm_idx'] = lm_idx

        return session

    def get_AB_sequence(self, session):
        if self.world == 'stable':
            sequence = 'ABAB'
        elif self.world == 'random':
            sequence = 'AB_shuffled'
        else:
            raise ValueError("Oops I don't know what to do about this type of world")
        
        session['sequence'] = sequence

        return session

    def get_licks_idx(self, session, lick_threshold=True):
        '''Get the idx of licks in the session'''

        if lick_threshold:
            session = self.threshold_licks(session)
        else:
            licks_idx = np.where(session['licks'])[0]
            session['licks_idx'] = licks_idx

        return session 

    def get_licks_per_lap(self, session):
        # Get position and frame index for each lick 
        lick_frames = {}
        lick_positions = {}
        for i in range(session['num_laps']):
            if session['num_laps'] == 1:
                lap_ix = np.where(session['lap_idx'] == i+1)[0]
            else:
                lap_ix = np.where(session['lap_idx'] == i)[0]
            # licks_per_lap_ix = np.intersect1d(lap_ix, session['thresholded_licks'])
            licks_per_lap_ix = np.intersect1d(lap_ix, session['licks_idx'])
            lick_frames[i] = licks_per_lap_ix
            lick_positions[i] = session['position'][licks_per_lap_ix]

        session['licks_per_lap'] = lick_positions
        session['licks_per_lap_frames'] = lick_frames

        return session

    def get_licked_lms(self, session):
        # Get licked landmarks
        licked_lms = np.zeros((session['num_laps'], len(session['landmarks'])))
        
        for i in range(session['num_laps']):
            lap_idx = np.where(session['lap_idx'] == i)[0]
            for j in range(len(session['landmarks'])):
                lm = np.where(session['lm_idx'] == j+1)[0]
                target_ix = np.intersect1d(lap_idx, lm)
                # if session['thresholded_licks'] exists, use that
                if 'thresholded_licks' in session:
                    thresholded_lick_idx = np.where(session["thresholded_licks"] == 1)[0]
                    target_licks = np.intersect1d(target_ix, thresholded_lick_idx)
                    # target_licks = np.intersect1d(target_ix, session['thresholded_licks'])
                # otherwise use all licks
                else:
                    target_licks = np.intersect1d(target_ix, session['licks'])
                if len(target_licks) > 0:
                    licked_lms[i,j] = 1
                else:
                    licked_lms[i,j] = 0

        session['licked_lms'] = licked_lms

        return session

    def get_rewarded_lms(self, session):
        '''Get a binary list of which landmarks were rewarded'''
        rewarded_lms = np.zeros((session['num_laps'], len(session['landmarks'])))

        for i in range(session['num_laps']):
            lap_idx = np.where(session['lap_idx'] == i)[0]
            for j in range(len(session['landmarks'])):
                lm = np.where(session['lm_idx'] == j+1)[0]
                target_ix = np.intersect1d(lap_idx, lm)    
                target_rewards = np.intersect1d(target_ix, session['rewards'])
                if len(target_rewards) > 0:
                    rewarded_lms[i,j] = 1
                else:
                    rewarded_lms[i,j] = 0

        session['rewarded_lms'] = rewarded_lms

        return session

    def get_rewarded_landmarks(self, session):
        '''Find the indices of rewarded (lick-triggered) landmarks.'''

        lm_entry_idx, lm_exit_idx = self.get_lm_entry_exit(session)

        # Find rewarded landmarks 
        reward_positions = session['position'][session['reward_idx']]

        rewarded_landmarks = [i for i, (start, end) in enumerate(zip(np.floor(session['position'][lm_entry_idx]), np.ceil(session['position'][lm_exit_idx]))) 
                                if np.any((np.ceil(reward_positions) >= start) & (np.floor(reward_positions) <= end))] 

        session['rewarded_landmarks'] = rewarded_landmarks

        return session

    def create_odour_lm_mapping(self):
        '''Create a list of rewarded and non-rewarded odours based on the order in which they are created in the session settings file'''
        odour_lm_id_mapping = []
        for lm_list in self.trial['landmarks']:
            for lm in lm_list:
                odour_id = extract_int(lm['odour'])
                if np.isin(odour_id, odour_lm_id_mapping) or odour_id == 0:
                    break
                else:
                    odour_lm_id_mapping.append(odour_id)

        return odour_lm_id_mapping
        
    def get_random_lm_sequence(self):
        '''Create a list with lm ids for each lm in the random world'''
        odour_lm_id_mapping = self.create_odour_lm_mapping()
        _, _, _, _, release_df = self.get_event_parsed()
        release_ids = release_df['Odour'].to_numpy()

        lm_ids_list = np.empty(len(release_ids), dtype=int)
        for i, odour in enumerate(release_ids):
            if odour == 0:
                continue
            matches = np.where(odour_lm_id_mapping == odour)[0]
            if len(matches) == 0:
                lm_ids_list[i] = np.nan
            else:
                lm_ids_list[i] = matches[0]

        return lm_ids_list
        
    def get_lms_visited(self, session):
        # Calculate number of landmarks visited
        if len(np.where(session['landmarks'][:,0] < session['position'][-1])[0]) != len(np.where(session['landmarks'][:,-1] < session['position'][-1])[0]):
            num_lms = len(session['landmarks']) - 1 # session ended before mouse exited last lm entered
        else:
            num_lms = len(session['landmarks'])  

        if self.world == 'stable':
            all_lms = np.array([])  # landmark ids
            for i in range(session['num_laps'] + 1):
                all_lms = np.append(all_lms, session['lm_ids'])
            all_lms = all_lms.astype(int)[:num_lms]
            
        elif self.world == 'random':
            all_lms = self.get_random_lm_sequence()
            all_lms = all_lms[:num_lms]
            
        all_landmarks = session['landmarks']  
        for i in range(1, session['num_laps']):  
            all_landmarks = np.concatenate((all_landmarks, session['landmarks']), axis=0)
        all_landmarks = all_landmarks[:num_lms]  # landmark positions

        # remove lms from last lap if not complete
        all_landmarks = all_landmarks[:len(all_lms)]

        session['all_landmarks'] = all_landmarks
        session['all_lms'] = all_lms

        return session

    def get_reward_idx(self, session):
        '''Get the data indices where rewards were delivered'''

        # Ensure mouse has left last rewarded landmark 
        reward_idx = session['rewards']
        if session['all_landmarks'][-1,1] < session['position'][reward_idx[-1]]:  
            reward_idx = reward_idx[:-1]  
            print('Mouse did not leave the last rewarded landmark. Removing landmark...')

        session['reward_idx'] = reward_idx

        return session 

    def get_landmark_categories(self, session):
        '''Find the landmarks in the entire session that belong to goals, non-goals and test.'''

        session = self.get_landmark_ids(session)

        # Get the landmarks that belong to each condition  
        goals_idx = np.where(np.isin(session['all_lms'], session['goal_landmark_id']))[0]
        non_goals_idx = np.where(np.isin(session['all_lms'], session['non_goal_landmark_id']))[0]
        test_idx = np.where(np.isin(session['all_lms'], session['test_landmark_id']))[0] if session['test_landmark_id'] is not None else None

        session['goals_idx'] = goals_idx
        session['non_goals_idx'] = non_goals_idx
        session['test_idx'] = test_idx

        return session

    def get_landmark_category_entries(self, session):
        '''Find the indices of landmark entry for different types of landmarks: rewarded, miss, non-goal, test.'''
        
        lm_entry_idx, _ = self.get_lm_entry_exit(session)

        # Find category for each landmark 
        session = self.get_landmark_categories(session)

        # Find the rewarded landmarks 
        session = self.get_rewarded_landmarks(session)

        # Find landmark entry indices for each landmark category
        rew_lm_entry_idx = [lm_entry_idx[i] for i in session['rewarded_landmarks']]
        miss_lm_entry_idx = np.array([lm_entry_idx[i] for i in session['goals_idx'] if i not in session['rewarded_landmarks']])
        nongoal_lm_entry_idx = np.array([lm_entry_idx[i] for i in session['non_goals_idx']])
        test_lm_entry_idx = np.array([lm_entry_idx[i] for i in session['test_idx']]) if session['test_idx'] is not None else np.array([])
        goal_lm_entry_idx = [lm_entry_idx[i] for i in session['goals_idx']]

        assert len(rew_lm_entry_idx) + len(miss_lm_entry_idx) + len(nongoal_lm_entry_idx) + len(test_lm_entry_idx) == len(session['all_lms']), 'Some landmarks have not been considered.'

        session['rew_lm_entry_idx'] = rew_lm_entry_idx
        session['miss_lm_entry_idx'] = miss_lm_entry_idx
        session['nongoal_lm_entry_idx'] = nongoal_lm_entry_idx
        session['test_lm_entry_idx'] = test_lm_entry_idx
        session['goal_lm_entry_idx'] = goal_lm_entry_idx

        return session

    def get_landmark_category_rew_idx(self, session):
            '''Find indices also in non-goal landmarks corresponding to the same time after landmark entry as mean reward time lag.'''
    
            session = self.get_landmark_category_entries(session)
            
            # Calculate time lag between landmark entry and reward delivery
            rew_time_lag = np.round(np.mean(session['reward_idx'] - session['rew_lm_entry_idx']))
            print('Reward time lag from lm entry: ', rew_time_lag)
    
            # Find where reward would be on average if these landmarks were rewarded
            miss_rew_idx = session['miss_lm_entry_idx'] + rew_time_lag
            nongoal_rew_idx = session['nongoal_lm_entry_idx'] + rew_time_lag  
            test_rew_idx = session['test_lm_entry_idx'] + rew_time_lag

            session['rew_time_lag'] = rew_time_lag
            session['miss_rew_idx'] = miss_rew_idx
            session['nongoal_rew_idx'] = nongoal_rew_idx
            session['test_rew_idx'] = test_rew_idx
    
            return session

    def get_landmark_ids(self, session):
        '''Define which landmarks belong to goals, non-goals and test.'''
    
        t = extract_int(session['stage'])

        session = self.get_AB_sequence(session)

        if t == 5 or t == 6:
            assert session['num_landmarks'] == 10, 'The number of landmarks in T5 or T6 should be 10.'
            
            if session['sequence'] == 'ABAB':
                goal_landmark_id = np.array([1, 3, 5, 7])
                test_landmark_id = 9
            elif session['sequence'] == 'AABB':  
                goal_landmark_id = np.array([0, 1, 4, 5])
                test_landmark_id = np.array([8, 9])
            non_goal_landmark_id = np.setxor1d(np.arange(0, session['num_landmarks']), np.append(goal_landmark_id, test_landmark_id))

        elif t == 3 or t == 4:
            assert session['num_landmarks'] == 2, 'The number of landmarks in T3 or T4 should be 2.'
            
            lms = np.unique(session['all_lms'])
            goal_mask = [i for i, landmark in enumerate(session['all_landmarks']) if landmark in session['goals']]
            goal_landmark_id = session['all_lms'][goal_mask[0]]
            non_goal_landmark_id = np.setdiff1d(lms, goal_landmark_id)[0]
            test_landmark_id = None

        elif t > 6:
            lms = np.arange(session['num_landmarks'])
            goal_landmark_id = session['goal_idx']
            non_goal_landmark_id = np.setdiff1d(lms, session['goal_idx'])
            test_landmark_id = None

        session['goal_landmark_id'] = goal_landmark_id
        session['non_goal_landmark_id'] = non_goal_landmark_id
        session['test_landmark_id'] = test_landmark_id

        return session

    def get_data_lm_idx(self, session):
        '''Get the landmark id of every data entry'''
        
        # Find landmark entry and exit idx
        lm_entry, lm_exit = self.get_lm_entry_exit(session)

        # Find datapoints within a landmark
        lm_idx = np.zeros(len(session['position']))
        for i in range(len(session['all_lms'])):
            lm_idx[lm_entry[i]:lm_exit[i]+1] = i+1

        session['data_lm_idx'] = lm_idx

        return session
    
    def analyse_session_pre7_behav(self, plot=True):
        '''Wrapper for session analysis using behaviour data'''

        session = self.create_session_struct()
        session = self.get_landmark_positions(session, data='odour')
        session = self.get_goal_positions(session)

        session['mouse'] = self.animal
        session['stage'] = self.stage
        session['world'] = self.world

        save_path = Path(self.session_path) / 'analysis'
        save_path.mkdir(parents=True, exist_ok=True)
        session['save_path'] = save_path

        session = self.get_lap_idx(session)
        session = self.get_lms_visited(session)
        session = self.get_lm_idx(session)
        session = self.get_licks_idx(session) # thresholding is also performed here
        session = self.get_licks_per_lap(session)
        session = self.get_licked_lms(session)
        session = self.get_rewarded_lms(session)
        session = self.get_reward_idx(session)
        # session = get_active_goal(session)
        # session = calc_acceleration(session)
        # session = calculate_frame_lick_rate(session)

        session = self.get_landmark_categories(session)
        session = self.get_rewarded_landmarks(session)
        session = self.get_landmark_category_rew_idx(session)

        # Get behaviour
        # session = get_behaviour(session, sess_dataframe, ses_settings, plot)
        
        self.sess = session 
        
        return session
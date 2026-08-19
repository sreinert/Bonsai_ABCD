from aeon.io.reader import Csv, Reader
import aeon.io.api as aeon
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import datetime
import json
import importlib
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
import re, os, sys
import palettes
import pickle
import seaborn as sns
from pynwb import NWBHDF5IO
np.set_printoptions(suppress=True, precision=2)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import parse_nwb_functions as parse_nwb_functions
importlib.reload(parse_nwb_functions)

'''This is a copy of parse_bonsai_functions with new functions added.'''

#%% ##### Loading #####
class AnalogData(Reader):
    def __init__(self, pattern, columns, channels, extension="bin"):
        super().__init__(pattern, columns, extension)
        self.channels = channels

    def read(self, file):
        data = np.fromfile(file, dtype=np.float64)
        data = np.reshape(data, (-1, self.channels))
        return pd.DataFrame(data, columns=self.columns)

def format_condition_label(cond=None, pattern=None):
    
    if cond is not None:
        # Determine the AB counts
        if 'abb' in cond and 'abbb' not in cond and 'aabb' not in cond:
            base = r"$\mathrm{AB}^2$"
        elif 'abbb' in cond:
            base = r"$\mathrm{AB}^3$"
        elif 'aabb' in cond:
            base = r"$\mathrm{A}^2\mathrm{B}^2$"
        else:
            base = cond
        
        # Preserve random/fixed part
        if 'random' in cond:
            return f"{base} random"
        elif 'fixed' in cond:
            return f"{base} fixed"
        else:
            return base
    
    elif pattern is not None: 
        pattern = np.asarray(pattern)

        symbol_map = {1: "A", 0: "B"}

        parts = []

        current = pattern[0]
        count = 1

        for x in pattern[1:]:
            if x == current:
                count += 1
            else:
                letter = symbol_map[current]
                parts.append(f"{letter}{count}")
                current = x
                count = 1

        # flush last run
        letter = symbol_map[current]
        parts.append(f"{letter}{count}")

        return "".join(parts)
    
from datetime import datetime

def find_base_path(mouse, date, root):
    mouse_path = Path(root) / f"sub-{mouse}"

    candidates = []

    for folder in mouse_path.iterdir():
        if folder.is_dir() and date in folder.name:
            try:
                # Extract timestamp after "date-"
                ts_str = folder.name.split("date-")[-1]
                ts = datetime.strptime(ts_str, "%Y%m%dT%H%M%S")
                candidates.append((ts, folder))
            except ValueError:
                # Skip folders with unexpected formatting
                continue

    if not candidates:
        raise FileNotFoundError(f"No session folder found for {mouse} on {date}")

    # Pick the latest timestamp
    latest_folder = max(candidates, key=lambda x: x[0])[1]
    print(f"Using latest folder: {latest_folder}")

    return latest_folder

def load_settings(base_path):
    settings_path = Path(base_path) / "behav/session-settings/"
    json_files = list(settings_path.glob("*.json")) or False # deal with first few sessions where config was saved as json
    if json_files:
        settings_file = json_files[0]
    else:
        try:
            settings_file = list(settings_path.glob("*.csv"))[0] # This still can be loaded with json.load(). wow.
        except:
            raise FileNotFoundError(f"No valid JSON found in {settings_path}")
    with open(settings_file, 'r') as file:
        settings = json.load(file)
    ses_settings = settings["value"]

    rig_path = Path(base_path) / "behav/rig-settings/"
    rig_file = list(rig_path.glob("*.json"))[0]
    with open(rig_file, 'r') as file:
        rig_settings = json.load(file)
    rig_settings = rig_settings["value"]
    return ses_settings, rig_settings

def load_data(base_path):
    '''Load raw behaviour data logged by Bonsai'''
    events_reader = Csv("behav/experiment-events/experiment-events_*", ["Seconds", "Value"])
    events_data = aeon.load(Path(base_path), events_reader)

    lick_reader = Csv("behav/licks/licks_*", ["Seconds", "Value"])
    lick_data = aeon.load(Path(base_path), lick_reader)

    rewards_reader = Csv("behav/reward/reward_*", ["Seconds", "Value"])
    rewards_data = aeon.load(Path(base_path), rewards_reader)

    position_reader = Csv("behav/current-position/current-position_*", ["Seconds","Value.X","Value.Y","Value.Z","Value.Length", "Value.LengthFast", "Value.LengthSquared"])
    position_data = aeon.load(Path(base_path), position_reader)

    treadmill_reader = Csv("behav/treadmill-speed/treadmill-speed_*", ["Seconds", "Value"])
    treadmill_data = aeon.load(Path(base_path), treadmill_reader)

    buffer_reader = Csv("behav/analog-data/analog-data_*", ["Seconds", "Value"])
    buffer_data = aeon.load(Path(base_path), buffer_reader)

    # Fix buffer resets
    buffer_vals = buffer_data['Value'].values.copy()

    offset = 0
    corrected = np.zeros_like(buffer_vals)

    for i in range(len(buffer_vals)):
        if i > 0 and buffer_vals[i] < buffer_vals[i-1]:
            offset += buffer_vals[i-1]

        corrected[i] = buffer_vals[i] + offset

    buffer_data['Value'] = corrected

    if os.path.exists(Path(base_path) / "behav/current-landmark/"):
        if 'cohort1' in str(base_path):
            lm_reader = Csv("behav/current-landmark/*", ["Seconds","Count","Size","Texture","Odour","SequencePosition","Position","Visited","RewardDelivered"])
        elif 'cohort2' in str(base_path):
            lm_reader = Csv("behav/current-landmark/*", ["Seconds","Count","Size","Texture","Odour","SequencePosition","Position","Visited","RewardDelivered","Gap","IgnoreInBoundaryCalculation"])
        lm_data = aeon.load(Path(base_path), lm_reader)
        # If RewardDelivered doesn't exist, it won't be in the dataframe
        if "RewardDelivered" not in lm_data.columns:
            lm_data["RewardDelivered"] = np.nan
        lm_data = lm_data[lm_data['Visited'] == False]
        sess_lm_data = lm_data.drop_duplicates(subset=['Position'], keep='first')

    sess_events_data = events_data[~events_data.index.duplicated(keep='first')]
    sess_lick_data = lick_data[~lick_data.index.duplicated(keep='first')]
    sess_treadmill_data = treadmill_data[~treadmill_data.index.duplicated(keep='first')]
    sess_position_data = position_data[~position_data.index.duplicated(keep='first')]
    sess_reward_data = rewards_data[~rewards_data.index.duplicated(keep='first')]
    sess_reward_data = sess_reward_data[sess_reward_data['Value'] != 'ManualReward']  # exclude experimenter-triggered rewards
    sess_buffer_data = buffer_data[~buffer_data.index.duplicated(keep='first')]

    if os.path.exists(Path(base_path) / "behav/current-landmark/"):
        sess_data = {
            'Events': pd.Series(sess_events_data['Value'], index=sess_events_data.index),
            'Licks': pd.Series(sess_lick_data['Value'], index=sess_lick_data.index),
            'Treadmill': pd.Series(sess_treadmill_data['Value'], index=sess_treadmill_data.index),
            'Position': pd.Series(sess_position_data['Value.Length'], index=sess_position_data.index),
            'Rewards': pd.Series(sess_reward_data['Value'], index=sess_reward_data.index),
            'Buffer': pd.Series(sess_buffer_data['Value'], index=sess_buffer_data.index),
            'LM_Count': pd.Series(sess_lm_data['Count'], index=sess_lm_data.index),
            'LM_Texture': pd.Series(sess_lm_data['Texture'], index=sess_lm_data.index),
            'LM_Odour': pd.Series(sess_lm_data['Odour'], index=sess_lm_data.index),
            'LM_Position': pd.Series(sess_lm_data['Position'], index=sess_lm_data.index),
            'Sequence_Position': pd.Series(sess_lm_data['SequencePosition'], index=sess_lm_data.index)
        }
        
        all_ix = sess_events_data.index.union(sess_lick_data.index).union(sess_treadmill_data.index).union(sess_position_data.index).union(sess_reward_data.index).union(sess_buffer_data.index).union(sess_lm_data.index)
        #take only unique indices
        all_ix = all_ix.unique()

    else:
        sess_data = {
            'Events': pd.Series(sess_events_data['Value'], index=sess_events_data.index),
            'Licks': pd.Series(sess_lick_data['Value'], index=sess_lick_data.index),
            'Treadmill': pd.Series(sess_treadmill_data['Value'], index=sess_treadmill_data.index),
            'Position': pd.Series(sess_position_data['Value.Length'], index=sess_position_data.index),
            'Rewards': pd.Series(sess_reward_data['Value'], index=sess_reward_data.index),
            'Buffer': pd.Series(sess_buffer_data['Value'], index=sess_buffer_data.index)
        }
        #combine indices
        all_ix = sess_events_data.index.union(sess_lick_data.index).union(sess_treadmill_data.index).union(sess_position_data.index).union(sess_reward_data.index).union(sess_buffer_data.index)
        #take only unique indices
        all_ix = all_ix.unique()

    sess_dataframe = pd.DataFrame(sess_data, index=all_ix)
    sess_dataframe['Position'] = sess_dataframe['Position'].interpolate()
    sess_dataframe['Treadmill'] = sess_dataframe['Treadmill'].interpolate()
    with pd.option_context("future.no_silent_downcasting", True):
        sess_dataframe['Licks'] = sess_dataframe['Licks'].fillna(False).astype(bool) 

    #crop sess_dataframe to when Buffer starts being non-zero
    sess_dataframe = sess_dataframe[sess_dataframe['Buffer'] >= 0]

    return sess_dataframe

def load_analog_data(base_path, ses_rig_settings):
    '''Load analog data'''
    channel_names = []
    for c in ses_rig_settings['analogInputChannels']:
        channel_names.append(c['alias'])
    print(f"Analog channels found: {channel_names}")
    analog_reader = AnalogData("behav/analog-data/*", channel_names, len(channel_names))
    analog_data = aeon.load(Path(base_path), analog_reader)
    analog_data = analog_data.reset_index() # aeon load assumes our indices are valid harp timestamps which they are not in this case
    analog_data = analog_data.drop(columns='time')

    return analog_data

def align_analog_to_events(analog_data, sess_dataframe, plot=False):
    '''Align analog data to Bonsai buffers'''
    buffer_data = sess_dataframe[['Buffer']].dropna().drop_duplicates()
    # buffer_data = sess_dataframe['Buffer']

    buffer_size = int(analog_data.shape[0] / buffer_data.shape[0]) # how many samples per buffer did we record?
    print(f'{len(analog_data)} analog samples were recorded.')
    print(f'{len(buffer_data)} buffer samples were recorded.')
    print(f'Buffer size: {buffer_size} samples')

    buffer_seconds = (buffer_data.index - datetime.datetime(1904, 1, 1)).total_seconds()
    # sliced_index = np.array(analog_data.index)[(buffer_size-1)::buffer_size]
    sliced_index = np.array(analog_data.index)[::buffer_size]

    o_m, o_b = np.polyfit(sliced_index, buffer_seconds, 1)
    index_to_timestamp = lambda x: x*o_m + o_b

    remapped_analog_index = aeon.aeon(index_to_timestamp(analog_data["rewards"].index))
    remapped_analog_data = analog_data
    remapped_analog_data = remapped_analog_data.set_index(remapped_analog_index)

    print('Converted analog index from sample to timestamps.')

    if plot: 
        plot_window = 100
        plt.figure(figsize=(4,4))
        plt.scatter(sliced_index[0:plot_window], buffer_seconds[0:plot_window], c='k', s=2, label='buffer')
        plt.plot(sliced_index[0:plot_window], index_to_timestamp(sliced_index)[0:plot_window], c='r', label='interp analog')
        plt.legend()

    return remapped_analog_data

def load_session_npz(base_path):
    '''Load behaviour data for valid funcimg frames'''
    data_path = base_path + '/behaviour_data.npz'
    data = np.load(data_path)

    if 'pd2' in data.files:
        return data

    if 'p2' not in data.files:
        print(f'No p2 key in {data_path}')
        return data
    
    fixed = dict(data)
    fixed['pd2'] = fixed.pop('p2')

    np.savez_compressed(data_path, **fixed)

    print(f'fixed pd2 naming in {data_path}') 

    return fixed 

#%% ##### Session functions #####
def get_event_parsed(sess_dataframe, ses_settings, threshold='below'):

    if threshold == 'below':
        licks = threshold_lick_events(sess_dataframe, ses_settings, below=True)
    elif threshold == 'above':
        licks = threshold_lick_events(sess_dataframe, ses_settings, below=False)
    elif threshold == 'all':
        licks = sess_dataframe['Licks'].values
    lick_position = sess_dataframe['Position'].values[licks > 0]
    lick_times = sess_dataframe.index[licks > 0]
    reward_times = sess_dataframe.index[sess_dataframe['Rewards'].notna()]
    reward_positions = sess_dataframe['Position'].values[sess_dataframe['Rewards'].notna()]

    if 'LM_Count' in sess_dataframe.columns:
        release_df = estimate_lm_events(sess_dataframe)
    else:
        release_df = estimate_release_events(sess_dataframe, ses_settings)

    # Fix the order of the first events
    lm_idx = np.asarray(release_df['Index'].to_numpy(), dtype=int)
    position = np.nan_to_num(sess_dataframe['Position'].values, nan=0.0)
    release_positions = position[lm_idx]
    
    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']
    reward_seq = np.array([lm[0]['rewardSequencePosition'] for lm in trial['landmarks']])

    if len(reward_seq) == 4:
        if np.diff(reward_seq)[0] == 0:    
            # AABB: re-order AB so that A is always first
            release_df = release_df[2:]
            
        elif len(np.where(reward_seq == -1)[0]) > 2:   
            # ABBB: get rid of first event if needed otherwise keep the order the same
            if release_positions[0] < lm_size:
                release_df = release_df[1:]
        else:    
            # ABAB: re-order AB so that A is always first
            release_df = release_df[1:]

    if len(reward_seq) == 3:
        # ABB: get rid of first event if needed otherwise keep the order the same
        if release_positions[0] < 2:
            release_df = release_df[1:]
    
    return lick_position, lick_times, reward_times, reward_positions, release_df

def parse_rew_lms(ses_settings):
    rew_odour = []
    rew_texture = []
    non_rew_odour = []
    non_rew_texture = []
    index = []

    for i in ses_settings['trial']['landmarks']:
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

def parse_stable_goal_ids(ses_settings):
    '''Identify the number of landmarks and goals for stable world sequences'''
    
    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']

    num_lms = len(trial['landmarks'])
    num_goals = ses_settings['availableRewardPositions']
    lm_ids = np.arange(num_lms)
    goal_counter = 0
    goals = []
    while goal_counter < num_goals:
        for i in range(num_lms):
            for j in trial['landmarks'][i]:
                if j['rewardSequencePosition'] == goal_counter:
                    goals.append(i)
                    goal_counter += 1
                    if goal_counter >= num_goals:
                        break
                    
    return goals, lm_ids

def parse_random_goal_ids(ses_settings):
    '''Identify the number of landmarks and goals for random world sequences'''
    rew_odour, _, non_rew_odour, _ = parse_rew_lms(ses_settings)

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']

    num_lms = len(rew_odour) + len(non_rew_odour)
    num_goals = ses_settings['availableRewardPositions']
    lm_ids = np.arange(num_lms)

    goal_counter = 0
    goals = []
    while goal_counter < num_goals:
        for i in range(num_lms):
            for j in trial['landmarks'][i]:
                if j['rewardSequencePosition'] == goal_counter:
                    goals.append(i)
                    goal_counter += 1
                    if goal_counter >= num_goals:
                        break

    return goals, lm_ids

def get_hit_fa_events_split(sess_dataframe, ses_settings):
    target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    A_landmarks, B_landmarks, A_idx, B_idx = get_A_B_landmarks(sess_dataframe, ses_settings)
    lick_position, *_ = get_event_parsed(sess_dataframe, ses_settings)

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']
    reward_seq = np.array([lm[0]['rewardSequencePosition'] for lm in trial['landmarks']])

    release_positions = np.sort(np.concatenate([target_positions, distractor_positions]))

    # --- split landmark groups exactly like your function ---
    A1 = A_landmarks
    A2 = []
    B2 = []
    B3 = []

    if len(reward_seq) == 3:
        if len(np.where(reward_seq == 0)[0]) == 1:
            B1 = B_landmarks[::2]
            B2 = B_landmarks[1::2]
        elif len(np.where(reward_seq == 0)[0]) == 2:
            A1 = A_landmarks[::2]
            A2 = A_landmarks[1::2]
            B1 = B_landmarks

    elif len(reward_seq) == 4:
        if len(np.where(reward_seq == -1)[0]) > 2:    # ABBB
            B1 = B_landmarks[::3]
            B2 = B_landmarks[1::3]
            B3 = B_landmarks[2::3]
        elif np.diff(reward_seq)[0] == 0:    # AABB
            A1 = A_landmarks[::2]
            A2 = A_landmarks[1::2]
            B1 = B_landmarks[::2]
            B2 = B_landmarks[1::2]
        else:    # ABAB
            A1 = A_landmarks
            B1 = B_landmarks

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
                int(np.any((lick_position > pos) & (lick_position < pos + lm_size)))
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

def calc_hit_fa(sess_dataframe, ses_settings):
    '''Calculate average hit and false alarm rate across a session'''

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']

    target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)

    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    release_positions = np.sort(np.concatenate([target_positions, distractor_positions]))

    licked_target = np.zeros(len(target_positions))
    for idx, pos in enumerate(target_positions):
        if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
            licked_target[idx] = 1

    licked_distractor = np.zeros(len(distractor_positions))
    for idx, pos in enumerate(distractor_positions):
        if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
            licked_distractor[idx] = 1

    licked_all = np.zeros(len(release_df), dtype=int)
    rewarded_all = np.zeros(len(release_df), dtype=int)
    for idx, pos in enumerate(release_positions):
        # only take into account licks/rewards that came later than the release
        licks = lick_position[lick_times >= release_df.index[idx]]
        rewards = reward_positions[reward_times >= release_df.index[idx]]
        # compare licks/rewards to position window (the LM position and logged position are offset by 3)
        if np.any((licks > (pos)) & (licks < (pos + lm_size))):
            licked_all[idx] = 1
        if np.any((rewards > (pos)) & (rewards < (pos + lm_size))):
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

def calc_sw_hit_fa(sess_dataframe, ses_settings, window=12, split_lms=False, plot=True):
    '''Calculate hit and false alarm rates as a sliding window across the session'''

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']
    reward_seq = np.array([lm[0]['rewardSequencePosition'] for lm in trial['landmarks']])

    target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    A_landmarks, B_landmarks, A_idx, B_idx = get_A_B_landmarks(sess_dataframe, ses_settings)
    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    release_positions = np.sort(np.concatenate([target_positions, distractor_positions]))
    
    if not split_lms:
        hit_rate_sw = np.zeros(len(release_positions[:-window]))
        fa_rate_sw = np.zeros(len(release_positions[:-window]))

        for idx, pos in enumerate(release_positions[:-window]):

            # Find landmark events within the specified window
            positions_range = release_positions[idx:idx + window]
            
            lick_pos_range = lick_position[(lick_position >= positions_range[0]) & (lick_position <= positions_range[-1] + lm_size)]
            target_pos_range = target_positions[(target_positions >= positions_range[0]) & (target_positions <= positions_range[-1])]
            distractor_pos_range = distractor_positions[(distractor_positions >= positions_range[0]) & (distractor_positions <= positions_range[-1])]
            
            # Find responses to targets and distractors inside the lms
            licked_target = np.zeros(len(target_pos_range))
            for a, a_pos in enumerate(target_pos_range):
                if np.any((lick_pos_range > a_pos) & (lick_pos_range < (a_pos + lm_size))):
                    licked_target[a] = 1
            
            licked_distractor = np.zeros(len(distractor_pos_range))
            for b, b_pos in enumerate(distractor_pos_range):
                if np.any((lick_pos_range > b_pos) & (lick_pos_range < (b_pos + lm_size))):
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
        A1 = A_landmarks
        A2 = []
        B2 = []
        B3 = []

        if len(reward_seq) == 3:
            if len(np.where(reward_seq == 0)[0]) == 1:
                B1 = B_landmarks[::2]
                B2 = B_landmarks[1::2]
            elif len(np.where(reward_seq == 0)[0]) == 2:
                A1 = A_landmarks[::2]
                A2 = A_landmarks[1::2]
                B1 = B_landmarks

        elif len(reward_seq) == 4:
            if len(np.where(reward_seq == -1)[0]) > 2:
                B1 = B_landmarks[::3]
                B2 = B_landmarks[1::3]
                B3 = B_landmarks[2::3]
            else:
                A1 = A_landmarks[::2]
                A2 = A_landmarks[1::2]
                B1 = B_landmarks[::2]
                B2 = B_landmarks[1::2]

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
            
            lick_pos_range = lick_position[(lick_position >= positions_range[0]) & (lick_position <= positions_range[-1] + lm_size)]
            
            A1_pos_range = A1_positions[(A1_positions >= positions_range[0]) & (A1_positions <= positions_range[-1])]
            A2_pos_range = A2_positions[(A2_positions >= positions_range[0]) & (A2_positions <= positions_range[-1])] if len(A2_positions) else np.array([])
            
            B1_pos_range = B1_positions[(B1_positions >= positions_range[0]) & (B1_positions <= positions_range[-1])]
            B2_pos_range = B2_positions[(B2_positions >= positions_range[0]) & (B2_positions <= positions_range[-1])]
            B3_pos_range = B3_positions[(B3_positions >= positions_range[0]) & (B3_positions <= positions_range[-1])] if len(B3_positions) else np.array([])

            # Find responses to targets and distractors inside the lms
            licked_A1 = np.zeros(len(A1_pos_range))
            for a, a_pos in enumerate(A1_pos_range):
                if np.any((lick_pos_range > a_pos) & (lick_pos_range < (a_pos + lm_size))):
                    licked_A1[a] = 1
            
            if len(A2_positions):
                licked_A2 = np.zeros(len(A2_pos_range))
                for a, a_pos in enumerate(A2_pos_range):
                    if np.any((lick_pos_range > a_pos) & (lick_pos_range < (a_pos + lm_size))):
                        licked_A2[a] = 1

            licked_B1 = np.zeros(len(B1_pos_range))
            for b, b_pos in enumerate(B1_pos_range):
                if np.any((lick_pos_range > b_pos) & (lick_pos_range < (b_pos + lm_size))):
                    licked_B1[b] = 1

            if len(B2_positions):
                licked_B2 = np.zeros(len(B2_pos_range))
                for b, b_pos in enumerate(B2_pos_range):
                    if np.any((lick_pos_range > b_pos) & (lick_pos_range < (b_pos + lm_size))):
                        licked_B2[b] = 1

            if len(B3_positions):
                licked_B3 = np.zeros(len(B3_pos_range))
                for a, a_pos in enumerate(B3_pos_range):
                    if np.any((lick_pos_range > a_pos) & (lick_pos_range < (a_pos + lm_size))):
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

def find_A_B_distance_and_positions(sess_dataframe, ses_settings, rewarded_As=False):
    '''Find the positions of As (either rewarded or not) and the consecutive Bs and the distance between them'''
    
    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    
    num_As, num_Bs = get_num_A_B(sess_dataframe, ses_settings)
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

def find_all_A_B_distance_and_positions(sess_dataframe, ses_settings):
    '''Find the positions of rewarded A1s and the consecutive As and Bs and the distance between them'''
    
    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']

    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    A_landmarks, B_landmarks, _, _ = get_A_B_landmarks(sess_dataframe, ses_settings)
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
    A1, A2, B1, B2, B3 = get_A_B_splits(A_landmarks, B_landmarks, ses_settings)

    # determine which A1s were rewarded
    rewarded_A1_positions = []
    for pos in release_positions[A1]:
        if np.any((reward_positions > pos) & (reward_positions <= pos + lm_size)):
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

def calc_distance_hit_fa(sess_dataframe, ses_settings, split_lms=False, plot=True):
    '''Calculate hit and fa rates for each distance'''

    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)
    A_landmarks, B_landmarks, A_idx, B_idx = get_A_B_landmarks(sess_dataframe, ses_settings)

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']
    reward_seq = np.array([lm[0]['rewardSequencePosition'] for lm in trial['landmarks']])

    distances = np.diff(release_df['Position']) - lm_size
    
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
                if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
                    licked_target[idx] = 1

            licked_distractor = np.zeros(len(distractor_pos_considered))
            for idx, pos in enumerate(distractor_pos_considered):
                if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
                    licked_distractor[idx] = 1

            hit_rate["A"][d] = (np.sum(licked_target) / len(licked_target)
                if len(licked_target) > 0 else np.nan)

            fa_rate["B"][d] = (np.sum(licked_distractor) / len(licked_distractor)
                if len(licked_distractor) > 0 else np.nan)
    
    else:
        A1 = A_landmarks
        A2 = []
        B3 = []

        if len(reward_seq) == 3:
            B1 = B_landmarks[::2]
            B2 = B_landmarks[1::2]

        elif len(reward_seq) == 4:
            if len(np.where(reward_seq == -1)[0]) > 2:
                B1 = B_landmarks[::3]
                B2 = B_landmarks[1::3]
                B3 = B_landmarks[2::3]
            else:
                A1 = A_landmarks[::2]
                A2 = A_landmarks[1::2]
                B1 = B_landmarks[::2]
                B2 = B_landmarks[1::2]

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
                if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
                    licked_A1[i] = 1

            licked_A2 = np.zeros(len(A2_pos_considered))
            for i, pos in enumerate(A2_pos_considered):
                if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
                    licked_A2[i] = 1
            
            licked_B1 = np.zeros(len(B1_pos_considered))
            for i, pos in enumerate(B1_pos_considered):
                if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
                    licked_B1[i] = 1
            
            licked_B2 = np.zeros(len(B2_pos_considered))
            for i, pos in enumerate(B2_pos_considered):
                if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
                    licked_B2[i] = 1

            licked_B3 = np.zeros(len(B3_pos_considered))
            for i, pos in enumerate(B3_pos_considered):
                if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
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

def calc_distance_from_A_hit_fa(sess_dataframe, ses_settings, plot=True, remove_disengagement=False, plot_disengagement=False):
    '''Calculate hit and fa rates based on distance from preceding A'''
    
    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)
    A_landmarks, B_landmarks, A_idx, B_idx = get_A_B_landmarks(sess_dataframe, ses_settings)

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']

    # Find number of landmarks between two consecutive As
    num_Bs = len(distractor_positions[(distractor_positions > target_positions[0]) & (distractor_positions < target_positions[1])])

    # Find distances between A and the following Bs
    A_A_diff, A_B_diff, A_positions, B_positions = find_A_B_distance_and_positions(sess_dataframe, ses_settings, rewarded_As=False)

    # Calculate hit rate
    licked_As = np.zeros((len(target_positions) - 1))
    for i, pos in enumerate(target_positions[:-1]):
        if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
            licked_As[i] = 1

    # Calculate false alarm rates
    licked_Bs = np.zeros((len(target_positions) - 1, num_Bs))
    for i, pos in enumerate(B_positions):
        for j in range(num_Bs):
            if np.any((lick_position > pos[j]) & (lick_position < (pos[j] + lm_size))):
                licked_Bs[i, j] = 1

    # Filter out disengaged trials 
    if remove_disengagement:
        _, _, _, _, [valid_mask_A, valid_mask_B], _ = calc_time_from_A_hit_fa(sess_dataframe, ses_settings, plot=False, remove_disengagement=True, plot_disengagement=plot_disengagement)
        
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

def calc_time_hit_fa(sess_dataframe, ses_settings, bins=10, plot=True):
    '''Calculate hit and fa rates based on time spent between landmarks'''

    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']

    release_positions = np.sort(np.concatenate([target_positions, distractor_positions]))

    # Bin time 
    dt, _ = get_time_between_landmarks(sess_dataframe, ses_settings, bins, plot=False)
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
            if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
                licked_target[idx] = 1

        licked_distractor = np.zeros(len(distractor_pos_considered))
        for idx, pos in enumerate(distractor_pos_considered):
            if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
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

def calc_distance_from_rew_hit_fa(sess_dataframe, ses_settings, plot=True, remove_disengagement=False, plot_disengagement=False):
    '''Calculate hit and fa rates based on distance from preceding **rewarded** A'''
    
    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)
    A_landmarks, B_landmarks, A_idx, B_idx = get_A_B_landmarks(sess_dataframe, ses_settings)

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']

    # Find number of landmarks between two consecutive As
    num_Bs = len(distractor_positions[(distractor_positions > target_positions[0]) & (distractor_positions < target_positions[1])])

    # Find distances between A and the following Bs
    A_A_diff, A_B_diff, A_positions, B_positions = find_A_B_distance_and_positions(sess_dataframe, ses_settings, rewarded_As=True)

    # Calculate hit rate
    licked_As = np.zeros((len(reward_positions) - 1))
    for i, pos in enumerate(A_positions):
        if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
            licked_As[i] = 1

    # Calculate false alarm rates
    licked_Bs = np.zeros((len(reward_positions) - 1, num_Bs))
    for i, pos in enumerate(B_positions):
        for j in range(num_Bs):
            if np.any((lick_position > pos[j]) & (lick_position < (pos[j] + lm_size))):
                licked_Bs[i, j] = 1
    
    # Filter out disengaged trials 
    if remove_disengagement:
        _, _, _, _, [valid_mask_A, valid_mask_B], _ = calc_time_from_rew_hit_fa(sess_dataframe, ses_settings, plot=False, remove_disengagement=remove_disengagement, plot_disengagement=plot_disengagement)
        
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

def calc_distance_from_A1rew_lick_rate(sess_dataframe, ses_settings, plot=True, remove_disengagement=False, plot_disengagement=False):
    '''Calculate hit and fa rates based on distance from preceding **rewarded** A1'''
    
    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    
    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']

    num_As, num_Bs = get_num_A_B(sess_dataframe, ses_settings)

    # Find distances between A and the following Bs
    A_A_diff, A_B_diff, A_positions, B_positions = find_all_A_B_distance_and_positions(sess_dataframe, ses_settings)

    num_rew_A1 = len(A_positions)
    
    # Calculate hit rate
    licked_As = np.zeros((num_rew_A1, num_As))
    for i, pos in enumerate(A_positions):
        for j in range(num_As):
            if np.any((lick_position > pos[j]) & (lick_position < (pos[j] + lm_size))):
                licked_As[i, j] = 1

    # Calculate false alarm rates
    licked_Bs = np.zeros((num_rew_A1, num_Bs))
    for i, pos in enumerate(B_positions):
        for j in range(num_Bs):
            if np.any((lick_position > pos[j]) & (lick_position < (pos[j] + lm_size))):
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
    
def calc_distance_from_rew_lick_frac(sess_dataframe, ses_settings, plot=True):
    '''Calculate fraction of trials per distance from **rewarded** A that were the mouse's first lick'''
    
    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']

    # Find number of landmarks between two consecutive As
    num_Bs = len(distractor_positions[(distractor_positions > target_positions[0]) & (distractor_positions < target_positions[1])])

    # Find distances between A and the following Bs
    A_A_diff, A_B_diff, A_positions, B_positions = find_A_B_distance_and_positions(sess_dataframe, ses_settings, rewarded_As=True)
    
    following_positions = np.array([
        np.sort(np.concatenate([np.atleast_1d(A_positions[i]), B_positions[i]]))
        for i in range(len(A_positions))
    ])

    # Find where first lick after a reward occured
    lm_licked = np.zeros_like(following_positions)
    for i, positions in enumerate(following_positions):
        for j, pos in enumerate(positions):
            if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
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

def get_time_between_landmarks(sess_dataframe, ses_settings, bins=20, plot=True):
    '''Calculate time spent between different landmark types (AA, BB or AB)'''

    if 'LM_Count' in sess_dataframe.columns:
        release_df = estimate_lm_events(sess_dataframe)
    else:
        release_df = estimate_release_events(sess_dataframe, ses_settings)

    _, _, A_idx, B_idx = get_A_B_landmarks(sess_dataframe, ses_settings)

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

def calc_time_from_A_hit_fa(sess_dataframe, ses_settings, plot=True, remove_disengagement=False, plot_disengagement=False):
    '''Calculate hit and fa rates for each time difference'''

    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    A_landmarks, B_landmarks, A_idx, B_idx = get_A_B_landmarks(sess_dataframe, ses_settings)

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']
    
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
        if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
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
            if np.any((lick_position > pos[j]) & (lick_position < (pos[j] + lm_size))):
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

def calc_time_from_rew_hit_fa(sess_dataframe, ses_settings, plot=True, remove_disengagement=False, plot_disengagement=False):
    '''Calculate hit and fa rates for each time difference from a **rewarded** A'''

    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    A_landmarks, B_landmarks, A_idx, B_idx = get_A_B_landmarks(sess_dataframe, ses_settings)

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']
    
    # Find number of landmarks between two consecutive As
    num_Bs = len(distractor_positions[(distractor_positions > target_positions[0]) & (distractor_positions < target_positions[1])])

    # Find distances between A and the following Bs
    A_A_diff, A_B_diff, A_positions, B_positions = find_A_B_distance_and_positions(sess_dataframe, ses_settings, rewarded_As=True)
    
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
        if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
            licked_As[i] = 1

    # Calculate false alarm rates
    licked_Bs = np.zeros((len(reward_positions) - 1, num_Bs))
    for i, pos in enumerate(B_positions):
        for j in range(num_Bs):
            if np.any((lick_position > pos[j]) & (lick_position < (pos[j] + lm_size))):
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

def calc_distance_from_rew_p_lick(sess_dataframe, ses_settings, plot=True, remove_disengagement=False):
    '''
    1. Probability of first lick at each landmark type per distance bin from **rewarded** A
       (probabilities sum to 1 within each distance bin)
    2. Number of first licks per landmark type per distance bin
    3. Number of trials per landmark type per distance bin
    4. Number of available trials (where no other lm was licked before) per landmark type per distance bin
    '''
    
    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    num_lms = len(lm_ids)

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']

    # Find number of landmarks between two consecutive As
    num_Bs = len(distractor_positions[(distractor_positions > target_positions[0]) & (distractor_positions < target_positions[1])])
    
    if num_Bs == 1:
        num_lms = 2 # abab

    # Find distances between A and the following Bs
    A_A_diff, A_B_diff, A_positions, B_positions = find_A_B_distance_and_positions(sess_dataframe, ses_settings, rewarded_As=True)
    
    if remove_disengagement:
        # We want to disregard trials where the mouse got a reward and disengaged until the next A, so all intermediate Bs are excluded based on valid As
        _, _, _, _, [valid_mask_A, valid_mask_B], _ = calc_time_from_rew_hit_fa(sess_dataframe, ses_settings, plot=False, remove_disengagement=remove_disengagement, plot_disengagement=False)
        
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
            if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
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

def get_lick_persistence(sess_dataframe, ses_settings, plot=False, include_A=False, remove_disengagement=False, plot_disengagement=False):
    '''Get lick persistence (lick counts) per landmark type'''

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']

    lick_position, lick_times, reward_times, reward_positions, _ = get_event_parsed(sess_dataframe, ses_settings)
    A_landmarks, B_landmarks, _, _ = get_A_B_landmarks(sess_dataframe, ses_settings)
    _, _, target_positions, distractor_positions, _, _ = find_targets_distractors(sess_dataframe, ses_settings)
    release_positions = np.sort(np.concatenate([target_positions, distractor_positions]))

    # Find total number of licks inside landmark - TODO redundant? 
    lick_counter = np.zeros(len(release_positions), dtype=int)

    for i, pos in enumerate(release_positions):
        # Licks within landmark boundaries
        mask = (
            (np.round(lick_position, 1) >= np.round(pos, 1)) &
            (np.round(lick_position, 1) < np.round(pos, 1) + lm_size)
        )

        # If this is an A landmark, and it is rewarded, only count licks before reward delivery
        if i in A_landmarks:
            rew_idx = np.where((reward_positions > pos) & (reward_positions <= pos + lm_size))[0]

            if len(rew_idx) > 0:
                reward_time = reward_times[rew_idx[0]]
                mask &= (lick_times <= reward_time)

        lick_counter[i] = np.sum(mask)

    # Counts licks for each type of landmark
    num_As, num_Bs = get_num_A_B(sess_dataframe, ses_settings)
    # num_Bs = len(distractor_positions[(distractor_positions > target_positions[0]) & (distractor_positions < target_positions[1])])

    # Find distances between A and the following Bs
    A_A_diff, A_B_diff, A_positions, B_positions = find_A_B_distance_and_positions(sess_dataframe, ses_settings, rewarded_As=True)

    # Licks in rewarded As
    rewarded_lms = []
    licked_As = np.zeros_like(A_positions)
    for i, pos in enumerate(A_positions): # NOTE the first A is not considered 
        mask = (
            (np.round(lick_position, 1) >= np.round(pos, 1)) &
            (np.round(lick_position, 1) <= np.round(pos, 1) + lm_size)
        )
        rew_idx = np.where((reward_positions > pos) & (reward_positions <= pos + lm_size))[0]
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
                (np.round(lick_position, 1) <= np.round(pos[j], 1) + lm_size)
            )
            licked_Bs[i, j] = np.sum(mask)

    if remove_disengagement:
        # We want to disregard trials where the mouse got a reward and disengaged until the next A, so all intermediate Bs are excluded based on valid As
        _, _, _, _, [valid_mask_A, valid_mask_B], _ = calc_time_from_rew_hit_fa(sess_dataframe, ses_settings, plot=False, remove_disengagement=remove_disengagement, plot_disengagement=False)
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

def get_lick_counts(sess_dataframe, ses_settings, plot=False, threshold=True, misses=False, omissions=False):
    '''Get lick persistence (lick counts) per landmark type'''

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']

    lick_position, lick_times, reward_times, reward_positions, _ = get_event_parsed(sess_dataframe, ses_settings, threshold=threshold)
    A_landmarks, B_landmarks, _, _ = get_A_B_landmarks(sess_dataframe, ses_settings)
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, _ = find_targets_distractors(sess_dataframe, ses_settings)
    release_positions = np.sort(np.concatenate([target_positions, distractor_positions]))

    num_As, num_Bs = get_num_A_B(sess_dataframe, ses_settings)

    # Get misses and omissions
    if len(trial) > 1:
        omitted_lms, omitted_pos = get_omissions(sess_dataframe, ses_settings)
    missed_lms, missed_pos = get_misses(sess_dataframe, ses_settings)

    # Find total number of licks inside each landmark 
    lick_counter = np.zeros(len(release_positions))

    for i, pos in enumerate(release_positions):
        # Licks within landmark boundaries
        mask = (
            (np.round(lick_position, 1) >= np.round(pos, 1)) &
            (np.round(lick_position, 1) < np.round(pos, 1) + lm_size)
        )

        # Count licks if this is an A landmark and it was rewarded up to the reward timepoint
        if i in A_landmarks:
            if misses and i not in missed_lms:
                continue
            rew_idx = np.where((reward_positions > pos) & (reward_positions <= pos + lm_size))[0]
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
    A1, A2, B1, B2, B3 = get_A_B_splits(A_landmarks, B_landmarks, ses_settings)

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
    A_A_diff, A_B_diff, A_positions, B_positions = find_all_A_B_distance_and_positions(sess_dataframe, ses_settings)
    
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
                (np.round(lick_position, 1) <= np.round(curr_pos, 1) + lm_size)
            )
            rew_idx = np.where((reward_positions > curr_pos) & (reward_positions <= curr_pos + lm_size))[0]
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
                (np.round(lick_position, 1) <= np.round(curr_pos, 1) + lm_size)
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
           
def get_response_end(target_licks):
    '''Find the index (if any) where the mouse stopped licking in the task'''

    # Find misses
    zero_idx = np.where(target_licks == 0)[0]
    if len(zero_idx) == 0:
        return None

    # Find breaks between consecutive indices and split into blocks
    breaks = np.where(np.diff(zero_idx) != 1)[0] + 1
    blocks = np.split(zero_idx, breaks)

    # Keep only blocks with length >= 5
    long_blocks = [b for b in blocks if len(b) >= 5]
    if len(long_blocks) > 0:
        cutoff_event = long_blocks[-1][0]
        # Check if any lick happens after this block
        if not np.any(target_licks[long_blocks[-1][-1] + 1:] == 1):
            print(f'Removing from trial {cutoff_event} until the end ({len(target_licks)}), because of likely satiety (no licking)')
            return cutoff_event
    
    return None

def get_disengagement_periods(A_A_dt, A_B_dt, plot=True):
    '''Find the trials (if any) where the mouse stopped engaging (no running) in the task'''

    # Find trial indices where reaching the next landmark took too long
    thresholds = []

    # AA
    thresholds.append(np.median(A_A_dt) + 0.5 * np.std(A_A_dt))
    ix_AA = np.where(A_A_dt > thresholds[0])[0]
    # print('A-A indices:', ix_AA)
    # print('A-A values', A_A_dt[ix_AA])

    # AB
    ix_AB = []
    for i in range(A_B_dt.shape[1]):
        col = A_B_dt[:, i]
        thresholds.append(np.median(col) + 0.5 * np.std(col))
        ix_AB.append(np.where(col > thresholds[i+1])[0])        
        # print(f'A-B{i+1} indices:', ix_AB[i])
        # print(f'A-B{i+1} values:', col[ix_AB[i]])

    # Find common indices
    # shared_ix = ix_AA.copy()
    # for ix in ix_AB:
    #     shared_ix = np.intersect1d(shared_ix, ix)
    # print("Shared disengagement indices:", shared_ix)
      
    if plot:
        _, ax = plt.subplots(1, A_B_dt.shape[1] + 1, figsize=(10,2))
        ax = ax.ravel()

        for i in range(A_B_dt.shape[1]):
            ax[i].hist(A_B_dt[:,i], bins=20)
            ax[i].set_title(f'A -> B{i+1}')
            ax[i].axvline(thresholds[i+1], linestyle='--', color='grey')
        for a in ax:
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)

        ax[i+1].hist(A_A_dt, bins=20)
        ax[i+1].set_title('A -> A')
        ax[i+1].axvline(thresholds[0], linestyle='--', color='grey')
        
    return ix_AA, ix_AB
      
def compute_binned_lick_rate(distances, licks, bins):
    
    bin_idx = np.digitize(distances, bins)
    
    means = []
    sems = []
    centers = []

    for b in range(1, len(bins)):
        mask = bin_idx == b
        
        if np.sum(mask) > 0:
            vals = licks[mask].astype(float)
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
        else:
            means.append(np.nan)
            sems.append(np.nan)
        
        centers.append(np.round((bins[b] + bins[b-1]) / 2, 2))

    return np.array(centers), np.array(means), np.array(sems)

def extract_int(s: str) -> int:
    m = re.search(r'\d+', s)
    if m:
        return int(m.group())
    else:
        raise ValueError(f"No digits found in string: {s!r}")

def get_landmarks(sess_dataframe, ses_settings):
    # Get landmark visits (full corridor)
    _, _, _, _, release_df = get_event_parsed(sess_dataframe, ses_settings)
    lm_idx = np.asarray(release_df['Index'].to_numpy(), dtype=int) # TODO rename because it conflicts with another definition
    landmarks = np.arange(len(lm_idx))
    
    return landmarks, lm_idx

def get_A_B_landmarks(sess_dataframe, ses_settings):
    '''Find which landmarks are rewarded (A) or non-rewarded (B)'''
    from itertools import zip_longest

    # Get landmark visits
    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    lm_idx = np.asarray(release_df['Index'].to_numpy(), dtype=int) # TODO rename because it conflicts with another definition

    # Get the sequence of landmarks
    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    reward_seq = np.array([lm[0]['rewardSequencePosition'] for lm in trial['landmarks']])

    if len(reward_seq) > 4: 
        reward_seq = reward_seq[:4]
    
    sorting = True

    # Split As and Bs into subtypes
    if len(reward_seq) == 4:
        if np.diff(reward_seq)[0] == 0:    # AABB
            A_landmarks = [i - 2 for i, r in enumerate(reward_seq) if r == 0]
            B_landmarks = [i + 2 for i, r in enumerate(reward_seq) if r == -1]
        elif len(np.where(reward_seq == -1)[0]) > 2:    # ABBB
            A_landmarks = list(np.where(reward_seq == 0)[0])
            if A_landmarks[0] == 0:
                A_landmarks[0] = 3 
            B_landmarks = [i for i in range(len(reward_seq)) if (i not in A_landmarks)]
        else:    # ABAB
            A_landmarks = [i - 1 for i, r in enumerate(reward_seq) if r == 0]
            B_landmarks = [i + 1 for i, r in enumerate(reward_seq) if r == -1]
    elif len(reward_seq) == 3:
        A_landmarks = list(np.where(reward_seq == 0)[0])
        if len(A_landmarks) == 1:     # ABB
            if A_landmarks[0] == 0:
                A_landmarks[0] = 2
        elif len(A_landmarks) == 2:   # AAB
            sorting = False
            if A_landmarks[0] == 0:
                seq1 = list(range(2, len(lm_idx), len(reward_seq)))
                seq2 = list(range(0, len(lm_idx), len(reward_seq)))
                
                A_landmarks = []
                for a, b in zip_longest(seq1, seq2):
                    if a is not None:
                        A_landmarks.append(a)
                    if b is not None:
                        A_landmarks.append(b)
                A_landmarks = np.sort(A_landmarks)

        B_landmarks = [i for i in range(len(reward_seq)) if (i not in A_landmarks)]

    if sorting:     
        for a in range(len(np.where(reward_seq == 0)[0])):
            A_landmarks.extend([i for i in range(A_landmarks[a]+len(reward_seq), len(lm_idx), len(reward_seq)) if i < len(lm_idx)])
    for b in range(len(np.where(reward_seq == -1)[0])):
        B_landmarks.extend([i for i in range(B_landmarks[b]+len(reward_seq), len(lm_idx), len(reward_seq)) if i < len(lm_idx)])
    
    if sorting:
        A_landmarks = np.sort(A_landmarks)
    B_landmarks = np.sort(B_landmarks)

    # Split the data indices into subtypes
    A_idx = [lm_idx[i] for i in A_landmarks]
    B_idx = [lm_idx[i] for i in B_landmarks]

    assert len(lm_idx) == (len(A_landmarks) + len(B_landmarks)), 'Some landmarks are missing!'

    return A_landmarks, B_landmarks, A_idx, B_idx

def get_A_B_splits(A_landmarks, B_landmarks, ses_settings):
    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']

    # TODO adapt for omissions
    reward_seq = np.array([lm[0]['rewardSequencePosition'] for lm in trial['landmarks']])

    A1 = A_landmarks
    A2 = []
    B2 = []
    B3 = []

    if len(reward_seq) == 3:
        if len(np.where(reward_seq == 0)[0]) == 1:
            B1 = B_landmarks[::2]
            B2 = B_landmarks[1::2]
        elif len(np.where(reward_seq == 0)[0]) == 2:
            # note for AAB the first lm is A2
            A1 = A_landmarks[1::2]
            A2 = A_landmarks[::2]
            B1 = B_landmarks

    elif len(reward_seq) == 4:
        if len(np.where(reward_seq == -1)[0]) > 2:    # ABBB
            B1 = B_landmarks[::3]
            B2 = B_landmarks[1::3]
            B3 = B_landmarks[2::3]
        elif np.diff(reward_seq)[0] == 0:    # AABB
            A1 = A_landmarks[::2]
            A2 = A_landmarks[1::2]
            B1 = B_landmarks[::2]
            B2 = B_landmarks[1::2]
        else:    # ABAB
            A1 = A_landmarks
            B1 = B_landmarks
    
    return A1, A2, B1, B2, B3

def get_num_A_B(sess_dataframe, ses_settings):
    '''Count the number of As and Bs in the binary pattern'''
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, _ = find_targets_distractors(sess_dataframe, ses_settings)

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

def get_omissions(sess_dataframe, ses_settings):
    '''Find which As were omitted'''
    # TODO incorporate into estimation of target positions if there are multiple types of trials

    A_landmarks, B_landmarks, A_idx, B_idx = get_A_B_landmarks(sess_dataframe, ses_settings)

    all_events = sess_dataframe['Sequence_Position'].dropna().to_numpy().astype(int)
    all_positions = sess_dataframe['Position'].values[sess_dataframe['Sequence_Position'].notna()]

    target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    seq_start = np.where(target_positions[0] == all_positions)[0][0]
    all_events = all_events[seq_start:]
    all_positions = all_positions[seq_start:]

    A_binary = np.zeros(len(all_events), dtype=bool)
    A_binary[A_landmarks] = 1

    omissions = np.where(A_binary & (all_events == -1))[0]
    omission_positions = all_positions[omissions]

    return omissions, omission_positions

def get_misses(sess_dataframe, ses_settings):
    '''Find which As were missed'''

    A_landmarks, B_landmarks, A_idx, B_idx = get_A_B_landmarks(sess_dataframe, ses_settings)
    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']

    all_events = sess_dataframe['Sequence_Position'].dropna().to_numpy().astype(int)
    all_positions = sess_dataframe['Position'].values[sess_dataframe['Sequence_Position'].notna()]

    target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)

    seq_start = np.where(target_positions[0] == all_positions)[0][0]
    all_events = all_events[seq_start:]
    all_positions = all_positions[seq_start:]

    misses = []
    miss_positions = []
    for A in A_landmarks:
        curr_pos = all_positions[A]
        rew_idx = np.where((reward_positions > curr_pos) & (reward_positions <= curr_pos + lm_size))[0]

        if len(rew_idx) > 0:
            continue
        misses.append(A)
        miss_positions.append(curr_pos)  
        
    return misses, miss_positions

def find_targets_distractors(sess_dataframe, ses_settings):
    '''Give an id to each type of landmark'''

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    reward_seq = np.array([lm[0]['rewardSequencePosition'] for lm in trial['landmarks']])

    if len(reward_seq) > 4: 
        reward_seq = reward_seq[:4]
      
    # Give ids to each type of landmark 
    # distractor_id = np.arange(0, len(np.where(reward_seq == -1)[0])) #[0,1]
    # target_id = np.arange(distractor_id[-1] + 1, len(np.where(reward_seq != -1)[0]) + distractor_id[-1] + 1)

    # Define order of landmark ids
    lm_id = np.arange(len(reward_seq))
    target_idx = np.where(reward_seq == 0)[0] 
    distractor_idx = np.where(reward_seq == -1)[0]
    
    if len(reward_seq) == 4:
        if np.diff(reward_seq)[0] == 0: # AABB
            if reward_seq[0] == -1:
                distractor_id = lm_id[distractor_idx] + 2
                target_id = lm_id[target_idx] - 2
            else:
                distractor_id = lm_id[distractor_idx]
                target_id = lm_id[target_idx]
        elif len(np.where(reward_seq == -1)[0]) > 2: # ABBB
            distractor_id = np.atleast_1d(lm_id[1:])
            target_id = np.atleast_1d(lm_id[0])
        else: # ABAB
            if reward_seq[0] == -1:
                distractor_id = lm_id[distractor_idx] + 1
                target_id = lm_id[target_idx] - 1
            else:
                distractor_id = lm_id[distractor_idx]
                target_id = lm_id[target_idx]

    elif len(reward_seq) == 3:
        if len(target_idx) == 1:
            distractor_id = np.atleast_1d(lm_id[len(target_idx):])
            target_id = np.atleast_1d(lm_id[0:len(target_idx)])
        elif len(target_idx) == 2:
            distractor_id = np.atleast_1d(lm_id[-1] - 1)
            target_id = [lm_id[target_idx[0]], lm_id[target_idx[1]] + 1]
    
    A_landmarks, B_landmarks, A_idx, B_idx = get_A_B_landmarks(sess_dataframe, ses_settings)

    # Get sequence of landmark ids 
    lm_id_sequence = np.zeros(len(A_landmarks) + len(B_landmarks), dtype=int)
    if len(reward_seq) == 4:
        if len(np.where(reward_seq == -1)[0]) > 2: # ABBB
            lm_id_sequence[A_landmarks] = np.tile(target_id, len(A_landmarks))
            lm_id_sequence[B_landmarks] = np.tile(distractor_id, int(np.ceil(len(B_landmarks)/2)))[:len(B_landmarks)]
        else:
            lm_id_sequence[A_landmarks] = np.tile(target_id, int(np.ceil(len(A_landmarks)/2)))[:len(A_landmarks)]
            lm_id_sequence[B_landmarks] = np.tile(distractor_id, int(np.ceil(len(B_landmarks)/2)))[:len(B_landmarks)]
    elif len(reward_seq) == 3:
        if len(target_id) == 1:
            lm_id_sequence[A_landmarks] = np.tile(target_id, len(A_landmarks))
            lm_id_sequence[B_landmarks] = np.tile(distractor_id, int(np.ceil(len(B_landmarks)/2)))[:len(B_landmarks)]
        elif len(target_id) == 2:
            lm_id_sequence[A_landmarks] = np.tile(target_id, int(np.ceil(len(A_landmarks)/2)))[:len(A_landmarks)]
            lm_id_sequence[B_landmarks] = np.tile(distractor_id, len(B_landmarks))

    # Get landmark visits
    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    lm_idx = np.asarray(release_df['Index'].to_numpy(), dtype=int)
    
    # Get positions of targets and distractors
    position = np.nan_to_num(sess_dataframe['Position'].values, nan=0.0)

    release_positions = position[lm_idx]
    # release_positions = release_df['Position'].to_numpy()     # less accurate

    target_positions = release_positions[A_landmarks]
    distractor_positions = release_positions[B_landmarks]

    return target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence

def calc_transition_matrix(sess_dataframe, ses_settings):
    
    target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)
    ideal_licks = get_ideal_performance(sess_dataframe, ses_settings)

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']

    num_landmarks = len(trial['landmarks'])

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

def calc_distance_transition_matrix(sess_dataframe, ses_settings, binning=True, distance_groups=None):
    '''
    Create a lick transition matrix based on distance between current and next licked landmark.
    If binning, distances will be grouped into small, medium, large.
    '''

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    num_landmarks = len(trial['landmarks'])

    A_landmarks, B_landmarks, A_idx, B_idx = get_A_B_landmarks(sess_dataframe, ses_settings)
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)
    ideal_licks = get_ideal_performance(sess_dataframe, ses_settings)

    # =========================
    # --- DISTANCE GROUPING ---
    # =========================
    A_A_diff, A_B_diff_list, licked_As, licked_Bs_list, _ = calc_distance_from_A_hit_fa(sess_dataframe, ses_settings, plot=False)

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

def get_ideal_performance(sess_dataframe,ses_settings):

    target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = find_targets_distractors(sess_dataframe,ses_settings)
    
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

def calc_conditional_matrix(sess_dataframe, ses_settings, n_steps=1):
    '''Calculate the transition probabilities given reward n_steps ahead of the reward'''

    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)

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

def get_speed_psth(ses_settings, sess_dataframe, events=None, bins=300):
    '''Get speed around landmark entry'''

    # Get session data
    session = create_session_struct(sess_dataframe, ses_settings)
    position = session['position']
    # licks = threshold_lick_events(sess_dataframe, ses_settings).astype(int)
    licks_idx = np.where(session['licks'] > 0)[0]

    if 'LM_Count' in sess_dataframe.columns:
        release_df = estimate_lm_events(sess_dataframe)
    else:
        release_df = estimate_release_events(sess_dataframe, ses_settings)

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
        binned_speed[i] = session['speed'][event_idx]
        # bin_edges = np.linspace(start_idx, end_idx, bins + 1).astype(int)
        # binned_speed[i], _, _ = stats.binned_statistic(event_idx, session['speed'][event_idx], statistic='mean', bins=bin_edges)

        # event_pos = position[np.where(release_df['Index'] == lm_idx)[0][0]]
        # lm_exit_idx = np.argmin(np.abs(position - (event_pos + session['lm_size'])))

    mean_binned_speed = np.mean(binned_speed, axis=0)
    sem_binned_speed = stats.sem(binned_speed, axis=0)

    return mean_binned_speed, sem_binned_speed

# def plot_speed_lick_rate_psth(ses_settings, sess_dataframe, session_id, bins=None):

#     if 'LM_Count' in sess_dataframe.columns:
#         release_df = estimate_lm_events(sess_dataframe)
#     else:
#         release_df = estimate_release_events(sess_dataframe, ses_settings)

#     dt_idx = np.diff(release_df['Index'])
#     dt_seconds = release_df.index.to_series().diff().dt.total_seconds().to_numpy()
    
#     if bins is None:
#         min_dt_idx = np.min(dt_idx)
#         min_dt_seconds = np.nanmin(dt_seconds)
#         window_seconds = np.round(min_dt_seconds * 2, 1)
#         bins = int(min_dt_idx * 2)
#     else:
#         window_seconds = np.round(dt_seconds[1:] / dt_idx * bins, 1)
#         window_seconds = window_seconds[~np.isnan(window_seconds)][0]

#     fig, axes = plt.subplots(1, 2, figsize=(10,4))
#     ax_speed, ax_lick = axes

#     # --- Get event indices ---
#     A_landmarks, B_landmarks, A_idx, B_idx = get_A_B_landmarks(sess_dataframe, ses_settings)

#     # --- Define groups dynamically ---
#     groups = {}

#     if 'abab' in session_id:
#         groups = {
#             'A': (A_idx, 'darkblue'),
#             'B': (B_idx, 'orange')
#         }

#     elif 'aabb' in session_id or 'a2b2' in session_id:
#         groups = {
#             'A1': (A_idx[::2], 'darkblue'),
#             'A2': (A_idx[1::2], 'blue'),
#             'B1': (B_idx[::2], 'orange'),
#             'B2': (B_idx[1::2], 'gold')
#         }

#     elif 'abb' in session_id and 'abbb' not in session_id:
#         groups = {
#             'A': (A_idx, 'darkblue'),
#             'B1': (B_idx[::2], 'orange'),
#             'B2': (B_idx[1::2], 'gold')
#         }

#     elif 'abbb' in session_id:
#         groups = {
#             'A': (A_idx, 'darkblue'),
#             'B1': (B_idx[::3], 'orange'),
#             'B2': (B_idx[1::3], 'gold'),
#             'B3': (B_idx[2::3], 'brown')
#         }

#     elif 'aab' in session_id and 'aabb' not in session_id:
#         groups = {
#             'A1': (A_idx[::2], 'darkblue'),
#             'A2': (A_idx[1::2], 'blue'),
#             'B1': (B_idx, 'orange'),
#         }

#     # --- Compute + plot ---
#     for label, (events, color) in groups.items():

#         (mean_s, sem_s), (mean_l, sem_l) = compute_psth_pair(
#             ses_settings, sess_dataframe, events, bins
#         )

#         plot_psth(ax_speed, mean_s, sem_s, color, label)
#         plot_psth(ax_lick, mean_l, sem_l, color, label)

#     ax_speed.axhline(ses_settings['velocityThreshold'], linestyle='--', color='grey')

#     # --- Styling ---
#     for ax in axes:
#         ax.legend()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#         ax.axvspan(bins/2, bins, color='grey', alpha=0.3)
#         ax.set_xticks([0, bins/2, bins], labels=[f'{-window_seconds/2:.1f}', 0, f'{window_seconds/2:.1f}'])

#     ax_speed.set_title('Speed')
#     ax_lick.set_title('Lick rate')

#     plt.tight_layout()
#     return fig

def get_lick_rate_psth(ses_settings, sess_dataframe, events=None, bins=300):
    '''Get lick rate around landmark entry'''

    # Get session data
    session = create_session_struct(sess_dataframe, ses_settings)

    # Threshold licks 
    licks = threshold_lick_events(sess_dataframe, ses_settings)

    if 'LM_Count' in sess_dataframe.columns:
        release_df = estimate_lm_events(sess_dataframe)
    else:
        release_df = estimate_release_events(sess_dataframe, ses_settings)

    if events is None:
        events = release_df['Index']
     
    # Bin licks
    binned_licks = np.zeros((len(events), bins))

    for i, lm_idx in enumerate(events):    
        start_idx = lm_idx - bins / 2
        end_idx = lm_idx + bins / 2
        if start_idx < 0:
            continue
        if end_idx > len(session['position']):
            break
        
        event_idx = np.arange(start_idx, end_idx).astype(int)
        bin_edges = np.linspace(start_idx, end_idx, bins + 1).astype(int)
        
        binned_licks[i], _, _ = stats.binned_statistic(event_idx, licks[event_idx], statistic='mean', bins=bin_edges)

    mean_binned_licks = np.mean(binned_licks, axis=0)
    sem_binned_licks = stats.sem(binned_licks, axis=0)

    return mean_binned_licks, sem_binned_licks

def get_speed_psth_by_distance(sess_dataframe, ses_settings, bins=300, binning=True, distance_groups=None, rewarded_As=False):
    '''Compute speed PSTH split by landmark type and distance groups'''

    # --- session data ---
    session = create_session_struct(sess_dataframe, ses_settings)
    speed = session['speed']

    if 'LM_Count' in sess_dataframe.columns:
        release_df = estimate_lm_events(sess_dataframe)
    else:
        release_df = estimate_release_events(sess_dataframe, ses_settings)

    # --- landmark structure ---
    target_id, distractor_id, target_positions, distractor_positions, _, _ = \
        find_targets_distractors(sess_dataframe, ses_settings)
    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)

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
    A_A_diff, A_B_diff, A_positions, B_positions = find_A_B_distance_and_positions(sess_dataframe, ses_settings, rewarded_As)
    
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

def safe_divide(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # result has same shape as a
    out = np.full_like(a, np.nan, dtype=float)

    # numpy handles broadcasting for where= and division
    np.divide(a, b, out=out, where=(b != 0))

    return out

def estimate_release_events(sess_dataframe, ses_settings):
    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']
    offset = trial['offsets'][0]

    lm_gap = lm_size + offset 

    tmp = sess_dataframe.reset_index(drop=False, inplace=False)
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
            chosen_idx, _, odour, chosen_pos = find_closest_events(tmp, closed_idx, pos_window = lm_size /2, event_priority=["release"], choose = "earliest")
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
            chosen_idx, _, odour, chosen_pos = find_closest_events(tmp, closed_idx, pos_window = lm_size /2, event_priority=["prepare", "flush"], choose = "average")
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
    first_release = extract_int(trial['landmarks'][0][0]['odour'])
    if first_release != 0 and (len(result) == 0 or first_release != result[0][3]):
        result = [[pd.NaT, 0, -1, first_release]] + result

    result_df = pd.DataFrame(result,
                              columns=["time", "Position", "Index", "Odour"]
                              ).set_index("time")

    return result_df

def find_closest_events(
    df: pd.DataFrame,
    idx: int,
    pos_window: float = 3.0,
    event_priority=["release", "prepare", "flush"],
    choose = 'earliest',
    verbose = False,
):
    """
    For each idx, find the nearest event based on Position.

    For each event type in priority:
        - Search in a zigzag pattern around the idx:
          row, row-1, row+1, row-2, row+2, ...
        - At each candidate row j:
            * Require |Position(j) - pos0| <= pos_window
            * Skip odour 0
        - Stop searching in a direction once Position falls outside pos_window
        - Each candidate row is saved into a list candidate_idx
    
    Final steps choose representative idx from candidate_idx based on: choose
    """
    events_col = df["Events"].astype("string")
    n_rows = len(df)
    pos0 = df.at[idx, "Position"]

    candidate_idx = []
    chosen_idx = None
    chosen_event = None
    odour = None
    chosen_pos = None

    for ev_type in event_priority:
        # ---------- Zigzag search around idx ----------
        offset = 0
        up_active = True
        down_active = True

        while up_active or down_active:
            # Check current / upward direction: idx - offset
            if up_active:
                j_up = idx - offset
                if j_up < 0:
                    up_active = False
                else:
                    if abs(df.at[j_up, "Position"] - pos0) > pos_window:
                        # we assume Position is monotonic, so further up is outside window
                        up_active = False
                    else:
                        ev = events_col.iat[j_up]
                        if ev is not None and not pd.isna(ev) and "odour0" not in ev:
                            if ev_type in ev:
                                candidate_idx.append(j_up)

            # Check downward direction only for offset > 0 to avoid double-checking idx
            if offset > 0 and down_active:
                j_down = idx + offset
                if j_down >= n_rows:
                    down_active = False
                else:
                    if abs(df.at[j_down, "Position"] - pos0) > pos_window:
                        # we assume Position is monotonic, so further down is outside window
                        down_active = False
                    else:
                        ev = events_col.iat[j_down]
                        if ev is not None and not pd.isna(ev) and "odour0" not in ev:
                            if ev_type in ev:
                                candidate_idx.append(j_down)

            offset += 1  # expand zigzag radius

    if len(candidate_idx) > 0:
        if choose == 'earliest':
            chosen_idx = min(candidate_idx)
            chosen_event = events_col.iat[chosen_idx]
            chosen_pos = df.at[chosen_idx, "Position"]
            odour = extract_int(chosen_event)
        elif choose == 'average':
            center_idx = candidate_idx[0]
            chosen_idx = int(np.average(candidate_idx))

            chosen_pos = df.at[chosen_idx, "Position"]
            odour = extract_int(events_col.iat[center_idx])
            chosen_event = 'Estimated release: odour' + str(odour)
        else:
            raise NotImplementedError

        if verbose:
            if chosen_event is None:
                raise ValueError(
                    f"No event of types {event_priority} found within ±{pos_window} "
                    f"for expected release event around idx: {idx}."
                )

    return chosen_idx, chosen_event, odour, chosen_pos

def sanity_check_parsing(sess_dataframe, ses_settings):

    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
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

def threshold_lick_eventslick_speed(sess_dataframe, speed_threshold=0.3):

    treadmill_speed = sess_dataframe['Treadmill'].to_numpy()
    lick_events = sess_dataframe['Licks'].to_numpy()

    # Create a boolean mask where speed is below threshold
    low_speed_mask = treadmill_speed < speed_threshold

    # Apply the mask to lick events
    filtered_licks = lick_events * low_speed_mask

    # Update the session dataframe with filtered licks
    sess_dataframe['Licks'] = filtered_licks

    return sess_dataframe

def estimate_lm_events(sess_dataframe):
    # print(sess_dataframe)

    lm_position = sess_dataframe['LM_Position'].values[sess_dataframe['LM_Count'].values >= 0]

    lm_time = sess_dataframe.index[sess_dataframe['LM_Count'].values >= 0]

    lm_odour = sess_dataframe['LM_Odour'].values[sess_dataframe['LM_Count'].values >= 0]
    lm_odour = [extract_int(odour) for odour in lm_odour]

    lm_index = sess_dataframe['Buffer'].values[sess_dataframe['LM_Count'].values >= 0]

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

def print_sess_summary(sess_dataframe,ses_settings):

    rew_odour, rew_texture, non_rew_odour, non_rew_texture = parse_rew_lms(ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)

    print(f'Session Summary:')
    print(f"Total Licks: {sess_dataframe['Licks'].sum()}")
    print(f"Total Landmarks: {licked_all.shape[0]}")
    print(f"Total Rewards: {sess_dataframe['Rewards'].notna().sum()}")
    print(f'Hit Rate: {hit_rate*100:.2f}%, False Alarm Rate: {fa_rate*100:.2f}%, D-prime: {d_prime:.2f}')
    print(f'Targets Licked: {np.sum(licked_target).astype(int)} of {len(licked_target)}, Distractors Licked: {np.sum(licked_distractor).astype(int)} of {len(licked_distractor)}')
    print(f'rewarded odours: {rew_odour}, rewarded textures: {rew_texture}')
    print(f'non-rewarded odours: {non_rew_odour}, non-rewarded textures: {non_rew_texture}')

# def get_num_landmarks(session):
#     # Get number of unique landmarks for the session
#     session['num_landmarks'] = len(session['lm_ids'])

#     return session

def get_licks_idx(session, lick_threshold=True):
    '''Get the idx of licks in the session'''

    if lick_threshold:
        session = threshold_licks(session)
    else:
        licks_idx = np.where(session['licks'])[0]
        session['licks_idx'] = licks_idx

    return session 

def threshold_licks(session):
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

def threshold_lick_events(sess_dataframe, ses_settings, below=True):

    session = create_session_struct(sess_dataframe, ses_settings)

    licks = sess_dataframe['Licks'].values.astype(int)
    
    if below == True:
        speed_ok = session['speed'] < session['lick_threshold']
    else:
        speed_ok = session['speed'] >= session['lick_threshold']
    licked = licks > 0
    threshold_mask = speed_ok & licked

    thresholded_licks = np.zeros(len(licks))
    thresholded_licks[threshold_mask] = licks[threshold_mask]

    return thresholded_licks

def create_odour_lm_mapping(ses_settings):
    '''Create a list of rewarded and non-rewarded odours based on the order in which they are created in the session settings file'''
    
    odour_lm_id_mapping = []
    for lm_list in ses_settings['trial']['landmarks']:
        for lm in lm_list:
            odour_id = extract_int(lm['odour'])
            if np.isin(odour_id, odour_lm_id_mapping) or odour_id == 0:
                break
            else:
                odour_lm_id_mapping.append(odour_id)

    return odour_lm_id_mapping

def calculate_frame_lick_rate(session):
    """Get lick rate per frame as a sliding window"""
    
    # Calculate lick rate as the mean number of licks over sliding window
    window = 100 # frames
    lick_rate = np.zeros(len(session['position']))
    for i in range(len(session['position'])-window):
        lick_num = len(np.where((session['licks_idx'] > i) & (session['licks_idx'] < i+window))[0])
        lick_rate[i] = lick_num / window
    
    session['frame_lick_rate'] = lick_rate

    return session

#%% ##### Functions that work with NIDAQ data only (after funcimg alignment) #####
def get_landmark_positions(session, sess_dataframe, ses_settings, data='pd'):
    '''Get the start and end of each landmark'''
    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']

    if data == 'odour':
        # Estimate landmark entries based on odour release positions 
        lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
        lm_idx = np.asarray(release_df['Index'].to_numpy(), dtype=int)
        
        position = np.nan_to_num(sess_dataframe['Position'].values, nan=0.0)
        release_positions = position[lm_idx]
        
        landmarks = np.zeros((len(release_positions), 2))
        for lm, pos in enumerate(release_positions):
            landmarks[lm,0] = pos
            landmarks[lm,1] = pos + lm_size

    elif data == 'pd':
        lm_entry_idx1, lm_exit_idx1 = estimate_pd_entry_exit(ses_settings, session, pd='pd1')
        lm_entry_idx2, lm_exit_idx2 = estimate_pd_entry_exit(ses_settings, session, pd='pd2')
        
        entry_pos1 = session['position'][lm_entry_idx1]
        entry_pos2 = session['position'][lm_entry_idx2]
        exit_pos1  = session['position'][lm_exit_idx1]
        exit_pos2  = session['position'][lm_exit_idx2]
        
        trial = ses_settings['trial']
        if isinstance(trial, list):
            trial = trial[0]['trial']
        lm_size = trial['landmarks'][0][0]['size']
        offset = ses_settings['trial']['offsets'][0]
        tol = lm_size * 0.5

        # Merge with "keep single" logic
        all_lm_entry = merge_positions_keep_single(entry_pos1, entry_pos2, tol, offset)
        all_lm_exit  = merge_positions_keep_single(exit_pos1,  exit_pos2,  tol, offset)

        # Fix last lm 
        if session['position'][-1] - all_lm_exit[-1] < lm_size:
            all_lm_exit = all_lm_exit[:-1]
        
        # Fix first lm 
        first_entries = all_lm_entry < offset
        first_exits  = all_lm_exit  < offset
        first_entry = all_lm_entry[first_entries][0] 
        first_exit = all_lm_exit[first_exits][-1]
        
        # Concatenate all landmarks 
        lm_entry = np.concatenate([[first_entry], all_lm_entry[~first_entries]])
        lm_exit = np.concatenate([[first_exit], all_lm_exit[~first_exits]])

        if len(lm_entry) != len(lm_exit):
            if len(lm_entry) - len(lm_exit) == 1:
                # Session ended before the mouse exited the last landmark 
                n = len(lm_exit)
                lm_entry = lm_entry[:n]
            else:
                raise ValueError(f'Something is wrong with landmark parsing using the photodiode data in {session['mouse']} {session['stage']}')

        # Store landmarks 
        landmarks = np.column_stack([lm_entry, lm_exit])

    session['landmarks'] = landmarks

    return session

def merge_positions_keep_single(pos1, pos2, tol, offset):
    """
    Merge two sorted position arrays.
    - If positions are within tol → average
    - If only one exists → keep it
    """
    i = j = 0
    merged = []

    while i < len(pos1) and j < len(pos2):
        if pos1[i] < offset:
            merged.append(pos1[i])
            i += 1
            continue

        if pos2[j] < offset:
            merged.append(pos2[j])
            j += 1
            continue

        if abs(pos1[i] - pos2[j]) <= tol:
            merged.append(np.mean([pos1[i], pos2[j]]))
            i += 1
            j += 1
        elif pos1[i] < pos2[j]:
            merged.append(pos1[i])
            i += 1
        else:
            merged.append(pos2[j])
            j += 1

    # append leftovers
    while i < len(pos1):
        merged.append(pos1[i])
        i += 1

    while j < len(pos2):
        merged.append(pos2[j])
        j += 1

    return np.array(merged)

def estimate_pd_entry_exit(ses_settings, session, pd='pd1'):
    '''Estimate lm entry and exit indices using photodiode data'''
    binary_pd = (session[pd] >= 100).astype(int)

    all_lm_entry_idx = np.where(np.diff(binary_pd) == 1)[0] + 1
    all_lm_exit_idx = np.where(np.diff(binary_pd) == -1)[0] + 1
    if binary_pd[0] == 1:
        all_lm_entry_idx = np.insert(all_lm_entry_idx, 0, 0)
    
    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']
    # lm_size = ses_settings['trial']['landmarks'][0][0]['size']
    offset = ses_settings['trial']['offsets'][0]

    # Filter out repeated lm visits
    # n = min(len(all_lm_entry_idx), len(all_lm_exit_idx))
    # entry_pos = session['position'][all_lm_entry_idx[:n]]
    # exit_pos  = session['position'][all_lm_exit_idx[:n]]
    entry_pos = session['position'][all_lm_entry_idx]
    exit_pos  = session['position'][all_lm_exit_idx]
    # pos_diff = np.where(exit_pos - entry_pos < lm_size - 1)[0]

    if offset == lm_size: # TODO bad fix 
        tol = 0
    else:
        tol = 1

    # Filter out re-entries - use earliest idx
    consecutive_diff = np.where(np.diff(entry_pos) < lm_size + tol)[0] + 1
    removed = []
    for i, idx in enumerate(all_lm_entry_idx):
        if i in consecutive_diff:
            if session['position'][idx] < offset:
                continue
            removed.append(i)
    lm_entry_idx = np.delete(all_lm_entry_idx, removed)

    # Filter out re-exits - use latest idx
    consecutive_diff = np.where(np.diff(exit_pos) < lm_size + tol)[0] 
    removed = []
    for i, idx in enumerate(all_lm_exit_idx):
        if i in consecutive_diff:
            if session['position'][idx] < offset:
                continue
            removed.append(i)
    lm_exit_idx = np.delete(all_lm_exit_idx, removed)

    # # Filter out re-entries - use earliest entry
    # removed = []
    # for i, idx in enumerate(all_lm_entry_idx):
    #     if i in pos_diff:
    #         # Check position of first outlier
    #         if session['position'][idx] < offset:
    #             continue
    #         removed.append(i)
    # lm_entry_idx = np.delete(all_lm_entry_idx, removed)

    # # Filter out re-exits - use latest exit
    # removed = []
    # for i, idx in enumerate(all_lm_exit_idx):
    #     if i in pos_diff:
    #         # Check position of first outlier
    #         if session['position'][idx] < offset:
    #             continue
    #         removed.append(i)
    # lm_exit_idx = np.delete(all_lm_exit_idx, removed)

    return lm_entry_idx, lm_exit_idx

def get_lm_entry_exit(session):
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

def get_rewarded_landmarks(session):
    '''Find the indices of rewarded (lick-triggered) landmarks.'''

    lm_entry_idx, lm_exit_idx = get_lm_entry_exit(session)

    # Find rewarded landmarks 
    reward_positions = session['position'][session['reward_idx']]

    rewarded_landmarks = [i for i, (start, end) in enumerate(zip(np.floor(session['position'][lm_entry_idx]), np.ceil(session['position'][lm_exit_idx]))) 
                            if np.any((np.ceil(reward_positions) >= start) & (np.floor(reward_positions) <= end))] 

    session['rewarded_landmarks'] = rewarded_landmarks

    return session


def get_reward_idx(session):
    # Ensure mouse has left last rewarded landmark 
    reward_idx = session['rewards']
    if session['all_landmarks'][-1,1] < session['position'][reward_idx[-1]]:  
        reward_idx = reward_idx[0:-1]  
        print('Mouse did not leave the last rewarded landmark. Removing landmark...')

    session['reward_idx'] = reward_idx

    return session 

#%% ##### Analysis wrappers #####
def create_session_struct(sess_dataframe, ses_settings):

    # Use the Buffer as datapoint idx
    position = np.nan_to_num(sess_dataframe['Position'].values, nan=0.0)
    speed = np.nan_to_num(sess_dataframe['Treadmill'].values, nan=0.0)
    licks = sess_dataframe['Licks'].values.astype(int)
    rewards = sess_dataframe['Buffer'][sess_dataframe['Rewards'].notna()].values    
    lick_threshold = ses_settings['velocityThreshold']

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']

    session = {'position': position,
               'speed': speed,
               'licks': licks, 
               'rewards': rewards,
               'lick_threshold': lick_threshold,
               'lm_size': lm_size
               }
    
    return session
    
#%% ##### Plotting #####
def plot_ethogram(sess_dataframe, ses_settings):
    lick_position = sess_dataframe['Position'].values[sess_dataframe['Licks'].values > 0]
    lick_times = sess_dataframe.index[sess_dataframe['Licks'].values > 0]
    reward_times = sess_dataframe.index[sess_dataframe['Rewards'].notna()]
    reward_positions = sess_dataframe['Position'].values[sess_dataframe['Rewards'].notna()]
    if 'LM_Count' in sess_dataframe.columns:
        release_df = estimate_lm_events(sess_dataframe)
    else:
        release_df = estimate_release_events(sess_dataframe, ses_settings)
    release_times = release_df.index.tolist() # time
    release_times = release_times[1:]  # remove first release for plotting because sometimes the timestamp is NaN
    release_positions = release_df["Position"].tolist()
    release_positions = release_positions[1:]  # remove first release for plotting because sometimes the timestamp is NaN   

    # num_laps, sess_dataframe = divide_laps(sess_dataframe, ses_settings)

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

def compute_psth_pair(ses_settings, sess_dataframe, events, bins):
    speed = get_speed_psth(ses_settings, sess_dataframe, events=events, bins=bins)
    licks = get_lick_rate_psth(ses_settings, sess_dataframe, events=events, bins=bins)
    return speed, licks

def plot_speed_lick_rate_psth(ses_settings, sess_dataframe, session_id, bins=None):

    if 'LM_Count' in sess_dataframe.columns:
        release_df = estimate_lm_events(sess_dataframe)
    else:
        release_df = estimate_release_events(sess_dataframe, ses_settings)

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
    if 'full' in session_id:
        landmarks, lm_idx = get_landmarks(sess_dataframe, ses_settings)

    else:
        # Binary sequence
        A_landmarks, B_landmarks, A_idx, B_idx = get_A_B_landmarks(sess_dataframe, ses_settings)

    # --- Define groups dynamically ---
    groups = {}

    if 'abab' in session_id:
        groups = {
            'A': (A_idx, 'darkblue'),
            'B': (B_idx, 'orange')
        }

    elif 'aabb' in session_id or 'a2b2' in session_id:
        groups = {
            'A1': (A_idx[::2], 'darkblue'),
            'A2': (A_idx[1::2], 'blue'),
            'B1': (B_idx[::2], 'orange'),
            'B2': (B_idx[1::2], 'gold')
        }

    elif 'abb' in session_id and 'abbb' not in session_id:
        groups = {
            'A': (A_idx, 'darkblue'),
            'B1': (B_idx[::2], 'orange'),
            'B2': (B_idx[1::2], 'gold')
        }

    elif 'abbb' in session_id:
        groups = {
            'A': (A_idx, 'darkblue'),
            'B1': (B_idx[::3], 'orange'),
            'B2': (B_idx[1::3], 'gold'),
            'B3': (B_idx[2::3], 'brown')
        }

    elif 'aab' in session_id and 'aabb' not in session_id:
        groups = {
            'A1': (A_idx[::2], 'darkblue'),
            'A2': (A_idx[1::2], 'blue'),
            'B1': (B_idx, 'orange'),
        }

    elif 'full' in session_id:
        groups = {
            'lm': (lm_idx, 'black')
        }

    # --- Compute + plot ---
    for label, (events, color) in groups.items():

        (mean_s, sem_s), (mean_l, sem_l) = compute_psth_pair(
            ses_settings, sess_dataframe, events, bins
        )

        plot_psth(ax_speed, mean_s, sem_s, color, label)
        plot_psth(ax_lick, mean_l, sem_l, color, label)

    ax_speed.axhline(ses_settings['velocityThreshold'], linestyle='--', color='grey')

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

def plot_speed_psth_distance_groups(sess_dataframe, ses_settings, psth_A, psth_B, distance_groups, bins=300):
    """
    Plot PSTHs for each distance group as subplots.
    Each subplot shows A + all B landmarks.
    """
    if 'LM_Count' in sess_dataframe.columns:
        release_df = estimate_lm_events(sess_dataframe)
    else:
        release_df = estimate_release_events(sess_dataframe, ses_settings)

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
        ax.axhline(ses_settings['velocityThreshold'], linestyle='--', color='grey')
        ax.set_xlabel('Time (bins)')
        ax.legend(frameon=False)

    # plt.tight_layout()
    return fig

def plot_transition_matrix(sess_dataframe, ses_settings):
    from matplotlib.colors import Normalize

    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    
    # Decide if matrix permuation is needed for plotting (currently only for AAB)
    perm = None
    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    reward_seq = np.array([lm[0]['rewardSequencePosition'] for lm in trial['landmarks']])
    if len(reward_seq) == 3:
        A_landmarks = list(np.where(reward_seq == 0)[0])
        if len(A_landmarks) == 2:   # AAB
            perm = np.array([target_id[0], target_id[1], distractor_id[0]])

    transition_matrix, lick_tm, ideal_tm = calc_transition_matrix(sess_dataframe, ses_settings)
    
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

def plot_distance_transition_matrix(sess_dataframe, ses_settings):

    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    transition_matrix, lick_tm, ideal_tm = calc_distance_transition_matrix(sess_dataframe, ses_settings)
    
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

def plot_conditional_matrix(sess_dataframe, ses_settings, n_steps=1):
    
    transition_prob, control_prob, ideal_prob = calc_conditional_matrix(sess_dataframe, ses_settings, n_steps)
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    
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

def plot_lick_lm(sess_dataframe,ses_settings):
    target_id, distractor_id, target_positions, distractor_positions, lm_id, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)

    A_landmarks, _, _, _ = get_A_B_landmarks(sess_dataframe, ses_settings)

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

def plot_full_corr(sess_dataframe,ses_settings):

    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)
    goals, lm_ids = parse_stable_goal_ids(ses_settings)
    #for the length of licked_all, repeat the lm_ids to fill the array
    all_lms = np.concatenate([lm_ids]* (licked_all.shape[0] // lm_ids.shape[0] + 1))[:licked_all.shape[0]]
    was_target = np.zeros_like(all_lms)
    for i in range(all_lms.shape[0]):
        if all_lms[i] in goals:
            match_id = goals.index(all_lms[i])
            was_target[i] = match_id + 1  #start from 1

    #reshape licked_all into 10 columns and the appropriate number of rows
    if licked_all.shape[0] % 10 != 0:
        #extend the array to make it divisible by 10
        licked_all = np.pad(licked_all, (0, 10 - (licked_all.shape[0] % 10)), 'constant')
    licked_all_reshaped = licked_all.reshape(np.round(licked_all.shape[0] / 10).astype(int), 10)
    if rewarded_all.shape[0] % 10 != 0:
        #extend the array to make it divisible by 10
        rewarded_all = np.pad(rewarded_all, (0, 10 - (rewarded_all.shape[0] % 10)), 'constant')
    rewarded_all_reshaped = rewarded_all.reshape(np.round(rewarded_all.shape[0] / 10).astype(int), 10)
    if was_target.shape[0] % 10 != 0:
        #extend the array to make it divisible by 10
        was_target = np.pad(was_target, (0, 10 - (was_target.shape[0] % 10)), 'constant')
    was_target_reshaped = was_target.reshape(np.round(was_target.shape[0] / 10).astype(int), 10)
    if all_lms.shape[0] % 10 != 0:
        #extend the array to make it divisible by 10
        all_lms = np.pad(all_lms, (0, 10 - (all_lms.shape[0] % 10)), 'constant')
    all_lms_reshaped = all_lms.reshape(np.round(all_lms.shape[0] / 10).astype(int), 10)

    plt.figure(figsize=(10,6))
    plt.subplot(3, 1, 1)
    plt.imshow(was_target_reshaped, aspect='auto', cmap='viridis', interpolation='none')
    plt.clim(0, len(goals))
    plt.title('Landmark ID (Full Corridor)')
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.imshow(licked_all_reshaped, aspect='auto', cmap='viridis', interpolation='none')
    plt.clim(0, 1)
    plt.title('Licked All (Full Corridor)')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    plt.imshow(rewarded_all_reshaped, aspect='auto', cmap='viridis', interpolation='none')
    plt.clim(0, 1)
    plt.title('Rewarded All (Full Corridor)')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def plot_sw_hit_fa(sess_dataframe,ses_settings,window=10):

    target_positions, distractor_positions, target_id, distractor_id, was_target, lm_id = find_targets_distractors(sess_dataframe,ses_settings)
    hit_rate, fa_rate,d_prime, licked_target, licked_distractor, licked_all,rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)

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

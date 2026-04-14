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

def format_condition_label(cond):
    # Determine the AB counts
    if 'abb' in cond and 'abbb' not in cond:
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

    if os.path.exists(Path(base_path) / "behav/current-landmark/"):
        lm_reader = Csv("behav/current-landmark/*", ["Seconds","Count","Size","Texture","Odour","SequencePosition","Position","Visited","RewardDelivered"])
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
            'LM_Position': pd.Series(sess_lm_data['Position'], index=sess_lm_data.index)
        }
        # print(sess_data)
        # #deal with index duplication (occurs if copying data breaks)
        # keys_to_remove = []
        # for k, v in sess_data.items():
        #     if not hasattr(v, "index"):
        #         continue

        #     if not v.index.is_unique:
        #         print(k, "has duplicate indices")
        #         keys_to_remove.append(k)
                
        # for k in keys_to_remove:
        #     sess_data.pop(k, None)
        
        # #combine indices
        # all_ix = None
        # for v in sess_data.values():
        #     if hasattr(v, "index"):
        #         all_ix = v.index if all_ix is None else all_ix.union(v.index)

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

def load_dF(base_path, red_chan=False):
    '''Load dF and valid frames and find valid neurons'''

    nwb_path = parse_nwb_functions.find_nwbfile(base_path)
    io = NWBHDF5IO(nwb_path, mode='r')
    nwb = io.read()
    
    # Find valid neurons 
    segmentation = nwb.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentationChan1Plane0'][:]
    neurons = np.where(segmentation['Accepted'] == 1)[0] 

    if not red_chan:
        # Load valid frames
        valid_frames = np.load(os.path.join(base_path, 'valid_frames.npz'))['valid_frames']
        
        # Load data 
        dF_all = nwb.processing['ophys'].data_interfaces['DfOverF'].roi_response_series['DfOverFChan1Plane0'].data[:]
        dF = dF_all[valid_frames,:].T

    else:
        print('Loading dG/R instead of dF')
        # Load data 
        dF_GR_path = os.path.join(base_path, 'funcimg', 'DG_R.npy')
        if os.path.exists(dF_GR_path):
            dF = np.load(dF_GR_path)
        else:
            print('dF_GR file not found - make sure it has been computed already or use dF/F0')            

    io.close()
    return dF, neurons

#%% ##### Session functions #####
def get_event_parsed(sess_dataframe, ses_settings):

    # TODO integrate better
    licks = threshold_lick_events(sess_dataframe, ses_settings)
    # licks = sess_dataframe['Licks'].values
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
    reward_seq = np.array([lm[0]['rewardSequencePosition'] for lm in trial['landmarks']])

    if len(reward_seq) == 4:
        if np.diff(reward_seq)[0] == 0:    
            # AABB: re-order AB so that A is always first
            release_df = release_df[2:]
            
        elif len(np.where(reward_seq == -1)[0]) > 2:    
            # ABBB: get rid of first event if needed otherwise keep the order the same
            if release_positions[0] < 2:
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

    # map to positions
    A1_pos = release_positions[A1]
    A2_pos = release_positions[A2] if len(A2) else np.array([])
    B1_pos = release_positions[B1]
    B2_pos = release_positions[B2]
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
        "B2": compute_events(B2_pos),
    }

    if len(A2_pos):
        events["A2"] = compute_events(A2_pos)
    if len(B3_pos):
        events["B3"] = compute_events(B3_pos)

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


        hit_rate_sw = {"A1": np.zeros(len(release_positions[:-window]))}
        if len(A2) > 0:
            hit_rate_sw["A2"] = np.zeros(len(release_positions[:-window]))

        fa_rate_sw  = {"B1": np.zeros(len(release_positions[:-window])),
                       "B2": np.zeros(len(release_positions[:-window]))}
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
            fa_rate_sw["B2"][idx] = np.sum(licked_B2) / len(licked_B2) 
            if len(B3_positions):
                fa_rate_sw["B3"][idx] = np.clip(np.sum(licked_B3) / len(licked_B3), 0.01, 0.99)

            # adjust hit rate and fa rate to avoid infinity in d-prime calculation
            hit_rate_sw["A1"][idx] = np.clip(hit_rate_sw["A1"][idx], 0.01, 0.99)
            fa_rate_sw["B1"][idx] = np.clip(fa_rate_sw["B1"][idx], 0.01, 0.99)
            fa_rate_sw["B2"][idx] = np.clip(fa_rate_sw["B2"][idx], 0.01, 0.99)

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
        hit_rate = {}
        fa_rate = {}

        for d in np.unique(distances):
            lms_considered = np.where(distances == d)[0]

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

            hit_rate[d] = (np.sum(licked_target) / len(licked_target)
                if len(licked_target) > 0 else np.nan)

            fa_rate[d] = (np.sum(licked_distractor) / len(licked_distractor)
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
            lms_considered = np.where(distances == d)[0]
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
                d_sorted = sorted(hit_rate.keys())
                plt.plot(d_sorted, [hit_rate[d] for d in d_sorted],
                        c='darkblue', linewidth=2, label='hit rate')
                plt.plot(d_sorted, [fa_rate[d] for d in d_sorted],
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
    
def calc_time_hit_fa(sess_dataframe, ses_settings, bins=10, plot=True):
    '''Calculate hit and fa rates based on time spent between landmarks'''

    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']
    # lm_size = ses_settings['trial']['landmarks'][0][0]['size']
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

def extract_int(s: str) -> int:
    m = re.search(r'\d+', s)
    if m:
        return int(m.group())
    else:
        raise ValueError(f"No digits found in string: {s!r}")

def get_A_B_landmarks(sess_dataframe, ses_settings):
    '''Find which landmarks are rewarded (A) or non-rewarded (B)'''

    # Get landmark visits
    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    lm_idx = np.asarray(release_df['Index'].to_numpy(), dtype=int) # TODO rename because it conflicts with another definition

    # Get the sequence of landmarks
    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    reward_seq = np.array([lm[0]['rewardSequencePosition'] for lm in trial['landmarks']])

    # Split As and Bs into A1-A2-B1-B2
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
        if A_landmarks[0] == 0:
            A_landmarks[0] = 2

        # first_rew = np.where(reward_positions > sess_dataframe['Position'].iloc[lm_idx[0]])[0][0]
        # first_A_lag = np.argmin(np.abs(sess_dataframe['Position'].iloc[lm_idx] - reward_positions[first_rew]))
        # A_landmarks = [first_A_lag]
        B_landmarks = [i for i in range(len(reward_seq)) if (i not in A_landmarks)]

    for a in range(len(np.where(reward_seq == 0)[0])):
        A_landmarks.extend([i for i in range(A_landmarks[a]+len(reward_seq), len(lm_idx), len(reward_seq)) if i < len(lm_idx)])
    for b in range(len(np.where(reward_seq == -1)[0])):
        B_landmarks.extend([i for i in range(B_landmarks[b]+len(reward_seq), len(lm_idx), len(reward_seq)) if i < len(lm_idx)])

    A_landmarks = np.sort(A_landmarks)
    B_landmarks = np.sort(B_landmarks)

    # Split the data indices into A1-A2-B1-B2
    A_idx = [lm_idx[i] for i in A_landmarks]
    B_idx = [lm_idx[i] for i in B_landmarks]

    assert len(lm_idx) == (len(A_landmarks) + len(B_landmarks)), 'Some landmarks are missing!'

    return A_landmarks, B_landmarks, A_idx, B_idx

def find_targets_distractors(sess_dataframe, ses_settings):
    '''Give an id to each type of landmark'''

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    reward_seq = np.array([lm[0]['rewardSequencePosition'] for lm in trial['landmarks']])

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
        distractor_id = np.atleast_1d(lm_id[1:])
        target_id = np.atleast_1d(lm_id[0])
    
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
        lm_id_sequence[A_landmarks] = np.tile(target_id, len(A_landmarks))
        lm_id_sequence[B_landmarks] = np.tile(distractor_id, int(np.ceil(len(B_landmarks)/2)))[:len(B_landmarks)]

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

def calc_distance_transition_matrix(sess_dataframe, ses_settings, binning=True):
    '''
    Create a lick transition matrix based on distance between current and next licked landmark.
    If binning, distances will be grouped into small, medium, large.
    '''
    release_df = estimate_lm_events(sess_dataframe)

    trial = ses_settings['trial']
    if isinstance(trial, list):
        trial = trial[0]['trial']
    lm_size = trial['landmarks'][0][0]['size']
    num_landmarks = len(trial['landmarks'])

    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)

    ideal_licks = get_ideal_performance(sess_dataframe, ses_settings) # TODO 
    ideal_sequence = lm_id_sequence[ideal_licks==1]

    distances = np.diff(release_df['Position']) - lm_size
    distance_range = np.sort(np.unique(distances))

    # Group distances
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

    # Calculate stimulus, lick, and ideal transition matrices
    transition_matrix = {}
    lick_tm = {}
    ideal_tm = {}

    for key, dist_vals in distance_groups.items():
        lms_considered = np.where(np.isin(distances, dist_vals))[0]

        # stimulus transition matrix
        transition_matrix[key] = np.zeros((num_landmarks, num_landmarks))
        for i in lms_considered:
            if i + 1 >= len(lm_id_sequence):
                break
            current_lm = int(lm_id_sequence[i])
            next_lm = int(lm_id_sequence[i+1])
            transition_matrix[key][current_lm, next_lm] += 1
        # transition_matrix[d][next_lm, current_lm] += 1 # TODO why is this here? 

        # lick transition matrix
        lick_tm[key] = np.zeros((num_landmarks, num_landmarks))
        for i in lms_considered:
            if licked_all[i] == 1: 
                current_lm = int(lm_id_sequence[i])
                next_licks = np.where(licked_all[i+1:] == 1)[0]
                if len(next_licks) > 0:
                    next_lm_idx = np.where(licked_all[i+1:] == 1)[0][0] + i + 1
                    next_lm = lm_id_sequence[next_lm_idx]
                else:
                    break
                lick_tm[key][current_lm, next_lm] += 1

        # ideal transition matrix
        ideal_tm[key] = np.zeros((num_landmarks, num_landmarks))
        for i in lms_considered:
            current_lm = int(lm_id_sequence[i])
            # print(current_lm)
            if current_lm in target_id:
                next_target = np.where(ideal_licks[i+1:] == 1)[0]
                if len(next_target) > 0:
                    next_target_idx = next_target[0] + i + 1
                    next_lm = lm_id_sequence[next_target_idx] 
                else:
                    break
                ideal_tm[key][current_lm, next_lm] += 1

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

def calc_sequencing_performance(sess_dataframe, ses_settings):
    '''Calculate sequencing performance i.e., the conditional A->A (hit) and A->B (fa) transitions, for each target (A)'''
    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)

    As = list(target_id)
    Bs = list(distractor_id)
    lm_ids = list(lm_ids)

    # 1. Find number of steps ahead to look for As and Bs
    # AABB: A1 +1+4 / +2+3 A2 +3+4 / +1+2
    # ABAB: A1 +2+4 / +1+3 A2 +2+4 / +1+3

    # calculate number of steps ahead to get to next A and back to the same A
    next_A_steps = {}
    for a, A in enumerate(As):
        next_A = As[a+1] if a+1 < len(As) else As[0]
        next_A_steps[A] = [(lm_ids.index(next_A) - lm_ids.index(A)) % len(lm_ids)]
        next_A_steps[A].append(len(lm_ids))
    # print(next_A_steps)

    # calculate number of steps ahead to get to Bs
    next_B_steps = {A: [] for A in As}
    for A in As:
        for B in Bs:
            next_B_steps[A].append((lm_ids.index(B) - lm_ids.index(A)) % len(lm_ids))
    # print(next_B_steps)

    # 2. Calculate transition probabilities for each number of steps ahead and take the mean per target (A) and distractor (B)
    transition_prob_A = {A: [] for A in As}
    for a, A in enumerate(As):
        for steps in next_A_steps[A][:1]:
            trans, _, _ = calc_conditional_matrix(sess_dataframe, ses_settings, n_steps=steps)
            transition_prob_A[A].append(trans[a])
    # print(transition_prob_A)

    mean_transition_prob_A = {A: np.mean(np.stack(steps, axis=0), axis=0) for A, steps in transition_prob_A.items()}
    # print(mean_transition_prob_A)

    distractor_prob_A = {A: [] for A in As}
    for a, A in enumerate(As):
        for steps in next_B_steps[A][:1]:
            trans, _, _ = calc_conditional_matrix(sess_dataframe, ses_settings, n_steps=steps)
            distractor_prob_A[A].append(trans[a])
    # print(distractor_prob_A)

    mean_distractor_prob_A = {A: np.mean(np.stack(steps, axis=0), axis=0) for A, steps in distractor_prob_A.items()}
    # print(mean_distractor_prob_A)

    # 3. Calculate performance metrics using the mean transition probs for As and Bs
    sequencing_hit_rate = {}
    sequencing_fa_rate = {}
    sequencing_d_prime = {}
    for A in As:
        sequencing_hit_rate[A] = np.mean(mean_transition_prob_A[A][target_id])
        sequencing_fa_rate[A] = np.mean(mean_distractor_prob_A[A][distractor_id])
        sequencing_d_prime[A] = np.log10(sequencing_hit_rate[A]/(1-sequencing_hit_rate[A])) - np.log10(sequencing_fa_rate[A]/(1-sequencing_fa_rate[A]))

    for a, A in enumerate(As):
        print(f'A{a+1} sequencing performance: hit rate {sequencing_hit_rate[A]*100:.1f}%, FA rate {sequencing_fa_rate[A]*100:.1f}%, d_prime {sequencing_d_prime[A]:.3f}')

    return sequencing_hit_rate, sequencing_fa_rate, sequencing_d_prime

def calc_performance(sess_dataframe, ses_settings):

    _ = calc_sequencing_performance(sess_dataframe, ses_settings)

    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)
    print(f'Overall performance: hit rate {hit_rate*100:.1f}%, FA rate {fa_rate*100:.1f}%, d_prime {d_prime:.3f}')

    return 


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

def calc_stable_conditional_matrix(sess_dataframe,ses_settings):

    goals, lm_ids = parse_stable_goal_ids(ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)

    transition_prob = np.zeros((np.unique(goals).shape[0], np.unique(lm_ids).shape[0]))
    control_prob = np.zeros((np.unique(goals).shape[0], np.unique(lm_ids).shape[0]))
    ideal_prob = np.zeros((np.unique(goals).shape[0], np.unique(lm_ids).shape[0]))
    licked_lm_ix = np.where(licked_all == 1)[0]
    all_lms = np.concatenate([lm_ids]* (licked_all.shape[0] // lm_ids.shape[0] + 1))[:licked_all.shape[0]]
    controlled_lm_ix = np.where(np.isin(all_lms, goals))[0]
    was_target = np.zeros_like(all_lms)
    for i in range(all_lms.shape[0]):
        if all_lms[i] in goals:
            match_id = goals.index(all_lms[i])
            was_target[i] = match_id + 1  #start from 1

    for g in range(np.unique(goals).shape[0]):
        goal = goals[g]
        rewards = np.intersect1d(np.where(rewarded_all == 1)[0],np.where(all_lms == goal)[0])
        for i,reward in enumerate(rewards):
            if i == len(rewards)-1:
                break
            next_lick_index = licked_lm_ix[licked_lm_ix > reward][0]
            next_control_index = controlled_lm_ix[controlled_lm_ix > reward][0]
            next_lm = all_lms[next_lick_index].astype(int)
            next_control_lm = all_lms[next_control_index].astype(int)
            transition_prob[g,next_lm] += 1
            control_prob[g,next_control_lm] += 1

    for g in range(np.unique(goals).shape[0]):
        next_goal = goals[g+1] if g+1 < len(goals) else goals[0]
        ideal_prob[g,next_goal] += 1

    return transition_prob, control_prob, ideal_prob

def calc_seq_fraction(sess_dataframe, ses_settings, test='transition'):
    
    transition_prob, control_prob, ideal_prob = calc_conditional_matrix(sess_dataframe, ses_settings)

    if test == 'transition':
        test_prob = transition_prob
    elif test == 'control':
        test_prob = control_prob
    elif test == 'ideal':
        test_prob = ideal_prob
    else:
        raise ValueError("Invalid test type. Choose from 'transition', 'control', or 'ideal'.")

    aa_prob = test_prob[0,0]
    ab_prob = test_prob[0,1]
    ac_prob = test_prob[0,2]
    bb_prob = test_prob[1,1]
    bc_prob = test_prob[1,2]
    ba_prob = test_prob[1,0]
    ca_prob = test_prob[2,0]
    cb_prob = test_prob[2,1]
    cc_prob = test_prob[2,2]

    perf_a = safe_divide(ab_prob, np.sum(test_prob[0,:3])) #ignore distractor column(s)
    perf_b = safe_divide(bc_prob, np.sum(test_prob[1,:3]))
    perf_c = safe_divide(ca_prob, np.sum(test_prob[2,:3]))

    performance = np.mean([perf_a, perf_b, perf_c])

    return performance, perf_a, perf_b, perf_c

def safe_divide(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # result has same shape as a
    out = np.full_like(a, np.nan, dtype=float)

    # numpy handles broadcasting for where= and division
    np.divide(a, b, out=out, where=(b != 0))

    return out

def calc_stable_seq_fraction(sess_dataframe,ses_settings,test='transition'):

    goals, lm_ids = parse_stable_goal_ids(ses_settings)
    transition_prob, control_prob, ideal_prob = calc_stable_conditional_matrix(sess_dataframe,ses_settings)

    if len(goals) != 4:
        raise ValueError("Stable sequencing performance calculation is only implemented for 4 goals.")
    
    a = goals[0]
    b = goals[1]
    c = goals[2]
    d = goals[3]

    if test == 'transition':
        test_prob = transition_prob
    elif test == 'control':
        test_prob = control_prob
    elif test == 'ideal':
        test_prob = ideal_prob
    else:
        raise ValueError("Invalid test type. Choose from 'transition', 'control', or 'ideal'.")

    ab_prob = test_prob[0,b]
    ac_prob = test_prob[0,c]
    ad_prob = test_prob[0,d]
    bc_prob = test_prob[1,c]
    ba_prob = test_prob[1,a]
    bd_prob = test_prob[1,d]
    ca_prob = test_prob[2,a]
    cb_prob = test_prob[2,b]
    cd_prob = test_prob[2,d]
    dc_prob = test_prob[3,c]
    db_prob = test_prob[3,b]
    da_prob = test_prob[3,a]

    #one performance metric is just comparing correct to incorrect (but relevant)
    # perf_a = ab_prob / (ab_prob + ac_prob + ad_prob)
    # perf_b = bc_prob / (bc_prob + ba_prob + bd_prob)
    # perf_c = cd_prob / (ca_prob + cb_prob + cd_prob)
    # perf_d = da_prob / (dc_prob + db_prob + da_prob)
    #the other is comparing correct to all other transitions
    perf_a = safe_divide(ab_prob, np.sum(test_prob[0,:]))
    perf_b = safe_divide(bc_prob, np.sum(test_prob[1,:]))
    perf_c = safe_divide(cd_prob, np.sum(test_prob[2,:]))
    perf_d = safe_divide(da_prob, np.sum(test_prob[3,:]))

    performance = np.nanmean([perf_a, perf_b, perf_c, perf_d])

    return performance, perf_a, perf_b, perf_c, perf_d

def calc_stable_seq_fraction_new(sess_dataframe,ses_settings,test='transition'):

    goals, lm_ids = parse_stable_goal_ids(ses_settings)
    transition_prob, control_prob, ideal_prob = calc_stable_conditional_matrix(sess_dataframe,ses_settings)

    if len(goals) != 4:
        raise ValueError("Stable sequencing performance calculation is only implemented for 4 goals.")
    
    if test == 'transition':
        test_prob = transition_prob
    elif test == 'control':
        test_prob = transition_prob
        sorted_goals = np.sort(goals)
        goals_new = [sorted_goals[sorted_goals > g][0] if np.any(sorted_goals > g) else sorted_goals[0]
                    for g in np.roll(goals,1)]
        goals = goals_new.copy()
    elif test == 'ideal':
        test_prob = ideal_prob
    else:
        raise ValueError("Invalid test type. Choose from 'transition', 'control', or 'ideal'.")

    a = goals[0]
    b = goals[1]
    c = goals[2]
    d = goals[3]

    ab_prob = test_prob[0,b]
    ac_prob = test_prob[0,c]
    ad_prob = test_prob[0,d]
    bc_prob = test_prob[1,c]
    ba_prob = test_prob[1,a]
    bd_prob = test_prob[1,d]
    ca_prob = test_prob[2,a]
    cb_prob = test_prob[2,b]
    cd_prob = test_prob[2,d]
    dc_prob = test_prob[3,c]
    db_prob = test_prob[3,b]
    da_prob = test_prob[3,a]

    #one performance metric is just comparing correct to incorrect (but relevant)
    # perf_a = ab_prob / (ab_prob + ac_prob + ad_prob)
    # perf_b = bc_prob / (bc_prob + ba_prob + bd_prob)
    # perf_c = cd_prob / (ca_prob + cb_prob + cd_prob)
    # perf_d = da_prob / (dc_prob + db_prob + da_prob)
    #the other is comparing correct to all other transitions
    perf_a = safe_divide(ab_prob, np.sum(test_prob[0,:]))
    perf_b = safe_divide(bc_prob, np.sum(test_prob[1,:]))
    perf_c = safe_divide(cd_prob, np.sum(test_prob[2,:]))
    perf_d = safe_divide(da_prob, np.sum(test_prob[3,:]))

    performance = np.nanmean([perf_a, perf_b, perf_c, perf_d])

    return performance, perf_a, perf_b, perf_c, perf_d

def calculate_corr_length(ses_settings):
    landmarks = ses_settings['trial']['landmarks']
    if len(ses_settings['trial']['offsets']) == 1:
        offset = ses_settings['trial']['offsets'][0]
    else:
        print("Cannot compute corridor lengths when offsets are randomised")

    sum_length = 0
    for lm in landmarks:
        sum_length += lm[0]['size']
        sum_length += offset
    return sum_length

def divide_laps(sess_dataframe, ses_settings):
    # Divide the session dataframe into laps based on the position and corridor length

    corridor_length = calculate_corr_length(ses_settings)
    num_laps = int(np.ceil(sess_dataframe['Position'].max() / corridor_length))

    #for each position, determine which lap it belongs to
    sess_dataframe['Lap'] = (sess_dataframe['Position'] // corridor_length).astype(int)

    return num_laps, sess_dataframe

def calc_laps_needed(ses_settings):

    goals,lm_ids = parse_stable_goal_ids(ses_settings)
    laps_needed = 1

    for i in range(len(goals)-1):
        if goals[i+1] - goals[i] < 0:
            laps_needed += 1
    if goals[0] - goals[-1] > 0:
        laps_needed -= 1

    return laps_needed

def give_state_id(sess_dataframe, ses_settings):

    goals,lm_ids = parse_stable_goal_ids(ses_settings)
    laps_needed = calc_laps_needed(ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)
    if rewarded_all.shape[0] % 10 != 0:
        #extend the array to make it divisible by 10
        rewarded_all = np.pad(rewarded_all, (0, 10 - (rewarded_all.shape[0] % 10)), 'constant')
    rewarded_all_reshaped = rewarded_all.reshape(np.round(rewarded_all.shape[0] / 10).astype(int), 10)

    num_lms = len(lm_ids)

    num_laps, sess_dataframe = divide_laps(sess_dataframe, ses_settings)

    state_id = np.zeros(len(rewarded_all_reshaped), dtype=int)
    state_id[0] = 0

    if laps_needed == 2:
        flips = np.where(np.diff(goals) < 0)[0]
        if len(flips) == 2:
            defining_goal_1 = goals[flips[0]]
            defining_goal_2 = goals[flips[1]]
        else:
            defining_goal_1 = goals[flips[0]]
            if goals[-1] - goals[0] > 0:
                defining_goal_2 = goals[-1]
            else:
                defining_goal_2 = goals[2]

        for i in range(0,num_laps-1):
            if rewarded_all_reshaped[i,defining_goal_1] == 1:
                state_id[i+1] = 1

            if state_id[i]==1 and rewarded_all_reshaped[i,defining_goal_2] == 1:
                state_id[i+1] = 0
            elif state_id[i]==1 and rewarded_all_reshaped[i,defining_goal_2] == 0:
                state_id[i+1] = state_id[i]

    elif laps_needed == 3:
        flips = np.where(np.diff(goals) < 0)[0]
        if len(flips) == 3:
            defining_goal_1 = goals[flips[0]]
            defining_goal_2 = goals[flips[1]]
            defining_goal_3 = goals[flips[2]]
        else:
            defining_goal_1 = goals[flips[0]]
            defining_goal_2 = goals[flips[1]]
            if goals[-1] - goals[0] > 0:
                defining_goal_3 = goals[-1]
            else:
                defining_goal_3 = goals[2]

        for i in range(0,num_laps-1):
            if rewarded_all_reshaped[i,defining_goal_1] == 1:
                state_id[i+1] = 1

            if state_id[i]==1 and rewarded_all_reshaped[i,defining_goal_2] == 1:
                state_id[i+1] = 2
            elif state_id[i]==1 and rewarded_all_reshaped[i,defining_goal_2] == 0:
                state_id[i+1] = state_id[i]
            elif state_id[i]==2 and rewarded_all_reshaped[i,defining_goal_3] == 1:
                state_id[i+1] = 0
            elif state_id[i]==2 and rewarded_all_reshaped[i,defining_goal_3] == 0:
                state_id[i+1] = state_id[i]

    return state_id

def calc_speed_per_lap(sess_dataframe, ses_settings):
    num_laps, sess_dataframe = divide_laps(sess_dataframe, ses_settings)
    # max position is the max of all positions where lap id is 0
    max_position = sess_dataframe['Position'][sess_dataframe['Lap'] == 0].max()
    max_position = np.round(max_position).astype(int)

    bins = 50
    bin_edges = np.linspace(0, max_position, bins+1)
    speed_per_bin = np.zeros((num_laps, bins))
    for i in range(num_laps):
        lap_idx = np.where(sess_dataframe['Lap'] == i)[0]
        speed_per_lap = sess_dataframe['Treadmill'][lap_idx]
        lap_positions = sess_dataframe['Position'][lap_idx] - sess_dataframe['Position'][lap_idx].min()
        bin_ix = np.digitize(lap_positions, bin_edges)
        for j in range(bins):
            speed_per_bin[i,j] = np.mean(speed_per_lap[bin_ix == j])
    speed_per_bin = speed_per_bin[:,1:]
    av_speed_per_bin = np.nanmean(speed_per_bin, axis=0)
    std_speed_per_bin = np.nanstd(speed_per_bin, axis=0)
    sem_speed_per_bin = std_speed_per_bin/np.sqrt(num_laps)

    return speed_per_bin

def calc_accel_per_lap_pre7(session, dt=1/45):
    actual_num_laps = np.round((len(session['all_lms']) // session['num_landmarks']))
    _, lm_exit_idx = get_lm_entry_exit(session)
    lap_change_idx = lm_exit_idx[session['num_landmarks']-1::session['num_landmarks']]

    x = 0
    if '3' in session['stage'] or '4' in session['stage']:
        bins = 15
        binned_lms = []
        binned_goals = []

        # acceleration per lm
        accel_per_bin = np.zeros((len(session['all_lms']), bins))
        for i, idx in enumerate(lm_exit_idx):
            lm_idx = np.arange(x, idx+1)

            speed_per_lm = session['speed'][lm_idx]
            pos_per_lm = session['position'][lm_idx]

            # acceleration (speed derivative / dt)
            accel = np.gradient(speed_per_lm) / dt
            
            accel_per_bin[i, :], bin_edges, _ = stats.binned_statistic(pos_per_lm, accel, bins=bins)
            
            # Bin goals and landmarks (as before)
            if session['goals_idx'][0] == i:
                binned_goals.append(np.digitize(session['goals'][0], bin_edges))
            if i <= 1:
                lm_bin = np.digitize(session['landmarks'][i], bin_edges)
                lm_bin_shifted = lm_bin + i * bins
                binned_lms.append(lm_bin_shifted)

            x = idx + 1

        # Split binned acceleration into goal and non-goal
        min_len = min(len([session['goals_idx']][0]), len([session['non_goals_idx']][0]))
        goal_accel = accel_per_bin[session['goals_idx'][:min_len], :]
        non_goal_accel = accel_per_bin[session['non_goals_idx'][:min_len], :]
        accel_per_bin = np.column_stack((goal_accel, non_goal_accel))

    else:
        bins = 120
        accel_per_bin = np.zeros((actual_num_laps, bins))

        for i, idx in enumerate(lap_change_idx):
            lap_idx = np.arange(x, idx+1)

            speed_per_lap = session['speed'][lap_idx]
            pos_per_lap = session['position'][lap_idx] 

            # acceleration
            accel = np.gradient(speed_per_lap) / dt
            accel_per_bin[i, :], bin_edges, _ = stats.binned_statistic(pos_per_lap, accel, bins=bins)
        
            x = idx + 1

    av_accel_per_bin = np.nanmean(accel_per_bin, axis=0)
    std_accel_per_bin = np.nanstd(accel_per_bin, axis=0)
    sem_accel_per_bin = std_accel_per_bin / np.sqrt(actual_num_laps)

    session['accel_per_bin'] = av_accel_per_bin
    session['sem_accel_per_bin'] = sem_accel_per_bin

    return session

def calc_decel_per_lap_pre7(session, dt=1/45):
    # TODO this doens't work as it should
    actual_num_laps = np.round((len(session['all_lms']) // session['num_landmarks']))
    _, lm_exit_idx = get_lm_entry_exit(session)
    lap_change_idx = lm_exit_idx[session['num_landmarks']-1::session['num_landmarks']]

    x = 0
    if '3' in session['stage'] or '4' in session['stage']:
        bins = 15
        binned_lms = []
        binned_goals = []

        # decel per lm
        decel_per_bin = np.zeros((len(session['all_lms']), bins))
        for i, idx in enumerate(lm_exit_idx):
            lm_idx = np.arange(x, idx+1)

            speed_per_lm = session['speed'][lm_idx]
            pos_per_lm = session['position'][lm_idx]

            # acceleration (speed derivative / dt)
            accel = np.gradient(speed_per_lm) / dt
            # keep only deceleration (negative accel)
            decel = np.where(accel < 0, accel, np.nan)

            decel_per_bin[i, :], bin_edges, _ = stats.binned_statistic(pos_per_lm, decel, bins=bins)
            
            # Bin goals and landmarks (as before)
            if session['goals_idx'][0] == i:
                binned_goals.append(np.digitize(session['goals'][0], bin_edges))
            if i <= 1:
                lm_bin = np.digitize(session['landmarks'][i], bin_edges)
                lm_bin_shifted = lm_bin + i * bins
                binned_lms.append(lm_bin_shifted)

            x = idx + 1

        # Split binned decel into goal and non-goal
        min_len = min(len([session['goals_idx']][0]), len([session['non_goals_idx']][0]))
        goal_decel = decel_per_bin[session['goals_idx'][:min_len], :]
        non_goal_decel = decel_per_bin[session['non_goals_idx'][:min_len], :]
        decel_per_bin = np.column_stack((goal_decel, non_goal_decel))

    else:
        bins = 120
        bin_edges = np.linspace(0, session['position'].max(), bins+1)
        decel_per_bin = np.zeros((actual_num_laps, bins))

        for i, idx in enumerate(lap_change_idx):
            lap_idx = np.arange(x, idx+1)

            speed_per_lap = session['speed'][lap_idx]
            pos_per_lap = session['position'][lap_idx] 

            # acceleration
            decel = np.gradient(speed_per_lap) / dt

            bin_ix = np.digitize(pos_per_lap, bin_edges)
            for j in range(bins):
                # Average speed per bin
                # Average deceleration per bin (negative values)
                decel_per_bin[i, j] = np.nanmean(decel[bin_ix == j])
        
            # decel = np.where(decel < 0, -decel, np.nan)

            # decel_per_bin[i, :], bin_edges, _ = stats.binned_statistic(pos_per_lap, decel, bins=bins)
        
            x = idx + 1

    av_decel_per_bin = np.nanmean(decel_per_bin, axis=0)
    std_decel_per_bin = np.nanstd(decel_per_bin, axis=0)
    sem_decel_per_bin = std_decel_per_bin / np.sqrt(actual_num_laps)

    session['decel_per_bin'] = av_decel_per_bin
    session['sem_decel_per_bin'] = sem_decel_per_bin

    return session

def calc_speed_per_lm(session):
    '''Calculate average speed per landmark'''
    actual_num_laps = np.round((len(session['all_lms']) // session['num_landmarks']) )

    _, lm_exit_idx = get_lm_entry_exit(session)
    lap_change_idx = lm_exit_idx[session['num_landmarks']-1::session['num_landmarks']]

    x = 0
    if '3' in session['stage'] or '4' in session['stage']:
        bins = 15
        binned_lms = []
        binned_goals = []

        # Calculate and bin speed per lm
        speed_per_bin = np.zeros((len(session['all_lms']), bins))
        for i, idx in enumerate(lm_exit_idx):
            lm_idx = np.arange(x, idx+1)

            speed_per_lm = session['speed'][lm_idx]
            pos_per_lm = session['position'][lm_idx]

            speed_per_bin[i, :], bin_edges, _ = stats.binned_statistic(pos_per_lm, speed_per_lm, bins=bins)
            
            # Bin goals and landmarks
            if session['goals_idx'][0] == i:
                binned_goals.append(np.digitize(session['goals'][0], bin_edges))
            if i <= 1:
                lm_bin = np.digitize(session['landmarks'][i], bin_edges)
                lm_bin_shifted = lm_bin + i * bins
                binned_lms.append(lm_bin_shifted)

            x = idx + 1

        # Split binned speed into goal and non-goal
        min_len = min(len([session['goals_idx']][0]), len([session['non_goals_idx']][0]))
        goal_speed = speed_per_bin[session['goals_idx'][:min_len], :]       # (min_len, bins)
        non_goal_speed = speed_per_bin[session['non_goals_idx'][:min_len], :]  # (min_len, bins)
        speed_per_bin = np.column_stack((goal_speed, non_goal_speed))    

    elif '5' in session['stage'] or '6' in session['stage']:
        bins = 120
        speed_per_bin = np.zeros((actual_num_laps, bins))

        for i, idx in enumerate(lap_change_idx):
            lap_idx = np.arange(x, idx+1)

            speed_per_lap = session['speed'][lap_idx]
            pos_per_lap = session['position'][lap_idx] 

            speed_per_bin[i, :], bin_edges, _ = stats.binned_statistic(pos_per_lap, speed_per_lap, bins=bins)
        
            # Only the last landmarks and goals will be used for binning
            goals_per_lap = session['goals'][i * 4 : (i + 1) * 4]
            lms_per_lap = session['landmarks'][i * session['num_landmarks'] : (i + 1) * session['num_landmarks']]
        
            x = idx + 1

        binned_goals = np.digitize(goals_per_lap, bin_edges)
        binned_lms = np.digitize(lms_per_lap, bin_edges)

    else:
        raise ValueError('This function only works for T3-T6 for now.')

    av_speed_per_bin = np.nanmean(speed_per_bin, axis=0)
    std_speed_per_bin = np.nanstd(speed_per_bin, axis=0)
    sem_speed_per_bin = std_speed_per_bin / np.sqrt(actual_num_laps)

    session['speed_per_bin'] = av_speed_per_bin
    session['sem_speed_per_bin'] = sem_speed_per_bin
    session['binned_goals'] = binned_goals
    session['binned_lms'] = binned_lms

    return session

def calc_sw_state_ratio(sess_dataframe, ses_settings):
    state_id = give_state_id(sess_dataframe,ses_settings)
    laps_needed = calc_laps_needed(ses_settings)
    num_laps, sess_dataframe = divide_laps(sess_dataframe, ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)
    if licked_all.shape[0] % 10 != 0:
        #extend the array to make it divisible by 10
        licked_all = np.pad(licked_all, (0, 10 - (licked_all.shape[0] % 10)), 'constant')
    licked_all_reshaped = licked_all.reshape(np.round(licked_all.shape[0] / 10).astype(int), 10)
    goals, lm_ids = parse_stable_goal_ids(ses_settings)

    if laps_needed == 2:
        window = 10
        state1_sw = np.zeros([num_laps,10])
        state2_sw = np.zeros([num_laps,10])
        state_diff_1 = np.zeros([num_laps,10])
        for i in range(num_laps):
            if i < window:
                state1_sw[i] = np.nan
                state2_sw[i] = np.nan
                state_diff_1[i] = np.nan

            else:
                lap_range = range(i-window, i)
                laps = licked_all_reshaped[lap_range]
                state1_laps = laps[np.where(state_id[lap_range] == 0)[0],:]
                state2_laps = laps[np.where(state_id[lap_range] == 1)[0],:]

                state1_sw[i] = safe_divide(np.sum(state1_laps,axis=0), state1_laps.shape[0])
                state2_sw[i] = safe_divide(np.sum(state2_laps,axis=0), state2_laps.shape[0])
                state_diff_1[i] = abs(state1_sw[i]-state2_sw[i])

        sw_state_ratio_a = state_diff_1[:,goals[0]]
        sw_state_ratio_b = state_diff_1[:,goals[1]]
        sw_state_ratio_c = state_diff_1[:,goals[2]]
        sw_state_ratio_d = state_diff_1[:,goals[3]]

    elif laps_needed == 3:
        window = 10
        state1_sw = np.zeros([num_laps,10])
        state2_sw = np.zeros([num_laps,10])
        state3_sw = np.zeros([num_laps,10])
        state_diff_1 = np.zeros([num_laps,10])
        state_diff_2 = np.zeros([num_laps,10])
        state_diff_3 = np.zeros([num_laps,10])
        for i in range(num_laps):
            if i < window:
                state1_sw[i] = np.nan
                state2_sw[i] = np.nan
                state3_sw[i] = np.nan
                state_diff_1[i] = np.nan
                state_diff_2[i] = np.nan
                state_diff_3[i] = np.nan

            else:
                lap_range = range(i-window, i)
                laps = licked_all_reshaped[lap_range]
                state1_laps = laps[np.where(state_id[lap_range] == 0)[0],:]
                state2_laps = laps[np.where(state_id[lap_range] == 1)[0],:]
                state3_laps = laps[np.where(state_id[lap_range] == 2)[0],:]

                state1_sw[i] = np.sum(state1_laps,axis=0)/state1_laps.shape[0]
                state2_sw[i] = np.sum(state2_laps,axis=0)/state2_laps.shape[0]
                state3_sw[i] = np.sum(state3_laps,axis=0)/state3_laps.shape[0]
                state_diff_1[i] = abs(state1_sw[i]-state2_sw[i])
                state_diff_2[i] = abs(state2_sw[i]-state3_sw[i])
                state_diff_3[i] = abs(state1_sw[i]-state3_sw[i])

        sw_state_ratio_a = (state_diff_1[:,goals[0]]+state_diff_3[:,goals[0]])/2
        sw_state_ratio_b = (state_diff_1[:,goals[1]]+state_diff_3[:,goals[1]])/2
        sw_state_ratio_c = (state_diff_1[:,goals[2]]+state_diff_2[:,goals[2]])/2
        sw_state_ratio_d = (state_diff_2[:,goals[3]]+state_diff_3[:,goals[3]])/2

    sw_state_ratio = np.nanmean([sw_state_ratio_a,sw_state_ratio_b,sw_state_ratio_c,sw_state_ratio_d],axis=0)

    return sw_state_ratio, sw_state_ratio_a, sw_state_ratio_b, sw_state_ratio_c, sw_state_ratio_d

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
    # if first_release != result[0][3]:
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

def threshold_lick_speed(sess_dataframe, speed_threshold=0.3):

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
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all,rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)

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

# def get_lap_idx(session):
#     # Divide the session dataframe into laps based on the position and corridor length
#     if 'world' in session:
#         if session['world'] == 'stable':
#             session['num_laps'] = int(np.ceil(session['position'].max() / session['tunnel_length']))
#         elif session['world'] == 'random':
#             session['num_laps'] = 1
#     else:
#         session['num_laps'] = int(np.ceil(session['position'].max() / session['tunnel_length']))

#     # For each position, determine which lap it belongs to
#     session['lap_idx'] = (session['position'] // session['tunnel_length']).astype(int)
#     print(session['num_laps'])
#     return session

def get_lm_idx(session):
    # Get landmark idx for each datapoint
    lm_idx = np.zeros(len(session['position']))
    for i in range(len(session['landmarks'])):
        lm = session['landmarks'][i]
        lm_entry = np.where((session['position'] > lm[0]) & (session['position'] < lm[1]))[0]
        lm_idx[lm_entry] = i+1

    session['lm_idx'] = lm_idx

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

def threshold_lick_events(sess_dataframe, ses_settings):

    session = create_session_struct(sess_dataframe, ses_settings)

    licks = sess_dataframe['Licks'].values.astype(int)
    speed_ok = session['speed'] < session['lick_threshold']
    licked = licks > 0
    threshold_mask = speed_ok & licked

    thresholded_licks = np.zeros(len(licks))
    thresholded_licks[threshold_mask] = licks[threshold_mask]

    return thresholded_licks

def get_licks_per_lap(session):
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

def get_licked_lms(session):
    # Get licked landmarks 

    if 'num_laps' in session:
        licked_lms = np.zeros((session['num_laps'], len(session['landmarks'])))
        
        for i in range(session['num_laps']):
            lap_idx = np.where(session['lap_idx'] == i)[0]
            target_ix = np.intersect1d(lap_idx, lm)
            for j in range(len(session['landmarks'])):
                lm = np.where(session['lm_idx'] == j+1)[0]
                target_ix = np.intersect1d(lap_idx, lm)
                target_licks = np.intersect1d(target_ix, session['licks_idx'])

                if len(target_licks) > 0:
                    licked_lms[i,j] = 1
                else:
                    licked_lms[i,j] = 0

    else:
        licked_lms = np.zeros(len(session['landmarks']))

        for i in range(len(session['landmarks'])):
            lm = np.where(session['lm_idx'] == i+1)[0]
            target_licks = np.intersect1d(lm, session['licks_idx'])
            
            if len(target_licks) > 0:
                licked_lms[i] = 1
            else:
                licked_lms[i] = 0

    session['licked_lms'] = licked_lms

    return session

def get_rewarded_lms(session):
    # Get rewarded landmarks

    if 'num_laps' in session:
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

    else:
        rewarded_lms = np.zeros((len(session['landmarks'])))

        for i in range(len(session['landmarks'])):
            lm = np.where(session['lm_idx'] == i+1)[0]
            target_rewards = np.intersect1d(lm, session['rewards'])
            if len(target_rewards) > 0:
                rewarded_lms[i] = 1
            else:
                rewarded_lms[i] = 0

    session['rewarded_lms'] = rewarded_lms

    return session

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

def get_random_lm_sequence(sess_dataframe, ses_settings):
    '''Create a list with lm ids for each lm in the random world'''
    odour_lm_id_mapping = create_odour_lm_mapping(ses_settings)

    _, _, _, _, release_df = get_event_parsed(sess_dataframe, ses_settings)

    release_ids = release_df['Odour'].to_numpy()
    lm_ids_list = np.empty(len(release_ids), dtype=int)
    for i, odour in enumerate(release_ids):
        matches = np.where(odour_lm_id_mapping == odour)[0]
        if len(matches) == 0:
            lm_ids_list[i] = np.nan
        else:
            lm_ids_list[i] = matches[0]

    return lm_ids_list

def get_lms_visited(session, sess_dataframe, ses_settings):
    # Calculate number of landmarks visited
    if len(np.where(session['landmarks'][:,0] < session['position'][-1])[0]) != len(np.where(session['landmarks'][:,-1] < session['position'][-1])[0]):
        num_lms = len(session['landmarks']) - 1 # session ended before mouse exited last lm entered
    else:
        num_lms = len(session['landmarks'])  

    all_lms = np.array([]) # landmark ids
    if 'world' in session:
        if session['world'] == 'stable':
            for i in range(session['num_laps']):
                all_lms = np.append(all_lms, session['lm_ids'])
            all_lms = all_lms.astype(int)[:num_lms]
        elif session['world'] == 'random':
            all_lms = get_random_lm_sequence(sess_dataframe, ses_settings)
            all_lms = all_lms[:num_lms]
    
    all_landmarks = session['landmarks'][:num_lms]  # landmark positions
    
    session['all_landmarks'] = all_landmarks
    session['all_lms'] = all_lms

    return session

def calc_acceleration(session, funcimg_frame_rate=45):
    # get acceleration 
    acceleration = gaussian_filter1d(np.gradient(session['speed'], 1/funcimg_frame_rate).reshape(1, -1), sigma=10)

    session['acceleration'] = acceleration[0]

    return session

def calc_goal_progress(session, bins=5):
    '''Create a goal progress vector'''
    
    binned_goal_progress = np.zeros(len(session['position'])) 

    if session['test_landmark_id'] is not None: 
        reward_ix = np.sort(np.concatenate([session['reward_idx'], session['test_rew_idx']])).astype(int)
    else:
        reward_ix = session['reward_idx']

    # first trial 
    phase_frames = np.arange(0, reward_ix[0])
    bin_edges = np.linspace(0, reward_ix[0], bins+1)
    bin_ix = np.digitize(phase_frames, bin_edges)
    binned_goal_progress[phase_frames] = bin_ix

    # intermediate trials
    for i in range(len(reward_ix)-1):
        phase_frames = np.arange(reward_ix[i], reward_ix[i+1])
        bin_edges = np.linspace(reward_ix[i], reward_ix[i+1], bins+1)
        bin_ix = np.digitize(phase_frames, bin_edges)
        binned_goal_progress[phase_frames] = bin_ix

    # last trial 
    phase_frames = np.arange(reward_ix[-1], len(session['position']))
    bin_edges = np.linspace(reward_ix[-1], len(session['position']), bins+1)
    bin_ix = np.digitize(phase_frames, bin_edges)
    binned_goal_progress[phase_frames] = bin_ix
    
    return binned_goal_progress

def get_active_goal(session):
    # get goal indices
    goal_idx = np.array([])
    if session['num_laps'] > 1:
        for goal in session['goal_ids']:
            goal_idx = np.append(goal_idx, np.where(session['all_lms'] == goal)[0][0])
    else:
        for goal in session['goal_ids']:
            matches = np.where(session['all_lms'][:session['num_landmarks']] == goal)[0]
            if matches.size > 0:
                goal_idx = np.append(goal_idx, matches[0])

    session['goal_idx'] = goal_idx.astype(int)

    # get active goal
    active_goal = np.zeros((session['num_laps'], len(session['all_lms'])))
    count = 0
    active_goal[0,0] = goal_idx[count]

    for i in range(session['num_laps']):
        for j in range(len(session['all_lms'])):
            active_goal[i,j] = goal_idx[count]
            if session['rewarded_lms'][i,j]==1:
                count += 1
                if count == len(goal_idx):
                    count = 0

    session['active_goal'] = active_goal

    return session

def get_data_lm_idx(session):
    """Get the landmark id of every data entry"""
    
    # Find landmark entry and exit idx
    lm_entry, lm_exit = get_lm_entry_exit(session)

    # Find datapoints within a landmark
    lm_idx = np.zeros(len(session['position']))
    for i in range(len(session['all_landmarks'])):
        lm_idx[lm_entry[i]:lm_exit[i]+1] = i+1

    session['data_lm_idx'] = lm_idx

    return session 

def get_binary_lick_map(session):
    """Create a binary map of licked landmarks"""
    
    # Get all datapoints within landmarks
    session = get_data_lm_idx(session)

    licked_lms = np.empty((len(session['all_lms'])))
    for lm in range(len(session['all_lms'])):
        # datapoints within landmarks for each lap 
        lm_idx = np.where(session['data_lm_idx'] == lm+1)[0]

        # Find all licks within the landmark
        target_licks = np.intersect1d(lm_idx, session['licks_idx'])
        if len(target_licks) > 0:
            licked_lms[lm] = 1
        else:
            licked_lms[lm] = 0

    # Check number of actual laps
    num_lms_considered = int(np.round((len(session['all_landmarks']) // session['num_landmarks']) * session['num_landmarks']))
    num_laps = int(num_lms_considered / session['num_landmarks'])

    # Reshape the data
    if '3' in session['stage'] or '4' in session['stage']:
        # The landmarks might not be in order so we need to be careful about binning 
        # Determine how many rows to keep
        min_len = min(len(session['goals_idx']), len(session['non_goals_idx']))
        goal_licked_lms = licked_lms[session['goals_idx'][:min_len]]
        non_goal_licked_lms = licked_lms[session['non_goals_idx'][:min_len]]

        goal_licked_lms = goal_licked_lms.reshape((num_laps, -1))        # -1 lets numpy figure out columns
        non_goal_licked_lms = non_goal_licked_lms.reshape((num_laps, -1))

        binary_licked_lms = np.column_stack((goal_licked_lms, non_goal_licked_lms))
    else: 
        # The landmarks are in order so we can simply reshape
        binary_licked_lms = np.array(licked_lms[:num_lms_considered]).reshape((num_laps, session['num_landmarks']))
    
    session['binary_licked_lms'] = binary_licked_lms

    return session

def get_lm_lick_rate(session, bins=16):  # TODO I really need to fix this and make it consistent across sessions
    """Get lick rate per frame bin as the mean per bin for each landmark"""
    
    # Get all datapoints within landmarks
    session = get_data_lm_idx(session)

    # Create a binary lick map for the entire session 
    binary_licks = np.zeros(len(session['position'])) # (actually not binary)
    if 'thresholded_licks' in session:
        binary_licks = session['thresholded_licks'] 
    else:
        binary_licks = session['licks'] 

    if bins is not None:
        if ('stage' in session) and ('3' in session['stage'] or '4' in session['stage']):
            
            lm_lick_rate = np.zeros((len(session['all_lms']), bins))
            for lm in range(len(session['all_lms'])):
                # datapoints within landmarks for each lap 
                lm_idx = np.where(session['data_lm_idx'] == lm+1)[0]

                # binary licks within landmark
                lm_licks = binary_licks[lm_idx[0]:lm_idx[-1]+1]
                
                # calculate lick rate within each landmark (mean in each bin)
                lm_lick_rate[lm], _, _ = stats.binned_statistic(lm_idx, lm_licks, bins=bins)

                # Reshape the data in goal - non-goal pairs
                goal_lm_lick_rate = lm_lick_rate[session['goals_idx']]
                non_goal_lm_lick_rate = lm_lick_rate[session['non_goals_idx']]

                min_len = min(len(goal_lm_lick_rate), len(non_goal_lm_lick_rate))
                goal_lm_lick_rate = goal_lm_lick_rate[:min_len, :]
                non_goal_lm_lick_rate = non_goal_lm_lick_rate[:min_len, :]

                lm_lick_rate = np.column_stack((goal_lm_lick_rate, non_goal_lm_lick_rate))
                
        else:
            lm_lick_rate = np.zeros((len(session['all_lms']), bins))
            for lm in range(len(session['all_lms'])):
                # datapoints within landmarks for each lap 
                lm_idx = np.where(session['data_lm_idx'] == lm+1)[0]

                # binary licks within landmark
                lm_licks = binary_licks[lm_idx[0]:lm_idx[-1]+1]
                
                # calculate lick rate within each landmark (mean in each bin)
                lm_lick_rate[lm], _, _ = stats.binned_statistic(lm_idx, lm_licks, bins=bins)

            # Calculate actual number of laps and lms
            num_lms_considered = int(np.round((len(session['all_landmarks']) // session['num_landmarks']) * session['num_landmarks']))
            num_laps = int(num_lms_considered / session['num_landmarks'])

            # Reshape the data 
            lm_lick_rate_reshape = [[] for _ in range(num_laps)]
            for lap in range(num_laps):
                start = lap * session['num_landmarks']
                end = start + session['num_landmarks']
                for lm_idx in range(start, end):
                    rate = lm_lick_rate[lm_idx]
                    lm_lick_rate_reshape[lap].extend(rate)                
            lm_lick_rate = np.array(lm_lick_rate_reshape)  # (num_laps, num_bins * num_landmarks)

    else:
        # Get a single value per landmark 
        lm_lick_rate = np.zeros((len(session['all_landmarks'])))
        for lm in range(len(session['all_landmarks'])):
            # datapoints within landmarks for each lap 
            lm_idx = np.where(session['data_lm_idx'] == lm+1)[0]
            
            if len(lm_idx) == 0:
                lm_lick_rate[lm] = 0
                continue

            # calculate mean lick rate within each landmark
            lm_lick_rate[lm] = np.mean(binary_licks[lm_idx])

        # Alternative 
        # lm_lick_rate = np.zeros(len(session['landmarks']))
        # for lm in range(len(session['landmarks'])):
        #     lm = np.where(session['lm_idx'] == lm+1)[0]
        #     target_licks = np.intersect1d(lm, session['licks_idx'])
        #     lm_lick_rate[lm] = len(target_licks) / len(lm)
        
    session['lm_lick_rate'] = lm_lick_rate

    return session

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

def get_goal_positions(session, sess_dataframe, ses_settings):
    '''Get the start and end of each goal landmark using odour release events to find targets'''
    target_positions, _, _, _, _, _ = find_targets_distractors(sess_dataframe, ses_settings)

    goals = np.zeros((len(target_positions), 2))
    for i, pos in enumerate(np.sort(target_positions)):
        # Find lm closest to release position
        closest_lm = np.argmin(np.abs(session['all_landmarks'][:,0] - pos))
        goals[i] = session['all_landmarks'][closest_lm]
    
    session['goals'] = goals

    return session

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

def get_landmark_category_rew_idx(session):
    '''Find indices also in non-goal landmarks corresponding to the same time after landmark entry as mean reward time lag.'''

    rew_lm_entry_idx, miss_lm_entry_idx, nongoal_lm_entry_idx, test_lm_entry_idx = get_landmark_category_entries(session)
    
    # Calculate time lag between landmark entry and reward delivery
    rew_time_lag = np.round(np.mean(session['reward_idx'] - rew_lm_entry_idx))
    print('Reward time lag from lm entry: ', rew_time_lag)

    # Find where reward would be on average if these landmarks were rewarded
    miss_rew_idx = miss_lm_entry_idx + rew_time_lag
    nongoal_rew_idx = nongoal_lm_entry_idx + rew_time_lag  
    test_rew_idx = test_lm_entry_idx + rew_time_lag

    session['rew_time_lag'] = rew_time_lag
    session['miss_rew_idx'] = miss_rew_idx
    session['nongoal_rew_idx'] = nongoal_rew_idx
    session['test_rew_idx'] = test_rew_idx

    return session

def get_landmark_category_entries(session):
    '''Find the indices of landmark entry for different types of landmarks: rewarded, miss, non-goal, test.'''
    
    lm_entry_idx, _ = get_lm_entry_exit(session)

    # Find category for each landmark 
    session = get_landmark_categories(session)

    # Find the rewarded landmarks 
    session = get_rewarded_landmarks(session)

    # Find landmark entry indices for each landmark category
    rew_lm_entry_idx = [lm_entry_idx[i] for i in session['rewarded_landmarks']]
    miss_lm_entry_idx = np.array([lm_entry_idx[i] for i in session['goals_idx'] if i not in session['rewarded_landmarks']])
    nongoal_lm_entry_idx = np.array([lm_entry_idx[i] for i in session['non_goals_idx']])
    test_lm_entry_idx = np.array([lm_entry_idx[i] for i in session['test_idx']]) if session['test_idx'] is not None else np.array([])

    assert len(rew_lm_entry_idx) + len(miss_lm_entry_idx) + len(nongoal_lm_entry_idx) + len(test_lm_entry_idx) == len(session['all_lms']), 'Some landmarks have not been considered.'

    return rew_lm_entry_idx, miss_lm_entry_idx, nongoal_lm_entry_idx, test_lm_entry_idx

def get_landmark_categories(session):
    '''Find the landmarks in the entire session that belong to goals, non-goals and test.'''

    session = get_landmark_ids(session)

    # Get the landmarks that belong to each condition  
    goals_idx = np.where(np.isin(session['all_lms'], session['goal_landmark_id']))[0]
    non_goals_idx = np.where(np.isin(session['all_lms'], session['non_goal_landmark_id']))[0]
    test_idx = np.where(np.isin(session['all_lms'], session['test_landmark_id']))[0] if session['test_landmark_id'] is not None else None
    
    session['goals_idx'] = goals_idx
    session['non_goals_idx'] = non_goals_idx
    session['test_idx'] = test_idx

    return session

def get_landmark_ids(session):
    '''Define which landmarks belong to goals, non-goals and test.'''
 
    t = extract_int(session['stage'])

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

def get_rewarded_landmarks(session):
    '''Find the indices of rewarded (lick-triggered) landmarks.'''

    lm_entry_idx, lm_exit_idx = get_lm_entry_exit(session)

    # Find rewarded landmarks 
    reward_positions = session['position'][session['reward_idx']]

    rewarded_landmarks = [i for i, (start, end) in enumerate(zip(np.floor(session['position'][lm_entry_idx]), np.ceil(session['position'][lm_exit_idx]))) 
                            if np.any((np.ceil(reward_positions) >= start) & (np.floor(reward_positions) <= end))] 

    session['rewarded_landmarks'] = rewarded_landmarks

    return session

def get_AB_sequence(session, world='stable'):
    if world == 'stable':
        sequence = 'ABAB'
    elif world == 'random':
        sequence = 'AB_shuffled'
    else:
        raise ValueError("Oops I don't know what to do about this type of world")
    
    session['sequence'] = sequence

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
def create_session_struct_npz(data, ses_settings, world):

    position = np.nan_to_num(data['position'], nan=0.0)
    rewards = np.where(data['rewards'])[0]
    speed = np.nan_to_num(data['speed'], nan=0.0)
    licks = data['licks'] # TODO licks has a different definition for cohort 2
    pd1 = data['pd1']
    pd2 = data['pd2']

    if world == 'stable':
        goal_ids, lm_ids = parse_stable_goal_ids(ses_settings)
    elif world == 'random':
        goal_ids, lm_ids = parse_random_goal_ids(ses_settings)
    num_landmarks = len(lm_ids) # unique number of lm ids

    tunnel_length = calculate_corr_length(ses_settings)
    lick_threshold = ses_settings['velocityThreshold']
    # transform bonsai speed to that computed using analog position
    bonsai_speed_factor = 0.10 # TODO compute here 
    lick_threshold = lick_threshold / bonsai_speed_factor

    session = {'position': position,
               'licks': licks, 
               'rewards': rewards, 
               'pd1': pd1,
               'pd2': pd2,
               'goal_ids': goal_ids, 
               'lm_ids': lm_ids,
               'num_landmarks': num_landmarks,
               'tunnel_length': tunnel_length,
               'lick_threshold': lick_threshold,
               'speed': speed}
    
    return session

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
    # lm_size = ses_settings['trial']['landmarks'][0][0]['size'] # assume same size for all lms

    session = {'position': position,
               'speed': speed,
               'licks': licks, 
               'rewards': rewards,
               'lick_threshold': lick_threshold,
               'lm_size': lm_size
               }
    
    return session

def get_behaviour(session, sess_dataframe, ses_settings, plot=True):
    transition_prob, control_prob, ideal_prob = calc_stable_conditional_matrix(sess_dataframe, ses_settings)
    laps_needed = calc_laps_needed(ses_settings)
    state_id = give_state_id(sess_dataframe, ses_settings)

    if plot:
        if int(session['stage'][-1]) > 6:
            print('Plotting the lick and speed profile for the 2 and 3 lap sequences.')
            plot_licks_per_state(sess_dataframe, ses_settings)
            plot_speed_per_state(sess_dataframe, ses_settings)

        if session['world'] == 'stable':
            _ = plot_lick_maps(session)
            _ = plot_speed_profile(session)
    
    session['transition_prob'] = transition_prob
    session['control_prob'] = control_prob
    session['ideal_prob'] = ideal_prob
    session['laps_needed'] = laps_needed
    session['state_id'] = state_id

    return session

def analyse_npz_pre7(mouse, session_id, root, stage, world='stable', plot=True):
    '''Wrapper for session analysis'''

    if '3' not in stage and '4' not in stage and '5' not in stage and '6' not in stage:
        raise ValueError('This function only works for T3-T6.')
    
    session_path = parse_nwb_functions.find_base_path(mouse, session_id, root)
    ses_settings, _ = parse_nwb_functions.load_settings(session_path)
    behav_data = load_session_npz(str(session_path))
    with open(list((session_path / 'behav').glob('*.pkl'))[0], 'rb') as f:
        sess_dataframe = pickle.load(f)

    session = create_session_struct_npz(behav_data, ses_settings, world=world)
    session = get_landmark_positions(session, sess_dataframe, ses_settings, data='pd')
    session = get_goal_positions(session, sess_dataframe, ses_settings)

    session['mouse'] = mouse
    session['session_id'] = session_id
    # session['date'] = date
    session['stage'] = stage
    session['world'] = world

    save_path = Path(session_path) / 'analysis'
    save_path.mkdir(parents=True, exist_ok=True)
    session['save_path'] = save_path
    
    session = get_lap_idx(session)
    session = get_lm_idx(session)
    session = get_licks_idx(session) # thresholding is also performed here
    session = get_licks_per_lap(session)
    session = get_licked_lms(session)
    session = get_rewarded_lms(session)
    session = get_lms_visited(session, sess_dataframe, ses_settings)
    session = get_reward_idx(session)
    session = get_active_goal(session)
    session = calc_acceleration(session)
    session = calculate_frame_lick_rate(session)

    session = get_AB_sequence(session, world)
    session = get_landmark_categories(session)
    # session = get_licks(data, session)
    session = get_rewarded_landmarks(session)
    session = get_landmark_category_rew_idx(session)

    # Get behaviour
    session = get_behaviour(session, sess_dataframe, ses_settings, plot)

    print('Number of laps = ', session['num_laps'])
    
    return session

def analyse_session_behav(session_path, mouse, plot=True):
    '''Wrapper for session analysis using behaviour data'''

    ses_settings, _ = load_settings(session_path)
    sess_dataframe = load_data(session_path)

    session = create_session_struct(sess_dataframe, ses_settings)
    session = get_landmark_positions(session, sess_dataframe, ses_settings, data='odour')
    session = get_lms_visited(session, sess_dataframe, ses_settings)
    session = get_goal_positions(session, sess_dataframe, ses_settings)

    session['mouse'] = mouse

    # save_path = Path(session_path) / 'analysis'
    # save_path.mkdir(parents=True, exist_ok=True)
    # session['save_path'] = save_path
    
    # session = get_lap_idx(session)
    session = get_lm_idx(session)
    session = get_licks_idx(session) # thresholding is also performed here
    # session = get_licks_per_lap(session)
    session = get_licked_lms(session)
    session = get_rewarded_lms(session)
    
    session = get_reward_idx(session)
    session = get_lm_lick_rate(session, bins=None)
    # session = get_active_goal(session)
    # session = calc_acceleration(session)
    # session = calculate_frame_lick_rate(session)

    # session = get_AB_sequence(session, world)
    # session = get_landmark_categories(session)
    # session = get_rewarded_landmarks(session)
    # session = get_landmark_category_rew_idx(session)

    # Get behaviour
    # session = get_behaviour(session, sess_dataframe, ses_settings, plot)

    # print('Number of laps = ', session['num_laps'])
    
    return session
    
#%% ##### Plotting #####
def plot_ethogram(sess_dataframe,ses_settings):
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

        event_pos = position[np.where(release_df['Index'] == lm_idx)[0][0]]
        lm_exit_idx = np.argmin(np.abs(position - (event_pos + session['lm_size'])))

    
    mean_binned_speed = np.mean(binned_speed, axis=0)
    sem_binned_speed = stats.sem(binned_speed, axis=0)

    return mean_binned_speed, sem_binned_speed

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

def plot_speed_lick_rate_psth(ses_settings, sess_dataframe, session_id, bins=300):
    with mpl.rc_context({
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 15,
    }):

        sequence_task = True
        if 'full' in session_id:
            sequence_task = False

        # Estimate dt 
        if 'LM_Count' in sess_dataframe.columns:
            release_df = estimate_lm_events(sess_dataframe)
        else:
            release_df = estimate_release_events(sess_dataframe, ses_settings)

        dt_idx = np.diff(release_df['Index'])
        dt_seconds = release_df.index.to_series().diff().dt.total_seconds().to_numpy()
        window_seconds = np.round(dt_seconds[1:] / dt_idx * bins, 1)
        window_seconds = window_seconds[~np.isnan(window_seconds)][0]

        # Plot data based on session id 
        if not sequence_task:
            # Shaping (10LM corridor)
            fig, axes = plt.subplots(1, 2, figsize=(10,4))
            axes = axes.ravel()

            mean_binned_speed, sem_binned_speed = get_speed_psth(ses_settings, sess_dataframe, bins=bins)
            mean_binned_licks, sem_binned_licks = get_lick_rate_psth(ses_settings, sess_dataframe, bins=bins)

            axes[0].plot(mean_binned_speed, color='black')
            axes[0].fill_between(range(len(mean_binned_speed)), 
                            mean_binned_speed + sem_binned_speed, 
                            mean_binned_speed - sem_binned_speed,
                            color='black', alpha=0.3)
            axes[0].set_title('Speed')

            axes[1].plot(mean_binned_licks, color='black')
            axes[1].fill_between(range(len(mean_binned_licks)), 
                            mean_binned_licks + sem_binned_licks, 
                            mean_binned_licks - sem_binned_licks,
                            color='black', alpha=0.3)
            axes[1].set_title('Lick rate')

            bins = mean_binned_speed.shape[0]
            for ax in axes:
                ax.axvspan(bins/2, bins, color='grey', alpha=0.3)
                ax.set_xticks([0, bins/2, bins], labels=[f'{-window_seconds/2:.1f}', 0, f'{window_seconds/2:.1f}'])
                # ax.set_xticks([bins/2, bins], labels=['lm entry', 'lm exit'])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        else:
            # Sequence training (ABAB or AABB)
            A_landmarks, B_landmarks, A_idx, B_idx = get_A_B_landmarks(sess_dataframe, ses_settings)

            if 'aabb' in session_id:
                A1 = A_idx[::2]
                A2 = A_idx[1::2]
                B1 = B_idx[::2]
                B2 = B_idx[1::2]

                fig, axes = plt.subplots(1, 2, figsize=(10,4))
                axes = axes.ravel()

                mean_binned_speed_A1, sem_binned_speed_A1 = get_speed_psth(ses_settings, sess_dataframe, events=A1, bins=bins)
                mean_binned_licks_A1, sem_binned_licks_A1 = get_lick_rate_psth(ses_settings, sess_dataframe, events=A1, bins=bins)

                mean_binned_speed_A2, sem_binned_speed_A2 = get_speed_psth(ses_settings, sess_dataframe, events=A2, bins=bins)
                mean_binned_licks_A2, sem_binned_licks_A2 = get_lick_rate_psth(ses_settings, sess_dataframe, events=A2, bins=bins)

                mean_binned_speed_B1, sem_binned_speed_B1 = get_speed_psth(ses_settings, sess_dataframe, events=B1, bins=bins)
                mean_binned_licks_B1, sem_binned_licks_B1 = get_lick_rate_psth(ses_settings, sess_dataframe, events=B1, bins=bins)

                mean_binned_speed_B2, sem_binned_speed_B2 = get_speed_psth(ses_settings, sess_dataframe, events=B2, bins=bins)
                mean_binned_licks_B2, sem_binned_licks_B2 = get_lick_rate_psth(ses_settings, sess_dataframe, events=B2, bins=bins)

                axes[0].axhline(ses_settings['velocityThreshold'], linestyle='--', color='grey')
                axes[0].plot(mean_binned_speed_A1, color='darkblue', label='A1')
                axes[0].fill_between(range(len(mean_binned_speed_A1)), 
                                mean_binned_speed_A1 + sem_binned_speed_A1, 
                                mean_binned_speed_A1 - sem_binned_speed_A1,
                                color='darkblue', alpha=0.3)
                axes[0].plot(mean_binned_speed_A2, color='blue', label='A2')
                axes[0].fill_between(range(len(mean_binned_speed_A2)), 
                                mean_binned_speed_A2 + sem_binned_speed_A2, 
                                mean_binned_speed_A2 - sem_binned_speed_A2,
                                color='blue', alpha=0.3)
                axes[0].plot(mean_binned_speed_B1, color='orange', label='B1')
                axes[0].fill_between(range(len(mean_binned_speed_B1)), 
                                mean_binned_speed_B1 + sem_binned_speed_B1, 
                                mean_binned_speed_B1 - sem_binned_speed_B1,
                                color='orange', alpha=0.3)
                axes[0].plot(mean_binned_speed_B2, color='gold', label='B2')
                axes[0].fill_between(range(len(mean_binned_speed_B2)), 
                                mean_binned_speed_B2 + sem_binned_speed_B2, 
                                mean_binned_speed_B2 - sem_binned_speed_B2,
                                color='gold', alpha=0.3)
                axes[0].set_title('Speed')

                axes[1].plot(mean_binned_licks_A1, color='darkblue', label='A1')
                axes[1].fill_between(range(len(mean_binned_licks_A1)), 
                                mean_binned_licks_A1 + sem_binned_licks_A1, 
                                mean_binned_licks_A1 - sem_binned_licks_A1,
                                color='darkblue', alpha=0.3)
                axes[1].plot(mean_binned_licks_A2, color='blue', label='A2')
                axes[1].fill_between(range(len(mean_binned_licks_A2)), 
                                mean_binned_licks_B1 + sem_binned_licks_A2, 
                                mean_binned_licks_A2 - sem_binned_licks_A2,
                                color='blue', alpha=0.3)
                axes[1].plot(mean_binned_licks_B1, color='orange', label='B1')
                axes[1].fill_between(range(len(mean_binned_licks_B1)), 
                                mean_binned_licks_B1 + sem_binned_licks_B1, 
                                mean_binned_licks_B1 - sem_binned_licks_B1,
                                color='orange', alpha=0.3)
                axes[1].plot(mean_binned_licks_B2, color='gold', label='B2')
                axes[1].fill_between(range(len(mean_binned_licks_B2)), 
                                mean_binned_licks_B2 + sem_binned_licks_B2, 
                                mean_binned_licks_B2 - sem_binned_licks_B2,
                                color='gold', alpha=0.3)
                axes[1].set_title('Lick rate')

                bins = mean_binned_speed_A1.shape[0]
                for ax in axes:
                    ax.axvspan(bins/2, bins, color='grey', alpha=0.3)
                    ax.set_xticks([0, bins/2, bins], labels=[f'{-window_seconds/2:.1f}', 0, f'{window_seconds/2:.1f}'])
                    # ax.set_xticks([bins/2, bins], labels=['lm entry', 'lm exit'])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.legend()

            elif 'abab' in session_id:
                fig, axes = plt.subplots(1, 2, figsize=(10,4))
                axes = axes.ravel()

                mean_binned_speed_A, sem_binned_speed_A = get_speed_psth(ses_settings, sess_dataframe, events=A_idx, bins=bins)
                mean_binned_licks_A, sem_binned_licks_A = get_lick_rate_psth(ses_settings, sess_dataframe, events=A_idx, bins=bins)

                mean_binned_speed_B, sem_binned_speed_B = get_speed_psth(ses_settings, sess_dataframe, events=B_idx, bins=bins)
                mean_binned_licks_B, sem_binned_licks_B = get_lick_rate_psth(ses_settings, sess_dataframe, events=B_idx, bins=bins)

                axes[0].axhline(ses_settings['velocityThreshold'], linestyle='--', color='grey')
                axes[0].plot(mean_binned_speed_A, color='darkblue', label='A')
                axes[0].fill_between(range(len(mean_binned_speed_A)), 
                                mean_binned_speed_A + sem_binned_speed_A, 
                                mean_binned_speed_A - sem_binned_speed_A,
                                color='darkblue', alpha=0.3)
                axes[0].plot(mean_binned_speed_B, color='orange', label='B')
                axes[0].fill_between(range(len(mean_binned_speed_B)), 
                                mean_binned_speed_B + sem_binned_speed_B, 
                                mean_binned_speed_B - sem_binned_speed_B,
                                color='orange', alpha=0.3)
                axes[0].set_title('Speed')

                axes[1].plot(mean_binned_licks_A, color='darkblue', label='A')
                axes[1].fill_between(range(len(mean_binned_licks_A)), 
                                mean_binned_licks_A + sem_binned_licks_A, 
                                mean_binned_licks_A - sem_binned_licks_A,
                                color='darkblue', alpha=0.3)
                axes[1].plot(mean_binned_licks_B, color='orange', label='B')
                axes[1].fill_between(range(len(mean_binned_licks_B)), 
                                mean_binned_licks_B + sem_binned_licks_B, 
                                mean_binned_licks_B - sem_binned_licks_B,
                                color='orange', alpha=0.3)
                axes[1].set_title('Lick rate')

                bins = mean_binned_speed_A.shape[0]   
                for ax in axes:
                    ax.axvspan(bins/2, bins, color='grey', alpha=0.3)
                    ax.set_xticks([0, bins/2, bins], labels=[f'{-window_seconds/2:.1f}', 0, f'{window_seconds/2:.1f}'])
                    # ax.set_xticks([bins/2, bins], labels=['lm entry', 'lm exit'])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.legend()
    
            elif 'abb' in session_id and 'abbb' not in session_id:                
                A1 = A_idx
                B1 = B_idx[::2]
                B2 = B_idx[1::2]

                fig, axes = plt.subplots(1, 2, figsize=(10,4))
                axes = axes.ravel()

                mean_binned_speed_A1, sem_binned_speed_A1 = get_speed_psth(ses_settings, sess_dataframe, events=A1, bins=bins)
                mean_binned_licks_A1, sem_binned_licks_A1 = get_lick_rate_psth(ses_settings, sess_dataframe, events=A1, bins=bins)

                mean_binned_speed_B1, sem_binned_speed_B1 = get_speed_psth(ses_settings, sess_dataframe, events=B1, bins=bins)
                mean_binned_licks_B1, sem_binned_licks_B1 = get_lick_rate_psth(ses_settings, sess_dataframe, events=B1, bins=bins)

                mean_binned_speed_B2, sem_binned_speed_B2 = get_speed_psth(ses_settings, sess_dataframe, events=B2, bins=bins)
                mean_binned_licks_B2, sem_binned_licks_B2 = get_lick_rate_psth(ses_settings, sess_dataframe, events=B2, bins=bins)

                axes[0].axhline(ses_settings['velocityThreshold'], linestyle='--', color='grey')
                axes[0].plot(mean_binned_speed_A1, color='darkblue', label='A')
                axes[0].fill_between(range(len(mean_binned_speed_A1)), 
                                mean_binned_speed_A1 + sem_binned_speed_A1, 
                                mean_binned_speed_A1 - sem_binned_speed_A1,
                                color='darkblue', alpha=0.3)
                axes[0].plot(mean_binned_speed_B1, color='orange', label='B1')
                axes[0].fill_between(range(len(mean_binned_speed_B1)), 
                                mean_binned_speed_B1 + sem_binned_speed_B1, 
                                mean_binned_speed_B1 - sem_binned_speed_B1,
                                color='orange', alpha=0.3)
                axes[0].plot(mean_binned_speed_B2, color='gold', label='B2')
                axes[0].fill_between(range(len(mean_binned_speed_B2)), 
                                mean_binned_speed_B2 + sem_binned_speed_B2, 
                                mean_binned_speed_B2 - sem_binned_speed_B2,
                                color='gold', alpha=0.3)
                axes[0].set_title('Speed')

                axes[1].plot(mean_binned_licks_A1, color='darkblue', label='A')
                axes[1].fill_between(range(len(mean_binned_licks_A1)), 
                                mean_binned_licks_A1 + sem_binned_licks_A1, 
                                mean_binned_licks_A1 - sem_binned_licks_A1,
                                color='darkblue', alpha=0.3)
                axes[1].plot(mean_binned_licks_B1, color='orange', label='B1')
                axes[1].fill_between(range(len(mean_binned_licks_B1)), 
                                mean_binned_licks_B1 + sem_binned_licks_B1, 
                                mean_binned_licks_B1 - sem_binned_licks_B1,
                                color='orange', alpha=0.3)
                axes[1].plot(mean_binned_licks_B2, color='gold', label='B2')
                axes[1].fill_between(range(len(mean_binned_licks_B2)), 
                                mean_binned_licks_B2 + sem_binned_licks_B2, 
                                mean_binned_licks_B2 - sem_binned_licks_B2,
                                color='gold', alpha=0.3)
                axes[1].set_title('Lick rate')

                bins = mean_binned_speed_A1.shape[0]
                for ax in axes:
                    ax.axvspan(bins/2, bins, color='grey', alpha=0.3)
                    ax.set_xticks([0, bins/2, bins], labels=[f'{-window_seconds/2:.1f}', 0, f'{window_seconds/2:.1f}'])
                    # ax.set_xticks([bins/2, bins], labels=['lm entry', 'lm exit'])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.legend()

            elif 'abbb' in session_id:                
                A1 = A_idx
                B1 = B_idx[::3]
                B2 = B_idx[1::3]
                B3 = B_idx[2::3]

                fig, axes = plt.subplots(1, 2, figsize=(10,4))
                axes = axes.ravel()

                mean_binned_speed_A1, sem_binned_speed_A1 = get_speed_psth(ses_settings, sess_dataframe, events=A1, bins=bins)
                mean_binned_licks_A1, sem_binned_licks_A1 = get_lick_rate_psth(ses_settings, sess_dataframe, events=A1, bins=bins)

                mean_binned_speed_B1, sem_binned_speed_B1 = get_speed_psth(ses_settings, sess_dataframe, events=B1, bins=bins)
                mean_binned_licks_B1, sem_binned_licks_B1 = get_lick_rate_psth(ses_settings, sess_dataframe, events=B1, bins=bins)

                mean_binned_speed_B2, sem_binned_speed_B2 = get_speed_psth(ses_settings, sess_dataframe, events=B2, bins=bins)
                mean_binned_licks_B2, sem_binned_licks_B2 = get_lick_rate_psth(ses_settings, sess_dataframe, events=B2, bins=bins)

                mean_binned_speed_B3, sem_binned_speed_B3 = get_speed_psth(ses_settings, sess_dataframe, events=B3, bins=bins)
                mean_binned_licks_B3, sem_binned_licks_B3 = get_lick_rate_psth(ses_settings, sess_dataframe, events=B3, bins=bins)

                axes[0].axhline(ses_settings['velocityThreshold'], linestyle='--', color='grey')
                axes[0].plot(mean_binned_speed_A1, color='darkblue', label='A')
                axes[0].fill_between(range(len(mean_binned_speed_A1)), 
                                mean_binned_speed_A1 + sem_binned_speed_A1, 
                                mean_binned_speed_A1 - sem_binned_speed_A1,
                                color='darkblue', alpha=0.3)
                axes[0].plot(mean_binned_speed_B1, color='orange', label='B1')
                axes[0].fill_between(range(len(mean_binned_speed_B1)), 
                                mean_binned_speed_B1 + sem_binned_speed_B1, 
                                mean_binned_speed_B1 - sem_binned_speed_B1,
                                color='orange', alpha=0.3)
                axes[0].plot(mean_binned_speed_B2, color='gold', label='B2')
                axes[0].fill_between(range(len(mean_binned_speed_B2)), 
                                mean_binned_speed_B2 + sem_binned_speed_B2, 
                                mean_binned_speed_B2 - sem_binned_speed_B2,
                                color='gold', alpha=0.3)
                axes[0].plot(mean_binned_speed_B3, color='brown', label='B3')
                axes[0].fill_between(range(len(mean_binned_speed_B3)), 
                                mean_binned_speed_B3 + sem_binned_speed_B3, 
                                mean_binned_speed_B3 - sem_binned_speed_B3,
                                color='brown', alpha=0.3)
                axes[0].set_title('Speed')

                axes[1].plot(mean_binned_licks_A1, color='darkblue', label='A')
                axes[1].fill_between(range(len(mean_binned_licks_A1)), 
                                mean_binned_licks_A1 + sem_binned_licks_A1, 
                                mean_binned_licks_A1 - sem_binned_licks_A1,
                                color='darkblue', alpha=0.3)
                axes[1].plot(mean_binned_licks_B1, color='orange', label='B1')
                axes[1].fill_between(range(len(mean_binned_licks_B1)), 
                                mean_binned_licks_B1 + sem_binned_licks_B1, 
                                mean_binned_licks_B1 - sem_binned_licks_B1,
                                color='orange', alpha=0.3)
                axes[1].plot(mean_binned_licks_B2, color='gold', label='B2')
                axes[1].fill_between(range(len(mean_binned_licks_B2)), 
                                mean_binned_licks_B2 + sem_binned_licks_B2, 
                                mean_binned_licks_B2 - sem_binned_licks_B2,
                                color='gold', alpha=0.3)
                axes[1].plot(mean_binned_licks_B3, color='brown', label='B3')
                axes[1].fill_between(range(len(mean_binned_licks_B3)), 
                                mean_binned_licks_B3 + sem_binned_licks_B3, 
                                mean_binned_licks_B3 - sem_binned_licks_B3,
                                color='brown', alpha=0.3)
                axes[1].set_title('Lick rate')

                bins = mean_binned_speed_A1.shape[0]
                for ax in axes:
                    ax.axvspan(bins/2, bins, color='grey', alpha=0.3)
                    ax.set_xticks([0, bins/2, bins], labels=[f'{-window_seconds/2:.1f}', 0, f'{window_seconds/2:.1f}'])
                    # ax.set_xticks([bins/2, bins], labels=['lm entry', 'lm exit'])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.legend()

        plt.tight_layout()

    return fig

def plot_transition_matrix(sess_dataframe, ses_settings):
    from matplotlib.colors import Normalize

    target_id, distractor_id, target_positions, distractor_positions, lm_ids, lm_id_sequence = find_targets_distractors(sess_dataframe, ses_settings)
    
    transition_matrix, lick_tm, ideal_tm = calc_transition_matrix(sess_dataframe, ses_settings)
    label_map = {}
    for i, tid in enumerate(target_id, start=1):
        label_map[tid] = f"A{i}"
    for i, did in enumerate(distractor_id, start=1):
        label_map[did] = f"B{i}"
    labels = [label_map[i] for i in lm_ids]
    
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

def plot_seq_fraction(sess_dataframe,ses_settings,test='transition'):

    performance, perf_a, perf_b, perf_c = calc_seq_fraction(sess_dataframe,ses_settings,test='transition')
    perf_ctrl, perf_a_ctrl, perf_b_ctrl, perf_c_ctrl = calc_seq_fraction(sess_dataframe,ses_settings,test='control')

    plt.figure(figsize=(6, 4))
    plt.bar(['A->B', 'B->C', 'C->A'], [perf_a, perf_b, perf_c], color=['blue', 'orange', 'green'])
    #add a dashed bar plot for control
    plt.bar(['A->B', 'B->C', 'C->A'], [perf_a_ctrl, perf_b_ctrl, perf_c_ctrl], color=['blue', 'orange', 'green'], alpha=0.3, hatch='//')
    plt.ylim(0, 1)
    plt.ylabel('Fraction of Correct Transitions')
    plt.title('Sequencing Performance per Transition')
    plt.show()

    print(f'Sequencing Performance: {performance*100:.2f}%')
    print(f'Control Performance: {perf_ctrl*100:.2f}%')

    return performance

def plot_stable_seq_fraction(sess_dataframe,ses_settings,test='transition'):

    performance, perf_a, perf_b, perf_c, perf_d = calc_stable_seq_fraction(sess_dataframe,ses_settings,test='transition')
    perf_ctrl, perf_a_ctrl, perf_b_ctrl, perf_c_ctrl, perf_d_ctrl = calc_stable_seq_fraction(sess_dataframe,ses_settings,test='control')

    plt.figure(figsize=(6, 4))
    plt.bar(['A->B', 'B->C', 'C->D', 'D->A'], [perf_a, perf_b, perf_c, perf_d], color=['blue', 'orange', 'green', 'purple'])
    #add a dashed bar plot for control
    plt.bar(['A->B', 'B->C', 'C->D', 'D->A'], [perf_a_ctrl, perf_b_ctrl, perf_c_ctrl, perf_d_ctrl], color=['blue', 'orange', 'green', 'purple'], alpha=0.3, hatch='//')
    plt.ylim(0, 1)
    plt.ylabel('Fraction of Correct Transitions')
    plt.title('Sequencing Performance per Transition')
    plt.show()

    print(f'Sequencing Performance: {performance*100:.2f}%, ({perf_a*100:.2f}%, {perf_b*100:.2f}%, {perf_c*100:.2f}%, {perf_d*100:.2f}%)')
    print(f'Control Performance: {perf_ctrl*100:.2f}%, ({perf_a_ctrl*100:.2f}%, {perf_b_ctrl*100:.2f}%, {perf_c_ctrl*100:.2f}%, {perf_d_ctrl*100:.2f}%)')

def plot_stable_seq_fraction_new(sess_dataframe,ses_settings):

    performance, perf_a, perf_b, perf_c, perf_d = calc_stable_seq_fraction_new(sess_dataframe,ses_settings,test='transition')
    perf_ctrl, perf_a_ctrl, perf_b_ctrl, perf_c_ctrl, perf_d_ctrl = calc_stable_seq_fraction_new(sess_dataframe,ses_settings,test='control')

    plt.figure(figsize=(6, 4))
    plt.bar(['A->B', 'B->C', 'C->D', 'D->A'], [perf_a, perf_b, perf_c, perf_d], color=['blue', 'orange', 'green', 'purple'])
    #add a dashed bar plot for control
    plt.bar(['A->B', 'B->C', 'C->D', 'D->A'], [perf_a_ctrl, perf_b_ctrl, perf_c_ctrl, perf_d_ctrl], color=['blue', 'orange', 'green', 'purple'], alpha=0.3, hatch='//')
    plt.ylim(0, 1)
    plt.ylabel('Fraction of Correct Transitions')
    plt.title('Sequencing Performance per Transition')
    plt.show()

    print(f'Sequencing Performance: {performance*100:.2f}%, ({perf_a*100:.2f}%, {perf_b*100:.2f}%, {perf_c*100:.2f}%, {perf_d*100:.2f}%)')
    print(f'Control Performance: {perf_ctrl*100:.2f}%, ({perf_a_ctrl*100:.2f}%, {perf_b_ctrl*100:.2f}%, {perf_c_ctrl*100:.2f}%, {perf_d_ctrl*100:.2f}%)')

def plot_switch_stay_AB(sess_dataframe,ses_settings):

    transition_prob, control_prob = calc_conditional_matrix(sess_dataframe,ses_settings)
    switch_prob = [transition_prob[i,i+1] for i in range(transition_prob.shape[0]-1)]
    stay_prob = [transition_prob[i,i] for i in range(transition_prob.shape[0]-1)]

    control_switch_prob = [control_prob[i,i+1] for i in range(control_prob.shape[0]-1)]
    control_stay_prob = [control_prob[i,i] for i in range(control_prob.shape[0]-1)]

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.bar([0, 1], [np.mean(control_switch_prob), np.mean(control_stay_prob)], color=['blue', 'orange'])
    plt.xticks([0, 1], ['Switch', 'Stay'])
    plt.ylabel('Probability')
    plt.title('Control Switch vs Stay Probability')
    plt.subplot(1, 2, 2)
    plt.bar([0, 1], [np.mean(switch_prob), np.mean(stay_prob)], color=['blue', 'orange'])
    plt.xticks([0, 1], ['Switch', 'Stay'])
    plt.ylabel('Probability')
    plt.title('Switch vs Stay Probability')
    plt.show()

def plot_sequencing_ABC(sess_dataframe,ses_settings):
    transition_prob, control_prob, ideal_prob = calc_conditional_matrix(sess_dataframe,ses_settings)

    ab_prob = transition_prob[0,1]
    ac_prob = transition_prob[0,2]
    bc_prob = transition_prob[1,2]
    ba_prob = transition_prob[1,0]
    ca_prob = transition_prob[2,0]
    cb_prob = transition_prob[2,1]

    correct_seq = np.mean([ab_prob, bc_prob, ca_prob])
    incorrect_seq = np.mean([ac_prob, ba_prob, cb_prob])

    ctrl_ab_prob = control_prob[0,1]
    ctrl_ac_prob = control_prob[0,2]
    ctrl_bc_prob = control_prob[1,2]
    ctrl_ba_prob = control_prob[1,0]
    ctrl_ca_prob = control_prob[2,0]
    ctrl_cb_prob = control_prob[2,1]
    ctrl_correct_seq = np.mean([ctrl_ab_prob, ctrl_bc_prob, ctrl_ca_prob])
    ctrl_incorrect_seq = np.mean([ctrl_ac_prob, ctrl_ba_prob, ctrl_cb_prob])

    ideal_ab_prob = ideal_prob[0,1]
    ideal_ac_prob = ideal_prob[0,2]
    ideal_bc_prob = ideal_prob[1,2]
    ideal_ba_prob = ideal_prob[1,0]
    ideal_ca_prob = ideal_prob[2,0]
    ideal_cb_prob = ideal_prob[2,1]
    ideal_correct_seq = np.mean([ideal_ab_prob, ideal_bc_prob, ideal_ca_prob])
    ideal_incorrect_seq = np.mean([ideal_ac_prob, ideal_ba_prob, ideal_cb_prob])

    #Calculate the maximum value across all data to set consistent y-axis max
    all_values = [correct_seq, incorrect_seq, ctrl_correct_seq, ctrl_incorrect_seq, 
                  ideal_correct_seq, ideal_incorrect_seq]
    y_max = max(all_values) * 1.1  #Add 10% padding at the top for space

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1)
    plt.bar([0, 1], [correct_seq, incorrect_seq], color=['green', 'red'])
    plt.title('Lick Sequence')
    plt.xticks([0, 1], ['Correct', 'Incorrect'])
    plt.ylim(0, y_max)  
    
    plt.subplot(1, 3, 2)
    plt.bar([0, 1], [ctrl_correct_seq, ctrl_incorrect_seq], color=['green', 'red'])
    plt.title('Control Sequence')
    plt.xticks([0, 1], ['Correct', 'Incorrect'])
    plt.ylim(0, y_max)  
    
    plt.subplot(1, 3, 3)
    plt.bar([0, 1], [ideal_correct_seq, ideal_incorrect_seq], color=['green', 'red'])
    plt.title('Ideal Sequence')
    plt.xticks([0, 1], ['Correct', 'Incorrect'])
    plt.ylim(0, y_max)  
    
    plt.tight_layout()
    plt.show()

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

def plot_licks_per_state(sess_dataframe, ses_settings):
    state_id = give_state_id(sess_dataframe,ses_settings)

    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)
    laps_needed = calc_laps_needed(ses_settings)
    if licked_all.shape[0] % 10 != 0:
        #extend the array to make it divisible by 10
        licked_all = np.pad(licked_all, (0, 10 - (licked_all.shape[0] % 10)), 'constant')
    licked_all_reshaped = licked_all.reshape(np.round(licked_all.shape[0] / 10).astype(int), 10)

    if laps_needed == 2:
        state1_laps = licked_all_reshaped[np.where(state_id == 0)[0],:]
        state2_laps = licked_all_reshaped[np.where(state_id == 1)[0],:]

        state1_hist = np.sum(state1_laps,axis=0)/state1_laps.shape[0]
        state2_hist = np.sum(state2_laps,axis=0)/state2_laps.shape[0]

    elif laps_needed == 3:
        state1_laps = licked_all_reshaped[np.where(state_id == 0)[0],:]
        state2_laps = licked_all_reshaped[np.where(state_id == 1)[0],:]
        state3_laps = licked_all_reshaped[np.where(state_id == 2)[0],:]

        state1_hist = np.sum(state1_laps,axis=0)/state1_laps.shape[0]
        state2_hist = np.sum(state2_laps,axis=0)/state2_laps.shape[0]
        state3_hist = np.sum(state3_laps,axis=0)/state3_laps.shape[0]

    plt.figure(figsize=(10,2))
    plt.plot(state1_hist, label='Lap1', color='g')
    plt.plot(state2_hist, label='Lap2', color='r')
    if laps_needed == 3:
        plt.plot(state3_hist, label='Lap3', color='y')
    plt.xlabel('Landmark')
    plt.ylabel('Fraction of Licks')
    plt.legend()
    plt.title('Licks per State/Lap')
    plt.show()

def plot_speed_per_state(sess_dataframe, ses_settings):

    speed_per_bin = calc_speed_per_lap(sess_dataframe, ses_settings)
    state_id = give_state_id(sess_dataframe,ses_settings)
    laps_needed = calc_laps_needed(ses_settings)

    if laps_needed == 2:
        state1_laps = speed_per_bin[np.where(state_id == 0)[0]]
        state2_laps = speed_per_bin[np.where(state_id == 1)[0]]
    elif laps_needed == 3:
        state1_laps = speed_per_bin[np.where(state_id == 0)[0]]
        state2_laps = speed_per_bin[np.where(state_id == 1)[0]]
        state3_laps = speed_per_bin[np.where(state_id == 2)[0]]

    state1_speed = np.nanmean(state1_laps, axis=0)
    state1_speed_sem = np.nanstd(state1_laps, axis=0)/np.sqrt(state1_laps.shape[0])
    state2_speed = np.nanmean(state2_laps, axis=0)
    state2_speed_sem = np.nanstd(state2_laps, axis=0)/np.sqrt(state2_laps.shape[0])
    if laps_needed == 3:
        state3_speed = np.nanmean(state3_laps, axis=0)
        state3_speed_sem = np.nanstd(state3_laps, axis=0)/np.sqrt(state3_laps.shape[0])
    
    plt.figure(figsize=(10,4))
    plt.plot(state1_speed, label='Lap1', color='g')
    plt.fill_between(range(state1_speed.shape[0]), state1_speed - state1_speed_sem, state1_speed + state1_speed_sem, color='g', alpha=0.3)
    plt.plot(state2_speed, label='Lap2', color='r')
    plt.fill_between(range(state2_speed.shape[0]), state2_speed - state2_speed_sem, state2_speed + state2_speed_sem, color='r', alpha=0.3)
    if laps_needed == 3:
        plt.plot(state3_speed, label='Lap3', color='y')
        plt.fill_between(range(state3_speed.shape[0]), state3_speed - state3_speed_sem, state3_speed + state3_speed_sem, color='y', alpha=0.3)
    plt.xlabel('Position Bin')
    plt.ylabel('Treadmill Speed')
    plt.legend()
    plt.title('Treadmill Speed per State/Lap')
    plt.show()

def plot_sw_state_ratio(sess_dataframe, ses_settings):

    sw_state_ratio, sw_state_ratio_a, sw_state_ratio_b, sw_state_ratio_c, sw_state_ratio_d = calc_sw_state_ratio(sess_dataframe, ses_settings)
    print(f'Average Switch-Stay Ratio: {np.nanmean(sw_state_ratio):.2f}')

    plt.figure(figsize=(10,4))
    plt.plot(sw_state_ratio, label='Average', color='k')
    plt.plot(sw_state_ratio_a, label='A', color='b')
    plt.plot(sw_state_ratio_b, label='B', color='g')
    plt.plot(sw_state_ratio_c, label='C', color='r')
    plt.plot(sw_state_ratio_d, label='D', color='m')
    plt.hlines(np.nanmean(sw_state_ratio),0,len(sw_state_ratio),colors='k',linestyles='dashed',label='Mean')
    plt.ylim(0, 1)
    plt.xlabel('Lap')
    plt.ylabel('Switch-Stay Ratio')
    plt.legend()
    plt.title('Switch-Stay Ratio per State/Lap')
    plt.show()

def plot_lick_maps(session):
    '''Plot binary lick and lick rate maps per landmark'''

    # ------- Get binary lick map (laps x landmarks) ------- #
    session = get_binary_lick_map(session)

    # ------- Get lick rate map (laps x landmarks) ------- #
    session = get_lm_lick_rate(session)

    # Plotting
    tm_palette = palettes.met_brew('Tam', n=123, brew_type="continuous")
    tm_palette = tm_palette[::-1]

    tick_positions = [i * 16 + 16 // 2 for i in range(session['num_landmarks'])]
    tick_labels = np.arange(1, session['num_landmarks']+1)  
    
    # Plot the binary and lick rate maps for each landmark 
    if session['num_landmarks'] == 2:
        _, ax = plt.subplots(1,2, figsize=(8,3), sharex=False, sharey=False)
    else:
        _, ax = plt.subplots(2, 1, figsize=(12,7), sharex=False, sharey=False)
    ax = ax.ravel()

    # Plot binary licks  
    sns.heatmap(session['binary_licked_lms'], ax=ax[0], cmap=[tm_palette[0], tm_palette[-1]], 
                vmin=0, vmax=1, cbar_kws={"ticks": [0, 1]}, xticklabels=(tick_labels), 
                yticklabels=[0, session['binary_licked_lms'].shape[0]])

    # Plot lick rate
    max_lick_rate = np.round(np.nanmax(session['lm_lick_rate']), 1)
    sns.heatmap(session['lm_lick_rate'], ax=ax[1], cmap=tm_palette, vmin=0, vmax=max_lick_rate, 
                cbar_kws={"ticks": [0, max_lick_rate]})
    for i in range(1, session['num_landmarks']):
        ax[1].axvline(i * 16, color='white', linestyle='--', linewidth=1)

    for axis in ax:
        axis.set_yticks([0, session['binary_licked_lms'].shape[0]])
        axis.set_yticklabels([0, session['binary_licked_lms'].shape[0]])
        axis.set_ylabel('Lap')

    ax[0].set_title('Licked Landmarks')
    ax[1].set_title('Lick Rate')
    ax[1].set_xlabel('Landmark')
    ax[1].set_xticks(tick_positions)
    ax[1].set_xticklabels(tick_labels, rotation=0)
    
    plt.tight_layout()

def plot_speed_profile(session, plot_threshold=False):
    '''Plot the speed profile per landmark'''
    session = calc_speed_per_lm(session)

    stage = extract_int(session['stage'])
    if stage == 3:
        color = '#325235'
    elif stage == 4:
        color = '#9E664C'
    elif stage == 5:
        color = ['darkgreen', 'yellowgreen'] #'blue'
    elif stage == 6:
        color = ['darkred', 'tomato'] #'orange'
    elif stage == 8:
        color = 'red'
    else: 
        color = 'black'

    if session['num_landmarks'] == 2:
        fig, ax = plt.subplots(1, 1, figsize=(8,3), sharex=False, sharey=False)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10,3))
    ax.plot(session['speed_per_bin'], color=color[0], label='speed')
    ax.fill_between(range(len(session['speed_per_bin'])),
                    session['speed_per_bin'] - session['sem_speed_per_bin'],
                    session['speed_per_bin'] + session['sem_speed_per_bin'],
                    color=color[0], alpha=0.3)

    if plot_threshold:
        ax.axhline(session['lick_threshold'], color='grey', linestyle='--', linewidth=2, label='speed threshold')
        ax.legend(loc='lower right')

    for lm in session['binned_lms']:
        ax.axvspan(lm[0], lm[1], color='grey', alpha=0.3)
    for goal in session['binned_goals']:
        ax.axvspan(goal[0], goal[1], color='grey', alpha=0.5)
    ax.set_xlabel('Landmark', fontsize=10)
    ax.set_ylabel('Speed (cm/s)', fontsize=10)

    tick_labels = np.arange(1,11)
    tick_positions = [(lm[0] + (lm[1]-lm[0])/2) for lm in session['binned_lms']]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=10)
    for i, label in enumerate(ax.get_xticklabels()):
        if i in [1,3,5,7]:  
            label.set_color('#1E2985')
        elif i in [0,2,4,6,8]:
            label.set_color('#9D1DA3')
        elif i == 9:
            label.set_color('darkorange')
            
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    return fig

def plot_acceleration_profile(session, data='acceleration'):

    if data == 'acceleration':
        if 'accel_per_bin' not in session:
            session = calc_accel_per_lap_pre7(session)
        avg_data = session['accel_per_bin']
        sem_data = session['sem_accel_per_bin']
        label = 'Acceleration (cm/s^2)'

    elif data == 'deceleration':
        if 'decel_per_bin' not in session:
            session = calc_decel_per_lap_pre7(session)
        avg_data = session['decel_per_bin']
        sem_data = session['sem_decel_per_bin']
        label = 'Deceleration (cm/s^2)'

    elif data == 'both':
        if 'accel_per_bin' not in session:
            session = calc_accel_per_lap_pre7(session)
        if 'decel_per_bin' not in session:
            session = calc_decel_per_lap_pre7(session)
        avg_data1 = session['accel_per_bin']
        sem_data1 = session['sem_accel_per_bin']
        avg_data2 = session['decel_per_bin']
        sem_data2 = session['sem_decel_per_bin']
        label1 = 'Acceleration (cm/s^2)'
        label2 = 'Deceleration (cm/s^2)'

    stage = extract_int(session['stage'])
    if stage == 3:
        color = '#325235'
    elif stage == 4:
        color = '#9E664C'
    elif stage == 5:
        color = ['darkgreen', 'yellowgreen'] #'blue'
    elif stage == 6:
        color = ['darkred', 'tomato'] #'orange'
    elif stage == 8:
        color = 'red'
    else: 
        color = 'black'

    if session['num_landmarks'] == 2:
        fig, ax = plt.subplots(1, 1, figsize=(8,3), sharex=False, sharey=False)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10,3))

    if data == 'both':
        ax.plot(avg_data1, color=color[0], label=label1)
        ax.plot(avg_data2, color=color[1], label=label2)
        ax.fill_between(range(len(avg_data1)),
                        avg_data1 - sem_data1,
                        avg_data1 + sem_data1,
                        color=color[0], alpha=0.3)
        ax.fill_between(range(len(avg_data2)),
                        avg_data2 - sem_data2,
                        avg_data2 + sem_data2,
                        color=color[1], alpha=0.3)
    else:
        ax.plot(avg_data, color=color[0])
        ax.axhline(y=0, color='grey', linestyle='--', linewidth=2)
        ax.fill_between(range(len(avg_data)),
                        avg_data - sem_data,
                        avg_data + sem_data,
                        color=color[0], alpha=0.3)

    for lm in session['binned_lms']:
        ax.axvspan(lm[0], lm[1], color='grey', alpha=0.3)
    for goal in session['binned_goals']:
        ax.axvspan(goal[0], goal[1], color='grey', alpha=0.5)
    ax.set_xlabel('Landmark', fontsize=10)
    if data != 'both':
        ax.set_ylabel(label, fontsize=10)
    else:
        ax.legend(loc='upper left')

    tick_labels = np.arange(1,11)
    tick_positions = [(lm[0] + (lm[1]-lm[0])/2) for lm in session['binned_lms']]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=10)
    for i, label in enumerate(ax.get_xticklabels()):
        if i in [1,3,5,7]:  
            label.set_color('#1E2985')
        elif i in [0,2,4,6,8]:
            label.set_color('#9D1DA3')
        elif i == 9:
            label.set_color('darkorange')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    return fig
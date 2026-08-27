from aeon.io.reader import Csv, Reader
import aeon.io.api as aeon
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import json
import importlib
import re, os, sys

np.set_printoptions(suppress=True, precision=2)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

#%% ##### Loading #####
class AnalogData(Reader):
    def __init__(self, pattern, columns, channels, extension="bin"):
        super().__init__(pattern, columns, extension)
        self.channels = channels

    def read(self, file):
        data = np.fromfile(file, dtype=np.float64)
        data = np.reshape(data, (-1, self.channels))
        return pd.DataFrame(data, columns=self.columns)

    
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



def get_licks_idx(session, lick_threshold=True):
    '''Get the idx of licks in the session'''

    if lick_threshold:
        session = threshold_licks(session)
    else:
        licks_idx = np.where(session['licks'])[0]
        session['licks_idx'] = licks_idx

    return session 

def threshold_licks(sess):
    # Threshold licks based on speed 
    speed_ok = sess['speed'] < sess['lick_threshold']
    licked = sess['licks'] > 0
    threshold_mask = speed_ok & licked

    licks_idx = np.where(threshold_mask)[0]
    thresholded_licks = np.zeros(len(sess['licks']))
    thresholded_licks[licks_idx] = sess['licks'][licks_idx]
    # thresholded_licks = session['licks'][licks_idx]

    sess['thresholded_licks'] = thresholded_licks
    sess['licks_idx'] = licks_idx

    return sess

# 

# def create_odour_lm_mapping(ses_settings):
#     TODO: remove? 
#     '''Create a list of rewarded and non-rewarded odours based on the order in which they are created in the session settings file'''
    
#     odour_lm_id_mapping = []
#     for lm_list in ses_settings['trial']['landmarks']:
#         for lm in lm_list:
#             odour_id = extract_int(lm['odour'])
#             if np.isin(odour_id, odour_lm_id_mapping) or odour_id == 0:
#                 break
#             else:
#                 odour_lm_id_mapping.append(odour_id)

#     return odour_lm_id_mapping

# def calculate_frame_lick_rate(session):
#     TODO: remove?
#     """Get lick rate per frame as a sliding window"""
    
#     # Calculate lick rate as the mean number of licks over sliding window
#     window = 100 # frames
#     lick_rate = np.zeros(len(session['position']))
#     for i in range(len(session['position'])-window):
#         lick_num = len(np.where((session['licks_idx'] > i) & (session['licks_idx'] < i+window))[0])
#         lick_rate[i] = lick_num / window
    
#     session['frame_lick_rate'] = lick_rate

#     return session

# def get_num_landmarks(session):
#     # Get number of unique landmarks for the session
#     session['num_landmarks'] = len(session['lm_ids'])

#     return session


# def print_sess_summary(sess_dataframe, ses_settings):
#     # TODO integrate
#     rew_odour, rew_texture, non_rew_odour, non_rew_texture = parse_rew_lms(ses_settings)
#     hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)

#     print(f'Session Summary:')
#     print(f"Total Licks: {sess_dataframe['Licks'].sum()}")
#     print(f"Total Landmarks: {licked_all.shape[0]}")
#     print(f"Total Rewards: {sess_dataframe['Rewards'].notna().sum()}")
#     print(f'Hit Rate: {hit_rate*100:.2f}%, False Alarm Rate: {fa_rate*100:.2f}%, D-prime: {d_prime:.2f}')
#     print(f'Targets Licked: {np.sum(licked_target).astype(int)} of {len(licked_target)}, Distractors Licked: {np.sum(licked_distractor).astype(int)} of {len(licked_distractor)}')
#     print(f'rewarded odours: {rew_odour}, rewarded textures: {rew_texture}')
#     print(f'non-rewarded odours: {non_rew_odour}, non-rewarded textures: {non_rew_texture}')

#%% ##### Functions that work with NIDAQ data only (after funcimg alignment) #####
# TODO 
# def get_landmark_positions(session, sess_dataframe, ses_settings, data='pd'):
#     '''Get the start and end of each landmark'''
#     trial = ses_settings['trial']
#     if isinstance(trial, list):
#         trial = trial[0]['trial']
#     lm_size = trial['landmarks'][0][0]['size']

#     if data == 'odour':
#         # Estimate landmark entries based on odour release positions 
#         lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
#         lm_idx = np.asarray(release_df['Index'].to_numpy(), dtype=int)
        
#         position = np.nan_to_num(sess_dataframe['Position'].values, nan=0.0)
#         release_positions = position[lm_idx]
        
#         landmarks = np.zeros((len(release_positions), 2))
#         for lm, pos in enumerate(release_positions):
#             landmarks[lm,0] = pos
#             landmarks[lm,1] = pos + self.session.lm_size

#     elif data == 'pd':
#         lm_entry_idx1, lm_exit_idx1 = estimate_pd_entry_exit(ses_settings, session, pd='pd1')
#         lm_entry_idx2, lm_exit_idx2 = estimate_pd_entry_exit(ses_settings, session, pd='pd2')
        
#         entry_pos1 = session['position'][lm_entry_idx1]
#         entry_pos2 = session['position'][lm_entry_idx2]
#         exit_pos1  = session['position'][lm_exit_idx1]
#         exit_pos2  = session['position'][lm_exit_idx2]
        
#         trial = ses_settings['trial']
#         if isinstance(trial, list):
#             trial = trial[0]['trial']
#         lm_size = trial['landmarks'][0][0]['size']
#         offset = ses_settings['trial']['offsets'][0]
#         tol = lm_size * 0.5

#         # Merge with "keep single" logic
#         all_lm_entry = merge_positions_keep_single(entry_pos1, entry_pos2, tol, offset)
#         all_lm_exit  = merge_positions_keep_single(exit_pos1,  exit_pos2,  tol, offset)

#         # Fix last lm 
#         if session['position'][-1] - all_lm_exit[-1] < lm_size:
#             all_lm_exit = all_lm_exit[:-1]
        
#         # Fix first lm 
#         first_entries = all_lm_entry < offset
#         first_exits  = all_lm_exit  < offset
#         first_entry = all_lm_entry[first_entries][0] 
#         first_exit = all_lm_exit[first_exits][-1]
        
#         # Concatenate all landmarks 
#         lm_entry = np.concatenate([[first_entry], all_lm_entry[~first_entries]])
#         lm_exit = np.concatenate([[first_exit], all_lm_exit[~first_exits]])

#         if len(lm_entry) != len(lm_exit):
#             if len(lm_entry) - len(lm_exit) == 1:
#                 # Session ended before the mouse exited the last landmark 
#                 n = len(lm_exit)
#                 lm_entry = lm_entry[:n]
#             else:
#                 raise ValueError(f'Something is wrong with landmark parsing using the photodiode data in {session['mouse']} {session['stage']}')

#         # Store landmarks 
#         landmarks = np.column_stack([lm_entry, lm_exit])

#     session['landmarks'] = landmarks

#     return session

# def merge_positions_keep_single(pos1, pos2, tol, offset):
#     """
#     Merge two sorted position arrays.
#     - If positions are within tol → average
#     - If only one exists → keep it
#     """
#     i = j = 0
#     merged = []

#     while i < len(pos1) and j < len(pos2):
#         if pos1[i] < offset:
#             merged.append(pos1[i])
#             i += 1
#             continue

#         if pos2[j] < offset:
#             merged.append(pos2[j])
#             j += 1
#             continue

#         if abs(pos1[i] - pos2[j]) <= tol:
#             merged.append(np.mean([pos1[i], pos2[j]]))
#             i += 1
#             j += 1
#         elif pos1[i] < pos2[j]:
#             merged.append(pos1[i])
#             i += 1
#         else:
#             merged.append(pos2[j])
#             j += 1

#     # append leftovers
#     while i < len(pos1):
#         merged.append(pos1[i])
#         i += 1

#     while j < len(pos2):
#         merged.append(pos2[j])
#         j += 1

#     return np.array(merged)

# def estimate_pd_entry_exit(ses_settings, session, pd='pd1'):
#     '''Estimate lm entry and exit indices using photodiode data'''
#     binary_pd = (session[pd] >= 100).astype(int)

#     all_lm_entry_idx = np.where(np.diff(binary_pd) == 1)[0] + 1
#     all_lm_exit_idx = np.where(np.diff(binary_pd) == -1)[0] + 1
#     if binary_pd[0] == 1:
#         all_lm_entry_idx = np.insert(all_lm_entry_idx, 0, 0)
    
#     trial = ses_settings['trial']
#     if isinstance(trial, list):
#         trial = trial[0]['trial']
#     lm_size = trial['landmarks'][0][0]['size']
#     # lm_size = ses_settings['trial']['landmarks'][0][0]['size']
#     offset = ses_settings['trial']['offsets'][0]

#     # Filter out repeated lm visits
#     # n = min(len(all_lm_entry_idx), len(all_lm_exit_idx))
#     # entry_pos = session['position'][all_lm_entry_idx[:n]]
#     # exit_pos  = session['position'][all_lm_exit_idx[:n]]
#     entry_pos = session['position'][all_lm_entry_idx]
#     exit_pos  = session['position'][all_lm_exit_idx]
#     # pos_diff = np.where(exit_pos - entry_pos < lm_size - 1)[0]

#     if offset == lm_size: # TODO bad fix 
#         tol = 0
#     else:
#         tol = 1

#     # Filter out re-entries - use earliest idx
#     consecutive_diff = np.where(np.diff(entry_pos) < lm_size + tol)[0] + 1
#     removed = []
#     for i, idx in enumerate(all_lm_entry_idx):
#         if i in consecutive_diff:
#             if session['position'][idx] < offset:
#                 continue
#             removed.append(i)
#     lm_entry_idx = np.delete(all_lm_entry_idx, removed)

#     # Filter out re-exits - use latest idx
#     consecutive_diff = np.where(np.diff(exit_pos) < lm_size + tol)[0] 
#     removed = []
#     for i, idx in enumerate(all_lm_exit_idx):
#         if i in consecutive_diff:
#             if session['position'][idx] < offset:
#                 continue
#             removed.append(i)
#     lm_exit_idx = np.delete(all_lm_exit_idx, removed)

#     # # Filter out re-entries - use earliest entry
#     # removed = []
#     # for i, idx in enumerate(all_lm_entry_idx):
#     #     if i in pos_diff:
#     #         # Check position of first outlier
#     #         if session['position'][idx] < offset:
#     #             continue
#     #         removed.append(i)
#     # lm_entry_idx = np.delete(all_lm_entry_idx, removed)

#     # # Filter out re-exits - use latest exit
#     # removed = []
#     # for i, idx in enumerate(all_lm_exit_idx):
#     #     if i in pos_diff:
#     #         # Check position of first outlier
#     #         if session['position'][idx] < offset:
#     #             continue
#     #         removed.append(i)
#     # lm_exit_idx = np.delete(all_lm_exit_idx, removed)

#     return lm_entry_idx, lm_exit_idx

# def get_lm_entry_exit(session):
#     '''Find data idx closest to landmark entry and exit. The results should be similar to estimate_pd_entry_exit.'''

#     positions = session['position']

#     lm_entry_idx = []
#     lm_exit_idx = []

#     if np.abs(positions[0] - session['landmarks'][-1,1]) < np.abs(positions[0] - session['landmarks'][0,0]):
#         search_start = np.where(positions <= session['all_landmarks'][0,0])[0][-1]  # the mouse accidentally moved backwards first
#     else: 
#         search_start = 0

#     for lm_start in session['all_landmarks'][:,0]:
#         lm_entry_idx.append(np.where(positions[search_start:] >= lm_start)[0][0] + search_start)

#     for lm_end in session['all_landmarks'][:,1]:
#         lm_exit_idx.append(np.where(positions[search_start:] <= lm_end)[0][-1] + search_start)

#     return np.array(lm_entry_idx), np.array(lm_exit_idx)

# def get_rewarded_landmarks(session):
#     '''Find the indices of rewarded (lick-triggered) landmarks.'''

#     lm_entry_idx, lm_exit_idx = get_lm_entry_exit(session)

#     # Find rewarded landmarks 
#     reward_positions = session['position'][session['reward_idx']]

#     rewarded_landmarks = [i for i, (start, end) in enumerate(zip(np.floor(session['position'][lm_entry_idx]), np.ceil(session['position'][lm_exit_idx]))) 
#                             if np.any((np.ceil(reward_positions) >= start) & (np.floor(reward_positions) <= end))] 

#     session['rewarded_landmarks'] = rewarded_landmarks

#     return session


# def get_reward_idx(session):
#     # Ensure mouse has left last rewarded landmark 
#     reward_idx = session['rewards']
#     if session['all_landmarks'][-1,1] < session['position'][reward_idx[-1]]:  
#         reward_idx = reward_idx[0:-1]  
#         print('Mouse did not leave the last rewarded landmark. Removing landmark...')

#     session['reward_idx'] = reward_idx

#     return session 
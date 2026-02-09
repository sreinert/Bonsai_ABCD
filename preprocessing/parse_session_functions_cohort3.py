from aeon.io.reader import Csv, Reader
import aeon.io.api as aeon
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def find_base_path(mouse, date, root):
    mouse_path = Path(root) / f"sub-{mouse}" 

    for folder in mouse_path.iterdir():
        if folder.is_dir() and date in folder.name:
            print(f"Found folder: {folder}")
            base_path = folder
    return base_path

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
    events_reader = Csv("behav/experiment-events/*", ["Seconds", "Value"])
    events_data = aeon.load(Path(base_path), events_reader)

    lick_reader = Csv("behav/licks/*", ["Seconds", "Value"])
    lick_data = aeon.load(Path(base_path), lick_reader)

    rewards_reader = Csv("behav/reward/*", ["Seconds", "Value"])
    rewards_data = aeon.load(Path(base_path), rewards_reader)

    position_reader = Csv("behav/current-position/*", ["Seconds","Value.X","Value.Y","Value.Z","Value.Length", "Value.LengthFast", "Value.LengthSquared"])
    position_data = aeon.load(Path(base_path), position_reader)

    treadmill_reader = Csv("behav/treadmill-speed/*", ["Seconds", "Value"])
    treadmill_data = aeon.load(Path(base_path), treadmill_reader)

    buffer_reader = Csv("behav/analog-data/*", ["Seconds", "Value"])
    buffer_data = aeon.load(Path(base_path), buffer_reader)

    if os.path.exists(Path(base_path) / "behav/current-landmark/"):
        lm_reader = Csv("behav/current-landmark/*", ["Seconds","Count","Size","Texture","Odour","SequencePosition","Position","Visited"])
        lm_data = aeon.load(Path(base_path), lm_reader)
        lm_data = lm_data[lm_data['Visited'] == False]
        sess_lm_data = lm_data.drop_duplicates(subset=['Position'], keep='first')

    sess_events_data = events_data[~events_data.index.duplicated(keep='first')]
    sess_lick_data = lick_data[~lick_data.index.duplicated(keep='first')]
    sess_treadmill_data = treadmill_data[~treadmill_data.index.duplicated(keep='first')]
    sess_position_data = position_data[~position_data.index.duplicated(keep='first')]
    sess_reward_data = rewards_data[~rewards_data.index.duplicated(keep='first')]
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
        #combine indices
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

    lick_position = sess_dataframe['Position'].values[sess_dataframe['Licks'].values > 0]
    lick_times = sess_dataframe.index[sess_dataframe['Licks'].values > 0]
    reward_times = sess_dataframe.index[sess_dataframe['Rewards'].notna()]
    reward_positions = sess_dataframe['Position'].values[sess_dataframe['Rewards'].notna()]

    if 'LM_Count' in sess_dataframe.columns:
        release_df = estimate_lm_events(sess_dataframe)
    else:
        release_df = estimate_release_events(sess_dataframe, ses_settings)

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

    num_lms = len(ses_settings['trial']['landmarks'])
    num_goals = ses_settings['availableRewardPositions']
    lm_ids = np.arange(num_lms)
    goal_counter = 0
    goals = []
    while goal_counter < num_goals:
        for i in range(num_lms):
            for j in ses_settings['trial']['landmarks'][i]:
                if j['rewardSequencePosition'] == goal_counter:
                    goals.append(i)
                    goal_counter += 1
                    if goal_counter >= num_goals:
                        break
                    
    return goals, lm_ids

def parse_random_goal_ids(ses_settings):
    '''Identify the number of landmarks and goals for random world sequences'''
    rew_odour, _, non_rew_odour, _ = parse_rew_lms(ses_settings)

    num_lms = len(rew_odour) + len(non_rew_odour)
    num_goals = ses_settings['availableRewardPositions']
    lm_ids = np.arange(num_lms)

    goal_counter = 0
    goals = []
    while goal_counter < num_goals:
        for i in range(num_lms):
            for j in ses_settings['trial']['landmarks'][i]:
                if j['rewardSequencePosition'] == goal_counter:
                    goals.append(i)
                    goal_counter += 1
                    if goal_counter >= num_goals:
                        break

    return goals, lm_ids

def calc_hit_fa(sess_dataframe,ses_settings):
    lm_size = ses_settings['trial']['landmarks'][0][0]['size']

    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)

    rew_odour, rew_texture, non_rew_odour, non_rew_texture = parse_rew_lms(ses_settings)

    target_positions, distractor_positions, target_id, distractor_id, was_target, lm_id = find_targets_distractors(sess_dataframe, ses_settings)

    licked_target = np.zeros(len(target_positions))
    for idx, pos in enumerate(target_positions):
        if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
            licked_target[idx] = 1

    licked_distractor = np.zeros(len(distractor_positions))
    for idx, pos in enumerate(distractor_positions):
        if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
            licked_distractor[idx] = 1

    if 'LM_Count' in sess_dataframe.columns:
        LM_offset = 3
    else:
        LM_offset = 0
    licked_all = np.zeros(len(release_df))
    rewarded_all = np.zeros(len(release_df))
    release_positions = release_df['Position'].to_numpy()
    for idx, pos in enumerate(release_positions):
        #only take into account licks/rewards that came later than the release
        licks = lick_position[lick_times >= release_df.index[idx]]
        rewards = reward_positions[reward_times >= release_df.index[idx]]
        #compare licks/rewards to position window (the LM position and logged position are offset by 3)
        if np.any((licks > (pos - LM_offset)) & (licks < (pos - LM_offset + lm_size))):
           licked_all[idx] = 1
        if np.any((rewards > (pos - LM_offset)) & (rewards < (pos - LM_offset + lm_size))):
           rewarded_all[idx] = 1

    hit_rate = np.sum(licked_target) / len(licked_target) 
    fa_rate = np.sum(licked_distractor) / len(licked_distractor) 
    #adjust hit rate and fa rate to avoid infinity in d-prime calculation
    if hit_rate == 1:
        hit_rate = 0.99
    if hit_rate == 0:
        hit_rate = 0.01
    if fa_rate == 1:
        fa_rate = 0.99
    if fa_rate == 0:
        fa_rate = 0.01

    d_prime = np.log10(hit_rate/(1-hit_rate)) - np.log10(fa_rate/(1-fa_rate))

    return hit_rate, fa_rate,d_prime, licked_target, licked_distractor, licked_all,rewarded_all

def extract_int(s: str) -> int:
    m = re.search(r'\d+', s)
    if m:
        return int(m.group())
    else:
        raise ValueError(f"No digits found in string: {s!r}")

def find_targets_distractors(sess_dataframe,ses_settings):
    
    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)
    rew_odour, rew_texture, non_rew_odour, non_rew_texture = parse_rew_lms(ses_settings)

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

def calc_transition_matrix(sess_dataframe,ses_settings):
    
    target_positions, distractor_positions, target_id, distractor_id, was_target, lm_id = find_targets_distractors(sess_dataframe,ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)
    ideal_licks = get_ideal_performance(sess_dataframe,ses_settings)

    lick_sequence = lm_id[licked_all==1]
    num_landmarks = int(np.max(lm_id)) + 1
    ideal_sequence = lm_id[ideal_licks==1]
        
    transition_matrix = np.zeros((num_landmarks, num_landmarks))
    lick_tm = np.zeros((num_landmarks, num_landmarks))
    ideal_tm = np.zeros((num_landmarks, num_landmarks))
    for i in range(len(lick_sequence)-1):
        current_lm = int(lick_sequence[i])
        next_lm = int(lick_sequence[i+1])
        lick_tm[current_lm, next_lm] += 1
    for i in range(len(lm_id)-1):
        current_lm = int(lm_id[i])
        next_lm = int(lm_id[i+1])
        transition_matrix[current_lm, next_lm] += 1
    for i in range(len(ideal_sequence)-1):
        current_lm = int(ideal_sequence[i])
        next_lm = int(ideal_sequence[i+1])
        ideal_tm[current_lm, next_lm] += 1

    return transition_matrix, lick_tm, ideal_tm

def get_ideal_performance(sess_dataframe,ses_settings):

    target_positions, distractor_positions, target_id, distractor_id, was_target, lm_id = find_targets_distractors(sess_dataframe,ses_settings)
    targets = np.unique(target_id)
    ideal_licks = np.zeros_like(lm_id)
    target_counter = 0
    for i in range(lm_id.shape[0]):

        if lm_id[i] == targets[target_counter]:
            ideal_licks[i] = 1  # ideal lick on target
            if target_counter < len(targets) - 1:
                target_counter += 1  # switch to the next target
            else:
                target_counter = 0  # reset to the first target
        else:
            ideal_licks[i] = 0  # no lick on distractor
    return ideal_licks

def calc_conditional_matrix(sess_dataframe,ses_settings):

    target_positions, distractor_positions, target_id, distractor_id, was_target, lm_id = find_targets_distractors(sess_dataframe, ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe, ses_settings)
    ideal_licks = get_ideal_performance(sess_dataframe, ses_settings)

    transition_prob = np.zeros((np.unique(target_id).shape[0], np.unique(lm_id).shape[0]))
    control_prob = np.zeros((np.unique(target_id).shape[0], np.unique(lm_id).shape[0])) 
    ideal_prob = np.zeros((np.unique(target_id).shape[0], np.unique(lm_id).shape[0]))
    licked_lm_ix = np.where(licked_all == 1)[0]
    controlled_lm_ix = np.where(was_target == 1)[0]

    for g in range(np.unique(target_id).shape[0]):
        rewards = np.intersect1d(np.where(rewarded_all == 1)[0],np.where(lm_id == g)[0])
        for i,reward in enumerate(rewards):
            if i == len(rewards)-1:
                break
            next_lick_index = licked_lm_ix[licked_lm_ix > reward][0]
            next_control_index = controlled_lm_ix[controlled_lm_ix > reward][0]
            next_lm = lm_id[next_lick_index].astype(int)
            next_control_lm = lm_id[next_control_index].astype(int)
            transition_prob[g,next_lm] += 1
            control_prob[g,next_control_lm] += 1
    
    for g in range(np.unique(target_id).shape[0]):
        ideal_rewards = np.intersect1d(np.where(ideal_licks == 1)[0],np.where(lm_id == g)[0])
        for i,ideal_reward in enumerate(ideal_rewards):
            if i == len(ideal_rewards)-1:
                break
            next_ideal_index = np.where(ideal_licks == 1)[0][np.where(np.where(ideal_licks == 1)[0] > ideal_reward)[0][0]]
            next_ideal_lm = lm_id[next_ideal_index].astype(int)
            ideal_prob[g,next_ideal_lm] += 1
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
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)
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
    lm_size = ses_settings['trial']['landmarks'][0][0]['size']
    lm_gap = lm_size + ses_settings['trial']['offsets'][0]

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
            chosen_idx, _, odour, chosen_pos = find_closest_events(tmp, closed_idx, pos_window = lm_size /2, event_priority=["release"], choose = "earlist")
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
    first_release = extract_int(ses_settings['trial']['landmarks'][0][0]['odour'])
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
    choose = 'earlist',
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
        if choose == 'earlist':
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

    if lm_df['Position'][0] != 0:
        # Add initial landmark at position 0 if not present
        initial_lm = pd.DataFrame({
            'time': [pd.NaT],
            'Position': [0],
            'Index': [-1],
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

def get_lap_idx(session):
    # Divide the session dataframe into laps based on the position and corridor length
    if session['world'] == 'stable':
        session['num_laps'] = int(np.ceil(session['position'].max() / session['tunnel_length']))
    elif session['world'] == 'random':
        session['num_laps'] = 1
    # For each position, determine which lap it belongs to
    session['lap_idx'] = (session['position'] // session['tunnel_length']).astype(int)

    return session

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
    licked_lms = np.zeros((session['num_laps'], len(session['landmarks'])))
    
    for i in range(session['num_laps']):
        lap_idx = np.where(session['lap_idx'] == i)[0]
        for j in range(len(session['landmarks'])):
            lm = np.where(session['lm_idx'] == j+1)[0]
            target_ix = np.intersect1d(lap_idx, lm)
            # if session['thresholded_licks'] exists, use that
            if 'thresholded_licks' in session:
                target_licks = np.intersect1d(target_ix, session['thresholded_licks'])
            # otherwise use all licks
            else:
                target_licks = np.intersect1d(target_ix, session['licks'])
            if len(target_licks) > 0:
                licked_lms[i,j] = 1
            else:
                licked_lms[i,j] = 0

    session['licked_lms'] = licked_lms

    return session

def get_rewarded_lms(session):
    # Get rewarded landmarks
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

    if session['world'] == 'stable':
        all_lms = np.array([])  # landmark ids
        for i in range(session['num_laps']):
            all_lms = np.append(all_lms, session['lm_ids'])
        all_lms = all_lms.astype(int)[:num_lms]
    elif session['world'] == 'random':
        all_lms = get_random_lm_sequence(sess_dataframe, ses_settings)
        all_lms = all_lms[:num_lms]
        
    all_landmarks = session['landmarks']  
    for i in range(1, session['num_laps']):  
        all_landmarks = np.concatenate((all_landmarks, session['landmarks']), axis=0)
    all_landmarks = all_landmarks[:num_lms]  # landmark positions

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
    for i in range(len(session['all_lms'])):
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
    
    lm_size = ses_settings['trial']['landmarks'][0][0]['size']

    if data == 'odour':
        result_df = estimate_release_events(sess_dataframe, ses_settings)
        release_positions = result_df['Position'].values
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
        
        # print(entry_pos1[:50])
        # print(entry_pos2[:50])
        # print(exit_pos1[:50])
        # print(exit_pos2[:50])
        lm_size = ses_settings['trial']['landmarks'][0][0]['size']
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
        closest_lm = np.argmin(np.abs(session['landmarks'][:,0] - pos))
        goals[i] = session['landmarks'][closest_lm]
    
    session['goals'] = goals

    return session

def estimate_pd_entry_exit(ses_settings, session, pd='pd1'):
    '''Estimate lm entry and exit indices using photodiode data'''
    binary_pd = (session[pd] >= 100).astype(int)

    all_lm_entry_idx = np.where(np.diff(binary_pd) == 1)[0] + 1
    all_lm_exit_idx = np.where(np.diff(binary_pd) == -1)[0] + 1
    if binary_pd[0] == 1:
        all_lm_entry_idx = np.insert(all_lm_entry_idx, 0, 0)
    
    lm_size = ses_settings['trial']['landmarks'][0][0]['size']
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

def create_session_struct(sess_dataframe, ses_settings, world):

    # Use the Buffer as datapoint idx
    position = np.nan_to_num(sess_dataframe['Position'].values, nan=0.0)
    speed = np.nan_to_num(sess_dataframe['Treadmill'].values, nan=0.0)
    licks = sess_dataframe['Licks'].values.astype(int)
    rewards = sess_dataframe['Buffer'][sess_dataframe['Rewards'].notna()].values
    
    if world == 'stable':
        goal_ids, lm_ids = parse_stable_goal_ids(ses_settings)
    elif world == 'random':
        goal_ids, lm_ids = parse_random_goal_ids(ses_settings)
    num_landmarks = len(lm_ids) # unique number of lm ids

    tunnel_length = calculate_corr_length(ses_settings)
    lick_threshold = ses_settings['velocityThreshold']

    session = {'position': position,
               'licks': licks, 
               'rewards': rewards, 
               'goal_ids': goal_ids, 
               'lm_ids': lm_ids,
               'num_landmarks': num_landmarks,
               'tunnel_length': tunnel_length,
               'lick_threshold': lick_threshold,
               'speed': speed}
    
    return session

def get_behaviour(session, sess_dataframe, ses_settings):
    transition_prob, control_prob, ideal_prob = calc_stable_conditional_matrix(sess_dataframe, ses_settings)
    laps_needed = calc_laps_needed(ses_settings)
    state_id = give_state_id(sess_dataframe, ses_settings)

    if int(session['stage'][-1]) > 6:
        print('Plotting the lick and speed profile for the 2 and 3 lap sequences.')
        plot_licks_per_state(sess_dataframe, ses_settings)
        plot_speed_per_state(sess_dataframe, ses_settings)

    if session['world'] == 'stable':
        _ = plot_lick_maps(session)
        plot_speed_profile(session)
    
    session['transition_prob'] = transition_prob
    session['control_prob'] = control_prob
    session['ideal_prob'] = ideal_prob
    session['laps_needed'] = laps_needed
    session['state_id'] = state_id

    return session

def analyse_npz_pre7(mouse, session_id, root, stage, world='stable'):
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
    session = get_behaviour(session, sess_dataframe, ses_settings)

    print('Number of laps = ', session['num_laps'])
    
    return session

def analyse_session_pre7_behav(session_path, mouse, stage, world='stable'):
    '''Wrapper for session analysis using behaviour data'''

    if '3' not in stage and '4' not in stage and '5' not in stage and '6' not in stage:
        raise ValueError('This function only works for T3-T6.')
    
    ses_settings, _ = load_settings(session_path)
    sess_dataframe = load_data(session_path)

    session = create_session_struct(sess_dataframe, ses_settings, world=world)
    session = get_landmark_positions(session, sess_dataframe, ses_settings, data='pd')
    session = get_goal_positions(session, sess_dataframe, ses_settings)

    session['mouse'] = mouse
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
    session = get_rewarded_landmarks(session)
    session = get_landmark_category_rew_idx(session)

    # Get behaviour
    session = get_behaviour(session, sess_dataframe, ses_settings)

    print('Number of laps = ', session['num_laps'])
    
    return session
    
#%% ##### Plotting #####
def plot_ethogram(sess_dataframe,ses_settings):
    lick_position = sess_dataframe['Position'].values[sess_dataframe['Licks'].values > 0]
    lick_times = sess_dataframe.index[sess_dataframe['Licks'].values > 0]
    reward_times = sess_dataframe.index[sess_dataframe['Rewards'].notna()]
    reward_positions = sess_dataframe['Position'].values[sess_dataframe['Rewards'].notna()]
    if 'LM_Count' in sess_dataframe.columns:
        release_df = estimate_lm_events(sess_dataframe,)
    else:
        release_df = estimate_release_events(sess_dataframe, ses_settings)
    release_times = release_df.index.tolist() # time
    release_times = release_times[1:]  # remove first release for plotting because sometimes the timestamp is NaN
    release_positions = release_df["Position"].tolist()
    release_positions = release_positions[1:]  # remove first release for plotting because sometimes the timestamp is NaN   

    num_laps, sess_dataframe = divide_laps(sess_dataframe, ses_settings)

    plt.figure(figsize=(12, 6))
    plt.plot(sess_dataframe.index, sess_dataframe['Treadmill']/np.max(sess_dataframe['Treadmill']), label='Treadmill Speed', color='purple')
    plt.plot(sess_dataframe.index, sess_dataframe['Position']/np.max(sess_dataframe['Position']), label='Position', color='blue')
    plt.plot(lick_times, lick_position/np.max(sess_dataframe['Position']), marker='o', linestyle='', label='Licks', color='orange')
    plt.plot(release_times, release_positions/np.max(sess_dataframe['Position']), marker='o', linestyle='', label='Releases', color='red')
    plt.plot(reward_times, reward_positions/np.max(sess_dataframe['Position']), marker='o', linestyle='', label='Rewards', color='green')
    plt.plot(sess_dataframe.index, sess_dataframe['Buffer']/np.max(sess_dataframe['Buffer']), label='Analog Buffer', color='black')
    plt.plot(sess_dataframe.index, sess_dataframe['Lap']/num_laps, label='Laps', color='brown')

    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Session Data Overview')
    plt.legend()
    plt.show()

def plot_transition_matrix(sess_dataframe,ses_settings):

    transition_matrix, lick_tm, ideal_tm = calc_transition_matrix(sess_dataframe,ses_settings)
    n_lms = transition_matrix.shape[0]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(transition_matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.clim(0, np.max(transition_matrix))
    plt.title('Stimulus Transition Matrix')
    plt.xlabel('Next Landmark ID')
    plt.ylabel('Current Landmark ID')
    plt.xticks([i for i in range(n_lms)])
    plt.yticks([i for i in range(n_lms)])

    plt.subplot(1, 3, 2)
    plt.imshow(lick_tm, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.clim(0, np.max(lick_tm))
    plt.title('Lick Transition Matrix')
    plt.xlabel('Next Landmark ID')
    plt.ylabel('Current Landmark ID')
    plt.xticks([i for i in range(n_lms)])
    plt.yticks([i for i in range(n_lms)])

    plt.subplot(1, 3, 3)
    plt.imshow(ideal_tm, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.clim(0, np.max(ideal_tm))
    plt.title('Ideal Transition Matrix')
    plt.xlabel('Next Landmark ID')
    plt.ylabel('Current Landmark ID')
    plt.xticks([i for i in range(n_lms)])
    plt.yticks([i for i in range(n_lms)])

    plt.tight_layout()
    plt.show()

def plot_conditional_matrix(sess_dataframe,ses_settings):

    transition_prob, control_prob, ideal_prob = calc_conditional_matrix(sess_dataframe,ses_settings)
    max_val = max(np.max(transition_prob), np.max(control_prob), np.max(ideal_prob))
    n_goals = transition_prob.shape[0]
    n_lms = transition_prob.shape[1]
    if n_goals == 3:
        y_labels = ['A', 'B', 'C']
        x_labels = [str(i) for i in range(n_lms)]
    elif n_goals == 4:
        y_labels = ['A', 'B', 'C', 'D']
        x_labels = [str(i) for i in range(n_lms)]
    else:
        y_labels = [str(i) for i in range(n_goals)]
        x_labels = [str(i) for i in range(n_lms)]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(transition_prob, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.clim(0, max_val)
    plt.title('Transition Probability Matrix (Licked)')
    plt.xlabel('Next Landmark ID')
    plt.xticks([i for i in range(n_lms)], x_labels)
    plt.yticks([i for i in range(n_goals)], y_labels)
    plt.ylabel('Goal ID')

    plt.subplot(1, 3, 2)
    plt.imshow(control_prob, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.clim(0, max_val)
    plt.title('Control Probability Matrix (All Targets)')
    plt.xlabel('Next Landmark ID')
    plt.xticks([i for i in range(n_lms)], x_labels)
    plt.yticks([i for i in range(n_goals)], y_labels)
    plt.ylabel('Goal ID')

    plt.subplot(1, 3, 3)
    plt.imshow(ideal_prob, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.clim(0, max_val)
    plt.title('Ideal Probability Matrix')
    plt.xlabel('Next Landmark ID')
    plt.xticks([i for i in range(n_lms)], x_labels)
    plt.yticks([i for i in range(n_goals)], y_labels)
    plt.ylabel('Goal ID')

    plt.tight_layout()
    plt.show()

def plot_stable_conditional_matrix(sess_dataframe,ses_settings):

    transition_prob, control_prob, ideal_prob = calc_stable_conditional_matrix(sess_dataframe,ses_settings)
    goals, lm_ids = parse_stable_goal_ids(ses_settings)
    max_val = max(np.max(transition_prob), np.max(control_prob), np.max(ideal_prob))
    n_goals = transition_prob.shape[0]
    n_lms = transition_prob.shape[1]
    if n_goals == 3:
        y_labels = ['A', 'B', 'C']
    elif n_goals == 4:
        y_labels = goals
    else:
        y_labels = [str(i) for i in range(n_goals)]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    im = plt.imshow(transition_prob, cmap='viridis', interpolation='none')
    plt.colorbar(im,fraction=0.02, pad=0.04)
    plt.clim(0, max_val)
    plt.title('Transition Probability Matrix (Licked)')
    plt.xlabel('Next Landmark ID')
    plt.xticks([i for i in range(n_lms)])
    plt.yticks([i for i in range(n_goals)], y_labels)
    plt.ylabel('Goal ID')

    plt.subplot(1, 3, 2)
    im = plt.imshow(control_prob, cmap='viridis', interpolation='none')
    plt.colorbar(im,fraction=0.02, pad=0.04)
    plt.clim(0, max_val)
    plt.title('Control Probability Matrix (All Targets)')
    plt.xlabel('Next Landmark ID')
    plt.xticks([i for i in range(n_lms)])
    plt.yticks([i for i in range(n_goals)], y_labels)
    plt.ylabel('Goal ID')

    plt.subplot(1, 3, 3)
    im = plt.imshow(ideal_prob, cmap='viridis', interpolation='none')
    plt.colorbar(im,fraction=0.02, pad=0.04)
    plt.clim(0, 1)
    plt.title('Ideal Probability Matrix')
    plt.xlabel('Next Landmark ID')
    plt.xticks([i for i in range(n_lms)])
    plt.yticks([i for i in range(n_goals)], y_labels)
    plt.ylabel('Goal ID')

    plt.tight_layout()
    plt.show()

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

    target_positions, distractor_positions, target_id, distractor_id, was_target, lm_id = find_targets_distractors(sess_dataframe,ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)

    was_target = was_target[:,np.newaxis]
    licked_all = licked_all[:,np.newaxis]
    lm_id = lm_id[:,np.newaxis]
    plt.figure(figsize=(10,4))
    plt.subplot(3, 1, 1)
    plt.imshow(was_target.T, aspect='auto', cmap='viridis')
    plt.clim(0, 1)
    plt.title('Was Target')
    #invert color map for better visibility
    plt.subplot(3, 1, 2)
    plt.imshow(lm_id.T, aspect='auto', cmap='viridis_r')
    plt.clim(0, np.max(lm_id))
    plt.title('Landmark ID')
    plt.subplot(3, 1, 3)
    plt.imshow(licked_all.T, aspect='auto', cmap='viridis')
    plt.clim(0, 1)
    plt.title('Licked All')
    plt.tight_layout()
    plt.show()

def plot_full_corr(sess_dataframe,ses_settings):

    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)
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
    hit_rate, fa_rate,d_prime, licked_target, licked_distractor, licked_all,rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)

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

    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)
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

def plot_speed_profile(session):
    '''Plot the speed profile per landmark'''
    session = calc_speed_per_lm(session)

    stage = extract_int(session['stage'])
    if stage == 3:
        color = '#325235'
    elif stage == 4:
        color = '#9E664C'
    elif stage == 5:
        color = 'blue'
    elif stage == 6:
        color = 'orange'
    elif stage == 8:
        color = 'red'
    else: 
        color = 'black'

    if session['num_landmarks'] == 2:
        _, ax = plt.subplots(1, 1, figsize=(8,3), sharex=False, sharey=False)
    else:
        _, ax = plt.subplots(1, 1, figsize=(10,3))
    ax.plot(session['speed_per_bin'], color=color)
    ax.fill_between(range(len(session['speed_per_bin'])),
                    session['speed_per_bin'] - session['sem_speed_per_bin'],
                    session['speed_per_bin'] + session['sem_speed_per_bin'],
                    color=color, alpha=0.3)

    for lm in session['binned_lms']:
        ax.add_patch(patches.Rectangle((lm[0],0), np.diff(lm)[0], ax.get_ylim()[1], color='grey', alpha=0.3))
    for goal in session['binned_goals']:
        ax.add_patch(patches.Rectangle((goal[0],0), np.diff(goal)[0], ax.get_ylim()[1], color='grey', alpha=0.5))
    ax.set_xlabel('Landmark')
    ax.set_ylabel('Speed (cm/s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return
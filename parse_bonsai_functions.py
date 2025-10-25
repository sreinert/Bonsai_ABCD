from aeon.io.reader import Csv, Reader
import aeon.io.api as aeon
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import json
from scipy.signal import find_peaks

class AnalogData(Reader):
    def __init__(self, pattern, columns, channels, extension="bin"):
        super().__init__(pattern, columns, extension)
        self.channels = channels

    def read(self, file):
        data = np.fromfile(file, dtype=np.float64)
        data = np.reshape(data, (-1, self.channels))
        return pd.DataFrame(data, columns=self.columns)

def find_base_path(mouse,date,root):
    mouse_path = Path(root) / f"sub-{mouse}" 

    for folder in mouse_path.iterdir():
        if folder.is_dir() and date in folder.name:
            print(f"Found folder: {folder}")
            base_path = folder
    return base_path

def load_settings(base_path):
    settings_path = Path(base_path) / "behav/session-settings/"
    settings_file = list(settings_path.glob("*.json"))[0]
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

    sess_events_data = events_data[~events_data.index.duplicated(keep='first')]
    sess_lick_data = lick_data[~lick_data.index.duplicated(keep='first')]
    sess_treadmill_data = treadmill_data[~treadmill_data.index.duplicated(keep='first')]
    sess_position_data = position_data[~position_data.index.duplicated(keep='first')]
    sess_reward_data = rewards_data[~rewards_data.index.duplicated(keep='first')]
    sess_buffer_data = buffer_data[~buffer_data.index.duplicated(keep='first')]

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

    return sess_dataframe

def get_event_parsed(sess_dataframe):

    lick_position = sess_dataframe['Position'].values[sess_dataframe['Licks'].values > 0]
    lick_times = sess_dataframe.index[sess_dataframe['Licks'].values > 0]
    reward_times = sess_dataframe.index[sess_dataframe['Rewards'].notna()]
    reward_positions = sess_dataframe['Position'].values[sess_dataframe['Rewards'].notna()]
    release_events = sess_dataframe[sess_dataframe['Events'].str.contains('release', na=False) & ~sess_dataframe['Events'].str.contains('odour0', na=False)]
    release_times = release_events.index
    release_positions = release_events['Position'].values

    #some times the mouse rocks back and forth triggering multiple release events at similar positions, so we filter these out by only keeping releases that are at least 3 units apart in position
    diff_ix = np.where(np.diff(release_positions)<3)[0]+1
    release_events = release_events.drop(release_events.index[diff_ix])
    release_times = release_events.index
    release_positions = release_events['Position'].values

    return lick_position, lick_times, reward_times, reward_positions, release_events, release_times, release_positions

def parse_rew_lms(ses_settings):
    rew_odour = []
    rew_texture = []
    non_rew_odour = []
    non_rew_texture = []

    for i in ses_settings['trial']['landmarks']:
        for j in i:
            if j['rewardSequencePosition'] > -1:
                rew_odour.append(j['odour'])
                rew_texture.append(j['texture'])
            else:
                
                non_rew_odour.append(j['odour'])
                non_rew_texture.append(j['texture'])

    rew_odour = np.unique(rew_odour)
    rew_texture = np.unique(rew_texture)
    non_rew_odour = np.unique(non_rew_odour)
    non_rew_texture = np.unique(non_rew_texture)
    non_rew_odour = non_rew_odour[non_rew_odour != 'odour0']
    non_rew_texture = non_rew_texture[non_rew_texture != 'grey']
    return rew_odour, rew_texture, non_rew_odour, non_rew_texture


def plot_ethogram(sess_dataframe,ses_settings):
    lick_position = sess_dataframe['Position'].values[sess_dataframe['Licks'].values > 0]
    lick_times = sess_dataframe.index[sess_dataframe['Licks'].values > 0]
    reward_times = sess_dataframe.index[sess_dataframe['Rewards'].notna()]
    reward_positions = sess_dataframe['Position'].values[sess_dataframe['Rewards'].notna()]
    release_events = sess_dataframe[sess_dataframe['Events'].str.contains('release', na=False) & ~sess_dataframe['Events'].str.contains('odour0', na=False)]
    release_events = release_events[release_events['Position'].notna()]
    release_times = release_events.index
    release_positions = release_events['Position'].values
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

def calc_hit_fa(sess_dataframe,ses_settings):

    lick_position, lick_times, reward_times, reward_positions, release_events, release_times, release_positions = get_event_parsed(sess_dataframe)

    rew_odour, rew_texture, non_rew_odour, non_rew_texture = parse_rew_lms(ses_settings)

    target_positions, distractor_positions, target_id, distractor_id, was_target, lm_id = find_targets_distractors(sess_dataframe,ses_settings)

    licked_target = np.zeros(len(target_positions))
    for idx, pos in enumerate(target_positions):
        if np.any((lick_position > pos) & (lick_position < (pos + 3))):
            licked_target[idx] = 1

    licked_distractor = np.zeros(len(distractor_positions))
    for idx, pos in enumerate(distractor_positions):
        if np.any((lick_position > pos) & (lick_position < (pos + 3))):
            licked_distractor[idx] = 1

    licked_all = np.zeros(len(release_positions))
    rewarded_all = np.zeros(len(release_positions))
    for idx, pos in enumerate(release_positions):
        if np.any((lick_position > pos) & (lick_position < (pos + 3))):
           licked_all[idx] = 1
        if np.any((reward_positions > pos) & (reward_positions < (pos + 3))):
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

def find_targets_distractors(sess_dataframe,ses_settings):
    
    lick_position, lick_times, reward_times, reward_positions, release_events, release_times, release_positions = get_event_parsed(sess_dataframe)
    rew_odour, rew_texture, non_rew_odour, non_rew_texture = parse_rew_lms(ses_settings)

    target_releases = pd.DataFrame()
    target_id = []
    for i in range(len(rew_odour)):
        test_str = 'release: '+rew_odour[i]
        new_events = release_events[release_events['Events'].str.fullmatch(test_str, na=False)]
        target_releases = pd.concat([target_releases, new_events])
        target_id.extend([i] * len(new_events))
        target_positions = target_releases['Position'].values
        target_positions = target_positions[~np.isnan(target_positions)]

    distractor_releases = pd.DataFrame()
    distractor_id = []
    for i in range(len(non_rew_odour)):
        test_str = 'release: '+non_rew_odour[i]
        new_events = release_events[release_events['Events'].str.fullmatch(test_str, na=False)]
        distractor_releases = pd.concat([distractor_releases, new_events])
        distractor_id.extend([i] * len(new_events))
        distractor_positions = distractor_releases['Position'].values
        distractor_positions = distractor_positions[~np.isnan(distractor_positions)]
    
    was_target = np.zeros(len(release_positions))
    lm_id = np.zeros(len(release_positions))
    for idx, pos in enumerate(release_positions):
        if pos in target_positions:
            was_target[idx] = 1
            lm_id[idx] = target_id[np.where(target_positions == pos)[0][0]]
        else:
            was_target[idx] = 0
            lm_id[idx] = distractor_id[np.where(distractor_positions == pos)[0][0]] + len(rew_odour)  #offset distractor IDs

    return target_positions, distractor_positions, target_id, distractor_id, was_target, lm_id

def calc_transition_matrix(sess_dataframe,ses_settings):
    
    target_positions, distractor_positions, target_id, distractor_id, was_target, lm_id = find_targets_distractors(sess_dataframe,ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)

    lick_sequence = lm_id[licked_all==1]
    num_landmarks = int(np.max(lm_id)) + 1

    transition_matrix = np.zeros((num_landmarks, num_landmarks))
    lick_tm = np.zeros((num_landmarks, num_landmarks))
    for i in range(len(lick_sequence)-1):
        current_lm = int(lick_sequence[i])
        next_lm = int(lick_sequence[i+1])
        lick_tm[current_lm, next_lm] += 1
    for i in range(len(lm_id)-1):
        current_lm = int(lm_id[i])
        next_lm = int(lm_id[i+1])
        transition_matrix[current_lm, next_lm] += 1

    return transition_matrix, lick_tm

def calc_conditional_matrix(sess_dataframe,ses_settings):


    target_positions, distractor_positions, target_id, distractor_id, was_target, lm_id = find_targets_distractors(sess_dataframe,ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)

    transition_prob = np.zeros((np.unique(target_id).shape[0], np.unique(lm_id).shape[0]))
    control_prob = np.zeros((np.unique(target_id).shape[0], np.unique(lm_id).shape[0])) 
    licked_lm_ix = np.where(licked_all == 1)[0]
    controlled_lm_ix = np.where(was_target == 1)[0]

    for g in range(np.unique(target_id).shape[0]):
        rewards = np.intersect1d(np.where(rewarded_all == 1)[0],np.where(lm_id == g)[0])
        for i,reward in enumerate(rewards):
            if i == len(rewards)-1:
                break
            next_lick_index = licked_lm_ix[licked_lm_ix>reward][0]
            next_control_index = controlled_lm_ix[controlled_lm_ix>reward][0]
            next_lm = lm_id[next_lick_index].astype(int)
            next_control_lm = lm_id[next_control_index].astype(int)
            transition_prob[g,next_lm] += 1
            control_prob[g,next_control_lm] += 1
    return transition_prob, control_prob

def plot_transition_matrix(sess_dataframe,ses_settings):

    transition_matrix, lick_tm = calc_transition_matrix(sess_dataframe,ses_settings)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(transition_matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.clim(0, np.max(transition_matrix))
    plt.title('Stimulus Transition Matrix')
    plt.xlabel('Next Landmark ID')
    plt.ylabel('Current Landmark ID')

    plt.subplot(1, 2, 2)
    plt.imshow(lick_tm, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.clim(0, np.max(lick_tm))
    plt.title('Lick Transition Matrix')
    plt.xlabel('Next Landmark ID')
    plt.ylabel('Current Landmark ID')

    plt.tight_layout()
    plt.show()

def plot_conditional_matrix(sess_dataframe,ses_settings):

    transition_prob, control_prob = calc_conditional_matrix(sess_dataframe,ses_settings)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(transition_prob, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.clim(0, np.max(transition_prob))
    plt.title('Transition Probability Matrix (Licked)')
    plt.xlabel('Next Landmark ID')
    plt.ylabel('Goal ID')

    plt.subplot(1, 2, 2)
    plt.imshow(control_prob, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.clim(0, np.max(control_prob))
    plt.title('Control Probability Matrix (All Targets)')
    plt.xlabel('Next Landmark ID')
    plt.ylabel('Goal ID')

    plt.tight_layout()
    plt.show()

def plot_switch_stay(sess_dataframe,ses_settings):

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

    target_positions, distractor_positions, target_id, distractor_id, was_target, lm_id = find_targets_distractors(sess_dataframe,ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)

    #reshape licked_all into 10 columns and the appropriate number of rows
    if licked_all.shape[0] % 10 != 0:
        #extend the array to make it divisible by 10
        licked_all = np.pad(licked_all, (0, 10 - (licked_all.shape[0] % 10)), 'constant')
    licked_all_reshaped = licked_all.reshape(np.round(licked_all.shape[0] / 10).astype(int), 10)
    if was_target.shape[0] % 10 != 0:
        #extend the array to make it divisible by 10
        was_target = np.pad(was_target, (0, 10 - (was_target.shape[0] % 10)), 'constant')
    was_target_reshaped = was_target.reshape(np.round(was_target.shape[0] / 10).astype(int), 10)
    if lm_id.shape[0] % 10 != 0:
        #extend the array to make it divisible by 10
        lm_id = np.pad(lm_id, (0, 10 - (lm_id.shape[0] % 10)), 'constant')
    lm_id_reshaped = lm_id.reshape(np.round(lm_id.shape[0] / 10).astype(int), 10)

    plt.figure(figsize=(10,4))
    plt.subplot(2, 1, 1)
    plt.imshow(was_target_reshaped, aspect='auto', cmap='viridis', interpolation='none')
    plt.clim(0, 1)
    plt.title('Landmark ID (Full Corridor)')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.imshow(licked_all_reshaped, aspect='auto', cmap='viridis', interpolation='none')
    plt.clim(0, 1)
    plt.title('Licked All (Full Corridor)')
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
        hit_rate_window[i] = np.sum(licked_all[i:i+window][was_target[i:i+window]==1])/all_window_goals
        false_alarm_rate_window[i] = np.sum(licked_all[i:i+window][was_target[i:i+window]==0])/all_window_distractors

    plt.figure(figsize=(10,2))
    plt.plot(hit_rate_window, label='Hit Rate', color='g')
    plt.plot(false_alarm_rate_window, label='False Alarm Rate', color='r')
    plt.xlabel('Landmark')
    plt.ylabel('Rate')
    plt.legend()
    plt.title('Sliding window Hit and False Alarm rates')
    plt.show()

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

from aeon.io.reader import Csv, Reader
import aeon.io.api as aeon
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import json
import re

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
    with pd.option_context("future.no_silent_downcasting", True):
        sess_dataframe['Licks'] = sess_dataframe['Licks'].fillna(False).astype(bool) 
        # make it true or false by making nan to false

    return sess_dataframe

def get_event_parsed(sess_dataframe, ses_settings):

    lick_position = sess_dataframe['Position'].values[sess_dataframe['Licks'].values > 0]
    lick_times = sess_dataframe.index[sess_dataframe['Licks'].values > 0]
    reward_times = sess_dataframe.index[sess_dataframe['Rewards'].notna()]
    reward_positions = sess_dataframe['Position'].values[sess_dataframe['Rewards'].notna()]

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


def plot_ethogram(sess_dataframe,ses_settings):
    lick_position = sess_dataframe['Position'].values[sess_dataframe['Licks'].values > 0]
    lick_times = sess_dataframe.index[sess_dataframe['Licks'].values > 0]
    reward_times = sess_dataframe.index[sess_dataframe['Rewards'].notna()]
    reward_positions = sess_dataframe['Position'].values[sess_dataframe['Rewards'].notna()]
    release_df = estimate_release_events(sess_dataframe, ses_settings)
    release_times = release_df.index.tolist() # time
    release_positions = release_df["Position"].tolist()

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
    lm_size = ses_settings['trial']['landmarks'][0][0]['size']

    lick_position, lick_times, reward_times, reward_positions, release_df = get_event_parsed(sess_dataframe, ses_settings)

    rew_odour, rew_texture, non_rew_odour, non_rew_texture = parse_rew_lms(ses_settings)

    target_positions, distractor_positions, target_id, distractor_id, was_target, lm_id = find_targets_distractors(sess_dataframe,ses_settings)

    licked_target = np.zeros(len(target_positions))
    for idx, pos in enumerate(target_positions):
        if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
            licked_target[idx] = 1

    licked_distractor = np.zeros(len(distractor_positions))
    for idx, pos in enumerate(distractor_positions):
        if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
            licked_distractor[idx] = 1

    licked_all = np.zeros(len(release_df))
    rewarded_all = np.zeros(len(release_df))
    release_positions = release_df['Position'].to_numpy()
    for idx, pos in enumerate(release_positions):
        if np.any((lick_position > pos) & (lick_position < (pos + lm_size))):
           licked_all[idx] = 1
        if np.any((reward_positions > pos) & (reward_positions < (pos + lm_size))):
           rewarded_all[idx] = 1
    
    #sometimes the VR drops the first release event, check for that and add a 0 as a first element if needed
    # first_release = ses_settings['trial']['landmarks'][0][0]['odour']
    # if not first_release in release_events['Events'].values[0]:
    #     licked_all = np.insert(licked_all, 0, 0)
    #     rewarded_all = np.insert(rewarded_all, 0, 0)

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

        distractor_id.extend([i] * len(pos))
        distractor_positions.extend(pos)
    
    all_release_positions = release_df["Position"].tolist()
    was_target = np.zeros(len(all_release_positions))
    lm_id = np.zeros(len(all_release_positions))
    for idx, pos in enumerate(all_release_positions):
        if pos in target_positions:
            was_target[idx] = 1
            lm_id[idx] = target_id[np.where(np.isclose(target_positions, pos))[0][0]]
        else:
            was_target[idx] = 0
            lm_id[idx] = distractor_id[np.where(np.isclose(distractor_positions, pos))[0][0]] + len(rew_odour)  #offset distractor IDs

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


    target_positions, distractor_positions, target_id, distractor_id, was_target, lm_id = find_targets_distractors(sess_dataframe,ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)
    ideal_licks = get_ideal_performance(sess_dataframe,ses_settings)

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
            next_lick_index = licked_lm_ix[licked_lm_ix>reward][0]
            next_control_index = controlled_lm_ix[controlled_lm_ix>reward][0]
            next_lm = lm_id[next_lick_index].astype(int)
            next_control_lm = lm_id[next_control_index].astype(int)
            transition_prob[g,next_lm] += 1
            control_prob[g,next_control_lm] += 1
    
    for g in range(np.unique(target_id).shape[0]):
        ideal_rewards = np.intersect1d(np.where(ideal_licks == 1)[0],np.where(lm_id == g)[0])
        for i,ideal_reward in enumerate(ideal_rewards):
            if i == len(ideal_rewards)-1:
                break
            next_ideal_index = np.where(ideal_licks == 1)[0][np.where(np.where(ideal_licks == 1)[0]>ideal_reward)[0][0]]
            next_ideal_lm = lm_id[next_ideal_index].astype(int)
            ideal_prob[g,next_ideal_lm] += 1
    return transition_prob, control_prob, ideal_prob

def calc_stable_conditional_matrix(sess_dataframe,ses_settings):

    goals, lm_ids = parse_stable_goal_ids(ses_settings)
    hit_rate, fa_rate, d_prime, licked_target, licked_distractor, licked_all, rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)

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
            next_lick_index = licked_lm_ix[licked_lm_ix>reward][0]
            next_control_index = controlled_lm_ix[controlled_lm_ix>reward][0]
            next_lm = all_lms[next_lick_index].astype(int)
            next_control_lm = all_lms[next_control_index].astype(int)
            transition_prob[g,next_lm] += 1
            control_prob[g,next_control_lm] += 1

    for g in range(np.unique(goals).shape[0]):
        next_goal = goals[g+1] if g+1 < len(goals) else goals[0]
        ideal_prob[g,next_goal] += 1

    return transition_prob, control_prob, ideal_prob

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
        x_labels = ['A', 'B', 'C', 'Dist']
    elif n_goals == 4:
        y_labels = ['A', 'B', 'C', 'D']
        x_labels = ['A', 'B', 'C', 'D', 'Dist']
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

def calc_seq_fraction(sess_dataframe,ses_settings,test='transition'):
    
    transition_prob, control_prob, ideal_prob = calc_conditional_matrix(sess_dataframe,ses_settings)

    if test == 'transition':
        test_prob = transition_prob
    elif test == 'control':
        test_prob = control_prob
    elif test == 'ideal':
        test_prob = ideal_prob
    else:
        raise ValueError("Invalid test type. Choose from 'transition', 'control', or 'ideal'.")

    ab_prob = test_prob[0,1]
    ac_prob = test_prob[0,2]
    bc_prob = test_prob[1,2]
    ba_prob = test_prob[1,0]
    ca_prob = test_prob[2,0]
    cb_prob = test_prob[2,1]

    perf_a = ab_prob / (ab_prob + ac_prob)
    perf_b = bc_prob / (bc_prob + ba_prob)
    perf_c = ca_prob / (ca_prob + cb_prob)

    performance = np.mean([perf_a, perf_b, perf_c])

    return performance, perf_a, perf_b, perf_c

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

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1)
    plt.bar([0, 1], [correct_seq, incorrect_seq], color=['green', 'red'])
    plt.title('Lick Sequence')
    plt.xticks([0, 1], ['Correct', 'Incorrect'])
    plt.subplot(1, 3, 2)
    plt.bar([0, 1], [ctrl_correct_seq, ctrl_incorrect_seq], color=['green', 'red'])
    plt.title('Control Sequence')
    plt.xticks([0, 1], ['Correct', 'Incorrect'])
    plt.subplot(1, 3, 3)
    plt.bar([0, 1], [ideal_correct_seq, ideal_incorrect_seq], color=['green', 'red'])
    plt.title('Ideal Sequence')
    plt.xticks([0, 1], ['Correct', 'Incorrect'])
    plt.tight_layout()
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
    if was_target.shape[0] % 10 != 0:
        #extend the array to make it divisible by 10
        was_target = np.pad(was_target, (0, 10 - (was_target.shape[0] % 10)), 'constant')
    was_target_reshaped = was_target.reshape(np.round(was_target.shape[0] / 10).astype(int), 10)
    if all_lms.shape[0] % 10 != 0:
        #extend the array to make it divisible by 10
        all_lms = np.pad(all_lms, (0, 10 - (all_lms.shape[0] % 10)), 'constant')
    all_lms_reshaped = all_lms.reshape(np.round(all_lms.shape[0] / 10).astype(int), 10)

    plt.figure(figsize=(10,4))
    plt.subplot(2, 1, 1)
    plt.imshow(was_target_reshaped, aspect='auto', cmap='viridis', interpolation='none')
    plt.clim(0, len(goals))
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

def give_state_id(sess_dataframe,ses_settings):

    goals,lm_ids = parse_stable_goal_ids(ses_settings)
    laps_needed = calc_laps_needed(ses_settings)
    hit_rate, fa_rate,d_prime, licked_target, licked_distractor, licked_all,rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)
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

    return state_id


def plot_licks_per_state(sess_dataframe, ses_settings):
    state_id = give_state_id(sess_dataframe,ses_settings)

    hit_rate, fa_rate,d_prime, licked_target, licked_distractor, licked_all,rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)
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
        state1_laps = licked_all[np.where(state_id == 0)[0],:]
        state2_laps = licked_all[np.where(state_id == 1)[0],:]
        state3_laps = licked_all[np.where(state_id == 2)[0],:]

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

def calc_speed_per_lap(sess_dataframe, ses_settings):
    num_laps, sess_dataframe = divide_laps(sess_dataframe, ses_settings)
    laps_needed = calc_laps_needed(ses_settings)
    #max position is the max of all positions where lap id is 0
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


def calc_sw_state_ratio(sess_dataframe, ses_settings):
    state_id = give_state_id(sess_dataframe,ses_settings)
    laps_needed = calc_laps_needed(ses_settings)
    num_laps, sess_dataframe = divide_laps(sess_dataframe, ses_settings)
    hit_rate, fa_rate,d_prime, licked_target, licked_distractor, licked_all,rewarded_all = calc_hit_fa(sess_dataframe,ses_settings)
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
        state_dprime = np.zeros([num_laps,10])
        for i in range(num_laps):
            if i < window:
                state1_sw[i] = np.nan
                state2_sw[i] = np.nan

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
    if first_release != result[0][3]:
        result = [[pd.NaT, np.nan, -1, first_release]] + result

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
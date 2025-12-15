from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import sys
import os
import yaml
import copy
import scipy.stats as stats
from scipy.stats import norm
from scipy.interpolate import make_interp_spline
import scipy.signal as signal
import seaborn as sns
from statistics import NormalDist
import wesanderson
from cycler import cycler
import palettes
import importlib

from analysis import neural_analysis_helpers
from cellTV import cellTV_functions as cellTV
importlib.reload(neural_analysis_helpers)
importlib.reload(cellTV)


hfs_palette = palettes.met_brew('Hiroshige',n=123, brew_type="continuous")
rev_hfs = hfs_palette[::-1]
tm_palette = palettes.met_brew('Tam',n=123, brew_type="continuous")
tm_palette = tm_palette[::-1]
color_scheme = wesanderson.film_palette('Darjeeling Limited',palette=0)
custom_cycler = cycler(color=color_scheme)

def find_base_path(mouse,date):
    data_dir = '/Volumes/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/' + mouse
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    sessions = [s for s in folders if date in s]
    if not sessions:
        data_dir = data_dir + '/TrainingData'
        date = date[2:]
        folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        sessions = [s for s in folders if date in s]
        if not sessions:
            print('No Data Found')
        else:
            print('Training Only')
            num_sessions = len(sessions)
            if num_sessions > 1:
                print('Multiple Training Sessions:')
                print(sessions)
                #ask for keyboard input
                session = int(input('Please select a session:'))
                base_path = os.path.join(data_dir, sessions[session])
                # print(base_path)
            else:
                base_path = os.path.join(data_dir, sessions[0])
                # print(base_path)
    else:
        print('Training and Imaging')
        base_path = os.path.join(data_dir, sessions[0],'behav',date[2:])
        # print(base_path)
    return base_path

def find_base_path_npz(mouse,date):
    data_dir = '/Volumes/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/' + mouse
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    sessions = [s for s in folders if date in s]
    base_path = os.path.join(data_dir, sessions[0])

    return base_path

def load_session(base_path):    
    # load position log data
    path = base_path + '/position_log.csv'
    data = pd.read_csv(path)
    return data

def load_session_npz(base_path):
    # load position log data
    data = np.load(os.path.join(base_path, 'behaviour_data.npz'))
    return data

def load_config(base_path):
    path = base_path + '/config.yaml'
    with open(path) as file:
        options = yaml.load(file, Loader=yaml.FullLoader)
    return options

def create_session_struct(data,options):
    position = data['Position']
    total_dist = data['TotalRunDistance']
    time = data['Time'].values
    licks = np.where(data['Event'] == 'challenged')[0]
    rewards = np.where(data['Event'] == 'rewarded')[0]
    assist_rewards = np.where(data['Event'] == 'assist-rewarded')[0]
    manual_rewards = np.where(data['Event'] == 'manually-rewarded')[0]
    goals = np.array(options['flip_tunnel']['goals']) 
    landmarks = np.array(options['flip_tunnel']['landmarks']) 
    tunnel_length = options['flip_tunnel']['length']
    lick_threshold = options['flip_tunnel']['speed_limit']
    # lick_threshold = 7
    print('Speed threshold:', lick_threshold)

    #set position to nan if it is -1 and then interpolate
    position = np.array(position)
    position[position == -1] = np.nan
    position = pd.Series(position)
    position = position.interpolate(method='linear', limit_direction='both')
    position = np.array(position)
    
    #same with total distance
    total_dist = np.array(total_dist)
    total_dist[total_dist == -1] = np.nan
    total_dist = pd.Series(total_dist)
    total_dist = total_dist.interpolate(method='linear', limit_direction='both')
    total_dist = np.array(total_dist)

    if 'Speed' in data:
        speed = data['Speed']
    else:
        # calculate speed
        window = 10
        position_diff = [total_dist[i+window]-total_dist[i] for i in range(len(total_dist)-window)]
        time_diff = [time[i+window]-time[i] for i in range(len(time)-window)]
        speed = np.array(position_diff)/np.array(time_diff)
        speed = np.append(speed, np.ones(window)*speed[-1])

    # create session struct 
    session = {'position': position,
               'total_dist': total_dist,
               'time': time, 
               'licks': licks, 
               'rewards': rewards, 
               'assist_rewards': assist_rewards, 
               'manual_rewards': manual_rewards, 
               'goals': goals, 
               'landmarks': landmarks, 
               'tunnel_length': tunnel_length,
                'lick_threshold': lick_threshold,
                'speed': speed}
    return session

def create_session_struct_npz(data,options):
    position = data['position']
    total_dist = data['distance']
    licks = np.where(data['licks'])[0]
    rewards = np.where(data['rewards'])[0]
    speed = data['speed']

    goals = np.array(options['flip_tunnel']['goals']) 
    landmarks = np.array(options['flip_tunnel']['landmarks']) 
    tunnel_length = options['flip_tunnel']['length']
    lick_threshold = options['flip_tunnel']['speed_limit']

    session = {'position': position,
               'total_dist': total_dist, 
               'licks': licks, 
               'rewards': rewards, 
               'goals': goals, 
               'landmarks': landmarks, 
               'tunnel_length': tunnel_length,
                'lick_threshold': lick_threshold,
                'speed': speed}
    
    return session  

def get_lap_idx(session):
    # get lap idx
    flip_ix = signal.find_peaks(session['position'], height=session['tunnel_length']-1,distance=100)[0]
    if (session['position'][0] - session['landmarks'][-1,1]) < (session['position'][0] - session['landmarks'][0,0]):
        flip_ix = flip_ix[1:]  # Remove the first peak index - the mouse accidentally moved backwards first
        
    if len(flip_ix) > 0:
        # a lap is between two flips
        lap_idx = np.zeros(len(session['position']))
        for i in range(len(flip_ix)-1):
            lap_idx[flip_ix[i]:flip_ix[i+1]] = i+1
        lap_idx[flip_ix[-1]:] = len(flip_ix)

        session['lap_idx'] = lap_idx
        session['num_laps'] = len(flip_ix) + 1
    else:
        # the session didn't have any flips
        session['lap_idx'] = np.ones(len(session['position']))
        session['num_laps'] = 1


    return session

def get_lm_idx(session):
    # get landmark idx
    lm_idx = np.zeros(len(session['position']))
    for i in range(len(session['landmarks'])):
        lm = session['landmarks'][i]
        lm_entry = np.where((session['position'] > lm[0]) & (session['position'] < lm[1]))[0]
        lm_idx[lm_entry] = i+1

    session['lm_idx'] = lm_idx

    return session

def calc_laps_needed(session):
    laps_needed = 1
    for i in range(len(session['goal_idx'])-1):
        if session['goal_idx'][i+1] - session['goal_idx'][i] < 0:
            laps_needed += 1
    if session['goal_idx'][0] - session['goal_idx'][-1] > 0:
        laps_needed -= 1

    session['laps_needed'] = laps_needed

    return session

def threshold_licks(session):
    # threshold licks
    # lick_threshold = 10
    
    lick_threshold = session['lick_threshold']
    threshold_licks = session['licks'][np.where(session['speed'][session['licks']] < lick_threshold)[0]]

    session['thresholded_licks'] = threshold_licks

    return session

def calculate_lick_rate(session):
    # calculate lick rate in a sliding window
    window = 100
    lick_rate = np.zeros(len(session['time']))
    for i in range(len(session['time'])-window):
        time_diff = session['time'][i+window]-session['time'][i]
        lick_num = len(np.where((session['licks'] > i) & (session['licks'] < i+window))[0])
        lick_rate[i] = lick_num/time_diff
    
    session['lick_rate'] = lick_rate

    return session

def get_licks_per_lap(session):
    #save lick indices for each lap in a dictionary
    lick_frames = {}
    lick_positions = {}
    for i in range(session['num_laps']):
        if session['num_laps'] == 1:
            lap_ix = np.where(session['lap_idx'] == i+1)[0]
        else:
            lap_ix = np.where(session['lap_idx'] == i)[0]
        licks_per_lap_ix = np.intersect1d(lap_ix,session['thresholded_licks'])
        lick_frames[i] = licks_per_lap_ix
        lick_positions[i] = session['position'][licks_per_lap_ix]

    session['licks_per_lap'] = lick_positions
    session['licks_per_lap_frames'] = lick_frames

    return session

def get_licked_lms(session):
    # get licked landmarks
    licked_lms = np.zeros((session['num_laps'],len(session['landmarks'])))
    
    for i in range(session['num_laps']):
        lap_idx = np.where(session['lap_idx'] == i)[0]
        for j in range(len(session['landmarks'])):
            lm = np.where(session['lm_idx'] == j+1)[0]
            target_ix = np.intersect1d(lap_idx,lm)
            #if session['thresholded_licks'] exists, use that
            if 'thresholded_licks' in session:
                target_licks = np.intersect1d(target_ix,session['thresholded_licks'])
            #otherwise use all licks
            else:
                target_licks = np.intersect1d(target_ix,session['licks'])
            if len(target_licks) > 0:
                licked_lms[i,j] = 1
            else:
                licked_lms[i,j] = 0

    session['licked_lms'] = licked_lms

    return session

def get_rewarded_lms(session):
    # get rewarded landmarks
    rewarded_lms = np.zeros((session['num_laps'],len(session['landmarks'])))
    
    for i in range(session['num_laps']):
        lap_idx = np.where(session['lap_idx'] == i)[0]
        for j in range(len(session['landmarks'])):
            lm = np.where(session['lm_idx'] == j+1)[0]
            target_ix = np.intersect1d(lap_idx,lm)
            target_rewards = np.intersect1d(target_ix,session['rewards'])
            if len(target_rewards) > 0:
                rewarded_lms[i,j] = 1
            else:
                rewarded_lms[i,j] = 0

    session['rewarded_lms'] = rewarded_lms

    return session

def give_lap_state_id(session): 
    state_id = np.zeros((session['num_laps']))
    state_id[0] = 0

    if session['laps_needed'] == 2: #either goal A, B or C defines the first state change, either B, C or D defines the second state change 
        if session['goal_idx'][1] - session['goal_idx'][2] > 0:
            defining_goal_1 = 1
            if session['goal_idx'][2] - session['goal_idx'][3] > 0:
                defining_goal_2 = 2
            else:
                defining_goal_2 = 3
        elif session['goal_idx'][2] - session['goal_idx'][3] > 0:
            defining_goal_1 = 2
            defining_goal_2 = 3
        else:
            defining_goal_1 = 0
            if session['goal_idx'][1] - session['goal_idx'][2] > 0:
                defining_goal_2 = 1
            elif session['goal_idx'][2] - session['goal_idx'][3] > 0:
                defining_goal_2 = 2
            else:
                defining_goal_2 = 3


        print('Defining goal 1 = ', defining_goal_1)
        print('Defining goal 2 = ', defining_goal_2)

        for i in range(0,session['num_laps']-1):
            if session['rewarded_lms'][i,session['goal_idx'][defining_goal_1]] == 1:
                state_id[i+1] = 1

            if state_id[i]==1 and session['rewarded_lms'][i,session['goal_idx'][defining_goal_2]] == 1:
                state_id[i+1] = 0
            elif state_id[i]==1 and session['rewarded_lms'][i,session['goal_idx'][defining_goal_2]] == 0:
                state_id[i+1] = state_id[i] 
    elif session['laps_needed'] == 3: #only one combination possible
        for i in range(0,session['num_laps']-1):
            if state_id[i]==0 and session['rewarded_lms'][i,session['goal_idx'][1]] == 1:
                state_id[i+1] = 1
            
            if state_id[i]==1 and session['rewarded_lms'][i,session['goal_idx'][2]] == 1:
                state_id[i+1] = 2
            elif state_id[i]==1 and session['rewarded_lms'][i,session['goal_idx'][2]] == 0:
                state_id[i+1] = state_id[i]
            
            if state_id[i]==2 and session['rewarded_lms'][i,session['goal_idx'][3]] == 1:
                state_id[i+1] = 0
            elif state_id[i]==2 and session['rewarded_lms'][i,session['goal_idx'][3]] == 0:
                state_id[i+1] = state_id[i]
    
    session['state_id'] = state_id

    return session

def get_lms_visited(options, session):
    # Calculate number of landmarks visited
    if len(np.where(session['landmarks'][:,0] < session['position'][-1])[0]) != 0:
        last_landmark = len(np.where(session['landmarks'][:,0] < session['position'][-1])[0]) # find the last landmark that was run through
    else:
        last_landmark = len(session['landmarks'])  # TODO: confirm
    
    num_lms = len(session['landmarks'])*(session['num_laps']-1) + last_landmark 
   
    lm_ids =  np.array(options['flip_tunnel']['landmarks_sequence'])
    all_lms = np.array([])  # landmark ids
    for i in range(session['num_laps']):
        all_lms = np.append(all_lms, lm_ids)
    all_lms = all_lms.astype(int)[:num_lms]

    all_landmarks = session['landmarks']  
    for i in range(1, session['num_laps']):  
        all_landmarks = np.concatenate((all_landmarks, session['landmarks']), axis=0)
    all_landmarks = all_landmarks[:num_lms]  # landmark positions
    session['all_landmarks'] = all_landmarks
    session['all_lms'] = all_lms

    return session

def sw_state_ratio(session):
    
    if session['laps_needed'] == 2:
        window = 10
        state1_sw = np.zeros([session['num_laps'],len(session['landmarks'])])
        state2_sw = np.zeros([session['num_laps'],len(session['landmarks'])])
        state_diff_1 = np.zeros([session['num_laps'],len(session['landmarks'])])
        state_dprime = np.zeros([session['num_laps'],len(session['landmarks'])])
        for i in range(session['num_laps']):
            if i < window:
                state1_sw[i] = np.nan
                state2_sw[i] = np.nan

            else:
                lap_range = range(i-window, i)
                laps = session['licked_lms'][lap_range]
                state1_laps = laps[np.where(session['state_id'][lap_range] == 0)[0],:]
                state2_laps = laps[np.where(session['state_id'][lap_range] == 1)[0],:]

                state1_sw[i] = np.sum(state1_laps,axis=0)/state1_laps.shape[0]
                state2_sw[i] = np.sum(state2_laps,axis=0)/state2_laps.shape[0]
                state_diff_1[i] = abs(state1_sw[i]-state2_sw[i])

        sw_state_ratio_a = state_diff_1[:,session['goal_idx'][0]]
        sw_state_ratio_b = state_diff_1[:,session['goal_idx'][1]]
        sw_state_ratio_c = state_diff_1[:,session['goal_idx'][2]]
        sw_state_ratio_d = state_diff_1[:,session['goal_idx'][3]]

    elif session['laps_needed'] == 3:
        window = 10
        state1_sw = np.zeros([session['num_laps'],len(session['landmarks'])])
        state2_sw = np.zeros([session['num_laps'],len(session['landmarks'])])
        state3_sw = np.zeros([session['num_laps'],len(session['landmarks'])])
        state_diff_1 = np.zeros([session['num_laps'],len(session['landmarks'])])
        state_diff_2 = np.zeros([session['num_laps'],len(session['landmarks'])])
        state_diff_3 = np.zeros([session['num_laps'],len(session['landmarks'])])
        for i in range(session['num_laps']):
            if i < window:
                state1_sw[i] = np.nan
                state2_sw[i] = np.nan
                state3_sw[i] = np.nan

            else:
                lap_range = range(i-window, i)
                laps = session['licked_lms'][lap_range]
                state1_laps = laps[np.where(session['state_id'][lap_range] == 0)[0],:]
                state2_laps = laps[np.where(session['state_id'][lap_range] == 1)[0],:]
                state3_laps = laps[np.where(session['state_id'][lap_range] == 2)[0],:]

                state1_sw[i] = np.sum(state1_laps,axis=0)/state1_laps.shape[0]
                state2_sw[i] = np.sum(state2_laps,axis=0)/state2_laps.shape[0]
                state3_sw[i] = np.sum(state3_laps,axis=0)/state3_laps.shape[0]
                state_diff_1[i] = abs(state1_sw[i]-state2_sw[i])
                state_diff_2[i] = abs(state2_sw[i]-state3_sw[i])
                state_diff_3[i] = abs(state1_sw[i]-state3_sw[i])
    
        sw_state_ratio_a = (state_diff_1[:,session['goal_idx'][0]]+state_diff_3[:,session['goal_idx'][0]])/2
        sw_state_ratio_b = (state_diff_1[:,session['goal_idx'][1]]+state_diff_3[:,session['goal_idx'][1]])/2
        sw_state_ratio_c = (state_diff_1[:,session['goal_idx'][2]]+state_diff_2[:,session['goal_idx'][2]])/2
        sw_state_ratio_d = (state_diff_2[:,session['goal_idx'][3]]+state_diff_3[:,session['goal_idx'][3]])/2

    sw_state_ratio = np.nanmean([sw_state_ratio_a,sw_state_ratio_b,sw_state_ratio_c,sw_state_ratio_d],axis=0)


    session['sw_state_ratio'] = sw_state_ratio
    session['sw_state_ratio_a'] = sw_state_ratio_a
    session['sw_state_ratio_b'] = sw_state_ratio_b
    session['sw_state_ratio_c'] = sw_state_ratio_c
    session['sw_state_ratio_d'] = sw_state_ratio_d

    return session


def calc_decel_per_lap_pre7(session, dt=1/45):
    actual_num_laps = np.round((len(session['all_lms']) // session['num_landmarks']))
    _, lm_exit_idx = neural_analysis_helpers.get_lm_entry_exit(session)
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
        decel_per_bin = np.zeros((actual_num_laps, bins))

        for i, idx in enumerate(lap_change_idx):
            lap_idx = np.arange(x, idx+1)

            speed_per_lap = session['speed'][lap_idx]
            pos_per_lap = session['position'][lap_idx] 

            # acceleration
            decel = np.gradient(speed_per_lap) / dt
            # decel = np.where(accel < 0, accel, np.nan)

            decel_per_bin[i, :], bin_edges, _ = stats.binned_statistic(pos_per_lap, decel, bins=bins)
        
            x = idx + 1

    av_decel_per_bin = np.nanmean(decel_per_bin, axis=0)
    std_decel_per_bin = np.nanstd(decel_per_bin, axis=0)
    sem_decel_per_bin = std_decel_per_bin / np.sqrt(actual_num_laps)

    session['decel_per_bin'] = av_decel_per_bin
    session['sem_decel_per_bin'] = sem_decel_per_bin

    return session


def calc_decel_per_lap(session):
    bins = 120
    bin_edges = np.linspace(0, session['position'].max(), bins+1)
    
    decel_per_bin = np.zeros((session['num_laps'], bins))
    
    for i in range(session['num_laps']):
        lap_idx = np.where(session['lap_idx'] == i)[0]
        
        lap_pos = session['position'][lap_idx]
        lap_speed = session['speed'][lap_idx]
        
        # Acceleration (derivative wrt time or samples)
        lap_accel = np.gradient(lap_speed, 1/45)
        
        # Keep only deceleration values (negative accel)
        lap_decel = np.where(lap_accel < 0, lap_accel, np.nan)
        
        # Bin data by position
        bin_ix = np.digitize(lap_pos, bin_edges)
        for j in range(bins):
            # Average speed per bin
            # Average deceleration per bin (negative values)
            decel_per_bin[i, j] = np.nanmean(lap_decel[bin_ix == j])
    
    # Across laps: mean and SEM
    av_decel_per_bin = np.nanmean(decel_per_bin, axis=0)
    sem_decel_per_bin = np.nanstd(decel_per_bin, axis=0) / np.sqrt(session['num_laps'])

    session['decel_per_bin'] = decel_per_bin
    session['av_decel_per_bin'] = av_decel_per_bin
    session['sem_decel_per_bin'] = sem_decel_per_bin
    
    return session


def calc_speed_per_lap(session):
    bins = 120
    bin_edges = np.linspace(0, session['position'].max(), bins+1)
    speed_per_bin = np.zeros((session['num_laps'], bins))
    for i in range(session['num_laps']):
        lap_idx = np.where(session['lap_idx'] == i)[0]
        speed_per_lap = session['speed'][lap_idx]
        bin_ix = np.digitize(session['position'][lap_idx], bin_edges)
        for j in range(bins):
            speed_per_bin[i,j] = np.mean(speed_per_lap[bin_ix == j])
    av_speed_per_bin = np.nanmean(speed_per_bin, axis=0)
    std_speed_per_bin = np.nanstd(speed_per_bin, axis=0)
    sem_speed_per_bin = std_speed_per_bin/np.sqrt(session['num_laps'])
    binned_goals = np.digitize(session['goals'], bin_edges)
    binned_lms = np.digitize(session['landmarks'], bin_edges)

    session['speed_per_bin'] = speed_per_bin
    session['binned_goals'] = binned_goals
    session['binned_lms'] = binned_lms

    return session


def calc_speed_per_lap_pre7(session):
    actual_num_laps = np.round((len(session['all_lms']) // session['num_landmarks']) )

    _, lm_exit_idx = neural_analysis_helpers.get_lm_entry_exit(session)
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

    else:
        bins = 120
        speed_per_bin = np.zeros((actual_num_laps, bins))

        for i, idx in enumerate(lap_change_idx):
            lap_idx = np.arange(x, idx+1)

            speed_per_lap = session['speed'][lap_idx]
            pos_per_lap = session['position'][lap_idx] 

            speed_per_bin[i, :], bin_edges, _ = stats.binned_statistic(pos_per_lap, speed_per_lap, bins=bins)
        
            # TODO remove from loop? 
            goals_per_lap = session['goals'][i * 4 : (i + 1) * 4]
            lms_per_lap = session['landmarks'][i * session['num_landmarks'] : (i + 1) * session['num_landmarks']]
        
            x = idx + 1

        binned_goals = np.digitize(goals_per_lap, bin_edges)
        binned_lms = np.digitize(lms_per_lap, bin_edges)

    av_speed_per_bin = np.nanmean(speed_per_bin, axis=0)
    std_speed_per_bin = np.nanstd(speed_per_bin, axis=0)
    sem_speed_per_bin = std_speed_per_bin / np.sqrt(actual_num_laps)

    session['speed_per_bin'] = av_speed_per_bin
    session['sem_speed_per_bin'] = sem_speed_per_bin
    session['binned_goals'] = binned_goals
    session['binned_lms'] = binned_lms

    return session

def calc_speed_per_state(session):
    if session['laps_needed'] == 2:
        state1_laps = session['speed_per_bin'][np.where(session['state_id'] == 0)[0]]
        state2_laps = session['speed_per_bin'][np.where(session['state_id'] == 1)[0]]
    elif session['laps_needed'] == 3:
        state1_laps = session['speed_per_bin'][np.where(session['state_id'] == 0)[0]]
        state2_laps = session['speed_per_bin'][np.where(session['state_id'] == 1)[0]]
        state3_laps = session['speed_per_bin'][np.where(session['state_id'] == 2)[0]]
    
    state1_speed = np.nanmean(state1_laps, axis=0)
    state1_speed_sem = np.nanstd(state1_laps, axis=0)/np.sqrt(state1_laps.shape[0])
    state2_speed = np.nanmean(state2_laps, axis=0)
    state2_speed_sem = np.nanstd(state2_laps, axis=0)/np.sqrt(state2_laps.shape[0])
    if session['laps_needed'] == 3:
        state3_speed = np.nanmean(state3_laps, axis=0)
        state3_speed_sem = np.nanstd(state3_laps, axis=0)/np.sqrt(state3_laps.shape[0])
    
    session['state1_speed'] = state1_speed
    session['state1_speed_sem'] = state1_speed_sem
    session['state2_speed'] = state2_speed
    session['state2_speed_sem'] = state2_speed_sem
    if session['laps_needed'] == 3:
        session['state3_speed'] = state3_speed
        session['state3_speed_sem'] = state3_speed_sem
    
    return session

def get_active_goal(session):
    # get goal indices
    goal_idx = np.array([])
    if session['num_laps'] > 1:
        for goal in session['goals']:
            goal_idx = np.append(goal_idx, np.where(session['landmarks'] == goal)[0][0])
    else:
        for goal in session['goals']:
            matches = np.where(session['landmarks'][:session['num_landmarks']] == goal)[0]
            if matches.size > 0:
                goal_idx = np.append(goal_idx, matches[0])

    session['goal_idx'] = goal_idx.astype(int)

    # get active goal
    active_goal = np.zeros((session['num_laps'],len(session['landmarks'])))
    count = 0
    active_goal[0,0] = goal_idx[count]
    for i in range(session['num_laps']):
        for j in range(len(session['landmarks'])):
            active_goal[i,j] = goal_idx[count]
            if session['rewarded_lms'][i,j]==1:
                count += 1
                if count == len(goal_idx):
                    count = 0

    session['active_goal'] = active_goal

    return session

def get_num_landmarks(session, options):
    rulename = options['sequence_task']['rulename']
    if rulename == 'run-auto' or rulename == 'run-lick':  # stages 1-2
        num_landmarks = 0
    elif rulename == 'olfactory_shaping' or rulename == 'olfactory_test':  # stages 3-6
        if rulename == 'olfactory_test':
            num_landmarks = 10
        else:
            num_landmarks = 2
    else:
        num_landmarks = 10

    session['num_landmarks'] = num_landmarks

    return session 

def get_transition_prob(session):
    #linearize session['licked_lms'] to a vector
    licked_lm_vector = session['licked_lms'].flatten()
    reward_lm_vector = session['rewarded_lms'].flatten()
    lm_sequence = np.tile(range(len(session['landmarks'])),session['num_laps'])
    licked_lm_ix = np.where(licked_lm_vector == 1)[0]

    transition_prob = np.zeros((len(session['goals']),len(session['landmarks'])))

    for g,goal in enumerate(session['goal_idx']):
        rewarded_laps = np.intersect1d(np.where(reward_lm_vector == 1)[0],np.where(lm_sequence == goal)[0])
        for i,reward in enumerate(rewarded_laps):
            if i == len(rewarded_laps)-1:
                break
            next_lick_index = licked_lm_ix[licked_lm_ix>rewarded_laps[i]][0]
            next_lm = lm_sequence[next_lick_index]
            transition_prob[g,next_lm] += 1
        transition_prob[g,:] = transition_prob[g,:]/len(rewarded_laps)
          
    session['transition_prob'] = transition_prob

    return session

def get_all_transitions(session):
    licked_lm_vector = session['licked_lms'].flatten()
    licked_lm_ix = np.where(licked_lm_vector == 1)[0]
    lm_sequence = np.tile(range(len(session['landmarks'])),session['num_laps'])
    licked_lms = lm_sequence[licked_lm_ix]
    transitions = np.zeros((len(session['landmarks']),len(session['landmarks'])))
    for i in range(len(licked_lms)-1):
        transitions[licked_lms[i],licked_lms[i+1]] += 1
    
    session['transitions'] = transitions

    return session

def get_ideal_transitions(session):
    ideal_transition_matrix = np.zeros((len(session['landmarks']), len(session['landmarks'])))
    for i in range(len(session['goal_idx'])-1):
        ideal_transition_matrix[session['goal_idx'][i], session['goal_idx'][i+1]] = 1
        ideal_transition_matrix[session['goal_idx'][i], session['goal_idx'][i]] = 0
    ideal_transition_matrix[session['goal_idx'][-1], session['goal_idx'][0]] = 1

    session['ideal_transition_matrix'] = ideal_transition_matrix

    return session

def get_sorted_transitions(session):
    sorted_goal_idx = np.sort(session['goal_idx'])
    wrong_transition_matrix = np.zeros((len(session['landmarks']), len(session['landmarks'])))
    for i in range(len(sorted_goal_idx)-1):
        wrong_transition_matrix[sorted_goal_idx[i], sorted_goal_idx[i+1]] = 1
        wrong_transition_matrix[sorted_goal_idx[i], sorted_goal_idx[i]] = 0
    wrong_transition_matrix[sorted_goal_idx[-1], sorted_goal_idx[0]] = 1

    session['wrong_transition_matrix'] = wrong_transition_matrix

    return session

def plot_licks_per_state(session):
    if session['laps_needed'] == 2:
        #sanity check, shuffle A and D of licked_lms
        session['shuff_licked_lms'] = copy.copy(session['licked_lms'])
        np.random.shuffle(session['shuff_licked_lms'][:,session['goal_idx'][0]])
        np.random.shuffle(session['shuff_licked_lms'][:,session['goal_idx'][-1]])
        shuff_state1 = session['shuff_licked_lms'][np.where(session['state_id'] == 0)[0]]
        shuff_state2 = session['shuff_licked_lms'][np.where(session['state_id'] == 1)[0]]
        state1_laps = session['licked_lms'][np.where(session['state_id'] == 0)[0]]
        state2_laps = session['licked_lms'][np.where(session['state_id'] == 1)[0]]
        state1_hist = np.sum(state1_laps,axis=0)/state1_laps.shape[0]
        state2_hist = np.sum(state2_laps,axis=0)/state2_laps.shape[0]
        shuff_state1_hist = np.sum(shuff_state1,axis=0)/shuff_state1.shape[0]
        shuff_state2_hist = np.sum(shuff_state2,axis=0)/shuff_state2.shape[0]

        session['state1_hist'] = state1_hist
        session['state2_hist'] = state2_hist
        session['shuff_state1_hist'] = shuff_state1_hist
        session['shuff_state2_hist'] = shuff_state2_hist

    elif session['laps_needed'] == 3:
        #sanity check, shuffle A and D of licked_lms
        session['shuff_licked_lms'] = copy.copy(session['licked_lms'])
        np.random.shuffle(session['shuff_licked_lms'][:,session['goal_idx'][0]])
        np.random.shuffle(session['shuff_licked_lms'][:,session['goal_idx'][-1]])
        shuff_state1 = session['shuff_licked_lms'][np.where(session['state_id'] == 0)[0]]
        shuff_state2 = session['shuff_licked_lms'][np.where(session['state_id'] == 1)[0]]
        shuff_state3 = session['shuff_licked_lms'][np.where(session['state_id'] == 2)[0]]
        state1_laps = session['licked_lms'][np.where(session['state_id'] == 0)[0]]
        state2_laps = session['licked_lms'][np.where(session['state_id'] == 1)[0]]
        state3_laps = session['licked_lms'][np.where(session['state_id'] == 2)[0]]
        state1_hist = np.sum(state1_laps,axis=0)/state1_laps.shape[0]
        state2_hist = np.sum(state2_laps,axis=0)/state2_laps.shape[0]
        state3_hist = np.sum(state3_laps,axis=0)/state3_laps.shape[0]
        shuff_state1_hist = np.sum(shuff_state1,axis=0)/shuff_state1.shape[0]
        shuff_state2_hist = np.sum(shuff_state2,axis=0)/shuff_state2.shape[0]
        shuff_state3_hist = np.sum(shuff_state3,axis=0)/shuff_state3.shape[0]

        session['state1_hist'] = state1_hist
        session['state2_hist'] = state2_hist
        session['state3_hist'] = state3_hist

        session['shuff_state1_hist'] = shuff_state1_hist
        session['shuff_state2_hist'] = shuff_state2_hist
        session['shuff_state3_hist'] = shuff_state3_hist

    return session

def plot_ethogram(session,npz=False):
    if not npz:
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(12,6))
        ax1.plot(session['time'],session['position'])
        ax1.plot(session['time'][session['licks']],session['position'][session['licks']],'ro')
        ax1.plot(session['time'][session['rewards']],session['position'][session['rewards']],'go')
        ax1.plot(session['time'][session['assist_rewards']],session['position'][session['assist_rewards']],'bo')
        if 'thresholded_licks' in session:
            ax1.plot(session['time'][session['thresholded_licks']],session['position'][session['thresholded_licks']],'mo')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (cm)')
        ax1.set_title('Ethogram')
        ax2.plot(session['time'],session['lap_idx'])
        ax2.plot(session['time'],session['lm_idx'])
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Lap/Landmark')
        ax3.plot(session['time'],session['speed'])
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Speed (cm/s)')
        ax4.plot(session['time'],session['lick_rate'])
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Lick rate (licks/s)')
        plt.tight_layout()
    else:
        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,6))
        ax1.plot(session['position'])
        ax1.plot(session['licks'],session['position'][session['licks']],'ro')
        ax1.plot(session['rewards'],session['position'][session['rewards']],'go')
        if 'thresholded_licks' in session:
            ax1.plot(session['thresholded_licks'],session['position'][session['thresholded_licks']],'mo')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (cm)')
        ax1.set_title('Ethogram')
        ax2.plot(session['lap_idx'])
        ax2.plot(session['lm_idx'])
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Lap/Landmark')
        ax3.plot(session['speed'])
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Speed (cm/s)')
        plt.tight_layout()

    fig,ax = plt.subplots(1,1,figsize=(10,6))
    for i in range(session['num_laps']):
        ax.eventplot(session['licks_per_lap'][i],lineoffsets=i,colors='k')
    #plot grey rectangles for landmarks
    for i in range(len(session['landmarks'])):
        ax.add_patch(patches.Rectangle((session['landmarks'][i][0],0),session['landmarks'][i][1]-session['landmarks'][i][0],session['num_laps'],color='grey',alpha=0.3))
    ax.set_xlabel('Position')
    ax.set_ylabel('Lap')
    ax.set_title('Licks per lap')
    plt.tight_layout()

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(8,4))
    # ax1.imshow(session['licked_lms'].T,aspect='auto')
    sns.heatmap(session['licked_lms'].T, ax=ax1, cmap=tm_palette)
    ax1.set_xlabel('Lap')
    ax1.set_ylabel('Landmark')
    ax1.set_title('Licked Landmarks')
    # ax2.imshow(session['rewarded_lms'].T,aspect='auto')
    sns.heatmap(session['rewarded_lms'].T, ax=ax2, cmap=tm_palette)
    ax2.set_xlabel('Lap')
    ax2.set_ylabel('Landmark')
    ax2.set_title('Rewarded Landmarks')
    plt.tight_layout()

    fig,ax1 = plt.subplots(1,1,figsize=(8,2))
    # ax1.imshow(session['licked_lms'].T,aspect='auto')
    sns.heatmap(session['shuff_licked_lms'].T, ax=ax1, cmap=tm_palette)
    ax1.set_xlabel('Lap')
    ax1.set_ylabel('Landmark')
    ax1.set_title('Shuffled A licked Landmarks')


    fig,ax = plt.subplots(1,1,figsize=(4,1.5))
    # cax1 = ax.matshow(session['transition_prob'],aspect='auto')
    sns.heatmap(session['transition_prob'], ax=ax, cmap=tm_palette)
    ax.set_xticks(range(len(session['landmarks'])))
    ax.set_yticks(range(len(session['goals'])))
    ax.set_yticklabels(session['goal_idx'])
    ax.set_xlabel('Next Landmark')
    ax.set_ylabel('Goal')
    ax.set_title('Post reward transitions')
    plt.tight_layout()

    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,3))
    # cax1 = ax1.matshow(session['ideal_transition_matrix'],aspect='auto')
    sns.heatmap(session['ideal_transition_matrix'], ax=ax1, cmap=tm_palette)
    ax1.set_xlabel('Next Landmark')
    ax1.set_ylabel('Current Landmark')
    ax1.set_title('Ideal Transitions')
    # cax2 = ax2.matshow(session['wrong_transition_matrix'],aspect='auto')
    sns.heatmap(session['wrong_transition_matrix'], ax=ax2, cmap=tm_palette)
    ax2.set_xlabel('Next Landmark')
    ax2.set_ylabel('Current Landmark')
    ax2.set_title('Disc Transitions')
    # cax3 = ax3.matshow(session['transitions'],aspect='auto')
    sns.heatmap(session['transitions'], ax=ax3, cmap=tm_palette)
    ax3.set_xlabel('Next Landmark')
    ax3.set_ylabel('Current Landmark')
    ax3.set_title('All Transitions')
    plt.tight_layout()

    fig,ax = plt.subplots(1,1,figsize=(5,3))
    ax.set_prop_cycle(custom_cycler)
    ax.plot(session['sw_state_ratio_a'])
    ax.plot(session['sw_state_ratio_b'])
    ax.plot(session['sw_state_ratio_c'])
    ax.plot(session['sw_state_ratio_d'])
    ax.plot(session['sw_state_ratio'],'k',linewidth=2)
    ax.hlines(np.nanmean(session['sw_state_ratio'][11:]),0,session['num_laps'],linestyle='--')
    ax.legend(['A','B','C','D','Mean'])
    ax.set_xlabel('Lap')
    ax.set_ylabel('State ratio')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    fig,ax = plt.subplots(1,1,figsize=(10,3))
    ax.set_prop_cycle(custom_cycler)
    ax.plot(session['state1_hist'])
    ax.plot(session['state2_hist'])
    if session['laps_needed'] == 3:
        ax.plot(session['state3_hist'])
    ax.set_xlabel('Landmark')
    ax.set_ylabel('Delta fraction lick')
    ax.set_ylim([0,1])
    # ax.legend(['Lap 1','Lap 2','Lap 3'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    # fig,ax = plt.subplots(1,1,figsize=(10,3))
    # ax.set_prop_cycle(custom_cycler)
    # ax.plot(session['shuff_state1_hist'])
    # ax.plot(session['shuff_state2_hist'])
    # if session['laps_needed'] == 3:
    #     ax.plot(session['shuff_state3_hist'])
    #     ax.legend(['Lap 1','Lap 2','Lap 3'])
    # else:
    #     ax.legend(['Lap 1','Lap 2'])
    # ax.set_xlabel('Landmark')
    # ax.set_ylabel('Delta fraction lick')
    

    fig,ax = plt.subplots(1,1,figsize=(10,3))
    ax.set_prop_cycle(custom_cycler)
    ax.plot(session['state1_speed'])
    ax.fill_between(range(len(session['state1_speed'])),session['state1_speed']-session['state1_speed_sem'],session['state1_speed']+session['state1_speed_sem'],alpha=0.3)
    ax.plot(session['state2_speed'])
    ax.fill_between(range(len(session['state2_speed'])),session['state2_speed']-session['state2_speed_sem'],session['state2_speed']+session['state2_speed_sem'],alpha=0.3)
    if session['laps_needed'] == 3:
        ax.plot(session['state3_speed'])
        ax.fill_between(range(len(session['state3_speed'])),session['state3_speed']-session['state3_speed_sem'],session['state3_speed']+session['state3_speed_sem'],alpha=0.3)
        # ax.legend(['Lap 1','Lap 2','Lap 3'])
 
    for lm in session['binned_lms']:
        ax.add_patch(patches.Rectangle((lm[0],0),np.diff(lm)[0],ax.get_ylim()[1],color='grey',alpha=0.3))
    for goal in session['binned_goals']:
        ax.add_patch(patches.Rectangle((goal[0],0),np.diff(goal)[0],ax.get_ylim()[1],color='grey',alpha=0.5))
    ax.set_xlabel('Landmark')
    ax.set_ylabel('Speed (cm/s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()


def plot_speed_profile(session, stage):
    import matplotlib.patches as patches

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
        fig, ax = plt.subplots(1, 1, figsize=(8,3), sharex=False, sharey=False)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10,3))
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
    plt.show()


def plot_deceleration_profile(session, stage):
    import matplotlib.patches as patches

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
        fig, ax = plt.subplots(1, 1, figsize=(8,3), sharex=False, sharey=False)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10,3))
    ax.plot(session['decel_per_bin'], color=color)
    ax.fill_between(range(len(session['decel_per_bin'])),
                    session['decel_per_bin'] - session['sem_decel_per_bin'],
                    session['decel_per_bin'] + session['sem_decel_per_bin'],
                    color=color, alpha=0.3)

    for lm in session['binned_lms']:
        ax.add_patch(patches.Rectangle((lm[0],0), np.diff(lm)[0], ax.get_ylim()[1], color='grey', alpha=0.3))
    for goal in session['binned_goals']:
        ax.add_patch(patches.Rectangle((goal[0],0), np.diff(goal)[0], ax.get_ylim()[1], color='grey', alpha=0.5))
    ax.set_xlabel('Landmark')
    ax.set_ylabel('Deceleration (cm/s^2)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

    
def analyse_session(mouse,date,plot=True):
    base_path = find_base_path(mouse,date)
    data = load_session(base_path)
    options = load_config(base_path)
    session = create_session_struct(data,options)
    session = get_num_landmarks(session, options)
    session = get_lap_idx(session)
    session = get_lm_idx(session)
    session = threshold_licks(session)
    session = get_licks_per_lap(session)
    session = get_licked_lms(session)
    session = get_rewarded_lms(session)
    session = get_active_goal(session)
    session = get_transition_prob(session)
    session = get_all_transitions(session)
    session = get_ideal_transitions(session)
    session = get_sorted_transitions(session)
    session = calculate_lick_rate(session)
    session = calc_laps_needed(session)
    session = give_lap_state_id(session)
    session = plot_licks_per_state(session)
    session = sw_state_ratio(session)
    session = calc_speed_per_lap(session)
    session = calc_speed_per_state(session)
    print('Performance = ', np.nanmean(session['sw_state_ratio'][11:]))
    print('Number of laps = ', session['num_laps'])
    if plot:
        plot_ethogram(session,npz=False)

    return session

def analyse_npz(mouse,date,plot=True):
    base_path = find_base_path_npz(mouse,date)
    data = load_session_npz(base_path)
    base_path2 = find_base_path(mouse,date)
    options = load_config(base_path2)
    
    session = create_session_struct_npz(data,options)
    session['mouse'] = mouse
    session['date'] = date

    session = get_num_landmarks(session, options)
    session = get_lap_idx(session)
    session = get_lm_idx(session)
    session = threshold_licks(session)
    session = get_licks_per_lap(session)
    session = get_licked_lms(session)
    session = get_rewarded_lms(session)
    session = get_lms_visited(options, session)

    VR_data = load_session(base_path2)
    session = neural_analysis_helpers.get_rewards(VR_data, data, session, print_output=True)
    
    session = get_active_goal(session)
    session = get_transition_prob(session)
    session = get_all_transitions(session)
    session = get_ideal_transitions(session)
    session = get_sorted_transitions(session)
    session = calc_laps_needed(session)
    session = give_lap_state_id(session)
    session = plot_licks_per_state(session)
    session = sw_state_ratio(session)
    session = calc_speed_per_lap(session)
    session = calc_speed_per_state(session)

    print('Performance = ', np.nanmean(session['sw_state_ratio'][11:]))
    print('Number of laps = ', session['num_laps'])
    
    if plot:
        plot_ethogram(session,npz=True)

    return session

def analyse_npz_pre7(mouse,date,stage,plot=False):
    base_path = find_base_path_npz(mouse,date)
    data = load_session_npz(base_path)
    base_path2 = find_base_path(mouse,date)
    options = load_config(base_path2)

    session = create_session_struct_npz(data,options)
    session['mouse'] = mouse
    session['date'] = date
    session['stage'] = stage
    
    session = get_num_landmarks(session, options)
    session = get_lap_idx(session)
    session = get_lm_idx(session)
    session = threshold_licks(session)
    session = get_licks_per_lap(session)
    session = get_licked_lms(session)
    session = get_rewarded_lms(session)
    session = get_lms_visited(options, session)

    VR_data = load_session(base_path2)
    session = neural_analysis_helpers.get_rewards(VR_data, data, session, print_output=True)
    session = neural_analysis_helpers.get_AB_sequence(session, mouse, stage)
    session = neural_analysis_helpers.get_landmark_categories(session)
    session = neural_analysis_helpers.get_licks(data, session)
    session = neural_analysis_helpers.get_rewarded_landmarks(VR_data, data, session)
    session = neural_analysis_helpers.get_landmark_category_rew_idx(session, VR_data, data)
    session = neural_analysis_helpers.get_lick_rate(data, session)

    session = get_active_goal(session)
    session = get_transition_prob(session)
    session = get_all_transitions(session)
    session = get_ideal_transitions(session)
    session = get_sorted_transitions(session)
    session = calc_laps_needed(session)
    session = give_lap_state_id(session)
    session = plot_licks_per_state(session)
    session = calc_speed_per_lap_pre7(session)

    # Get lick profile 
    neural_analysis_helpers.plot_lick_maps(session)

    # Get speed profile
    stage = int(stage[-1])
    plot_speed_profile(session, stage=stage)

    print('Number of laps = ', session['num_laps'])
    
    if plot:
        plot_ethogram(session,npz=True)

    return session
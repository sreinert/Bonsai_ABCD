import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.patches as patches
import os, re, sys, yaml, math, copy
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, resample
from scipy.stats import norm
from scipy.interpolate import make_interp_spline
import pandas as pd
from math import log10, floor
import seaborn as sns
import palettes
import importlib
from pathlib import Path
import wesanderson
from cycler import cycler

hfs_palette = palettes.met_brew('Hiroshige',n=123, brew_type="continuous")
rev_hfs = hfs_palette[::-1]
tm_palette = palettes.met_brew('Tam',n=123, brew_type="continuous")
tm_palette = tm_palette[::-1]
color_scheme = wesanderson.film_palette('Darjeeling Limited',palette=0)
custom_cycler = cycler(color=color_scheme)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

#%% ##### Loading #####
def get_session_folders(base_path, mouse, stage):
    """
    Get the session folders for a given mouse and state.
    """
    mouse_path = os.path.join(base_path, mouse)
    session_folder = [f for f in os.listdir(mouse_path) if stage in f][0]
    session_path = os.path.join(mouse_path, session_folder)
    date = session_folder.split('_')[1]
    date1 = date[7:]
    date2 = date[5:]

    imaging_path = os.path.join(session_path, 'funcimg/Session/suite2p/plane0/')
    config_path = os.path.join(session_path, 'behav/', date1)
    if os.path.exists(os.path.join(session_path, 'valid_frames.npz')):
        frame_ix = np.load(os.path.join(session_path, 'valid_frames.npz'))
    else:
        frame_ix = None

    return imaging_path, config_path, frame_ix, date1, date2

def load_session(base_path):    
    # load position log data 
    path = base_path + '/position_log.csv'
    data = pd.read_csv(path)
    return data

def load_session_npz(base_path):
    # load position log data - after barcode alignment
    data = np.load(base_path + '/behaviour_data.npz')
    return data

def load_config(base_path):
    path = base_path + '/config.yaml'
    with open(path) as file:
        options = yaml.load(file, Loader=yaml.FullLoader)
    return options

def find_session_base_path(mouse, date, root):
    data_dir = root + mouse
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    sessions = [s for s in folders if date in s]
    base_path = os.path.join(data_dir, sessions[0])

    return base_path

def find_base_path(mouse, date):
    root = '/Volumes/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/'
    data_dir = root + mouse
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

def find_base_path_npz(mouse, date):
    data_dir = '/Volumes/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/' + mouse
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    sessions = [s for s in folders if date in s]
    base_path = os.path.join(data_dir, sessions[0])

    return base_path

def load_vr_session_info(base_path, VR_data=None, options=None):  # TODO: deprecated? 
    '''Get landmark, goal, and lap information from VR data.'''

    # Load VR data 
    if VR_data is None and options is None:
        VR_data = load_session(base_path)
        options = load_config(base_path)

    #### Determine behaviour stage: (1) what defines VR start and (2) number of distinct landmarks
    rulename = options['sequence_task']['rulename']
    if rulename == 'run-auto' or rulename == 'run-lick':  # stages 1-2
        start_odour = False  # VR started with reward delivery
    elif rulename == 'olfactory_shaping' or rulename == 'olfactory_test':  # stages 3-6
        start_odour = True  # first VR event was the odour delivery prep

        if rulename == 'olfactory_test':
            num_landmarks = 10
        else:
            num_landmarks = 2
            # print('Please specify the number of landmarks in the corridor!')  # TODO: read this from config file
    
    #### Deal with VR data from a table with Time, Position, Event, TotalRunDistance
    _, position, _, total_dist = get_position_info(VR_data)
    corrected_position = position - np.array(options['flip_tunnel']['margin_start'])

    goals = np.array(options['flip_tunnel']['goals']) #- np.array(options['flip_tunnel']['margin_start'])
    landmarks = np.array(options['flip_tunnel']['landmarks']) #- np.array(options['flip_tunnel']['margin_start'])
    tunnel_length = options['flip_tunnel']['length']

    num_laps = np.ceil([total_dist.max()/position.max()])
    # num_laps = np.ceil([total_dist.max()/corrected_position.max()])
    num_laps = num_laps.astype(int)[0]
    print(f'{num_laps} laps were completed.')

    # find the last landmark that was run through
    last_landmark = np.where(landmarks[:,0] < position[-1])[0][-1]
    num_lms = len(landmarks)*(num_laps-1) + last_landmark 

    lm_ids =  np.array(options['flip_tunnel']['landmarks_sequence'])
    goal_ids = np.array(options['goal_ids'])
    all_lms = np.array([])
    all_goals = np.array([])
    for i in range(num_laps):
        all_lms = np.append(all_lms, lm_ids)
        all_goals = np.append(all_goals, goal_ids)
    all_lms = all_lms.astype(int)
    all_goals = all_goals.astype(int)
    all_lms = all_lms[:num_lms]
    all_goals = all_goals[:num_lms]

    # create a variable that indexes the laps by finding flips first
    flip_ix = np.where(np.diff(position) < -50)[0]
    # a lap is between two flips
    lap_num = np.zeros(len(position))
    for i in range(len(flip_ix)-1):
        lap_num[flip_ix[i]:flip_ix[i+1]] = i+1
    if num_laps > 1:
        lap_num[flip_ix[-1]:] = len(flip_ix)

    # find the landmarks that were completed
    total_lm_position = np.array([])
    for i in range(num_laps):
        lap_lms = landmarks + i*tunnel_length
        total_lm_position = np.append(total_lm_position, lap_lms[:,0])
    total_lm_position = total_lm_position[:num_lms].astype(int)
    print(f"{total_lm_position.shape[0]} landmarks were visited")

    return num_landmarks, all_goals, all_lms, total_lm_position, landmarks, start_odour, num_laps

def extract_int(s: str) -> int:
    m = re.search(r'\d+', s)
    if m:
        return int(m.group())
    else:
        raise ValueError(f"No digits found in string: {s!r}")
    
#%% ##### Session analysis functions - for both VR and NIDAQ data #####
def get_position_info(VR_data):
    '''Find position, speed, total distance, times from VR data.'''
    position_idx = np.where(VR_data['Position'] > -1)[0]
    
    times = VR_data['Time'][position_idx].values
    position = VR_data['Position'][position_idx].values 
    total_dist = VR_data['TotalRunDistance'][position_idx].values #- np.array(options['flip_tunnel']['margin_start'])

    if 'Speed' not in VR_data.keys():
        speed = np.diff(total_dist)/np.diff(times)
        speed = np.append(speed, speed[-1])
    else:
        speed = VR_data['Speed'][position_idx].values
    
    return times, position, speed, total_dist

def get_VR_rewards(VR_data):
    '''Find different types of rewards from VR data.'''
    # rewards_root_VR = np.where(VR_data['Event'] == 'rewarded')[0]
    # rewards_VR = VR_data['Index'][rewards_root_VR].values
    rewards_VR = np.where(VR_data['Event'] == 'rewarded')[0]

    # assistant_reward_root_idx = np.where(VR_data['Event'] == 'assist-rewarded')[0]
    # assistant_reward_idx = VR_data['Index'][assistant_reward_root_idx].values
    assistant_reward_idx = np.where(VR_data['Event'] == 'assist-rewarded')[0]

    # manual_reward_root_idx = np.where(VR_data['Event'] == 'manually-rewarded')[0]
    # manual_reward_idx = VR_data['Index'][manual_reward_root_idx].values
    manual_reward_idx = np.where(VR_data['Event'] == 'manually-rewarded')[0]

    return rewards_VR, assistant_reward_idx, manual_reward_idx

def get_lap_idx(session):
    # get lap idx
    flip_ix = find_peaks(session['position'], height=session['tunnel_length']-1,distance=100)[0]
    if np.abs(session['position'][0] - session['landmarks'][-1,1]) < np.abs(session['position'][0] - session['landmarks'][0,0]):
        flip_ix = flip_ix[1:]  # Remove the first peak index - the mouse accidentally moved backwards first

    if len(flip_ix) > 0:
        # a lap is between two flips
        lap_idx = np.zeros(len(session['position']))
        for i in range(len(flip_ix)-1):
            lap_idx[flip_ix[i]:flip_ix[i+1]] = i+1
        lap_idx[flip_ix[-1]:] = len(flip_ix)

        session['lap_idx'] = lap_idx
        session['num_laps'] = len(flip_ix) #+ 1
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
    if len(np.where(session['landmarks'][:,0] < session['position'][-1])[0]) != len(np.where(session['landmarks'][:,-1] < session['position'][-1])[0]):
        last_landmark = len(np.where(session['landmarks'][:,-1] < session['position'][-1])[0]) # session ended before mouse exited last lm entered
    else:
        last_landmark = len(session['all_landmarks'])     
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

def get_num_landmarks(session, options):
    # Get number of landmarks in the corridor according to behaviour timepoint
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

#%% ##### Functions that work with NIDAQ data only (after funcimg alignment) #####
def get_lm_entry_exit(session, positions=None):
    '''Find data idx closest to landmark entry and exit.'''

    if positions is None:
        base_path = find_base_path_npz(session['mouse'], session['date'])
        nidaq_data = load_session_npz(base_path)
        positions = nidaq_data['position']

    lm_entry_idx = []
    lm_exit_idx = []

    if session['num_laps'] > 1:
        search_start = 0  

        for i, (lm_start, lm_end) in enumerate(session['all_landmarks'][:-1]):

            next_lm_start = session['all_landmarks'][i+1,0]
            next_lm_start_idx = np.where(positions[search_start:] >= next_lm_start)[0][0] + search_start                
            if next_lm_start < lm_start:    # position reset 
                # print('Lap change')
                lap_change_idx = find_peaks(positions[search_start:], height=session['tunnel_length']-1, distance=100)[0][0] + 10 
                next_lm_start_idx = search_start + lap_change_idx + 1
                
            start_candidates = np.where(positions[search_start:next_lm_start_idx] >= lm_start)[0]
            entry_idx = start_candidates[0] + search_start

            end_candidates = np.where(positions[entry_idx:next_lm_start_idx] >= lm_end)[0]
            exit_idx = end_candidates[0] + entry_idx

            search_start = next_lm_start_idx 

            lm_entry_idx.append(entry_idx)
            lm_exit_idx.append(exit_idx)

        # last landmark 
        last_lm_start_idx = np.where(positions[search_start:] >= session['all_landmarks'][-1,0])[0][0] + search_start
        last_lm_end_idx = np.where(positions[search_start:] >= session['all_landmarks'][-1,1])[0]
        if len(last_lm_end_idx) != 0:
            last_lm_end_idx = last_lm_end_idx[0] + search_start
            lm_entry_idx.append(last_lm_start_idx)  
            lm_exit_idx.append(last_lm_end_idx)
        else:
            return np.array(lm_entry_idx), np.array(lm_exit_idx)  # terminate early 
    
    else:
        if np.abs(positions[0] - session['landmarks'][-1,1]) < np.abs(positions[0] - session['landmarks'][0,0]):
            search_start = np.where(positions <= session['all_landmarks'][0,0])[0][-1]  # the mouse accidentally moved backwards first
        else: 
            search_start = 0

        for lm_start in session['all_landmarks'][:,0]:
            lm_entry_idx.append(np.where(positions[search_start:] >= lm_start)[0][0] + search_start)

        for lm_end in session['all_landmarks'][:,1]:
            lm_exit_idx.append(np.where(positions[search_start:] <= lm_end)[0][-1] + search_start)

    return np.array(lm_entry_idx), np.array(lm_exit_idx)

def get_before_lm_entry_exit(session):
    '''Find entry and exit indices for the between-landmark points'''
    lm_entry_idx, lm_exit_idx = get_lm_entry_exit(session)

    before_lm_entry_idx = [0]
    before_lm_exit_idx = [int(lm_entry_idx[0])-1]
    for entry, exit in zip(lm_entry_idx[1:], lm_exit_idx[:-1]):  
        before_lm_entry_idx.append(int(exit) + 1)
        before_lm_exit_idx.append(int(entry) - 1)

    return before_lm_entry_idx, before_lm_exit_idx

def get_imag_rew_idx(session, lm_idx):
    '''Find indices after landmark entry where reward would be expected.'''
    
    lm_entry_idx, _ = get_lm_entry_exit(session)
    lm_entry_idx = np.array([lm_entry_idx[i] for i in lm_idx])
    imag_rew_idx = lm_entry_idx + session['rew_time_lag']

    return imag_rew_idx
    
def get_rewarded_landmarks(VR_data, nidaq_data, session):
    '''Find the indices of rewarded (lick-triggered) landmarks.'''

    session = get_rewards(VR_data, nidaq_data, session, print_output=False)
    lm_entry_idx, lm_exit_idx = get_lm_entry_exit(session)

    # Find rewarded landmarks 
    reward_positions = nidaq_data['distance'][session['reward_idx']]  # using flattened position array 

    rewarded_landmarks = [i for i, (start, end) in enumerate(zip(np.floor(nidaq_data['distance'][lm_entry_idx]), np.ceil(nidaq_data['distance'][lm_exit_idx]))) 
                            if np.any((np.ceil(reward_positions) >= start) & (np.floor(reward_positions) <= end))] 
    
    session['rewarded_landmarks'] = rewarded_landmarks

    return session

def get_rewards(VR_data, nidaq_data, session, print_output=False):
    '''Find the indices of lick-triggered rewards in the nidaq logging file.'''

    # Find different types of rewards from VR data
    rewards_VR, assistant_reward_idx, manual_reward_idx = get_VR_rewards(VR_data)
    all_rewards_VR = np.sort(np.concatenate([rewards_VR, assistant_reward_idx, manual_reward_idx]))

    # Find rewards in NIDAQ data
    reward_idx = np.where(nidaq_data['rewards'] == 1)[0]  
    rewards_to_remove = []

    for r, rew in enumerate(all_rewards_VR):
        if (rew in assistant_reward_idx) or (rew in manual_reward_idx):
            rewards_to_remove.append(r)

    reward_idx = np.delete(reward_idx, rewards_to_remove)

    # Confirm number of rewards makes sense
    if session['all_landmarks'][-1,1] < nidaq_data['position'][reward_idx[-1]]:  # ensure mouse has left last rewarded landmark 
        reward_idx = reward_idx[0:-1]  
    num_rewards = len(reward_idx)  

    session['reward_idx'] = reward_idx

    if print_output:
        print('Total rewards considered here: ', num_rewards)
        print('Total rewards not considered here: ', len(rewards_to_remove))
        print('Total assistant and manual rewards: ', len(assistant_reward_idx) + len(manual_reward_idx))

    return session

def get_reward_idx_by_goal(session, ABCD_goals=[1,2,3,4]): # TODO
    rew_idx = {}    
    for g, goal in enumerate(ABCD_goals):
        rew_idx[goal] = np.array([idx for idx in session['rewarded_landmarks'] \
                                if session['all_lms'][idx] == session['goal_landmark_id'][goal-1]])
        
    return rew_idx

def get_events_in_surrounding_landmarks(session, events, time_around, funcimg_frame_rate=45, label="Event"):
    """
    Identify time windows around events that fall within the previous of following landmarks.

    Args:
        events (list or array): List of event indices (e.g. licks, rewards).
        time_around (float): Time in seconds around the event to consider.
        funcimg_frame_rate (float): Frame rate of functional imaging.
        label (str): Descriptive label for printing.
    """
    lm_entry_idx, lm_exit_idx = get_lm_entry_exit(session)

    time_window = int(time_around * funcimg_frame_rate)
    print(f"{label}: {len(events)}")

    for i, event in enumerate(events):
        # Find previous landmark exit
        lm_exit_candidates = np.where(lm_exit_idx < event)[0]
        if lm_exit_candidates.size > 0:
            prev_landmark_idx = lm_exit_candidates[-1]
        else:
            prev_landmark_idx = 0

        # Find next landmark entry
        lm_entry_candidates = np.where(lm_entry_idx > event)[0]
        if lm_entry_candidates.size > 0:
            next_landmark_idx = lm_entry_candidates[0]
        else:
            next_landmark_idx = len(lm_entry_idx) - 1

        # Check if window overlaps with a landmark
        in_landmark = (
            (lm_entry_idx[next_landmark_idx] <= event + time_window) or
            (lm_exit_idx[prev_landmark_idx] >= event - time_window)
        )

        if in_landmark:
            print(i, event - time_window, event + time_window)

    return

def get_licks(nidaq_data, session, print_output=False):
    '''Find the indices of licks in the nidaq logging file.'''

    # Find licks in NIDAQ data
    lick_idx = np.where(nidaq_data['licks'] >= 1)[0]  

    # Confirm number of rewards makes sense
    if session['all_landmarks'][-1,1] < nidaq_data['position'][lick_idx[-1]]:  # TODO ensure mouse has left last licked landmark 
        lick_idx = lick_idx[0:-1]  
    num_licks = len(lick_idx)  

    session['lick_idx'] = lick_idx
    if print_output:
        print('Total licks considered here: ', num_licks)
        
    return session

def threshold_nidaq_licks(nidaq_data, session):
    """Threshold which licks are considered based on the animal's speed."""
    
    lick_threshold = session['lick_threshold']
    threshold_licks = session['lick_idx'][np.where(nidaq_data['speed'][session['lick_idx']] < lick_threshold)[0]]

    session['thresholded_lick_idx'] = threshold_licks

    return session

def calculate_frame_lick_rate(nidaq_data, session):
    """Get lick rate per frame as a sliding window - similar to calculate_lick_rate"""
    
    # Threshold licks
    session = threshold_nidaq_licks(nidaq_data, session)

    # Calculate lick rate as the mean number of licks over sliding window
    window = 100 # frames
    lick_rate = np.zeros(len(nidaq_data['position']))
    for i in range(len(nidaq_data['position'])-window):
        if 'thresholded_lick_idx' in session:
            lick_num = len(np.where((session['thresholded_lick_idx'] > i) & (session['thresholded_lick_idx'] < i+window))[0])
        else:
            lick_num = len(np.where((session['lick_idx'] > i) & (session['lick_idx'] < i+window))[0])
        lick_rate[i] = lick_num / window
    
    session['frame_lick_rate'] = lick_rate

    return session

def get_smoothed_lick_rate(nidaq_data, session):
    """Get lick rate per frame as a sliding window - similar to calculate_lick_rate"""
    
    # Threshold licks
    session = threshold_nidaq_licks(nidaq_data, session)

    # Calculate smoothed lick rate 
    binary_licks = np.zeros_like(nidaq_data['position'], dtype=int)
    binary_licks[session['thresholded_lick_idx']] = 1

    lick_rate = gaussian_filter1d(binary_licks.astype(float), sigma=1.5)
    lick_rate = lick_rate.reshape(1,-1)

    session['smooth_lick_rate'] = lick_rate

    return session

def get_event_lick_rate(session, nidaq_data, event_idx, time_around=(-1,3), funcimg_frame_rate=45):
    """Get lick rate per frame as a smoothed sliding window around an event"""
    
    # Handle single int input as symmetric window
    if isinstance(time_around, (int, float)):
        start_time = -time_around
        end_time = time_around
    elif isinstance(time_around, (tuple, list)) and len(time_around) == 2:
        start_time, end_time = time_around
    else:
        raise ValueError("time_around must be a single number or a tuple/list of (start, end)")

    start_frames = int(np.floor(start_time * funcimg_frame_rate))
    end_frames = int(np.ceil(end_time * funcimg_frame_rate))

    # Get indices for each event
    window = np.arange(start_frames, end_frames)
    window_indices = np.add.outer(event_idx, window).astype(int)

    # Find licks within this window
    binary_licks = np.zeros_like(window_indices, dtype=int)
    mask = np.isin(window_indices, session['thresholded_lick_idx'])
    binary_licks[mask] = nidaq_data['licks'][window_indices[mask]]

    # Get smoothed lick rate 
    event_lick_rate = np.empty_like(window_indices, dtype=float)
    for i in range(window_indices.shape[0]):
        event_lick_rate[i,:] = gaussian_filter1d(binary_licks[i,:].astype(float), sigma=1.5)
    
    return event_lick_rate

def get_lm_lick_rate(nidaq_data, session, bins=16):  # TODO I really need to fix this and make it consistent across sessions
    """Get lick rate per frame bin as the mean per bin for each landmark"""
    
    # Get all datapoints within landmarks
    session = get_data_lm_idx(nidaq_data, session)

    # Create a binary lick map for the entire session 
    binary_licks = np.zeros(len(nidaq_data['position']))
    binary_licks[session['thresholded_lick_idx']] = nidaq_data['licks'][session['thresholded_lick_idx']] # (actually not binary)

    if ('stage' in session) and ('3' in session['stage'] or '4' in session['stage']):
        
        lm_lick_rate = np.zeros((len(session['all_lms']), bins))
        for lm in range(len(session['all_lms'])):
            # datapoints within landmarks for each lap 
            lm_idx = np.where(session['data_lm_idx'] == lm+1)[0]

            # binary licks within landmark
            lm_licks = binary_licks[lm_idx[0]:lm_idx[-1]+1]
            
            # calculate lick rate within each landmark (mean in each bin)
            lm_lick_rate[lm], _, _ = stats.binned_statistic(lm_idx, lm_licks, bins=bins)

    else:
        lm_lick_rate = {}
        for lap in range(session['num_laps']):
            for lm in range(len(session['all_lms'])):
                key = (lap, lm)

                # datapoints within landmarks for each lap 
                lm_idx = np.where(session['data_lm_idx'] == lm+1)[0]

                # binary licks within landmark
                lm_licks = binary_licks[lm_idx[0]:lm_idx[-1]+1]
                
                # calculate lick rate within each landmark (mean in each bin)
                lm_lick_rate[key], _, _ = stats.binned_statistic(lm_idx, lm_licks, bins=bins)

    session['lm_lick_rate'] = lm_lick_rate

    return session

def get_binned_lick_rate(nidaq_data, session):  # TODO
    """Get lick rate per frame bin as the mean per bin for each landmark and the gray zones before"""
    
    # Threshold licks
    session = threshold_nidaq_licks(nidaq_data, session)

    # Get all datapoints within landmarks
    session = get_data_lm_idx(nidaq_data, session)

    lm_lick_rate = {}

    # Create a binary lick map for the entire session
    binary_licks = np.zeros(len(nidaq_data['position']))
    binary_licks[session['thresholded_lick_idx']] = nidaq_data['licks'][session['thresholded_lick_idx']] # (actually not binary)

    for lap in range(session['num_laps']):
        for lm in range(len(session['all_lms']) * 2):
            key = (lap, lm)

            # datapoints within landmarks for each lap 
            lm_idx = np.where(session['data_lm_idx'] == lm+1)[0]

            # binary licks within landmark
            lm_licks = binary_licks[lm_idx[0]:lm_idx[-1]+1]
            
            # calculate lick rate within each landmark (mean in each bin)
            lm_lick_rate[key], _, _ = stats.binned_statistic(lm_idx, lm_licks, bins=16)

    session['lm_lick_rate'] = lm_lick_rate

    return session

def get_binary_lick_map(nidaq_data, session):
    """Create a binary map of licked landmarks - similar to get_licked_lms"""
    
    # Get all datapoints within landmarks
    session = get_data_lm_idx(nidaq_data, session)

    licked_lms = np.empty((len(session['all_lms'])))
    for lm in range(len(session['all_lms'])):
        # datapoints within landmarks for each lap 
        lm_idx = np.where(session['data_lm_idx'] == lm+1)[0]

        # Find all licks within the landmark
        if 'thresholded_lick_idx' in session:
            target_licks = np.intersect1d(lm_idx, session['thresholded_lick_idx'])
        else:
            target_licks = np.intersect1d(lm_idx, session['lick_idx'])
        if len(target_licks) > 0:
            licked_lms[lm] = 1
        else:
            licked_lms[lm] = 0

    session['binary_licked_lms'] = licked_lms

    return session

def get_licks_per_lap(session):
    #save lick indices for each lap in a dictionary
    lick_positions = {}
    for i in range(session['num_laps']):
        lap_ix = np.where(session['lap_idx'] == i)[0]
        licks_per_lap_ix = np.intersect1d(lap_ix, session['thresholded_licks'])
        lick_positions[i] = session['position'][licks_per_lap_ix]

    session['licks_per_lap'] = lick_positions

    return session

def get_data_lm_idx(nidaq_data, session):
    """Get the landmark id of every data entry - similar to get_lm_idx"""
    
    # Find landmark entry and exit idx
    lm_entry, lm_exit = get_lm_entry_exit(session, nidaq_data['position'])

    # Find datapoints within a landmark
    lm_idx = np.zeros(len(nidaq_data['position']))
    for i in range(len(session['all_lms'])):
        lm_idx[lm_entry[i]:lm_exit[i]+1] = i+1

    session['data_lm_idx'] = lm_idx

    return session

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

def get_landmark_category_rew_idx(session, VR_data, nidaq_data):
    '''Find indices also in non-goal landmarks corresponding to the same time after landmark entry as mean reward time lag.'''
    
    session = get_rewards(VR_data, nidaq_data, session, print_output=True)

    rew_lm_entry_idx, miss_lm_entry_idx, nongoal_lm_entry_idx, test_lm_entry_idx = get_landmark_category_entries(VR_data, nidaq_data, session)
    
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

def get_landmark_category_entries(VR_data, nidaq_data, session):
    '''Find the indices of landmark entry for different types of landmarks: rewarded, miss, non-goal, test.'''
    
    lm_entry_idx, _ = get_lm_entry_exit(session, positions=nidaq_data['position'])

    # Find category for each landmark 
    session = get_landmark_categories(session)

    # Find the rewarded landmarks 
    session = get_rewarded_landmarks(VR_data, nidaq_data, session)

    # Find landmark entry indices for each landmark category
    rew_lm_entry_idx = [lm_entry_idx[i] for i in session['rewarded_landmarks']]
    miss_lm_entry_idx = np.array([lm_entry_idx[i] for i in session['goals_idx'] if i not in session['rewarded_landmarks']])
    nongoal_lm_entry_idx = np.array([lm_entry_idx[i] for i in session['non_goals_idx']])
    test_lm_entry_idx = np.array([lm_entry_idx[i] for i in session['test_idx']]) if session['test_idx'] is not None else np.array([])

    assert len(rew_lm_entry_idx) + len(miss_lm_entry_idx) + len(nongoal_lm_entry_idx) + len(test_lm_entry_idx) == len(session['all_lms']), 'Some landmarks have not been considered.'

    return rew_lm_entry_idx, miss_lm_entry_idx, nongoal_lm_entry_idx, test_lm_entry_idx

#%% ##### Analysis wrappers #####
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

def create_session_struct(data, options):
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

def analyse_session(mouse, date, plot=True):
    base_path = find_base_path(mouse, date)
    data = load_session(base_path)
    options = load_config(base_path)
    session = create_session_struct(data, options)

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
    session = get_lms_visited(options, session)
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

def analyse_session_pre7(mouse, date):
    base_path = find_base_path(mouse, date)
    data = load_session(base_path)
    options = load_config(base_path)
    session = create_session_struct(data, options)

    session = create_session_struct(data, options)
    session = get_num_landmarks(session, options)
    session = get_lap_idx(session)
    session = get_lm_idx(session)
    session = get_rewarded_lms(session)
    session = get_active_goal(session)
    session = calc_laps_needed(session)
    session = get_lms_visited(options, session)

    print('Number of laps:', session['num_laps'])
    num_lms = len(session['all_landmarks'])
    print('Number of landmarks visited:', num_lms)

    all_rewards_VR = np.sort(np.concatenate([session['rewards'], session['assist_rewards'], session['manual_rewards']]))
    first_reward = np.min(all_rewards_VR)
    print('First reward was delivered', np.round(session['time'][first_reward] - session['time'][0], 2), 's after VR start.')

    return session

def analyse_npz(mouse, date, stage, plot=True): 
    base_path = find_base_path_npz(mouse, date)
    data = load_session_npz(base_path)
    base_path2 = find_base_path(mouse, date)
    options = load_config(base_path2)
    
    session = create_session_struct_npz(data, options)
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
    session = get_licks(data, session)
    session = get_active_goal(session)

    VR_data = load_session(base_path2)
    session = get_rewarded_landmarks(VR_data, data, session)
    session = get_landmark_category_rew_idx(session, VR_data, data)
    session = calculate_frame_lick_rate(data, session)

    session = calc_acceleration(session)
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

def analyse_npz_pre7(mouse, date, stage, plot=False):

    if '3' not in stage and '4' not in stage and '5' not in stage and '6' not in stage:
        raise ValueError('This function only works for T3-T6.')
    
    base_path = find_base_path_npz(mouse, date)
    data = load_session_npz(base_path)
    base_path2 = find_base_path(mouse, date)
    options = load_config(base_path2)

    session = create_session_struct_npz(data, options)
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
    session = get_AB_sequence(session, mouse, stage)
    session = get_landmark_categories(session)
    session = get_licks(data, session)
    session = get_rewarded_landmarks(VR_data, data, session)
    session = get_landmark_category_rew_idx(session, VR_data, data)
    session = calculate_frame_lick_rate(data, session)

    session = calc_acceleration(session)
    session = get_active_goal(session)
    session = get_transition_prob(session)
    session = get_all_transitions(session)
    session = get_ideal_transitions(session)
    session = get_sorted_transitions(session)
    session = calc_laps_needed(session)
    session = give_lap_state_id(session)
    session = plot_licks_per_state(session)
    session = calc_speed_per_lap_pre7(session)

    plot_lick_maps(session)
    plot_speed_profile(session, stage=int(stage[-1]))

    print('Number of laps = ', session['num_laps'])
    
    if plot:
        plot_ethogram(session,npz=True)

    return session

#%% ##### Plotting #####
def plot_ethogram(session, npz=False):
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

def plot_lick_maps(session):

    # Load data
    base_path = find_base_path_npz(session['mouse'], session['date'])
    nidaq_data = load_session_npz(base_path)
    session = get_licks(nidaq_data, session, print_output=True)
    session = threshold_nidaq_licks(nidaq_data, session)

    # ------- Get binary lick map (laps x landmarks) ------- #
    session = get_binary_lick_map(nidaq_data, session)

    # Reshape if laps are not repeating
    # if np.array(session['binary_licked_lms']).ndim != 2:

    # Check number of actual laps
    num_lms_considered = int(np.round((len(session['all_landmarks']) // session['num_landmarks']) * session['num_landmarks']))
    num_laps = int(num_lms_considered / session['num_landmarks'])

    if '3' in session['stage'] or '4' in session['stage']:
        # The landmarks might not be in order so we need to be careful about binning 
        # Determine how many rows to keep
        min_len = min(len(session['goals_idx']), len(session['non_goals_idx']))
        goal_licked_lms = session['binary_licked_lms'][session['goals_idx'][:min_len]]
        non_goal_licked_lms = session['binary_licked_lms'][session['non_goals_idx'][:min_len]]

        goal_licked_lms = goal_licked_lms.reshape((num_laps, -1))        # -1 lets numpy figure out columns
        non_goal_licked_lms = non_goal_licked_lms.reshape((num_laps, -1))

        binary_licked_lms = np.column_stack((goal_licked_lms, non_goal_licked_lms))
    else: 
        # the landmarks are in order so we can simply reshape
        # binary_licked_lms = np.array(session['binary_licked_lms'][0][:num_lms_considered]).reshape((num_laps, session['num_landmarks']))
        binary_licked_lms = np.array(session['binary_licked_lms'][:num_lms_considered]).reshape((num_laps, session['num_landmarks']))
    
        # else:
        #     # binary_licked_lms = np.array(session['binary_licked_lms'][0])
        #     binary_licked_lms = np.array(session['binary_licked_lms'])

    # ------- Get lick rate map (laps x landmarks) ------- #
    session = get_lm_lick_rate(nidaq_data, session)

    if '3' in session['stage'] or '4' in session['stage']:
        goal_lm_lick_rate = session['lm_lick_rate'][session['goals_idx']]
        non_goal_lm_lick_rate = session['lm_lick_rate'][session['non_goals_idx']]

        min_len = min(len(goal_lm_lick_rate), len(non_goal_lm_lick_rate))
        goal_lm_lick_rate = goal_lm_lick_rate[:min_len, :]
        non_goal_lm_lick_rate = non_goal_lm_lick_rate[:min_len, :]

        lm_lick_rate = np.column_stack((goal_lm_lick_rate, non_goal_lm_lick_rate))
        
    else: # TODO
        keys = sorted(session['lm_lick_rate'].keys())  
        first_keys = set(k[0] for k in keys)

        if len(first_keys) == 1:
            lm_lick_rate = [[] for _ in range(num_laps)]
            for lm_idx in range(num_lms_considered):
                lap = lm_idx // session['num_landmarks']
                key = (0, lm_idx)
                if key in session['lm_lick_rate']:
                    rate = session['lm_lick_rate'][key]
                    lm_lick_rate[lap].extend(rate)
        else:
            lm_lick_rate = [[] for _ in range(num_laps)]
            for i in range(num_laps):
                for lm_idx in range(num_lms_considered):
                    lap = lm_idx // session['num_landmarks']
                    key = (i, lm_idx)
                    if key in session['lm_lick_rate']:
                        rate = session['lm_lick_rate'][key]
                        lm_lick_rate[lap].extend(rate)

        lm_lick_rate = np.array(lm_lick_rate)  # (num_laps, num_bins * num_landmarks)

    # Plotting
    tm_palette = palettes.met_brew('Tam', n=123, brew_type="continuous")
    tm_palette = tm_palette[::-1]

    tick_positions = [i * 16 + 16 // 2 for i in range(session['num_landmarks'])]
    tick_labels = np.arange(1, session['num_landmarks']+1)  
    
    # Plot the binary and lick rate maps for each landmark 
    if session['num_landmarks'] == 2:
        fig, ax = plt.subplots(1,2, figsize=(8,3), sharex=False, sharey=False)
    else:
        fig, ax = plt.subplots(2, 1, figsize=(10,4), sharex=False, sharey=False)
    
    ax = ax.ravel()

    # Plot binary licks  
    sns.heatmap(binary_licked_lms, ax=ax[0], cmap=[tm_palette[0], tm_palette[-1]], 
                vmin=0, vmax=1, cbar_kws={"ticks": [0, 1]}, xticklabels=(tick_labels), 
                yticklabels=[0, binary_licked_lms.shape[0]])

    # Plot lick rate
    max_lick_rate = np.round(np.nanmax(lm_lick_rate), 1)
    sns.heatmap(lm_lick_rate, ax=ax[1], cmap=tm_palette, vmin=0, vmax=max_lick_rate, 
                cbar_kws={"ticks": [0, max_lick_rate]})
    for i in range(1, session['num_landmarks']):
        ax[1].axvline(i * 16, color='white', linestyle='--', linewidth=1)

    for axis in ax:
        axis.set_yticks([0, binary_licked_lms.shape[0]])
        axis.set_yticklabels([0, binary_licked_lms.shape[0]])
        axis.set_xlabel('Landmark')
        axis.set_ylabel('Lap')

    ax[0].set_title('Licked Landmarks')
    ax[1].set_title('Lick Rate')
    ax[1].set_xticks(tick_positions)
    ax[1].set_xticklabels(tick_labels, rotation=0)
    
    # plt.tight_layout()

    return binary_licked_lms, lm_lick_rate

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

#%% ##### T3-T6 specific functions #####
def get_AB_sequence(session, mouse, stage):
    if mouse == 'TAA0000066' or mouse == 'TAA0000059':
        if int(stage[-1]) == 3 or int(stage[-1]) == 4:
            sequence = 'AB_shuffled'
        elif int(stage[-1]) == 5 or int(stage[-1]) == 6:
            sequence = 'ABAB'
        else:
            print('This code does not work before T3 or beyond T6 yet.')
    elif mouse == 'TAA0000061' or mouse == 'TAA0000064':
        if int(stage[-1]) == 3 or int(stage[-1]) == 4:
            sequence = 'AABB'
        elif int(stage[-1]) == 5 or int(stage[-1]) == 6:
            sequence = 'AABB'
        else:
            print('This code does not work before T3 or beyond T6 yet.')
    elif mouse == 'TAA0000062' or mouse == 'TAA0000065':
        if int(stage[-1]) == 3 or int(stage[-1]) == 4:
            sequence = 'AB_shuffled'
        elif int(stage[-1]) == 5 or int(stage[-1]) == 6:
            sequence = 'AABB'
        else:
            print('This code does not work before T3 or beyond T6 yet.')
    else:
        raise ValueError("Oops I don't know what to do about this mouse")
    
    session['sequence'] = sequence

    return session

def get_lick_types(session, VR_data, nidaq_data):
    """
    Give an ID to each lick type:
    1: licks inside goal landmarks & rewarded (hit)
    2: licks inside goal landmarks & not rewarded (miss)
    3: licks inside non-goal landmarks (false alarm)
    4: licks inside test landmark
    5: licks before goal landmarks 
    6: licks before non-goal landmarks 
    7: licks before test landmark
    8: imaginary lick inside goal (miss)
    9: imaginary lick inside non-goal 
    10: imaginary lick inside test

    Returns:
    -------
    licks: Dict where the key corresponds to the lick ID
    """

    if ('stage' not in session) or ('3' not in session['stage'] and '4' not in session['stage'] and '5' not in session['stage'] and '6' not in session['stage']):
        raise ValueError('This function only works for T3-T6.')
    
    # Get landmark entry and exit indices 
    lm_entry_idx, lm_exit_idx = get_lm_entry_exit(session)

    # Get lick indices
    session = get_licks(nidaq_data, session)
    lick_idx = session['lick_idx']

    # Get rewarded landmarks
    session = get_rewarded_landmarks(VR_data, nidaq_data, session)

    # Collect all licks 
    licks = {id: {} for id in range(1,11)}

    for id in range(1,11):
        collected_licks = []  
        
        if id == 1:
            collected_licks = [
                lick_idx[(lick_idx >= lm_entry_idx[i]) & (lick_idx <= lm_exit_idx[i])]
                for i in range(len(lm_entry_idx)) if i in session['rewarded_landmarks']
            ]

        elif id == 2:
            collected_licks = [
                lick_idx[(lick_idx >= lm_entry_idx[i]) & (lick_idx <= lm_exit_idx[i])]
                for i in range(len(lm_entry_idx)) if i in session['goals_idx'] and i not in session['rewarded_landmarks']
            ]

        elif id == 3:
            collected_licks = [
                lick_idx[(lick_idx >= lm_entry_idx[i]) & (lick_idx <= lm_exit_idx[i])]
                for i in range(len(lm_entry_idx)) if i in session['non_goals_idx']
            ]

        elif id == 4:
            if session['test_idx'] is not None:
                collected_licks = [
                    lick_idx[(lick_idx >= lm_entry_idx[i]) & (lick_idx <= lm_exit_idx[i])]
                    for i in range(len(lm_entry_idx)) if i in session['test_idx']
                ]
            else:
                collected_licks = [] 

        elif id == 5:
            if 0 in session['goals_idx']:
                collected_licks.append(lick_idx[lick_idx < lm_entry_idx[0]])
            collected_licks += [
                lick_idx[(lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1])]
                for i in range(len(lm_entry_idx) - 1) if i + 1 in session['goals_idx']
            ]

        elif id == 6:
            if 0 in session['non_goals_idx']:
                collected_licks.append(lick_idx[lick_idx < lm_entry_idx[0]])
            collected_licks += [
                lick_idx[(lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1])]
                for i in range(len(lm_entry_idx) - 1) if i + 1 in session['non_goals_idx']
            ]

        elif id == 7:
            if session['test_idx'] is not None:
                if 0 in session['test_idx']:
                    collected_licks.append(lick_idx[lick_idx < lm_entry_idx[0]])  
                collected_licks += [
                    lick_idx[(lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1])]
                    for i in range(len(lm_entry_idx) - 1) if i + 1 in session['test_idx']
                ]
            else: 
                collected_licks = []

        elif id == 8:
            collected_licks = [
                session['miss_rew_idx'][(session['miss_rew_idx'] >= lm_entry_idx[i]) & (session['miss_rew_idx'] <= lm_exit_idx[i])]
                for i in range(len(lm_entry_idx)) if i in session['goals_idx'] and i not in session['rewarded_landmarks'] 
                and not np.any((lick_idx >= lm_entry_idx[i]) & (lick_idx <= lm_exit_idx[i]))
            ]

        elif id == 9:
            collected_licks = [
                session['nongoal_rew_idx'][(session['nongoal_rew_idx'] >= lm_entry_idx[i]) & (session['nongoal_rew_idx'] <= lm_exit_idx[i])]
                for i in range(len(lm_entry_idx)) if i in session['non_goals_idx']
                and not np.any((lick_idx >= lm_entry_idx[i]) & (lick_idx <= lm_exit_idx[i]))
            ]

        elif id == 10:
            collected_licks = [
                session['test_rew_idx'][(session['test_rew_idx'] >= lm_entry_idx[i]) & (session['test_rew_idx'] <= lm_exit_idx[i])]
                for i in range(len(lm_entry_idx)) if i in session['test_idx']
                and not np.any((lick_idx >= lm_entry_idx[i]) & (lick_idx <= lm_exit_idx[i]))
            ]

        if collected_licks:
            licks[id] = np.concatenate(collected_licks).astype(int)
        else:
            licks[id] = np.array([], dtype=int)
            
    return licks

def get_first_licks(session, VR_data=None, nidaq_data=None):
    """
    Get the first lick from each type in a block of licks

    Returns:
    -------
    first_licks: Dict where the key corresponds to the lick ID
    """
    
    if ('stage' not in session) or ('3' not in session['stage'] and '4' not in session['stage'] and '5' not in session['stage'] and '6' not in session['stage']):
        raise ValueError('This function only works for T3-T6.')
    
    buffer = 1  # in case lick is right at the lm entry boundary

    # Load data if needed
    if VR_data is None:
        base_path2 = find_base_path(session['mouse'], session['date'])
        VR_data = load_session(base_path2)

    if nidaq_data is None:
        base_path = find_base_path_npz(session['mouse'], session['date'])
        nidaq_data = load_session_npz(base_path)
        
    # Get landmark entry and exit indices 
    lm_entry_idx, lm_exit_idx = get_lm_entry_exit(session, nidaq_data['position'])

    # Get lick indices
    session = get_licks(nidaq_data, session)
    lick_idx = session['lick_idx']

    # Get lick ids
    licks = get_lick_types(session, VR_data, nidaq_data)

    # Find first licks for each type 
    first_licks = {}

    for id in range(1,11):
        if id == 1:
            first_licks[id] = np.array([lick_idx[(lick_idx >= entry - buffer) & (lick_idx <= exit)][0]
                    for i, (entry, exit) in enumerate(zip(lm_entry_idx, lm_exit_idx))
                    if i in session['rewarded_landmarks'] and np.any((lick_idx >= entry - buffer) & (lick_idx <= exit))])
            
        elif id == 2:
            first_licks[id] = np.array([lick_idx[(lick_idx >= entry - buffer) & (lick_idx <= exit)][0]
                    for i, (entry, exit) in enumerate(zip(lm_entry_idx, lm_exit_idx))
                    if i in session['goals_idx'] and i not in session['rewarded_landmarks'] and np.any((lick_idx >= entry - buffer) & (lick_idx <= exit))])
            
        elif id == 3:
            first_licks[id] = np.array([lick_idx[(lick_idx >= entry - buffer) & (lick_idx <= exit)][0]
                    for i, (entry, exit) in enumerate(zip(lm_entry_idx, lm_exit_idx))
                    if i in session['non_goals_idx'] and np.any((lick_idx >= entry - buffer) & (lick_idx <= exit))])
            
        elif id == 4:
            if session['test_idx'] is not None:
                first_licks[id] = np.array([lick_idx[(lick_idx >= entry - buffer) & (lick_idx <= exit)][0]
                        for i, (entry, exit) in enumerate(zip(lm_entry_idx, lm_exit_idx))
                        if i in session['test_idx'] and np.any((lick_idx >= entry - buffer) & (lick_idx <= exit))])
            else:
                first_licks[id] = []
            
        elif id == 5:
            licks5 = []
            if 0 in session['goals_idx']:
                if np.any(lick_idx < lm_entry_idx[0]):
                    licks5.append(lick_idx[lick_idx < lm_entry_idx[0]][0])
            licks5 += [
                lick_idx[(lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1])][0]
                for i in range(len(lm_entry_idx) - 1)
                if i + 1 in session['goals_idx'] and np.any((lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1]))
            ]
            first_licks[id] = np.array(licks5)

        elif id == 6:
            licks6 = []
            if 0 in session['non_goals_idx']:
                if np.any(lick_idx < lm_entry_idx[0]):
                    licks6.append(lick_idx[lick_idx < lm_entry_idx[0]][0])
            licks6 += [
                lick_idx[(lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1])][0]
                for i in range(len(lm_entry_idx) - 1)
                if i + 1 in session['non_goals_idx'] and np.any((lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1]))
            ]
            first_licks[id] = np.array(licks6)

        elif id == 7:
            if session['test_idx'] is not None:
                licks7 = []
                if 0 in session['test_idx']:
                    if np.any(lick_idx < lm_entry_idx[0]):
                        licks7.append(lick_idx[lick_idx < lm_entry_idx[0]][0])
                licks7 += [
                    lick_idx[(lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1])][0]
                    for i in range(len(lm_entry_idx) - 1)
                    if i + 1 in session['test_idx'] and np.any((lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1]))
                ]
                first_licks[id] = np.array(licks7)
            else:
                first_licks[id] = []

        elif id == 8 or id == 9 or id == 10: 
            first_licks[id] = licks[id]

    # Concatenate all first licks together and sort them by data index
    all_first_licks = np.sort(np.concatenate([first_licks[id] for id in range(1,11)]))
    print('Number of licks considered:', len(all_first_licks))

    # Confirm the licks and imaginary licks in landmarks make sense
    all_reward_idx = np.sort(np.concatenate([session['reward_idx'], session['miss_rew_idx'], session['nongoal_rew_idx'], session['test_rew_idx']]))

    assert len (all_reward_idx) == (len(first_licks[1]) + len(first_licks[2]) + len(first_licks[3]) + \
                                    len(first_licks[4]) + len(first_licks[8]) + len(first_licks[9]) + \
                                        len(first_licks[10])), 'Some licks or rewards are missing...'
    return first_licks, all_first_licks

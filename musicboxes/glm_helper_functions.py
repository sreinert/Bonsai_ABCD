import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import find_peaks
import pickle
import seaborn as sns
import importlib as imp
import pickle as pkl
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import preprocessing.parse_session_functions as parse_session_functions
from suite2p.extraction import dcnv
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
import matplotlib.patches as patches
import cellTV.cellTV_functions as cellTV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
#suppress the warnings
import warnings
imp.reload(parse_session_functions)
warnings.filterwarnings("ignore", category=RuntimeWarning)

## collection of GLM helper functions
def create_design_matrix(session_data, params, LS_match=False):
    frame_range = np.arange(session_data['dF'].shape[1])
    session = session_data['session']
    #create a design matrix with all predictors
    predictors = {}
    if params['position']:
        position = session['position']
        #bin the position data and create a vector for each bin that is 1 if the animal was in that bin, 0 otherwise
        bins_pos = 100
        bin_edges = np.linspace(9, session['position'].max(), bins_pos+1)
        bin_indices = np.digitize(position, bin_edges) - 1  # -1 to make it zero-indexed
        position_vectors = np.zeros((len(frame_range), len(bin_edges)-1))
        for i in range(bins_pos):
            pos_frames = np.where(bin_indices == i)[0]
            # Ensure pos_frames are within the range of frame_range
            position_vectors[pos_frames, i] = 1
        # Add the position vector to the predictors dictionary
        predictors['position'] = position_vectors
    
    if params['landmarks']:
        num_lms = len(np.unique(session['lm_idx']))-1  # Exclude the 'no landmark' category
        lm_vectors = np.zeros((len(frame_range), num_lms))
        for i in range(num_lms):
            lm_frames = np.where(session['lm_idx'] == i+1)[0]
            # Ensure lm_frames are within the range of frame_range
            lm_vectors[lm_frames, i] = 1
        # Add the landmark vectors to the predictors dictionary
        predictors['landmarks'] = lm_vectors
    
    if params['goal_progress']:
    #first bin times between rewards into n bins
        bins_goal = 9
        reward_ix = session['rewards']
        goal_vec = []
        goal_progress = np.zeros((len(frame_range), bins_goal))
        for i in range(len(reward_ix)-1):
            phase_frames = np.arange(reward_ix[i], reward_ix[i+1]-1)
            # Ensure phase_frames are within the range of frame_range
            bin_edges = np.linspace(reward_ix[i], reward_ix[i+1]-1, bins_goal+1)
            bin_ix = np.digitize(phase_frames, bin_edges)

            for j in range(bins_goal):
                goal_frames = np.where(bin_ix == j+1)[0]
                target_frames = phase_frames[goal_frames]
                goal_progress[target_frames, j] = 1

        predictors['goal_progress'] = goal_progress
    
    if params['task_state']:
        reward_ix = session['rewards']
        num_states = len(np.unique(session['goal_idx']))
        state_id_vec = np.arange(num_states)
        state_id_vec = np.tile(state_id_vec, len(session['rewards'])//num_states)
        state_id_vec = state_id_vec[:-1]

        task_state = np.zeros((len(frame_range), num_states))
        for i in range(num_states):
            state_frames = np.where(state_id_vec == i)[0]
            # Ensure state_frames are within the range of frame_range
            state_on = reward_ix[state_frames]
            state_off = reward_ix[state_frames + 1]-1 if np.all(state_frames + 1 < len(reward_ix)) else reward_ix[state_frames[:-1]] + 1
            if len(state_off) < len(state_on):
                #add a last state_off if it is missing
                state_off = np.append(state_off, frame_range[-1])
            for j in range(len(state_on)):
                task_state[state_on[j]:state_off[j], i] = 1
        # Add the task state vector to the predictors dictionary
        predictors['task_state'] = task_state

    if params['latent_state']:
        if LS_match:
            num_latent_states = 3
        else:
            num_latent_states = len(np.unique(session['state_id']))
        latent_state = np.zeros((len(frame_range), num_latent_states))
        for i in range(session['num_laps']):
            lap_frames = np.where(session['lap_idx'] == i)[0]
            for j in range(num_latent_states):
                if session['state_id'][i] == j:
                    # Set the latent state for the frames in this lap
                    latent_state[lap_frames, j] = 1
        # Add the latent state vector to the predictors dictionary
        predictors['latent_state'] = latent_state

    if params['lm_ls']:
        num_lms = len(np.unique(session['lm_idx']))-1  # Exclude the 'no landmark' category
        lm_vectors = np.zeros((len(frame_range), num_lms))
        for i in range(num_lms):
            lm_frames = np.where(session['lm_idx'] == i+1)[0]
            # Ensure lm_frames are within the range of frame_range
            lm_vectors[lm_frames, i] = 1
        if LS_match:
            num_latent_states = 3
        else:
            num_latent_states = len(np.unique(session['state_id']))
        latent_state = np.zeros((len(frame_range), num_latent_states))
        for i in range(session['num_laps']):
            lap_frames = np.where(session['lap_idx'] == i)[0]
            for j in range(num_latent_states):
                if session['state_id'][i] == j:
                    # Set the latent state for the frames in this lap
                    latent_state[lap_frames, j] = 1
        interaction_terms = np.zeros((len(frame_range), num_lms * num_latent_states))
        for i in range(num_lms):
            for j in range(num_latent_states):
                interaction_terms[:, i * num_latent_states + j] = lm_vectors[:, i] * latent_state[:, j]
        predictors['lm_ls'] = interaction_terms

    if params['gp_state']:
        bins_goal = 3
        reward_ix = session['rewards']
        goal_vec = []
        goal_progress = np.zeros((len(frame_range), bins_goal))
        for i in range(len(reward_ix)-1):
            phase_frames = np.arange(reward_ix[i], reward_ix[i+1]-1)
            # Ensure phase_frames are within the range of frame_range
            bin_edges = np.linspace(reward_ix[i], reward_ix[i+1]-1, bins_goal+1)
            bin_ix = np.digitize(phase_frames, bin_edges)

            for j in range(bins_goal):
                goal_frames = np.where(bin_ix == j+1)[0]
                target_frames = phase_frames[goal_frames]
                goal_progress[target_frames, j] = 1
        
        num_states = len(np.unique(session['goal_idx']))
        state_id_vec = np.arange(num_states)
        state_id_vec = np.tile(state_id_vec, len(session['rewards'])//num_states)
        state_id_vec = state_id_vec[:-1]

        task_state = np.zeros((len(frame_range), num_states))
        for i in range(num_states):
            state_frames = np.where(state_id_vec == i)[0]
            # Ensure state_frames are within the range of frame_range
            state_on = reward_ix[state_frames]
            state_off = reward_ix[state_frames + 1]-1 if np.all(state_frames + 1 < len(reward_ix)) else reward_ix[state_frames[:-1]] + 1
            if len(state_off) < len(state_on):
                #add a last state_off if it is missing
                state_off = np.append(state_off, frame_range[-1])
            for j in range(len(state_on)):
                task_state[state_on[j]:state_off[j], i] = 1
        interaction_terms_gp = np.zeros((len(frame_range), bins_goal * num_states))
        for i in range(bins_goal):
            for j in range(num_states):
                interaction_terms_gp[:, i * num_states + j] = goal_progress[:, i] * task_state[:, j]
        predictors['gp_state'] = interaction_terms_gp

    if params['licks']:
        # Assuming licks are defined by the session['licks'] variable
        lick_frames = np.zeros((len(frame_range), 1))
        lick_times = session['licks']
        for lick in lick_times:
            if lick < frame_range[-1] and lick >= frame_range[0]:
                lick_frames[lick, 0] = 1
        # Add the lick vector to the predictors dictionary
        predictors['licks'] = lick_frames

    if params['rewards']:
        # Assuming rewards are defined by the session['rewards'] variable
        reward_frames = np.zeros((len(frame_range), 1))
        reward_times = session['rewards']
        for reward in reward_times:
            if reward < frame_range[-1] and reward >= frame_range[0]:
                reward_frames[reward, 0] = 1
        # Add the reward vector to the predictors dictionary
        predictors['rewards'] = reward_frames

    if params['speed']:
        # Assuming speed is defined by the session['speed'] variable
        speed = session['speed']
        # Bin the speed data into 10 bins
        bins_speed = 5
        bin_edges = np.linspace(0, speed.max(), bins_speed+1)
        bin_indices = np.digitize(speed, bin_edges) - 1  # -1 to make it zero-indexed
        speed_vectors = np.zeros((len(frame_range), bins_speed))
        for i in range(bins_speed):
            speed_frames = np.where(bin_indices == i)[0]
            #if speed frames are within the range of frame_range, set them to 1
            if len(speed_frames) > 0:
                speed_vectors[speed_frames, i] = 1
        # Add the speed vector to the predictors dictionary
        predictors['speed'] = speed_vectors
    
    design_matrix = np.hstack([predictors[key] for key in predictors.keys() if key in params and params[key]])
    
    stored_ids= {}
    idx_count = 0
    for key in predictors.keys(): 
        if key in params and params[key]: 
            indices = range(predictors[key].shape[1])
            indices = [idx_count + i for i in indices]
            idx_count += len(indices)
            print(f'Predictor: {key}, Indices: {indices}')
            stored_ids[key] = indices


    return design_matrix, stored_ids, predictors 

def plot_design_matrix(design_matrix, stored_ids):
    plt.figure(figsize=(15, 8))
    #inverse gray colormap so that 1s are black and 0s are white
    plt.imshow(design_matrix[:30000].T, aspect='auto', cmap='gray_r', interpolation='none')
    plt.xlabel('Frames')
    plt.ylabel('Predictors')
    plt.title('Design Matrix')
    # Add lines to separate different predictor groups based on stored_ids
    for key, indices in stored_ids.items():
        if len(indices) > 0:
            plt.axhline(y=indices[0]-0.5, color='white', linestyle='--',alpha=0.7)
            plt.text(100, indices[0]+0.5, key, color='black', fontsize=10, verticalalignment='bottom')
    plt.show()

def fit_ridge_regression(X_train, y_train, X_test, y_test):
    model = Ridge(alpha=1)
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    sse = calc_sse(y_test, model.predict(X_test))
    return model.coef_, r2, sse

def calc_sse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

def fit_reduced_model(X_train, y_train, X_test, y_test, predictor_indices):
    X_train_reduced = np.delete(X_train, predictor_indices, axis=1)
    X_test_reduced = np.delete(X_test, predictor_indices, axis=1)
    model = Ridge(alpha=1)
    model.fit(X_train_reduced, y_train)
    r2_reduced = model.score(X_test_reduced, y_test)
    sse_reduced = calc_sse(y_test, model.predict(X_test_reduced))
    return model.coef_,r2_reduced, sse_reduced

def calc_cpd(sse_full, sse_reduced):
    return (sse_reduced - sse_full) / sse_reduced

def shuffle_predictor(X_train, predictor_indices):
    X_train_shuffled = X_train.copy()
    for idx in predictor_indices:
        np.random.shuffle(X_train_shuffled[:, idx])
    return X_train_shuffled
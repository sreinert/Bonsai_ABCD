import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.gridspec as gridspec
import os, re, sys
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, resample
import pandas as pd
import yaml
import math
from math import log10, floor
import itertools
import seaborn as sns
import palettes
import importlib
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.path.join('..', 'preprocessing')))
sys.path.append(os.path.abspath(os.path.join('..', 'cellTV')))

import preprocessing.parse_session_functions_cohort2 as parse_session_functions
import cellTV.cellTV_functions_cohort2 as cellTV

importlib.reload(parse_session_functions)
importlib.reload(cellTV)

def load_dF_session_data(base_path, mouse, stage, calculate_DF_F=False):
    session_folder = [f for f in os.listdir(os.path.join(base_path, mouse)) if stage in f][0]
    sess_data_path = os.path.join(base_path, mouse, session_folder)

    imaging_path, config_path, frame_ix, date1, date2 = cellTV.get_session_folders(base_path, mouse, stage)

    save_path = Path(sess_data_path) / 'analysis'
    save_path.mkdir(parents=True, exist_ok=True)

    # Load or calculate dF/F0
    DF_F_file = os.path.join(imaging_path, 'DF_F0.npy')

    if os.path.exists(DF_F_file) and calculate_DF_F is False:
        print('DF_F0 file found. Loading...')
        
        DF_F_all = np.load(DF_F_file)
        dF = DF_F_all[:, frame_ix['valid_frames']]
        print(dF.shape)
        
    else:
        DF_F_file = os.path.join(imaging_path, 'DF_F0_valid_frames.npy')
        if os.path.exists(DF_F_file):
            print('DF_F0 file with valid frames found. Loading...')
            dF = np.load(DF_F_file)
        else:
            # TODO: incorporate this into the main analysis and ensure DF_F is the same everywhere
            f, fneu, iscell, ops, seg, frame_rate = cellTV.load_img_data(imaging_path)
            dF = cellTV.get_dff(f, fneu, frame_ix, ops)

            np.save(DF_F_file, dF)
            
    # Get session data 
    if stage in ['-t3','-t4','-t5', '-t6']:
        session = parse_session_functions.analyse_npz_pre7(mouse, date2, stage=stage, plot=False)
    else:
        session = parse_session_functions.analyse_npz(mouse, date2, plot=True)

    session['save_path'] = save_path

    return save_path, dF, session


def get_psth(data, neurons, event_idx, time_around=(-1, 3), funcimg_frame_rate=45):
    num_neurons = len(neurons)

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
    time_bins = end_frames - start_frames

    # Get indices for each event
    window = np.arange(start_frames, end_frames)
    window_indices = np.add.outer(event_idx, window).astype(int)

    # Remove last events if close to session end 
    valid_mask = window_indices[:, -1] < data.shape[1]
    valid_window_indices = window_indices[valid_mask]

    # Preallocate PSTH array
    num_events = valid_window_indices.shape[0]
    psth = np.zeros((num_neurons, num_events, time_bins))
    for n, neuron in enumerate(neurons):
        psth[n, :, :] = data[neuron, valid_window_indices]

    # Compute average PSTH across events
    average_psth = np.mean(psth, axis=1)

    return psth, average_psth


def plot_avg_psth(average_psth, event='reward', zscoring=True, time_around=(-1, 1), funcimg_frame_rate=45, save_psth=False, savepath='', filename=''):

    # Handle time window
    if isinstance(time_around, (int, float)):
        start_time = -time_around
        end_time = time_around
    elif isinstance(time_around, (tuple, list)) and len(time_around) == 2:
        start_time, end_time = time_around
    else:
        raise ValueError("time_around must be a number or a tuple/list (start, end)")

    start_frames = int(np.floor(start_time * funcimg_frame_rate))
    end_frames = int(np.ceil(end_time * funcimg_frame_rate))
    num_timebins = average_psth.shape[1]
    num_neurons = average_psth.shape[0]

    # Index corresponding to event time (0s)
    zero_bin = int(round(-start_frames))

    # Sort cells by time of max response
    sortidx = np.argsort(np.argmax(average_psth, axis=1))

    data = average_psth.copy()
    if zscoring:
        data = stats.zscore(data, axis=1)

    fig, ax = plt.subplots(figsize=(3, 4))
    im = ax.imshow(data[sortidx, :], aspect='auto')

    # Event marker line (time 0)
    ax.vlines(zero_bin - 0.5, ymin=-0.5, ymax=num_neurons - 0.5, color='k')

    ax.set_xlabel('Time (s)')
    ax.set_xticks([0, zero_bin, num_timebins - 1])
    ax.set_xticklabels([round(start_time, 2), 0, round(end_time, 2)])

    ax.set_ylabel('Neuron')
    ax.set_yticks([-0.5, num_neurons - 0.5])
    ax.set_yticklabels([0, num_neurons])
    fig.suptitle(f'{event} PSTH')

    cbar = fig.colorbar(im, ax=ax)
    vmin, vmax = im.get_clim()
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=2, fontsize=8)

    plt.tight_layout()

    if save_psth:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, f'{filename}.png'))

    return


def split_psth(psth, event_idx, event='reward', zscoring=True, time_around=1, funcimg_frame_rate=45):

    if isinstance(time_around, int):
        time_window = time_around * funcimg_frame_rate
    else:
        time_window = int(np.floor(time_around * funcimg_frame_rate))

    num_timebins = psth.shape[2]
    num_neurons = psth.shape[0]
    num_events = len(event_idx)

    # Split trials in half (randomly) to confirm event tuning
    num_sort_trials = np.floor(num_events/2).astype(int)
    event_array = np.arange(0, num_events)

    random_rew_sort = np.random.choice(event_array, num_sort_trials, replace=False)  # used for sorting
    random_rew_test = np.setdiff1d(event_array, random_rew_sort)  # used for testing

    # Average firing rates for sort trials and test trials separately
    sorting_data = np.mean(psth[:, random_rew_sort, :], axis=1)
    testing_data = np.mean(psth[:, random_rew_test, :], axis=1)

    if zscoring:
        sorting_data = stats.zscore(sorting_data, axis=1)
        testing_data = stats.zscore(testing_data, axis=1)
        # sorting_data = stats.zscore(sorting_data, axis=None)
        # testing_data = stats.zscore(testing_data, axis=None)
    
    vmin = min(np.min(sorting_data), np.min(testing_data))
    vmax = max(np.max(sorting_data), np.max(testing_data))

    sortidx = np.argsort(np.argmax(sorting_data[:, :], axis=1))

    # Plotting 
    fig = plt.figure(figsize=(6, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])  # third slot for colorbar

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    cax = fig.add_subplot(gs[2])

    im0 = ax0.imshow(sorting_data[sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)
    ax0.vlines(time_window-0.5, ymin=-0.5, ymax=num_neurons-0.5, color='k')
    ax0.set_xlabel('Time')
    ax0.set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
    if time_around == int(time_around):
        xticklabels = [int(-time_around), 0, int(time_around)]
    else:
        xticklabels = [round(-time_around, 1), 0, round(time_around, 1)]
    ax0.set_xticklabels(xticklabels)
    ax0.set_title(f'Sorting trials')

    im1 = ax1.imshow(testing_data[sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)
    ax1.vlines(time_window-0.5, ymin=-0.5, ymax=num_neurons-0.5, color='k')
    ax1.set_xlabel('Time')
    ax1.set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
    if time_around == int(time_around):
        xticklabels = [int(-time_around), 0, int(time_around)]
    else:
        xticklabels = [round(-time_around, 1), 0, round(time_around, 1)]
    ax1.set_xticklabels(xticklabels)    
    ax1.set_title(f'Testing trials')

    ax0.set_ylabel('Neuron')
    ax0.set_yticks([-0.5, num_neurons-0.5])
    ax0.set_yticklabels([0, num_neurons])

    cbar = fig.colorbar(im1, cax=cax)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    cbar.set_ticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=2, fontsize=8)

    fig.suptitle(f'{event} PSTH')
    plt.tight_layout()


def get_tuned_neurons(psth, event='reward', time_around=1, funcimg_frame_rate=45, plot_neurons=True):
    """
    Mann–Whitney U test comparing firing before and after event.
    
    Parameters
    ----------
    psth : array, shape (neurons, trials, timebins)
    event : str
        Event name
    time_around : int/float or tuple
        If int/float: window in seconds, symmetric around 0 (e.g. 1 → -1 to +1).
        If tuple: (start, end) in seconds (e.g. (-1, 2)).
    funcimg_frame_rate : int
        Imaging frame rate (Hz).
    plot_neurons : bool
        Whether to plot significant neurons.
    """
    # TODO: bootstrapping / permutation test instead? 

    # Handle time window
    if isinstance(time_around, (int, float)):
        start_time = -time_around
        end_time = time_around
    elif isinstance(time_around, (tuple, list)) and len(time_around) == 2:
        start_time, end_time = time_around
    else:
        raise ValueError("time_around must be a number or a tuple/list (start, end)")

    start_frames = int(np.floor(start_time * funcimg_frame_rate))
    end_frames = int(np.ceil(end_time * funcimg_frame_rate))
    num_neurons = psth.shape[0]

    event_frame = -start_frames  # event aligned at t=0
    before_idx = slice(0, event_frame)          # frames before event
    after_idx  = slice(event_frame, event_frame + end_frames)  # frames after event

    # Average across timebins for each trial
    before_event_firing = np.mean(psth[:, :, before_idx], axis=2)
    after_event_firing  = np.mean(psth[:, :, after_idx], axis=2)
    # print(before_event_firing.shape, after_event_firing.shape)

    # Perform the test using all trials for each neuron
    wilcoxon_stat = np.zeros((num_neurons, 1))
    wilcoxon_pval = np.zeros((num_neurons, 1))
    for n in range(num_neurons):
        wilcoxon_stat[n], wilcoxon_pval[n] = stats.wilcoxon(before_event_firing[n, :], after_event_firing[n, :]) #, method=stats.PermutationMethod(n_resamples=1000))

    # Criteria to define tuned neurons
    # 1. p-value
    criterion1 = np.where(wilcoxon_pval < 0.001)[0]   

    # 2. peak in the 1s after event > mean + 2*std of the 1s before the event
    average_psth = np.mean(psth, axis=1)
    before_event_avg_firing = average_psth[:, before_idx]
    after_event_avg_firing = average_psth[:, after_idx]
    criterion2 = np.where(np.max(after_event_avg_firing, axis=1) > (np.mean(before_event_avg_firing, axis=1) + 2 * np.std(before_event_avg_firing, axis=1)))[0]

    tuned_neurons = np.intersect1d(criterion1, criterion2)
    print(f'{len(tuned_neurons)} neurons are tuned to {event}.')

    # Plot firing for a few significant neurons
    if plot_neurons:
        for n in tuned_neurons:
            fig, ax = plt.subplots(1, 1, figsize=(2,2), sharey=True)
            ax.plot(average_psth[n, :])      
            event_frame = -start_frames  
            # ax.plot(average_psth[n, :])      
            ax.axvspan(event_frame, event_frame + end_frames, color='gray', alpha=0.5)
            ax.set_xlabel('Time')
            ax.set_xticks([event_frame + start_frames, event_frame, event_frame + end_frames])
            ax.set_xticklabels([start_time, 0, end_time])
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_ylabel('DF/F')
            ax.set_title(f'neuron {n}, p-value {float(wilcoxon_pval[n]):.5f}')

    return tuned_neurons, wilcoxon_stat, wilcoxon_pval


def plot_psth_single_neurons(psth, average_psth, neurons, time_around=(-1, 1), num_neurons=10, avg_only=False, zscoring=True, event_lick_rate=None, pvalues=None, color=None, axis=None):
    '''Plot the PSTH around events for specific neurons.'''

    num_timebins = average_psth.shape[1]
    num_events = psth.shape[1]

    if isinstance(neurons, int) or np.isscalar(neurons):
        neurons = [neurons]

    # Handle time window input
    if isinstance(time_around, (int, float)):
        start_time = -time_around
        end_time = time_around
    elif isinstance(time_around, (tuple, list)) and len(time_around) == 2:
        start_time, end_time = time_around
    else:
        raise ValueError("time_around must be a single number or a tuple/list of (start, end)")

    # z-scoring
    if zscoring:
        average_psth = stats.zscore(np.array(average_psth), axis=1)       
        psth = stats.zscore(np.array(psth), axis=2)

    for i, n in enumerate(neurons[:num_neurons]):
        if axis is None:
            fig, ax = plt.subplots(1, 1, figsize=(2,2), sharey=True)
        else: 
            ax = axis
        if not avg_only:
            for r in range(num_events):
                ax.plot(psth[n, r, :])
            linewidth = 3
        else:
            linewidth = 2

        if color == 'blue':
            label = '2/3 Alternation'
        elif color == 'red':
            label = '3/3 Discrimination'
        else:
            label = None

        # Plot PSTH
        ax.plot(average_psth[n, :], color=color if color is not None else 'black', linewidth=linewidth, label=label) 

        if avg_only and not zscoring:  # add SEM
            ax.fill_between(np.arange(num_timebins),
                            average_psth[n, :] - stats.sem(psth[n, :, :], axis=0),
                            average_psth[n, :] + stats.sem(psth[n, :, :], axis=0),
                            color=color if color is not None else 'black',
                            alpha=0.3)
        
        # Plot lick rate if available
        if event_lick_rate is not None:
            # Get mean lick rate across events 
            avg_event_lick_rate = np.mean(event_lick_rate, axis=0)
            sem_event_lick_rate = stats.sem(event_lick_rate, axis=0)

            ax2 = ax.twinx()
            ax2.plot(avg_event_lick_rate, color='orange', linestyle='-', label='Lick Rate', linewidth=2)

            ax2.fill_between(np.arange(num_timebins),
                            avg_event_lick_rate - sem_event_lick_rate,
                            avg_event_lick_rate + sem_event_lick_rate,
                            color='orange', alpha=0.3)

            # Label the second y-axis
            ax2.set_ylabel('Lick Rate (Hz)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')

        # Event alignment marker (time 0 is at -start_time in bins)
        zero_bin = int(round(-start_time / (end_time - start_time) * num_timebins))
        ax.axvspan(zero_bin, num_timebins, color='gray', alpha=0.5)

        # X-axis labels
        ax.set_xlabel('Time (s)')
        ax.set_xticks([0, zero_bin, num_timebins - 1])
        ax.set_xticklabels([round(start_time, 2), 0, round(end_time, 2)])

        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylabel(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0')

        if pvalues is not None:
            ax.set_title(f'p-value {round(pvalues[i], 3 - int(floor(log10(abs(pvalues[i])))) - 1)}')

    return


def get_tuned_neurons_shohei(DF_F, average_psth, neurons, event='reward', time_around=1, funcimg_frame_rate=45, plot_neurons=True, zscoring=True):
    # The response to an event is calculated using the mean z-scored ΔF/F calcium signal 
    # averaged over a window from 0.4 s to 1 s after event onset, baseline-subtracted using 
    # the mean z-scored ΔF/F signal during 0.5 s before event onset for each event. 
    # Neurons are classified as event-responsive if their mean response is bigger than 0.5 z-scored ΔF/F. 
    
    time_window = time_around * funcimg_frame_rate # frames
    time_before = int(np.floor(0.5 * funcimg_frame_rate))
    time_after = int(0.4 * funcimg_frame_rate)
    num_timebins = average_psth.shape[1]
    num_neurons = average_psth.shape[0]

    num_neurons = len(neurons)

    data = average_psth.copy()
    if zscoring:
        data = stats.zscore(np.array(data), axis=1)
        # data = stats.zscore(np.array(data), axis=None)

    before_firing = data[:, time_before:time_window]
    after_firing = data[:, time_window+time_after:]
    
    mean_before = np.mean(before_firing, axis=1)
    mean_after = np.mean(after_firing, axis=1)

    total_response = mean_after - mean_before

    tuned_neurons = []
    for n in range(num_neurons):
        if total_response[n] > 0.5 * np.mean(DF_F[n,:]):
            tuned_neurons.append(n)
    
    print(f'{len(tuned_neurons)} neurons are tuned to {event}.')

    if plot_neurons:
        # Plot firing for a few significant neurons
        for n in tuned_neurons[0:10]:
            fig, ax = plt.subplots(1, 1, figsize=(2,2), sharey=True)
            ax.plot(average_psth[n, :])      
            ax.axvspan(num_timebins/2, num_timebins, color='gray', alpha=0.5)
            ax.set_xlabel('Time')
            ax.set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
            ax.set_xticklabels([int(-time_around), 0, int(time_around)])
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_ylabel('DF/F')

    return tuned_neurons


def plot_avg_goal_psth(neurons, event_idxs, psths, average_psths, \
                        goals=['A','B','C','D'], time_around=1, funcimg_frame_rate=45, \
                        plot_all_neurons=False, save_plot=False, savepath='', savedir=''):
    
    time_window = time_around * funcimg_frame_rate # frames
    num_timebins = 2*time_window

    num_goals = len(goals)

    if plot_all_neurons:
        for n, neuron in enumerate(neurons):

            fig, ax = plt.subplots(1, num_goals, figsize=(10,2), sharey=True, sharex=True)
            ax = ax.ravel()
            
            for goal in range(num_goals):
                psth = psths[goal]
                avg_psth = average_psths[goal]
                event_idx = event_idxs[goal]

                for i in range(len(event_idx)):
                    ax[goal].plot(psth[n, i, :], alpha=0.5)

                ax[goal].plot(avg_psth[n, :], 'k', linewidth=2)
                ax[goal].axvspan(num_timebins / 2, num_timebins, color='gray', alpha=0.5)
                ax[goal].set_xticks([-0.5, num_timebins/2 - 0.5, num_timebins - 0.5])
                ax[goal].set_xticklabels([int(-time_around), 0, int(time_around)])
                ax[goal].spines[['right', 'top']].set_visible(False)
                ax[goal].set_title(goals[goal])

            ax[0].set_ylabel('DF/F')
            plt.suptitle(f'Neuron {neuron}')

            if save_plot:
                output_path = os.path.join(savepath, savedir)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                plt.savefig(os.path.join(output_path, f'neuron{neuron}.png'))
                plt.close()

    else:
        for n, neuron in enumerate(neurons[0:10]):

            fig, ax = plt.subplots(1, num_goals, figsize=(10,2), sharey=True, sharex=True)
            ax = ax.ravel()
            
            for goal in range(num_goals):
                psth = psths[goal]
                avg_psth = average_psths[goal]
                event_idx = event_idxs[goal]

                for i in range(len(event_idx)):
                    ax[goal].plot(psth[n, i, :], alpha=0.5)

                ax[goal].plot(avg_psth[n, :], 'k', linewidth=2)
                ax[goal].axvspan(num_timebins / 2, num_timebins, color='gray', alpha=0.2)
                ax[goal].set_xticks([-0.5, num_timebins/2 - 0.5, num_timebins - 0.5])
                ax[goal].set_xticklabels([int(-time_around), 0, int(time_around)])
                ax[goal].spines[['right', 'top']].set_visible(False)
                ax[goal].set_title(goals[goal])

            ax[0].set_ylabel('DF/F')
            plt.suptitle(f'Neuron {neuron}')
            plt.show()


def get_landmark_psth(data, neurons, event_idx, num_landmarks=10, time_around=1, funcimg_frame_rate=45):
    '''This function is similar to get_psth, but the average PSTH is calculated for each landmark separately.'''

    # Handle time window
    if isinstance(time_around, (int, float)):
        start_time = -time_around
        end_time = time_around
    elif isinstance(time_around, (tuple, list)) and len(time_around) == 2:
        start_time, end_time = time_around
    else:
        raise ValueError("time_around must be a number or a tuple/list (start, end)")
    
    # Convert to frames
    start_frames = int(np.floor(start_time * funcimg_frame_rate))
    end_frames = int(np.ceil(end_time * funcimg_frame_rate))
    window = np.arange(start_frames, end_frames)
    num_timebins = len(window)
    
    # Handle neurons input
    if isinstance(neurons, int) or np.isscalar(neurons):
        neurons = [neurons]
    num_neurons = len(neurons)

    # Build window indices (events × timebins)
    window_indices = np.add.outer(event_idx, window).astype(int)  

    # Remove last events if close to session end 
    valid_mask = window_indices[:, -1] < data.shape[1]
    valid_window_indices = window_indices[valid_mask]
    num_events = valid_window_indices.shape[0]

    # Preallocate PSTH array
    psth = np.zeros((num_neurons, num_events, num_timebins))
    for n, neuron in enumerate(neurons):
        psth[n, :, :] = data[neuron, valid_window_indices]

    # Average PSTH for all events per landmark
    average_landmark_psth = np.zeros([num_neurons, num_landmarks, num_timebins])
    for i in range(num_landmarks):
        average_landmark_psth[:, i, :] = np.mean(psth[:, i::num_landmarks, :], axis=1)

    return psth, average_landmark_psth


def get_landmark_id_psth(data, neurons, event_idx, session, num_landmarks=2, time_around=1, funcimg_frame_rate=45):
    '''This function is similar to get_psth, but the average PSTH is calculated for each landmark separately.'''

    assert num_landmarks == 2, 'This function only deals with 2 landmark sequences.'

    if isinstance(time_around, int):
        time_window = time_around * funcimg_frame_rate
    else:
        time_window = int(np.floor(time_around * funcimg_frame_rate))

    num_timebins = 2*time_window
    num_neurons = len(neurons)
    num_events = len(event_idx)

    window_indices = np.add.outer(event_idx, np.arange(-time_window, time_window)).astype(int)  

    psth = np.zeros((num_neurons, num_events, num_timebins))
    for n, neuron in enumerate(neurons):
        psth[n, :, :] = data[neuron, window_indices]

    # Average PSTH for all events per landmark
    average_landmark_psth = np.zeros([num_neurons, num_landmarks, num_timebins])
    for i in range(num_landmarks):
        if i == 0:
            average_landmark_psth[:, i, :] = np.mean(psth[:, session['goals_idx'], :], axis=1)
        elif i == 1:
            average_landmark_psth[:, i, :] = np.mean(psth[:, session['non_goals_idx'], :], axis=1)

    return psth, average_landmark_psth


def plot_avg_landmark_psth(neurons, psth, average_psth, num_landmarks=10, time_around=1, funcimg_frame_rate=45, \
                           plot_all_neurons=False, plot_trials=False, save_plot=False, savepath='', savedir=''):
    
    # Handle time window properly
    if isinstance(time_around, (int, float)):
        start_time = -time_around
        end_time = time_around
    elif isinstance(time_around, (tuple, list)) and len(time_around) == 2:
        start_time, end_time = time_around
    else:
        raise ValueError("time_around must be a number or a (start, end) tuple/list")

    start_frames = int(np.floor(start_time * funcimg_frame_rate))
    end_frames = int(np.ceil(end_time * funcimg_frame_rate))
    num_timebins = end_frames - start_frames

    # Choose how many neurons to plot 
    neurons_to_plot = neurons if plot_all_neurons else neurons[:10]

    for n, neuron in enumerate(neurons_to_plot):   # neuron idx in psth array, not neuron id
        fig, ax = plt.subplots(1, 10, figsize=(15, 2), sharey=True, sharex=True)
        ax = ax.ravel()

        for i in range(num_landmarks):
            if plot_trials:
                ax[i].plot(psth[neuron, i::num_landmarks, :].T, alpha=0.5)  
            ax[i].plot(average_psth[neuron, i, :], 'k', linewidth=3)
            ax[i].axvspan(-start_frames, -start_frames + end_frames, color='gray', alpha=0.5)
            ax[i].set_xlabel('Time')
            ax[i].set_xticks([0, -start_frames, num_timebins])
            ax[i].set_xticklabels([start_time, 0, end_time])
            ax[i].spines[['right', 'top']].set_visible(False)

        ax[0].set_ylabel('DF/F')
        plt.tight_layout()
        plt.suptitle(f'Neuron {neuron}')
    
        if save_plot:
            output_path = os.path.join(savepath, savedir)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.savefig(os.path.join(output_path, f'neuron{neuron}.png'))
            plt.close()


def plot_landmark_psth_map(average_psth, session, zscoring=True, sorting_lm=0, num_landmarks=10, time_around=1, funcimg_frame_rate=45, save_plot=False, savepath='', savedir='', filename=''):
    '''Plot firing maps of all selected neurons for all landmarks, sorted by specific landmark.'''

    if sorting_lm >= num_landmarks:
        raise ValueError(f'The sorting landmark should be one of the {num_landmarks} landmarks.')
    
    if isinstance(time_around, int):
        time_window = time_around * funcimg_frame_rate
    else:
        time_window = int(np.floor(time_around * funcimg_frame_rate))

    num_timebins = average_psth.shape[2]

    fig, ax = plt.subplots(1, num_landmarks, figsize=(num_landmarks*1.5+2,3), sharey=True, sharex=True)
    ax = ax.ravel()

    data = average_psth.copy()
    if zscoring:
        data = stats.zscore(data, axis=1)
        # data = stats.zscore(data, axis=None)

    vmin = min([np.nanmin(data)])
    vmax = max([np.nanmax(data)])

    sortidx = np.argsort(np.argmax(data[:, sorting_lm, :], axis=1))

    for i in range(num_landmarks):
        img = ax[i].imshow(data[sortidx, i, :], aspect='auto', vmin=vmin, vmax=vmax)
        ax[i].vlines(time_window-0.5, ymin=-0.5, ymax=data.shape[0]-0.5, color='k', linewidth=0.5)
        ax[i].set_xlabel('Time')
        ax[i].set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
        if time_around == int(time_around):
            xticklabels = [int(-time_around), 0, int(time_around)]
        else:
            xticklabels = [round(-time_around, 1), 0, round(time_around, 1)]
        ax[i].set_xticklabels(xticklabels)
        ax[i].spines[['right', 'top']].set_visible(False)
        if num_landmarks == 10:
            ax[i].set_title(f'{i+1}')
        else:
            lm = session['all_lms'][session['goals_idx'][0]] if i == 0 else session['all_lms'][session['non_goals_idx'][0]] 
            ax[i].set_title(f'{lm+1}')

    ax[0].set_yticks([-0.5, data.shape[0]-0.5])
    ax[0].set_yticklabels([0, data.shape[0]])
    ax[0].set_ylabel('Neuron', labelpad=-10)

    cbar = fig.colorbar(img, ax=ax.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_ticks([vmin, vmax])
    cbar.ax.set_yticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=10, fontsize=8)

    # plt.tight_layout()

    if save_plot:
        output_path = os.path.join(savepath, savedir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, f'{filename}.png'))
        plt.show()


def plot_goal_psth_map(average_psths, zscoring=True, sorting_goal=1, time_around=1, funcimg_frame_rate=45, save_plot=False, savepath='', savedir='', filename=''):
    '''Plot firing maps of all selected neurons for each goal, sorted by specific goal.'''

    num_goals = len(average_psths)
    if num_goals == 4:
        goals = ['A','B','C','D']
    else:
        goals = ['A','B']

    if sorting_goal not in average_psths:
        raise ValueError(f'The sorting landmark should be one of the {num_goals} landmarks.')
    
    time_window = time_around * funcimg_frame_rate # frames
    num_timebins = 2*time_window

    data = average_psths.copy()
    if zscoring:
        for goal in data.keys():
            data[goal] = stats.zscore(data[goal], axis=1)
            # data[goal] = stats.zscore(data[goal], axis=None)

    # Find global vmin and vmax across all goals
    vmin = min([np.nanmin(arr) for arr in data.values()])
    vmax = max([np.nanmax(arr) for arr in data.values()])

    im = [[] for _ in range(num_goals)]
    fig, ax = plt.subplots(1, num_goals, figsize=(3*num_goals, 4), sharey=True, sharex=True)
    ax = ax.ravel()

    sortidx = np.argsort(np.argmax(data[sorting_goal], axis=1))  # expects a dict with keys = goals

    for i, goal in enumerate(sorted(data.keys())):
        im[i] = ax[i].imshow(data[goal][sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)
        ax[i].vlines(time_window-0.5, ymin=-0.5, ymax=data[goal].shape[0]-0.5, color='k', linewidth=0.5)
        ax[i].set_xlabel('Time')
        ax[i].set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
        ax[i].set_xticklabels([int(-time_around), 0, int(time_around)])
        ax[i].spines[['right', 'top']].set_visible(False)
        ax[i].set_title(goals[i])

    ax[0].set_yticks([-0.5, data[goal].shape[0]-0.5])
    ax[0].set_yticklabels([0, data[goal].shape[0]])
    ax[0].set_ylabel('Neuron')

    cbar = fig.colorbar(im[-1], ax=fig.axes, shrink=0.6)

    cbar.set_ticks([vmin, vmax])
    cbar.ax.set_yticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=10, fontsize=8)
    
    if save_plot:
        output_path = os.path.join(savepath, savedir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, f'{filename}.png'))
        plt.show()


def plot_all_sessions_goal_psth_map(all_average_psths, conditions, zscoring=True, ref_session=0, sorting_goal=1, time_around=1, funcimg_frame_rate=45, save_plot=False, savepath='', savedir='', filename=''):
    '''Plot firing maps for all sessions and each goal, sorted by a specific goal. 
    If there is one goal per session, or the average across goals, it behaves like plot_condition_psth_map.'''

    time_window = time_around * funcimg_frame_rate # frames
    num_timebins = 2*time_window

    num_sessions = len(all_average_psths)

    # Copy and optionally z-score data
    data = []
    goals_per_session = [[] for _ in range(num_sessions)]
    if isinstance(all_average_psths, list):
        for s, session in enumerate(all_average_psths):
            if isinstance(session, dict):
                session_data = {}
                for goal in session.keys():
                    session_data[goal] = stats.zscore(session[goal], axis=1) if zscoring else session[goal]
                    # session_data[goal] = stats.zscore(session[goal], axis=None) if zscoring else session[goal]
                data.append(session_data)
            else:  # transform data to follow the same structure
                session_data = {}
                session_data[1] = stats.zscore(session, axis=1) if zscoring else session
                # session_data[1] = stats.zscore(session, axis=None) if zscoring else session
                data.append(session_data)
            
            goals_per_session[s] = list(session_data.keys())

    elif isinstance(all_average_psths, dict):
        # Flatten the data
        for session_id, session in all_average_psths.items():  
            if isinstance(session, dict):
                assert sorting_goal in all_average_psths[ref_session], 'This goal does not exist in the reference session.'

                session_data = {}
                for goal in session.keys():
                    session_data[goal] = stats.zscore(session[goal], axis=1) if zscoring else session[goal]
                    # session_data[goal] = stats.zscore(session[goal], axis=None) if zscoring else session[goal]
                data.append(session_data)

            else:  # transform data to follow the same structure
                session_data = {}
                session_data[1] = stats.zscore(session, axis=1) if zscoring else session
                # session_data[1] = stats.zscore(session, axis=None) if zscoring else session
                data.append(session_data)

        goals_per_session = [sorted(data[s].keys()) for s in range(num_sessions)]

    # Compute global vmin/vmax
    vmin = min([np.nanmin(session[goal]) for session in data for goal in session.keys()])
    vmax = max([np.nanmax(session[goal]) for session in data for goal in session.keys()])

    # Sort neurons consistently across sessions (using sorting_goal)
    sortidx = np.argsort(np.argmax(data[ref_session][sorting_goal], axis=1))  # reference the first session for sorting

    # === Plotting ===
    # Set up figure
    max_goals = max(len(goals) for goals in goals_per_session)
    goal_label_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', '1': 'A', '2': 'B', '3': 'C', '4': 'D', 'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}
    protocol_nums = sorted(set([cond.split()[0] for cond in conditions]))
    ylabel = [f'{protocol_nums[s]}\nNeuron' if max_goals > 1 else 'Neuron' for s in range(num_sessions)]
    # titles = [protocol_nums[s] for s in range(num_sessions)]
    titles = [conditions[s] for s in range(num_sessions)]

    if max_goals == 1: 
        nrows = 1
        ncols = num_sessions
    else:
        nrows = num_sessions
        ncols = max_goals
        
    fig, ax = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows), sharex=True, sharey=True)

    # if num_sessions == 1 or max_goals == 1:
    ax = np.atleast_2d(ax)
    ax = np.array(ax)

    for s in range(num_sessions):
        for g, goal in enumerate(goals_per_session[s]):
            if max_goals == 1:  # one row, multiple columns
                row = 0
                col = s  
            else:
                row = s
                col = g  
            ax[row, col].imshow(data[s][goal][sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)
            ax[row, col].vlines(time_window-0.5, ymin=-0.5, ymax=data[s][goal].shape[0]-0.5, color='k', linewidth=0.5)
            ax[row, col].set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
            if time_around == int(time_around):
                xticklabels = [int(-time_around), 0, int(time_around)]
            else:
                xticklabels = [round(-time_around, 1), 0, round(time_around, 1)]
            ax[row, col].set_xticklabels(xticklabels)
            ax[row, col].spines[['right', 'top']].set_visible(False)
            if max_goals != 1:
                ax[row, col].set_title(goal_label_map.get(goal, str(goal)))
            else:
                ax[row, col].set_title(titles[s])
            
        ax[row,0].set_ylabel(ylabel[s], labelpad=-5)
        ax[row,0].set_yticks([-0.5, data[ref_session][goals_per_session[0][0]].shape[0]-0.5])  
        ax[row,0].set_yticklabels([0, data[ref_session][goals_per_session[0][0]].shape[0]])
            
        # Hide unused axes in that row
        for g_unused in range(len(goals_per_session[s]), max_goals):
            ax[s, g_unused].axis('off')

    cbar = fig.colorbar(ax[0,0].images[0], ax=ax.ravel().tolist(), shrink=0.6)
    cbar.set_ticks([vmin, vmax])
    cbar.ax.set_yticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=0, fontsize=8)

    if save_plot:
        output_path = os.path.join(savepath, savedir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, f'{filename}.png'))
    plt.show()


def plot_condition_psth_map(average_psths, conditions, zscoring=True, time_around=1, funcimg_frame_rate=45, save_plot=False, savepath='', savedir=''):
    '''Compare average PSTH map across different conditions.'''

    time_window = time_around * funcimg_frame_rate # frames
    # num_timebins = 2*time_window
    num_timebins = average_psths[0].shape[1]
    num_neurons = average_psths[0].shape[0]

    data = [[] for i in range(len(conditions))]
    for i in range(len(conditions)):
        data[i] = average_psths[i].copy()
        if zscoring:
            data[i] = stats.zscore(data[i], axis=1)
            # data[i] = stats.zscore(data[i], axis=None)

    # Find global vmin and vmax across all conditions
    vmin = min([np.nanmin(d) for d in data if d.size > 0])
    vmax = max([np.nanmax(d) for d in data if d.size > 0])

    # === Plotting ===
    for c, condition in enumerate(conditions):
        sortidx = np.argsort(np.argmax(data[c], axis=1))  # Sort by different conditions
        
        im = [[] for _ in range(len(conditions))]
        fig, ax = plt.subplots(1, len(conditions), figsize=(3*len(conditions),3), sharex=True, sharey=True)
        ax = ax.ravel()
        
        for i in range(len(conditions)):
            im[i] = ax[i].imshow(data[i][sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)    
            ax[i].set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
            if time_around == int(time_around):
                xticklabels = [int(-time_around), 0, int(time_around)]
            else:
                xticklabels = [round(-time_around, 1), 0, round(time_around, 1)]
            ax[i].set_xticklabels(xticklabels, fontsize=8)
            ax[i].spines[['right', 'top']].set_visible(False)
            ax[i].set_xlabel('Time', fontsize=8)
            ax[i].set_title(f'{conditions[i]}', fontsize=10)
            ax[i].vlines(time_window-0.5, ymin=-0.5, ymax=num_neurons-0.5, color='k')
        
        ax[0].set_yticks([-0.5, num_neurons-0.5])
        ax[0].set_yticklabels([0, num_neurons], fontsize=8)
        ax[0].set_ylabel('Neuron', fontsize=8, labelpad=-5)

        cbar = fig.colorbar(im[-1], ax=ax.ravel().tolist(), shrink=0.6)
        cbar.set_ticks([vmin, vmax])
        cbar.ax.set_yticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
        cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=10, fontsize=8)

        plt.suptitle(f'Sorting by {condition} trials', fontsize=10)

        if save_plot:
            output_path = os.path.join(savepath, savedir)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.savefig(os.path.join(output_path, f'{condition}_sorting.png'))
        plt.show()
        

def get_rolling_map_correlation(average_psths, conditions, population=False, zscoring=True, color_scheme=None, ax=None, save_plot=False, savepath='', savedir='', filename=''):
    '''
    Get the firing map correlation among different conditions against a reference. 
    The correlation for the reference is calculated by randomly selecting half the trials.
    If population is True, the correlation is computed across the entire activity map. Otherwise it is calculated on a neuron-by-neuron basis.
    NOTE: The reference is the index of the data if the data are either a list or a nested dict (will get flattened into a list), but it is a key of the data if the data are a dict. 
    '''
    num_neurons = average_psths[0].shape[0]
    num_windows = average_psths[0].shape[1]
    num_timebins = average_psths[0].shape[2]
    num_conditions = len(conditions)

    if zscoring:
        average_psths = [stats.zscore(psth, axis=2) for psth in average_psths]
    
    # 1. Within-condition correlations (rolling across windows)
    within_corrs = [[[] for _ in range(num_windows - 1)] for _ in range(num_conditions)]

    # 2. Across-condition correlations (between same windows)
    condition_pairs = list(itertools.combinations(range(num_conditions), 2))
    across_corrs = [[[] for _ in range(num_windows)] for _ in range(len(condition_pairs))]

    # Calculate within-condition rolling correlations
    for c in range(num_conditions):
        for i in range(num_windows - 1):
            if population:
                for t in range(num_timebins):
                    v1 = average_psths[c][:, i, t]
                    v2 = average_psths[c][:, i + 1, t]
                    if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                        r, _ = stats.pearsonr(v1, v2)
                        within_corrs[c][i].append(r)
                    else:
                        within_corrs[c][i].append(np.nan)
            else:
                for n in range(num_neurons):
                    v1 = average_psths[c][n, i, :]
                    v2 = average_psths[c][n, i + 1, :]
                    if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                        r, _ = stats.pearsonr(v1, v2)
                        within_corrs[c][i].append(r)
                    else:
                        within_corrs[c][i].append(np.nan)

    # Calculate across-condition correlations (same window index)
    for pair_idx, (c1, c2) in enumerate(condition_pairs):
        for i in range(num_windows):
            if population:
                for t in range(num_timebins):
                    v1 = average_psths[c1][:, i, t]
                    v2 = average_psths[c2][:, i, t]
                    if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                        r, _ = stats.pearsonr(v1, v2)
                        across_corrs[pair_idx][i].append(r)
                    else:
                        across_corrs[pair_idx][i].append(np.nan)
            else:
                for n in range(num_neurons):
                    v1 = average_psths[c1][n, i, :]
                    v2 = average_psths[c2][n, i, :]
                    if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                        r, _ = stats.pearsonr(v1, v2)
                        across_corrs[pair_idx][i].append(r)
                    else:
                        across_corrs[pair_idx][i].append(np.nan)

    # === Plotting ===
    # 1. Each neuron correlation trace & mean
    fig, ax = plt.subplots(1, len(within_corrs), figsize=(8,3))
    ax = ax.ravel()

    for i in range(len(within_corrs)):
        mean_across_neurons = [np.mean(within_corrs[i][w]) for w in range(len(within_corrs[i]))]
        ax[i].plot(within_corrs[i])
        ax[i].plot(mean_across_neurons, color='black')
        ax[i].set_title(f"{conditions[i]} vs {conditions[i]}")

    fig, ax = plt.subplots(1, len(across_corrs), figsize=(4,3))
    if len(across_corrs) > 1:
        ax = ax.ravel()
    for i in range(len(across_corrs)):
        mean_across_neurons = [np.mean(across_corrs[i][w]) for w in range(len(across_corrs[i]))]
        if len(across_corrs) > 1:
            ax.plot(across_corrs[i])
            ax.plot(mean_across_neurons, color='black')
            ax.set_title(f"{conditions[int(condition_pairs[0][0])]} vs {conditions[int(condition_pairs[0][1])]}")
        else:
            ax.plot(across_corrs[i])
            ax.plot(mean_across_neurons, color='black')
            ax.set_title(f"{conditions[int(condition_pairs[0][0])]} vs {conditions[int(condition_pairs[0][1])]}")

    # 2. Mean +/- sem correlation trace 
    fig, ax = plt.subplots(1, len(within_corrs) + len(across_corrs), figsize=(12,3), sharey=True, sharex=True)
    ax = ax.ravel()

    k = 0 
    for i in range(len(within_corrs)):
        mean_across_neurons = np.array([np.mean(within_corrs[i][w]) for w in range(len(within_corrs[i]))])
        sem_across_neurons = stats.sem(np.array([within_corrs[i][w] for w in range(len(within_corrs[i]))]), axis=1)

        ax[k].fill_between(np.arange(len(within_corrs[i])),
                        mean_across_neurons - sem_across_neurons,
                        mean_across_neurons + sem_across_neurons,
                        color='black',
                        alpha=0.3)
        # ax[i].plot(within_corrs[i])
        ax[k].plot(mean_across_neurons, color='black')
        ax[k].set_title(f"{conditions[i]} vs {conditions[i]}")
        ax[k].set_xlabel('Lap block')
        k += 1

    for j in range(len(across_corrs)):
        mean_across_neurons = np.array([np.mean(across_corrs[j][w]) for w in range(len(across_corrs[j]))])
        sem_across_neurons = stats.sem(np.array([across_corrs[j][w] for w in range(len(across_corrs[j]))]), axis=1)

        ax[k].plot(mean_across_neurons, color='black')
        ax[k].fill_between(np.arange(len(across_corrs[j])),
                    mean_across_neurons - sem_across_neurons,
                    mean_across_neurons + sem_across_neurons,
                    color='black',
                    alpha=0.3)
        ax[k].set_title(f"{conditions[int(condition_pairs[0][0])]} vs {conditions[int(condition_pairs[0][1])]}")
        ax[k].set_xlabel('Lap block')
        k += 1

    # TODO: add saving options
        
    return within_corrs, across_corrs, condition_pairs


def get_window_similarity_matrix(average_psths, conditions, population=False, zscoring=True, plot=True):
    """
    Compute full window-by-window correlation matrices for each condition.
    Returns one similarity matrix per condition.

    Parameters:
        average_psths: list of np.arrays (num_neurons x num_windows x num_timebins) per condition
        conditions: list of condition names (same order as average_psths)
        population: if True, correlate population vectors, otherwise per-neuron average
        zscoring: whether to z-score each neuron's time series before computing similarity
        plot: whether to show the matrices

    Returns:
        similarity_matrices: list of np.arrays (num_windows x num_windows) for each condition
    """
    num_neurons = average_psths[0].shape[0]
    num_windows = average_psths[0].shape[1]
    num_timebins = average_psths[0].shape[2]
    num_conditions = len(conditions)

    if zscoring:
        average_psths = [stats.zscore(psth, axis=2) for psth in average_psths]

    similarity_matrices = []

    # Compute within condition similarity matrix
    for c, psth in enumerate(average_psths):

        if population:
            # Store one sim_matrix per timebin, then average
            sim_matrix_all_timebins = np.full((num_timebins, num_windows, num_windows), np.nan)

            for i in range(num_windows):
                for j in range(num_windows):
                    for t in range(num_timebins):
                        v1 = psth[:, i, t]
                        v2 = psth[:, j, t]
                        if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                            r, _ = stats.pearsonr(v1, v2)
                            sim_matrix_all_timebins[t, i, j] = r
                        else:
                            sim_matrix_all_timebins[t, i, j] = np.nan
                    
            # Average across timebins
            sim_matrix = np.nanmean(sim_matrix_all_timebins, axis=0)

        else:
            # Store one sim_matrix per neuron, then average
            sim_matrix_all_neurons = np.full((num_neurons, num_windows, num_windows), np.nan)

            for i in range(num_windows):
                for j in range(num_windows):
                    for n in range(num_neurons):
                        v1 = psth[n, i, :]
                        v2 = psth[n, j, :]
                        if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                            r, _ = stats.pearsonr(v1, v2)
                            sim_matrix_all_neurons[n, i, j] = r
                        else:
                            sim_matrix_all_neurons[n, i, j] = np.nan

            # Average across neurons
            sim_matrix = np.nanmean(sim_matrix_all_neurons, axis=0)

        similarity_matrices.append(sim_matrix)

    # Set diagonal to nan to avoid skewing the colormap
    for sim_matrix in similarity_matrices:
        np.fill_diagonal(sim_matrix, np.nan)
        
    # Compute across condition similarity matrix
    condition_pairs = list(itertools.combinations(range(num_conditions), 2))
    for pair_idx, (c1, c2) in enumerate(condition_pairs):

        if population:
            # Store one sim_matrix per timebin, then average
            sim_matrix_all_timebins = np.full((num_timebins, num_windows, num_windows), np.nan)

            for i in range(num_windows):
                for j in range(num_windows):                    
                    for t in range(num_timebins):
                        v1 = average_psths[c1][:, i, t]
                        v2 = average_psths[c2][:, j, t]
                        if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                            r, _ = stats.pearsonr(v1, v2)
                            sim_matrix_all_timebins[t, i, j] = r
                        else:
                            sim_matrix_all_timebins[t, i, j] = np.nan
                    
            # Average across timebins
            sim_matrix = np.nanmean(sim_matrix_all_timebins, axis=0)

        else:
            # Store one sim_matrix per neuron, then average
            sim_matrix_all_neurons = np.full((num_neurons, num_windows, num_windows), np.nan)

            for i in range(num_windows):
                for j in range(num_windows): 
                    for n in range(num_neurons):
                        v1 = average_psths[c1][n, i, :]
                        v2 = average_psths[c2][n, j, :]
                        if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                            r, _ = stats.pearsonr(v1, v2)
                            sim_matrix_all_neurons[n, i, j] = r
                        else:
                            sim_matrix_all_neurons[n, i, j] = np.nan

            # Average across neurons
            sim_matrix = np.nanmean(sim_matrix_all_neurons, axis=0)

        similarity_matrices.append(sim_matrix)

        # Optional plot
        if plot:
            # Individual colormaps
            fig, ax = plt.subplots(1, len(similarity_matrices), figsize=(12,4), sharex=True, sharey=True)
            ax = ax.ravel()
            
            for i, sim_matrix in enumerate(similarity_matrices):
                vmax = np.round(np.nanmax(sim_matrix), 2)
                if vmax < 1e-3:
                    vmax = 1e-3
                vmin = -vmax
                im = ax[i].imshow(sim_matrix, vmin=vmin, vmax=vmax, cmap='bwr', origin='lower')
                
                cb = fig.colorbar(im, ax=ax[i], shrink=0.8, ticks=[vmin, vmax])  
                cb.set_label('Correlation (r)', labelpad=-5)

                if i < len(average_psths):
                    ax[i].set_title(f"{conditions[i]} vs {conditions[i]}")
                else:
                    ax[i].set_title(f"{conditions[int(condition_pairs[0][0])]} vs {conditions[int(condition_pairs[0][1])]}")
                ax[i].set_xlabel("Lap block")
                ax[i].set_ylabel("Lap block")
            plt.tight_layout()

            # Global colormap
            all_values = np.concatenate([sim[~np.isnan(sim)].flatten() for sim in similarity_matrices])
            vmax = np.round(np.max(all_values), 2)
            vmin = -vmax
            
            fig, ax = plt.subplots(1, len(similarity_matrices), figsize=(12,4), sharex=True, sharey=True)
            ax = ax.ravel()
            
            for i, sim_matrix in enumerate(similarity_matrices):
                im = ax[i].imshow(sim_matrix, vmin=vmin, vmax=vmax, cmap='bwr', origin='lower')

                if i < len(average_psths):
                    ax[i].set_title(f"{conditions[i]} vs {conditions[i]}")
                else:
                    ax[i].set_title(f"{conditions[int(condition_pairs[0][0])]} vs {conditions[int(condition_pairs[0][1])]}")
                ax[i].set_xlabel("Lap block")
                ax[i].set_ylabel("Lap block")
            # plt.tight_layout()
            fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.8, label='Correlation (r)', ticks=[vmin, vmax])

    return similarity_matrices


def get_map_correlation(psths, average_psths, conditions, population=False, zscoring=True, reference=0, color_scheme=None, ax=None, save_plot=False, savepath='', savedir='', filename=''):
    '''
    Get the firing map correlation among different conditions against a reference. 
    The correlation for the reference is calculated by randomly selecting half the trials.
    If population is True, the correlation is computed across the entire activity map. Otherwise it is calculated on a neuron-by-neuron basis.
    NOTE: The reference is the index of the data if the data are either a list or a nested dict (will get flattened into a list), but it is a key of the data if the data are a dict. 
    '''
    # Check data format
    if isinstance(average_psths, list):
        if reference > len(conditions):
            raise ValueError('The reference data should be within the range of input average PSTHs.')
    
        average_psth_data = []
        psth_data = []
        # psth_data = [psths[c] for c in range(len(conditions))]
        if zscoring:
            # average_psth_data = stats.zscore(np.array(average_psth_data), axis=2)
            for c in range(len(conditions)):
                average_psth_data.append(stats.zscore(np.array(average_psths[c]), axis=1))
                # average_psth_data.append(stats.zscore(np.array(average_psths[c]), axis=None))
                # psth_data = stats.zscore(np.array(psth_data), axis=2)
                psth_data.append(stats.zscore(np.array(psths[c]), axis=2))
                # psth_data.append(stats.zscore(np.array(psths[c]), axis=None))
        else: 
            average_psth_data = [average_psths[c] for c in range(len(conditions))]
            psth_data = [psths[c] for c in range(len(conditions))]

        data_indices = np.arange(0, len(conditions))
        ref_cond = reference

    elif isinstance(average_psths, dict):
        first_entry = next(iter(average_psths))  

        if isinstance(average_psths[first_entry], dict):
            # Flatten all data: [(session 0 goal A), (session 0 goal B), ..., (session 1 goal A), ...]
            average_psth_data = []  
            psth_data = []  
            for s in average_psths.keys():
                for goal in average_psths[s].keys():  
                    d = average_psths[s][goal]
                    ref = psths[s][goal]
                    if zscoring:
                        d = stats.zscore(d, axis=1)  
                        # d = stats.zscore(d, axis=None)  
                        ref = stats.zscore(ref, axis=2)
                        # ref = stats.zscore(ref, axis=None)
                    average_psth_data.append(d)
                    psth_data.append(ref)

            assert len(average_psth_data) == len(conditions), 'The length of the input data does not match the number of conditions.'
            
            # Create array of indexing into the data 
            data_indices = np.arange(0, len(average_psth_data))
            if reference not in data_indices:
                raise ValueError(f'Reference condition {reference} should be within the range of input average PSTHs.')
            ref_cond = reference
            
        else:
            average_psth_data = average_psths.copy()
            psth_data = psths.copy()
            if zscoring:
                for i in average_psth_data.keys():  
                    average_psth_data[i] = stats.zscore(average_psth_data[i], axis=1)
                    # average_psth_data[i] = stats.zscore(average_psth_data[i], axis=None)
                    psth_data[i] = stats.zscore(psth_data[i], axis=2)
                    # psth_data[i] = stats.zscore(psth_data[i], axis=None)

            data_indices = list(average_psth_data.keys())
            if reference not in average_psth_data.keys():
                raise ValueError(f'Reference condition {reference} should be one of the keys of the input dict.')
            ref_cond = data_indices.index(reference)

    num_neurons = average_psth_data[reference].shape[0]
    num_timebins = average_psth_data[reference].shape[1]
    
    corrs = [[] for c in data_indices]

    # Split reference PSTH data into random half trials 
    num_sort_trials = np.floor(psth_data[reference].shape[1]/2).astype(int)
    event_array = np.arange(0, psth_data[reference].shape[1])

    random_rew_sort = np.random.choice(event_array, num_sort_trials, replace=False)  # used for sorting
    random_rew_test = np.setdiff1d(event_array, random_rew_sort)  # used for testing

    sorting_data = np.mean(psth_data[reference][:, random_rew_sort, :], axis=1)
    testing_data = np.mean(psth_data[reference][:, random_rew_test, :], axis=1)

    # Calculate correlations
    for c, idx in enumerate(data_indices):
        if population is True:
            for t in range(num_timebins):
                if idx == reference:
                    if np.all(np.isfinite(sorting_data[:,t])) and np.all(np.isfinite(testing_data[:,t])):
                        r, _ = stats.pearsonr(sorting_data[:,t], testing_data[:,t])
                        corrs[c].append(r)
                    else:
                        corrs[c].append(np.nan)
                else:
                    if np.all(np.isfinite(average_psth_data[reference][:,t])) and np.all(np.isfinite(average_psth_data[idx][:,t])):
                        r, _ = stats.pearsonr(average_psth_data[reference][:,t], average_psth_data[idx][:,t])
                        corrs[c].append(r)
                    else:
                        corrs[c].append(np.nan)
        else:
            for n in range(num_neurons):
                if idx == reference:
                    if np.all(np.isfinite(sorting_data[n])) and np.all(np.isfinite(testing_data[n])):
                        r, _ = stats.pearsonr(sorting_data[n], testing_data[n])
                        corrs[c].append(r)
                    else:
                        corrs[c].append(np.nan)
                else:
                    if np.all(np.isfinite(average_psth_data[reference][n])) and np.all(np.isfinite(average_psth_data[idx][n])):
                        r, _ = stats.pearsonr(average_psth_data[reference][n], average_psth_data[idx][n])
                        corrs[c].append(r)
                    else:
                        corrs[c].append(np.nan)
    
    # Convert to numpy arrays
    for c in range(len(conditions)):
        corrs[c] = np.array(corrs[c])

    # === Plotting ===
    # Set up labels
    labels = []
    for i, cond in enumerate(conditions):
        if isinstance(average_psths, list):
            labels.append(f"{cond}\nvs\n{conditions[ref_cond]}")
        elif isinstance(average_psths, dict):
            if len(cond) > 10:
                labels.append(f"{cond}\nvs\n{conditions[ref_cond]}")
            else:
                labels.append(f"{cond} vs {conditions[ref_cond]}")

    if color_scheme is None:
        color_scheme = sns.color_palette("Set2", len(corrs))   # Fallback color scheme if none is given

    # Compute mean and SEM for each condition's correlations
    bar_data = []
    sem_data = []
    for c in corrs:
        if np.all(np.isnan(c)):
            bar_data.append(0.0)          
            sem_data.append(0.0)          
        else:
            bar_data.append(np.nanmean(c))
            sem_data.append(stats.sem(c[~np.isnan(c)]) if np.sum(~np.isnan(c)) > 1 else 0)

    # Plot    
    if ax is None: 
        _, ax = plt.subplots(figsize=(len(corrs)+1, 4))
        ax.set_ylabel('Mean correlation')
        if population is True:
            ax.set_title('Population vector correlations')
        else:
            ax.set_title('Per-neuron PSTH correlations')
    ax.bar(labels, bar_data, yerr=sem_data, capsize=3, color=color_scheme)
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    plt.xticks(rotation=0)
    plt.tight_layout()

    if save_plot:
        output_path = os.path.join(savepath, savedir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if population:
            plt.savefig(os.path.join(output_path, filename + '_population.png'))
        else:
            plt.savefig(os.path.join(output_path, filename + '.png'))

    return corrs


def get_map_correlation_matrix(all_average_psths, conditions, population=False, zscoring=True, save_plot=False, savepath='', savedir='', filename=''):
    '''
    Calculate pairwise PSTH correlation across all sessions and goals. 
    If population is True, the correlation is computed across the entire activity map. Otherwise it is calculated on a neuron-by-neuron basis.
    '''
    num_sessions = len(all_average_psths)

    # Flatten all data: [(session 0 goal A), (session 0 goal B), ..., (session 1 goal A), ...]
    data = []
    for s in range(num_sessions):
        for goal in all_average_psths[s].keys():  
            d = all_average_psths[s][goal]
            if zscoring:
                d = stats.zscore(d, axis=1)  # z-score along time
                # d = stats.zscore(d, axis=None)
            data.append(d)

    num_conditions = len(data)  
    assert num_conditions == len(data), 'The length of the input data does not match the number of conditions.'

    # Initialize correlation matrix
    correlation_matrix = np.zeros((num_conditions, num_conditions))

    # Calculate correlations
    for i in range(num_conditions):
        for j in range(num_conditions):
            correlations = []
            if population is True:
                for t in range(data[i].shape[1]):
                    if np.all(np.isfinite(data[i][:,t])) and np.all(np.isfinite(data[j][:,t])):
                        r, _ = stats.pearsonr(data[i][:,t], data[j][:,t])
                        correlations.append(r)
                if correlations:
                    correlation_matrix[i,j] = np.nanmean(correlations)
                else:
                    correlation_matrix[i,j] = np.nan  # If no valid timebins
            else:
                for n in range(data[i].shape[0]):  # loop over neurons
                    if np.all(np.isfinite(data[i][n])) and np.all(np.isfinite(data[j][n])):
                        r, _ = stats.pearsonr(data[i][n], data[j][n])
                        correlations.append(r)
                if correlations:
                    correlation_matrix[i,j] = np.nanmean(correlations)
                else:
                    correlation_matrix[i,j] = np.nan  # If no valid neurons

    # === Plot ===
    fig, ax = plt.subplots(figsize=(5,4))
    if population:
        label = 'Mean population correlation'
    else:
        label = 'Mean neuron correlation'
    im = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='bwr', vmin=-1, vmax=1,
            cbar_kws={'label': label}, cbar=False, square=True, annot_kws={"size": 8}, 
            xticklabels=[f"{c}" for c in conditions],
            yticklabels=[f"{c}" for c in conditions])

    cbar = fig.colorbar(im.collections[0], ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label(label, fontsize=10, rotation=270, labelpad=10)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['-1', '0', '1'])
    ax.set_title('All Sessions and Goals PSTH Correlation')

    plt.tight_layout()

    if save_plot:
        output_path = os.path.join(savepath, savedir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if population:
            plt.savefig(os.path.join(output_path, filename + '_population.png'))
        else:
            plt.savefig(os.path.join(output_path, filename + '.png'))

    plt.show()

    return correlation_matrix


def get_glm_residuals(x, y, plot=True, conditions = ['reward', 'test']):
    # Fit a GLM of the format y = f(x, beta) + noise
    import statsmodels.api as sm
    
    glm_residuals = []

    for i in range(len(y)):
        endog_flat = np.array(y[i]).flatten() # y variable
        exog_flat = np.array(x[i]).flatten() # x variable

        mask = ~np.isnan(exog_flat) & ~np.isnan(endog_flat)
        exog = exog_flat[mask]
        endog = endog_flat[mask]

        X = sm.add_constant(exog)  # Adds a column of ones

        # Fit the model
        model = sm.OLS(endog, X).fit()

        print(model.summary())

        # Get residuals
        residuals = np.full_like(exog_flat, np.nan, dtype=np.float64)
        residuals[mask] = model.resid
        residuals_2d = residuals.reshape(np.array(x[i]).shape)

        glm_residuals.append(residuals_2d)

    
    if plot:
        condition_pairs = list(itertools.combinations(range(len(conditions)), 2))
        
        # Individual colormaps
        fig, ax = plt.subplots(1, len(glm_residuals), figsize=(12,4), sharex=True, sharey=True)
        ax = ax.ravel()

        for i, sim_matrix in enumerate(glm_residuals):
            vmax = np.round(np.nanmax(sim_matrix), 2)
            if vmax < 1e-3:
                vmax = 1e-3
            vmin = -vmax
            im = ax[i].imshow(sim_matrix, vmin=vmin, vmax=vmax, cmap='bwr', origin='lower')
            
            cb = fig.colorbar(im, ax=ax[i], shrink=0.8, ticks=[vmin, vmax])  
            cb.set_label('Correlation (r)', labelpad=-5)

            if i < len(conditions):
                ax[i].set_title(f"{conditions[i]} vs {conditions[i]}")
            else:
                ax[i].set_title(f"{conditions[int(condition_pairs[0][0])]} vs {conditions[int(condition_pairs[0][1])]}")
            ax[i].set_xlabel("Lap block")
            ax[i].set_ylabel("Lap block")
        plt.tight_layout()

        # Global colormap
        all_values = np.concatenate([sim[~np.isnan(sim)].flatten() for sim in glm_residuals])
        vmax = np.round(np.max(all_values), 2)
        vmin = -vmax

        fig, ax = plt.subplots(1, len(glm_residuals), figsize=(12,4), sharex=True, sharey=True)
        ax = ax.ravel()

        for i, sim_matrix in enumerate(glm_residuals):
            im = ax[i].imshow(sim_matrix, vmin=vmin, vmax=vmax, cmap='bwr', origin='lower')

            if i < len(conditions):
                ax[i].set_title(f"{conditions[i]} vs {conditions[i]}")
            else:
                ax[i].set_title(f"{conditions[int(condition_pairs[0][0])]} vs {conditions[int(condition_pairs[0][1])]}")
            ax[i].set_xlabel("Lap block")
            ax[i].set_ylabel("Lap block")
        # plt.tight_layout()
        fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.8, label='Correlation (r)', ticks=[vmin, vmax])

    return glm_residuals


def get_lm_firing(dF, cell, session, lm_entry_idx=None, lm_exit_idx=None, bins=90, shuffle=False):

    # Get landmark entry and exit idx 
    if lm_entry_idx is None and lm_exit_idx is None:
        lm_entry_idx, lm_exit_idx = parse_session_functions.get_lm_entry_exit(session)

    # Create landmark vector TODO: remove? 
    num_landmarks = session['num_landmarks']
    lm_vec = np.arange(num_landmarks)
    lm_vec = np.tile(lm_vec, len(session['all_landmarks'])//num_landmarks)
    num_lms_considered = len(lm_vec)

    # Get firing rate per landmark 
    binned_firing = np.zeros((num_lms_considered, bins))
    for i, (entry, exit) in enumerate(zip(lm_entry_idx[:num_lms_considered], lm_exit_idx[:num_lms_considered])):
        if shuffle: 
                # Generate null distribution with circular shifts.
                shift = np.random.randint(len(dF[cell]))
                dF_shuffled = np.roll(dF[cell], shift)
                lm_firing = dF_shuffled[entry:exit]
        else:
                lm_firing = dF[cell, entry:exit]

        binned_firing[i], _, _ = stats.binned_statistic(np.arange(0, len(lm_firing)), lm_firing, bins=bins)
        if np.isnan(binned_firing[i]).any():
            print(f"NaNs found! Lap {i}, entry={entry}, exit={exit}. Consider reducing the number of bins.")

    # Get mean firing rate per landmark 
    mean_firing = np.nanmean(binned_firing, axis=1)

    # Get firing rate per landmark id
    binned_lm_firing = np.zeros((num_landmarks, bins))
    for k in range(num_landmarks):
        binned_lm_firing[k] = np.nanmean(binned_firing[k::num_landmarks], axis=0)

    # Get mean firing rate per landmark id - regressor
    mean_lm_firing = np.mean(binned_lm_firing, axis=1)

    return mean_firing, binned_firing, mean_lm_firing, binned_lm_firing


def get_high_lm_firing_cells(dF, lm_tunings, session, goal_lms, test_lm, bins=90, 
                             stage=None, lm_entry_idx=None, lm_exit_idx=None, plot=True,
                             saveplot=True, figpath='test_lm_cells'):
    mean_goal_lm_firing = {}
    mean_test_lm_firing = {}
    mean_firing = {}
    binned_firing = {}
    binned_lm_firing = {}
    wilcoxon_stat = np.zeros((len(lm_tunings)))
    wilcoxon_pval = np.zeros((len(lm_tunings)))

    for c, cell in enumerate(lm_tunings):
        mean_firing[cell], binned_firing[cell], _, binned_lm_firing[cell] = get_lm_firing(dF, cell, session, lm_entry_idx=lm_entry_idx, lm_exit_idx=lm_exit_idx, bins=bins, shuffle=False)

        # Gather mean firing for all goal landmarks
        arrs = [mean_firing[cell][i::session['num_landmarks']] for i in goal_lms]
        stacked = np.stack(arrs, axis=1)   # shape (n_rows, n_goal_lms)
        mean_goal_lm_firing[cell] = stacked.mean(axis=1)

        # Gather mean firing for all goal landmarks
        mean_test_lm_firing[cell] = np.concatenate([mean_firing[cell][test_lm::session['num_landmarks']]])

        # Perform Wilcoxon test on test vs goal
        wilcoxon_stat[c], wilcoxon_pval[c] = stats.wilcoxon(mean_goal_lm_firing[cell], mean_test_lm_firing[cell]) 

    # Find cells with higher test vs goal firing and plot binned firing rate per landmark per lap and polar plot of fiirng rate per lm. 
    high_lm_cells = []
    for c, cell in enumerate(lm_tunings):
        # Criterion 1: p-value < 0.05
        if wilcoxon_pval[c] < 0.05:

            # Criterion 2: max test firing > max goal firing
            goal_arr = [binned_lm_firing[cell][i::session['num_landmarks'], :] for i in goal_lms]
            goal_arr = np.stack(goal_arr, axis=1).flatten()
            test_arr = binned_lm_firing[cell][test_lm::session['num_landmarks'], :].flatten()

            if np.max(test_arr) > np.max(goal_arr):
                
                high_lm_cells.append(cell)

                if plot:
                    # Format data for plotting
                    arrs = [binned_firing[cell][i::session['num_landmarks'], :] for i in goal_lms]
                    binned_goal_firing = np.stack(arrs, axis=1)   # shape (n_rows, n_goal_lms, n_cols)
                    binned_test_firing = binned_firing[cell][test_lm::session['num_landmarks'], :]
                    
                    combined_rows = np.concatenate([binned_goal_firing, binned_test_firing[:, None, :]], axis=1)  # axis1 = landmarks
                    plot_data = combined_rows.reshape(binned_goal_firing.shape[0], -1)

                    # Plot
                    fig = plt.figure(figsize=(10, 4))
                    ax0 = fig.add_subplot(121)
                    
                    im = ax0.imshow(plot_data, aspect='auto', cmap='viridis')
                    fig.colorbar(im, ax=ax0, label='Firing rate (Hz)')

                    cols_per_lm = binned_firing[cell].shape[1]  # number of bins per landmark
                    for i in range(1, len(goal_lms) + 1):
                        ax0.axvline(i*cols_per_lm, color='white', linestyle='--', lw=0.5)  # reward LMs

                    tick_positions = np.arange(len(goal_lms) + 1) * cols_per_lm + cols_per_lm/2
                    if test_lm == 9:
                        tick_labels = ['A', 'B', 'C', 'D', 'Test']
                    elif test_lm == 8:
                        tick_labels = ['1', '3', '5', '7', '9']
                    ax0.set_xticks(tick_positions, tick_labels, fontsize=10)
                    ax0.set_xlabel('Landmarks')
                    ax0.set_yticks([0, binned_goal_firing.shape[0]-1])
                    ax0.set_ylabel('Lap')
                    ax0.set_title(f'Cell {cell}, p-value {np.round(wilcoxon_pval[c], 2)}')

                    # Polar subplot
                    data = binned_lm_firing[cell].flatten()
                    if stage == 5:
                        color = 'blue'
                    elif stage == 6:
                        color = 'orange'
                    elif stage == 8:
                        color = 'red'
                    else:
                        color = 'blue'
                    ax1 = fig.add_subplot(122, projection='polar')
                    ax1.set_theta_zero_location('N')
                    ax1.set_theta_direction(-1)
                    angles = np.linspace(0, 2 * np.pi, binned_lm_firing[cell].shape[1] * session['num_landmarks'], endpoint=False)
                    # add the first angle to close the circle
                    angles = np.concatenate((angles, [angles[0]]))
                    avg_bin = np.concatenate((data, [data[0]]))
                    # sem_bin = np.concatenate((sem_bin, [sem_bin[0]]))
                    ax1.plot(angles, avg_bin, color=color, linewidth=2)
                    # ax1.fill_between(angles, avg_bin - sem_bin, avg_bin + sem_bin, color='blue', alpha=0.2)
                    #label the cardinal directions
                    ax1.set_xticks(np.linspace(0, 2 * np.pi, session['num_landmarks'], endpoint=False))
                    ax1.set_xticklabels(np.arange(1,11))
                    ax1.set_title(f'Cell {cell} - Average Firing Rate (Polar)')

                    if saveplot is True:
                        plt.savefig(figpath / f'cell{cell}.png')
                    plt.show()
    
    return high_lm_cells


def get_test_peak_tuned_cells(dF, goal_firing, event_idx, session_idx, neurons, bins,
                         rew_goals, test_goal, session, save_path, plot=True, add_lick_rate=False):
    """
    Find neurons with strong test-goal tuning using multiple criteria.
    - rew_goals: list of goal indices considered as "reward"
    - test_goal: index of the goal to test against
    """

    test_goal_cells = []

    wilcoxon_stat = np.zeros(len(neurons))
    wilcoxon_pval = np.zeros(len(neurons))

    templates, peaks = create_templates(peaks=[1, 4, 5], bins=360, plot=False)

    for c, cell in enumerate(neurons):

        # Split data by goal
        goal_data = [
            goal_firing[cell][:, bins*i:bins*(i+1)]
            for i in range(len(rew_goals) + 1)  # total goals
        ]

        rew_data = np.hstack([goal_data[i] for i in rew_goals])  # concat reward goals
        test_data = goal_data[test_goal]

        # Mean activity per lap
        mean_rew_data = np.mean(rew_data, axis=1)
        mean_test_data = np.mean(test_data, axis=1)

        # Mean activity per bin
        mean_bin_rew_data = np.mean(rew_data, axis=0)
        mean_bin_test_data = np.mean(test_data, axis=0)

        # Condition 1: Wilcoxon test (test > rew)
        wilcoxon_stat[c], wilcoxon_pval[c] = stats.wilcoxon(mean_rew_data, mean_test_data, alternative='less')

        if wilcoxon_pval[c] < 0.05:
            # Condition 2: test firing stronger than rew
            if np.max(mean_bin_test_data) > np.max(mean_bin_rew_data):

                # Condition 3: high tuning score
                tuning_scores = []
                avg_goal_firing = np.mean(goal_firing[cell], axis=0)
                for i in np.sort(np.concatenate([rew_goals, [test_goal]])):
                    state = avg_goal_firing[bins*i:bins*(i+1)]
                    state_max, state_min, state_mean = np.max(state), np.min(state), np.mean(state)
                    tuning_scores.append((state_max - state_min) / state_mean)

                if tuning_scores[test_goal] > 1.2 and (
                    tuning_scores[test_goal] > np.median([tuning_scores[i] for i in rew_goals]) + 0.2
                ):
                    # Condition 4: highest correlation with single-peak template
                    avg_binned_data = np.mean(goal_firing[cell], axis=0)
                    n_cell_peaks, _ = get_template_ccg(cell, avg_binned_data, templates, peaks, plot=False)

                    if n_cell_peaks == 1:
                        test_goal_cells.append(cell)

                        if plot:
                            if add_lick_rate:
                                dF_lick = np.array(session['frame_lick_rate']).reshape(1, -1)

                                cellTV.plot_arb_progress_2cells(dF=[dF, dF_lick], cell=[cell, 0], 
                                                                sessions=[session, session],
                                                                event_frames=[event_idx, event_idx], 
                                                                ngoals=len(rew_goals)+1, bins=bins, 
                                                                stages=np.array([session_idx, session_idx]), 
                                                                labels=[f'cell {cell}', 'lick rate'], 
                                                                plot=True, shuffle=False)
                            else:
                                plot_arb_progress(dF, cell, event_idx, len(rew_goals) + 1, bins, session_idx, ax=None)
                            
    if save_path is not None:
        np.savez(save_path, high_test_goal_cells=test_goal_cells)

    return test_goal_cells


def get_high_peak_tuned_cells(dF, goal_firing, event_idx, session_idx, neurons, bins,
                         rew_goals, test_goal, session, save_path, plot=True, add_lick_rate=False):
    """
    Find neurons with stronger test-goal than rew-goal tuning using multiple criteria.
    - rew_goals: list of goal indices considered as "reward"
    - test_goal: index of the goal to test against
    """

    high_test_goal_cells = []

    wilcoxon_stat = np.zeros(len(neurons))
    wilcoxon_pval = np.zeros(len(neurons))

    templates, peaks = create_templates(peaks=[4, 5], bins=360, plot=False)

    for c, cell in enumerate(neurons):

        # Split data by goal
        goal_data = [
            goal_firing[cell][:, bins*i:bins*(i+1)]
            for i in range(len(rew_goals) + 1)  # total goals
        ]

        rew_data = np.hstack([goal_data[i] for i in rew_goals])  # concat reward goals
        test_data = goal_data[test_goal]

        # Mean activity per lap
        mean_rew_data = np.mean(rew_data, axis=1)
        mean_test_data = np.mean(test_data, axis=1)

        # Mean activity per bin
        mean_bin_rew_data = np.mean(rew_data, axis=0)
        mean_bin_test_data = np.mean(test_data, axis=0)

        # Condition 1: Wilcoxon test (test > rew)
        wilcoxon_stat[c], wilcoxon_pval[c] = stats.wilcoxon(mean_rew_data, mean_test_data, alternative='less')

        if wilcoxon_pval[c] < 0.05:
            # Condition 2: test firing stronger than rew
            if np.max(mean_bin_test_data) > np.max(mean_bin_rew_data):

                # Condition 3: high tuning score
                tuning_scores = []
                avg_goal_firing = np.mean(goal_firing[cell], axis=0)
                for i in np.sort(np.concatenate([rew_goals, [test_goal]])):
                    state = avg_goal_firing[bins*i:bins*(i+1)]
                    state_max, state_min, state_mean = np.max(state), np.min(state), np.mean(state)
                    tuning_scores.append((state_max - state_min) / state_mean)

                if tuning_scores[test_goal] > 1.2 and (
                    tuning_scores[test_goal] > np.median([tuning_scores[i] for i in rew_goals]) + 0.2
                ):
                    high_test_goal_cells.append(cell)

                avg_binned_data = np.mean(goal_firing[cell], axis=0)
                n_cell_peaks, _ = get_template_ccg(cell, avg_binned_data, templates, peaks, plot=False)

                if n_cell_peaks == 5:
                    # test_goal_cells.append(cell)
                    high_test_goal_cells.append(cell)

                    if plot:
                        if add_lick_rate:
                            dF_lick = np.array(session['frame_lick_rate']).reshape(1, -1)

                            cellTV.plot_arb_progress_2cells(dF=[dF, dF_lick], cell=[cell, 0], event_frames=[event_idx, event_idx], 
                                                            ngoals=len(rew_goals)+1, bins=bins, 
                                                            stages=[session_idx, session_idx], labels=[f'cell {cell}', 'lick rate'], 
                                                            plot=True, shuffle=False)
                        else:
                            plot_arb_progress(dF, cell, event_idx, len(rew_goals) + 1, bins, session_idx, ax=None)

    if save_path is not None:                         
        np.savez(save_path, high_test_goal_cells=high_test_goal_cells)

    return high_test_goal_cells


def plot_arb_progress(dF, cell, event_frames, ngoals, bins, stage, session, period='goal', labels=None, ax=None):
    """
    Extract the progress tuning between arbitrary events.
    If ax1/ax2 are given, plot into them. Otherwise, create a new figure.
    """
    dF_cell = cellTV.extract_cell_trace(dF, cell, plot=False)
    binned_phase_firing = np.zeros((len(event_frames)-1, bins))

    # Create a goal vector 
    if period == 'goal':
        # Events are organised based on whether they are a goal or not
        if ('shuffled' in session['sequence']):
            assert ngoals == 2
            goal_vec = np.empty((len(event_frames)), dtype=int)
            for i in range(len(event_frames)):
                if i in session['goals_idx']:
                    goal_vec[i] = 0
                elif i in session['non_goals_idx']:
                    goal_vec[i] = 1
        else:
            goal_vec = np.arange(ngoals)
            goal_vec = np.tile(goal_vec, len(event_frames)//ngoals) 

    elif period == 'landmark':
        # Events are organised based on the order in which they occur
        goal_vec = np.arange(ngoals)
        goal_vec = np.tile(goal_vec, len(event_frames)//ngoals)  
    goal_vec = goal_vec[:-1]
    
    num_trials = np.array([np.sum(goal_vec == i) for i in range(ngoals)])
    max_trials = np.max(num_trials)

    for i in range(len(event_frames)-1):
        phase_frames = np.arange(event_frames[i], event_frames[i+1])
        bin_edges = np.linspace(event_frames[i], event_frames[i+1], bins+1)
        phase_firing = dF_cell[phase_frames]
        bin_ix = np.digitize(phase_frames, bin_edges)
        for j in range(bins):
            binned_phase_firing[i, j] = np.mean(phase_firing[bin_ix == j+1])

    binned_segment = np.zeros((ngoals, max_trials, binned_phase_firing.shape[1]))
    for i in range(ngoals):
        idx = np.where(goal_vec == i)[0]
        binned_segment[i, :len(idx), :] = binned_phase_firing[idx, :]

    min_state = min(seg.shape[0] for seg in binned_segment)
    binned_all = np.concatenate([binned_segment[i][:min_state, :] for i in range(ngoals)], axis=1)

    avg_bin = np.nanmean(binned_all, axis=0)
    std_bin = np.nanstd(binned_all, axis=0)
    sem_bin = std_bin / np.sqrt(binned_all.shape[0])

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
    elif stage == 12:
        color = 'teal'
    else:
        color = 'gray'

    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='polar')

    angles = np.linspace(0, 2 * np.pi, bins*ngoals, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    avg_bin = np.concatenate((avg_bin, [avg_bin[0]]))
    # avg_bin = avg_bin / np.max(avg_bin)
    sem_bin = np.concatenate((sem_bin, [sem_bin[0]]))
    # sem_bin = sem_bin / np.max(avg_bin)
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.plot(angles[:bins*(ngoals-1)], avg_bin[:bins*(ngoals-1)], color=color, linewidth=2)
    ax.plot(angles[bins*(ngoals-1):], avg_bin[bins*(ngoals-1):], color=color, linewidth=2)
    ax.fill_between(angles, avg_bin - sem_bin, avg_bin + sem_bin, color=color, alpha=0.2)
    ax.set_xticks(np.linspace(0, 2 * np.pi, ngoals, endpoint=False))
    # if ngoals == 10:
    #     ax.set_xticklabels([])    
    ax.set_rticks([np.round(np.min(avg_bin),1), np.round(np.max(avg_bin),1)])
    if labels is None:
        ax.set_title(f'T{stage} Cell {cell}')
    else:
        ax.set_title(labels)

    return ax, avg_bin, sem_bin


def create_templates(peaks=[1,4,5], bins=360, plot=True):

    templates = [np.zeros((bins)) for _ in range(len(peaks))]

    # Arbitrarily adds peaks to simulate activity, scaled to max at 1:
    fake_peak = 25*stats.norm.pdf(range(50), 25, 10)
    for i in range(5):
        for j, peak in enumerate(peaks):
            if i < peak:
                templates[j][72*i:72*i + 50] = fake_peak

    # Plot the templates
    if plot:
        _, ax = plt.subplots(1, len(peaks), figsize=(10,2))
        ax = ax.ravel()
        for i in range(len(peaks)):
            ax[i].plot(templates[i])
            ax[i].set_title(f'{peaks[i]}-peaks template')
        
    return templates, peaks


def get_goal_progress_cells(dF, neurons, session, event_frames, save_path, ngoals=4, bins=90, period='goal', reload=False, plot=True, shuffle=False):
    # Find goal progress tuned cells - takes long if shuffling
    stage = int(session['stage'][-1])

    if period == 'goal':
        filename = f'T{stage}_{ngoals}goal_progress_tracked_neurons.npz'
    else:
        filename = f'T{stage}_{ngoals}period_tracked_neurons.npz'

    if os.path.exists(os.path.join(save_path, filename)) and not reload:
        print(f'Goal progress and tracked neurons found. Loading...')
        data = np.load(os.path.join(save_path, filename), allow_pickle=True)
        goal_progress_tuned = data['goal_progress_tuned']
        real_scores = data['real_scores'].item()
        shuffled_scores = data['shuffled_scores'].item()

    else:
        goal_progress_tuned = []
        real_scores = {}
        shuffled_scores = {}
        for cell in neurons:
            real_scores[cell], shuffled_scores[cell], _, _ = cellTV.calc_goal_tuningix(dF, cell, session, condition='arb', period=period, event_frames=event_frames, n_goals=ngoals, frame_rate=45, bins=bins, shuffle=shuffle, plot=False)

            if 'shuffled' in session['sequence']:
                if real_scores[cell] - np.median(shuffled_scores[cell]) > 0.07:
                    goal_progress_tuned.append(cell)
            else:
                if (real_scores[cell] > 1) & (np.abs(real_scores[cell] - np.median(shuffled_scores[cell])) > 0.5):
                    goal_progress_tuned.append(cell)

        # Plot firing rates for goal progress tuned cells
        if plot:
            for cell in goal_progress_tuned:
                _ = cellTV.extract_arb_progress(dF, cell, session, event_frames, ngoals=ngoals, 
                                                bins=bins, period=period, stage=stage, 
                                                plot=plot, shuffle=False)

        # Save these neurons
        np.savez(os.path.join(save_path, filename), 
                 goal_progress_tuned=np.array(goal_progress_tuned), 
                 real_scores=np.array(real_scores), 
                 shuffled_scores=np.array(shuffled_scores),
                 allow_pickle=True)

    print(f"{len(goal_progress_tuned)} out of {len(neurons)} tracked neurons are goal progress tuned in T{stage}")

    return goal_progress_tuned, real_scores, shuffled_scores 


def circular_crosscorr(x, y):
    # center the signals to not inflate by baseline offsets
    X = np.fft.fft(x - np.mean(x))
    Y = np.fft.fft(y - np.mean(y))
    corr = np.fft.ifft(X * Y.conj()).real
    # shift so that lag=0 is first
    corr = np.fft.fftshift(corr)
    # normalize like Pearson correlation
    corr = corr / (np.std(x) * np.std(y) * len(x))
    return corr


def get_acg_template_ccg(dF, cell, session, event_idx, ngoals, templates, plot_firing=True, plot_corr=True):
    """
    Extracts the autocorrelogram of the binned firing rate of a neuron and the 
    crosscorrelogram of the binned firing rate with two templates testing for 
    4-fold or 5-fold goal progress.
    """
    template_size = templates[0].shape[0]

    # Extract the firing rate
    binned_firing_rate = cellTV.extract_arb_progress(dF, cell, session, event_idx, ngoals=ngoals, bins=90, plot=plot_firing, shuffle=False)
    avg_binned_firing_rate = np.mean(binned_firing_rate, axis=0)

    # Resize firing rate if needed
    if avg_binned_firing_rate.shape[0] > template_size:
        firing_rate_resized = resample(avg_binned_firing_rate, template_size)
    else:
        firing_rate_resized = avg_binned_firing_rate

    # Get the autocorrelogram
    acg = circular_crosscorr(firing_rate_resized, firing_rate_resized) 
    ccg0 = circular_crosscorr(firing_rate_resized, templates[0]) 
    ccg1 = circular_crosscorr(firing_rate_resized, templates[1]) 

    # Plot the correlation
    if plot_corr:
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12,3), sharey=False)
        ax1.sharey(ax2)   # explicit sharing
        ax0.plot(np.arange(0, len(acg)), acg)
        ax0.set_title('Autocorrelogram')
        ax1.plot(ccg0)
        ax1.set_title('Firing rate vs 5-fold template')
        ax2.plot(ccg1)
        ax2.set_title('Firing rate vs 4-fold template')
        plt.suptitle(f'Neuron {cell}')
        plt.tight_layout()

    return acg, ccg0, ccg1


def get_template_ccg(cell, binned_firing, templates, peaks, plot=True):
    # Resize firing rate if needed
    template_size = templates[0].shape[0]
    if binned_firing.shape[0] > template_size:
        firing_rate_resized = resample(binned_firing, template_size)
    else:
        firing_rate_resized = binned_firing

    # Get the autocorrelogram
    ccg = []
    for i in range(len(templates)):
        ccg.append(circular_crosscorr(firing_rate_resized, templates[i]))
    ccg = np.vstack(ccg)
    template_maxima = ccg.max(axis=1)

    # find index of template with the global maximum
    best_template = np.argmax(template_maxima)
    n_cell_peaks = peaks[best_template]

    # Plot the correlation
    if plot:
        _, ax = plt.subplots(1, len(templates), figsize=(10,2))
        ax = ax.ravel()
        y_min = min(cc.min() for cc in ccg)
        y_max = max(cc.max() for cc in ccg)

        for i in range(len(peaks)):
            ax[i].plot(ccg[i])
            ax[i].set_title(f'Firing rate vs {peaks[i]}-peaks template')
            ax[i].set_ylim([y_min, y_max])   # enforce same y-scale

        plt.suptitle(f'Neuron {cell}: # peaks = {n_cell_peaks}')
        plt.tight_layout()

    return n_cell_peaks, ccg


def classify_4_or_5_peak_neurons(neurons, mean_goal_firing, peaks=[1,4,5], plot=True):
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    import matplotlib.gridspec as gridspec

    # Create templates 
    templates, peaks = create_templates(peaks=peaks, bins=360, plot=False)

    neurons_1peaks = []
    neurons_4peaks = []
    neurons_5peaks = []

    for cell in neurons:
        # ----- Criterion 1 ----- #
        
        # Detect number of peaks on the binned firing
        binned_firing = np.mean(mean_goal_firing[cell], axis=0)

        # Smooth the firing to detect peaks
        smooth_avg_bin = gaussian_filter1d(binned_firing, sigma=4, mode='wrap').copy()
        N = binned_firing.size

        # Triple the signal to detect boundary peaks
        avg_tripled = np.concatenate((smooth_avg_bin, smooth_avg_bin, smooth_avg_bin))
        peaks_tripled, props = find_peaks(avg_tripled, distance=10, prominence=0.3, wlen=50, height=0.7)
        
        # Discard very small peaks
        mask = props["peak_heights"] >= 0.4 * np.mean(props['peak_heights'])
        peaks_tripled = peaks_tripled[mask]

        # Identify peaks on the first third of the tripled signal
        polar_peaks = peaks_tripled[(peaks_tripled >= N) & (peaks_tripled < 2*N)] - N
        
        # ----- Criterion 2 ----- #

        # Get maximum cross-correlation with either 4-peak or 5-peak templates
        n_cell_peaks, ccg = get_template_ccg(cell, binned_firing, templates, peaks, plot=False)

        # Overwrite template-based cell classification if needed 
        if (n_cell_peaks == 4) and (len(polar_peaks) > n_cell_peaks):
            n_cell_peaks = 5

        # ----- Classification ----- #
        if n_cell_peaks == 4:
            neurons_4peaks.append(cell)
        elif n_cell_peaks == 5:
            neurons_5peaks.append(cell)
        elif n_cell_peaks == 1:
            neurons_1peaks.append(cell)

        # ----- Plotting ----- #
        if plot: 

            fig = plt.figure(figsize=(8, 5))
            gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2])  # top row 3/5, bottom row 2/5

            # top row (polar plots)
            ax1 = fig.add_subplot(gs[0, 0], projection='polar')
            ax2 = fig.add_subplot(gs[0, 1], projection='polar')

            # bottom row (CCGs)
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
            # fig = plt.figure(figsize=(10, 5))

            angles = np.linspace(0, 2 * np.pi, 90*5, endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # add the first angle to close the circle
            peak_angles = angles[polar_peaks]

            # ax1 = fig.add_subplot(221, projection='polar')
            ax1.set_theta_zero_location('N')
            ax1.set_theta_direction(-1)
            smooth_avg_bin = np.concatenate((smooth_avg_bin, [smooth_avg_bin[0]]))
            ax1.plot(angles, smooth_avg_bin, color='blue', linewidth=2)
            ax1.scatter(peak_angles, smooth_avg_bin[polar_peaks], color="red", s=40, zorder=3)
            ax1.set_xticks(np.linspace(0, 2 * np.pi, 5, endpoint=False))
            ax1.set_title(f'Smooth firing rate (sigma=4)')

            # ax2 = fig.add_subplot(222, projection='polar')
            ax2.set_theta_zero_location('N')
            ax2.set_theta_direction(-1)
            binned_firing = np.concatenate((binned_firing, [binned_firing[0]]))
            ax2.plot(angles, binned_firing, color='blue', linewidth=2)
            ax2.scatter(peak_angles, binned_firing[polar_peaks], color="red", s=40, zorder=3)
            ax2.set_xticks(np.linspace(0, 2 * np.pi, 5, endpoint=False))
            ax2.set_title(f'Raw firing rate')

            y_min = min(cc.min() for cc in ccg)
            y_max = max(cc.max() for cc in ccg)

            # ax3 = fig.add_subplot(223)
            ax3.plot(ccg[0])
            ax3.set_ylim([y_min, y_max])
            ax3.set_title(f'CCG with {peaks[0]}-peak template')

            # ax4 = fig.add_subplot(224)
            ax4.plot(ccg[1])
            ax4.set_ylim([y_min, y_max])
            ax4.set_title(f'CCG with {peaks[1]}-peak template')

            plt.suptitle(f'Neuron {cell} n_peaks {n_cell_peaks}')
            plt.tight_layout()
    
    return neurons_1peaks, neurons_4peaks, neurons_5peaks

def annotate_cell(cell, binned_firing, n_peaks=None):
    """
    Show polar plot for one cell and let the user assign number of peaks.
    Returns:
        - int: assigned peak count (0-9)
    """
    smooth_avg_bin = gaussian_filter1d(binned_firing, sigma=4, mode='wrap')
    angles = np.linspace(0, 2 * np.pi, len(binned_firing), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    smooth_plot = np.concatenate([smooth_avg_bin, smooth_avg_bin[:1]])

    result = {"peaks": None, "back": False, "quit": False}

    def on_key(event):
        if event.key.isdigit():
            result["peaks"] = int(event.key)
            plt.close('all')
        elif event.key == 'escape':
            result["peaks"] = "keep" 
            plt.close('all')
        elif event.key.lower() == 'q':
            result["quit"] = True
            plt.close('all')
        elif event.key.lower() == 'b':
            result["back"] = True
            plt.close('all')

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5, endpoint=False))
    ax.plot(angles, smooth_plot, lw=2)
    if n_peaks:
        ax.set_title(
            f"Cell {cell} n_peaks = {n_peaks}\n"
            "0-9 = assign peaks | B = back | Q = quit | ESC = keep")
    else:
        ax.set_title(
        f"Cell {cell}\n"
        "0-9 = assign peaks | B = back | Q = quit | ESC = keep")
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show(block=True)
    
    if result["quit"]:
        return "quit"
    if result["back"]:
        return "back"
    return result["peaks"]

def classify_4_or_5_peak_neurons_with_gui(neurons, mean_goal_firing, peaks=[1,4,5], plot=False, gui=True):
    """
    Initial peak counting using CCG with templates and peak counting. 
    Optional manual curation by running a GUI.
    """
    # Create templates 
    templates, peaks = create_templates(peaks=peaks, bins=360, plot=False)

    peak_counts = {}       # cell_id -> number of peaks
    n_cell_peaks = {}

    for cell in neurons:
        # ----- Criterion 1 ----- #
        
        # Detect number of peaks on the binned firing
        binned_firing = np.mean(mean_goal_firing[cell], axis=0)

        # Smooth the firing to detect peaks
        smooth_avg_bin = gaussian_filter1d(binned_firing, sigma=4, mode='wrap').copy()
        N = binned_firing.size

        # Triple the signal to detect boundary peaks
        avg_tripled = np.concatenate((smooth_avg_bin, smooth_avg_bin, smooth_avg_bin))
        peaks_tripled, props = find_peaks(avg_tripled, distance=10, prominence=0.3, wlen=50, height=0.7)
        
        # Discard very small peaks
        mask = props["peak_heights"] >= 0.4 * np.mean(props['peak_heights'])
        peaks_tripled = peaks_tripled[mask]

        # Identify peaks on the first third of the tripled signal
        polar_peaks = peaks_tripled[(peaks_tripled >= N) & (peaks_tripled < 2*N)] - N
        
        # ----- Criterion 2 ----- #

        # Get maximum cross-correlation with either 4-peak or 5-peak templates
        n_cell_peaks[cell], ccg = get_template_ccg(cell, binned_firing, templates, peaks, plot=False)

        # Overwrite template-based cell classification if needed 
        if (n_cell_peaks[cell] == 4) and (len(polar_peaks) > n_cell_peaks[cell]):
            n_cell_peaks[cell] = 5

        # ----- Classification ----- #
        peak_counts[cell] = n_cell_peaks[cell]

        # ----- Plotting ----- #
        if plot: 

            fig = plt.figure(figsize=(8, 5))
            gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2])  # top row 3/5, bottom row 2/5

            # top row (polar plots)
            ax1 = fig.add_subplot(gs[0, 0], projection='polar')
            ax2 = fig.add_subplot(gs[0, 1], projection='polar')

            # bottom row (CCGs)
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
            # fig = plt.figure(figsize=(10, 5))

            angles = np.linspace(0, 2 * np.pi, 90*5, endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # add the first angle to close the circle
            peak_angles = angles[polar_peaks]

            # ax1 = fig.add_subplot(221, projection='polar')
            ax1.set_theta_zero_location('N')
            ax1.set_theta_direction(-1)
            smooth_avg_bin = np.concatenate((smooth_avg_bin, [smooth_avg_bin[0]]))
            ax1.plot(angles, smooth_avg_bin, color='blue', linewidth=2)
            ax1.scatter(peak_angles, smooth_avg_bin[polar_peaks], color="red", s=40, zorder=3)
            ax1.set_xticks(np.linspace(0, 2 * np.pi, 5, endpoint=False))
            ax1.set_title(f'Smooth firing rate (sigma=4)')

            # ax2 = fig.add_subplot(222, projection='polar')
            ax2.set_theta_zero_location('N')
            ax2.set_theta_direction(-1)
            binned_firing = np.concatenate((binned_firing, [binned_firing[0]]))
            ax2.plot(angles, binned_firing, color='blue', linewidth=2)
            ax2.scatter(peak_angles, binned_firing[polar_peaks], color="red", s=40, zorder=3)
            ax2.set_xticks(np.linspace(0, 2 * np.pi, 5, endpoint=False))
            ax2.set_title(f'Raw firing rate')

            y_min = min(cc.min() for cc in ccg)
            y_max = max(cc.max() for cc in ccg)

            # ax3 = fig.add_subplot(223)
            ax3.plot(ccg[0])
            ax3.set_ylim([y_min, y_max])
            ax3.set_title(f'CCG with {peaks[0]}-peak template')

            # ax4 = fig.add_subplot(224)
            ax4.plot(ccg[1])
            ax4.set_ylim([y_min, y_max])
            ax4.set_title(f'CCG with {peaks[1]}-peak template')

            plt.suptitle(f'Neuron {cell} n_peaks {n_cell_peaks[cell]}')
            plt.tight_layout()

    # ----- GUI ----- #
    if gui:
        peak_counts = dict(n_cell_peaks)

        i = 0
        while i < len(neurons):
            cell = neurons[i]
            binned_firing = np.mean(mean_goal_firing[cell], axis=0)

            assigned = annotate_cell(cell, binned_firing, n_cell_peaks[cell])

            # ----- QUIT -----
            if assigned == "quit":
                print("Annotation stopped by user.")
                break

            # ----- BACK -----
            if assigned == "back":
                if i > 0:
                    i -= 1
                    print(f"Going back to cell {neurons[i]}")
                else:
                    print("Already at first cell")
                continue

            # ----- KEEP DEFAULT -----
            if assigned == "keep":
                # Do nothing, keep existing value
                i += 1
                continue

            # ----- OVERWRITE -----
            if assigned is not None:
                peak_counts[cell] = assigned
                i += 1

        print("Final peak counts:", peak_counts)

    return peak_counts

def get_state_tuned_cells(dF, session, event_idx, neurons, bins=90, ngoals=5, sigma_smooth=10, plot=True, shuffled_scores_path=''):
    """
    Get state-tuned neurons following the z-scoring method in El Gaby et al., and adding a tuning score criterion.
    """
    stage = int(session['stage'][-1])
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
    elif stage == 12:
        color = 'teal'
    else:
        color = 'gray'

    # Load shuffled scores or compute them if needed 
    tuning_score_criterion = True
    if ngoals > 5:
        print('Shuffled scores do not exist for this number of goals. Ignoring this criterion for now.')
        tuning_score_criterion = False

    if shuffled_scores_path and tuning_score_criterion:
        print('Shuffled scores found. Loading...')
        _, _, shuffled_scores = get_goal_progress_cells(dF, neurons, session, event_idx, shuffled_scores_path, ngoals=ngoals, bins=bins, period='goal', plot=False, shuffle=True)
    elif not shuffled_scores_path and tuning_score_criterion:
        print('Shuffled scores path not defined. Scores are computed now...')
        _, _, shuffled_scores = get_goal_progress_cells(dF, neurons, session, event_idx, session['save_path'], ngoals=ngoals, bins=bins, period='goal', plot=False, shuffle=True)
    
    sigma_bins = sigma_smooth / (360 / bins * ngoals) # sigma_smooth in deg
    angles = np.linspace(0, 2 * np.pi, 90*5, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # add the first angle to close the circle
    states = ['A','B','C','D','test']
    if ngoals == 10:
        states = ['A','A','B','B','C','C','D','D','test','test']
    state_number = np.arange(1,11)
    state_width = 2 * np.pi / ngoals

    state_tuned = []
    state_number_preference = {}
    keep_cell = []

    binned_goal_activity = {}
    for cell in neurons:
        # Bin activity per goal 
        binned_goal_activity[cell] = cellTV.extract_arb_progress(dF, cell, session, event_idx, ngoals=ngoals, bins=bins, 
                                                            stage=stage, plot=False, shuffle=False)

        ntrials = binned_goal_activity[cell].shape[0]

        # (1) Find the preferred state 
        av_binned = np.nanmean(binned_goal_activity[cell], axis=0)
        avg_tripled = np.concatenate((av_binned, av_binned, av_binned))
        avg_smoothed = gaussian_filter1d(avg_tripled, sigma=sigma_bins, mode='nearest').copy()
        N = av_binned.size
        smooth_center = avg_smoothed[N:2*N]  

        state_max = np.array([np.max(av_binned[bins*i:bins*(i+1)]) for i in range(ngoals)])
        state_min = np.array([np.min(av_binned[bins*i:bins*(i+1)]) for i in range(ngoals)])
        state_mean = np.array([np.mean(av_binned[bins*i:bins*(i+1)]) for i in range(ngoals)])
        state_preference = np.where(state_max == np.max(state_max))[0][0]
        tuning_score = (state_max - state_min) / state_mean

        if tuning_score_criterion:
            if tuning_score[state_preference] - np.mean(shuffled_scores[cell]) > 0.9:
                keep_cell.append(cell)
        else:
            keep_cell.append(cell)

        # (2) Take the peak firing rate in each state and trial (n_trials x n_goals)
        trial_state_max = np.zeros((ntrials, ngoals))
        for i in range(ngoals):
            trial_state_max[:,i] = np.max(binned_goal_activity[cell][:,bins*i:bins*(i+1)], axis=1)
            
        # (3) z-score across each trial 
        trial_state_max_zscored = stats.zscore(trial_state_max, axis=1)

        # (4) Extract the z-score of the preferred state 
        pref_state_zscore = np.array([trial_state_max_zscored[t, state_preference] for t in range(ntrials)])

        # (5) t-test of z-scores across all trials against 0
        result = stats.ttest_1samp(pref_state_zscore, popmean=0.0, alternative='two-sided')
    
        # Define state-tuned cells 
        if (result.pvalue < 0.05) and np.isin(cell, keep_cell):
            state_tuned.append(cell)
            state_number_preference[cell] = state_number[state_preference]

            # Plot state-tuned cells
            if plot:
                fig = plt.figure(figsize=(3, 3))
                ax1 = fig.add_subplot(projection='polar')
                ax1.set_theta_zero_location('N')
                ax1.set_theta_direction(-1)
                data = av_binned # smooth_center
                data = np.concatenate((data, [data[0]]))
                ax1.plot(angles, data, color=color, linewidth=2)
                ax1.set_xticks(np.linspace(0, 2 * np.pi, 5, endpoint=False))
                ax1.set_title(f'Cell {cell} - pref state: {states[state_preference]}')
                ax1.set_rlim(0, np.nanmax(data) * 1.1)
                theta_start = state_preference * state_width
                
                ax1.bar(
                    theta_start + state_width/2,
                    ax1.get_rmax(),
                    width=state_width,
                    bottom=0,
                    color=color,
                    alpha=0.15,
                    edgecolor=None
                )

            # elif (result.pvalue < 0.05) and not np.isin(cell, keep_cell):
            #     fig = plt.figure(figsize=(3, 3))
            #     ax1 = fig.add_subplot(projection='polar')
            #     ax1.set_theta_zero_location('N')
            #     ax1.set_theta_direction(-1)
            #     data = av_binned
            #     data = np.concatenate((data, [data[0]]))
            #     ax1.plot(angles, data, color='gray', linewidth=2)
            #     ax1.set_xticks(np.linspace(0, 2 * np.pi, 5, endpoint=False))
            #     ax1.set_title(f'Cell {cell}: pref state {states[state_preference]}')
            #     ax1.set_rlim(0, np.nanmax(data) * 1.1)
            #     theta_start = state_preference * state_width
            #     theta_end   = (state_preference + 1) * state_width
            #     theta_arc = np.linspace(theta_start, theta_end, 200)

            #     ax1.bar(
            #         theta_start + state_width/2,
            #         ax1.get_rmax(),
            #         width=state_width,
            #         bottom=0,
            #         color='gray',
            #         alpha=0.15,
            #         edgecolor=None
            #     )

    return state_tuned, state_number_preference
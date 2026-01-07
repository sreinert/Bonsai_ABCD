import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.segmentation import find_boundaries
from scipy.signal import find_peaks
from sklearn.preprocessing import normalize
# from suite2p.extraction import dcnv
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
import matplotlib.patches as patches
import scipy.cluster.hierarchy as sch
import pandas as pd
#suppress the warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

## New version


## Loading functions 

def get_session_folders(base_path,mouse,stage):
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
    frame_ix = np.load(os.path.join(session_path, 'valid_frames.npz'))

    return imaging_path, config_path, frame_ix, date1, date2

def load_img_data(imaging_path):
    """
    Load the imaging data (no spikes yet) from the specified path.
    """
    f = np.load(os.path.join(imaging_path, 'F.npy'))
    fneu = np.load(os.path.join(imaging_path, 'Fneu.npy'))
    iscell = np.load(os.path.join(imaging_path, 'iscell.npy'))
    ops = np.load(os.path.join(imaging_path, 'ops.npy'), allow_pickle=True).item()
    seg = np.load(os.path.join(imaging_path, 'meanImg_seg.npy'), allow_pickle=True).item()
    frame_rate = ops['fs']

    return f, fneu, iscell, ops, seg, frame_rate

def get_dff(f,fneu, frame_ix, ops):
    """
    Calculate the dF/F for the imaging data (suite2p default method).
    """
    from suite2p.extraction import dcnv

    all_f = f[:, frame_ix['valid_frames']]
    all_fneu = fneu[:, frame_ix['valid_frames']]
    all_cells_f_corr = all_f - all_fneu*0.7
    dF = dcnv.preprocess(all_cells_f_corr, ops['baseline'], ops['win_baseline'], 
                                    ops['sig_baseline'], ops['fs'], ops['prctile_baseline'])
    print(f"Calculated dF/F with the following parameters: "
        f"baseline={ops['baseline']}, win_baseline={ops['win_baseline']}, "
        f"sig_baseline={ops['sig_baseline']}, fs={ops['fs']},perctile_baseline={ops['prctile_baseline']}")

    return dF

## Displaying cell properties

def show_fov(ops, seg):
    """
    Show the field of view of the imaging data.
    """
    meanImg = ops['meanImg']
    mask = seg['outlines']
    mask = mask.astype(float)
    bool = mask > 0
    bool = bool.astype(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(meanImg, cmap='gray')
    ax.imshow(mask, alpha=bool, cmap='jet')
    ax.set_title('Field of View')
    plt.tight_layout()
    plt.show()

def concat_masks(image_mask: pd.Series):
    masks = np.zeros(image_mask.loc[0].shape)
    for n in range(len(image_mask)):
        tmp = image_mask.loc[n]

        mask_bool = find_boundaries((tmp > 0).astype(int), mode='inner').astype(bool)
        write_here = mask_bool & (masks == 0)
        masks[write_here] = n + 1
    return masks

def show_cell_fov(cell,ops,seg):
    """
    Show the cell's field of view, as a zoom in and the full field of view.
    """
    meanImg = ops['meanImg']
    mask = seg['outlines']
    mask = mask.astype(float)
    bool = mask > 0
    bool = bool.astype(float)
    #find the cool cells in the mask
    cell_mask = np.zeros((mask.shape[0], mask.shape[1]))
    cell_mask[mask == cell+1] = 1

    #create a crop box around the cell
    x, y = np.where(cell_mask == 1)
    x_min = max(0, x.min() - 10)
    x_max = min(mask.shape[0], x.max() + 10)
    y_min = max(0, y.min() - 10)
    y_max = min(mask.shape[1], y.max() + 10)
    crop_mask = meanImg[x_min:x_max, y_min:y_max]

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(crop_mask, cmap='gray')
    ax[0].imshow(cell_mask[x_min:x_max, y_min:y_max], alpha=bool[x_min:x_max, y_min:y_max], cmap='jet')
    ax[0].set_title(f'Cell {cell} zoomed in')

    ax[1].imshow(meanImg, cmap='gray')
    ax[1].imshow(cell_mask,alpha=bool,cmap='jet')
    ax[1].set_title(f'Cell {cell} in full field of view')
    plt.tight_layout()
    plt.show()

# new version 
def show_cell_fov_new(cell:int, meanImg: np.array, mask: np.array):
    """
    Show the cell's field of view, as a zoom in and the full field of view.
    This is similar to show_cell_fov, but uses different input formats.
    """
    mask_br = np.where(
        mask == cell, 1.0,          # exact match
        np.where(mask != 0, -1, 0) # nonzero but not match → 0.5, else 0
    )
    mask_bool = (mask_br != 0).astype(float)

    #create a crop box around the cell
    x, y = np.where(mask == cell)
    x_min = max(0, x.min() - 10)
    x_max = min(mask.shape[0], x.max() + 10)
    y_min = max(0, y.min() - 10)
    y_max = min(mask.shape[1], y.max() + 10)
    crop_mask = meanImg[x_min:x_max, y_min:y_max]

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(crop_mask, cmap='gray')
    ax[0].imshow(mask_br[x_min:x_max, y_min:y_max], alpha=mask_bool[x_min:x_max, y_min:y_max], cmap='bwr')
    ax[0].set_title(f'Cell {cell} zoomed in')

    ax[1].imshow(meanImg, cmap='gray')
    ax[1].imshow(mask_br,alpha=mask_bool,cmap='bwr')
    ax[1].set_title(f'Cell {cell} in full field of view')
    plt.tight_layout()
    plt.show()

def extract_cell_trace(dF,cell,plot=False,session=None,frame_range=None):
    """
    Extract the dF/F trace for a specific cell. The plotting option shows the dF/F trace for the cell, for any desired frame range. 
    If session is provided, it will also plot the rewards, odour events, and position of the mouse.
    """
    dF_cell = dF[cell, :]

    if plot:
        if frame_range is not None:
            plt.figure(figsize=(15, 3))
            plt.plot(dF_cell[frame_range[0]:frame_range[1]], label='Fluorescence (F)', color='blue', alpha=0.5)
            if session is not None:
                for r in session['rewards']:
                    if frame_range[0] <= r < frame_range[1]:
                        plt.axvline(r - frame_range[0], color='black', linestyle='--', label='Reward' if r == session['rewards'][0] else "")
                for m1 in session['modd1']:
                    if frame_range[0] <= m1 < frame_range[1]:
                        plt.axvline(m1 - frame_range[0], color='orange', linestyle='--', label='Modulation 1', alpha=0.5 if m1 == session['modd1'][0] else 0.5)
                for m2 in session['modd2']:
                    if frame_range[0] <= m2 < frame_range[1]:
                        plt.axvline(m2 - frame_range[0], color='purple', linestyle='--', label='Modulation 2', alpha=0.5 if m2 == session['modd2'][0] else 0.5)
                plt.plot(session['position'][frame_range[0]:frame_range[1]], label='Position X', color='red', alpha=0.5)
                #plot landmarks as dots on the position line
                for l in session['landmarks']:
                    l = l[0]
                    l_ix = np.where((session['position'] > l-0.5) & (session['position'] < l+0.5))[0]
                    l_ix = l_ix[l_ix > frame_range[0]]
                    l_ix = l_ix[l_ix < frame_range[1]]
                    l_ix = l_ix - frame_range[0]
                    plt.plot(l_ix, np.ones_like(l_ix) * l, 'o', color='green', label='Landmark')
        else:
            if session is not None:
                plt.figure(figsize=(15, 3))
                for r in session['rewards']:
                    plt.axvline(r, color='black', linestyle='--', label='Reward', alpha=0.5 if r == session['rewards'][0] else 0.2)
                plt.plot(dF_cell, label='Fluorescence (F)', color='blue', alpha=0.5)
                plt.plot(session['position'], label='Position X', color='red', alpha=0.5)
                for l in session['landmarks']:
                    l = l[0]
                    l_ix = np.where((session['position'] > l-0.5) & (session['position'] < l+0.5))[0]
                    plt.plot(l_ix, np.ones_like(l_ix) * l, 'o', color='green', label='Landmark',markersize=2) 
            else:
                plt.figure(figsize=(15, 3))
                plt.plot(dF_cell, label=f'Cell {cell}')
        plt.title(f'dF/F trace for cell {cell}')
        plt.xlabel('Time (frames)')
        plt.ylabel('dF/F')
        plt.show()

    return dF_cell

## Basic tuning properties 

def extract_lick_rate(dF, cell, session):
    #extract licks to calculate lick rate
    licks = session['licks']
    #get lick rate around landmarks
    lick_rate = np.zeros(dF.shape[1])
    for lick in licks:
        if lick < dF.shape[1]:
            lick_rate[lick] += 1
    lick_rate = gaussian_filter1d(lick_rate, sigma=1.5)

    return lick_rate

def extract_reward_tuning(dF, cell, session, frame_rate=45 ,window_size = [-1,5], plot=False):
    """
    Extract the reward tuning for a specific cell. 
    The plotting option shows the average dF/F trace around rewards, the mouses licking rate, and the position of the mouse relative to the reward.
    """
    dF_cell = extract_cell_trace(dF, cell, plot=False, session=session)
    lick_rate = extract_lick_rate(dF, cell, session)

    #outlier detection and removal
    print(f"Number of rewards: {len(session['rewards'])}")
    #calculate the frame intervals between rewards
    frame_intervals = np.diff(session['rewards'])
    #get rid of reward outliers of under 2s (manual rewards, find better way of extracting in the future)
    outliers = np.where(frame_intervals < 2 * frame_rate)[0]
    print(f"Number of outliers: {len(outliers)}")
    #remove outliers from session['rewards']
    session['rewards'] = np.delete(session['rewards'], outliers + 1)
    print(f"Number of rewards after removing outliers: {len(session['rewards'])}")

    window_start = window_size[0]* frame_rate
    window_end = window_size[1] * frame_rate
    window = np.arange(window_start, window_end)
    window_frames = len(window)

    cell_rewards = np.zeros((len(session["rewards"]), window_frames))
    lick_rewards = np.zeros((len(session["rewards"]), window_frames))
    position_rewards = np.zeros((len(session["rewards"]), window_frames))
    for i,r in enumerate(session["rewards"]):
        rew_frames = np.arange(r + window_start, r + window_end)
        if max(rew_frames) < len(dF_cell):
            cell_rewards[i,:] = dF_cell[rew_frames]
            lick_rewards[i,:] = lick_rate[rew_frames]
            position_rewards[i,:] = session["position"][rew_frames] - session["position"][r]
        else:
            print("ignoring last reward,too close to end of session")
            continue
    cell_rewards_avg = np.mean(cell_rewards, axis=0)
    cell_rewards_std = np.std(cell_rewards, axis=0)
    cell_rewards_sem = cell_rewards_std/np.sqrt(len(session["rewards"]))

    lick_rewards_avg = np.mean(lick_rewards, axis=0)
    lick_rewards_std = np.std(lick_rewards, axis=0)
    lick_rewards_sem = lick_rewards_std/np.sqrt(len(session["rewards"]))

    position_rewards_avg = np.mean(position_rewards, axis=0)
    position_rewards_std = np.std(position_rewards, axis=0)
    position_rewards_sem = position_rewards_std/np.sqrt(len(session["rewards"]))

    if plot:
        fig, ax = plt.subplots(1,2,figsize=(10, 5))
        ax[0].plot(cell_rewards_avg, label='Average', color='blue')
        ax[0].fill_between(np.arange(len(cell_rewards_avg)), cell_rewards_avg-cell_rewards_sem, cell_rewards_avg+cell_rewards_sem, color='blue', alpha=0.2)
        ax[0].axvline(np.where(window == 0)[0][0], color='grey', linestyle='--', label='Reward time')
        ax2 = ax[0].twinx()
        ax2.plot(lick_rewards_avg, label='Lick rate', color='grey', alpha=0.5)
        ax2.fill_between(np.arange(len(lick_rewards_avg)), lick_rewards_avg-lick_rewards_sem, lick_rewards_avg+lick_rewards_sem, color='grey', alpha=0.2)
        ax[0].plot(position_rewards_avg, label='Position', color='orange', alpha=0.5)
        ax[0].fill_between(np.arange(len(position_rewards_avg)), position_rewards_avg-position_rewards_sem, position_rewards_avg+position_rewards_sem, color='orange', alpha=0.2)
        ax[0].legend(loc='best')
        ax[0].set_title(f'Cell {cell} - Reward Aligned')
        ax[0].set_xlabel('Time (frames)')
        ax[0].set_ylabel('dF/F')
        ax[1].imshow(cell_rewards, aspect='auto', cmap='viridis', interpolation='none')
        ax[1].axvline(45, color='black', linestyle='--', label='Reward time')
        ax[1].set_title(f'Cell {cell} - All trials {cell_rewards.shape[0]}')
    return cell_rewards

def extract_position_tuning(dF, cell, stage, session, frame_rate=45, bins=200, plot=False):
    """
    Extract the position tuning for a specific cell. 
    The plotting option shows the firing rate as a function of position (average and split into states if applicable).
    """
    dF_cell = extract_cell_trace(dF, cell, plot=False, session=session)
    lick_rate = extract_lick_rate(dF, cell, session)

    lm = np.copy(session['landmarks'])
    goal_a = session['goal_idx'][0]
    goal_b = session['goal_idx'][1]
    goal_c = session['goal_idx'][2]
    goal_d = session['goal_idx'][3]

    bins_pos = bins
    if stage in ['-t3', '-t4', '-t5', '-t6']:
        bin_edges = np.linspace(0, session['position'].max(), bins_pos+1)
    else:
        bin_edges = np.linspace(9, session['position'].max(), bins_pos+1)
    fr_per_bin = np.zeros((session['num_laps'], bins_pos))
    lr_per_bin = np.zeros((session['num_laps'], bins_pos))

    for i in range(session['num_laps']):
        lap_idx = np.where(session['lap_idx'] == i)[0]
        fr_per_lap = dF_cell[lap_idx]
        lr_per_lap = lick_rate[lap_idx]
        bin_ix = np.digitize(session['position'][lap_idx], bin_edges)
        for j in range(bins_pos):
            fr_per_bin[i,j] = np.mean(fr_per_lap[bin_ix == j])
            lr_per_bin[i,j] = np.mean(lr_per_lap[bin_ix == j])


    av_fr_per_bin = np.nanmean(fr_per_bin, axis=0)
    std_fr_per_bin = np.nanstd(fr_per_bin, axis=0)
    sem_fr_per_bin = std_fr_per_bin/np.sqrt(session['num_laps'])
    av_lr_per_bin = np.nanmean(lr_per_bin, axis=0)
    std_lr_per_bin = np.nanstd(lr_per_bin, axis=0)
    sem_lr_per_bin = std_lr_per_bin/np.sqrt(session['num_laps'])
    if session['laps_needed'] > 1:
        state1_av = np.nanmean(fr_per_bin[session['state_id'] == 0], axis=0)
        state1_std = np.nanstd(fr_per_bin[session['state_id'] == 0], axis=0)
        state1_sem = state1_std/np.sqrt(np.sum(session['state_id'] == 0))
        state2_av = np.nanmean(fr_per_bin[session['state_id'] == 1], axis=0)
        state2_std = np.nanstd(fr_per_bin[session['state_id'] == 1], axis=0)
        state2_sem = state2_std/np.sqrt(np.sum(session['state_id'] == 1))
        if session['laps_needed'] == 3:
            state3_av = np.nanmean(fr_per_bin[session['state_id'] == 2], axis=0)
            state3_std = np.nanstd(fr_per_bin[session['state_id'] == 2], axis=0)
            state3_sem = state3_std/np.sqrt(np.sum(session['state_id'] == 2))
    
    if plot:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(fr_per_bin, cmap='viridis', aspect='auto', interpolation='none')
        plt.title(f'Cell {cell} - All Trials Aligned by Position')
        plt.xlabel('Position (cm)')
        plt.ylabel('Lap Number')
        plt.subplot(1, 3, 2)
        ax = plt.gca()
        plt.plot(bin_edges[:-1], av_fr_per_bin, label='Average', color='blue')
        plt.fill_between(bin_edges[:-1], av_fr_per_bin-sem_fr_per_bin, av_fr_per_bin+sem_fr_per_bin, color='blue', alpha=0.2)
        #get y limits
        y_min, y_max = plt.ylim()
        for i in range(len(lm)):
                ax.add_patch(patches.Rectangle((lm[i][0],0),lm[i][1]-lm[i][0],y_max,color='grey',alpha=0.3))
        ax2 = ax.twinx()
        ax2.plot(bin_edges[:-1], av_lr_per_bin, label='Lick Rate', color='orange')
        ax2.fill_between(bin_edges[:-1], av_lr_per_bin-sem_lr_per_bin, av_lr_per_bin+sem_lr_per_bin, color='orange', alpha=0.2)
        plt.title(f'Cell {cell} - Position Aligned')
        plt.xlabel('Position (cm)')
        plt.subplot(1, 3, 3)
        ax = plt.gca()
        if session['laps_needed'] > 1:
            plt.plot(bin_edges[:-1], state1_av, label='State 1', color='blue')
            plt.fill_between(bin_edges[:-1], state1_av-state1_sem, state1_av+state1_sem, color='blue', alpha=0.2)
            plt.plot(bin_edges[:-1], state2_av, label='State 2', color='orange')
            plt.fill_between(bin_edges[:-1], state2_av-state2_sem, state2_av+state2_sem, color='orange', alpha=0.2)
            y_min, y_max = plt.ylim()
            if session['laps_needed'] == 3:
                plt.plot(bin_edges[:-1], state3_av, label='State 3', color='green')
                plt.fill_between(bin_edges[:-1], state3_av-state3_sem, state3_av+state3_sem, color='green', alpha=0.2)
                y_min, y_max = plt.ylim()
                ax.add_patch(patches.Rectangle((lm[goal_a][0],0),lm[goal_a][1]-lm[goal_a][0],y_max,color='blue',alpha=0.3))
                ax.add_patch(patches.Rectangle((lm[goal_b][0],0),lm[goal_b][1]-lm[goal_b][0],y_max,color='blue',alpha=0.3))
                ax.add_patch(patches.Rectangle((lm[goal_c][0],0),lm[goal_c][1]-lm[goal_c][0],y_max,color='orange',alpha=0.3))
                ax.add_patch(patches.Rectangle((lm[goal_d][0],0),lm[goal_d][1]-lm[goal_d][0],y_max,color='green',alpha=0.3))
            else:
                ax.add_patch(patches.Rectangle((lm[goal_a][0],0),lm[goal_a][1]-lm[goal_a][0],y_max,color='blue',alpha=0.3))
                ax.add_patch(patches.Rectangle((lm[goal_b][0],0),lm[goal_b][1]-lm[goal_b][0],y_max,color='blue',alpha=0.3))
                ax.add_patch(patches.Rectangle((lm[goal_c][0],0),lm[goal_c][1]-lm[goal_c][0],y_max,color='orange',alpha=0.3))
                ax.add_patch(patches.Rectangle((lm[goal_d][0],0),lm[goal_d][1]-lm[goal_d][0],y_max,color='orange',alpha=0.3))
        else:
            plt.plot(bin_edges[:-1], av_fr_per_bin, label='Average', color='blue')
            plt.fill_between(bin_edges[:-1], av_fr_per_bin-sem_fr_per_bin, av_fr_per_bin+sem_fr_per_bin, color='blue', alpha=0.2)
            y_min, y_max = plt.ylim()
            ax.add_patch(patches.Rectangle((lm[goal_a][0],0),lm[goal_a][1]-lm[goal_a][0],y_max,color='blue',alpha=0.3))
            ax.add_patch(patches.Rectangle((lm[goal_b][0],0),lm[goal_b][1]-lm[goal_b][0],y_max,color='blue',alpha=0.3))
            ax.add_patch(patches.Rectangle((lm[goal_c][0],0),lm[goal_c][1]-lm[goal_c][0],y_max,color='blue',alpha=0.3))
            ax.add_patch(patches.Rectangle((lm[goal_d][0],0),lm[goal_d][1]-lm[goal_d][0],y_max,color='blue',alpha=0.3))
        for i in range(len(lm)):
                ax.add_patch(patches.Rectangle((lm[i][0],0),lm[i][1]-lm[i][0],y_max,color='grey',alpha=0.3))
        plt.title(f'Cell {cell} - Position Aligned by State')
        plt.xlabel('Position (cm)')
    return fr_per_bin, bin_edges


def extract_speed_tuning(dF, cell, session, bins=40, plot=False):
    """
    Extract the speed tuning for a specific cell. 
    The plotting option shows both the behaviour speed histogram and the cell's firing rate as a function of speed.
    """
    dF_cell = extract_cell_trace(dF, cell, plot=False, session=session)
    dF_cell = dF_cell[1:]  # Remove the first frame to have the same length as speed
    speed = session['speed']
    speed_bins = np.linspace(0, np.max(session['speed']), 41)
    speed_hist, speed_edges = np.histogram(session['speed'], bins=speed_bins)
    speed_hist = speed_hist / np.sum(speed_hist)  # normalize histogram

    bin_edges = np.linspace(0, np.max(session['speed']), bins+1)
    fr_per_spbin = np.zeros((bins))
    std_fr_per_bin = np.zeros((bins))
    sem_fr_per_bin = np.zeros((bins))
    bin_ix = np.digitize(session['speed'], bin_edges)
    n_per_bin = np.zeros(bins)
    for j in range(bins):
        n_per_bin[j] = np.sum(bin_ix == j)
        fr_per_spbin[j] = np.mean(dF_cell[bin_ix == j])
        std_fr_per_bin[j] = np.std(dF_cell[bin_ix == j])
        sem_fr_per_bin[j] = std_fr_per_bin[j] / np.sqrt(n_per_bin[j]) if n_per_bin[j] > 0 else np.nan

    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(speed_edges[:-1], speed_hist, width=np.diff(speed_edges), align='edge', alpha=0.5, color='blue')
        plt.xlabel('Speed (cm/s)')
        plt.ylabel('Normalized Frequency')
        plt.title(f'Behaviour - Speed Histogram')
        plt.subplot(1, 2, 2)
        plt.plot(bin_edges[:-1], fr_per_spbin, label='Average', color='blue')
        plt.fill_between(bin_edges[:-1], fr_per_spbin-sem_fr_per_bin, fr_per_spbin+sem_fr_per_bin, color='blue', alpha=0.2)
        plt.xlabel('Speed (cm/s)')
        plt.ylabel('dF/F')
        plt.title(f'Cell {cell} - Speed Tuning')
    return fr_per_spbin, speed_hist,speed_edges

def extract_modd_tuning(dF, cell, session, frame_rate=45, window_size=[-1, 2], plot=False):
    """
    Extract the tuning of the cell's activity to landmarks.
    """
    dF_cell = extract_cell_trace(dF, cell, plot=False, session=session)
    window_start = window_size[0]* frame_rate
    window_end = window_size[1] * frame_rate
    window = np.arange(window_start, window_end)
    window_frames = len(window)

    #landmark aligned firing rate - fix dropped modd frames first
    all_modd_frames = np.concatenate((session['modd1'], session['modd2']), axis=0)
    all_lm_frames = np.sort(all_modd_frames)
    num_lms = len(all_lm_frames)
    print(f"Found {num_lms} modd frames in the session")
    modd_source = np.zeros(num_lms, dtype=int)
    for i, frame in enumerate(all_lm_frames):
        if frame in session['modd1']:
            modd_source[i] = 1  # Modd 1
        elif frame in session['modd2']:
            modd_source[i] = 2  # Modd 2
    # print(f"Modd source: {modd_source}")
    if np.all(np.diff(modd_source) != 0):
        print("Modd source is always alternating")
    else:
        while np.any(np.diff(modd_source) == 0):
            # If not alternating, print a warning and fix it
            non_alternating_ix = np.where(np.diff(modd_source) == 0)[0][0]
            print(f"Fixing non-alternating modd source at index {non_alternating_ix}")
            # Insert a 0 at the non-alternating index
            modd_source = np.insert(modd_source, non_alternating_ix + 1, 0)
            all_lm_frames = np.insert(all_lm_frames, non_alternating_ix + 1, 0)
    
    num_lms = len(all_lm_frames)
    print(f"Number of landmarks after correction: {num_lms}")
    goal_a = session['goal_idx'][0]
    goal_b = session['goal_idx'][1]
    goal_c = session['goal_idx'][2]
    goal_d = session['goal_idx'][3]
    unique_lms = len(np.unique(session['lm_idx']))-1
    landmark_frames = np.zeros((unique_lms, window_frames))
    if session['laps_needed'] > 1:
        state1_frames = np.zeros((unique_lms, window_frames))
        state2_frames = np.zeros((unique_lms, window_frames))
        if session['laps_needed'] == 3:
            state3_frames = np.zeros((unique_lms, window_frames))
    std_frames = np.zeros((unique_lms, window_frames))
    sem_frames = np.zeros((unique_lms, window_frames))
    for i in range(unique_lms):
        trials = np.zeros((num_lms//unique_lms+1, window_frames))
        for j,r in enumerate(all_lm_frames[i::unique_lms]):
            trial_frames = np.arange(r + window_start, r + window_end)
            if np.min(trial_frames) < 0 or np.max(trial_frames) >= len(dF_cell):
                trials[j,:] = np.full((window_frames,), np.nan)  # Fill with NaNs if out of bounds
            else:
                trials[j,:] = dF_cell[trial_frames]
        landmark_frames[i, :] = np.nanmean(trials, axis=0)
        std_frames[i, :] = np.nanstd(trials, axis=0)
        sem_frames[i, :] = std_frames[i, :] / np.sqrt(trials.shape[0])
        if session['laps_needed'] > 1:
            state1_frames[i, :] = np.nanmean(trials[session['state_id'] == 0], axis=0)
            state2_frames[i, :] = np.nanmean(trials[session['state_id'] == 1], axis=0)
            if session['laps_needed'] == 3:
                state3_frames[i, :] = np.nanmean(trials[session['state_id'] == 2], axis=0)
    if plot:
        plt.figure(figsize=(15, 5))
        if session['laps_needed'] > 1:
            y_absmax = np.max([np.abs(landmark_frames), np.abs(state1_frames), np.abs(state2_frames)]) + 0.1
            y_absmin = np.min([np.abs(landmark_frames), np.abs(state1_frames), np.abs(state2_frames)]) - 0.1
            if session['laps_needed'] == 3:
                y_absmax = np.max([np.abs(landmark_frames), np.abs(state1_frames), np.abs(state2_frames), np.abs(state3_frames)]) + 0.1
                y_absmin = np.min([np.abs(landmark_frames), np.abs(state1_frames), np.abs(state2_frames), np.abs(state3_frames)]) - 0.1
        else:
            y_absmax = np.max(np.abs(landmark_frames)) + 0.1
            y_absmin = np.min(np.abs(landmark_frames)) - 0.1

        for i in range(unique_lms):
            plt.subplot(1, unique_lms, i+1)
            if session['laps_needed'] > 1:
                plt.plot(state1_frames[i, :], label='State 1', color='blue', alpha=0.5)
                plt.plot(state2_frames[i, :], label='State 2', color='orange', alpha=0.5)
                if session['laps_needed'] == 3:
                    plt.plot(state3_frames[i, :], label='State 3', color='green', alpha=0.5)
            plt.plot(landmark_frames[i, :], color='black', label=f'Landmark {i+1}', alpha=0.7)
            plt.fill_between(np.arange(window_frames), landmark_frames[i, :] - sem_frames[i, :], 
                             landmark_frames[i, :] + sem_frames[i, :], color='black', alpha=0.2)
            plt.axvline(np.where(window == 0)[0][0], color='grey', linestyle='--', label='Landmark Time')
            plt.ylim(y_absmin, y_absmax)
            if i == goal_a:
                plt.title('Goal A')
            elif i == goal_b:
                plt.title('Goal B')
            elif i == goal_c:
                plt.title('Goal C')
            elif i == goal_d:
                plt.title('Goal D')
    return landmark_frames, all_lm_frames, modd_source

def extract_lm_tuning(dF, cell, session, frame_rate=45, window_size=[-1, 2], plot=False):
    """Extract the tuning of the cell's activity to landmarks.
    This function finds the first frame where the position is greater than or equal to the landmark position for each lap.
    It then extracts the dF/F trace for the cell around these landmark frames.
    """

    lm_entries = []
    for l in range(session['num_laps']):
        lap_frames = np.where(session['lap_idx']==l+1)[0]
        lap_pos = session['position'][lap_frames]
        lap_pos = lap_pos[3:]  # Remove first 3 frames to avoid NaN values
        # for lm in session['landmarks']: # might also consider non-visited lms
        for lm in session['all_landmarks']:
            #find the first index where the lap position is greater than or equal to the landmark position
            if np.all(lap_pos < lm[0]):
                print(f"Landmark {lm[0]} not found in lap {l+1}")
                continue
            lm_pos = np.where((lap_pos >= lm[0]))[0][0]
            lm_ix = lap_frames[lm_pos]
            lm_entries.append(lm_ix+3)
    lm_entries = np.array(lm_entries)
    num_lms = len(lm_entries)

    dF_cell = extract_cell_trace(dF, cell, plot=False, session=session)
    window_start = window_size[0] * frame_rate
    window_end = window_size[1] * frame_rate
    window = np.arange(window_start, window_end)
    window_frames = len(window)

    goal_a = session['goal_idx'][0]
    goal_b = session['goal_idx'][1]
    goal_c = session['goal_idx'][2]
    goal_d = session['goal_idx'][3]
    unique_lms = len(np.unique(session['all_lms'])) #TODO
    # unique_lms = len(np.unique(session['lm_idx']))-1  # frame idx before first lm is 0

    landmark_frames = np.zeros((unique_lms, window_frames))
    if session['laps_needed'] > 1:
        state1_frames = np.zeros((unique_lms, window_frames))
        state2_frames = np.zeros((unique_lms, window_frames))
        if session['laps_needed'] == 3:
            state3_frames = np.zeros((unique_lms, window_frames))
    std_frames = np.zeros((unique_lms, window_frames))
    sem_frames = np.zeros((unique_lms, window_frames))
    for i in range(unique_lms):
        lm_subset = lm_entries[i::unique_lms]         # all entries for this landmark
        trials = np.zeros((len(lm_subset), window_frames))
        for j, r in enumerate(lm_subset):
    # for i in range(unique_lms):
    #     trials = np.zeros((num_lms//unique_lms, window_frames))
    #     for j,r in enumerate(lm_entries[i::unique_lms]):
            trial_frames = np.arange(r + window_start, r + window_end)
            if np.min(trial_frames) < 0 or np.max(trial_frames) >= len(dF_cell):
                trials[j,:] = np.full((window_frames,), np.nan)  # Fill with NaNs if out of bounds
            else:
                trials[j,:] = dF_cell[trial_frames]
        landmark_frames[i, :] = np.nanmean(trials, axis=0)
        std_frames[i, :] = np.nanstd(trials, axis=0)
        sem_frames[i, :] = std_frames[i, :] / np.sqrt(trials.shape[0])
        if session['laps_needed'] > 1:
            state1_frames[i, :] = np.nanmean(trials[session['state_id'] == 0], axis=0)
            state2_frames[i, :] = np.nanmean(trials[session['state_id'] == 1], axis=0)
            if session['laps_needed'] == 3:
                state3_frames[i, :] = np.nanmean(trials[session['state_id'] == 2], axis=0)
    if plot:
        plt.figure(figsize=(15, 5))
        if session['laps_needed'] > 1:
            y_absmax = np.max([np.abs(landmark_frames), np.abs(state1_frames), np.abs(state2_frames)]) + 0.1
            y_absmin = np.min([np.abs(landmark_frames), np.abs(state1_frames), np.abs(state2_frames)]) - 0.1
            if session['laps_needed'] == 3:
                y_absmax = np.max([np.abs(landmark_frames), np.abs(state1_frames), np.abs(state2_frames), np.abs(state3_frames)]) + 0.1
                y_absmin = np.min([np.abs(landmark_frames), np.abs(state1_frames), np.abs(state2_frames), np.abs(state3_frames)]) - 0.1
        else:
            y_absmax = np.max(np.abs(landmark_frames)) + 0.1
            y_absmin = np.min(np.abs(landmark_frames)) - 0.1

        for i in range(unique_lms):
            plt.subplot(1, unique_lms, i+1)
            if session['laps_needed'] > 1:
                plt.plot(state1_frames[i, :], label='State 1', color='blue', alpha=0.5)
                plt.plot(state2_frames[i, :], label='State 2', color='orange', alpha=0.5)
                if session['laps_needed'] == 3:
                    plt.plot(state3_frames[i, :], label='State 3', color='green', alpha=0.5)
            plt.plot(landmark_frames[i, :], color='black', label=f'Landmark {i+1}', alpha=0.7)
            plt.fill_between(np.arange(window_frames), landmark_frames[i, :] - sem_frames[i, :], 
                             landmark_frames[i, :] + sem_frames[i, :], color='black', alpha=0.2)
            plt.axvline(np.where(window == 0)[0][0], color='grey', linestyle='--', label='Landmark Time')
            plt.ylim(y_absmin, y_absmax)
            if i == goal_a:
                plt.title('Goal A')
            elif i == goal_b:
                plt.title('Goal B')
            elif i == goal_c:
                plt.title('Goal C')
            elif i == goal_d:
                plt.title('Goal D')
    return landmark_frames, lm_entries

def extract_lick_tuning(dF, cell, session, frame_rate=45, window_size=[-1,2], plot=False):
    """
    Extract the tuning of the cell's activity to licks.
    """
    dF_cell = extract_cell_trace(dF, cell, plot=False, session=session)
    window_start = window_size[0] * frame_rate
    window_end = window_size[1] * frame_rate
    window = np.arange(window_start, window_end)
    window_frames = len(window)

    #find first licks inside each landmark
    first_licks = np.zeros((session['num_laps'], len(session['landmarks'])), dtype=int)
    for l in range(session['num_laps']):
        for i, landmark in enumerate(session['landmarks']):
            lap_licks = session['licks_per_lap_frames'][l]
            licks_in_landmark = lap_licks[(session['position'][lap_licks] > landmark[0]) & (session['position'][lap_licks] < landmark[1])]
            if len(licks_in_landmark) == 0:
                first_lick = -1
                first_licks[l, i] = first_lick
            else:
                first_lick = np.min(licks_in_landmark)
                first_licks[l, i] = first_lick
    print(f"Found {np.sum(first_licks > 0)} first licks in the session")
    # Convert first_licks to a boolean to sanity check
    # bool_first_licks = first_licks > 0
    # plt.imshow(bool_first_licks, aspect='auto', cmap='gray', interpolation='none')
    print(f"First licks first lap: {first_licks[0, :]}")
    lin_first_licks = np.reshape(first_licks, -1)
    print(f"First licks flattened: {lin_first_licks[:10]}")

    lick_frames = np.zeros((len(lin_first_licks), window_frames))
    for i, lick in enumerate(lin_first_licks):
        lick_range = np.arange(lick + window_start, lick + window_end)
        if np.min(lick_range) < 0 or np.max(lick_range) >= len(dF_cell):
            lick_frames[i, :] = np.full((window_frames,), np.nan)  # Fill with NaNs if out of bounds
        else:
            lick_frames[i, :] = dF_cell[lick_range]
    non_nan_lick_frames = lick_frames[~np.isnan(lick_frames).any(axis=1)]
    avg_lick_frame = np.nanmean(non_nan_lick_frames, axis=0)
    std_lick_frame = np.nanstd(non_nan_lick_frames, axis=0)
    sem_lick_frame = std_lick_frame / np.sqrt(non_nan_lick_frames.shape[0])

    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(avg_lick_frame, label='Average Lick Response', color='blue')
        plt.fill_between(np.arange(window_frames), avg_lick_frame - sem_lick_frame, 
                        avg_lick_frame + sem_lick_frame, color='blue', alpha=0.2)
        plt.axvline(np.where(window == 0)[0][0], color='grey', linestyle='--', label='Lick Time')
        plt.title(f'Cell {cell} - Lick Tuning')
        plt.xlabel('Time (frames)')
        plt.ylabel('dF/F')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.imshow(non_nan_lick_frames, aspect='auto', cmap='viridis', interpolation='none')
        plt.title(f'Cell {cell} - All Licks Aligned')
        plt.xlabel('Time (frames)')
        plt.ylabel('Lick Trials')
        plt.colorbar(label='dF/F')
        plt.tight_layout()
    return lick_frames, lin_first_licks

def extract_arb_peth(dF,cell,event_frames, frame_rate=45, window_size=[-1,5], plot=False):
    """
    Extract the peri-event time histogram (PETH) for a specific cell around arbitrary events. (window[0] before, window[1] after)
    """
    dF_cell = extract_cell_trace(dF, cell, plot=False)
    window_start = window_size[0] * frame_rate
    window_end = window_size[1] * frame_rate
    window = np.arange(window_start, window_end)
    window_frames = len(window)

    peth_frames = np.zeros((len(event_frames), window_frames))
    for i, event in enumerate(event_frames):
        event_range = np.arange(event + window_start, event + window_end)
        if np.min(event_range) < 0 or np.max(event_range) >= len(dF_cell):
            peth_frames[i, :] = np.full((window_frames,), np.nan)  # Fill with NaNs if out of bounds
        else:
            peth_frames[i, :] = dF_cell[event_range]

    non_nan_peth_frames = peth_frames[~np.isnan(peth_frames).any(axis=1)]
    avg_peth_frame = np.nanmean(non_nan_peth_frames, axis=0)
    std_peth_frame = np.nanstd(non_nan_peth_frames, axis=0)
    sem_peth_frame = std_peth_frame / np.sqrt(non_nan_peth_frames.shape[0])

    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(avg_peth_frame, label='Average PETH', color='blue')
        plt.fill_between(np.arange(window_frames), avg_peth_frame - sem_peth_frame, 
                        avg_peth_frame + sem_peth_frame, color='blue', alpha=0.2)
        plt.axvline(np.where(window == 0)[0][0], color='grey', linestyle='--', label='Event Time')
        plt.title(f'Cell {cell} - Event PETH')
        plt.xlabel('Time (frames)')
        plt.ylabel('dF/F')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.imshow(non_nan_peth_frames, aspect='auto', cmap='viridis', interpolation='none')
        plt.title(f'Cell {cell} - All Events Aligned')
        plt.xlabel('Time (frames)')
        plt.ylabel('Event Trials')
        plt.colorbar(label='dF/F')
        plt.tight_layout()
        plt.show()
    
    return peth_frames

# Goal or arbitrary (e.g. licked landmarks/internal goals) progress coding

def extract_goal_progress(dF,cell,session,frame_rate = 45,bins=90,plot=False,shuffle=False):
    """
    Extract the goal progress for a specific cell.
    """
    #outlier detection and removal
    #calculate the frame intervals between rewards
    frame_intervals = np.diff(session['reward_idx'])
    #get rid of reward outliers of under 2s (manual rewards, find better way of extracting in the future)
    outliers = np.where(frame_intervals < 2 * frame_rate)[0]
    #remove outliers from session['rewards']
    session['reward_idx'] = np.delete(session['reward_idx'], outliers + 1)
    reward_ix = session['reward_idx']

    binned_phase_firing = np.zeros((len(reward_ix)-1, bins))
    ngoals = len(np.unique(session['goal_idx']))
    goal_rew_vec = np.arange(ngoals)
    goal_rew_vec = np.tile(goal_rew_vec, len(session['reward_idx'])//ngoals)
    goal_rew_vec = goal_rew_vec[:-1]

    dF_cell = extract_cell_trace(dF, cell, plot=False, session=session)

    for i in range(len(reward_ix)-1):
        phase_frames = np.arange(reward_ix[i], reward_ix[i+1]-1)
        bin_edges = np.linspace(reward_ix[i], reward_ix[i+1]-1, bins+1)
        phase_firing = dF_cell[phase_frames]
        if shuffle:
            np.random.shuffle(phase_firing)
        bin_ix = np.digitize(phase_frames, bin_edges)
        for j in range(bins):
            binned_phase_firing[i,j] = np.mean(phase_firing[bin_ix == j+1])
    if ngoals == 4:
        binned_B = binned_phase_firing[np.where(goal_rew_vec==0)[0],:]
        binned_C = binned_phase_firing[np.where(goal_rew_vec==1)[0],:]
        binned_D = binned_phase_firing[np.where(goal_rew_vec==2)[0],:]
        binned_A = binned_phase_firing[np.where(goal_rew_vec==3)[0],:]
        min_state = np.min([binned_A.shape[0],binned_B.shape[0],binned_C.shape[0],binned_D.shape[0]], axis=0)
        binned_all = np.concatenate(( binned_B[:min_state,:], binned_C[:min_state,:], binned_D[:min_state,:],binned_A[:min_state,:]), axis=1)
    else:
        print("Only 4 goals are supported for this goal progress extraction for now. Check extract_arb_progress for other specs.")
    avg_bin = np.nanmean(binned_all, axis=0)
    std_bin = np.nanstd(binned_all, axis=0)
    sem_bin = std_bin / np.sqrt(binned_all.shape[0])

    if plot:
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        angles = np.linspace(0, 2 * np.pi, bins*4, endpoint=False)
        # add the first angle to close the circle
        angles = np.concatenate((angles, [angles[0]]))
        avg_bin = np.concatenate((avg_bin, [avg_bin[0]]))
        sem_bin = np.concatenate((sem_bin, [sem_bin[0]]))
        ax1.plot(angles, avg_bin, color='blue', linewidth=2)
        ax1.fill_between(angles, avg_bin - sem_bin, avg_bin + sem_bin, color='blue', alpha=0.2)
        #label the cardinal directions
        ax1.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        ax1.set_xticklabels(['A', 'B', 'C', 'D'])
        ax1.set_title(f'Cell {cell} - Average Firing Rate (Polar)')
        ax2 = fig.add_subplot(122)
        cax = ax2.imshow(binned_all, aspect='auto', cmap='viridis', interpolation='none')
        ax2.set_title(f'Cell {cell} - Binned Firing Rates')
        plt.colorbar(cax, ax=ax2, label='dF/F')
        plt.tight_layout()
    return binned_all, reward_ix

def extract_arb_progress(dF, cell, session, event_frames, ngoals, bins, period='goal', stage=None, plot=False, shuffle=False):
    """
    Extract the progress tuning between arbitrary events.
    """
    dF_cell = extract_cell_trace(dF, cell, plot=False)
    binned_phase_firing = np.zeros((len(event_frames)-1, bins))

    if period == 'goal':
        # Events are organised based on whether they are a goal or not
        if 'sequence' in session and 'shuffled' in session['sequence']:
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

    num_trials = np.zeros(ngoals)
    for i in range(ngoals):
        num_trials[i] = np.sum(goal_vec == i)
    num_trials = num_trials.astype(int)
    max_trials = np.max(num_trials)

    for i in range(len(event_frames)-1):
        phase_frames = np.arange(event_frames[i], event_frames[i+1])
        bin_edges = np.linspace(event_frames[i], event_frames[i+1], bins+1)
        phase_firing = dF_cell[phase_frames]
        if shuffle:
            np.random.shuffle(phase_firing)
        bin_ix = np.digitize(phase_frames, bin_edges)
        for j in range(bins):
            binned_phase_firing[i,j] = np.mean(phase_firing[bin_ix == j+1])

    binned_segment = np.zeros((ngoals, max_trials, binned_phase_firing.shape[1]))
    for i in range(ngoals):
        if num_trials[i] < max_trials:
            binned_segment[i, :num_trials[i], :] = binned_phase_firing[np.where(goal_vec == i)[0], :]
        else:
            binned_segment[i] = binned_phase_firing[np.where(goal_vec == i)[0], :]
    min_state = np.min([binned_segment[i].shape[0] for i in range(ngoals)], axis=0)
    binned_all = np.concatenate([binned_segment[i][:min_state,:] for i in range(ngoals)], axis=1)

    avg_bin = np.nanmean(binned_all, axis=0)
    std_bin = np.nanstd(binned_all, axis=0)
    sem_bin = std_bin / np.sqrt(binned_all.shape[0])

    if plot:
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

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        angles = np.linspace(0, 2 * np.pi, bins*ngoals, endpoint=False)
        # add the first angle to close the circle
        angles = np.concatenate((angles, [angles[0]]))
        avg_bin = np.concatenate((avg_bin, [avg_bin[0]]))
        sem_bin = np.concatenate((sem_bin, [sem_bin[0]]))
        ax1.plot(angles, avg_bin, color=color, linewidth=2)
        ax1.fill_between(angles, avg_bin - sem_bin, avg_bin + sem_bin, color=color, alpha=0.2)
        #label the cardinal directions
        ax1.set_xticks(np.linspace(0, 2 * np.pi, ngoals, endpoint=False))
        ax1.set_title(f'Cell {cell} - Average Firing Rate (Polar)')
        ax2 = fig.add_subplot(122)
        cax = ax2.imshow(binned_all, aspect='auto', cmap='viridis', interpolation='none')
        ax2.set_title(f'Cell {cell} - Binned Firing Rates')
        plt.colorbar(cax, ax=ax2, label='dF/F')
        plt.tight_layout()

    return binned_all

def calc_goal_tuningix(dF, cell, session, condition='goal', period='goal', event_frames=None, n_goals=4, frame_rate=45, bins=90, shuffle=True, plot=False):

    """
    Calculate the goal tuning index for a specific cell by comparing the real score to shuffled scores.
    The tuning index is calculated as the difference between the maximum and minimum firing rates across phases, divided by the mean firing rate across phases.
    """
    if condition == 'goal':
        binned_all, _ = extract_goal_progress(dF, cell, session, frame_rate=frame_rate, bins=bins, plot=False, shuffle=False)
    elif condition == 'arb':
        binned_all = extract_arb_progress(dF, cell, session, event_frames, n_goals, bins, period=period, plot=False, shuffle=False)

    av_binned = np.nanmean(binned_all, axis=0)
    ngoals = av_binned.shape[0]/bins
    ngoals = int(ngoals)
    state_max = np.zeros(ngoals)
    state_min = np.zeros(ngoals)
    state_mean = np.zeros(ngoals)
    pref_phase = np.zeros(ngoals)

    for i in range(ngoals):
        state_max[i] = np.max(av_binned[bins*i:bins*(i+1)])
        state_min[i] = np.min(av_binned[bins*i:bins*(i+1)])
        state_mean[i] = np.mean(av_binned[bins*i:bins*(i+1)])
        pref_phase[i] = np.where(av_binned[bins*i:bins*(i+1)] == state_max[i])[0][0] 
    tuning_score = (state_max - state_min)/state_mean
    real_score = np.mean(tuning_score)
    phase_preference = np.mean(pref_phase)
    state_preference = np.where(state_max == np.max(state_max))[0][0]  # Find the state with the maximum firing rate
    print(f'Real score for cell {cell} is {real_score:.2f}, phase preference is {phase_preference:.2f}, state preference is {state_preference:.2f}')

    if shuffle:
        nreps = 100
        shuffled_scores = []
        for i in range(nreps):
            bins = 90
            if condition == 'goal':
                binned_all, _ = extract_goal_progress(dF, cell, session, frame_rate=frame_rate, bins=bins, plot=False, shuffle=True)
            elif condition == 'arb':
                binned_all = extract_arb_progress(dF, cell, session, event_frames, n_goals, bins, period=period, plot=False, shuffle=True)
            av_binned = np.nanmean(binned_all, axis=0)
            ngoals = av_binned.shape[0]/bins
            ngoals = int(ngoals)
            state_max = np.zeros(ngoals)
            state_min = np.zeros(ngoals)
            state_mean = np.zeros(ngoals)
            for i in range(ngoals):
                state_max[i] = np.max(av_binned[bins*i:bins*(i+1)])
                state_min[i] = np.min(av_binned[bins*i:bins*(i+1)])
                state_mean[i] = np.mean(av_binned[bins*i:bins*(i+1)])
            tuning_score = (state_max - state_min)/state_mean
            tuning_score = np.mean(tuning_score)
            shuffled_scores.append(tuning_score)
        p_value = np.sum(shuffled_scores >= real_score) / nreps
        print(f'P-value for cell {cell} is {p_value:.8f}')
    else:
        shuffled_scores = None
        p_value = None

    if plot:
        plt.figure(figsize=(10, 5))
        plt.hist(shuffled_scores, bins=30, alpha=0.7, color='blue')
        plt.axvline(real_score, color='red', linestyle='dashed', linewidth=2, label='Observed Tuning Score')
        plt.title(f'Histogram of Shuffled Tuning Scores for Cell {cell}')
        plt.xlabel('Tuning Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    return real_score,shuffled_scores,phase_preference,state_preference

## Correlations with other cells

def extract_cell_correlation(dF, cell, ops, seg, session, plot=False):
    """
    Extract the correlation of the cell's activity all other cells in that recording.
    """
    dF_cell = extract_cell_trace(dF, cell, plot=False, session=None)
    correlations = np.zeros(dF.shape[0])
    for i in range(dF.shape[0]):
        correlations[i] = np.corrcoef(dF_cell, dF[i])[0, 1]
    
    highest_five_cells = np.argsort(correlations)[-5:]
    lowest_five_cells = np.argsort(correlations)[:5]
    print(f"Top 5 correlated cells: {highest_five_cells}, correlations: {correlations[highest_five_cells]}")
    print(f"Bottom 5 correlated cells: {lowest_five_cells}, correlations: {correlations[lowest_five_cells]}")

    if plot:
        mask = seg['outlines']
        mask = mask.astype(float)
        bool = mask > 0
        bool = bool.astype(float)
        corr_mask = mask.copy()
        meanImg = ops['meanImg']

        for c in range(dF.shape[0]):
            #create a mask for the cell
            corr_mask[np.where(mask == c+1)] = correlations[c]

        corr_mask[np.where(mask == cell+1)] = np.nan  # Highlight the target cell in the mask

        max_abs_corr = np.nanmax(np.abs(corr_mask))
        plot_corrs = correlations.copy()
        plot_corrs[cell] = np.nan  # Exclude the target cell from the histogram

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(meanImg, cmap='gray')
        plt.imshow(corr_mask, alpha=bool, cmap='coolwarm', vmin=-max_abs_corr, vmax=max_abs_corr)
        cbar = plt.colorbar()
        plt.title(f'Correlation of Cell {cell} with all other cells')
        plt.scatter(np.mean(np.where(mask == cell+1)[1]), np.mean(np.where(mask == cell+1)[0]), color='red', s=100, label=f'Cell {cell}')
        plt.subplot(1, 2, 2)
        plt.hist(plot_corrs, bins=50, color='blue', alpha=0.7)
        plt.axvline(np.nanmean(plot_corrs), color='red', linestyle='dashed', linewidth=1, label='Mean Correlation')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Frequency')


def cluster_all_corr(dF,plot=False):
    """ Cluster all cells based on their correlation with each other.
    """
    correlation_all = np.zeros((dF.shape[0],dF.shape[0]))
    for i in range(dF.shape[0]):
        for j in range(dF.shape[0]):
            correlation_all[i,j] = np.corrcoef(dF[i,:], dF[j,:])[0,1]

    #convert correlation_all to float with 3 decimal places
    correlation_all = correlation_all.astype(float)
    correlation_all = np.round(correlation_all, 3)

    pairwise_distances = sch.distance.pdist(correlation_all)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    correlation_sorted = correlation_all[idx, :][:, idx]

    if plot:
        #set diagonal to NaN
        np.fill_diagonal(correlation_all, np.nan)
        np.fill_diagonal(correlation_sorted, np.nan)
        cmin = np.nanmin(correlation_all)
        cmax = np.nanmax(correlation_all)
        clim = np.max(np.abs([cmin, cmax]))
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        cax1 = ax[0].imshow(correlation_all, cmap='coolwarm', interpolation='none')
        cbar1 = fig.colorbar(cax1, ax=ax[0], orientation='vertical')
        cax1.set_clim(-clim, clim)
        cbar1.set_label('Correlation Coefficient')
        cax2 = ax[1].imshow(correlation_sorted, cmap='coolwarm', interpolation='none')
        cbar2 = fig.colorbar(cax2, ax=ax[1], orientation='vertical')
        cax2.set_clim(-clim, clim)
        cbar2.set_label('Correlation Coefficient')
        ax[0].set_title('Original Correlation Matrix')
        ax[1].set_title('Sorted Correlation Matrix')
        ax[0].set_xlabel('Cells')
        ax[0].set_ylabel('Cells')
        ax[1].set_xlabel('Cells')
        ax[1].set_ylabel('Cells')
        plt.tight_layout()

    return correlation_all, correlation_sorted


def plot_arb_progress_2cells(dF, cell, sessions, event_frames, ngoals, bins, stages, period='goal', labels=None, plot=False, shuffle=False, plot_firing=False, plot_speed_limit=False):

    """
    Extract the progress tuning between arbitrary events for 2 cells.
    """
    binned_all = []
    avg_bin = []
    std_bin = []
    sem_bin = []

    # Check if data are from cell or fake neuron (lick rate)
    fake_neurons = [c for c in range(len(dF)) if dF[c].shape[0] == 1]

    for c in range(2):
        # Extract the neural activity trace
        dF_cell = extract_cell_trace(dF[c], cell[c], plot=False)
        
        # z-score to remove differences across sessions 
        # if not all(s == stages[0] for s in stages):
        dF_cell = stats.zscore(dF_cell)
        
        # Bin the activity
        binned_phase_firing = np.zeros((len(event_frames[c])-1, bins))

        # Create a goal vector 
        if period == 'goal':
            # Events are organised based on whether they are a goal or not
            if ('sequence' in sessions[c]) and ('shuffled' in sessions[c]['sequence']):
                assert ngoals == 2
                goal_vec = np.empty((len(event_frames[c])), dtype=int)
                for i in range(len(event_frames[c])):
                    if i in sessions[c]['goals_idx']:
                        goal_vec[i] = 0
                    elif i in sessions[c]['non_goals_idx']:
                        goal_vec[i] = 1
            else:
                goal_vec = np.arange(ngoals)
                goal_vec = np.tile(goal_vec, len(event_frames[c])//ngoals) 

        elif period == 'landmark':
            # Events are organised based on the order in which they occur
            goal_vec = np.arange(ngoals)
            goal_vec = np.tile(goal_vec, len(event_frames[c])//ngoals)  
        goal_vec = goal_vec[:-1]
        
        # Find number of trials per goal 
        num_trials = np.zeros(ngoals)
        for i in range(ngoals):
            num_trials[i] = np.sum(goal_vec == i)
        num_trials = num_trials.astype(int)
        max_trials = np.max(num_trials)

        # Bin firing between events
        for i in range(len(event_frames[c])-1):
            phase_frames = np.arange(event_frames[c][i], event_frames[c][i+1])
            bin_edges = np.linspace(event_frames[c][i], event_frames[c][i+1], bins+1)
            phase_firing = dF_cell[phase_frames]
            if shuffle:
                np.random.shuffle(phase_firing)
            bin_ix = np.digitize(phase_frames, bin_edges)
            for j in range(bins):
                binned_phase_firing[i,j] = np.mean(phase_firing[bin_ix == j+1])

        # Get binned phase firing per goal
        binned_segment = np.zeros((ngoals, max_trials, binned_phase_firing.shape[1]))

        for i in range(ngoals):
            if num_trials[i] < max_trials:
                binned_segment[i, :num_trials[i], :] = binned_phase_firing[np.where(goal_vec == i)[0], :]
            else:
                binned_segment[i] = binned_phase_firing[np.where(goal_vec == i)[0], :]
        min_state = np.min([binned_segment[i].shape[0] for i in range(ngoals)], axis=0)
        binned_all.append(np.concatenate([binned_segment[i][:min_state,:] for i in range(ngoals)], axis=1))

        avg_bin.append(np.nanmean(binned_all[c], axis=0))
        std_bin.append(np.nanstd(binned_all[c], axis=0))
        sem_bin.append(std_bin[c] / np.sqrt(binned_all[c].shape[0]))

    if plot:
        cell = np.array(cell).astype(int)

        # Define colors 
        colors = np.empty(len(stages), dtype=object)
        
        if all(s == stages[0] for s in stages):
            
            if stages[0] == 3:
                colors[0] = '#325235'
                colors[1] = '#6AC272'
            elif stages[0] == 4:
                colors[0] = '#9E664C'
                colors[1] = '#E68558'
            elif stages[0] == 5:
                colors[0] = 'blue'
                colors[1] = 'deepskyblue'
            elif stages[0] == 6:
                colors[0] = 'orange'
                colors[1] = 'gold'
            elif stages[0] == 8:
                colors[0] = 'red'
                colors[1] = 'tomato'
            elif stages[0] == 12:
                colors[0] = 'teal'
                colors[1] = 'lightseagreen'
                
            if labels is None:
                labels = np.empty(len(stages), dtype=object)
                labels[0] = f'T{stages[0]} - rewards'
                labels[1] = f'T{stages[1]} - licks'
        else:
            for i, s in enumerate(stages):
                if s == 5:
                    colors[i] = 'blue'
                elif s == 6:
                    colors[i] = 'orange'
                elif s == 8:
                    colors[i] = 'red'
            if labels is None:
                labels = np.empty(len(stages), dtype=object)
                for i, s in enumerate(stages):
                    labels[i] = f'T{stages[i]} - cell {cell[i]}'

        # Plot
        fig = plt.figure(figsize=(10, 5))
        
        if not plot_firing:
            ax1 = fig.add_subplot(111, projection='polar')
        else:
            ax1 = fig.add_subplot(121, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        angles = np.linspace(0, 2 * np.pi, bins*ngoals, endpoint=False)
        angles = np.concatenate((angles, [angles[0]])) # add the first angle to close the circle
        
        for s in range(len(stages)):
            avg_bin[s] = np.concatenate((avg_bin[s], [avg_bin[s][0]]))
            sem_bin[s] = np.concatenate((sem_bin[s], [sem_bin[s][0]]))
            
            if not len(fake_neurons) == len(cell):
                stage_max = np.max(avg_bin[s])  
                avg_bin[s] = avg_bin[s] / stage_max
                sem_bin[s] = sem_bin[s] / stage_max

            ax1.plot(angles, avg_bin[s], color=colors[s], linewidth=2)
            ax1.fill_between(angles, avg_bin[s] - sem_bin[s], avg_bin[s] + sem_bin[s], color=colors[s], alpha=0.5, label=labels[s])
            
        # label the cardinal directions
        ax1.set_xticks(np.linspace(0, 2 * np.pi, ngoals, endpoint=False))
        if len(fake_neurons) == len(cell):
            ax1.set_title("Lick Rate and Speed (Polar)")

            if plot_speed_limit:
                ax1.plot(angles, np.full_like(angles, sessions[0]['lick_threshold']), 
                    color='black', linestyle='--', linewidth=1.5, label='speed threshold')
         
        elif len(fake_neurons) > 0:
            real_idx = [i for i in range(len(cell)) if i not in fake_neurons][0]  # first non-fake idx
            ax1.set_title(f'Cell {cell[real_idx]} - Average Firing Rate (Polar)')
        else:
            ax1.set_title(f'Cells {cell} - Average Firing Rate (Polar)')

        plt.legend(loc='upper right')

        if plot_firing: 
            ax2 = fig.add_subplot(122)
            cax = ax2.imshow(binned_all[0], aspect='auto', cmap='viridis', interpolation='none')
            ax2.set_yticks([0, len(binned_all[0])-1])
            ax2.set_yticklabels([0, len(binned_all[0])])
            ax2.set_ylabel('Lap')
            
            if stages[0] == 3 or stages[0] == 4:
                xtick_positions = [i * bins + bins // 2 for i in range(ngoals)]
                ax2.set_xlabel('Landmark')
                ax2.set_xticks(xtick_positions)
                ax2.set_xticklabels(['A', 'B'])
            elif stages[0] == 5 or stages[0] == 6:
                xtick_positions = [i * bins for i in range(ngoals)]
                ax2.set_xlabel('Goal')
                ax2.set_xticks(xtick_positions)
                ax2.set_xticklabels(['A', 'B', 'C', 'D', 'test'])

            if len(fake_neurons) == len(cell):
                ax2.set_title("Binned Lick Rate and Speed (Polar)")
            elif len(fake_neurons) > 0:
                real_idx = [i for i in range(len(cell)) if i not in fake_neurons][0]  # first non-fake idx
                ax2.set_title(f'Cell {cell[real_idx]} - Binned Firing Rate')
            else:
                ax2.set_title(f'Cells {cell} - Binned Firing Rate')

            plt.colorbar(cax, ax=ax2, label='dF/F')

        plt.tight_layout()

    return binned_all


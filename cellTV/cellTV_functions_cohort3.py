import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.segmentation import find_boundaries
import scipy.stats as stats
import parse_nwb_functions as parse_nwb_functions
from pynwb import NWBHDF5IO

# import h5py
# from scipy.signal import find_peaks
# from sklearn.preprocessing import normalize

# import pickle
# import parse_session_functions
# from suite2p.extraction import dcnv
# import scipy.stats as stats
# from scipy.ndimage import gaussian_filter1d
# import matplotlib.patches as patches
# import scipy.cluster.hierarchy as sch
# import pandas as pd


## Loading functions 

def load_img_data(imaging_path):
    """
    Load the imaging data (no spikes yet) from the specified path.
    """
    f = np.load(os.path.join(imaging_path, 'F.npy'))
    fneu = np.load(os.path.join(imaging_path, 'Fneu.npy'))
    iscell = np.load(os.path.join(imaging_path, 'iscell.npy'))
    ops = np.load(os.path.join(imaging_path, 'ops.npy'), allow_pickle=True).item()
    frame_rate = ops['fs']

    funcimg_data = {
        'f': f,
        'fneu': fneu,
        'iscell': iscell,
        'ops': ops,
        'frame_rate': frame_rate
    }

    # Check for red channel data 
    f2_path = os.path.join(imaging_path, 'F_chan2.npy')
    if os.path.exists(f2_path):
        f2 = np.load(f2_path)
        funcimg_data['f2'] = f2

    fneu2_path = os.path.join(imaging_path, 'Fneu_chan2.npy')
    if os.path.exists(fneu2_path):
        fneu2 = np.load(fneu2_path)
        funcimg_data['fneu2'] = fneu2

    redcell_path = os.path.join(imaging_path, 'redcell.npy') # TODO what is this
    if os.path.exists(redcell_path):
        redcell = np.load(redcell_path)
        funcimg_data['redcell'] = redcell

    return funcimg_data

def load_dF_data(funcimg_data, frame_ix, save_path, reload=False):
    '''Load or calculate dF and dG/R data if two channels are available'''
    
    # Load or calculate dF
    dF_path = os.path.join(save_path, 'DF_F0.npy')
    if os.path.exists(dF_path) and not reload:
        print('dF file found. Loading...')
        dF = np.load(dF_path)
    else:
        dF = get_dff(funcimg_data, frame_ix, chan=1)
        np.save(dF_path, dF)

    # Check for red channel data
    if 'f2' in funcimg_data:
        dF2_path = os.path.join(save_path, 'DF_F02.npy')
        if os.path.exists(dF2_path) and not reload:
            print('dF2 file found. Loading...')
            dFred = np.load(dF2_path)
        else:
            dFred = get_dff(funcimg_data, frame_ix, chan=2)
            np.save(dF2_path, dFred)

        dF_GR_path = os.path.join(save_path, 'DG_R.npy')
        if os.path.exists(dF_GR_path) and not reload:
            print('dF_GR file found. Loading...')
            dF_GR = np.load(dF_GR_path)
        else:
            dF_GR = get_dff_GR(funcimg_data, frame_ix)
            np.save(dF_GR_path, dF_GR)
    else:
        dFred = None
        dF_GR = None
        
    return dF, dFred, dF_GR

def load_dF(mouse, session_id, funcimg_root, behav_root, save_path, reload=False):
    '''Load dF and valid frames and find valid neurons'''

    # Load funcimg data 
    imaging_path = get_funcimg_path(mouse, session_id, funcimg_root)
    funcimg_data = load_img_data(imaging_path)
    
    # Load valid frames
    frame_ix = load_valid_frames(mouse, session_id, behav_root)

    # Load or calculate dF
    dF, dFred, dF_GR = load_dF_data(funcimg_data, frame_ix, save_path, reload=reload)

    # Find valid neurons
    neurons = get_responsive_neurons(dF, imaging_path, plot_deltas=False, plot_responses=False)

    return dF, dFred, dF_GR, neurons

def get_responsive_neurons(dF, imaging_path, save_path=None, plot_deltas=False, n_deltas=20, plot_responses=False, n_responses=20):
    '''Get ROIs that not just artifacts or constitutively active neurons.'''
        
    # 1. Select neurons that have been manually selected as valid ROIs
    iscell = np.load(os.path.join(imaging_path, 'iscell.npy'))[:,0]
    green = np.where(iscell == 1)[0]

    redcell = np.load(os.path.join(imaging_path, 'redcell.npy')) # TODO what is this
    red = np.where(redcell == 1)[0]

    neurons = np.intersect1d(red, green)
    # nwb_path = parse_nwb_functions.find_nwbfile(nwb_base_path)
    # io = NWBHDF5IO(nwb_path, mode='r')
    # nwb = io.read()
    
    # segmentation = nwb.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentationChan1Plane0'][:]
    # neurons = np.where(segmentation['Accepted'] == 1)[0] 

    # io.close()

    # 2. Select neurons based on response patterns during the session
    neurons_considered = []

    for i, n in enumerate(neurons): 
        deltas = dF[n,:] - np.mean(dF[n,:])

        # Skip cells with very large deviations
        if len(np.where(~np.isfinite(deltas))[0]) > 0:
            print(f'{n} is a strange neuron')
            continue

        if np.max(deltas) > 500:
            continue

        # Keep cells with different responses below or above the mean
        skew = stats.skew(deltas)
        if np.abs(skew) > 0.4: 
            neurons_considered.append(n)

        if plot_deltas and (i < n_deltas):
            plt.figure()
            plt.hist(deltas, bins=100)
            plt.title(f'{n}, {skew}, {np.max(deltas)}')
        
    print(f'Out of the {dF.shape[0]} ROIs, {len(neurons)} are classified as neurons, and {len(neurons_considered)} have time-varying Ca2+ signals and will be considered here.')

    neurons_considered = np.array(neurons_considered)
    if save_path is None:
        save_path = imaging_path
    np.savez(os.path.join(save_path, f'responsive_neurons.npz'), neurons_considered)

    # Plot dF/F for a few responsive neurons
    if plot_responses:
        for n in neurons_considered[:n_responses]:
            plot_range = [10000, 20000]
            xdata = np.arange(plot_range[0], plot_range[1]) 
            fig, ax = plt.subplots(1, 1, figsize=(15,4), sharey=True)
            ax.plot(xdata, dF[n, xdata])
            ax.set_xlabel('Frame')
            ax.set_ylabel('DF/F')
            plt.suptitle(f'neuron {n}')

    return neurons_considered

def get_dff(funcimg_data, frame_ix, chan=1):
    """
    Calculate the dF/F for the imaging data (suite2p default method).
    """
    from suite2p.extraction import dcnv

    ops = funcimg_data['ops']
    if chan == 1:
        print('Calculating dF using green channel')
        f = funcimg_data['f']
        fneu = funcimg_data['fneu']
    elif chan == 2:
        print('Calculating dF using red channel')
        f = funcimg_data['f2']
        fneu = funcimg_data['fneu2']
        
    all_f = f[:, frame_ix['valid_frames']]
    all_fneu = fneu[:, frame_ix['valid_frames']]
    all_cells_f_corr = all_f - all_fneu*0.7
    dF = dcnv.preprocess(all_cells_f_corr, ops['baseline'], ops['win_baseline'], 
                                    ops['sig_baseline'], ops['fs'], ops['prctile_baseline'])
    
    print(f"Calculated dF/F with the following parameters: "
        f"baseline={ops['baseline']}, win_baseline={ops['win_baseline']}, "
        f"sig_baseline={ops['sig_baseline']}, fs={ops['fs']},perctile_baseline={ops['prctile_baseline']}")

    return dF

def get_dff_GR(funcimg_data, frame_ix):
    from suite2p.extraction import dcnv
    from scipy.signal import medfilt

    ops = funcimg_data['ops']
    f = funcimg_data['f']
    f2 = funcimg_data['f2']
    fneu = funcimg_data['fneu']
    fneu2 = funcimg_data['fneu2']

    all_f = f[:, frame_ix['valid_frames']]
    all_fneu = fneu[:, frame_ix['valid_frames']]
    all_f2 = f2[:, frame_ix['valid_frames']]
    all_fneu2 = fneu2[:, frame_ix['valid_frames']]

    # Neuropil correction
    f_corr = all_f - all_fneu * 0.7
    f_corr2 = all_f2 - all_fneu2 * 0.7

    # Median filter on each trace 
    w = int(ops['fs'] * 0.25) | 1  # force odd
    # print(f'Smoothing both green and red channel data with a median filter of kernel size {w}')
    print(f'Smoothing only red channel data with a median filter of kernel size {w}')
    # f_corr = medfilt(f_corr, kernel_size=(1, w))
    f_corr2 = medfilt(f_corr2, kernel_size=(1, w))

    # ensure denominator is not too low, otherwise the G/R ratio will blow up
    f_corr2_floor = np.percentile(f_corr2, 5, axis=1, keepdims=True) 
    f_corr2_floor = np.maximum(f_corr2_floor, 1e-3)

    # Compute G/R ratio
    f_ratio = f_corr / np.maximum(f_corr2, f_corr2_floor)
    # f_ratio = f_corr / f_corr2

    # Compute dG/R
    dF_GR = dcnv.preprocess(f_ratio, ops['baseline'], ops['win_baseline'], 
                                    ops['sig_baseline'], ops['fs'], ops['prctile_baseline'])

    return dF_GR

def get_funcimg_path(mouse, session_id, root):

    mouse_id = f"sub-{mouse}"
    mouse_path = root / mouse_id
    session_folder = [p for p in mouse_path.iterdir()
        if p.is_dir() and f"{session_id}" in p.name][0]

    imaging_path = session_folder / 'funcimg' / 'suite2p' / 'plane0'

    return imaging_path

def load_valid_frames(mouse, session_id, root):

    mouse_id = f"sub-{mouse}"
    mouse_path = root / mouse_id
    session_folder = [p for p in mouse_path.iterdir()
        if p.is_dir() and f"{session_id}" in p.name][0]
    
    frame_ix = np.load(os.path.join(session_folder, 'valid_frames.npz'))

    return frame_ix

def plot_fluorescence_traces(neurons, funcimg_data, dF, dFred=None, dF_GR=None):

    for n in neurons[0:10]:
        if dFred is not None:
            _, ax = plt.subplots(3, 1, figsize=(10,8), sharey=False)
            ax = ax.ravel()
        else:
            _, ax = plt.subplots(1, 1, figsize=(10,3))

        ax[0].plot(funcimg_data['f'][n,0:2000], label='F', color='green')
        ax[0].plot(funcimg_data['fneu'][n,0:2000], label='Fneu', color='darkgreen')
        ax[0].plot(dF[n,0:2000], label='dF', color='gray')
        ax[0].legend()

        if dFred is not None:
            ax[1].plot(funcimg_data['f2'][n,0:2000], label='F2', color='red')
            ax[1].plot(funcimg_data['fneu2'][n,0:2000], label='Fneu2', color='darkred')
            ax[1].plot(dFred[n,0:2000], label='dFred', color='gray')
            ax[1].legend()

            if dF_GR is not None:
                ax[2].plot(dF_GR[n,0:2000], label='dF(G/R)', c='black')
                ax[2].set_ylim([-0.2,1])
            else:
                ax[2].plot(dFred[n,0:2000], label='dFred', c='red')
                ax[2].plot(dF[n,0:2000], label='dF', c='green')
            ax[2].legend()
        
        plt.tight_layout()


## Displaying cell properties

def concat_masks(image_mask: pd.Series):
    masks = np.zeros(image_mask.loc[0].shape)
    for n in range(len(image_mask)):
        tmp = image_mask.loc[n]

        mask_bool = find_boundaries((tmp > 0).astype(int), mode='inner').astype(bool)
        write_here = mask_bool & (masks == 0)
        masks[write_here] = n + 1
    return masks

def show_cell_fov(cell:int, meanImg: np.array, mask: np.array):
    """
    Show the cell's field of view, as a zoom in and the full field of view.
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

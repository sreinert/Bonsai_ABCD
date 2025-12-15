import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from tkinter.filedialog import askdirectory, askopenfilename
# import defopt
from barcode_util import read_h5, read_h5_with_key
from scipy.signal import find_peaks
import pickle

def align_barcode(sess_data_path:str, 
                            *, 
                    func_img_path:str = '',
                    func_img_header_fname:str = '', 
                    behav_data_header_fname:str = '', 
                    n_channels:int = 3, 
                    convert_frames:bool = True, 
                    save_trans:bool = True):
    """
    Display a friendly greeting.

    :param sess_data_path: sess_data_path
    :param func_img_header_fname: sess_data_path
    :param behav_data_header_fname: sess_data_path
    :param expected_sample_rate: Number of times to display the greeting
    :param height: Number of times to display the greeting
    """
    # n_channels = 3
    height = 1
    # Input variables
    main_sample_rate = 9000 # Expected sample rate of main data, in Hz
    secondary_sample_rate = 5000 # Expected sample rate of secondary data, in Hz
    convert_timestamp_column = 0 # Column that timestamps are located in secondary data

    if os.name == 'nt': # Windows system
        sess_data_path = Path(sess_data_path)
    else:
        sess_data_path = Path(sess_data_path)

    vdaq_signals_file = sess_data_path / func_img_path / func_img_header_fname
    vdaq_signals_file_fname = func_img_header_fname.split('.')[0]
    # vdaq_FrameTTL = read_h5(vdaq_signals_file, key_idx=1)
    vdaq_FrameTTL = read_h5_with_key(vdaq_signals_file, key='FrameTTL')
    vdaq_FrameTTL = np.array(vdaq_FrameTTL)
    vdaq_SyncTTL_ts = np.load(sess_data_path / func_img_path / (vdaq_signals_file_fname + '_barcode.npy'))

    behav_signals_file = sess_data_path / 'behav' / behav_data_header_fname
    arr = np.fromfile(behav_signals_file, dtype=np.float64) # Needs to be float64!
    behav_DAQ_data = np.reshape(arr, (-1,n_channels)).T  
    behav_signals_file_fname = behav_data_header_fname.split('.')[0]
    behav_SyncTTL_ts = np.load(sess_data_path / 'behav' / (behav_signals_file_fname + '_barcode.npy'))


    # Pull the barcode row from the data. 1st column is timestamps, second is barcodes
    barcode_timestamps_row = 0 # Same for both main and secondary, because we used our own code
    barcodes_row = 1 # Same for both main and secondary

    main_numpy_barcode = behav_SyncTTL_ts[barcodes_row, :]
    secondary_numpy_barcode = vdaq_SyncTTL_ts[barcodes_row, :]

    main_numpy_timestamp = behav_SyncTTL_ts[barcode_timestamps_row, :]
    secondary_numpy_timestamp = vdaq_SyncTTL_ts[barcode_timestamps_row, :]

    # Pull the index values from barcodes shared by both groups of data
    shared_barcodes, main_index, second_index = np.intersect1d(main_numpy_barcode,
                                                secondary_numpy_barcode, return_indices=True)
    # Note: To intersect more than two arrays, use functools.reduce

    # Use main_index and second_index arrays to extract related timestamps
    main_shared_barcode_times = main_numpy_timestamp[main_index]
    secondary_shared_barcode_times = secondary_numpy_timestamp[second_index]

    # Determine slope (m) between main/secondary timestamps
    m = (main_shared_barcode_times[-1]-main_shared_barcode_times[0])/(secondary_shared_barcode_times[-1]-secondary_shared_barcode_times[0]) 
    # Determine offset (b) between main and secondary barcode timestamps
    b = main_shared_barcode_times[0] - secondary_shared_barcode_times[0] * m

    print('Linear conversion from secondary timestamps to main:\ny = ', m, 'x + ', b)

    ##################################################################
    ### Apply Linear Conversion to Secondary Data (in .npy Format) ###
    ##################################################################

    # From original code: 
    # vdaq_SyncTTL_ts[convert_timestamp_column,:] = vdaq_SyncTTL_ts[convert_timestamp_column,:] * secondary_sample_rate * m + b
    vdaq_SyncTTL_ts[convert_timestamp_column,:] = vdaq_SyncTTL_ts[convert_timestamp_column,:] * m + b
    secondary_data_converted = vdaq_SyncTTL_ts # To show conversion complete.

    # Clean up conversion of values to nearest whole number
    for index in range(0,len(secondary_data_converted[convert_timestamp_column,:])):
        value = secondary_data_converted[convert_timestamp_column, index]
        rounded_val = value.astype('int')
        secondary_data_converted[convert_timestamp_column, index] = rounded_val

    ##################################################################
    ### Apply Linear Conversion to Imaging Frames (in .npy Format) ###
    ##################################################################

    if convert_frames: 
        # Get the index of FrameTTL
        binary_TTL = vdaq_FrameTTL > 4.5  # Convert to binary: 0 (low) / 1 (high)
        FrameTTL_index = np.where(np.diff(binary_TTL.astype(int)) == 1)[0] -1

        # FrameTTL_converted = FrameTTL_index * m_frame + b_frame
        FrameTTL_converted = FrameTTL_index * m + b

        # Clean up conversion of values to nearest whole number
        print("Total number of index values: ", len(FrameTTL_converted))
        FrameTTL_converted_rounded = []
        for index in range(0,len(FrameTTL_converted)):
            value = FrameTTL_converted[index]
            rounded_val = value.astype('int')
            FrameTTL_converted_rounded.append(rounded_val)
        FrameTTL_converted_rounded = [int(value) for value in FrameTTL_converted]

    ################################################################
    ### Print out final output and save to chosen file format(s) ###
    ################################################################

    # Test to see output here:
    print("Final output for LJ Data:\n", secondary_data_converted)

    output_file = sess_data_path / func_img_path / 'nidaq_aligned_ts.npy'
    np.save(output_file, secondary_data_converted)

    # Save behav aligned frame ts
    if convert_frames: 
        output_file = sess_data_path / func_img_path / 'nidaq_aligned_frames.npy'
        np.save(output_file, FrameTTL_converted_rounded)

    # Save slope and offset as pickle
    if save_trans:
        trans = {"slope": m, "offset": b}
        with open(sess_data_path / func_img_path / 'nidaq_alignment_trans_vars.pkl', "wb") as f:
            pickle.dump(trans, f)

    return m, b

if __name__ == '__main__':
    # defopt.run(align_barcode)

    sess_data_path = '/media/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/TAA0000066/ses-012_date-20250318_protocol-t6'
    func_img_path = 'funcimg/Session'
    func_img_header_fname = '20250318_152251__TAA0000066_00001.h5'
    behav_data_header_fname = '20250318_M8_ai.bin'
    n_channels = 6

    align_barcode(sess_data_path = sess_data_path, 
                            func_img_path = func_img_path, 
                            func_img_header_fname = func_img_header_fname, 
                            behav_data_header_fname = behav_data_header_fname, 
                            n_channels=n_channels, 
                            convert_frames=True, 
                            save_trans=False)
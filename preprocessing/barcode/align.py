import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from tkinter.filedialog import askdirectory, askopenfilename
import defopt
from barcode_util import read_h5
from scipy.signal import find_peaks

def align_barcode_timestamp(sess_data_path:str, 
                            *, 
                    func_img_header_fname:str = '', 
                    behav_data_header_fname:str = '', 
                    n_channels:int = 3):
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
    # main_sample_rate = 9000 # Expected sample rate of main data, in Hz
    # secondary_sample_rate = 5000 # Expected sample rate of secondary data, in Hz
    # convert_timestamp_column = 0 # Column that timestamps are located in secondary data

    if os.name == 'nt': # Windows system
        sess_data_path = Path(sess_data_path)
    else:
        sess_data_path = Path(sess_data_path)

    vdaq_signals_file = sess_data_path / 'funcimg' / func_img_header_fname
    vdaq_signals_file_fname = func_img_header_fname.split('.')[0]
    vdaq_FrameTTL = read_h5(vdaq_signals_file, key_idx=1)
    vdaq_FrameTTL = np.array(vdaq_FrameTTL)
    vdaq_SyncTTL_ts = np.load(sess_data_path / 'funcimg' / (vdaq_signals_file_fname + '_barcode.npy'))

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

    # Get the index of FrameTTL
    FrameTTL_index, _ = find_peaks(np.abs(np.diff(vdaq_FrameTTL)), height=height)
    secondary_data_converted = FrameTTL_index * m + b

    # Clean up conversion of values to nearest whole number
    print("Total number of index values: ", len(secondary_data_converted))
    r = []
    for index in range(0,len(secondary_data_converted)):
        value = secondary_data_converted[index]
        rounded_val = value.astype('int')
        r.append(rounded_val)

    behav_DAQ_data

    ################################################################
    ### Print out final output and save to chosen file format(s) ###
    ################################################################

    # Test to see output here:
    print("Final output for LJ Data:\n", secondary_data_converted)

    time_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    output_file = sess_data_path / 'funcimg' / ('behav_aligned_' + time_now)
    np.save(output_file, secondary_data_converted)

if __name__ == '__main__':
    defopt.run(align_barcode_timestamp)
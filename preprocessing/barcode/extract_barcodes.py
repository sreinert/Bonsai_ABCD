import numpy as np
import sys
from scipy.signal import find_peaks
from datetime import datetime
from pathlib import Path
import os
# import defopt
from .barcode_util import read_h5, read_h5_with_key
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

def extract_barcode(sess_data_path:str, 
                    *, 
                    func_img_path:str = '',
                    func_img_header_fname:str = '', 
                    behav_data_header_fname:str = '', 
                    expected_sample_rate:int = 5000, height:float = 2, n_channels:int = 3):
    """
    Display a friendly greeting.

    :param sess_data_path: sess_data_path
    :param func_img_header_fname: sess_data_path
    :param behav_data_header_fname: sess_data_path
    :param expected_sample_rate: Number of times to display the greeting
    :param height: Number of times to display the greeting

    NOTE: the barcode extraction is very sensitive to the height parameter. 
    In a future version, this should be removed e.g., by smoothing the barcode signal (?).
    """
    if os.name == 'nt': # Windows system
        sess_data_path = Path(sess_data_path)
    else:
        sess_data_path = Path(sess_data_path)

    # NP events come in as indexed values, with 1 to indicate when a TTL pulse changed
    # on to off (directionality). Other DAQ files (like LJ) come in 'raw' digital format,
    # with 0 to indicate TTL off state, and 1 to indicate on state.
    raw_data_format = True  # Set raw_data_format to True for LJ-like data, False for NP.
    signals_column = 0 # Column in which sorted timestamps or raw barcode data
    
    # appears (Base zero; 1st column = 0)
    # 30k Hz for the Neuropixel. Choose based on your DAQ's sample rate.
    global_tolerance = .20 # The fraction (percentage) of tolerance allowed for
    # duration measurements.
    # (Ex: If global_tolerance = 0.2 and ind_wrap_duration = 10, then any signal
    # change between 8-12 ms long will be considered a barcode wrapper.)

    if (func_img_header_fname != '') & (behav_data_header_fname == ''):
        signals_file = sess_data_path / func_img_path / func_img_header_fname
        output_path = sess_data_path / func_img_path
        output_name = func_img_header_fname.split('.')[0]

        # data = read_h5(signals_file, key_idx=2)
        data = read_h5_with_key(signals_file, key='SyncTTL')
        data = np.array(data)

    elif (behav_data_header_fname != '') & (func_img_header_fname == ''):
        signals_file = sess_data_path / 'behav' / behav_data_header_fname
        output_path = sess_data_path / 'behav'
        output_name = behav_data_header_fname.split('.')[0]

        arr = np.fromfile(signals_file, dtype=np.float64) # Needs to be float64!
        arr = np.reshape(arr, (-1,n_channels)).T  
        data = arr[signals_column,:]
    else:
        print('Please set either input')

    # General variables; make sure these align with the timing format of
    # your Arduino-generated barcodes.
    nbits = 16
    inter_barcode_interval = 1000  # In milliseconds
    ind_wrap_duration = 10  # In milliseconds
    ind_bar_duration = 30 # In milliseconds

    ################################################################################
    ############################ END OF USER INPUT #################################
    ################################################################################

    ###########################################################
    ### Set Global Variables/Tolerances Based on User Input ###
    ###########################################################

    wrap_duration = 3 * ind_wrap_duration # Off-On-Off
    total_barcode_duration = nbits * ind_bar_duration + 2 * wrap_duration

    # Tolerance conversions
    min_wrap_duration = ind_wrap_duration - ind_wrap_duration * global_tolerance
    max_wrap_duration = ind_wrap_duration + ind_wrap_duration * global_tolerance
    min_bar_duration = ind_bar_duration - ind_bar_duration * global_tolerance
    max_bar_duration = ind_bar_duration + ind_bar_duration * global_tolerance
    sample_conversion = 1000 / expected_sample_rate # Convert sampling rate to msec

    ##########################################################
    ### Select Data Input File / Barcodes Output Directory ###
    ##########################################################
    print("Reading signals")

    ##############################################
    ### Signals Data Extraction & Manipulation ###
    ##############################################
    try:
        signals_numpy_data = data# np.load(signals_file)
        signals_located = True
    except:
        signals_numpy_data = ''
        print("Signals .npy file not located; please check your filepath")
        signals_located = False

    # Check whether signals_numpy_data exists; if not, end script with sys.exit().
    if signals_located:
        #LJ = If data is in raw format and has not been sorted by "peaks"
        if raw_data_format:

            # Extract the signals_column from the raw data
            barcode_column = signals_numpy_data #[:, signals_column]
            barcode_array = barcode_column #.transpose()
            # Extract the indices of all events when TTL pulse changed value.
            event_index, _ = find_peaks(np.abs(np.diff(barcode_array)), height=height)
            # Convert the event_index to indexed_times to align with later code.
            indexed_times = event_index # Just take the index values of the raw data

            # fig = go.Figure()
            # fig.add_trace(go.Scatter(y=barcode_array, mode='lines'))
            # fig.add_trace(go.Scatter(x=indexed_times, y=[barcode_array[i] for i in indexed_times], mode='markers', marker=dict(size=4, color='red')))
            # fig.write_html('Barcodes_idx_test.html')
            # fig.show()
        # NP = Collect the pre-extracted indices from the signals_column.
        else:
            indexed_times = signals_numpy_data 
        
        # Find time difference between index values (ms), and extract barcode wrappers.
        events_time_diff = np.diff(indexed_times) * sample_conversion # convert to ms
        wrapper_array = indexed_times[np.where(
                        np.logical_and(min_wrap_duration < events_time_diff,
                                    events_time_diff < max_wrap_duration))[0]]
        

        # Isolate the wrapper_array to wrappers with ON values, to avoid any
        # "OFF wrappers" created by first binary value.
        false_wrapper_check = np.diff(wrapper_array) * sample_conversion # Convert to ms
        # Locate indices where two wrappers are next to each other.
        nexttoeachother = np.where(np.abs(np.diff(false_wrapper_check)) < ind_wrap_duration * global_tolerance)[0]
        false_wrapper_check[nexttoeachother+1] = np.nan
        false_wrappers = np.where(false_wrapper_check < max_wrap_duration)[0]
        
        # Delete the "second" wrapper (it's an OFF wrapper going into an ON bar)
        # wrapper_array = np.delete(wrapper_array, false_wrappers+1)
        wrapper_array = wrapper_array[false_wrappers]

        # Find the barcode "start" wrappers, set these to wrapper_start_times, then
        # save the "real" barcode start times to signals_barcode_start_times, which
        # will be combined with barcode values for the output .npy file.
        # wrapper_time_diff = np.diff(wrapper_array) * sample_conversion # convert to ms
        # barcode_index = np.where(wrapper_time_diff < wrap_duration)[0] # total_barcode_duration
        wrapper_start_times = wrapper_array  #[barcode_index]
        signals_barcode_start_times = wrapper_start_times #- ind_wrap_duration / sample_conversion
        # Actual barcode start is 10 ms before first 10 ms ON value.
        # Using the wrapper_start_times, collect the rest of the indexed_times events
        # into on_times and off_times for barcode value extraction.
        on_times = []
        off_times = []
        for idx, ts in enumerate(indexed_times):    # Go through indexed_times
            # Find where ts = first wrapper start time
            if ts == wrapper_start_times[0]:
                # All on_times include current ts and every second value after ts.
                on_times = indexed_times[idx::2]
                off_times = indexed_times[idx+1::2] # Everything else is off_times

        # Convert wrapper_start_times, on_times, and off_times to ms
        wrapper_start_times = wrapper_start_times * sample_conversion
        on_times = on_times * sample_conversion
        off_times = off_times * sample_conversion

        signals_barcodes = []
        for start_time in wrapper_start_times:
            oncode = on_times[
                np.where(
                    np.logical_and(on_times > start_time,
                                on_times < start_time + total_barcode_duration)
                )[0]
            ]
            offcode = off_times[
                np.where(
                    np.logical_and(off_times > start_time,
                                off_times < start_time + total_barcode_duration)
                )[0]
            ]

            # if len(signals_barcodes) > 0:
            #     if signals_barcodes[-1] >= 9795:
            #         print('Start debugging now...')

            #         indices = np.concatenate([np.array(oncode/sample_conversion).astype(int), np.array(offcode/sample_conversion).astype(int)])
            #         fig = go.Figure()
            #         fig.add_trace(go.Scatter(y=barcode_array, mode='lines'))
            #         fig.add_trace(go.Scatter(x=indices, y=[barcode_array[i] for i in indices], mode='markers', marker=dict(size=4, color='red')))
            #         fig.write_html(f'Barcodes_idx_test_{signals_barcodes[-1]+1}.html')
            #         fig.show()
            

            if len(offcode) > 0 and len(oncode) > 0:
                if offcode[0] < oncode[0]:
                    curr_time = offcode[0] + 2*ind_wrap_duration # Jumps ahead to start of barcode
                else:
                    # somehow messed up so here is dirty fix. so sad.
                    offcode, oncode = oncode.copy(), offcode.copy()
                    curr_time = offcode[0] + 2*ind_wrap_duration # Jumps ahead to start of barcode

                bits = np.zeros((nbits,))
                interbit_OFF = False # Changes to "True" during multiple OFF

                for bit in range(0, nbits):
                    next_on_idx = np.where(oncode >= (curr_time - ind_bar_duration * global_tolerance))[0]
                    next_off_idx = np.where(offcode >= (curr_time - ind_bar_duration * global_tolerance))[0]

                    # Handle missing values properly
                    if next_on_idx.size > 0:  # Don't include the ending wrapper
                        next_on = oncode[next_on_idx[0]]
                    else:
                        next_on = curr_time + ind_bar_duration  

                    if next_off_idx.size > 0:  # Don't include the ending wrapper
                        next_off = offcode[next_off_idx[0]]
                    else:
                        next_off = curr_time + ind_bar_duration 
                    # next_on = np.where(oncode >= (curr_time - ind_bar_duration * global_tolerance))[0]
                    # next_off = np.where(offcode >= (curr_time - ind_bar_duration * global_tolerance))[0]

                    # if next_on.size > 1:    # Don't include the ending wrapper
                    #     next_on = oncode[next_on[0]]
                    # else:
                    #     next_on = start_time + inter_barcode_interval

                    # if next_off.size > 1:    # Don't include the ending wrapper
                    #     next_off = offcode[next_off[0]]
                    # else:
                    #     next_off = start_time + inter_barcode_interval
                        

                    # Recalculate min/max bar duration around curr_time
                    min_bar_duration = curr_time - ind_bar_duration * global_tolerance
                    max_bar_duration = curr_time + ind_bar_duration * global_tolerance

                    if min_bar_duration <= next_on <= max_bar_duration:
                        bits[bit] = 1
                        interbit_OFF = False
                    elif min_bar_duration <= next_off <= max_bar_duration:
                        interbit_OFF = True
                    elif interbit_OFF == False:
                        bits[bit] = 1
                    # elif min_bar_duration <= next_off < next_on: #interbit_ON == True:
                    #     bits[bit] = 1
                    # elif next_on < next_off: #interbit_ON == True:
                    #     bits[bit] = 0

                    curr_time += ind_bar_duration

                barcode = sum(bits[bit] * pow(2, bit) for bit in range(nbits))
            
                signals_barcodes.append(barcode)    

            else:
                sys.exit("Something is wrong. The number of on and off times do not match.")

    else: # If signals_located = False
        sys.exit("Data not found. Program has stopped.")

    # ################################################################
    # ### Print out final output and save to chosen file format(s) ###
    # ################################################################

    # Create merged array with timestamps stacked above their barcode values
    signals_time_and_bars_array = np.vstack((signals_barcode_start_times,
                                            np.array(signals_barcodes)))
    actualbarcode_idx = np.where(np.logical_and(np.array(signals_barcodes) >= 1,
                                np.array(signals_barcodes) < 65535))[0]
    signals_time_and_bars_array = signals_time_and_bars_array[:,actualbarcode_idx].astype(np.int64)
    print("Final Ouput: ", signals_time_and_bars_array)

    # time_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # output_file = output_path / (output_name + '_barcode_' + time_now)
    output_file = output_path / (output_name + '_barcode')
    np.save(output_file, signals_time_and_bars_array)
    print('Barcode saved!!')
    print('Note that timestamps are actually just samples now, they will be converted to sec later on if needed. ')

    return None


if __name__ == '__main__':
    # defopt.run(extract_barcode)
    
    sess_data_path = '/media/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/TAA0000066/ses-012_date-20250318_protocol-t6/funcimg/Session'
    func_img_header_fname = '20250318_152251__TAA0000066_00001.h5'
    behav_data_header_fname = '20250214_M8_ai.bin'
    n_channels = 4

    extract_barcode(sess_data_path = sess_data_path, 
                func_img_header_fname = func_img_header_fname, 
                expected_sample_rate = 5000)

    # extract_barcode(sess_data_path = sess_data_path, 
    #             behav_data_header_fname = behav_data_header_fname, 
    #             expected_sample_rate = 9000, n_channels=n_channels)
    

    
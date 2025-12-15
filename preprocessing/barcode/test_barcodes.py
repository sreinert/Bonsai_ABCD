#%%
import numpy as np
import sys
from scipy.signal import find_peaks
from datetime import datetime
from pathlib import Path
import os
# import defopt
from barcode_util import read_h5, read_h5_with_key
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

#%%
sess_data_path = '/media/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/TAA0000066/ses-012_date-20250318_protocol-t6'
func_img_path = 'funcimg/Session'
func_img_header_fname = '20250318_152251__TAA0000066_00001.h5'
expected_sample_rate = 5000
behav_data_header_fname = ''
height = 2

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
    signals_file = sess_data_path / func_img_path /func_img_header_fname
    output_path = sess_data_path / func_img_path
    output_name = func_img_header_fname.split('.')[0]
    
    vdaq_barcode_file = sess_data_path / func_img_path / (func_img_header_fname.split('.')[0] + '_barcode.npy')

    if os.path.exists(vdaq_barcode_file):
        print('VDAQ barcodes detected. Loading...')
        vdaq_SyncTTL_ts = np.load(vdaq_barcode_file)

        weird_idx = np.where(np.diff(vdaq_SyncTTL_ts[1,:]) != 1)[0]
        print(weird_idx)    
        print(vdaq_SyncTTL_ts[1,weird_idx[0]])
        print(vdaq_SyncTTL_ts[1,weird_idx[0]+1])
        print(vdaq_SyncTTL_ts[1,weird_idx[0]+2])

    data = read_h5_with_key(signals_file, key='SyncTTL')
    data = np.array(data)

else:
    print('Please set either input')

#%%
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

#%%
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

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=barcode_array, mode='lines'))
        fig.add_trace(go.Scatter(x=indexed_times, y=[barcode_array[i] for i in indexed_times], mode='markers', marker=dict(size=4, color='red')))
        fig.add_trace(go.Scatter(x=[vdaq_SyncTTL_ts[0,weird_idx[0]]], y=[barcode_array[weird_idx[0]]], mode='markers', marker=dict(size=4, color='blue')))
        fig.add_trace(go.Scatter(x=[vdaq_SyncTTL_ts[0,weird_idx[0]+1]], y=[barcode_array[weird_idx[0]+1]], mode='markers', marker=dict(size=4, color='green')))
        fig.add_trace(go.Scatter(x=[vdaq_SyncTTL_ts[0,weird_idx[0]+2]], y=[barcode_array[weird_idx[0]+2]], mode='markers', marker=dict(size=4, color='goldenrod')))

        fig.write_html('Barcodes_idx_test.html')
        fig.show()
# %%

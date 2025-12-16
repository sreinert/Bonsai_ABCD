"""
  Optogenetics and Neural Engineering Core ONE Core
  University of Colorado, School of Medicine
  31.Oct.2021
  See bit.ly/onecore for more information, including a more detailed write up.
  alignment_barcodes.py
################################################################################
  This code takes two Numpy files ("main" and "secondary" data) that contain the 
  timestamp and barcode values collected using "extraction_barcodes.py", and 
  finds the linear conversion variables needed to align the timestamps. Then it
  takes the original (or .npy converted) secondary data file, applies the linear
  conversion to its timestamp column, and outputs this data as a Numpy (.npy) or
  CSV (.csv) file into a chosen directory.
################################################################################
  USER INPUTS EXPLAINED:

  Input Variables:
  = main_sample_rate = (int) The sampling rate (in Hz) of the "main" DAQ, to
                       which the secondary data will be aligned.
  = secondary_sample_rate = (int) The sampling rate (in Hz) of the "secondary"
                            DAQ, which will be aligned to the "main" DAQ.
  = convert_timestamp_column = (int) The timestamp column in the original 
                               "secondary" data file; this will be converted to
                               match the timestamps in the original "main" data.

  Output Variables: 
  = alignment_name = (str) The name of the file(s) in which the output will be 
                     saved in the chosen directory.
  = save_npy = (bool) Set to "True" to save the aligned data as a .npy file.
  = save_csv = (bool) Set to "True" to save the aligned data as a .csv file.
  
################################################################################
  References

"""
#######################
### Imports Section ###
#######################

import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from tkinter.filedialog import askdirectory, askopenfilename

################################################################################
############################ USER INPUT SECTION ################################
################################################################################

# Input variables
main_sample_rate = 30000 # Expected sample rate of main data, in Hz
secondary_sample_rate = 2000 # Expected sample rate of secondary data, in Hz
convert_timestamp_column = 0 # Column that timestamps are located in secondary data

# Output variables
alignment_name = 'LabJackAlignedToNeuroPixelTest' # Name of output file.
save_npy = False   # Save output file as a .npy file
save_csv = False    # Save output file as a .csv file

################################################################################
############################ END OF USER INPUT #################################
################################################################################

##########################################
### Select Files for Barcode Alignment ###
##########################################

# Have user select files to be used.
# Main data's barcodes input file:
try:
    main_dir_and_name = Path(askopenfilename(title = 'Select Main Data Barcodes File'))
except:
    print("No Main Data Barcodes File Chosen")
    sys.exit()
# Secondary data's barcodes input file
try:
    secondary_dir_and_name = Path(askopenfilename(title = 'Select Secondary Data Barcodes File'))
except:
    print("No Secondary Data Barcodes File Chosen")
    sys.exit()
# Secondary data file to be aligned with main data.
try:
    secondary_raw_data = Path(askopenfilename(title = 'Select Secondary Data File to Align'))
except:
    print("No Secondary Data File to Align Chosen")
    sys.exit()

# Try to load the selected files; if they fail, inform the user.
try:
    main_numpy_data = np.load(main_dir_and_name)
except:
    main_numpy_data = ''
    print("Main .npy file not located/failed to load; please check the filepath")

try:
    secondary_numpy_data = np.load(secondary_dir_and_name)
except:
    secondary_numpy_data = ''
    print("Secondary .npy file not located/failed to load; please check the filepath")

try:
    secondary_data_original = np.load(secondary_raw_data)
except:
    secondary_data_original = ''
    print("Data file to be aligned not located/failed to load; please check your filepath")

# Have user select folder into which aligned data will be saved; if no format
# selected, inform the user.
if save_npy or save_csv or save_dat:
    try:
        alignment_dir = Path(askdirectory(title = 'Select Folder to Save Aligned Data'))
    except:
        print("No Output Directory Selected")
        sys.exit()
else:
    print("Aligned data will not be saved to file in any format.")


##########################################################################
### Extract Barcodes and Index Values, then Calculate Linear Variables ###
##########################################################################

# Pull the barcode row from the data. 1st column is timestamps, second is barcodes
barcode_timestamps_row = 0 # Same for both main and secondary, because we used our own code
barcodes_row = 1 # Same for both main and secondary

main_numpy_barcode = main_numpy_data[barcodes_row, :]
secondary_numpy_barcode = secondary_numpy_data[barcodes_row, :]

main_numpy_timestamp = main_numpy_data[barcode_timestamps_row, :]
secondary_numpy_timestamp = secondary_numpy_data[barcode_timestamps_row, :]

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

secondary_data_original[:,convert_timestamp_column] = secondary_data_original[:,convert_timestamp_column] * secondary_sample_rate * m + b
secondary_data_converted = secondary_data_original # To show conversion complete.

# Clean up conversion of values to nearest whole number
#print("Total number of index values: ", len(secondary_data_converted[:,convert_timestamp_column]))
for index in range(0,len(secondary_data_converted[:,convert_timestamp_column])):
    value = secondary_data_converted[index, convert_timestamp_column]
    rounded_val = value.astype('int')
    secondary_data_converted[index, convert_timestamp_column] = rounded_val

################################################################
### Print out final output and save to chosen file format(s) ###
################################################################

# Test to see output here:
print("Final output for LJ Data:\n", secondary_data_converted)

time_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

if save_npy:
    output_file = alignment_dir / (alignment_name + time_now)
    np.save(output_file, secondary_data_converted)

if save_csv:
    output_file = alignment_dir / (alignment_name + time_now + '.csv')
    np.savetxt(output_file, secondary_data_converted, delimiter=',', fmt="%s")

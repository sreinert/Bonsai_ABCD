import os 
import numpy as np
from pathlib import Path
import importlib
import re
import sys

import roi_tracking_helpers
importlib.reload(roi_tracking_helpers)

def visualize_selected_tracked_clusters(roicat_dir, roicat_data_name, sessions_to_align, tracked_neuron_ids_path, session_keys, dir_save, filename):
    """
    Visualize on a slider the ROIs of selected neurons that have been tracked across sessions.
    The neuron ids need to have been saved in a .npz file beforehand.
    """
    # Load tracked neuron ids
    data = np.load(tracked_neuron_ids_path)
    tracked_neuron_ids = [np.array(data[key], dtype=int) for key in session_keys]

    # Visualize ROIs on a slider
    roi_tracking_helpers.roicat_visualize_tracked_rois(roicat_dir, roicat_data_name, sessions_to_align,
                                                    tracked_neuron_ids=tracked_neuron_ids, dir_save=dir_save, filename=filename)
    

if __name__ == "__main__":
    # Parse CLI args
    roicat_dir = sys.argv[1]
    roicat_data_name = sys.argv[2]
    sessions_to_align = sys.argv[3].split(",")  
    tracked_neuron_ids_path = sys.argv[4]
    session_keys = sys.argv[5].split(",")
    dir_save = sys.argv[6]
    filename = sys.argv[7]

    visualize_selected_tracked_clusters(
        roicat_dir, roicat_data_name, sessions_to_align,
        tracked_neuron_ids_path, session_keys, dir_save, filename
    )
    


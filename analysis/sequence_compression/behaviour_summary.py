import os, re, sys
import numpy as np
from pathlib import Path
from scipy.signal import resample
import scipy.stats as stats
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import importlib

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]   # adjust if needed
sys.path.append(str(PROJECT_ROOT))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))

import neural_analysis_helpers
import alternation_analysis_helpers as alternation 

importlib.reload(alternation)
importlib.reload(neural_analysis_helpers)

cohort = '3'

#%% 
def training_date_to_int(name):
    # Extract 6-digit date
    match = re.search(r"(\d{6})", name)
    if not match:
        return None
    # Convert YYMMDD -> YYYYMMDD assuming 20XX
    yymmdd = match.group(1)
    return int("20" + yymmdd)  # e.g., '250326' -> 20250326


def main():
    # Load functions according to cohort 
    if cohort == '2':
        import preprocessing.parse_session_functions_cohort2 as parse_session_functions
        import cellTV.cellTV_functions_cohort2 as cellTV
    elif cohort == '3':
        import preprocessing.parse_session_functions_cohort3 as parse_session_functions
        import cellTV.cellTV_functions_cohort2 as cellTV

    importlib.reload(parse_session_functions)
    importlib.reload(cellTV)

    alternation.set_parse_session_functions(parse_session_functions)

    # Define data paths
    if cohort == '2':
        base_path = Path("/ceph/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/")
        # base_path = Path('/Volumes/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/')
        mouse_ids = ["TAA0000066", "TAA0000059"]
        session_ids = ['t5', 't6']
    elif cohort == '3':
        PREPROCESSED_BEHAV = 'preprocessed_behav_Nov2025'
        base_path = Path(f"/ceph/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/{PREPROCESSED_BEHAV}/derivatives")
        # base_path = f"/Volumes/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/{PREPROCESSED_BEHAV}/derivatives" 
        mouse_ids = [f"sub-{mouse:03d}" for mouse in (4,6,7,8,10,11,12,13)]
        session_ids = {}
        for mouse in mouse_ids:
            if '4' in mouse:
                session_ids[mouse] = ['full020', 'full030']
            elif '6' in mouse:
                session_ids[mouse] = ['full011', 'full014']
            elif '7' in mouse:
                session_ids[mouse] = ['full010', 'full012']
            elif '8' in mouse:
                session_ids[mouse] = ['full011', 'full014']
            elif '10' in mouse:
                session_ids[mouse] = ['full010', 'full012']
            elif '11' in mouse:
                session_ids[mouse] = ['full009', 'full013']
            elif '12' in mouse:
                session_ids[mouse] = ['full011', 'full017']
            elif '13' in mouse:
                session_ids[mouse] = ['full010', 'full014']

    mouse_sessions = {mouse: [] for mouse in mouse_ids}
    for mouse in mouse_ids:
        if cohort == '2':
            mouse_path = base_path / mouse

            # Find funcimg dates 
            funcimg_dates = []
            for s in session_ids:
                session = [p for p in mouse_path.iterdir()
                    if p.is_dir() and f"{s}" in p.name and "ses" in p.name][0]
                mouse_sessions[mouse].append(session.name)

                match = re.search(r"date-(\d{8})", session.name)
                if match:
                    funcimg_dates.append(match.group(1))
                funcimg_dates = [int(d) for d in funcimg_dates] # convert to int

            # Find intermediate training sessions
            training_path = mouse_path / 'TrainingData'
            training_sessions = [p for p in training_path.iterdir() if p.is_dir()]
            
            training_dates = [
                f"TrainingData/{p.name}"  
                for p in training_sessions
                if (d := training_date_to_int(p.name)) is not None
                and min(funcimg_dates) <= d <= max(funcimg_dates)
            ]

            # Append training folders to the mouse_sessions list
            mouse_sessions[mouse].extend(training_dates)

        elif cohort == '3':
            mouse_path = base_path / mouse

            # Find funcimg dates 
            funcimg_dates = []
            for s in session_ids[mouse]:
                session = [p for p in mouse_path.iterdir()
                    if p.is_dir() and f"{s}" in p.name][0]
                mouse_sessions[mouse].append(session.name)
            print(mouse_sessions)

    # print(mouse_sessions)
if __name__ == "__main__":
    main()
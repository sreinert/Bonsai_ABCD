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

# def extract_date(session):
#     date = session.split('_')[1]
#     date1 = date[7:]
#     date2 = date[5:]
#     return date1, date2

def extract_date(name):
    """
    Extracts date from:
    - TrainingData/YYMMDD
    - ses-xxx_date-YYYYMMDD_protocol-...
    Returns date as int YYYYMMDD, or None if not found
    """
    # Case 1: imaging session (YYYYMMDD)
    match_8 = re.search(r"date-(\d{8})", name)
    if match_8:
        date_8 = match_8.group(1)
        date_6 = date_8[2:]   # YYYYMMDD -> YYMMDD
        return date_6, date_8

    # Case 2: training session (YYMMDD, possibly with suffix)
    match_6 = re.search(r"(\d{6})", name)
    if match_6:
        date_6 = match_6.group(1)
        date_8 = "20" + date_6
        return date_6, date_8

    return None, None

def find_session_index_by_substring(sessions, substring):
    for i, s in enumerate(sessions):
        if substring in s:
            return i
    return None

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
        base_path = Path(f"/ceph/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/rawdata/")
        # PREPROCESSED_BEHAV = 'preprocessed_behav_Nov2025'
        # base_path = Path(f"/ceph/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/{PREPROCESSED_BEHAV}/derivatives")
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

            all_dates = []
            session_names = []

            # Find funcimg dates 
            funcimg_dates = []
            for s in session_ids:
                session = [p for p in mouse_path.iterdir()
                    if p.is_dir() and f"{s}" in p.name and "ses" in p.name][0]
                session_names.append(session.name)

                match = re.search(r"date-(\d{8})", session.name)
                if match:
                    date_int = int(match.group(1))
                    funcimg_dates.append(date_int)
                    all_dates.append(date_int)
                    
            # Find intermediate training sessions
            training_path = mouse_path / 'TrainingData'
            training_sessions = [p for p in training_path.iterdir() if p.is_dir()]
            
            for p in training_sessions:
                d = training_date_to_int(p.name)
                if d is None:
                    continue
                if min(funcimg_dates) <= d <= max(funcimg_dates):
                    session_names.append(f"TrainingData/{p.name}")
                    all_dates.append(d)

            # Sort dates             
            mouse_sessions[mouse] = [session_names[i] for i in np.argsort(all_dates)]

        elif cohort == '3':
            mouse_path = base_path / mouse

            # Identify the first and last funcimg sessions
            all_sessions = sorted([p.name for p in mouse_path.iterdir() if p.is_dir()])

            start_idx = find_session_index_by_substring(all_sessions, session_ids[mouse][0])
            end_idx   = find_session_index_by_substring(all_sessions, session_ids[mouse][-1])

            # Find intermediate training sessions
            training_and_funcimg = all_sessions[start_idx:end_idx+1]  
            pattern = re.compile(r'full\d{3}')
            training_and_funcimg = [s for s in training_and_funcimg if pattern.search(s)]

            mouse_sessions[mouse] = training_and_funcimg

    # Analyse each session and collect data  
    lm_lick_rate = {mouse: {} for mouse in mouse_ids}
    for m, mouse in enumerate(mouse_ids):
        if m == 0:
        #     continue
            session_id = mouse_sessions[mouse][0]

        # for session_id in mouse_sessions[mouse]:
            if cohort == '2':
                if any(s in str(session_id) for s in session_ids):  # funcimg
                    date, _ = extract_date(session_id)
                    session_path = os.path.join(base_path, mouse, session_id, 'behav', date)
                else:  # training
                    date, _ = extract_date(session_id)
                    session_path = os.path.join(base_path, mouse, session_id)
                session = parse_session_functions.analyse_session_pre7_behav(session_path, mouse, date, 't5')

            elif cohort == '3':
                session_path = Path(os.path.join(base_path, mouse, session_id))
            # session = parse_session_functions.analyse_npz_pre7(mouse, date2, 't5')
                session = parse_session_functions.analyse_session_pre7_behav(session_path, mouse, 't5')
            
            # Collect lick rate date per session
            # lm_lick_rate[mouse][session_id] = session['lm_lick_rate']

if __name__ == "__main__":
    main()
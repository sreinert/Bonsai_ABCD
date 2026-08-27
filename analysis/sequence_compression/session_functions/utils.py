import pandas as pd
import numpy as np
import re, os, sys
import matplotlib.pyplot as plt
from pathlib import Path

np.set_printoptions(suppress=True, precision=2)

def threshold_lick_events(sess, sess_dataframe, below=True):

    licks = sess_dataframe['Licks'].values.astype(int)
    
    if below == True:
        speed_ok = sess['speed'] < sess['lick_threshold']
    else:
        speed_ok = sess['speed'] >= sess['lick_threshold']
    licked = licks > 0
    threshold_mask = speed_ok & licked

    thresholded_licks = np.zeros(len(licks))
    thresholded_licks[threshold_mask] = licks[threshold_mask]

    return thresholded_licks

def extract_int(s: str) -> int:
    m = re.search(r'\d+', s)
    if m:
        return int(m.group())
    else:
        raise ValueError(f"No digits found in string: {s!r}")

def extract_session_id(base_path: Path):
    name = base_path.name  # e.g. "ses-full003_date-20260212T161727"
    session_id = name.split("ses-")[1].split("_date-")[0]
    return session_id

def find_closest_events(
    df: pd.DataFrame,
    idx: int,
    pos_window: float = 3.0,
    event_priority=["release", "prepare", "flush"],
    choose = 'earliest',
    verbose = False,
):
    """
    For each idx, find the nearest event based on Position.

    For each event type in priority:
        - Search in a zigzag pattern around the idx:
          row, row-1, row+1, row-2, row+2, ...
        - At each candidate row j:
            * Require |Position(j) - pos0| <= pos_window
            * Skip odour 0
        - Stop searching in a direction once Position falls outside pos_window
        - Each candidate row is saved into a list candidate_idx
    
    Final steps choose representative idx from candidate_idx based on: choose
    """
    events_col = df["Events"].astype("string")
    n_rows = len(df)
    pos0 = df.at[idx, "Position"]

    candidate_idx = []
    chosen_idx = None
    chosen_event = None
    odour = None
    chosen_pos = None

    for ev_type in event_priority:
        # ---------- Zigzag search around idx ----------
        offset = 0
        up_active = True
        down_active = True

        while up_active or down_active:
            # Check current / upward direction: idx - offset
            if up_active:
                j_up = idx - offset
                if j_up < 0:
                    up_active = False
                else:
                    if abs(df.at[j_up, "Position"] - pos0) > pos_window:
                        # we assume Position is monotonic, so further up is outside window
                        up_active = False
                    else:
                        ev = events_col.iat[j_up]
                        if ev is not None and not pd.isna(ev) and "odour0" not in ev:
                            if ev_type in ev:
                                candidate_idx.append(j_up)

            # Check downward direction only for offset > 0 to avoid double-checking idx
            if offset > 0 and down_active:
                j_down = idx + offset
                if j_down >= n_rows:
                    down_active = False
                else:
                    if abs(df.at[j_down, "Position"] - pos0) > pos_window:
                        # we assume Position is monotonic, so further down is outside window
                        down_active = False
                    else:
                        ev = events_col.iat[j_down]
                        if ev is not None and not pd.isna(ev) and "odour0" not in ev:
                            if ev_type in ev:
                                candidate_idx.append(j_down)

            offset += 1  # expand zigzag radius

    if len(candidate_idx) > 0:
        if choose == 'earliest':
            chosen_idx = min(candidate_idx)
            chosen_event = events_col.iat[chosen_idx]
            chosen_pos = df.at[chosen_idx, "Position"]
            odour = extract_int(chosen_event)
        elif choose == 'average':
            center_idx = candidate_idx[0]
            chosen_idx = int(np.average(candidate_idx))

            chosen_pos = df.at[chosen_idx, "Position"]
            odour = extract_int(events_col.iat[center_idx])
            chosen_event = 'Estimated release: odour' + str(odour)
        else:
            raise NotImplementedError

        if verbose:
            if chosen_event is None:
                raise ValueError(
                    f"No event of types {event_priority} found within ±{pos_window} "
                    f"for expected release event around idx: {idx}."
                )

    return chosen_idx, chosen_event, odour, chosen_pos

def get_response_end(target_licks):
    '''Find the index (if any) where the mouse stopped licking in the task'''

    # Find misses
    zero_idx = np.where(target_licks == 0)[0]
    if len(zero_idx) == 0:
        return None

    # Find breaks between consecutive indices and split into blocks
    breaks = np.where(np.diff(zero_idx) != 1)[0] + 1
    blocks = np.split(zero_idx, breaks)

    # Keep only blocks with length >= 5
    long_blocks = [b for b in blocks if len(b) >= 5]
    if len(long_blocks) > 0:
        cutoff_event = long_blocks[-1][0]
        # Check if any lick happens after this block
        if not np.any(target_licks[long_blocks[-1][-1] + 1:] == 1):
            print(f'Removing from trial {cutoff_event} until the end ({len(target_licks)}), because of likely satiety (no licking)')
            return cutoff_event
    
    return None

def get_disengagement_periods(A_A_dt, A_B_dt, plot=True):
    '''Find the trials (if any) where the mouse stopped engaging (no running) in the task'''

    # Find trial indices where reaching the next landmark took too long
    thresholds = []

    # AA
    thresholds.append(np.median(A_A_dt) + 0.5 * np.std(A_A_dt))
    ix_AA = np.where(A_A_dt > thresholds[0])[0]
    # print('A-A indices:', ix_AA)
    # print('A-A values', A_A_dt[ix_AA])

    # AB
    ix_AB = []
    for i in range(A_B_dt.shape[1]):
        col = A_B_dt[:, i]
        thresholds.append(np.median(col) + 0.5 * np.std(col))
        ix_AB.append(np.where(col > thresholds[i+1])[0])        
        # print(f'A-B{i+1} indices:', ix_AB[i])
        # print(f'A-B{i+1} values:', col[ix_AB[i]])

    # Find common indices
    # shared_ix = ix_AA.copy()
    # for ix in ix_AB:
    #     shared_ix = np.intersect1d(shared_ix, ix)
    # print("Shared disengagement indices:", shared_ix)
      
    if plot:
        _, ax = plt.subplots(1, A_B_dt.shape[1] + 1, figsize=(10,2))
        ax = ax.ravel()

        for i in range(A_B_dt.shape[1]):
            ax[i].hist(A_B_dt[:,i], bins=20)
            ax[i].set_title(f'A -> B{i+1}')
            ax[i].axvline(thresholds[i+1], linestyle='--', color='grey')
        for a in ax:
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)

        ax[i+1].hist(A_A_dt, bins=20)
        ax[i+1].set_title('A -> A')
        ax[i+1].axvline(thresholds[0], linestyle='--', color='grey')
        
    return ix_AA, ix_AB
      
def compute_binned_lick_rate(distances, licks, bins):
    
    bin_idx = np.digitize(distances, bins)
    
    means = []
    sems = []
    centers = []

    for b in range(1, len(bins)):
        mask = bin_idx == b
        
        if np.sum(mask) > 0:
            vals = licks[mask].astype(float)
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
        else:
            means.append(np.nan)
            sems.append(np.nan)
        
        centers.append(np.round((bins[b] + bins[b-1]) / 2, 2))

    return np.array(centers), np.array(means), np.array(sems)

def format_condition_label(cond=None, pattern=None):
    
    if cond is not None:
        # Determine the AB counts
        if 'abb' in cond and 'abbb' not in cond and 'aabb' not in cond:
            base = r"$\mathrm{AB}^2$"
        elif 'abbb' in cond:
            base = r"$\mathrm{AB}^3$"
        elif 'aabb' in cond:
            base = r"$\mathrm{A}^2\mathrm{B}^2$"
        else:
            base = cond
        
        # Preserve random/fixed part
        if 'random' in cond:
            return f"{base} random"
        elif 'fixed' in cond:
            return f"{base} fixed"
        else:
            return base
    
    elif pattern is not None: 
        pattern = np.asarray(pattern)

        symbol_map = {1: "A", 0: "B"}

        parts = []

        current = pattern[0]
        count = 1

        for x in pattern[1:]:
            if x == current:
                count += 1
            else:
                letter = symbol_map[current]
                parts.append(f"{letter}{count}")
                current = x
                count = 1

        # flush last run
        letter = symbol_map[current]
        parts.append(f"{letter}{count}")

        return "".join(parts)

def safe_divide(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # result has same shape as a
    out = np.full_like(a, np.nan, dtype=float)

    # numpy handles broadcasting for where= and division
    np.divide(a, b, out=out, where=(b != 0))

    return out
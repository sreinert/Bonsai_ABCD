"""Read-only comparison of processed (frame-aligned npz) vs raw (Bonsai) licked-landmark calls.

musicboxes/behaviour_performance_summary.py shows the two side by side and they visibly disagree on
a fair number of landmark visits. This script pins down WHY, visit by visit, before anything in the
classification is changed.

Both sides use the same visit list (estimate_lm_events/estimate_release_events on the raw pickle) and
the same landmark windows [release_pos - LM_offset, + lm_size]:
 - raw (calc_hit_fa): any speed-thresholded Bonsai 'Licks' event at or after the visit's release
   event, with its exact logged position strictly inside the window;
 - processed (get_licks_rewards_npz): any frame inside the visit's contiguous entry..exit frame range,
   whose frame-mean position is inside the window, with an analog lick edge and frame-mean speed
   below velocityThreshold.

Every disagreement gets the first matching cause, checked in this order:
 processed licked, raw didn't:
   'before_release'  raw has a lick in the window, but before the release event (raw ignores it)
   'speed_gate'      raw has a lick in the window that its instantaneous-speed threshold dropped
   'edge_position'   raw has a lick within EDGE_CM of the window (frame-mean position pulled it in)
   'lick_source'     no Bonsai lick near this visit at all -- analog lick channel only
 raw licked, processed didn't:
   'outside_imaging' the raw lick falls outside the imaged frames
   'speed_gate'      the processed frame has an analog lick but failed the frame-mean speed gate
   'edge_position'   the raw lick's frame has frame-mean position outside the window / visit range
   'lick_source'     no analog lick on (or next to) the raw lick's frame

Raw lick times are put on the frame axis by a linear fit of raw reward times -> npz reward frames
(the two have equal counts, checked by verify_behaviour_data), which is exact to within a frame.

Usage (bonsai_abcd env, from repo root):
    python -m preprocessing.quality_control.lick_classification_diagnostics <behav_root> <mouse> <session_id> [...] \\
        --out ~/Desktop/diagnostics/lick_classification
    e.g. ... .../cohort1/BehaviourAligned_robust 004 2LM004 004 2LM003
"""

import argparse
import contextlib
import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import era_brewer
import parse_nwb_functions
import preprocessing.parse_session_functions_cohort3_summary as pf

EDGE_CM = 3
CAUSES = ['before_release', 'speed_gate', 'edge_position', 'lick_source', 'outside_imaging']
CAUSE_COLORS = dict(zip(CAUSES, era_brewer.era_brew('Lover1', n=5, brew_type='discrete')))


def _raw_frame_mapping(sess_dataframe, session):
    reward_times = (sess_dataframe.index[sess_dataframe['Rewards'].notna()]
                    - pd.Timestamp('1904-01-01')).total_seconds().values
    reward_frames = session['rewards']
    if len(reward_times) != len(reward_frames) or len(reward_frames) < 2:
        raise ValueError(f"{len(reward_times)} raw rewards vs {len(reward_frames)} npz reward frames -- can't map.")
    slope, intercept = np.polyfit(reward_times, reward_frames, 1)
    residual = reward_frames - (slope * reward_times + intercept)
    print(f"  raw time -> frame fit: {1 / slope:.4f} s/frame^-1, max residual {np.abs(residual).max():.2f} frames")
    return lambda t: np.round(slope * t + intercept).astype(int)


def classify_session(behav_root, mouse, session_id, world='stable'):
    with contextlib.redirect_stdout(io.StringIO()):
        session = pf.analyse_npz_sequences(mouse, session_id, behav_root, stage=session_id, world=world)
        session_path = parse_nwb_functions.find_base_path(mouse, session_id, behav_root)
    ses_settings, _ = parse_nwb_functions.load_settings(session_path)
    with open(list((Path(session_path) / 'behav').glob('*.pkl'))[0], 'rb') as f:
        raw_all = pickle.load(f)
    raw_df = pf.threshold_lick_speed(raw_all.copy(), speed_threshold=ses_settings['velocityThreshold'])
    _, _, _, _, _, raw_licked, _ = pf.calc_hit_fa(raw_df, ses_settings)

    to_frame = _raw_frame_mapping(raw_df, session)
    t0 = pd.Timestamp('1904-01-01')
    all_lick_mask = raw_all['Licks'].values > 0
    all_lick_t = (raw_all.index[all_lick_mask] - t0).total_seconds().values
    all_lick_pos = raw_all['Position'].values[all_lick_mask]
    kept = raw_df['Licks'].values[all_lick_mask] > 0          # passed the instantaneous speed gate
    all_lick_frames = to_frame(all_lick_t)

    _, _, _, _, release_df = pf.get_event_parsed(raw_df, ses_settings)
    release_t = (release_df.index - t0).total_seconds().values  # NaT (prepended LM 0) -> NaN
    n_frames = len(session['position'])
    analog_licks = session['licks'] > 0
    proc_lick_frames = set(np.where(session['thresholded_licks'] > 0)[0])

    proc_licked = session['licked_lms'].astype(int)
    n = min(len(proc_licked), len(raw_licked))
    rows = []
    for i in range(n):
        p, r = proc_licked[i], int(raw_licked[i])
        cause = ''
        lm_start, lm_end = session['landmarks'][i]
        in_win = (all_lick_pos > lm_start) & (all_lick_pos < lm_end)
        after = np.ones_like(in_win) if np.isnan(release_t[i]) else all_lick_t >= release_t[i]
        visit_frames = np.where(session['data_lm_idx'] == i + 1)[0]
        if p and not r:
            # restrict to licks around this visit's frames, so other laps' licks at the same position don't count
            near_visit = (np.isin(all_lick_frames, visit_frames) if len(visit_frames)
                          else np.zeros_like(in_win))
            if np.any(in_win & ~after & kept):
                cause = 'before_release'
            elif np.any(in_win & after & ~kept):
                cause = 'speed_gate'
            elif np.any(near_visit & (all_lick_pos > lm_start - EDGE_CM) & (all_lick_pos < lm_end + EDGE_CM)):
                cause = 'edge_position'
            else:
                cause = 'lick_source'
        elif r and not p:
            frames = all_lick_frames[in_win & after & kept]
            if np.all((frames < 0) | (frames >= n_frames)):
                cause = 'outside_imaging'
            else:
                frames = frames[(frames >= 0) & (frames < n_frames)]
                nb = np.unique(np.concatenate([frames - 1, frames, frames + 1]).clip(0, n_frames - 1))
                if not np.any(analog_licks[nb]):
                    cause = 'lick_source'
                elif not any(f in proc_lick_frames for f in nb):
                    cause = 'speed_gate'
                else:
                    cause = 'edge_position'
        rows.append(dict(visit=i, processed=p, raw=r, cause=cause))
    df = pd.DataFrame(rows)
    df.attrs.update(mouse=mouse, session_id=session_id,
                    n_raw_licks=int(raw_all['Licks'].sum()), n_analog_licks=int(np.nansum(session['licks'])))
    return df


def plot_session(df, out_dir):
    label = f"sub-{df.attrs['mouse']}_{df.attrs['session_id']}"
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), gridspec_kw=dict(height_ratios=[1.3, 1]))
    ax = axes[0]
    ax.imshow(np.vstack([df['raw'], df['processed']]), aspect='auto', cmap='Greys', interpolation='none')
    for cause, color in CAUSE_COLORS.items():
        v = df.loc[df['cause'] == cause, 'visit']
        ax.scatter(v, np.full(len(v), 0.5), color=color, s=18, marker='v', label=cause, zorder=3)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['raw', 'processed'])
    ax.set_xlabel('Landmark visit'); ax.legend(fontsize=7, ncol=5, loc='upper right')
    n_dis = (df['cause'] != '').sum()
    ax.set_title(f"{label}: {n_dis}/{len(df)} visits disagree "
                 f"(Bonsai licks {df.attrs['n_raw_licks']}, analog licks {df.attrs['n_analog_licks']})", fontsize=10)

    ax = axes[1]
    for k, (direction, sel) in enumerate([('processed only', (df['processed'] == 1) & (df['raw'] == 0)),
                                          ('raw only', (df['processed'] == 0) & (df['raw'] == 1))]):
        counts = df.loc[sel, 'cause'].value_counts()
        left = 0
        for cause in CAUSES:
            c = counts.get(cause, 0)
            if c:
                ax.barh(k, c, left=left, color=CAUSE_COLORS[cause])
                ax.text(left + c / 2, k, str(c), ha='center', va='center', fontsize=8)
                left += c
    ax.set_yticks([0, 1]); ax.set_yticklabels(['processed only', 'raw only'])
    ax.set_xlabel('Disagreeing visits')
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f'{label}_lick_classification.png', dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('behav_root')
    parser.add_argument('mouse_session', nargs='+', help='pairs: <mouse> <session_id>, e.g. 004 2LM004')
    parser.add_argument('--world', default='stable')
    parser.add_argument('--out', default=str(Path.home() / 'Desktop' / 'diagnostics' / 'lick_classification'))
    args = parser.parse_args()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(zip(args.mouse_session[0::2], args.mouse_session[1::2]))
    summary = []
    for mouse, session_id in pairs:
        print(f"sub-{mouse} {session_id}")
        df = classify_session(args.behav_root, mouse, session_id, world=args.world)
        plot_session(df, out_dir)
        df.to_csv(out_dir / f"sub-{mouse}_{session_id}_visits.csv", index=False)
        counts = df['cause'].value_counts().drop('', errors='ignore').to_dict()
        print(f"  {(df['cause'] != '').sum()}/{len(df)} visits disagree: {counts}")
        summary.append(dict(mouse=mouse, session_id=session_id, n_visits=len(df),
                            n_disagree=int((df['cause'] != '').sum()), **counts))
    pd.DataFrame(summary).fillna(0).to_csv(out_dir / 'summary.csv', index=False)


if __name__ == '__main__':
    main()

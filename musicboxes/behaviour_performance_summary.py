"""Behavioural-performance summary plots for a single cohort3/GoalProgress-style session.

Builds two parallel combined figures for the same session so they can be compared side by
side: one from the processed, frame-aligned behaviour struct (npz pipeline, same 'session'
struct musicboxes/session_overview.py uses -- via parse_session_functions_cohort3_summary.
analyse_npz_sequences), and one from the raw, real-timestamp-indexed Bonsai sess_dataframe
(the same pickle analyse_npz_sequences itself loads internally, before frame-alignment). Each
figure bundles all of that run's panels as subplots in one grid, saved as a single PNG.

Session type (sequence vs shaping) is auto-detected from the number of rewarded landmarks
(len(session['goal_ids'])), same convention as session_overview.py:
 - sequence (>1 rewarded landmark, cyclic A->B->C->D...): 1) binary licked-landmark x lap map,
   2) unconditional landmark-to-landmark transition matrix, 3) conditional transition matrix
   per goal + ideal-agent matrix, 4) binary lick map split by cycle-state, 5) sliding-window
   per-goal state-discrimination ratio, 6) correct-transition rate per sequence step.
 - shaping (1 rewarded landmark): 1) binary licked-landmark vector (+ rewarded vector), 2)
   peri-reward lick trace pooled across rewards, 3) same, in chronological blocks of rewards,
   4) average speed around reward (mean +/- SEM).

This is a copy of parse_session_functions_cohort3.py (as
preprocessing/parse_session_functions_cohort3_summary.py) with a handful of new functions
added -- see that file's own added-functions section -- rather than edits to the original,
per instruction.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from pynwb import NWBHDF5IO

import parse_nwb_functions
import era_brewer
import preprocessing.parse_session_functions_cohort3_summary as pf
import preprocessing.parse_session_functions_cohort2 as pf2


# ---------------------------------------------------------------------------
# Colors -- era_brewer substitutes chosen to look similar to the plain matplotlib
# colors/colormaps they replace (not imported from session_overview.py to avoid a circular
# import, since that module imports this one -- MATRIX_CMAP duplicates its HEATMAP_CMAP
# formula so the two stay visually consistent).
# ---------------------------------------------------------------------------

MATRIX_CMAP = LinearSegmentedColormap.from_list(  # ~ 'viridis'
    'NineteenEightyNine_r', era_brewer.era_brew('NineteenEightyNine', n=256, brew_type='continuous')[::-1])
LICK_CMAP = LinearSegmentedColormap.from_list(  # ~ 'Reds' -- light dusty pink to near-black maroon
    'era_lick', list(reversed(era_brewer.ERA_PALETTES['Red1']['colors'])))
REWARD_CMAP = LinearSegmentedColormap.from_list(  # ~ 'Greens' -- cream to forest green
    'era_reward', era_brewer.ERA_PALETTES['Showgirl2']['colors'][3:])
BLOCK_CMAP = LinearSegmentedColormap.from_list(  # ~ plt.cm.viridis, for block-of-N gradients
    'era_block', era_brewer.ERA_PALETTES['Fearless']['colors'])
SPEED_COLOR = '#4478ac'  # ~ 'steelblue', era_brewer Lover1/Lover2's blue
AVERAGE_COLOR = '#2B2B2B'  # ~ 'k', era_brewer Reputation's near-black
BAR_COLOR = SPEED_COLOR
GOAL_COLORS = era_brewer.era_brew('Lover1', n=5, brew_type='discrete')  # per-goal A/B/C/D...
# Lover1 (pink/blue/cream/brown) over SpeakNow1: SpeakNow1 is one hue family shading light to
# near-black, which was hard to tell apart and clashed with AVERAGE_COLOR; Lover1's colors are
# distinct hues instead of just different lightnesses of the same one, and none are near-black.


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_processed_session(mouse, session_id, stage, behav_root, world='stable'):
    """Frame-aligned/npz session struct -- behaviour-only, no dF/F or suite2p needed."""
    return pf.analyse_npz_sequences(mouse, session_id, behav_root, stage, world=world)


def load_raw_session(mouse, session_id, behav_root, apply_speed_threshold=True):
    """Raw, real-timestamp-indexed sess_dataframe + ses_settings -- the same pickle
    analyse_npz_sequences loads internally, before any frame-alignment happens.

    :param apply_speed_threshold: if True (default), filter sess_dataframe['Licks'] to only
        licks made below ses_settings['velocityThreshold'] (see
        parse_session_functions_cohort3_summary.threshold_lick_speed), matching the
        speed-gating the processed pipeline already applies (get_licks_idx/threshold_licks) --
        both now read the same raw Treadmill channel (see build_behaviour_data.py's speed
        fix), so this makes the two pipelines' lick counts directly comparable instead of raw
        counting every lick regardless of running speed.
    """
    session_path = Path(parse_nwb_functions.find_base_path(mouse, session_id, behav_root))
    ses_settings, _ = parse_nwb_functions.load_settings(session_path)
    with open(list((session_path / 'behav').glob('*.pkl'))[0], 'rb') as f:
        sess_dataframe = pickle.load(f)
    if apply_speed_threshold:
        sess_dataframe = pf.threshold_lick_speed(sess_dataframe, speed_threshold=ses_settings['velocityThreshold'])
    return sess_dataframe, ses_settings


def is_sequence_session(session):
    return len(session['goal_ids']) > 1


def _get_frame_rate(session_path):
    """Empirical funcimg frame rate, same source as session_overview.load_session_data --
    needed to window session['licks']/session['speed'] (frame-indexed) around rewards."""
    nwb_path = list(Path(session_path).glob('*.nwb'))[0]
    with NWBHDF5IO(str(nwb_path), mode='r') as io:
        nwb = io.read()
        funcimg_timestamps = nwb.acquisition['vDAQ_TTLsTable'][:]['timestamp'].values
    return 1 / np.median(np.diff(funcimg_timestamps))


def default_save_dir(root_path, mouse, session_id):
    """root_path/analysis/sub-mouse/session -- shared default save location for behavioural
    (and, via session_overview.run_overview, neural) summary plots for one session. root_path
    is the project's own root folder (e.g. .../SuKuSaRe_20250923_HFScohort3), not behav_root --
    for cohort3-style dual-tree projects behav_root is already a deep subpath
    (.../preprocessed_behav_Nov2025/derivatives) that isn't itself the right place for a
    shared analysis/ folder."""
    return Path(root_path) / 'analysis' / f'sub-{mouse}' / session_id


def _finish_figure(fig, save_path, show, suptitle=None):
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def _plot_matrix(ax, mat, title, xlabel='Next landmark ID', ylabel='Goal',
                  y_labels=None, vmin=0, vmax=None, cmap=MATRIX_CMAP):
    im = ax.imshow(mat, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
    return im


def _reshape_by_landmark(arr, n_landmarks, n_rows=None):
    if arr.shape[0] % n_landmarks != 0:
        arr = np.pad(arr, (0, n_landmarks - (arr.shape[0] % n_landmarks)), 'constant')
    reshaped = arr.reshape(-1, n_landmarks)
    return reshaped[:n_rows] if n_rows is not None else reshaped


def _placeholder_axis(fig, gs_slice, message):
    """Fills a GridSpec cell with a centered text note instead of a real plot, for a panel
    that isn't applicable to this session (see STATE_LAPS_NEEDED below)."""
    ax = fig.add_subplot(gs_slice)
    ax.text(0.5, 0.5, message, ha='center', va='center', wrap=True, fontsize=9)
    ax.axis('off')
    return ax


# give_state_id/give_state_id_npz only distinguish a laps_needed-cycle of 2 or 3 states
# (the mouse's reward pattern at two or three "defining" landmarks) -- laps_needed==1 means
# the goal identities are laid out in increasing position order within a single lap, so
# there's no multi-lap state cycle to differentiate at all (every lap is the same "state").
# Plots 5/6 below are meaningless in that case -- see plot_sequence_processed/plot_sequence_raw.
STATE_LAPS_NEEDED = (2, 3)


# ---------------------------------------------------------------------------
# Sequence sessions
# ---------------------------------------------------------------------------

def plot_sequence_processed(session, save_dir=None, show=True,
                            suptitle='Sequence session -- processed (frame-aligned)',
                            filename='processed_summary.png'):
    n_landmarks = session['num_landmarks']
    goals = session['goal_ids']
    n_goals = len(goals)
    goal_labels = [chr(65 + i) for i in range(n_goals)]
    goal_colors = GOAL_COLORS[:n_goals]

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 6, figure=fig)
    print(f"Building sequence behavioural summary ({filename})...")

    # 1) binary lick/reward map (licked landmark x lap)
    licked_reshaped = _reshape_by_landmark(session['licked_lms'], n_landmarks)
    rewarded_reshaped = _reshape_by_landmark(session['rewarded_lms'], n_landmarks)
    ax = fig.add_subplot(gs[0, 0:2])
    sns.heatmap(licked_reshaped, cmap=LICK_CMAP, ax=ax, cbar_kws={'label': 'Licked'})
    ax.set_xlabel('Landmark ID'); ax.set_ylabel('Lap'); ax.set_title('1) Binary lick map')
    ax = fig.add_subplot(gs[0, 2:4])
    sns.heatmap(rewarded_reshaped, cmap=REWARD_CMAP, ax=ax, cbar_kws={'label': 'Rewarded'})
    ax.set_xlabel('Landmark ID'); ax.set_ylabel('Lap'); ax.set_title('1) Binary reward map')

    # 2) unconditional landmark-to-landmark transition matrix
    tm = pf.get_all_transitions_npz(session)
    ax = fig.add_subplot(gs[0, 4:6])
    _plot_matrix(ax, tm, '2) Landmark transition counts',
                 xlabel='Next landmark ID', ylabel='Current landmark ID')

    # 3) conditional transition matrix per goal + ideal-agent matrix
    transition_prob = session['transition_prob']
    transition_prob = transition_prob / np.sum(transition_prob, axis=1, keepdims=True)
    control_prob = session['control_prob']
    control_prob = control_prob / np.sum(control_prob, axis=1, keepdims=True)
    ideal_prob = session['ideal_prob']
    max_val = max(np.max(transition_prob), np.max(control_prob), np.max(ideal_prob))
    ax = fig.add_subplot(gs[1, 0:2])
    _plot_matrix(ax, transition_prob, '3) Transition prob. (licked)', y_labels=goal_labels, vmax=max_val)
    ax = fig.add_subplot(gs[1, 2:4])
    _plot_matrix(ax, control_prob, '3) Control prob. (all targets)', y_labels=goal_labels, vmax=max_val)
    ax = fig.add_subplot(gs[1, 4:6])
    _plot_matrix(ax, ideal_prob, '4) Ideal-agent prob.', y_labels=goal_labels, vmax=max_val)

    # 5/6) state-dependent panels -- only meaningful when give_state_id_npz actually
    # distinguishes states. Two independent ways that can fail to happen:
    #  - laps_needed not in {2, 3} (see STATE_LAPS_NEEDED): with laps_needed==1 every lap is
    #    the same "state" (goal identities fit within one lap), so there's structurally
    #    nothing to split by;
    #  - laps_needed IS 2 or 3, but state_id still never left 0 -- seen for a session (mouse
    #    010/full065) where only the LAST goal in the 4-goal chain ever carried an individual
    #    'Rewarded' flag in the raw data (the other 3 had zero, vs. ~11-12 each in a normal
    #    session), so the intermediate defining-goal reward give_state_id_npz's state machine
    #    watches for never fires. That looks like a genuine reward-logging anomaly for that
    #    session, not a bug in this analysis code, so it's reported rather than worked around.
    laps_needed = session['laps_needed']
    n_states_observed = len(np.unique(session['state_id']))
    if laps_needed in STATE_LAPS_NEEDED and n_states_observed > 1:
        # 5) binary lick map split by cycle-state
        by_state = pf.get_binary_licks_by_state_npz(session)
        n_states = len(by_state)
        span = 6 // n_states
        for s, mat in by_state.items():
            ax = fig.add_subplot(gs[2, s * span:(s + 1) * span])
            sns.heatmap(mat, cmap=LICK_CMAP, ax=ax, cbar_kws={'label': 'Licked'})
            ax.set_xlabel('Landmark ID'); ax.set_ylabel('Lap (within state)')
            ax.set_title(f'5) Binary licks, state {s}')

        # 6) sliding-window state-discrimination ratio per goal
        sw_ratio, sw_per_goal = pf.calc_sw_state_ratio_npz(session)
        ax = fig.add_subplot(gs[3, 0:4])
        ax.plot(sw_ratio, color=AVERAGE_COLOR, label='Average', linewidth=3)
        for color, label, curve in zip(goal_colors, goal_labels, sw_per_goal):
            ax.plot(curve, label=label, color=color, linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Lap'); ax.set_ylabel('State-discrimination ratio')
        ax.set_title('6) Sliding-window (10-lap) state-discrimination ratio')
        ax.legend(fontsize=8)
    elif laps_needed not in STATE_LAPS_NEEDED:
        print(f"  laps_needed={laps_needed}: no state cycle to differentiate, skipping panels 5/6")
        _placeholder_axis(fig, gs[2, 0:6],
                           f"5) Binary licks by state: not applicable (laps_needed={laps_needed})")
        _placeholder_axis(fig, gs[3, 0:4],
                           f"6) State-discrimination ratio: not applicable (laps_needed={laps_needed})")
    else:
        print(f"  laps_needed={laps_needed} but state_id never left 0 (likely one or more "
              "defining goals never individually rewarded in the raw data) -- skipping panels 5/6")
        _placeholder_axis(fig, gs[2, 0:6],
                           "5) Binary licks by state: skipped -- state_id stayed at 0 all "
                           "session (one or more defining goals never individually rewarded)")
        _placeholder_axis(fig, gs[3, 0:4],
                           "6) State-discrimination ratio: skipped -- state_id stayed at 0 "
                           "all session")

    # 7) correct-transition rate per sequence step
    performance, perf_list = pf.calc_stable_seq_fraction_npz(session)
    step_labels = [f'{chr(65 + i)}->{chr(65 + (i + 1) % n_goals)}' for i in range(n_goals)]
    ax = fig.add_subplot(gs[3, 4:6])
    ax.bar(step_labels, perf_list, color=goal_colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction correct')
    ax.set_title(f'7) Correct-transition rate (mean={performance:.2f})')

    save_path = Path(save_dir) / filename if save_dir is not None else None
    _finish_figure(fig, save_path, show, suptitle=suptitle)


def plot_sequence_raw(sess_dataframe, ses_settings, save_dir=None, show=True):
    goals, lm_ids = pf.parse_stable_goal_ids(ses_settings)
    n_landmarks = lm_ids.shape[0]
    n_goals = len(goals)
    goal_labels = [chr(65 + i) for i in range(n_goals)]
    goal_colors = GOAL_COLORS[:n_goals]

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 6, figure=fig)
    print("Building sequence behavioural summary (raw)...")

    # 1) binary lick/reward map (licked landmark x lap)
    _, _, _, _, _, licked_all, rewarded_all = pf.calc_hit_fa(sess_dataframe, ses_settings)
    ax = fig.add_subplot(gs[0, 0:2])
    sns.heatmap(_reshape_by_landmark(licked_all, n_landmarks), cmap=LICK_CMAP, ax=ax, cbar_kws={'label': 'Licked'})
    ax.set_xlabel('Landmark ID'); ax.set_ylabel('Lap'); ax.set_title('1) Binary lick map')
    ax = fig.add_subplot(gs[0, 2:4])
    sns.heatmap(_reshape_by_landmark(rewarded_all, n_landmarks), cmap=REWARD_CMAP, ax=ax, cbar_kws={'label': 'Rewarded'})
    ax.set_xlabel('Landmark ID'); ax.set_ylabel('Lap'); ax.set_title('1) Binary reward map')

    # 2) unconditional landmark-to-landmark transition matrix
    tm = pf.calc_stable_transition_matrix(sess_dataframe, ses_settings)
    ax = fig.add_subplot(gs[0, 4:6])
    _plot_matrix(ax, tm, '2) Landmark transition counts',
                 xlabel='Next landmark ID', ylabel='Current landmark ID')

    # 3) conditional transition matrix per goal + ideal-agent matrix
    transition_prob, control_prob, ideal_prob = pf.calc_stable_conditional_matrix(sess_dataframe, ses_settings)
    max_val = max(np.max(transition_prob), np.max(control_prob), np.max(ideal_prob))
    ax = fig.add_subplot(gs[1, 0:2])
    _plot_matrix(ax, transition_prob, '3) Transition prob. (licked)', y_labels=goal_labels, vmax=max_val)
    ax = fig.add_subplot(gs[1, 2:4])
    _plot_matrix(ax, control_prob, '3) Control prob. (all targets)', y_labels=goal_labels, vmax=max_val)
    ax = fig.add_subplot(gs[1, 4:6])
    _plot_matrix(ax, ideal_prob, '4) Ideal-agent prob.', y_labels=goal_labels, vmax=max_val)

    # 5/6) state-dependent panels -- see plot_sequence_processed's equivalent comment
    laps_needed = pf.calc_laps_needed(ses_settings)
    n_states_observed = (len(np.unique(pf.give_state_id(sess_dataframe, ses_settings)))
                          if laps_needed in STATE_LAPS_NEEDED else 1)
    if laps_needed in STATE_LAPS_NEEDED and n_states_observed > 1:
        # 5) binary lick map split by cycle-state
        by_state = pf.get_binary_licks_by_state(sess_dataframe, ses_settings)
        n_states = len(by_state)
        span = 6 // n_states
        for s, mat in by_state.items():
            ax = fig.add_subplot(gs[2, s * span:(s + 1) * span])
            sns.heatmap(mat, cmap=LICK_CMAP, ax=ax, cbar_kws={'label': 'Licked'})
            ax.set_xlabel('Landmark ID'); ax.set_ylabel('Lap (within state)')
            ax.set_title(f'5) Binary licks, state {s}')

        # 6) sliding-window state-discrimination ratio per goal
        sw_ratio, sw_per_goal = pf.calc_sw_state_ratio(sess_dataframe, ses_settings)
        ax = fig.add_subplot(gs[3, 0:4])
        ax.plot(sw_ratio, color=AVERAGE_COLOR, label='Average', linewidth=3)
        for color, label, curve in zip(goal_colors, goal_labels, sw_per_goal):
            ax.plot(curve, label=label, color=color, linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Lap'); ax.set_ylabel('State-discrimination ratio')
        ax.set_title('6) Sliding-window (10-lap) state-discrimination ratio')
        ax.legend(fontsize=8)
    elif laps_needed not in STATE_LAPS_NEEDED:
        print(f"  laps_needed={laps_needed}: no state cycle to differentiate, skipping panels 5/6")
        _placeholder_axis(fig, gs[2, 0:6],
                           f"5) Binary licks by state: not applicable (laps_needed={laps_needed})")
        _placeholder_axis(fig, gs[3, 0:4],
                           f"6) State-discrimination ratio: not applicable (laps_needed={laps_needed})")
    else:
        print(f"  laps_needed={laps_needed} but state_id never left 0 (likely one or more "
              "defining goals never individually rewarded in the raw data) -- skipping panels 5/6")
        _placeholder_axis(fig, gs[2, 0:6],
                           "5) Binary licks by state: skipped -- state_id stayed at 0 all "
                           "session (one or more defining goals never individually rewarded)")
        _placeholder_axis(fig, gs[3, 0:4],
                           "6) State-discrimination ratio: skipped -- state_id stayed at 0 "
                           "all session")

    # 7) correct-transition rate per sequence step
    performance, perf_list = pf.calc_correct_transition_rate(transition_prob, goals)
    step_labels = [f'{chr(65 + i)}->{chr(65 + (i + 1) % n_goals)}' for i in range(n_goals)]
    ax = fig.add_subplot(gs[3, 4:6])
    ax.bar(step_labels, perf_list, color=goal_colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction correct')
    ax.set_title(f'7) Correct-transition rate (mean={performance:.2f})')

    save_path = Path(save_dir) / 'raw_summary.png' if save_dir is not None else None
    _finish_figure(fig, save_path, show, suptitle='Sequence session -- raw')


# ---------------------------------------------------------------------------
# Shaping sessions
# ---------------------------------------------------------------------------

def plot_shaping_processed(session, frame_rate, save_dir=None, show=True,
                            block_size=10, window_size=(-5, 5), smooth_sigma=3):
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(4, 2, figure=fig)
    print("Building shaping behavioural summary (processed)...")

    # 1) binary licked-landmark vector (+ rewarded, for comparison against raw -- a landmark
    # processed calls "licked" but that got no reward either way is worth a second look)
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(session['licked_lms'][np.newaxis, :], aspect='auto', cmap=LICK_CMAP)
    ax.set_yticks([]); ax.set_xlabel('Landmark visit (chronological)')
    ax.set_title('1) Licked (1) / not licked (0) per visited landmark')

    ax = fig.add_subplot(gs[1, :])
    ax.imshow(session['rewarded_lms'][np.newaxis, :], aspect='auto', cmap=REWARD_CMAP)
    ax.set_yticks([]); ax.set_xlabel('Landmark visit (chronological)')
    ax.set_title('Rewarded (1) / not rewarded (0) per visited landmark')

    # 2) peri-reward lick trace, pooled -- lightly smoothed (frame-binned per-frame signal is
    # much noisier at 45Hz than raw's coarser time-histogram, see smooth_sigma) so the two are
    # easier to visually compare
    windows, avg, sem = pf.extract_peri_reward_signal_npz(
        session['licks'], session['rewards'], frame_rate, window_size)
    t = np.linspace(window_size[0], window_size[1], windows.shape[1])
    avg_smooth = gaussian_filter1d(avg, sigma=smooth_sigma)
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, avg_smooth, color=AVERAGE_COLOR)
    ax.axvline(0, color='grey', linestyle='--')
    ax.set_xlabel('Time from reward (s)'); ax.set_ylabel('Lick signal')
    ax.set_title(f'2) Peri-reward licks, pooled (n={windows.shape[0]})')

    # 3) peri-reward lick trace, blocks of rewards (same smoothing as 2)
    ax = fig.add_subplot(gs[2, 1])
    n_blocks = int(np.ceil(windows.shape[0] / block_size))
    colors = BLOCK_CMAP(np.linspace(0, 1, max(n_blocks, 1)))
    for b in range(n_blocks):
        block = windows[b * block_size:(b + 1) * block_size]
        end = min((b + 1) * block_size, windows.shape[0])
        block_smooth = gaussian_filter1d(block.mean(axis=0), sigma=smooth_sigma)
        ax.plot(t, block_smooth, color=colors[b], label=f'{b * block_size + 1}-{end}')
    ax.axvline(0, color='grey', linestyle='--')
    ax.set_xlabel('Time from reward (s)'); ax.set_ylabel('Lick signal')
    ax.set_title(f'3) Peri-reward licks, blocks of {block_size}')
    ax.legend(fontsize=7)

    # 4) average speed around reward (mean +/- SEM)
    _, speed_avg, speed_sem = pf.extract_peri_reward_signal_npz(
        session['speed'], session['rewards'], frame_rate, window_size)
    ax = fig.add_subplot(gs[3, :])
    ax.plot(t, speed_avg, color=SPEED_COLOR)
    ax.fill_between(t, speed_avg - speed_sem, speed_avg + speed_sem, color=SPEED_COLOR, alpha=0.3)
    ax.axvline(0, color='grey', linestyle='--')
    ax.set_xlabel('Time from reward (s)'); ax.set_ylabel('Speed')
    ax.set_title('4) Average speed around reward (mean ± SEM)')

    save_path = Path(save_dir) / 'processed_summary.png' if save_dir is not None else None
    _finish_figure(fig, save_path, show, suptitle='Shaping session -- processed (frame-aligned)')


def plot_shaping_raw(sess_dataframe, ses_settings, save_dir=None, show=True,
                      block_size=10, window_sec=(-5, 5), hist_bins=41):
    lick_position, lick_times, reward_times, reward_positions, release_df = \
        pf.get_event_parsed(sess_dataframe, ses_settings)
    _, _, _, _, _, licked_all, rewarded_all = pf.calc_hit_fa(sess_dataframe, ses_settings)
    _plot_shaping_raw_panels(licked_all, rewarded_all, lick_times, reward_times, sess_dataframe,
                             save_dir=save_dir, show=show, block_size=block_size,
                             window_sec=window_sec, hist_bins=hist_bins)


def _plot_shaping_raw_panels(licked_all, rewarded_all, lick_times, reward_times, speed_dataframe,
                             save_dir=None, show=True, block_size=10, window_sec=(-5, 5),
                             hist_bins=41, speed_label='Speed (raw treadmill units)'):
    """lick_times/reward_times are timestamps; speed_dataframe has a timestamp index and a
    'Treadmill' column (see pf.extract_peri_reward_speed)."""
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(4, 2, figure=fig)
    print("Building shaping behavioural summary (raw)...")

    # 1) binary licked-landmark vector (+ rewarded, for comparison against processed)
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(licked_all[np.newaxis, :], aspect='auto', cmap=LICK_CMAP)
    ax.set_yticks([]); ax.set_xlabel('Landmark visit (chronological)')
    ax.set_title('1) Licked (1) / not licked (0) per visited landmark (raw)')

    ax = fig.add_subplot(gs[1, :])
    ax.imshow(rewarded_all[np.newaxis, :], aspect='auto', cmap=REWARD_CMAP)
    ax.set_yticks([]); ax.set_xlabel('Landmark visit (chronological)')
    ax.set_title('Rewarded (1) / not rewarded (0) per visited landmark (raw)')

    # 2) peri-reward lick histogram, pooled
    per_reward = pf.extract_peri_reward_lick_times(lick_times, reward_times, window_sec)
    bins = np.linspace(window_sec[0], window_sec[1], hist_bins)
    centers = (bins[:-1] + bins[1:]) / 2
    pooled = np.concatenate(per_reward) if len(per_reward) else np.array([])
    counts, _ = np.histogram(pooled, bins=bins)
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(centers, counts, color=AVERAGE_COLOR)
    ax.axvline(0, color='grey', linestyle='--')
    ax.set_xlabel('Time from reward (s)'); ax.set_ylabel('Lick count')
    ax.set_title(f'2) Peri-reward licks, pooled (n={len(per_reward)})')

    # 3) peri-reward lick histogram, blocks of rewards
    ax = fig.add_subplot(gs[2, 1])
    n_blocks = int(np.ceil(len(per_reward) / block_size))
    colors = BLOCK_CMAP(np.linspace(0, 1, max(n_blocks, 1)))
    for b in range(n_blocks):
        block = per_reward[b * block_size:(b + 1) * block_size]
        pooled_block = np.concatenate(block) if len(block) else np.array([])
        counts, _ = np.histogram(pooled_block, bins=bins)
        end = min((b + 1) * block_size, len(per_reward))
        ax.plot(centers, counts, color=colors[b], label=f'{b * block_size + 1}-{end}')
    ax.axvline(0, color='grey', linestyle='--')
    ax.set_xlabel('Time from reward (s)'); ax.set_ylabel('Lick count')
    ax.set_title(f'3) Peri-reward licks, blocks of {block_size} (raw)')
    ax.legend(fontsize=7)

    # 4) average speed around reward (mean +/- SEM)
    rel_time, windows, speed_avg, speed_sem = pf.extract_peri_reward_speed(
        speed_dataframe, reward_times, window_sec)
    ax = fig.add_subplot(gs[3, :])
    ax.plot(rel_time, speed_avg, color=SPEED_COLOR)
    ax.fill_between(rel_time, speed_avg - speed_sem, speed_avg + speed_sem, color=SPEED_COLOR, alpha=0.3)
    ax.axvline(0, color='grey', linestyle='--')
    ax.set_xlabel('Time from reward (s)'); ax.set_ylabel(speed_label)
    ax.set_title('4) Average speed around reward (mean ± SEM, raw)')

    save_path = Path(save_dir) / 'raw_summary.png' if save_dir is not None else None
    _finish_figure(fig, save_path, show, suptitle='Shaping session -- raw')


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_behaviour_summary(mouse, session_id, stage, behav_root, world='stable',
                           root_path=None, save_dir=None, show=True):
    """Build the processed and raw behavioural-performance summary figures for one session,
    auto-detecting sequence vs shaping. Each is saved as a single combined PNG: save_dir/
    processed_summary.png and save_dir/raw_summary.png. If save_dir is not given, it defaults
    to default_save_dir(root_path, mouse, session_id) (root_path/analysis/sub-mouse/session) --
    pass save_dir=False to skip saving entirely. root_path is the project's own root folder
    (e.g. .../SuKuSaRe_20250923_HFScohort3), NOT behav_root -- for cohort3-style dual-tree
    projects behav_root is already a deep subpath (.../preprocessed_behav_Nov2025/derivatives).
    Defaults to behav_root if not given (correct for GoalProgress-style single-tree projects,
    where behav_root already is the project root's BehaviourAligned_robust child). Returns
    (session, sess_dataframe, ses_settings)."""
    if root_path is None:
        root_path = behav_root
    session = load_processed_session(mouse, session_id, stage, behav_root, world=world)
    sess_dataframe, ses_settings = load_raw_session(mouse, session_id, behav_root)
    sequence = is_sequence_session(session)

    if save_dir is None:
        save_dir = default_save_dir(root_path, mouse, session_id)
    elif save_dir is False:
        save_dir = None
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"Session type: {'sequence' if sequence else 'shaping'} "
          f"({len(session['goal_ids'])} rewarded landmark(s))")

    if sequence:
        plot_sequence_processed(session, save_dir=save_dir, show=show)
        plot_sequence_raw(sess_dataframe, ses_settings, save_dir=save_dir, show=show)
    else:
        session_path = Path(parse_nwb_functions.find_base_path(mouse, session_id, behav_root))
        frame_rate = _get_frame_rate(session_path)
        plot_shaping_processed(session, frame_rate, save_dir=save_dir, show=show)
        plot_shaping_raw(sess_dataframe, ses_settings, save_dir=save_dir, show=show)

    return session, sess_dataframe, ses_settings


# ---------------------------------------------------------------------------
# Cohort2 (sessions rebuilt into BehaviourAligned_robust by build_session_nwb_cohort2.py)
# ---------------------------------------------------------------------------
# Both figures are parsed with parse_session_functions_cohort2: "processed" from the robust,
# frame-aligned behaviour_data.npz, "raw" from the VR position_log.csv (VR-clock timestamps).
# The parsed lap x landmark lick/reward maps, goal slots and lap states are then handed to the
# same cohort3 metric helpers and plotting functions used above, so the panels are computed
# and drawn identically across cohorts.
#
# Session type: "sequence" needs the loop corridor (t7+: every odour exactly once per lap) and
# more than one goal; everything else (t1-t6 linear corridors) gets the shaping layout. t0
# has no landmarks and is skipped.

def _cohort2_vr_dir(session_path, mouse):
    """Raw VR folder (<project>/<mouse>/<session>/behav/<yymmdd>, holding config.yaml and
    position_log.csv) for a session folder <project>/BehaviourAligned_robust/sub-<mouse>/<session>."""
    session_path = Path(session_path)
    raw_session = session_path.parents[2] / mouse / session_path.name
    return sorted((raw_session / 'behav').glob('*/config.yaml'))[0].parent


def _parse_cohort2_session(session, options):
    """Lap/landmark parsing shared by the processed and raw cohort2 structs."""
    session = pf2.get_lap_idx(session)
    session = pf2.get_lm_idx(session)
    session = pf2.threshold_licks(session)
    session = pf2.get_licked_lms(session)
    session = pf2.get_rewarded_lms(session)
    session = pf2.get_lms_visited(options, session)
    return session


def load_cohort2_sessions(mouse, session_id, behav_root):
    """(session_path, processed, raw, options): cohort2 session structs from the robust
    behaviour_data.npz and from the VR position_log.csv, parsed alike. Returns None for the
    struct pair if the session has no landmarks (t0)."""
    session_path = Path(parse_nwb_functions.find_base_path(mouse, session_id, behav_root))
    vr_dir = str(_cohort2_vr_dir(session_path, mouse))
    options = pf2.load_config(vr_dir)
    if not options['flip_tunnel'].get('landmarks'):
        return session_path, None, None, options
    # t0-t1 have no lick speed limit: keep every lick
    options['flip_tunnel'].setdefault('speed_limit', np.inf)

    data = pf2.load_session_npz(str(session_path))
    processed = pf2.create_session_struct_npz(data, options)
    processed['lick_frames'] = data['licks']  # per frame, for the peri-reward lick trace

    raw = pf2.create_session_struct(pf2.load_session(vr_dir), options)
    # NIDAQ reward pulses (processed) include assisted/manual rewards; count them in raw too
    raw['rewards'] = np.sort(np.concatenate([raw['rewards'], raw['assist_rewards'],
                                             raw['manual_rewards']]))

    processed = _parse_cohort2_session(processed, options)
    raw = _parse_cohort2_session(raw, options)
    return session_path, processed, raw, options


def is_cohort2_sequence_session(options):
    ft = options['flip_tunnel']
    loop_corridor = len(ft['landmarks']) == len(set(ft['landmarks_sequence']))
    return loop_corridor and 1 < len(ft['goals']) < len(ft['landmarks'])


def cohort2_sequence_view(session):
    """Cohort3-style per-visit struct (see plot_sequence_processed) from a parsed cohort2
    struct. Goals and landmark IDs are landmark slots, which in the cohort2 loop corridor are
    also the odour ids."""
    session = pf2.get_active_goal(session)
    session = pf2.calc_laps_needed(session)
    session = pf2.give_lap_state_id(session)
    n_landmarks = len(session['landmarks'])
    n_laps = session['licked_lms'].shape[0]
    view = dict(num_landmarks=n_landmarks, num_laps=n_laps,
                goal_ids=[int(g) for g in session['goal_idx']],
                licked_lms=session['licked_lms'].flatten().astype(int),
                rewarded_lms=session['rewarded_lms'].flatten().astype(int),
                all_lms=np.tile(np.arange(n_landmarks), n_laps),
                laps_needed=session['laps_needed'], state_id=session['state_id'])
    return pf.calc_conditional_probs_npz(view)


def _cohort2_visit_vectors(session):
    """Licked/rewarded per visited landmark, chronological (truncated to the landmarks the
    mouse actually reached, see get_lms_visited)."""
    n_visits = len(session['all_lms'])
    return (session['licked_lms'].flatten()[:n_visits].astype(int),
            session['rewarded_lms'].flatten()[:n_visits].astype(int))


def plot_shaping_cohort2(processed, raw, frame_rate, save_dir=None, show=True):
    licked, rewarded = _cohort2_visit_vectors(processed)
    view = dict(licked_lms=licked, rewarded_lms=rewarded, licks=processed['lick_frames'],
                rewards=processed['rewards'], speed=processed['speed'])
    plot_shaping_processed(view, frame_rate, save_dir=save_dir, show=show)

    licked, rewarded = _cohort2_visit_vectors(raw)
    vr_time = pd.to_datetime(raw['time'], unit='s')
    speed_dataframe = pd.DataFrame({'Treadmill': np.asarray(raw['speed'], dtype=float)}, index=vr_time)
    _plot_shaping_raw_panels(licked, rewarded, vr_time[raw['thresholded_licks']],
                             vr_time[raw['rewards']], speed_dataframe, save_dir=save_dir,
                             show=show, speed_label='Speed (VR, cm/s)')


def run_behaviour_summary_cohort2(mouse, session_id, stage, behav_root, root_path=None,
                                  save_dir=None, show=True):
    """Cohort2 counterpart of run_behaviour_summary (same save_dir/root_path conventions and
    output files: processed_summary.png, raw_summary.png). Returns (processed, raw, options)."""
    if root_path is None:
        root_path = behav_root
    session_path, processed, raw, options = load_cohort2_sessions(mouse, session_id, behav_root)
    if processed is None:
        print("  no landmarks in this session (t0): skipping behavioural summary")
        return None, None, options

    if save_dir is None:
        save_dir = default_save_dir(root_path, mouse, session_id)
    elif save_dir is False:
        save_dir = None
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    sequence = is_cohort2_sequence_session(options)
    print(f"Session type: {'sequence' if sequence else 'shaping'} (cohort2, stage {stage})")
    if sequence:
        plot_sequence_processed(cohort2_sequence_view(processed), save_dir=save_dir, show=show)
        plot_sequence_processed(cohort2_sequence_view(raw), save_dir=save_dir, show=show,
                                suptitle='Sequence session -- raw', filename='raw_summary.png')
    else:
        plot_shaping_cohort2(processed, raw, _get_frame_rate(session_path),
                             save_dir=save_dir, show=show)

    return processed, raw, options

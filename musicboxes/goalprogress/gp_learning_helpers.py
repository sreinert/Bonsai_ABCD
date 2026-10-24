"""Goal-progress (GP) coding while the mouse updates its model of the task -- holography cohort.

Backs musicboxes/goalprogress/GP_learning_holography.ipynb, which asks what GP-tuned cells and
the population code do when performance is poor (t0: first shaping session; t4: first ABC
session; t6: new sequence) compared with good sessions (t3: end of shaping; t5: ABC learned),
and on single error trials (false alarms, misses).

Provenance
----------
The per-session GP machinery is lifted from GP_reward_prediction_holography.ipynb (which is left
untouched) so both notebooks can share it:
  - rewarded_slots_from_settings, goal_visit_for_rewards: goal slots from the trial settings
    (session['goal_ids'] stops at availableRewardPositions and misses goals in shaping), and
    the visit that earned each reward (lm_idx at the reward frame is wrong when the reward lands
    after the mouse has left the goal landmark).
  - set_all_reward_transitions -> reward_segments: every consecutive reward pair is one GP
    segment, NOT only "correct" transitions -- in poor sessions there are hardly any.
  - gp_frame_bins / gp_tuning_from_groups / shuffled_groups -> segment_layout / tuning_from_groups
    / shuffled_groups: vectorised binning (same np.digitize edges as
    cellTV_functions_goalprogress.extract_goal_progress_pooled) and the circular-shift shuffle.
  - preferred_phase, rowwise_corr, circular_distance, sorted_display_curves.

New here
--------
  - Two GP "clocks": 'reward' (reward -> reward, what the mouse experienced) and 'task' (goal
    entry -> next goal entry, licked or not, what the task defines). They coincide when the
    mouse hits every goal and diverge on misses. A "previous task" clock is NOT offered: the
    shaping (11 slots, goals 3/7, 66 units) and ABC (8 slots, goals 1/3/6, 68 units) corridors
    differ, so old slot indices don't refer to the same places.
  - visit_table: one row per landmark visit with outcome hit / miss / FA / CR. Hits come from
    goal_visit_for_rewards, not session['rewarded_lms'] (which marks the NEXT visit when the
    reward lands late); FA = licked (after release) at a non-goal landmark. In the shaping
    corridor landmarks touch (no gap), so licking that runs on into a goal or carries on from one
    lands in the neighbouring non-goal landmark; those visits are labelled 'anticipatory' /
    'post-goal' instead of FA and are left out of FA, CR and the FA rate.
  - Two normalisations of dF: 'dFz' (z-score per session; decoders, PCA) and 'dFn' (divided by
    a robust noise estimate, median/MAD). dF here is F - 0.7 Fneu minus baseline in raw units,
    which drift between days, and a per-session z-score would hide a cell going silent; noise
    units keep "silent" visible (S1, S2).
  - Single-segment field metrics (reliability / gain / jitter), sequence scores, LDA decoders of
    GP phase and lap position with segment-grouped CV, cross-session decoding with tracked cells,
    entry-aligned decoded-phase posteriors by visit outcome, and PCA projection of trial-averaged
    GP trajectories onto a good-session subspace.
"""

import io
import os
import contextlib
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

import era_brewer
import parse_nwb_functions
import musicboxes.session_overview as session_overview
import preprocessing.parse_session_functions_cohort3_summary as pfs
import preprocessing.quality_control.tracking_overview as tracking_overview

PROJECT_ROOT = '/Volumes/mrsic_flogel/public/projects/SaReSuKuMaRa_20260911_holography/cohort1'
BEHAV_ROOT = f'{PROJECT_ROOT}/BehaviourAligned_robust'
TRACKING_ROOT = f'{PROJECT_ROOT}/analysis'

STAGES = ['t0', 't1', 't2', 't3', 't4', 't5', 't6']
GOOD, POOR = ['t3', 't5'], ['t0', 't4', 't6']
SHAPING, ABC = ['t0', 't1', 't2', 't3'], ['t4', 't5', 't6']
OUTCOMES = ['hit', 'miss', 'FA', 'CR', 'anticipatory', 'post-goal']
PHASE_CLASSES = {'post-reward': (0.0, 1 / 3), 'middle': (1 / 3, 2 / 3), 'pre-reward': (2 / 3, 1.0)}

SESSION_COLORS = dict(zip(STAGES, era_brewer.era_brew('SpeakNow1', n=9, brew_type='continuous')[1:8]))
MOUSE_COLORS = dict(zip(['001', '002', '003', '004'], era_brewer.era_brew('Lover2', n=4, brew_type='discrete')))
_lover = era_brewer.era_brew('Lover1', n=5, brew_type='discrete')
OUTCOME_COLORS = {'hit': _lover[1], 'miss': _lover[0], 'FA': _lover[3], 'CR': _lover[4], 'anticipatory': _lover[2],
                  'post-goal': '0.7', 'lick': '0.5'}
CLASS_COLORS = dict(zip(PHASE_CLASSES, era_brewer.era_brew('Midnight1', n=3, brew_type='discrete')))
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    'NineteenEightyNine_r', era_brewer.era_brew('NineteenEightyNine', n=256, brew_type='continuous')[::-1])
PHASE_CMAP = LinearSegmentedColormap.from_list(
    'Midnight1', era_brewer.era_brew('Midnight1', n=256, brew_type='continuous'))


# ---------------------------------------------------------------------------
# Circular helpers (phase in [0, 1))
# ---------------------------------------------------------------------------

def circular_distance(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 1
    return np.minimum(d, 1 - d)


def circular_mean(phases, axis=None, weights=None):
    angle = 2 * np.pi * np.asarray(phases)
    w = np.ones_like(angle) if weights is None else weights
    return (np.arctan2(np.nansum(w * np.sin(angle), axis=axis), np.nansum(w * np.cos(angle), axis=axis))
            / (2 * np.pi)) % 1


def circular_std(phases):
    """Circular SD in phase units (0 = identical, ~0.4 = uniform)."""
    angle = 2 * np.pi * np.asarray(phases)[np.isfinite(phases)]
    if len(angle) < 2:
        return np.nan
    R = np.hypot(np.mean(np.sin(angle)), np.mean(np.cos(angle)))
    return np.sqrt(-2 * np.log(max(R, 1e-12))) / (2 * np.pi)


def circular_corr(a, b):
    """Circular-circular correlation (Jammalamadaka & SenGupta) between two phase arrays."""
    a, b = 2 * np.pi * np.asarray(a), 2 * np.pi * np.asarray(b)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 3:
        return np.nan
    sa = np.sin(a - np.arctan2(np.sin(a).sum(), np.cos(a).sum()))
    sb = np.sin(b - np.arctan2(np.sin(b).sum(), np.cos(b).sum()))
    return np.sum(sa * sb) / np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2))


def rowwise_corr(a, b):
    """Pearson r between matching rows of two (n_cells, bins) arrays, over bins finite in both."""
    good = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
    a = a[:, good] - a[:, good].mean(axis=1, keepdims=True)
    b = b[:, good] - b[:, good].mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))


def mean_curve(mat):
    return np.nanmean(mat, axis=1)


def preferred_phase(curves, sigma=1):
    """Peak of each smoothed (n, bins) curve as a phase (bin centre, 0-1); NaN bins filled with
    the row mean first. Smoothing wraps, since phase is circular."""
    curves = np.atleast_2d(curves)
    filled = np.where(np.isnan(curves), np.nanmean(curves, axis=1, keepdims=True), curves)
    filled = np.nan_to_num(filled)
    smoothed = gaussian_filter1d(filled, sigma=sigma, axis=1, mode='wrap') if sigma else filled
    return (np.argmax(smoothed, axis=1) + 0.5) / curves.shape[1]


def phase_class(phase):
    out = np.full(len(phase), '', dtype=object)
    for name, (lo, hi) in PHASE_CLASSES.items():
        out[(phase >= lo) & (phase < hi)] = name
    return out


def zscore_rows(x):
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    return (x - mu) / np.where(sd > 0, sd, np.nan)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def rewarded_slots_from_settings(mouse, tag, behav_root=BEHAV_ROOT):
    """Landmark slots with any rewarded variant (rewardSequencePosition > -1) in the trial
    build: [3, 7] in shaping, [1, 3, 6] in ABC."""
    session_path = parse_nwb_functions.find_base_path(mouse, tag, behav_root)
    ses_settings, _ = parse_nwb_functions.load_settings(session_path)
    trial = pfs.get_trial_settings(ses_settings)
    return [i for i, variants in enumerate(trial['landmarks'])
            if any(v['rewardSequencePosition'] > -1 for v in variants)]


def session_exists(mouse, tag, behav_root=BEHAV_ROOT):
    with contextlib.redirect_stdout(io.StringIO()):
        return os.path.isdir(f'{behav_root}/sub-{mouse}') and \
            parse_nwb_functions.find_base_path(mouse, tag, behav_root) is not None


def load_session(mouse, tag, behav_root=BEHAV_ROOT, cache_dir=None):
    """session_overview.load_session_data plus goal slots, a common frame count for dF and
    behaviour (they can differ by a frame), and the 'dFz' / 'dFn' normalisations. ~1.5 min per
    session from the share; set cache_dir to a LOCAL folder to pickle the result. The visit table
    and GP segments are rebuilt on every call (cheap), so changes to their definitions also
    apply to cached sessions."""
    cache = os.path.join(cache_dir, f'sub-{mouse}_{tag}.pkl') if cache_dir else None
    if cache and os.path.exists(cache):
        with open(cache, 'rb') as f:
            out = pickle.load(f)
        out['visits'] = visit_table(out)
        out['segments'] = {'reward': reward_segments(out), 'task': task_segments(out)}
        return out
    with contextlib.redirect_stdout(io.StringIO()):
        loaded = session_overview.load_session_data(mouse, tag, tag, behav_root=behav_root, cohort='3',
                                                    world='stable', stage_type='sequences')
        loaded['session']['rewarded_slots'] = rewarded_slots_from_settings(mouse, tag, behav_root)
    session = loaded['session']
    n = min(loaded['dF'].shape[1], len(session['position']))
    dF = np.asarray(loaded['dF'][:, :n], dtype=float)
    med = np.nanmedian(dF, axis=1, keepdims=True)
    noise = 1.4826 * np.nanmedian(np.abs(dF - med), axis=1, keepdims=True)
    out = dict(mouse=mouse, tag=tag, dF=dF, dFz=zscore_rows(dF), dFn=(dF - med) / noise,
               neurons=np.asarray(loaded['neurons']), session=session, frame_rate=loaded['frame_rate'],
               n_frames=n)
    out['visits'] = visit_table(out)
    out['segments'] = {'reward': reward_segments(out), 'task': task_segments(out)}
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache, 'wb') as f:
            pickle.dump(out, f)
    return out


# ---------------------------------------------------------------------------
# Behaviour: visit outcomes and performance
# ---------------------------------------------------------------------------

def goal_visit_for_rewards(session):
    """Visit index that earned each reward: the latest goal-slot visit whose entry frame is at or
    before the reward frame (see GP_reward_prediction_holography.ipynb for why not lm_idx)."""
    n_visits = len(session['lm_entry'])
    visit_slots = np.asarray(session['all_lms'][:n_visits])
    goal_visits = np.where(np.isin(visit_slots, session['rewarded_slots']))[0]
    k = np.searchsorted(session['lm_entry'][goal_visits], session['rewards'], side='right') - 1
    if np.any(k < 0):
        raise ValueError(f"{np.sum(k < 0)} reward(s) occur before the first goal visit")
    return goal_visits[k]


def lick_onset_mask(licks):
    onset = np.zeros(len(licks), dtype=bool)
    onset[0] = licks[0] > 0
    onset[1:] = (licks[1:] > 0) & (licks[:-1] <= 0)
    return onset


def visit_table(loaded, anticip_window=(-1.0, 0.0), speed_windows=((-1.0, 0.0), (-3.0, -2.0)),
                max_gap=0.5, boundary_s=0.5):
    """One row per landmark visit: slot, goal flag, outcome, entry/exit frames, the reward frame
    (hits), anticipatory lick onsets before entry and the pre-entry speed ratio (late/early
    window; < 1 = slowing down). 'predicted' marks hits with >= 1 anticipatory lick onset in
    anticip_window (s, relative to entry).

    Outcomes: hit / miss (goal visits), FA / CR (non-goal visits), plus two non-goal categories
    for licking that crosses into or out of an ADJACENT goal (landmarks <= max_gap units apart --
    the shaping corridor has no gaps, ABC has 2.5-unit gaps so this never applies there):
      'anticipatory' : licked non-goal visit, next landmark is a goal, and the mouse licks in the
                       last boundary_s before leaving -- licking that runs on into the goal.
      'post-goal'    : licked non-goal visit, previous landmark is a goal, and the mouse licks in
                       the last boundary_s before entering -- licking carried on from the goal.
    These are excluded from FA, CR and the FA rate (see session_performance)."""
    session, fr, n = loaded['session'], loaded['frame_rate'], loaded['n_frames']
    n_visits = len(session['lm_entry'])
    slots = np.asarray(session['all_lms'][:n_visits]).astype(int)
    is_goal = np.isin(slots, session['rewarded_slots'])
    rewards = np.asarray(session['rewards'])
    earned = goal_visit_for_rewards(session)
    reward_frame = np.full(n_visits, -1)
    for v, r in zip(earned[::-1], rewards[::-1]):   # first reward per visit wins
        reward_frame[v] = r
    licked = np.asarray(session['licked_lms'][:n_visits]) > 0
    outcome = np.where(is_goal, np.where(reward_frame >= 0, 'hit', 'miss'), np.where(licked, 'FA', 'CR')).astype(object)

    entry, exit_ = np.asarray(session['lm_entry']), np.asarray(session['lm_exit'])
    licks = np.asarray(session['licks'][:n]) > 0
    lms = np.asarray(session['landmarks'])[:n_visits]
    gap = lms[1:, 0] - lms[:-1, 1]                      # gap between visit i and i+1
    touches_next_goal = np.r_[is_goal[1:] & (gap <= max_gap), False]
    touches_prev_goal = np.r_[False, is_goal[:-1] & (gap <= max_gap)]
    b = int(round(boundary_s * fr))
    licks_before = lambda f: licks[max(f - b, 0):max(f + 1, 0)].any()
    for i in np.where(outcome == 'FA')[0]:
        if touches_next_goal[i] and licks_before(exit_[i]):
            outcome[i] = 'anticipatory'
        elif touches_prev_goal[i] and licks_before(entry[i]):
            outcome[i] = 'post-goal'

    onset = lick_onset_mask(session['licks'][:n])
    lo, hi = (int(round(w * fr)) for w in anticip_window)
    speed = np.asarray(session['speed'][:n])
    (a0, a1), (b0, b1) = [[int(round(w * fr)) for w in win] for win in speed_windows]
    anticip, speed_ratio = np.zeros(n_visits, int), np.full(n_visits, np.nan)
    for i, e in enumerate(entry):
        anticip[i] = onset[max(e + lo, 0):max(e + hi, 0)].sum()
        if e + b0 >= 0 and e + a1 <= n:
            speed_ratio[i] = np.mean(speed[e + a0:e + a1]) / max(np.mean(speed[e + b0:e + b1]), 1e-6)
    vt = pd.DataFrame(dict(slot=slots, is_goal=is_goal, licked=licked, outcome=outcome, entry=entry,
                           exit=exit_, reward_frame=reward_frame, anticip_licks=anticip,
                           speed_ratio=speed_ratio))
    vt['predicted'] = (vt.outcome == 'hit') & (vt.anticip_licks > 0)
    return vt


def dprime(hit_rate, fa_rate, clip=(0.01, 0.99)):
    h, f = np.clip(hit_rate, *clip), np.clip(fa_rate, *clip)
    from scipy.stats import norm
    return norm.ppf(h) - norm.ppf(f)


def session_performance(vt):
    """Hit rate over goal visits, FA rate over FA + CR visits (anticipatory / post-goal licks at
    landmarks touching a goal are left out), d'."""
    goal, non = vt[vt.is_goal], vt[vt.outcome.isin(['FA', 'CR'])]
    h = (goal.outcome == 'hit').mean() if len(goal) else np.nan
    f = (non.outcome == 'FA').mean() if len(non) else np.nan
    return dict(hit_rate=h, fa_rate=f, dprime=dprime(h, f), n_goal=len(goal), n_hits=int((goal.outcome == 'hit').sum()))


def rolling_performance(vt, window=20):
    """Per visit: hit rate over a centred window of `window` GOAL visits and FA rate over a window
    of `window` NON-goal visits (goal visits are sparse in shaping, so a window of visits would
    hold too few), forward/back-filled onto every visit; plus d'."""
    def roll(mask, outcome):
        r = (vt.outcome[mask] == outcome).astype(float).rolling(window, center=True, min_periods=window // 3).mean()
        return r.reindex(vt.index).ffill().bfill().to_numpy()
    h, f = roll(vt.is_goal, 'hit'), roll(vt.outcome.isin(['FA', 'CR']), 'FA')
    return pd.DataFrame(dict(hit_rate=h, fa_rate=f, dprime=dprime(h, f)))


def block_performance(vt, frame_edges):
    """d' / hit / FA rate of the visits whose entry falls in each [frame_edges[i], frame_edges[i+1])."""
    rows = []
    for lo, hi in zip(frame_edges[:-1], frame_edges[1:]):
        rows.append(session_performance(vt[(vt.entry >= lo) & (vt.entry < hi)]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GP segments, binning, tuning
# ---------------------------------------------------------------------------

def reward_segments(loaded, dedupe=True):
    """(start_frame, end_frame) for every consecutive reward pair (the 'reward' clock). With
    dedupe, repeat rewards earned in the same visit are dropped first (zero-length segments)."""
    session = loaded['session']
    rewards = np.asarray(session['rewards'])
    if dedupe and len(rewards):
        visit = goal_visit_for_rewards(session)
        rewards = rewards[np.r_[True, visit[1:] != visit[:-1]]]
    rewards = rewards[rewards < loaded['n_frames']]
    return np.stack([rewards[:-1], rewards[1:]], axis=1) if len(rewards) > 1 else np.empty((0, 2), int)


def task_segments(loaded):
    """(start_frame, end_frame) between consecutive goal-landmark ENTRIES, rewarded or not (the
    'task' clock)."""
    vt = loaded['visits']
    entries = vt.entry[vt.is_goal].to_numpy()
    entries = entries[entries < loaded['n_frames']]
    return np.stack([entries[:-1], entries[1:]], axis=1) if len(entries) > 1 else np.empty((0, 2), int)


def segment_labels(loaded, segments):
    """'miss' if a goal visit was missed inside the segment, else 'FA' if any FA, else 'clean'."""
    vt = loaded['visits']
    labels = []
    for s, e in segments:
        inside = vt[(vt.entry > s) & (vt.entry < e)]
        if (inside.outcome == 'miss').any():
            labels.append('miss')
        elif (inside.outcome == 'FA').any():
            labels.append('FA')
        else:
            labels.append('clean')
    return np.array(labels)


def segment_layout(segments, bins):
    """Frame layout of a segment list, binned exactly as extract_goal_progress_pooled
    (np.digitize on np.linspace(start, end, bins + 1)). Generalises the old notebook's
    gp_frame_bins from session['correct_transitions'] to any (start, end) array."""
    frames, seg, group = [], [], []
    for i, (s, e) in enumerate(np.asarray(segments, dtype=int)):
        if e <= s:
            continue
        f = np.arange(s, e)
        bin_ix = np.digitize(f, np.linspace(s, e, bins + 1)) - 1
        frames.append(f)
        seg.append(np.full(len(f), i))
        group.append(i * bins + bin_ix)
    n_seg = len(segments)
    frames, seg, group = np.concatenate(frames), np.concatenate(seg), np.concatenate(group)
    seg_start = np.searchsorted(seg, np.arange(n_seg))
    seg_len = np.bincount(seg, minlength=n_seg)
    pos = np.arange(len(frames)) - seg_start[seg]
    return dict(frames=frames, seg=seg, pos=pos, group=group, seg_start=seg_start, seg_len=seg_len,
                n_seg=n_seg, bins=bins)


def binned_from_groups(dF_frames, group, layout):
    """(n_cells, n_seg, bins) bin means from per-frame group labels."""
    n_groups = layout['n_seg'] * layout['bins']
    indicator = sparse.csr_matrix((np.ones(len(group)), (np.arange(len(group)), group)),
                                  shape=(len(group), n_groups))
    sums = np.asarray((indicator.T @ dF_frames.T).T)
    counts = np.bincount(group, minlength=n_groups).astype(float)
    counts[counts == 0] = np.nan
    return (sums / counts).reshape(-1, layout['n_seg'], layout['bins'])


def binned_matrix(dF, segments, bins):
    layout = segment_layout(segments, bins)
    return binned_from_groups(dF[:, layout['frames']], layout['group'], layout)


def tuning_index(mat):
    curve = mean_curve(mat)
    return (np.nanmax(curve, axis=1) - np.nanmin(curve, axis=1)) / np.nanmean(curve, axis=1)


def shuffled_groups(layout, rng):
    """Circularly shift each segment's frames by an independent random amount."""
    seg, pos, seg_start, seg_len = layout['seg'], layout['pos'], layout['seg_start'], layout['seg_len']
    shift = rng.integers(0, np.maximum(seg_len, 1))
    src = seg_start[seg] + (pos + shift[seg]) % seg_len[seg]
    new_group = np.empty_like(layout['group'])
    new_group[src] = layout['group']
    return new_group


def gp_tuning(loaded, clock='reward', bins=30, n_shuffles=500, alpha=0.05, seed=0, signal='dF', segments=None):
    """Per-cell GP tuning on one clock (or on an explicit `segments` subset). Tuning index
    ((max-min)/mean of the curve) is computed on raw dF as in the earlier notebooks (scale-free,
    but needs a positive mean); the returned 'mat' is in noise units (dFn) for plotting and field
    metrics. Significance: circular-shift shuffle within segments (n_shuffles=0 skips it). Also
    returns split-half (even/odd segment) curve reliability."""
    segments = loaded['segments'][clock] if segments is None else np.asarray(segments)
    if len(segments) < 4:
        return None
    rng = np.random.default_rng(seed)
    layout = segment_layout(segments, bins)
    frames = layout['frames']
    raw = binned_from_groups(loaded[signal][:, frames], layout['group'], layout)
    observed = tuning_index(raw)
    if n_shuffles:
        shuf = np.stack([tuning_index(binned_from_groups(loaded[signal][:, frames], shuffled_groups(layout, rng),
                                                         layout)) for _ in range(n_shuffles)], axis=1)
        p = (1 + np.sum(shuf >= observed[:, None], axis=1)) / (1 + n_shuffles)
        shuffle_z = (observed - np.nanmean(shuf, 1)) / np.nanstd(shuf, 1)
    else:
        p, shuffle_z = np.full(len(observed), np.nan), np.full(len(observed), np.nan)
    mat = binned_from_groups(loaded['dFn'][:, frames], layout['group'], layout)
    return dict(clock=clock, bins=bins, segments=segments, mat=mat, tuning=observed, p=p, significant=p < alpha,
                shuffle_z=shuffle_z,
                phase=preferred_phase(mean_curve(mat)),
                phase_even=preferred_phase(mean_curve(mat[:, 0::2])),
                phase_odd=preferred_phase(mean_curve(mat[:, 1::2])),
                split_r=rowwise_corr(mean_curve(mat[:, 0::2]), mean_curve(mat[:, 1::2])))


def sorted_display_curves(sort_mat, display_mat):
    """Z-scored trial-averaged curves of display_mat ordered by each cell's peak in sort_mat."""
    sort_curve = zscore_rows(mean_curve(sort_mat))
    display_curve = zscore_rows(mean_curve(display_mat))
    order = np.argsort(preferred_phase(sort_curve, sigma=0))
    return display_curve[order]


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def load_tracking(mouse, tracking_root=TRACKING_ROOT):
    path = f'{tracking_root}/sub-{mouse}/tracking'
    return tracking_overview.load_cell_matrix(path) if os.path.exists(f'{path}/cell_matrix.npz') else None


def tracked_rows(cm, loaded_by_tag, tags, max_shift_um=5.0):
    """dF row indices (n_cells x len(tags)) of cells tracked in all `tags`, or None if any tag is
    not in the tracking. Checks that the tracking's iscell ROIs equal the NWB's accepted ROIs."""
    if cm is None or any(t not in list(cm['tags']) or t not in loaded_by_tag for t in tags):
        return None
    roi, _ = tracking_overview.select_tracked_cells(cm, tags, max_shift_um=max_shift_um)
    rows = np.zeros(roi.shape, int)
    for j, tag in enumerate(tags):
        neurons = loaded_by_tag[tag]['neurons']
        col = cm['roi'][:, list(cm['tags']).index(tag)]
        if set(col[col >= 0].tolist()) != set(neurons.tolist()):
            raise ValueError(f'{tag}: tracking iscell ROIs != NWB accepted ROIs -- suite2p rerun after tracking?')
        to_row = {int(r): i for i, r in enumerate(neurons)}
        rows[:, j] = [to_row[int(r)] for r in roi[:, j]]
    return rows


# ---------------------------------------------------------------------------
# S1: event-aligned responses
# ---------------------------------------------------------------------------

def peri_event(signal, events, frame_rate, window=(-2, 4)):
    """(n_cells, n_events, n_t) windows of a (n_cells, n_frames) signal (1D signals are treated as
    one cell); events whose window leaves the recording are dropped."""
    signal = np.atleast_2d(signal)
    w = np.arange(int(round(window[0] * frame_rate)), int(round(window[1] * frame_rate)))
    events = np.asarray([e for e in events if e + w[0] >= 0 and e + w[-1] < signal.shape[1]], int)
    t = w / frame_rate
    if not len(events):
        return np.empty((signal.shape[0], 0, len(w))), t
    return signal[:, events[:, None] + w[None, :]], t


def response_index(windows, t, pre=(-2, 0), post=(0, 2)):
    """Mean post minus mean pre, per cell and event."""
    pre_m = (t >= pre[0]) & (t < pre[1])
    post_m = (t >= post[0]) & (t < post[1])
    return np.nanmean(windows[..., post_m], -1) - np.nanmean(windows[..., pre_m], -1)


# ---------------------------------------------------------------------------
# S2: single-segment field metrics
# ---------------------------------------------------------------------------

def field_metrics(mat, ref_phase, halfwidth=0.15, sd_thresh=2.0, min_amp=0.5):
    """Per cell, from a (n_cells, n_seg, bins) matrix in noise units and each cell's reference
    preferred phase:
      amplitude   : in-field minus out-of-field mean of the trial-averaged curve
      reliability : fraction of segments with an in-field transient (in-field peak bin above the
                    segment's out-of-field mean + sd_thresh * SD AND above min_amp noise units)
      gain        : in-field peak minus out-of-field mean, on segments with a transient
      jitter      : circular SD of the per-segment peak phase relative to ref_phase, on
                    segments with a transient
      offset      : circular mean of that per-segment peak phase minus ref_phase (signed)
      reliability_null : the same reliability with the field rotated by half a cycle -- the
                    false-positive floor
    """
    n_cells, n_seg, bins = mat.shape
    centres = (np.arange(bins) + 0.5) / bins

    def reliability_for(phase):
        in_field = circular_distance(centres[None], phase[:, None]) <= halfwidth
        peak = np.where(in_field[:, None, :], mat, np.nan)
        out = np.where(~in_field[:, None, :], mat, np.nan)
        peak_v = np.nanmax(peak, axis=2)
        out_m, out_sd = np.nanmean(out, axis=2), np.nanstd(out, axis=2)
        transient = (peak_v > out_m + sd_thresh * out_sd) & (peak_v > min_amp)
        return in_field, peak_v, out_m, transient

    in_field, peak_v, out_m, transient = reliability_for(ref_phase)
    curve = mean_curve(mat)
    amplitude = (np.nanmean(np.where(in_field, curve, np.nan), 1) - np.nanmean(np.where(~in_field, curve, np.nan), 1))
    gain = np.nanmean(np.where(transient, peak_v - out_m, np.nan), axis=1)
    seg_phase = (np.nanargmax(np.nan_to_num(gaussian_filter1d(np.nan_to_num(mat), 1, axis=2, mode='wrap'),
                                            nan=-np.inf), axis=2) + 0.5) / bins
    rel = (seg_phase - ref_phase[:, None] + 0.5) % 1 - 0.5
    jitter = np.array([circular_std(rel[c][transient[c]]) for c in range(n_cells)])
    offset = np.array([(circular_mean(rel[c][transient[c]]) + 0.5) % 1 - 0.5 if transient[c].any() else np.nan
                       for c in range(n_cells)])
    _, _, _, transient_null = reliability_for((ref_phase + 0.5) % 1)
    return pd.DataFrame(dict(amplitude=amplitude, reliability=transient.mean(1), gain=gain, jitter=jitter,
                             offset=offset, reliability_null=transient_null.mean(1)))


def event_rate(dFn, thresh=3.0):
    """Fraction of frames above `thresh` noise units -- overall activity level, comparable across
    days because it is in noise units."""
    return np.mean(dFn > thresh, axis=1)


# ---------------------------------------------------------------------------
# S4: sequence scores
# ---------------------------------------------------------------------------

def segment_sequence_scores(mat, template_phase, min_active=5, sd_thresh=2.0, min_amp=0.5):
    """For each segment, circular correlation between the active cells' peak phases and their
    template phases (cells active = segment peak above out-of-segment mean + sd_thresh*SD and
    min_amp). Returns (scores, n_active) per segment."""
    n_cells, n_seg, bins = mat.shape
    sm = gaussian_filter1d(np.nan_to_num(mat), 1, axis=2, mode='wrap')
    peak_bin = np.argmax(sm, axis=2)
    peak_v = np.take_along_axis(sm, peak_bin[..., None], 2)[..., 0]
    active = (peak_v > np.nanmean(mat, 2) + sd_thresh * np.nanstd(mat, 2)) & (peak_v > min_amp)
    peak_phase = (peak_bin + 0.5) / bins
    scores, n_active = np.full(n_seg, np.nan), active.sum(0)
    for s in range(n_seg):
        if n_active[s] >= min_active:
            scores[s] = circular_corr(peak_phase[active[:, s], s], template_phase[active[:, s]])
    return scores, n_active


# ---------------------------------------------------------------------------
# Decoders (P1-P3)
# ---------------------------------------------------------------------------

def downsample(x, factor):
    """Mean over consecutive blocks of `factor` frames along the last axis."""
    n = x.shape[-1] // factor
    return x[..., :n * factor].reshape(x.shape[:-1] + (n, factor)).mean(-1)


def frame_phase(n_frames, segments):
    """Per-frame GP phase in [0, 1) and segment id (-1 / NaN outside segments)."""
    phase, seg_id = np.full(n_frames, np.nan), np.full(n_frames, -1)
    for i, (s, e) in enumerate(np.asarray(segments, int)):
        if e > s:
            phase[s:e] = (np.arange(s, e) - s) / (e - s)
            seg_id[s:e] = i
    return phase, seg_id


def decoding_data(loaded, rows=None, clock='reward', bin_s=0.2, n_phase_bins=10, n_pos_bins=20):
    """Downsampled population (dFz, rows = cells to use) with phase/position labels.
    Returns dict(X (n_bins, n_cells), frame (centre frame of each bin), phase, phase_bin, seg,
    pos, pos_bin, lap, speed, factor)."""
    fr, n = loaded['frame_rate'], loaded['n_frames']
    factor = max(1, int(round(bin_s * fr)))
    session = loaded['session']
    dFz = loaded['dFz'] if rows is None else loaded['dFz'][rows]
    X = np.nan_to_num(downsample(dFz, factor)).T
    centre = np.arange(X.shape[0]) * factor + factor // 2
    phase, seg = frame_phase(n, loaded['segments'][clock])
    pos = np.asarray(session['position'][:n]) % session['tunnel_length'] / session['tunnel_length']
    return dict(X=X, frame=centre, phase=phase[centre], seg=seg[centre],
                phase_bin=np.floor(np.nan_to_num(phase[centre]) * n_phase_bins).astype(int),
                pos=pos[centre], pos_bin=np.floor(pos[centre] * n_pos_bins).astype(int),
                lap=np.asarray(session['lap_idx'][:n])[centre], speed=np.asarray(session['speed'][:n])[centre],
                factor=factor, n_phase_bins=n_phase_bins, n_pos_bins=n_pos_bins, frame_rate=fr)


def _lda():
    return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')


def posterior_to_phase(post, n_bins):
    """Posterior-weighted circular mean over bin centres."""
    centres = (np.arange(n_bins) + 0.5) / n_bins
    return circular_mean(np.broadcast_to(centres, post.shape), axis=1, weights=post)


def full_posterior(model, X, classes, n_bins):
    post = np.zeros((X.shape[0], n_bins))
    post[:, classes] = model.predict_proba(X)
    return post


def cv_decode(X, y, groups, n_bins, mask, n_splits=5):
    """Grouped-CV posteriors (n, n_bins) for the samples in `mask` (others NaN)."""
    post = np.full((len(y), n_bins), np.nan)
    idx = np.where(mask)[0]
    n_splits = min(n_splits, len(np.unique(groups[idx])))
    for train, test in GroupKFold(n_splits=n_splits).split(idx, groups=groups[idx]):
        model = _lda().fit(X[idx[train]], y[idx[train]])
        post[idx[test]] = full_posterior(model, X[idx[test]], model.classes_, n_bins)
    return post


def cv_decode_all(dd, target='phase', n_splits=5):
    """CV posteriors for every downsampled bin: labelled bins get out-of-fold predictions, bins
    outside any segment get the model fit on all labelled bins (P3 needs the whole session)."""
    y, groups, n_bins, mask = _target(dd, target)
    post = cv_decode(dd['X'], y, groups, n_bins, mask, n_splits)
    model = _lda().fit(dd['X'][mask], y[mask])
    post[~mask] = full_posterior(model, dd['X'][~mask], model.classes_, n_bins)
    return post


def _target(dd, target):
    if target == 'phase':
        return dd['phase_bin'], dd['seg'], dd['n_phase_bins'], dd['seg'] >= 0
    if target == 'position':
        return dd['pos_bin'], dd['lap'], dd['n_pos_bins'], np.ones(len(dd['pos_bin']), bool)
    raise ValueError(target)


def decode_error(post, truth, n_bins):
    """Circular error (fraction of a cycle, 0-0.5) of the posterior-mean phase."""
    return circular_distance(posterior_to_phase(post, n_bins), truth)


def chance_error(post, truth, n_bins, n_perm=20, seed=0):
    """Error after permuting predictions across samples (keeps the prediction distribution)."""
    rng = np.random.default_rng(seed)
    pred = posterior_to_phase(post, n_bins)
    return np.nanmean([np.nanmean(circular_distance(rng.permutation(pred), truth)) for _ in range(n_perm)])


def within_session_decoding(dd, n_splits=5):
    """Median CV error (and chance) for GP phase and lap position, plus a speed-only control
    decoder of GP phase."""
    out = {}
    for target, truth in [('phase', dd['phase']), ('position', dd['pos'])]:
        y, groups, n_bins, mask = _target(dd, target)
        post = cv_decode(dd['X'], y, groups, n_bins, mask, n_splits)
        err = decode_error(post[mask], truth[mask], n_bins)
        out[target] = dict(post=post, error=np.nanmedian(err), chance=chance_error(post[mask], truth[mask], n_bins),
                           truth=truth[mask], pred=posterior_to_phase(post[mask], n_bins))
    y, groups, n_bins, mask = _target(dd, 'phase')
    post = cv_decode(dd['speed'][:, None], y, groups, n_bins, mask, n_splits)
    out['phase_speed_only'] = np.nanmedian(decode_error(post[mask], dd['phase'][mask], n_bins))
    return out


def cross_session_decoding(dd_train, dd_test, target='phase'):
    """Train on all labelled bins of one session, posteriors for every bin of another (same cells,
    same order). Returns (post, error over labelled test bins)."""
    y, _, n_bins, mask = _target(dd_train, target)
    model = _lda().fit(dd_train['X'][mask], y[mask])
    post = full_posterior(model, dd_test['X'], model.classes_, n_bins)
    _, _, _, test_mask = _target(dd_test, target)
    truth = dd_test['phase'] if target == 'phase' else dd_test['pos']
    return post, np.nanmedian(decode_error(post[test_mask], truth[test_mask], n_bins))


def event_aligned_posterior(post, dd, event_frames, window=(-2, 4)):
    """Mean posterior (n_t, n_bins) around event frames, in downsampled bins."""
    step = dd['factor'] / dd['frame_rate']
    lags = np.arange(int(np.floor(window[0] / step)), int(np.ceil(window[1] / step)))
    idx = np.asarray(event_frames, int) // dd['factor']
    idx = idx[(idx + lags[0] >= 0) & (idx + lags[-1] < post.shape[0])]
    if not len(idx):
        return None, lags * step, 0
    return np.nanmean(post[idx[:, None] + lags[None]], axis=0), lags * step, len(idx)


def inter_landmark_lick_bouts(loaded, min_gap_s=2.0):
    """Lick-bout onsets between landmarks (lm_idx == 0), >= min_gap_s after the previous onset --
    licking without a landmark, a control for licking-related signals in FA alignments."""
    session, n = loaded['session'], loaded['n_frames']
    onsets = np.where(lick_onset_mask(session['licks'][:n]))[0]
    keep = np.r_[True, np.diff(onsets) >= min_gap_s * loaded['frame_rate']]
    onsets = onsets[keep]
    return onsets[np.asarray(session['lm_idx'])[onsets] == 0]


# ---------------------------------------------------------------------------
# P4 / P5: subspace projection and population similarity
# ---------------------------------------------------------------------------

def fit_subspace(ref_curves, n_components=10):
    """PCA with GP bins as samples and cells as features. ref_curves: list of (n_cells, bins)
    z-scored trial-averaged curves (same cells), concatenated along bins."""
    X = np.concatenate([np.nan_to_num(c).T for c in ref_curves], axis=0)
    return PCA(n_components=min(n_components, X.shape[0] - 1, X.shape[1])).fit(X)


def project(pca, curves):
    return pca.transform(np.nan_to_num(curves).T)


def alignment_index(pca, curves, k=3):
    """Variance of `curves` captured by the reference top-k PCs relative to its own top-k PCs."""
    X = np.nan_to_num(curves).T
    X = X - X.mean(0)
    ref_var = np.sum(np.var(X @ pca.components_[:k].T, axis=0))
    own_var = np.sum(np.sort(np.linalg.eigvalsh(np.cov(X.T)))[::-1][:k])
    return ref_var / own_var


def pv_similarity(curves_a, curves_b):
    """Mean over GP bins of the population-vector correlation between two (n_cells, bins) curves."""
    a, b = zscore_rows(curves_a), zscore_rows(curves_b)
    return np.nanmean([np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(a.shape[1])
                       if np.isfinite(a[:, i]).all() and np.isfinite(b[:, i]).all()])


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def style_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)


def heatmap_row(axes, curves_by_tag, title_fn, vmin=None, vmax=None, cmap=HEATMAP_CMAP):
    """One heatmap per axis from {tag: (n_cells, bins)} curves; grey for NaN rows."""
    cmap = cmap.copy()
    cmap.set_bad('0.85')
    im = None
    for ax, (tag, curves) in zip(axes, curves_by_tag.items()):
        im = ax.imshow(curves, aspect='auto', cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax,
                       extent=[0, 1, curves.shape[0], 0])
        ax.set_title(title_fn(tag), fontsize=9)
        ax.set_xlabel('GP phase')
    return im


def mean_sem(x, axis=0):
    x = np.asarray(x, float)
    n = np.sum(np.isfinite(x), axis=axis)
    return np.nanmean(x, axis=axis), np.nanstd(x, axis=axis) / np.sqrt(np.maximum(n, 1))

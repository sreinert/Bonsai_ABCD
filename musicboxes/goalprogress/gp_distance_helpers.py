"""Does a goal-progress (GP) cell's amplitude scale with the distance to the next reward? -- across cohorts.

Backs musicboxes/goalprogress/GP_amplitude_distance.ipynb. Within a sequence session the goal-to-goal segments
have different lengths (e.g. TAA0000066 t10: 6, 8, 9 and 7 landmarks; holography t7: 5, 6 and 7), and between
sequences they change again. GP cells keep their preferred phase across segments; the question here is whether
their amplitude at that phase grows or shrinks with segment length.

Design
------
  - Segments: correct transitions only (reward at goal X, and the next licked landmark is the next goal Y), as in
    GP_posterplots_panel1 (calc_gp_tuning_corr_trans). A segment type is the goal pair (X, Y). Its distance is the
    median distance actually travelled over its transitions (one fixed task distance, since correct transitions
    contain no detours), and its duration the median time.
  - Preferred phase: ONE phase per cell, the centre of the von Mises fit over all of the session's correct
    transitions (gp_learning_helpers.gp_tuning -> gp_circular_fit.vm_tuning, the default GP method in this repo).
    The amplitude of every segment type is read out at that same phase: the mean of the cell's trial-averaged curve
    for that type (dFn, noise units) over bins within AMP_HALFWIDTH of the phase. The out-of-field baseline (bins
    >= OUT_DIST from the phase) is kept alongside, to tell a field that grows from a cell whose activity rises
    everywhere.
  - Units: dFn (dF divided by a robust noise estimate, median/MAD, as in gp_learning_helpers.load_session), so
    cells and sessions with different raw dF scales are comparable without a per-session z-score.
  - Two cohorts, one format: load_cohort2_session turns a cohort-2 img_session_data.pkl into the same dict that
    gp_learning_helpers.load_session returns for the holography cohort (dF, dFn, session, frame_rate, n_frames,
    mouse, tag, neurons), so tuning, caching and amplitudes run unchanged on both.
  - Tracking: cohort 2 uses the ROICaT aligned-ROI matrix (one row per protocol t0..t17, values = dF row or NaN, as
    in GP_posterplots_panel1); the holography cohort uses its tracking cell matrix
    (gp_learning_helpers.tracked_rows), linked pairwise so a cell tracked in only some sessions still counts.
"""

import os
import pickle

import numpy as np
import pandas as pd

from ceph_paths import ceph_path
import cellTV.cellTV_functions_goalprogress as cellTV_gp
import preprocessing.parse_session_functions as parse_session_functions_c2
import musicboxes.goalprogress.gp_learning_helpers as h

COHORT2_ROOT = ceph_path('/Volumes/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2')
COHORT2_ROICAT = 'ROICaT/alignment/t0_t17/roicat_aligned_ROIs_t0_t17.npy'   # rows = protocols t0 ... t17
AMP_HALFWIDTH = 0.05   # phase bins within this of the preferred phase make up the "in-field" amplitude
OUT_DIST = 0.25        # bins at least this far from the preferred phase make up the baseline


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _noise_units(dF):
    med = np.nanmedian(dF, axis=1, keepdims=True)
    noise = 1.4826 * np.nanmedian(np.abs(dF - med), axis=1, keepdims=True)
    return (dF - med) / noise


def load_cohort2_session(mouse, stage, root=COHORT2_ROOT, cache_dir=None):
    """One cohort-2 session (stage = protocol, e.g. 't10') from its img_session_data.pkl, as a
    gp_learning_helpers.load_session-style dict. The pickle is ~330 MB on the share; with cache_dir only what is
    needed (dF as float32, the session dict and the correct transitions) is pickled locally. 'trans' holds
    preprocessing.parse_session_functions.get_corr_transitions' correct_transitions (from goal, to goal, start
    frame, end frame)."""
    cache = os.path.join(cache_dir, f'c2_{mouse}_{stage}.pkl') if cache_dir else None
    if cache and os.path.exists(cache):
        with open(cache, 'rb') as f:
            out = pickle.load(f)
    else:
        folder = [p for p in os.listdir(f'{root}/{mouse}') if p.endswith(f'protocol-{stage}')]
        if len(folder) != 1:
            raise FileNotFoundError(f'{mouse} {stage}: {len(folder)} folders end in protocol-{stage}')
        with open(f'{root}/{mouse}/{folder[0]}/img_session_data.pkl', 'rb') as f:
            d = pickle.load(f)
        session = parse_session_functions_c2.get_corr_transitions(dict(d['session']))
        n = min(d['dF'].shape[1], len(session['position']))
        out = dict(mouse=mouse, tag=stage, cohort='2', dF=np.asarray(d['dF'][:, :n], np.float32), session=session,
                   frame_rate=d['frame_rate'], n_frames=n, neurons=np.arange(d['dF'].shape[0]),
                   trans=np.asarray(session['correct_transitions']))
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache, 'wb') as f:
                pickle.dump(out, f)
    out['dFn'] = _noise_units(out['dF'].astype(float))
    return out


def shaping_transitions(loaded):
    """Correct transitions of a shaping session (several goal slots that all pay out, one entry in goal_ids):
    consecutive rewards whose goal landmarks follow each other in running order (no goal visit missed in between)
    and with no licked non-goal landmark in between (FA, anticipatory or post-goal), i.e. the next licked landmark
    after the reward is the next goal, as in the sequence criterion. get_goal_progress_transitions returns none here:
    get_shaping_transitions works from goal_ids, which lists only the first goal."""
    session, vt = loaded['session'], loaded['visits']
    seg = h.reward_segments(loaded)
    if not len(seg):
        return np.empty((0, 4), int)
    earned = h.goal_visit_for_rewards(session)
    rewards = np.asarray(session['rewards'])
    visit_of = dict(zip(rewards.tolist(), earned.tolist()))
    slots, licked = vt.slot.to_numpy(), vt.licked.to_numpy()
    is_goal = vt.is_goal.to_numpy()
    out = []
    for s, e in seg:
        a, b = visit_of.get(int(s)), visit_of.get(int(e))
        if a is None or b is None or b <= a:
            continue
        between = np.arange(a + 1, b)
        if is_goal[between].any() or licked[between].any():
            continue
        out.append((slots[a], slots[b], s, e))
    return np.asarray(out, int).reshape(-1, 4)


def load_holography_session(mouse, tag, cache_dir=None):
    """gp_learning_helpers.load_session plus 'trans': the correct transitions (from goal, to goal, start frame,
    end frame). Sequence sessions: cellTV_functions_goalprogress.get_goal_progress_transitions(correct_only=True),
    the same transitions as gp_learning_helpers.correct_segments. Shaping sessions: shaping_transitions."""
    L = h.load_session(mouse, tag, cache_dir=cache_dir)
    if len(L['session']['goal_ids']) > 1:
        session = dict(L['session'])
        cellTV_gp.get_goal_progress_transitions(session, correct_only=True)
        trans = np.asarray(session['correct_transitions']).reshape(-1, 4)
    else:
        trans = shaping_transitions(L)
    L.update(cohort='holo', trans=trans)
    return L


# ---------------------------------------------------------------------------
# Segment types and their distances
# ---------------------------------------------------------------------------

def cumulative_distance(session, n):
    """Distance run since the session start, per frame: session['total_dist'] where it exists (cohort 2), else
    the unwrapped corridor position (a step of more than half a corridor length is a wrap). Not lap index *
    length + position: lap_idx does not switch exactly where the position wraps, which gives negative distances."""
    if 'total_dist' in session:
        return np.asarray(session['total_dist'][:n], float)
    tl = session['tunnel_length']
    step = np.diff(np.asarray(session['position'][:n], float) % tl)
    step = np.where(step < -tl / 2, step + tl, np.where(step > tl / 2, step - tl, step))
    return np.r_[0, np.cumsum(step)]


def transition_table(loaded):
    """One row per correct transition, sorted by start frame and without duplicate (start, end) pairs: from / to
    goal, start / end frame, segment type 'from->to', distance run, duration (s). Also sets loaded['segments_ct']
    (start, end) in the same order, the segments every GP fit and amplitude in this module uses."""
    trans = np.asarray(loaded['trans'])
    trans = trans[(trans[:, 3] > trans[:, 2]) & (trans[:, 3] <= loaded['n_frames'])]
    _, first = np.unique(trans[:, 2:4].astype(int), axis=0, return_index=True)
    trans = trans[np.sort(first)]
    trans = trans[np.argsort(trans[:, 2], kind='stable')]
    s, e = trans[:, 2].astype(int), trans[:, 3].astype(int)
    cum = cumulative_distance(loaded['session'], loaded['n_frames'])
    lm = np.asarray(loaded['session']['landmarks'], float)
    spacing = np.median(np.diff(lm[:, 0])) if len(lm) > 1 else np.nan
    tab = pd.DataFrame(dict(from_goal=trans[:, 0].astype(int), to_goal=trans[:, 1].astype(int), start=s, end=e,
                            distance=cum[np.minimum(e, len(cum) - 1)] - cum[s], duration=(e - s) / loaded['frame_rate']))
    tab['type'] = tab.from_goal.astype(str) + '->' + tab.to_goal.astype(str)
    tab['distance_lm'] = tab.distance / spacing
    loaded['segments_ct'] = tab[['start', 'end']].to_numpy()
    return tab


def type_table(tab):
    """Per segment type: number of transitions, median distance (VR units and landmark spacings), median duration (s)
    and median speed (distance / duration)."""
    g = tab.groupby('type', sort=False)
    out = pd.DataFrame(dict(n=g.size(), distance=g.distance.median(), distance_lm=g.distance_lm.median(),
                            duration=g.duration.median(), speed=(g.distance.median() / g.duration.median())))
    return out.sort_values('distance')


# ---------------------------------------------------------------------------
# Amplitude at the preferred phase, per segment type
# ---------------------------------------------------------------------------

def phase_masks(phase, n_bins, halfwidth=AMP_HALFWIDTH, out_dist=OUT_DIST):
    """In-field (within halfwidth of the preferred phase; at least the nearest bin) and out-of-field (at least
    out_dist away) bin masks, (cells, n_bins) each."""
    centres = (np.arange(n_bins) + 0.5) / n_bins
    d = h.circular_distance(centres[None], np.asarray(phase)[:, None])
    infield = (d <= halfwidth) | (d == d.min(1, keepdims=True))
    return infield, d >= out_dist


def type_amplitudes(mat, phase, types, type_order, halfwidth=AMP_HALFWIDTH, out_dist=OUT_DIST):
    """mat: (cells, segments, bins) binned dFn (gp_tuning's 'mat'), phase: (cells,) preferred phases, types:
    (segments,) segment-type label of each segment, type_order: the types to return. Each type's trial-averaged
    curve is read at the cell's ONE preferred phase. Returns dict of (cells, n_types) arrays: peak (mean in-field),
    baseline (mean out-of-field), amp (peak - baseline)."""
    infield, outfield = phase_masks(phase, mat.shape[2], halfwidth, out_dist)
    types = np.asarray(types)
    curves = np.stack([np.nanmean(mat[:, types == t], axis=1) for t in type_order], axis=1)   # cells, types, bins
    mean_where = lambda m: np.nansum(curves * m[:, None], -1) / m.sum(1)[:, None]
    peak, base = mean_where(infield), mean_where(outfield)
    return dict(peak=peak, baseline=base, amp=peak - base)


def normalise_rows(v):
    """Each cell's values divided by its largest value (cells whose largest value is <= 0 become NaN)."""
    m = np.nanmax(v, axis=1, keepdims=True)
    return np.where(m > 0, v / np.where(m > 0, m, 1), np.nan)


def row_slopes(y, x):
    """OLS slope of each row of y (cells, n) on x ((n,) or (cells, n)), ignoring NaN; NaN with < 2 points."""
    x = np.broadcast_to(np.asarray(x, float), y.shape)
    ok = np.isfinite(y) & np.isfinite(x)
    n = ok.sum(1)
    xm = np.where(ok, x, 0).sum(1) / np.maximum(n, 1)
    ym = np.where(ok, y, 0).sum(1) / np.maximum(n, 1)
    dx, dy = np.where(ok, x - xm[:, None], 0), np.where(ok, y - ym[:, None], 0)
    den = (dx ** 2).sum(1)
    return np.where((n >= 2) & (den > 0), (dx * dy).sum(1) / np.where(den > 0, den, 1), np.nan)


def shuffled_slopes(mat, phase, types, type_order, log_dist, n_shuffles=1000, measure='peak', seed=0):
    """Null for the per-cell slopes of normalised amplitude on log2 distance: segment-type labels permuted across
    segments (each type keeps its number of segments), amplitudes and slopes recomputed. Returns (n_shuffles,
    cells)."""
    rng = np.random.default_rng(seed)
    types = np.asarray(types)
    out = np.empty((n_shuffles, mat.shape[0]))
    for i in range(n_shuffles):
        a = type_amplitudes(mat, phase, rng.permutation(types), type_order)[measure]
        out[i] = row_slopes(normalise_rows(a), log_dist)
    return out


# ---------------------------------------------------------------------------
# Tracking across sessions
# ---------------------------------------------------------------------------

def cohort2_track_table(mouse, stages, root=COHORT2_ROOT):
    """ROICaT-tracked cells of a cohort-2 mouse: DataFrame (one row per tracked cell, one column per stage) of dF
    rows, -1 where the cell was not found; only cells found in >= 2 of `stages`."""
    roicat = np.load(f'{root}/{mouse}/{COHORT2_ROICAT}')
    rows = roicat[[int(s.lstrip('t')) for s in stages]].T
    rows = np.where(np.isfinite(rows), rows, -1).astype(int)
    return pd.DataFrame(rows[(rows >= 0).sum(1) >= 2], columns=list(stages))


def holography_track_table(cm, loaded_by_tag, stages):
    """Holography-cohort cells tracked in >= 2 of `stages`, from pairwise matches (gp_learning_helpers.tracked_rows
    on every pair of stages), joined into one identity per cell. Cells whose pairwise matches conflict (one ROI
    linked to two different ROIs of another stage) are dropped. DataFrame as in cohort2_track_table."""
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    stages = [s for s in stages if s in loaded_by_tag]
    for i, a in enumerate(stages):
        for b in stages[i + 1:]:
            rows = h.tracked_rows(cm, loaded_by_tag, [a, b])
            for ra, rb in ([] if rows is None else rows):
                parent[find((a, int(ra)))] = find((b, int(rb)))
    groups = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    out = []
    for members in groups.values():
        stage_of = [m[0] for m in members]
        if len(set(stage_of)) < 2 or len(stage_of) != len(set(stage_of)):
            continue
        row = {s: -1 for s in stages}
        row.update({s: r for s, r in members})
        out.append(row)
    return pd.DataFrame(out, columns=stages)

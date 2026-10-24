"""Cohort-agnostic overview plots for a single 2P imaging + VR-ABCD session.

Produces:
  1. Mean image with ROI mask outlines, restricted to accepted cells (iscell/Accepted == True)
  2. Raw dF/F traces for a handful of cells over a window of frames (visual transient check)
  3. Peri-reward dF/F heatmap across all cells (z-scored per cell, sorted by peak time)
  4. Same, but cross-validated: sort order computed on even-numbered reward trials,
     applied to a heatmap built from odd-numbered trials
  5. Position-tuning heatmap across all cells (z-scored per cell, sorted by peak position
     bin), plain and even/odd-lap cross-validated
  6. Goal-progress tuning heatmap, pooled across all A->B/B->C/... segments, plain and
     even/odd-transition cross-validated -- auto-detects sequence (>1 rewarded landmark,
     correct-transitions-verified segments) vs shaping (exactly 1 rewarded landmark,
     consecutive-rewarded-visit segments) sessions from ses_settings, via
     cellTV/cellTV_functions_goalprogress.py. Sequence sessions additionally get a
     per-segment-type breakdown (one heatmap panel per A->B, B->C, ..., all sorted on the first).
  7. Behavioural-performance summary (cohort='3' and '2nwb') -- see
     musicboxes/behaviour_performance_summary.py: two combined figures (processed
     frame-aligned vs raw) covering lick/reward maps, transition matrices, and (for shaping
     sessions) peri-reward lick/speed traces.

Plots are saved by default to behav_root/analysis/sub-mouse/session (see
behaviour_performance_summary.default_save_dir); pass save_dir=False to run_overview to
display only, or a path to save elsewhere.

Mirrors musicboxes/recording_overview_cohort3.ipynb's session-loading and mean-image/
mask-overlay conventions, and reuses its cellTV.extract_reward_tuning/extract_position_tuning
helpers -- deliberately not reproducing that notebook's behavioural ethogram plots, per
request (the goal-progress tuning above is a new addition, not from that notebook).

Supports cohort='3' (cohort3 and GoalProgress-style sessions -- NWB-based DfOverF +
ImageSegmentation) and cohort='2' (older, pre-NWB sessions -- suite2p .npy +
behaviour_data.npz read directly, mirroring preprocessing/parse_session_functions_cohort2.py
and cellTV/cellTV_functions_cohort2.py).

NOTE: the cohort='2' path has NOT been run against real cohort2 data -- only cohort='3'
(cohort3 and GoalProgress) has been verified. Check its output carefully before trusting
it for a cohort2 session. Also note parse_session_functions_cohort2.find_base_path has a
project root hardcoded inside it (unlike cohort3's, which takes root as an argument), so
`behav_root` is ignored for cohort='2'.

cohort='2nwb' is for cohort2 sessions rebuilt by nwb_barcoding/build_session_nwb_cohort2.py:
imaging from the merged NWB as for cohort='3', behaviour from the robust behaviour_data.npz
plus the raw session's VR config.yaml (see cohort2_nwb_session). behav_root is the cohort2
project's BehaviourAligned_robust folder; pass the ses-XXX prefix as session_id, since
find_base_path matches substrings (session_id "t1" would also match t10-t17). Position
tuning is skipped for the one-lap linear corridors (t0-t6), and goal-progress tuning for
sessions without a goal identity (t0-t2). The behavioural-performance summary uses
behaviour_performance_summary.run_behaviour_summary_cohort2 (skipped for t0).

Run in the `bonsai_abcd` conda env (same as the rest of this repo -- needs aeon, seaborn, etc).

Command-line usage -- run as a module from the repo root so the repo imports resolve
(`python musicboxes/session_overview.py ...` fails on `import parse_nwb_functions`):

    conda activate bonsai_abcd
    cd .../Code/Bonsai_ABCD

    # cohort3 sequence session
    python -m musicboxes.session_overview 010 full037 full037 \\
        /Volumes/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3/preprocessed_behav_Nov2025/derivatives \\
        --root-path /Volumes/mrsic_flogel/public/projects/SuKuSaRe_20250923_HFScohort3

    # GoalProgress session (--world random for random-world sessions; --show displays
    # each figure as it is built)
    python -m musicboxes.session_overview 002 t2 t2 \\
        /Volumes/mrsic_flogel/public/projects/SaReMaRa_20260629_GoalProgress/BehaviourAligned_robust \\
        --world random --show

    # cohort2 session rebuilt by nwb_barcoding (pass the ses-XXX prefix as session_id)
    python -m musicboxes.session_overview TAA0000066 ses-XXX t6 \\
        /Volumes/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/BehaviourAligned_robust \\
        --cohort 2nwb

Positional arguments are mouse, session_id, stage, behav_root; see `--help` for the
options. Call run_overview() directly for the tuning parameters the CLI doesn't expose.
"""

import contextlib
import io as io_
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from pynwb import NWBHDF5IO

import parse_nwb_functions
import preprocessing.parse_session_functions_cohort3 as parse_session_functions_cohort3
import cellTV.cellTV_functions as cellTV
import cellTV.cellTV_functions_goalprogress as cellTV_gp
import era_brewer
from ceph_paths import ceph_path
import musicboxes.behaviour_performance_summary as behaviour_performance_summary

# Continuous colormap for the z-scored heatmaps (plots 3/4/5), built from
# era_brewer's 'NineteenEightyNine' palette. Reversed so high values are red/warm and
# low values are blue/cool.
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    'NineteenEightyNine_r',
    era_brewer.era_brew('NineteenEightyNine', n=256, brew_type='continuous')[::-1])


def _quiet():
    """cellTV.extract_reward_tuning/extract_position_tuning print a couple of lines on
    every call; looping over all cells would otherwise flood stdout with nothing new."""
    return contextlib.redirect_stdout(io_.StringIO())


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------

def _load_nwb_imaging(session_path):
    """dF (all ROIs x valid frames), accepted-ROI indices, mean image, per-ROI masks and
    empirical frame rate from a session folder's merged NWB."""
    dF_all, neurons = parse_session_functions_cohort3.load_dF(session_path)

    nwb_path = list(session_path.glob('*.nwb'))[0]
    with NWBHDF5IO(str(nwb_path), mode='r') as io:
        nwb = io.read()
        # key names vary by which pipeline wrote the NWB: cohort3's native pipeline uses
        # the Chan1Plane0-suffixed names; nwb_barcoding's (stock neuroconv
        # add_segmentation_to_nwbfile) uses the plain, unsuffixed ones.
        seg_images = nwb.processing['ophys'].data_interfaces['SegmentationImages'].images
        try:
            mean_img = seg_images['MeanImageChan1Plane0'][:]
        except KeyError:
            mean_img = seg_images['MeanImage'][:]
        try:
            plane_seg = nwb.processing['ophys'].data_interfaces['ImageSegmentation'] \
                .plane_segmentations['PlaneSegmentationChan1Plane0'][:]
        except KeyError:
            plane_seg = nwb.processing['ophys'].data_interfaces['ImageSegmentation'] \
                .plane_segmentations['PlaneSegmentation'][:]
        image_mask_all = plane_seg['image_mask']
        # empirical, not a hardcoded nominal value -- see nwb_barcoding's barcode_alignment.py
        # for why that distinction matters in this codebase.
        funcimg_timestamps = nwb.acquisition['vDAQ_TTLsTable'][:]['timestamp'].values
        frame_rate = 1 / np.median(np.diff(funcimg_timestamps))

    return dF_all, neurons, mean_img, image_mask_all, frame_rate


def _cohort2_goal_ids(goal_odours, n_landmarks):
    """Goal identities (odour ids) in cycle order from the config's per-goal odour list:
    the shortest prefix that tiles the whole list. None when goal identity isn't defined --
    no goals (t0) or every landmark rewarded (t1-t2)."""
    if len(goal_odours) == 0 or len(goal_odours) == n_landmarks:
        return None
    for k in range(1, len(goal_odours) + 1):
        if all(goal_odours[i] == goal_odours[i % k] for i in range(len(goal_odours))):
            return list(goal_odours[:k])


def cohort2_nwb_session(mouse, session_id, stage, session_path, world='stable'):
    """Behavioural session struct for a cohort2 session rebuilt by
    nwb_barcoding/build_session_nwb_cohort2.py, in the cohort3 field layout the plotting and
    goal-progress code expects.

    Reads the robust behaviour_data.npz (frame-binned, same convention as cohort3) from
    session_path, and the VR config.yaml from the matching raw session folder
    (<project>/<mouse>/<session>/behav/<yymmdd>/config.yaml, where session_path is
    <project>/BehaviourAligned_robust/sub-<mouse>/<session>).

    Landmark visits are contiguous runs of frames inside one landmark's [start, end]
    interval; each visit's identity is that landmark's odour (flip_tunnel.landmarks_sequence),
    so the ABCD loop (t7+, one odour per slot) and the long linear t3-t6 corridors are handled
    alike. Fields set: position, speed, licks/thresholded_licks (per frame), licks_idx
    (thresholded lick frames), rewards, landmarks,
    num_landmarks, tunnel_length, lap_idx, num_laps, all_lms, lm_idx (per-frame visit id,
    1-indexed), licked_lms/rewarded_lms (flat, per visit), goal_ids (None if undefined).
    """
    import yaml

    session_path = Path(session_path)
    raw_session = session_path.parents[2] / mouse / session_path.name
    config_path = sorted((raw_session / 'behav').glob('*/config.yaml'))[0]
    with open(config_path) as f:
        options = yaml.safe_load(f)
    ft = options['flip_tunnel']

    data = np.load(session_path / 'behaviour_data.npz')
    position = data['position']
    speed = data['speed']
    licks = data['licks']  # per frame, as in cohort3
    rewards = np.where(data['rewards'])[0]
    lick_threshold = ft.get('speed_limit')
    licks_idx = np.where(licks > 0)[0]
    if lick_threshold is not None:
        licks_idx = licks_idx[speed[licks_idx] < lick_threshold]
    thresholded_licks = np.zeros(len(licks))
    thresholded_licks[licks_idx] = licks[licks_idx]

    landmarks = np.array(ft.get('landmarks') or [], dtype=float).reshape(-1, 2)
    lm_odours = np.array(ft.get('landmarks_sequence') or [], dtype=int)
    tunnel_length = ft['length']

    # laps: the position wraps back to the start of the corridor (loop corridors, t7+); the
    # long linear corridors (t0-t6) never wrap, so they're one lap.
    wraps = np.where(np.diff(position) < -tunnel_length / 2)[0] + 1
    lap_idx = np.zeros(len(position), dtype=int)
    for w in wraps:
        lap_idx[w:] += 1
    num_laps = len(wraps) + 1

    # per-frame landmark slot (-1 = between landmarks), then split into visits. Position jitter
    # at a landmark edge makes the mouse leave and re-enter the same landmark; those runs are
    # merged into one visit as long as no other landmark or lap boundary comes in between.
    slot = np.full(len(position), -1)
    for j, (start, end) in enumerate(landmarks):
        slot[(position >= start) & (position <= end)] = j
    inside = np.where(slot >= 0)[0]
    new_visit = np.ones(len(inside), dtype=bool)
    new_visit[1:] = (slot[inside[1:]] != slot[inside[:-1]]) | (lap_idx[inside[1:]] != lap_idx[inside[:-1]])
    visit_of_frame = np.cumsum(new_visit)  # 1-indexed visit id for each frame in `inside`
    lm_idx = np.zeros(len(position), dtype=int)
    lm_idx[inside] = visit_of_frame
    all_lms = lm_odours[slot[inside[new_visit]]]
    licked_lms = np.zeros(len(all_lms), dtype=int)
    licked_lms[np.unique(lm_idx[licks_idx][lm_idx[licks_idx] > 0]) - 1] = 1
    rewarded_lms = np.zeros(len(all_lms), dtype=int)
    rewarded_lms[np.unique(lm_idx[rewards][lm_idx[rewards] > 0]) - 1] = 1
    run_start = inside[new_visit]

    goal_odours = [int(lm_odours[np.where((landmarks == np.asarray(g, dtype=float)).all(axis=1))[0][0]])
                   for g in (ft.get('goals') or [])]
    goal_ids = _cohort2_goal_ids(goal_odours, len(landmarks))

    n_outside = int(np.sum(lm_idx[rewards] == 0))
    print(f"cohort2 session: {num_laps} lap(s), {len(run_start)} landmark visits, "
          f"{len(rewards)} rewards ({n_outside} outside a landmark visit), goal_ids={goal_ids}")

    return dict(position=position, speed=speed, total_dist=data['distance'], licks=licks,
                thresholded_licks=thresholded_licks, licks_idx=licks_idx, rewards=rewards, landmarks=landmarks,
                num_landmarks=len(landmarks), tunnel_length=tunnel_length,
                lick_threshold=lick_threshold, lap_idx=lap_idx, num_laps=num_laps,
                # lap states (state_id) aren't computed here; laps_needed=1 tells
                # cellTV.extract_position_tuning to skip its per-state averages (plot-only)
                laps_needed=1,
                all_lms=all_lms, lm_idx=lm_idx, licked_lms=licked_lms,
                rewarded_lms=rewarded_lms, goal_ids=goal_ids,
                mouse=mouse, session_id=session_id, stage=stage, world=world)

def load_session_data(mouse, session_id, stage, behav_root=None, cohort='3', world='stable',
                       stage_type='sequences', restrict_to_accepted=True):
    """Load everything the plotting functions below need for one session.

    :param cohort: '3' for cohort3/GoalProgress-style sessions, '2nwb' for cohort2 sessions
        rebuilt into BehaviourAligned_robust, '2' for cohort2-style (see module docstring --
        cohort='2' is best-effort/unverified).
    :param stage_type: which session-analysis wrapper to use for cohort='3' -- 'sequences'
        (stage > T6, parse_session_functions_cohort3.analyse_npz_sequences, matches
        recording_overview_cohort3.ipynb) or 'pre7' (T3-T6, analyse_npz_pre7). Ignored
        for cohort='2' (always uses parse_session_functions_cohort2.analyse_npz).
    :param restrict_to_accepted: if True (default), dF and image_mask are both restricted
        to accepted cells (iscell/Accepted == True) and re-indexed 0..n_accepted-1, so
        every plotting function below only ever sees identified cells.

    Returns a dict: dF (n_cells x n_frames), neurons (accepted-cell indices into the
    ORIGINAL, unfiltered ROI set), session (behavioural struct, frame-indexed to match
    dF), mean_img, image_mask (per-ROI boolean masks, pandas Series matching dF's cell
    axis if restrict_to_accepted), frame_rate (Hz, empirically derived), cohort.
    """
    if cohort == '3':
        if behav_root is None:
            raise ValueError("behav_root is required for cohort='3'")
        session_path = Path(parse_nwb_functions.find_base_path(mouse, session_id, behav_root))

        if stage_type == 'sequences':
            session = parse_session_functions_cohort3.analyse_npz_sequences(
                mouse, session_id, behav_root, stage, world=world)
        elif stage_type == 'pre7':
            session = parse_session_functions_cohort3.analyse_npz_pre7(
                mouse, session_id, behav_root, stage, world=world)
        else:
            raise ValueError(f"Unknown stage_type {stage_type!r}, expected 'sequences' or 'pre7'")

        dF_all, neurons, mean_img, image_mask_all, frame_rate = _load_nwb_imaging(session_path)

    elif cohort == '2nwb':
        if behav_root is None:
            raise ValueError("behav_root is required for cohort='2nwb'")
        session_path = Path(parse_nwb_functions.find_base_path(mouse, session_id, behav_root))
        session = cohort2_nwb_session(mouse, session_id, stage, session_path, world=world)
        dF_all, neurons, mean_img, image_mask_all, frame_rate = _load_nwb_imaging(session_path)

    elif cohort == '2':
        import preprocessing.parse_session_functions_cohort2 as parse_session_functions_cohort2
        import cellTV.cellTV_functions_cohort2 as cellTV_cohort2

        session = parse_session_functions_cohort2.analyse_npz(mouse, session_id, stage, plot=False)
        base_path = parse_session_functions_cohort2.find_base_path(mouse, session_id)
        f, fneu, iscell, ops, seg, frame_rate = cellTV_cohort2.load_img_data(base_path)
        frame_ix = parse_session_functions_cohort2.load_session_npz(base_path)
        dF_all = cellTV_cohort2.get_dff(f, fneu, frame_ix, ops)
        neurons = np.where(iscell[:, 0] == 1)[0]
        mean_img = ops['meanImg']
        # cohort2 masks are one labelled outline image (label n+1 == cell n), not per-ROI
        # boolean arrays -- convert to the same per-ROI Series format plot_fov_masks expects.
        image_mask_all = pd.Series([(seg['outlines'] == (n + 1)) for n in range(len(iscell))])

    else:
        raise ValueError(f"Unknown cohort {cohort!r}, expected '2', '2nwb' or '3'")

    if restrict_to_accepted:
        dF = dF_all[neurons, :]
        image_mask = image_mask_all.iloc[neurons].reset_index(drop=True)
    else:
        dF = dF_all
        image_mask = image_mask_all.reset_index(drop=True)

    return dict(dF=dF, neurons=neurons, session=session, mean_img=mean_img,
                image_mask=image_mask, frame_rate=frame_rate, cohort=cohort)


# ---------------------------------------------------------------------------
# Plot 1 -- FOV with accepted-cell mask outlines
# ---------------------------------------------------------------------------

def plot_fov_masks(mean_img, image_mask, title='Field of view', save_path=None):
    """Mean image with ROI outlines overlaid. Pass an already accepted-cell-only
    `image_mask` (the default from load_session_data) to only show identified cells."""
    masks = cellTV.concat_masks(image_mask)
    mask_bool = (masks != 0).astype(float)

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(mean_img.T, cmap='gray')
    plt.imshow(masks.T, alpha=mask_bool.T, cmap='rainbow')
    plt.title(f"{title} (n={len(image_mask)} accepted cells)")
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Plot 2 -- raw dF/F traces over a window of frames
# ---------------------------------------------------------------------------

def plot_raw_traces(dF, n_cells=20, start_frame=0, n_frames=20000, cell_indices=None,
                     frame_rate=None, offset_step=6.0, save_path=None):
    """Stacked raw dF/F traces for a subset of cells over a window of frames, to visually
    check for calcium transients.

    :param cell_indices: explicit cells to plot; if None, `n_cells` evenly spaced across
        the full population are chosen (so a quick look covers the whole FOV, not just
        the first few ROIs).
    :param offset_step: vertical spacing between traces, in units of each trace's own
        std (traces are normalized by their own std first, so cells are comparably visible
        regardless of absolute dF/F amplitude).
    """
    n_total_cells, n_total_frames = dF.shape
    end_frame = min(start_frame + n_frames, n_total_frames)
    if cell_indices is None:
        cell_indices = np.unique(np.linspace(0, n_total_cells - 1, min(n_cells, n_total_cells)).astype(int))

    window = dF[cell_indices][:, start_frame:end_frame]
    scale = np.nanstd(window, axis=1, keepdims=True)
    scale[scale == 0] = 1
    normalized = window / scale

    time = np.arange(start_frame, end_frame)
    xlabel = 'Frame'
    if frame_rate:
        time = time / frame_rate
        xlabel = 'Time (s)'

    fig = plt.figure(figsize=(14, 0.35 * len(cell_indices) + 2))
    for i, cell in enumerate(cell_indices):
        plt.plot(time, normalized[i] + i * offset_step, lw=0.6, color='k')
    plt.yticks(np.arange(len(cell_indices)) * offset_step, [f"cell {c}" for c in cell_indices])
    plt.xlabel(xlabel)
    plt.title(f"Raw dF/F, {len(cell_indices)} cells, frames {start_frame}-{end_frame}")
    plt.xlim(time[0], time[-1])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Shared machinery for plots 3/4 (reward) and 5 (position): build a per-cell
# (trials x window) matrix, then z-score/sort/display -- optionally cross-validated.
# ---------------------------------------------------------------------------

def reward_tuning_matrix(dF, session, frame_rate, window_size=(-5, 5), quiet=True):
    """(n_cells, n_trials, n_window) array via cellTV.extract_reward_tuning, called once
    per cell (all cells share the same reward-trial set and window)."""
    n_cells = dF.shape[0]
    # extract_reward_tuning uses frame_rate*window_size directly as np.arange bounds for
    # frame indexing, so it needs an integer (matches how it's always been called
    # elsewhere in this codebase, with a hardcoded frame_rate=45) -- an empirically precise
    # but non-integer rate (e.g. 44.64 Hz) would otherwise raise inside np.arange/indexing.
    frame_rate_int = int(round(frame_rate))
    if abs(frame_rate_int - frame_rate) > 0.1:
        print(f"  rounding frame_rate {frame_rate:.3f} -> {frame_rate_int} Hz for frame-index math")
    ctx = _quiet() if quiet else contextlib.nullcontext()
    with ctx:
        first = cellTV.extract_reward_tuning(dF, 0, session, frame_rate=frame_rate_int,
                                              window_size=list(window_size), plot=False)
        mat = np.zeros((n_cells,) + first.shape)
        mat[0] = first
        for c in range(1, n_cells):
            mat[c] = cellTV.extract_reward_tuning(dF, c, session, frame_rate=frame_rate_int,
                                                   window_size=list(window_size), plot=False)
    return mat


def position_tuning_matrix(dF, session, bins=200, quiet=True):
    """(n_cells, n_laps, n_bins) array via cellTV.extract_position_tuning, called once per cell."""
    n_cells = dF.shape[0]
    ctx = _quiet() if quiet else contextlib.nullcontext()
    with ctx:
        first, bin_edges = cellTV.extract_position_tuning(dF, 0, session, bins=bins, plot=False)
        mat = np.zeros((n_cells,) + first.shape)
        mat[0] = first
        for c in range(1, n_cells):
            mat[c], _ = cellTV.extract_position_tuning(dF, c, session, bins=bins, plot=False)
    return mat, bin_edges


def _zscore_sort_plot(mat_for_sort, mat_for_display, x_values, xlabel, title,
                       event_x=None, save_path=None, drop_axis='cells'):
    """Average each (n_cells, n_trials, n_window) array over its trial axis, z-score per
    cell, sort rows by the argmax of `mat_for_sort`'s per-cell curve, and display
    `mat_for_display` (same order) as a heatmap. Pass the same array for both to get a
    plain (non-cross-validated) sort+display; pass even/odd-trial subsets for a
    cross-validated one.

    :param drop_axis: how to handle NaNs (e.g. a cell with too few valid trials, or --
        for position tuning -- a position bin that's undersampled across ALL laps for at
        least one cell). 'cells' (default, used for reward tuning) drops whole cells;
        'bins' drops whole columns instead, which is the right call when a handful of
        bins are chronically empty (e.g. right at a lap-position wraparound) rather than
        genuinely reflecting bad cells -- matches recording_overview_cohort3.ipynb's own
        `zsc_av_first_half[:, ~np.isnan(...).any(axis=0)]` approach for position tuning.
    """
    sort_curve = stats.zscore(np.nanmean(mat_for_sort, axis=1), axis=1, nan_policy='omit')
    display_curve = stats.zscore(np.nanmean(mat_for_display, axis=1), axis=1, nan_policy='omit')

    if drop_axis == 'cells':
        valid = ~np.isnan(sort_curve).any(axis=1) & ~np.isnan(display_curve).any(axis=1)
        n_dropped = (~valid).sum()
        if n_dropped:
            print(f"  dropping {n_dropped} cell(s) with all-NaN tuning (e.g. too few valid trials)")
        sort_curve, display_curve = sort_curve[valid], display_curve[valid]
    elif drop_axis == 'bins':
        good_bins = ~np.isnan(sort_curve).any(axis=0) & ~np.isnan(display_curve).any(axis=0)
        n_dropped = (~good_bins).sum()
        if n_dropped:
            print(f"  dropping {n_dropped} bin(s) undersampled for at least one cell")
        sort_curve, display_curve = sort_curve[:, good_bins], display_curve[:, good_bins]
        x_values = np.asarray(x_values)[good_bins]
    else:
        raise ValueError(f"Unknown drop_axis {drop_axis!r}, expected 'cells' or 'bins'")

    order = np.argsort(np.nanargmax(sort_curve, axis=1))

    fig = plt.figure(figsize=(8, 6))
    extent = [x_values[0], x_values[-1], sort_curve.shape[0], 0]
    plt.imshow(display_curve[order], aspect='auto', cmap=HEATMAP_CMAP, interpolation='none', extent=extent)
    plt.colorbar(label='z-scored dF/F')
    if event_x is not None:
        plt.axvline(event_x, color='white', linestyle='--', lw=1)
    plt.xlabel(xlabel)
    plt.ylabel(f'Cells (sorted, n={sort_curve.shape[0]})')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Plots 3 & 4 -- peri-reward heatmap, plain and even/odd cross-validated
# ---------------------------------------------------------------------------

def plot_reward_heatmap(dF, session, frame_rate, window_size=(-5, 5), cross_validated=False,
                         save_path=None):
    """Plot 3 (cross_validated=False): all cells' peri-reward dF/F, z-scored per cell,
    sorted by their own peak time within the window.

    Plot 4 (cross_validated=True): sort order computed from even-numbered reward trials,
    applied to a heatmap built from odd-numbered trials.
    """
    mat = reward_tuning_matrix(dF, session, frame_rate, window_size=window_size)
    n_window = mat.shape[2]
    x_values = np.linspace(window_size[0], window_size[1], n_window)

    if cross_validated:
        sort_mat, display_mat = mat[:, 0::2, :], mat[:, 1::2, :]
        title = 'Peri-reward tuning (sorted on even trials, shown on odd trials)'
    else:
        sort_mat = display_mat = mat
        title = 'Peri-reward tuning (all cells)'

    return _zscore_sort_plot(sort_mat, display_mat, x_values, 'Time from reward (s)', title,
                              event_x=0, save_path=save_path)


# ---------------------------------------------------------------------------
# Plot 5 -- position-tuning heatmap, plain and even/odd-lap cross-validated
# ---------------------------------------------------------------------------

def plot_position_heatmap(dF, session, bins=200, cross_validated=False, save_path=None):
    """Plot 5 (cross_validated=False): all cells' position tuning, z-scored per cell,
    sorted by their own peak position bin.

    Cross-validated version (cross_validated=True): sort order computed from even-numbered
    laps, applied to a heatmap built from odd-numbered laps.
    """
    mat, bin_edges = position_tuning_matrix(dF, session, bins=bins)
    # bin 0 is unreliable by construction (np.digitize edge behaviour right at the
    # lap-start position subtraction in extract_position_tuning) -- same reason
    # recording_overview_cohort3.ipynb drops it via av_fr[:, 1:] before z-scoring.
    mat, x_values = mat[:, :, 1:], bin_edges[1:-1]

    if cross_validated:
        sort_mat, display_mat = mat[:, 0::2, :], mat[:, 1::2, :]
        title = 'Position tuning (sorted on even laps, shown on odd laps)'
    else:
        sort_mat = display_mat = mat
        title = 'Position tuning (all cells)'

    # fewer laps go into each half here (esp. cross-validated), so more bins can end up
    # undersampled for at least one cell -- drop those bins, not whole cells.
    return _zscore_sort_plot(sort_mat, display_mat, x_values, 'Position (cm)', title,
                              save_path=save_path, drop_axis='bins')


# ---------------------------------------------------------------------------
# Goal-progress tuning -- pooled (plain + even/odd cross-validated), and, for
# sequence sessions only, split by individual segment (A->B, B->C, ...)
# ---------------------------------------------------------------------------

def goal_progress_tuning_matrix(dF, session, bins=90, quiet=True):
    """(n_cells, n_transitions, bins) array via
    cellTV_functions_goalprogress.extract_goal_progress_pooled, called once per cell.
    Requires session['correct_transitions'] to already be set -- see
    cellTV_functions_goalprogress.get_goal_progress_transitions()."""
    n_cells = dF.shape[0]
    ctx = _quiet() if quiet else contextlib.nullcontext()
    with ctx:
        first = cellTV_gp.extract_goal_progress_pooled(dF, 0, session, bins=bins)
        mat = np.full((n_cells,) + first.shape, np.nan)
        mat[0] = first
        for c in range(1, n_cells):
            mat[c] = cellTV_gp.extract_goal_progress_pooled(dF, c, session, bins=bins)
    return mat


def plot_goal_progress_heatmap(dF, session, bins=90, cross_validated=False, save_path=None):
    """All cells' goal-progress tuning, pooled across every segment type/goal-transition
    together, z-scored per cell and sorted by peak phase. Works for both sequence sessions
    (correct-transitions-verified segments) and shaping sessions (consecutive-rewarded-visit
    segments) -- same session['correct_transitions'] interface either way.

    Cross-validated version: sort order computed from even-numbered transitions, applied to
    a heatmap built from odd-numbered transitions.
    """
    mat = goal_progress_tuning_matrix(dF, session, bins=bins)
    x_values = np.arange(bins)

    if cross_validated:
        if mat.shape[1] < 2:
            raise ValueError(
                f"Cannot cross-validate goal-progress tuning: only {mat.shape[1]} valid "
                "segment(s) found (need >= 2 to split into even/odd halves).")
        sort_mat, display_mat = mat[:, 0::2, :], mat[:, 1::2, :]
        title = 'Goal-progress tuning (sorted on even transitions, shown on odd transitions)'
    else:
        sort_mat = display_mat = mat
        title = f'Goal-progress tuning (all cells, pooled across {mat.shape[1]} segments)'

    return _zscore_sort_plot(sort_mat, display_mat, x_values, 'Goal-progress phase (bin)', title,
                              save_path=save_path, drop_axis='bins')


def plot_goal_progress_by_segment(dF, session, bins=90, save_path=None, quiet=True):
    """Sequence sessions only: one heatmap panel per segment type (starting-goal identity,
    e.g. A->B, B->C, ...), each z-scored per cell. Cells are sorted by their peak phase in
    the FIRST segment type and that same order is applied to every other panel, so the
    panels can be compared row-by-row (does a cell keep its preferred phase across segments?).
    """
    transitions = np.asarray(session['correct_transitions'])
    goals = np.unique(transitions[:, 0].astype(int))
    n_cells = dF.shape[0]

    ctx = _quiet() if quiet else contextlib.nullcontext()
    per_goal_mats = {}
    with ctx:
        for goal in goals:
            mats = []
            for c in range(n_cells):
                segment_session = dict(session)  # shallow copy -- only swapping correct_transitions
                segment_session['correct_transitions'] = transitions[transitions[:, 0].astype(int) == goal]
                mats.append(cellTV_gp.extract_goal_progress_pooled(dF, c, segment_session, bins=bins))
            per_goal_mats[int(goal)] = np.stack(mats)  # (n_cells, n_trials_of_this_type, bins)

    fig, axes = plt.subplots(1, len(goals), figsize=(5 * len(goals), 6), squeeze=False)
    axes = axes[0]
    curves = {}
    for goal in goals:
        curve = stats.zscore(np.nanmean(per_goal_mats[int(goal)], axis=1), axis=1, nan_policy='omit')
        curves[int(goal)] = curve[:, ~np.isnan(curve).any(axis=0)]
    order = np.argsort(np.nanargmax(curves[int(goals[0])], axis=1))  # shared order from first segment

    im = None
    for ax, goal in zip(axes, goals):
        im = ax.imshow(curves[int(goal)][order], aspect='auto', cmap=HEATMAP_CMAP, interpolation='none')
        ax.set_title(f'From goal {goal} (n={per_goal_mats[int(goal)].shape[1]} trials)')
        ax.set_xlabel('Goal-progress phase (bin)')
    axes[0].set_ylabel(f'Cells (sorted on goal {goals[0]} segment)')
    fig.colorbar(im, ax=list(axes), label='z-scored dF/F', fraction=0.02)
    fig.suptitle('Goal-progress tuning, split by segment')
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def compute_goal_progress_tuning_indices(dF, session, bins=90, quiet=True):
    """Per-cell goal-progress tuning index and preferred phase for every cell (see
    cellTV_functions_goalprogress.calc_goal_progress_tuning). Used to select "high-tuning"
    cells for plot_goal_progress_heatmap_high_tuning/plot_goal_progress_polar_high_tuning.
    """
    n_cells = dF.shape[0]
    tuning = np.zeros(n_cells)
    phase_pref = np.zeros(n_cells, dtype=int)
    ctx = _quiet() if quiet else contextlib.nullcontext()
    with ctx:
        for c in range(n_cells):
            _, phase_pref[c], tuning[c] = cellTV_gp.calc_goal_progress_tuning(dF, c, session, bins=bins)
    return tuning, phase_pref


def _select_high_tuning_cells(tuning, top_percentile=80, tuning_index_threshold=None):
    """Shared cell-selection logic for the two high-tuning plots below.

    :param top_percentile: keep cells at/above this percentile of tuning index (default 80
        = top 20%). Ignored if tuning_index_threshold is given.
    :param tuning_index_threshold: absolute tuning-index cutoff; overrides top_percentile.
    """
    valid = ~np.isnan(tuning)
    threshold = (tuning_index_threshold if tuning_index_threshold is not None
                 else np.nanpercentile(tuning[valid], top_percentile))
    high_idx = np.where(valid & (tuning >= threshold))[0]
    label = f"tuning index >= {threshold:.2f} ({'given threshold' if tuning_index_threshold is not None else f'{top_percentile}th percentile'})"
    print(f"  {len(high_idx)}/{len(tuning)} cells at/above {label}")
    return high_idx, threshold


def plot_goal_progress_tuning_index_histogram(tuning, hist_bins=30, save_path=None):
    """Bar histogram of the goal-progress tuning index across ALL cells (not just
    high-tuning ones) -- gives a sense of the full distribution the high-tuning threshold
    in the other plots is being drawn from.
    """
    valid = tuning[~np.isnan(tuning)]
    fig = plt.figure(figsize=(6, 4))
    plt.hist(valid, bins=hist_bins, color='C0', edgecolor='white')
    plt.xlabel('Goal-progress tuning index')
    plt.ylabel('Cell count')
    plt.title(f'Goal-progress tuning index distribution (n={len(valid)} cells)')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_goal_progress_phase_pref_histogram(phase_pref, high_idx, bins=90, hist_bins=30, save_path=None):
    """Line histogram (counts plotted as a connected line, not bars) of preferred
    goal-progress phase, restricted to the high-tuning cells in `high_idx`.
    """
    counts, edges = np.histogram(phase_pref[high_idx], bins=hist_bins, range=(0, bins))
    centers = (edges[:-1] + edges[1:]) / 2

    fig = plt.figure(figsize=(6, 4))
    plt.plot(centers, counts, color='C1', marker='o', markersize=3)
    plt.xlabel('Preferred goal-progress phase (bin)')
    plt.ylabel('Cell count')
    plt.title(f'Preferred phase distribution, high-tuning cells (n={len(high_idx)})')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_goal_progress_heatmap_high_tuning(dF, session, bins=90, top_percentile=80,
                                            tuning_index_threshold=None, save_path=None,
                                            tuning=None):
    """Same cross-validated pooled goal-progress heatmap as plot_goal_progress_heatmap
    (sort on even transitions, display odd transitions), but restricted to cells with a
    high goal-progress tuning index (see _select_high_tuning_cells).

    :param tuning: precomputed tuning indices (see compute_goal_progress_tuning_indices),
        to avoid recomputing them if the caller already has them. Computed internally if
        not given.
    """
    if tuning is None:
        tuning, _ = compute_goal_progress_tuning_indices(dF, session, bins=bins)
    high_idx, _ = _select_high_tuning_cells(tuning, top_percentile, tuning_index_threshold)

    mat = goal_progress_tuning_matrix(dF[high_idx], session, bins=bins)
    x_values = np.arange(bins)
    if mat.shape[1] < 2:
        raise ValueError(
            f"Cannot cross-validate goal-progress tuning: only {mat.shape[1]} valid "
            "segment(s) found (need >= 2 to split into even/odd halves).")
    sort_mat, display_mat = mat[:, 0::2, :], mat[:, 1::2, :]
    title = f'Goal-progress tuning, high-tuning cells only (n={len(high_idx)}), even/odd CV'
    return _zscore_sort_plot(sort_mat, display_mat, x_values, 'Goal-progress phase (bin)', title,
                              save_path=save_path, drop_axis='bins')


def plot_goal_progress_polar_high_tuning(dF, session, bins=90, top_percentile=80,
                                          tuning_index_threshold=None, save_path=None, quiet=True,
                                          tuning=None):
    """Sequence sessions only: one polar-plot subplot per high-tuning cell, showing its
    goal-progress tuning curve concatenated across the full goal cycle (A->B->C->...->A).

    A transition type with zero valid entries in session['correct_transitions'] (e.g. under
    get_goal_progress_transitions(correct_only=True), if the mouse never completed that
    transition) shows as a flat-zero wedge rather than being omitted (see
    cellTV_functions_goalprogress.calc_goal_progress_polar_curve), so every subplot always
    spans the same full circle.

    :param tuning: precomputed tuning indices, see plot_goal_progress_heatmap_high_tuning.
    """
    if len(session['goal_ids']) <= 1:
        raise ValueError("plot_goal_progress_polar_high_tuning is only meaningful for sequence "
                          "sessions (session['goal_ids'] has > 1 entry) -- this is a shaping session.")

    if tuning is None:
        tuning, _ = compute_goal_progress_tuning_indices(dF, session, bins=bins)
    high_idx, _ = _select_high_tuning_cells(tuning, top_percentile, tuning_index_threshold)

    n_goals = len(session['goal_ids'])
    n_cols = min(5, len(high_idx)) or 1
    n_rows = max(1, int(np.ceil(len(high_idx) / n_cols)))

    angles = np.linspace(0, 2 * np.pi, bins * n_goals, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(3 * n_cols, 3 * n_rows))
    ctx = _quiet() if quiet else contextlib.nullcontext()
    with ctx:
        for i, cell in enumerate(high_idx):
            curve = cellTV_gp.calc_goal_progress_polar_curve(dF, cell, session, bins=bins)
            curve_closed = np.concatenate([curve, [curve[0]]])
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='polar')
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.plot(angles_closed, curve_closed, color='C1', lw=1.2)
            ax.fill(angles_closed, curve_closed, color='C1', alpha=0.15)
            ax.set_xticks(np.linspace(0, 2 * np.pi, n_goals, endpoint=False))
            ax.set_xticklabels([str(g) for g in session['goal_ids']], fontsize=7)
            ax.set_yticklabels([])
            ax.set_title(f'Cell {cell} (idx={tuning[cell]:.2f})', fontsize=9)

    fig.suptitle(f'Goal-progress tuning (polar), high-tuning cells (n={len(high_idx)})')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Goal-progress tuning section (plots 7-13) -- run once for "all" reward-to-reward
# transitions and once for "correct" ones only (see run_overview)
# ---------------------------------------------------------------------------

def _run_goal_progress_section(dF, session, is_sequence, bins, tuning_top_percentile, save_dir,
                                label, filename_suffix):
    """Runs the goal-progress tuning plots (7-13) against whatever transitions are currently
    in session['correct_transitions'] -- caller sets these via
    cellTV_gp.get_goal_progress_transitions first. `label` is a short human-readable tag used
    in print statements (e.g. 'all transitions' / 'correct transitions only'); `filename_suffix`
    tags the saved PNGs (e.g. '_all' / '_correct') so the two runs don't overwrite each other.
    Returns a figs dict with the same keys run_overview's other plots use, each suffixed with
    filename_suffix.
    """
    def _p(name):
        return str(save_dir / f"{name}{filename_suffix}.png") if save_dir else None

    figs = {}
    n_transitions = len(session['correct_transitions'])
    print(f"Plot 7 [{label}]: goal-progress tuning (pooled across {n_transitions} segments)...")
    figs[f'goal_progress{filename_suffix}'] = plot_goal_progress_heatmap(
        dF, session, bins=bins, cross_validated=False, save_path=_p('7_goal_progress'))

    print(f"Plot 8 [{label}]: goal-progress tuning (even/odd cross-validated)...")
    try:
        figs[f'goal_progress_cv{filename_suffix}'] = plot_goal_progress_heatmap(
            dF, session, bins=bins, cross_validated=True, save_path=_p('8_goal_progress_cv'))
    except ValueError as e:
        print(f"  skipping: {e}")
        figs[f'goal_progress_cv{filename_suffix}'] = None

    # computed once and reused by every plot below, instead of each one recomputing the
    # same per-cell tuning index/preferred phase from scratch
    tuning, phase_pref = compute_goal_progress_tuning_indices(dF, session, bins=bins)

    print(f"Plot 9 [{label}]: goal-progress tuning index histogram (all cells)...")
    figs[f'goal_progress_tuning_hist{filename_suffix}'] = plot_goal_progress_tuning_index_histogram(
        tuning, save_path=_p('9_goal_progress_tuning_hist'))

    high_idx, _ = _select_high_tuning_cells(tuning, tuning_top_percentile)

    print(f"Plot 10 [{label}]: preferred-phase histogram (high-tuning cells)...")
    figs[f'goal_progress_phase_hist{filename_suffix}'] = plot_goal_progress_phase_pref_histogram(
        phase_pref, high_idx, bins=bins, save_path=_p('10_goal_progress_phase_hist'))

    print(f"Plot 11 [{label}]: goal-progress tuning, high-tuning cells only (even/odd cross-validated)...")
    try:
        figs[f'goal_progress_cv_high_tuning{filename_suffix}'] = plot_goal_progress_heatmap_high_tuning(
            dF, session, bins=bins, top_percentile=tuning_top_percentile,
            tuning=tuning, save_path=_p('11_goal_progress_cv_high_tuning'))
    except ValueError as e:
        print(f"  skipping: {e}")
        figs[f'goal_progress_cv_high_tuning{filename_suffix}'] = None

    if is_sequence:
        print(f"Plot 12 [{label}]: goal-progress tuning, split by segment (sequence session)...")
        figs[f'goal_progress_by_segment{filename_suffix}'] = plot_goal_progress_by_segment(
            dF, session, bins=bins, save_path=_p('12_goal_progress_by_segment'))

        print(f"Plot 13 [{label}]: goal-progress tuning (polar), high-tuning cells (sequence session)...")
        figs[f'goal_progress_polar_high_tuning{filename_suffix}'] = plot_goal_progress_polar_high_tuning(
            dF, session, bins=bins, top_percentile=tuning_top_percentile,
            tuning=tuning, save_path=_p('13_goal_progress_polar_high_tuning'))

    return figs


# ---------------------------------------------------------------------------
# Orchestrator -- build all overview plots at once
# ---------------------------------------------------------------------------

def run_overview(mouse, session_id, stage, behav_root=None, cohort='3', world='stable',
                  stage_type='sequences', n_trace_cells=20, trace_start_frame=0,
                  trace_n_frames=20000, reward_window=(-5, 5), position_bins=200,
                  goal_progress_bins=90, min_correct_transitions=10,
                  tuning_top_percentile=80, root_path=None, save_dir=None, show=True):
    """Load one session and produce all overview plots. Returns the loaded session dict
    (in case you want to keep working with dF/session/etc afterward) plus the list of
    created figures.

    Goal-progress plots auto-detect session type from len(session['goal_ids']) (which
    tracks ses_settings['availableRewardPositions']): >1 rewarded landmark -> sequence
    task (cellTV_functions_goalprogress.get_corr_transitions_cohort3), plus a
    per-segment-type breakdown (A->B, B->C, ...) and a per-cell polar-plot breakdown for
    high-tuning cells; exactly 1 -> shaping session
    (cellTV_functions_goalprogress.get_shaping_transitions), pooled only (no per-segment or
    polar plots -- there's only one segment type).

    The goal-progress tuning plots (7-13, see _run_goal_progress_section) are produced TWICE
    by default: once using every reward-to-reward transition ('_all' filename suffix), and
    once restricted to "correct" transitions only (get_corr_transitions_cohort3/
    get_shaping_transitions -- the mouse's next lick after a reward must land on the correct
    next goal for sequence sessions, or the next landmark visit must also have been rewarded
    for shaping sessions; '_correct' filename suffix). The 'correct' run is skipped entirely
    (not just its cross-validated plots) if fewer than min_correct_transitions are found --
    with too few segments, even the non-cross-validated heatmap isn't meaningful, whereas the
    'all' run almost always has enough data to stay informative.

    :param min_correct_transitions: minimum number of correct-only transitions required to
        run that block at all (default 10); below this, only the 'all transitions' block is
        produced.
    :param tuning_top_percentile: for the two "high-tuning cells" plots, keep cells at/above
        this percentile of goal-progress tuning index (default 80 = top 20%).
    :param root_path: the project's own root folder (e.g. .../SuKuSaRe_20250923_HFScohort3),
        used only to build the default save_dir below -- NOT behav_root, which for
        cohort3-style dual-tree projects is already a deep subpath
        (.../preprocessed_behav_Nov2025/derivatives). Defaults to behav_root if not given
        (correct for GoalProgress-style single-tree projects).
    :param save_dir: where each figure (including the behavioural-performance summary, see
        below) is saved as a PNG. Defaults to
        behaviour_performance_summary.default_save_dir(root_path, mouse, session_id)
        (root_path/analysis/sub-mouse/session) -- pass save_dir=False to skip saving and only
        display.
    """
    if root_path is None:
        root_path = behav_root
    if save_dir is None:
        save_dir = behaviour_performance_summary.default_save_dir(root_path, mouse, session_id)
    elif save_dir is False:
        save_dir = None
    else:
        save_dir = Path(save_dir)
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    def _p(name):
        return str(save_dir / f"{name}.png") if save_dir else None

    data = load_session_data(mouse, session_id, stage, behav_root=behav_root, cohort=cohort,
                              world=world, stage_type=stage_type)
    dF, session, mean_img, image_mask = data['dF'], data['session'], data['mean_img'], data['image_mask']
    frame_rate = data['frame_rate']
    print(f"Loaded {dF.shape[0]} accepted cells, {dF.shape[1]} frames, {frame_rate:.2f} Hz")

    num_goals = len(session['goal_ids']) if session['goal_ids'] is not None else 0
    is_sequence = num_goals > 1
    n_steps = 7  # 1-6 below, plus the final behavioural-performance-summary step; the
                 # goal-progress section (7-13) runs twice (see _run_goal_progress_section)
                 # and is numbered/labelled separately since it's not a fixed step count

    figs = {}
    print(f"Plot 1/{n_steps}: FOV with accepted-cell masks...")
    figs['fov'] = plot_fov_masks(mean_img, image_mask,
                                  title=f"{mouse} {session_id} {stage}", save_path=_p('1_fov_masks'))

    print(f"Plot 2/{n_steps}: raw dF/F traces...")
    figs['raw_traces'] = plot_raw_traces(dF, n_cells=n_trace_cells, start_frame=trace_start_frame,
                                          n_frames=trace_n_frames, frame_rate=frame_rate,
                                          save_path=_p('2_raw_traces'))

    print(f"Plot 3/{n_steps}: peri-reward heatmap (all trials)...")
    figs['reward'] = plot_reward_heatmap(dF, session, frame_rate, window_size=reward_window,
                                          cross_validated=False, save_path=_p('3_reward_heatmap'))

    print(f"Plot 4/{n_steps}: peri-reward heatmap (even/odd cross-validated)...")
    figs['reward_cv'] = plot_reward_heatmap(dF, session, frame_rate, window_size=reward_window,
                                             cross_validated=True, save_path=_p('4_reward_heatmap_cv'))

    if session['num_laps'] > 1:
        print(f"Plot 5/{n_steps}: position-tuning heatmap (all laps)...")
        figs['position'] = plot_position_heatmap(dF, session, bins=position_bins,
                                                  cross_validated=False, save_path=_p('5_position_heatmap'))

        print(f"Plot 6/{n_steps}: position-tuning heatmap (even/odd cross-validated)...")
        figs['position_cv'] = plot_position_heatmap(dF, session, bins=position_bins,
                                                     cross_validated=True, save_path=_p('5b_position_heatmap_cv'))
    else:
        # cohort2 t0-t6 run a long linear corridor that never loops
        print(f"Plots 5-6/{n_steps}: skipped, position tuning needs more than one lap")

    if session['goal_ids'] is None:
        # cohort2 t0 (no landmarks) and t1-t2 (every landmark rewarded)
        print("Goal-progress tuning: skipped, this session has no goal identity")
        _run_behaviour_summary_if_supported(mouse, session_id, stage, behav_root, cohort, world,
                                            save_dir, show, n_steps)
        if show:
            plt.show()
        return data, figs

    print(f"Detecting goal-progress session type: {num_goals} rewarded landmark(s) "
          f"-> {'sequence' if is_sequence else 'shaping'} session")

    print("Goal-progress tuning, ALL transitions:")
    cellTV_gp.get_goal_progress_transitions(session, correct_only=False)
    n_all = len(session['correct_transitions'])
    print(f"  found {n_all} segment(s)")
    figs.update(_run_goal_progress_section(
        dF, session, is_sequence, goal_progress_bins, tuning_top_percentile, save_dir,
        label='all transitions', filename_suffix='_all'))

    print("Goal-progress tuning, CORRECT transitions only:")
    cellTV_gp.get_goal_progress_transitions(session, correct_only=True)
    n_correct = len(session['correct_transitions'])
    print(f"  found {n_correct} valid segment(s)")
    if n_correct < min_correct_transitions:
        print(f"  skipping this block: {n_correct} < min_correct_transitions={min_correct_transitions}")
    else:
        figs.update(_run_goal_progress_section(
            dF, session, is_sequence, goal_progress_bins, tuning_top_percentile, save_dir,
            label='correct transitions only', filename_suffix='_correct'))

    _run_behaviour_summary_if_supported(mouse, session_id, stage, behav_root, cohort, world,
                                        save_dir, show, n_steps)

    if show:
        plt.show()

    return data, figs


def _run_behaviour_summary_if_supported(mouse, session_id, stage, behav_root, cohort, world,
                                        save_dir, show, n_steps):
    if cohort == '3':
        print(f"Plot {n_steps}/{n_steps}: behavioural-performance summary (processed + raw)...")
        # save_dir is already resolved above (Path or None-meaning-skip) -- pass False rather
        # than None for "skip", since run_behaviour_summary treats a bare None as "compute your
        # own default" (for when it's called standalone, outside run_overview).
        try:
            behaviour_performance_summary.run_behaviour_summary(
                mouse, session_id, stage, behav_root, world=world,
                save_dir=save_dir if save_dir else False, show=show)
        except Exception as e:
            # This legacy behavioural-ethogram module is otherwise unrelated to the neural/
            # tuning plots above -- a failure here shouldn't take down an otherwise-successful
            # session_overview run.
            print(f"  behavioural-performance summary failed, skipping: {type(e).__name__}: {e}")
    elif cohort == '2nwb':
        print(f"Plot {n_steps}/{n_steps}: behavioural-performance summary (processed + raw)...")
        try:
            behaviour_performance_summary.run_behaviour_summary_cohort2(
                mouse, session_id, stage, behav_root,
                save_dir=save_dir if save_dir else False, show=show)
        except Exception as e:
            print(f"  behavioural-performance summary failed, skipping: {type(e).__name__}: {e}")
    else:
        print(f"Skipping behavioural-performance summary: not implemented for cohort={cohort!r}.")


def _cli():
    """Command-line entry point: load one session and build every overview plot (neural +
    behavioural), saved to root_path/analysis/sub-mouse/session by default. Uses plain
    argparse (not build_session_nwb.py's defopt) since this module runs in the bonsai_abcd
    conda env, which doesn't have defopt installed. Import the module and call run_overview()
    directly for the full set of tuning parameters this CLI doesn't expose."""
    import argparse
    parser = argparse.ArgumentParser(description=_cli.__doc__)
    parser.add_argument('mouse', help='e.g. "010"')
    parser.add_argument('session_id', help='session tag to match, e.g. "full037" or "t2"')
    parser.add_argument('stage', help='session stage label, e.g. "full037" or "t7"')
    parser.add_argument('behav_root', help='the project\'s behavioural-derivatives root, e.g. '
                         '".../SuKuSaRe_20250923_HFScohort3/preprocessed_behav_Nov2025/derivatives" '
                         'or ".../SaReMaRa_20260629_GoalProgress/BehaviourAligned_robust"')
    parser.add_argument('--root-path', default=None,
                         help='the project\'s own root folder, e.g. '
                         '".../SuKuSaRe_20250923_HFScohort3" -- used only for the default save '
                         'location; defaults to behav_root if not given')
    parser.add_argument('--cohort', default='3', choices=['2', '2nwb', '3'],
                         help='"3" (cohort3/GoalProgress-style, NWB-based, default), "2nwb" (cohort2 '
                         'rebuilt into BehaviourAligned_robust by build_session_nwb_cohort2.py) or '
                         '"2" (older, pre-NWB)')
    parser.add_argument('--world', default='stable', choices=['stable', 'random'])
    parser.add_argument('--show', action='store_true',
                         help='display each figure interactively as it\'s built '
                         '(default off, for unattended/batch terminal use)')
    args = parser.parse_args()
    run_overview(args.mouse, args.session_id, args.stage, behav_root=ceph_path(args.behav_root),
                 cohort=args.cohort, world=args.world,
                 root_path=ceph_path(args.root_path) if args.root_path else None, show=args.show)


if __name__ == '__main__':
    _cli()

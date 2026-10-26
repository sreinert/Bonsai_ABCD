"""Overview plots for a barcode-aligned 2P + sequence-compression session.

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
     per-segment-type breakdown (one heatmap panel per A->B, B->C, ...).
Behaviour parsing is delegated to the current modular
``analysis.sequence_compression.session_functions`` implementation. The raw Bonsai session
provides settings and landmark identities; ``behaviour_data.npz`` provides the
barcode-aligned, imaging-frame-indexed position, speed, lick, reward and photodiode arrays.
The old ``parse_session_functions_*`` modules are deliberately not used.

Pass the project root containing ``rawdata/cohort2`` and
``BehaviourAligned_robust/cohort2``. Sessions may be identified by their full directory
name or short name (for example ``full002``). Plots default to
``<project>/analysis/sub-<mouse>/<full-session>``; pass ``save_dir=False`` for display only.
"""

import contextlib
import io as io_
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

import cellTV.cellTV_functions_goalprogress as cellTV_gp
import era_brewer
from analysis.sequence_compression.session_functions import io as session_io
from analysis.sequence_compression.session_functions import session as session_models
from nwb_barcoding.sequence_compression import resolve_raw_session, short_session_name

# Continuous colormap for the z-scored heatmaps (plots 3/4/5), built from
# era_brewer's 'NineteenEightyNine' palette. Reversed so high values are red/warm and
# low values are blue/cool.
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    'NineteenEightyNine_r',
    era_brewer.era_brew('NineteenEightyNine', n=256, brew_type='continuous')[::-1])


def _quiet():
    """Silence verbose per-cell helpers while constructing population matrices."""
    return contextlib.redirect_stdout(io_.StringIO())


def _concat_masks(image_mask):
    """Combine per-ROI boolean masks into one labelled image."""
    masks = [np.asarray(mask, dtype=bool) for mask in image_mask]
    if not masks:
        raise ValueError("No ROI masks were found")
    combined = np.zeros(masks[0].shape, dtype=int)
    for index, mask in enumerate(masks, start=1):
        combined[mask] = index
    return combined


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------

def _load_nwb_imaging(session_path):
    """dF (all ROIs x valid frames), accepted-ROI indices, mean image, per-ROI masks and
    empirical frame rate from a session folder's merged NWB."""
    from pynwb import NWBHDF5IO

    session_path = Path(session_path)
    nwb_paths = sorted(session_path.glob('*.nwb'))
    if len(nwb_paths) != 1:
        raise ValueError(f"Expected exactly one merged NWB in {session_path}, found {nwb_paths}")
    nwb_path = nwb_paths[0]
    with NWBHDF5IO(str(nwb_path), mode='r') as io:
        nwb = io.read()
        ophys = nwb.processing['ophys'].data_interfaces

        dff_series = next(iter(ophys['DfOverF'].roi_response_series.values()))
        dff_on_disk = np.asarray(dff_series.data[:])

        # key names vary by which pipeline wrote the NWB: cohort3's native pipeline uses
        # the Chan1Plane0-suffixed names; nwb_barcoding's (stock neuroconv
        # add_segmentation_to_nwbfile) uses the plain, unsuffixed ones.
        seg_images = ophys['SegmentationImages'].images
        try:
            mean_img = seg_images['MeanImageChan1Plane0'][:]
        except KeyError:
            mean_img = seg_images['MeanImage'][:]
        try:
            plane_seg = ophys['ImageSegmentation'] \
                .plane_segmentations['PlaneSegmentationChan1Plane0'][:]
        except KeyError:
            plane_seg = ophys['ImageSegmentation'] \
                .plane_segmentations['PlaneSegmentation'][:]

        n_rois = len(plane_seg)
        if dff_on_disk.ndim != 2:
            raise ValueError(f"Expected a 2-D DfOverF array, got {dff_on_disk.shape}")
        if dff_on_disk.shape[1] == n_rois:
            dff_frames_by_roi = dff_on_disk
        elif dff_on_disk.shape[0] == n_rois:
            dff_frames_by_roi = dff_on_disk.T
        else:
            raise ValueError(
                f"Cannot match DfOverF shape {dff_on_disk.shape} to {n_rois} segmented ROIs"
            )

        valid_frames = np.load(session_path / 'valid_frames.npz')['valid_frames']
        if valid_frames.size == 0 or valid_frames.min() < 0 or valid_frames.max() >= len(dff_frames_by_roi):
            raise IndexError(
                f"valid_frames range is incompatible with DfOverF shape {dff_frames_by_roi.shape}"
            )
        dF_all = dff_frames_by_roi[valid_frames].T

        if 'Accepted' in plane_seg.columns:
            neurons = np.flatnonzero(np.asarray(plane_seg['Accepted']).astype(bool))
        else:
            neurons = np.arange(n_rois)
        image_mask_all = plane_seg['image_mask']
        # empirical, not a hardcoded nominal value -- see nwb_barcoding's barcode_alignment.py
        # for why that distinction matters in this codebase.
        funcimg_timestamps = nwb.acquisition['vDAQ_TTLsTable'][:]['timestamp'].values
        frame_rate = 1 / np.median(np.diff(funcimg_timestamps))

    return dF_all, neurons, mean_img, image_mask_all, frame_rate


def _build_aligned_session(parser, aligned_data, mouse, stage, full_session_name):
    """Apply the current ABCD_Session parser to imaging-frame-aligned arrays."""
    session = parser.create_session_struct()
    session.update(
        position=np.nan_to_num(aligned_data['position'], nan=0.0),
        speed=np.nan_to_num(aligned_data['speed'], nan=0.0),
        licks=np.asarray(aligned_data['licks']),
        rewards=np.flatnonzero(aligned_data['rewards']),
        pd1=np.asarray(aligned_data['pd1']),
        pd2=np.asarray(aligned_data['pd2']),
        mouse=mouse,
        stage=stage,
        session_id=short_session_name(full_session_name),
        world=parser.world,
        laps_needed=1,
    )

    session = parser.get_landmark_positions(session, data='odour')
    session = parser.get_goal_positions(session)
    session = parser.get_lap_idx(session)
    session = parser.get_lms_visited(session)
    session = parser.get_lm_idx(session)
    session = parser.get_licks_idx(session)
    session = parser.get_licks_per_lap(session)

    # ABCD_Session's historical helpers store these as lap x landmark matrices.
    # Goal-progress code needs one flat flag per visit, aligned with all_lms.
    n_visits = len(session['all_lms'])
    licked_lms = np.zeros(n_visits, dtype=int)
    rewarded_lms = np.zeros(n_visits, dtype=int)
    lick_visits = session['lm_idx'][session['licks_idx']].astype(int)
    reward_visits = session['lm_idx'][session['rewards']].astype(int)
    licked_lms[np.unique(lick_visits[lick_visits > 0]) - 1] = 1
    rewarded_lms[np.unique(reward_visits[reward_visits > 0]) - 1] = 1
    session['licked_lms'] = licked_lms
    session['rewarded_lms'] = rewarded_lms
    session['total_dist'] = session['position'] - session['position'][0]

    n_outside = int(np.sum(session['lm_idx'][session['rewards']] == 0))
    print(f"sequence-compression session: {n_visits} landmark visits, "
          f"{len(session['rewards'])} rewards ({n_outside} outside a visit), "
          f"goal_ids={session['goal_ids']}")
    return session


def load_session_data(mouse, session_id, stage, project_root, restrict_to_accepted=True):
    """Load one sequence-compression session using the modular session parser.

    ``project_root`` must contain ``rawdata/cohort2`` and
    ``BehaviourAligned_robust/cohort2``. ``session_id`` may be short or full; ambiguous
    matches are rejected by the same resolver used by the NWB build pipeline.
    """
    project_root = Path(project_root).expanduser().resolve()
    subject = mouse if str(mouse).startswith('sub-') else f'sub-{mouse}'
    mouse_id = subject.removeprefix('sub-')

    aligned_subject = project_root / 'BehaviourAligned_robust' / 'cohort2' / subject
    session_path = resolve_raw_session(aligned_subject, session_id)
    raw_session = project_root / 'rawdata' / 'cohort2' / subject / session_path.name
    if not raw_session.is_dir():
        raise FileNotFoundError(f"Matching raw session does not exist: {raw_session}")

    settings, rig_settings = session_io.load_settings(raw_session)
    dataframe = session_io.load_data(raw_session)
    parser = session_models.ABCD_Session(
        settings, rig_settings, dataframe, raw_session, mouse_id, stage,
        short_session_name(session_path.name),
    )
    with np.load(session_path / 'behaviour_data.npz') as aligned_data:
        session = _build_aligned_session(
            parser, aligned_data, mouse_id, stage, session_path.name
        )

    dF_all, neurons, mean_img, image_mask_all, frame_rate = _load_nwb_imaging(session_path)
    if dF_all.shape[1] != len(session['position']):
        raise ValueError(
            f"Aligned dF/F has {dF_all.shape[1]} frames but behavior has "
            f"{len(session['position'])}"
        )

    if restrict_to_accepted:
        dF = dF_all[neurons, :]
        image_mask = image_mask_all.iloc[neurons].reset_index(drop=True)
    else:
        dF = dF_all
        image_mask = image_mask_all.reset_index(drop=True)

    return dict(dF=dF, neurons=neurons, session=session, mean_img=mean_img,
                image_mask=image_mask, frame_rate=frame_rate,
                session_path=session_path, raw_session_path=raw_session,
                cohort='sequence-compression')


# ---------------------------------------------------------------------------
# Plot 1 -- FOV with accepted-cell mask outlines
# ---------------------------------------------------------------------------

def plot_fov_masks(mean_img, image_mask, title='Field of view', save_path=None):
    """Mean image with ROI outlines overlaid. Pass an already accepted-cell-only
    `image_mask` (the default from load_session_data) to only show identified cells."""
    masks = _concat_masks(image_mask)
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
    """Return cells x valid reward trials x peri-event frames without mutating session."""
    frame_rate_int = int(round(frame_rate))
    if abs(frame_rate_int - frame_rate) > 0.1:
        print(f"  rounding frame_rate {frame_rate:.3f} -> {frame_rate_int} Hz for frame-index math")
    rewards = np.asarray(session['rewards'], dtype=int)
    if rewards.size > 1:
        keep = np.ones(rewards.size, dtype=bool)
        keep[1:][np.diff(rewards) < 2 * frame_rate_int] = False
        rewards = rewards[keep]
    offsets = np.arange(
        int(round(window_size[0] * frame_rate_int)),
        int(round(window_size[1] * frame_rate_int)),
    )
    valid = rewards[(rewards + offsets[0] >= 0) & (rewards + offsets[-1] < dF.shape[1])]
    if valid.size == 0:
        raise ValueError("No rewards have a complete peri-event imaging window")
    indices = valid[:, None] + offsets[None, :]
    return dF[:, indices]


def position_tuning_matrix(dF, session, bins=200, quiet=True):
    """Return cells x laps x position bins using the aligned frame axis."""
    if session['num_laps'] <= 1:
        raise ValueError("Position tuning requires more than one lap")
    position = np.asarray(session['position'], dtype=float)
    finite = np.isfinite(position)
    bin_edges = np.linspace(np.nanmin(position[finite]), np.nanmax(position[finite]), bins + 1)
    mat = np.full((dF.shape[0], session['num_laps'], bins), np.nan)
    for lap in range(session['num_laps']):
        lap_frames = np.flatnonzero((session['lap_idx'] == lap) & finite)
        if not len(lap_frames):
            continue
        bin_index = np.digitize(position[lap_frames], bin_edges[1:-1])
        for bin_number in range(bins):
            frames = lap_frames[bin_index == bin_number]
            if len(frames):
                mat[:, lap, bin_number] = np.nanmean(dF[:, frames], axis=1)
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
    e.g. A->B, B->C, ...), each z-scored per cell and sorted by that panel's OWN peak phase
    (not a shared sort order -- this is about checking whether each individual segment type
    shows its own sequential tuning, not comparing them against a common order).
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
    im = None
    for ax, goal in zip(axes, goals):
        mat = per_goal_mats[int(goal)]
        curve = stats.zscore(np.nanmean(mat, axis=1), axis=1, nan_policy='omit')
        good_bins = ~np.isnan(curve).any(axis=0)
        curve = curve[:, good_bins]
        order = np.argsort(np.nanargmax(curve, axis=1))
        im = ax.imshow(curve[order], aspect='auto', cmap=HEATMAP_CMAP, interpolation='none')
        ax.set_title(f'From goal {goal} (n={mat.shape[1]} trials)')
        ax.set_xlabel('Goal-progress phase (bin)')
    axes[0].set_ylabel('Cells (sorted per-segment)')
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

def run_overview(mouse, session_id, stage, project_root,
                  n_trace_cells=20, trace_start_frame=0,
                  trace_n_frames=20000, reward_window=(-5, 5), position_bins=200,
                  goal_progress_bins=90, min_correct_transitions=10,
                  tuning_top_percentile=80, save_dir=None, show=True):
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
    :param project_root: sequence-compression project containing ``rawdata/cohort2`` and
        ``BehaviourAligned_robust/cohort2``.
    :param save_dir: where each figure is saved as a PNG. Defaults to
        ``project_root/analysis/sub-<mouse>/<full-session>``; pass False to skip saving.
    """
    data = load_session_data(mouse, session_id, stage, project_root)
    dF, session, mean_img, image_mask = data['dF'], data['session'], data['mean_img'], data['image_mask']
    frame_rate = data['frame_rate']
    print(f"Loaded {dF.shape[0]} accepted cells, {dF.shape[1]} frames, {frame_rate:.2f} Hz")

    if save_dir is None:
        subject = mouse if str(mouse).startswith('sub-') else f'sub-{mouse}'
        save_dir = Path(project_root) / 'analysis' / subject / data['session_path'].name
    elif save_dir is False:
        save_dir = None
    else:
        save_dir = Path(save_dir)
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    def _p(name):
        return str(save_dir / f"{name}.png") if save_dir else None

    num_goals = len(session['goal_ids']) if session['goal_ids'] is not None else 0
    is_sequence = num_goals > 1
    n_steps = 6

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
        print("Goal-progress tuning: skipped, this session has no goal identity")
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

    if show:
        plt.show()

    return data, figs


def _cli():
    """Load one aligned sequence-compression session and build its overview plots."""
    import argparse
    parser = argparse.ArgumentParser(description=_cli.__doc__)
    parser.add_argument('mouse', help='e.g. "02" or "sub-02"')
    parser.add_argument('session_id', help='full or short identifier, e.g. "full002"')
    parser.add_argument('stage', help='session stage label, e.g. "full002"')
    parser.add_argument('project_root', help='sequence-compression project root')
    parser.add_argument('--show', action='store_true',
                         help='display each figure interactively as it\'s built '
                         '(default off, for unattended/batch terminal use)')
    args = parser.parse_args()
    run_overview(args.mouse, args.session_id, args.stage, args.project_root, show=args.show)


if __name__ == '__main__':
    _cli()

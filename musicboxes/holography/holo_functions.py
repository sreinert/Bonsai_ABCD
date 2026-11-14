"""Analysis functions for holographic photostimulation sessions (musicboxes/holography/).

Used by session_overview_holography.py and meant for notebooks:

    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
    import musicboxes.holography.holo_functions as holo

Contents
--------
- Display helpers: mask outlines as an RGBA overlay, era_brewer colours.
- load_holo_session: dF/F (suite2p ROIs + projected source masks), stim table and alignment in one dict.
- psth / response_amplitude: trial-aligned dF/F with the stimulation frames masked out (the spiral
  scans leave a light artefact in the frames they overlap, so those frames are NaN by default).
- stim_triggered_movie (task item 5): crop-out movie of one stimulated cell, averaged over its trials.
- stim_cells_by_phase (task item 6): stimulated cells joined with the source session's gp_cell_table
  and sorted by preferred goal-progress phase.

Conventions: frames are indices into the holo suite2p traces (F.npy rows' columns); times are on the
vDAQ clock in seconds (see nwb_holography.py). 'cell' arguments are SOURCE ROI ids (= ensemble
number = gp_cell_table cell_id), because those are what the stimulation targeted.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from scipy import ndimage

import era_brewer

# stimulated + matched holo ROI, stimulated + projected source mask, other holo ROI, extra
ROLE_COLOURS = dict(zip(['matched', 'projected', 'other', 'extra'],
                        era_brewer.era_brew('Lover1', n=4, brew_type='discrete')))


# ---------------------------------------------------------------- display helpers

def outline_overlay(stat, shape, colour, alpha=1.0, width=1):
    """RGBA image (Ly, Lx, 4) with the outlines of the ROIs in `stat` in one colour, for
    ax.imshow on top of a mean image."""
    edge = np.zeros(shape, bool)
    for s in stat:
        m = np.zeros(shape, bool)
        m[s['ypix'], s['xpix']] = True
        edge |= m & ~ndimage.binary_erosion(m, iterations=width)
    rgba = np.zeros(shape + (4,))
    rgba[edge] = to_rgba(colour, alpha)
    return rgba


def display_range(img, lo=1, hi=99.7):
    return np.percentile(img, lo), np.percentile(img, hi)


def colormap(name='NineteenEightyNine', reverse=False):
    from matplotlib.colors import LinearSegmentedColormap
    cols = era_brewer.era_brew(name, n=256, brew_type='continuous')
    return LinearSegmentedColormap.from_list(name, cols[::-1] if reverse else cols)


# ---------------------------------------------------------------- loading

def load_holo_session(mouse, holo_tag, source_tag, project_root=None, holo_suffix='SR'):
    """Everything the holo analyses need, in one dict:

    dff (n_holo_rois, n_frames), dff_projected (n_projected, n_frames) or None, trials (stim_trials.csv),
    matrix (stim_matrix.npz), frame_times, fs (measured frame rate), stat, iscell, stat_projected,
    mean_img, mean_img2, match (roi_match_<source>.csv), wheel_speed (deg/s at frames), paths.
    """
    import musicboxes.holography.holo_io as hio
    from musicboxes.holography.stim_table import load_stim_table
    project_root = project_root or hio.PROJECT_ROOT
    paths = hio.session_paths(mouse, holo_tag, project_root, holo_suffix)
    out = paths['output']
    a = np.load(out / 'holo_alignment.npz', allow_pickle=True)
    trials, matrix = load_stim_table(mouse, holo_tag, project_root, holo_suffix)
    rois = hio.load_rois(paths['plane'])
    proj = out / 'funcimg' / 'projected' / 'dFF_projected.npy'
    stat_proj_path = out / 'stat_projected.npy'
    mean_img, mean_img2 = hio.load_mean_images(paths['plane'])
    return {
        'mouse': str(mouse), 'holo_tag': holo_tag, 'source_tag': source_tag, 'paths': paths,
        'dff': np.load(paths['dff'], mmap_mode='r'),
        'dff_projected': np.load(proj) if proj.exists() else None,
        'trials': trials, 'matrix': matrix, 'frame_times': a['frame_times'], 'fs': 1 / float(a['frame_period']),
        'stat': rois['stat'], 'iscell': rois['iscell'],
        'stat_projected': list(np.load(stat_proj_path, allow_pickle=True)) if stat_proj_path.exists() else [],
        'mean_img': mean_img, 'mean_img2': mean_img2,
        'match': pd.read_csv(out / f'roi_match_{source_tag}.csv'),
        'wheel_speed': a['wheel_speed_frames'],
    }


def stimulated_cells(session):
    """One row per stimulated cell (source ROI) with holo_roi, status, trace_source, trace_row."""
    t = session['trials']
    return (t[~t.is_blank].drop_duplicates('condition')
            [['condition', 'source_roi', 'holo_roi', 'status', 'trace_source', 'trace_row']]
            .sort_values('source_roi').reset_index(drop=True))


def trace_for(session, source_roi):
    """dF/F of a stimulated cell (by source ROI id): its holo ROI row or its projected row. None if
    the projected trace has not been extracted yet."""
    c = stimulated_cells(session).set_index('source_roi').loc[source_roi]
    if c.trace_source == 'suite2p':
        return np.asarray(session['dff'][c.trace_row])
    if c.trace_source == 'projected' and c.trace_row >= 0 and session['dff_projected'] is not None:
        return session['dff_projected'][c.trace_row]
    return None


def onsets(session, source_roi=None, blank=False):
    """Onset frames of one cell's trials (or of the blank trials)."""
    t = session['trials']
    sel = t.is_blank if blank else t.source_roi == source_roi
    return t.onset_frame[sel].values, t.offset_frame[sel].values


# ---------------------------------------------------------------- trial-aligned responses

def psth(traces, onset_frames, fs, window=(-2, 6), offset_frames=None):
    """Trial-aligned traces.

    traces: (n_frames,) or (n_cells, n_frames). Returns (t, mat) with mat (n_trials, n_t) or
    (n_cells, n_trials, n_t); samples outside the recording are NaN. With offset_frames given,
    frames from onset to offset of each trial (the spiral train) are set to NaN; the default keeps
    them, because the holo-001 traces show no light artefact during the train (median z ~0 over the
    stimulated cells' 890 trials, frames 0-21 after onset).
    """
    traces = np.asarray(traces, float)
    one = traces.ndim == 1
    traces = np.atleast_2d(traces)
    lags = np.arange(int(np.floor(window[0] * fs)), int(np.ceil(window[1] * fs)) + 1)
    idx = np.asarray(onset_frames)[:, None] + lags[None, :]
    ok = (idx >= 0) & (idx < traces.shape[1])
    mat = np.where(ok[None], traces[:, np.clip(idx, 0, traces.shape[1] - 1)], np.nan)
    if offset_frames is not None:
        stim = (lags[None, :] >= 0) & (idx <= np.asarray(offset_frames)[:, None])
        mat[:, stim] = np.nan
    return lags / fs, mat[0] if one else mat


def response_amplitude(t, mat, baseline=(-2, 0), response=(0.5, 2)):
    """Per trial mean(response window) - mean(baseline window); works on (..., n_t) arrays."""
    b = (t >= baseline[0]) & (t < baseline[1])
    r = (t >= response[0]) & (t <= response[1])
    return np.nanmean(mat[..., r], -1) - np.nanmean(mat[..., b], -1)


def target_responses(session, window=(-2, 6), baseline=(-2, 0), response=(0.5, 2), alpha=0.05):
    """Per stimulated cell: its response to its own stimulation vs to blank trials.

    Columns: source_roi, holo_roi, status, n_trials, amp_mean, amp_sem, blank_mean, p_vs_blank
    (one-sided Mann-Whitney, own trials > blank trials), responsive (p < alpha), z_vs_other (own
    mean amplitude as a z-score of its responses to all other cells' trials, a null with ~1000 trials).
    """
    from scipy.stats import mannwhitneyu
    fs = session['fs']
    t_all = session['trials']
    rows = []
    for c in stimulated_cells(session).itertuples():
        tr = trace_for(session, c.source_roi)
        if tr is None:
            rows.append({'source_roi': c.source_roi, 'holo_roi': c.holo_roi, 'status': c.status, 'n_trials': 0})
            continue
        t, own = psth(tr, onsets(session, c.source_roi)[0], fs, window)
        _, blank = psth(tr, onsets(session, blank=True)[0], fs, window)
        _, other = psth(tr, t_all.onset_frame[~t_all.is_blank & (t_all.source_roi != c.source_roi)].values, fs, window)
        a_own, a_blank, a_other = (response_amplitude(t, m, baseline, response) for m in (own, blank, other))
        p = mannwhitneyu(a_own, a_blank, alternative='greater').pvalue
        rows.append({'source_roi': c.source_roi, 'holo_roi': c.holo_roi, 'status': c.status, 'n_trials': len(a_own),
                     'amp_mean': np.nanmean(a_own), 'amp_sem': np.nanstd(a_own) / np.sqrt(len(a_own)),
                     'blank_mean': np.nanmean(a_blank), 'p_vs_blank': p, 'responsive': p < alpha,
                     'z_vs_other': (np.nanmean(a_own) - np.nanmean(a_other)) / np.nanstd(a_other)})
    return pd.DataFrame(rows)


def response_matrix(session, window=(-2, 6), baseline=(-2, 0), response=(0.5, 2)):
    """Mean response amplitude of every holo ROI to every condition: (n_holo_rois, n_conditions)
    plus the condition labels and each ROI's distance (px) to each condition's target (holo FOV)."""
    import musicboxes.holography.holo_io as hio
    t_all = session['trials']
    conds = session['matrix']['conditions']
    amp = np.full((len(session['stat']), len(conds)), np.nan)
    dff = np.asarray(session['dff'])
    for j, c in enumerate(conds):
        t, m = psth(dff, t_all.onset_frame[t_all.condition == c].values, session['fs'], window)
        amp[:, j] = np.nanmean(response_amplitude(t, m, baseline, response), 1)
    xy = hio.roi_centroids(session['stat'])
    tg = t_all.drop_duplicates('condition').set_index('condition').loc[conds, ['target_x_holo', 'target_y_holo']].values
    dist = np.hypot(xy[:, None, 0] - tg[None, :, 0], xy[:, None, 1] - tg[None, :, 1])
    return amp, conds, dist


# ---------------------------------------------------------------- item 5: crop-out movie

def stim_triggered_movie(cell, mouse, holo_tag, source_tag, crop_px=40, window=(-2, 6), bin_frames=2,
                         project_root=None, holo_suffix='SR', out_dir=None, fps=15, session=None):
    """Trial-averaged crop-out movie of one stimulated cell around its own stimulations.

    cell: source ROI id (= ensemble number). Reads its trials' frames from the registered movie
    (suite2p plane0 data.bin, int16; ~1.6 GB per cell over the share for an 8 s window), crops a
    (2*crop_px)^2 square around the cell (holo ROI centroid, or the projected mask's), averages over
    trials, bins bin_frames frames, and normalises each pixel by its pre-stimulus mean:
    (F - F0) / F0 with F0 the mean over window[0]..0 s.

    Saves <out_dir>/stim_movie_<source>-<cell>.mp4 (with mask outline, target, time and a
    'stim' marker during the spiral train) and .tif (float32 stack of the dF/F crop), out_dir
    defaulting to <analysis_dir>/stim_movies. Returns (t, movie, raw_mean) with movie (n_t, y, x).
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import tifffile
    import musicboxes.holography.holo_io as hio

    session = session or load_holo_session(mouse, holo_tag, source_tag, project_root, holo_suffix)
    paths = session['paths']
    c = stimulated_cells(session).set_index('source_roi').loc[cell]
    if c.status == 'matched':
        s = session['stat'][c.holo_roi]
    else:
        s = next(p for p in session['stat_projected'] if int(p['source_roi']) == cell)
    cy, cx = int(round(np.mean(s['ypix']))), int(round(np.mean(s['xpix'])))
    ops = hio.load_ops(paths['plane'])
    Ly, Lx, n_frames = int(ops['Ly']), int(ops['Lx']), int(ops['nframes'])
    y0, y1 = max(cy - crop_px, 0), min(cy + crop_px, Ly)
    x0, x1 = max(cx - crop_px, 0), min(cx + crop_px, Lx)

    fs = session['fs']
    lags = np.arange(int(np.floor(window[0] * fs)), int(np.ceil(window[1] * fs)) + 1)
    on, off = onsets(session, cell)
    movie = np.memmap(paths['plane'] / 'data.bin', mode='r', dtype=np.int16, shape=(n_frames, Ly, Lx))
    acc, n = np.zeros((len(lags), y1 - y0, x1 - x0)), 0
    for o in on:
        if o + lags[0] < 0 or o + lags[-1] >= n_frames:
            continue
        acc += movie[o + lags[0]:o + lags[-1] + 1, y0:y1, x0:x1].astype(float)
        n += 1
    raw = acc / n
    nb = len(lags) // bin_frames
    raw_b = raw[:nb * bin_frames].reshape(nb, bin_frames, *raw.shape[1:]).mean(1)
    t = lags[:nb * bin_frames].reshape(nb, bin_frames).mean(1) / fs
    f0 = raw_b[t < 0].mean(0)
    f0 = np.maximum(f0, np.percentile(f0, 5))
    dff = (raw_b - f0) / f0

    out_dir = Path(out_dir) if out_dir else paths['analysis'] / 'stim_movies'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f'stim_movie_{source_tag}-{cell}'
    tifffile.imwrite(f'{stem}.tif', dff.astype(np.float32), imagej=True, metadata={'axes': 'TYX', 'finterval': 1 / fs * bin_frames})

    train_end = np.median(off - on + 1) / fs
    outline = outline_overlay([{'ypix': np.asarray(s['ypix']) - y0, 'xpix': np.asarray(s['xpix']) - x0}],
                              (y1 - y0, x1 - x0), ROLE_COLOURS['other'])
    lim = np.percentile(np.abs(dff[t > 0]), 99.5)
    fig, ax = plt.subplots(figsize=(4.5, 4.8))
    im = ax.imshow(dff[0], cmap=colormap(), vmin=-lim / 3, vmax=lim)
    ax.imshow(outline)
    tg = session['trials'].set_index('source_roi').loc[cell].iloc[0]
    ax.plot(tg.target_x_holo - x0, tg.target_y_holo - y0, '+', color='w', ms=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label='dF/F (trial average)')
    ax.set_axis_off()
    title = ax.set_title('')
    def draw(i):
        im.set_data(dff[i])
        stim = '  STIM' if 0 <= t[i] <= train_end else ''
        title.set_text(f'{source_tag} cell {cell} ({c.status}), n={n} trials\nt = {t[i]:+.2f} s{stim}')
        return im, title
    anim = animation.FuncAnimation(fig, draw, frames=len(t), blit=False)
    anim.save(f'{stem}.mp4', writer=animation.FFMpegWriter(fps=fps), dpi=150)
    plt.close(fig)
    print(f'Saved {stem}.mp4 and .tif ({n} trials, {len(t)} binned frames, crop {y1 - y0}x{x1 - x0} px)')
    return t, dff, raw_b


# ---------------------------------------------------------------- item 6: goal-progress phase

def stim_cells_by_phase(mouse, holo_tag, source_tag, project_root=None, holo_suffix='SR', gp_table=None):
    """Stimulated cells with the source session's goal-progress tuning, sorted by preferred phase.

    Joins stim_cells.csv (stim_table.py) with gp_cell_table.csv (cell_id = source ROI, see
    musicboxes/goalprogress/gp_cell_table.py) and returns one row per stimulated cell:
    source_roi, holo_roi, status, trace_source, trace_row, gp_tuned, preferred_phase (0-1, 0 =
    reward), p, split_r, phase_rank. Untuned cells (preferred_phase NaN) come last.
    Note: the sub-001 seq3-007 table's phases sit on a 30-bin grid (0.0167 + k/30 -> binned peaks),
    i.e. it was made with --method binned or before von Mises fits became the default.
    """
    import musicboxes.holography.holo_io as hio
    project_root = project_root or hio.PROJECT_ROOT
    out = hio.session_paths(mouse, holo_tag, project_root, holo_suffix)['output']
    cells = pd.read_csv(out / 'stim_cells.csv')
    cells = cells[cells.condition != 'blank'].drop(columns=['gp_tuned', 'preferred_phase'], errors='ignore')
    gp = pd.read_csv(gp_table or hio.gp_table_path(mouse, source_tag, project_root))
    t = cells.merge(gp.rename(columns={'cell_id': 'source_roi', 'x': 'source_x', 'y': 'source_y'}),
                    on='source_roi', how='left', validate='one_to_one')
    missing = t.gp_tuned.isna().sum()
    if missing:
        print(f'{missing} stimulated cells are not in the gp table (not iscell in {source_tag}?)')
    t = t.sort_values('preferred_phase', na_position='last').reset_index(drop=True)
    t['phase_rank'] = np.arange(len(t))
    return t

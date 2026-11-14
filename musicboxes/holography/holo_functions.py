"""Analysis functions for holographic photostimulation sessions (musicboxes/holography/).

Used by session_overview_holography.py and meant for notebooks:

    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
    import musicboxes.holography.holo_functions as holo

Contents
--------
- Display helpers: mask outlines as an RGBA overlay, era_brewer colours.
- load_holo_session: dF/F (suite2p ROIs + projected source masks), stim table and alignment in one dict.
- ensure_dfof: true dF/F (divided by F0), cached next to the other outputs; see its docstring for why
  F0 comes from raw F and not from the neuropil-corrected trace.
- psth / response_amplitude: trial-aligned dF/F (stimulation frames can be masked out, off by default:
  holo-001 shows no light artefact during the spiral train).
- target_responses: per-cell significance against a far-target null with Benjamini-Hochberg FDR across
  cells, and per-trial reliability (how many of a cell's trials exceed the null's 95th percentile).
- stim_triggered_movie (task item 5): crop-out movie of one stimulated cell: one example trial of the
  registered frames (grey) next to the trial-averaged dF/F.
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


def colormap(name='NineteenEightyNine', reverse=True):
    """era_brewer colormap; reversed by default so that red = high (active) and blue = low, the
    convention of the other analyses in this repo."""
    from matplotlib.colors import LinearSegmentedColormap
    cols = era_brewer.era_brew(name, n=256, brew_type='continuous')
    return LinearSegmentedColormap.from_list(name, cols[::-1] if reverse else cols)


# ---------------------------------------------------------------- loading

UNITS = {'dfof': 'dF/F', 'df': 'dF'}


def _raw_baseline(F, ops):
    """Slow baseline of raw fluorescence: the same suite2p dcnv.preprocess (maximin) baseline that
    compute_dff removes from F - 0.7*Fneu, here applied to F itself (preprocess returns F - baseline)."""
    from suite2p.extraction import dcnv
    from nwb_barcoding.compute_dff import _get_baseline_params
    baseline, win, sig, prctile = _get_baseline_params(ops)
    F = np.asarray(F, np.float32)
    return F - dcnv.preprocess(F.copy(), baseline, win, sig, ops['fs'], prctile)


def ensure_dfof(paths, force=False, f0_floor=1.0):
    """dF/F for the holo ROIs and the projected masks, cached as <output>/funcimg/dFoF.npy and
    <output>/funcimg/projected/dFoF_projected.npy (n_rois, n_frames, float32). Returns the paths.

    dF/F = dF / F0, with
      dF = the lab's dFF.npy (nwb_barcoding.compute_dff: F - 0.7*Fneu minus its maximin baseline),
           which is baseline-subtracted but NOT divided, so it is in raw fluorescence units;
      F0 = the same maximin baseline computed on raw F (not neuropil-subtracted), floored at f0_floor.
    F0 is not taken from F - 0.7*Fneu because that baseline sits near or below zero for many ROIs
    (holo-001: median 17 across ROIs, negative for ~5%, while raw F has median ~94), so dividing by
    it blows up. With this F0 the amplitudes are an underestimate of dF/F relative to a
    neuropil-free F0, but they are stable and comparable across cells.
    """
    import musicboxes.holography.holo_io as hio
    ops = hio.load_ops(paths['plane'])
    out = paths['output'] / 'funcimg'
    proj = out / 'projected'
    jobs = [(paths['plane'] / 'F.npy', paths['dff'], out / 'dFoF.npy')]
    if (proj / 'F_projected.npy').exists():
        jobs.append((proj / 'F_projected.npy', proj / 'dFF_projected.npy', proj / 'dFoF_projected.npy'))
    for f_path, df_path, dst in jobs:
        if dst.exists() and not force:
            continue
        print(f'Computing {dst} (dF / raw-F baseline) ...')
        f0 = _raw_baseline(np.load(f_path), ops)
        print(f'  F0 median {np.median(f0):.1f}, {(f0 < f0_floor).mean() * 100:.2f}% of samples floored at {f0_floor}')
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, (np.load(df_path) / np.maximum(f0, f0_floor)).astype(np.float32))
    return [j[2] for j in jobs]


def load_holo_session(mouse, holo_tag, source_tag, project_root=None, holo_suffix='SR', units='dfof'):
    """Everything the holo analyses need, in one dict:

    dff (n_holo_rois, n_frames), dff_projected (n_projected, n_frames) or None, units, trials
    (stim_trials.csv), matrix (stim_matrix.npz), frame_times, fs (measured frame rate), stat, iscell,
    stat_projected, mean_img, mean_img2, match (roi_match_<source>.csv), wheel_speed (deg/s at frames),
    paths. units: 'dfof' (default; dF/F from ensure_dfof, computed on first use) or 'df' (the
    baseline-subtracted, undivided dFF.npy).
    """
    import musicboxes.holography.holo_io as hio
    from musicboxes.holography.stim_table import load_stim_table
    project_root = project_root or hio.PROJECT_ROOT
    paths = hio.session_paths(mouse, holo_tag, project_root, holo_suffix)
    out = paths['output']
    a = np.load(out / 'holo_alignment.npz', allow_pickle=True)
    trials, matrix = load_stim_table(mouse, holo_tag, project_root, holo_suffix)
    rois = hio.load_rois(paths['plane'])
    if units == 'dfof':
        ensure_dfof(paths)
        dff_path, proj = out / 'funcimg' / 'dFoF.npy', out / 'funcimg' / 'projected' / 'dFoF_projected.npy'
    else:
        dff_path, proj = paths['dff'], out / 'funcimg' / 'projected' / 'dFF_projected.npy'
    stat_proj_path = out / 'stat_projected.npy'
    mean_img, mean_img2 = hio.load_mean_images(paths['plane'])
    return {
        'mouse': str(mouse), 'holo_tag': holo_tag, 'source_tag': source_tag, 'paths': paths,
        'dff': np.load(dff_path, mmap_mode='r'), 'units': UNITS[units],
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


def bh_fdr(p):
    """Benjamini-Hochberg q-values; NaNs stay NaN and do not count towards the number of tests."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qv = np.empty(len(pv))
    qv[order] = np.minimum(ranked, 1)
    q[ok] = qv
    return q


def cell_centroid(session, source_roi):
    """(x, y) in the holo FOV of a stimulated cell: its holo ROI, or its projected source mask."""
    import musicboxes.holography.holo_io as hio
    c = stimulated_cells(session).set_index('source_roi').loc[source_roi]
    if c.status == 'matched':
        return hio.roi_centroids([session['stat'][c.holo_roi]])[0]
    s = [p for p in session['stat_projected'] if int(p['source_roi']) == source_roi]
    return hio.roi_centroids(s)[0] if s else np.array([np.nan, np.nan])


def target_responses(session, window=(-2, 6), baseline=(-2, 0), response=(0.5, 2), alpha=0.05,
                     trial_alpha=0.05, null_dist_px=50, null_exclude_s=8, return_trials=False):
    """Per stimulated cell: response to its own stimulation, its significance and its reliability.

    Amplitude per trial = mean(response window) - mean(baseline window) of the cell's trace.

    Null distribution (per cell): its amplitudes on all trials that stimulated a target at least
    null_dist_px from the cell (off-target responses vanish by ~25 px in holo-001) plus the blank
    trials, leaving out trials starting within null_exclude_s after one of the cell's own trials
    (their baseline window still holds the decay of its own response). ~900 trials per cell.

    Cell-level tests, each Benjamini-Hochberg corrected across the cells with a trace (q_*):
      p_vs_null   one-sided Mann-Whitney, own trials > null trials (primary test)
      p_vs_blank  one-sided Mann-Whitney, own trials > blank trials (10 vs 10, low power)
      p_binom     binomial test of n_success against trial_alpha (is the cell reliable beyond chance)
    responsive = q_vs_null < alpha and amp_mean > 0.

    Reliability: a trial is a success if its amplitude exceeds the (1 - trial_alpha) quantile of the
    cell's null; n_success of n_trials (by chance ~trial_alpha * n_trials, i.e. 0.5 of 10).

    Returns a DataFrame with one row per cell (source_roi, holo_roi, status, n_trials, n_null,
    amp_mean, amp_sem, blank_mean, null_mean, null_thr, z_vs_other, n_success, frac_success, the p and
    q columns, responsive); with return_trials=True also a long DataFrame of the own trials (source_roi,
    trial, rep, amp, success).
    """
    from scipy.stats import mannwhitneyu, binomtest
    fs = session['fs']
    t_all = session['trials']
    onset_all = t_all.onset_frame.values
    tx, ty = t_all.target_x_holo.values, t_all.target_y_holo.values
    rows, trial_rows = [], []
    for c in stimulated_cells(session).itertuples():
        tr = trace_for(session, c.source_roi)
        if tr is None:
            rows.append({'source_roi': c.source_roi, 'holo_roi': c.holo_roi, 'status': c.status, 'n_trials': 0})
            continue
        t, m = psth(tr, onset_all, fs, window)
        amp = response_amplitude(t, m, baseline, response)
        own = (t_all.source_roi == c.source_roi).values
        blank = t_all.is_blank.values
        x, y = cell_centroid(session, c.source_roi)
        far = np.hypot(tx - x, ty - y) >= null_dist_px
        own_times = t_all.onset_time.values[own]
        dt = t_all.onset_time.values[:, None] - own_times[None, :]
        after_own = ((dt > 0) & (dt <= null_exclude_s)).any(1)
        null = (blank | far) & ~own & ~after_own
        a_own, a_blank, a_null = amp[own], amp[blank], amp[null]
        a_null = a_null[~np.isnan(a_null)]
        thr = np.quantile(a_null, 1 - trial_alpha)
        success = a_own > thr
        valid = ~np.isnan(a_own)
        rows.append({'source_roi': c.source_roi, 'holo_roi': c.holo_roi, 'status': c.status,
                     'n_trials': int(valid.sum()), 'n_null': len(a_null),
                     'amp_mean': np.nanmean(a_own), 'amp_sem': np.nanstd(a_own) / np.sqrt(valid.sum()),
                     'blank_mean': np.nanmean(a_blank), 'null_mean': a_null.mean(), 'null_thr': thr,
                     'z_vs_other': (np.nanmean(a_own) - a_null.mean()) / a_null.std(),
                     'n_success': int(success.sum()), 'frac_success': success.sum() / valid.sum(),
                     'p_vs_null': mannwhitneyu(a_own[valid], a_null, alternative='greater').pvalue,
                     'p_vs_blank': mannwhitneyu(a_own[valid], a_blank[~np.isnan(a_blank)], alternative='greater').pvalue,
                     'p_binom': binomtest(int(success.sum()), int(valid.sum()), trial_alpha, alternative='greater').pvalue})
        tc = t_all[own]
        trial_rows.append(pd.DataFrame({'source_roi': c.source_roi, 'trial': tc.trial.values,
                                        'rep': tc.rep.values, 'amp': a_own, 'success': success}))
    df = pd.DataFrame(rows)
    for col in ('p_vs_null', 'p_vs_blank', 'p_binom'):
        df['q' + col[1:]] = bh_fdr(df[col])
    df['responsive'] = (df.q_vs_null < alpha) & (df.amp_mean > 0)
    if return_trials:
        return df, pd.concat(trial_rows, ignore_index=True)
    return df


def response_matrix(session, window=(-2, 6), baseline=(-2, 0), response=(0.5, 2)):
    """Mean response amplitude of every holo ROI to every condition: (n_holo_rois, n_conditions)
    plus the condition labels and each ROI's distance (px) to each condition's target (holo FOV)."""
    import musicboxes.holography.holo_io as hio
    t_all = session['trials']
    conds = session['matrix']['condition']
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

def _example_trial(session, cell, window, baseline=(-2, 0), response=(0.5, 2)):
    """rep of the cell's successful trial with the largest amplitude (target_trial_responses.csv from
    the overview); without that file, or without a successful trial, the trial with the largest amplitude."""
    path = session['paths']['output'] / 'target_trial_responses.csv'
    if path.exists():
        tr = pd.read_csv(path)
        tr = tr[tr.source_roi == cell]
        if len(tr):
            pool = tr[tr.success] if tr.success.any() else tr
            return int(pool.sort_values('amp').rep.iloc[-1]), bool(tr.success.any())
    t, m = psth(trace_for(session, cell), onsets(session, cell)[0], session['fs'], window)
    return int(np.nanargmax(response_amplitude(t, m, baseline, response))), False


def stim_triggered_movie(cell, mouse, holo_tag, source_tag, crop_px=40, window=(-2, 6), bin_frames=2,
                         project_root=None, holo_suffix='SR', out_dir=None, fps=15, session=None,
                         example_rep=None, smooth_px=1.0, smooth_bins=3, raw_smooth_px=0, raw_avg_frames=20):
    """Crop-out movie of one stimulated cell around its own stimulations: one example trial of the
    registered frames next to the trial-averaged dF/F.

    cell: source ROI id (= ensemble number). Reads its trials' frames from the registered movie
    (suite2p plane0 data.bin = functional channel, green; int16; ~1.6 GB per cell over the share for an
    8 s window) and crops a (2*crop_px)^2 square around the cell (holo ROI centroid, or the projected
    mask's). Frames are binned by bin_frames.
      left:  registered frames of a single trial, grey scale, as a raw_avg_frames trailing sliding
             average at full frame rate (frame k = mean of frames k-raw_avg_frames+1..k, so nothing from
             after the train onset leaks before it; 20 frames = 0.45 s), shown at the binned time points;
             raw_avg_frames=1 gives single frames. raw_smooth_px > 0 adds a spatial Gaussian. Grey range = 1st-99.5th percentile of
             all the trial's pixels, since single frames are shot-noise dominated (cell 57: median 3,
             99th percentile ~1460, so a mean-image range saturates everything). example_rep picks the trial (0-based rep); default is the cell's successful
             trial with the largest amplitude (target_trial_responses.csv, written by the overview).
      right: trial average, normalised per pixel by its pre-stimulus mean, (F - F0) / F0 with F0 the
             mean over window[0]..0 s; red = high colormap, smoothed for display (Gaussian smooth_px
             in space, smooth_bins running mean in time) since a trial-averaged bin holds only
             n_trials * bin_frames frames.
    Both panels show the mask outline and the target (+); the title gives time and 'STIM' during the train.

    Saves to <out_dir> (default <analysis_dir>/stim_movies):
      stim_movie_<source>-<cell>.mp4
      stim_movie_<source>-<cell>.tif                 float32 dF/F stack (trial average, unsmoothed)
      stim_movie_<source>-<cell>_trial<rep>_raw.tif  float32 registered stack of the example trial
                                                     (sliding average, at the binned time points)
    Returns (t, dff, raw_mean, raw_trial), each (n_t, y, x).
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
    if example_rep is None:
        example_rep, from_success = _example_trial(session, cell, window)
    else:
        from_success = False
    movie = np.memmap(paths['plane'] / 'data.bin', mode='r', dtype=np.int16, shape=(n_frames, Ly, Lx))
    acc, n, trial = np.zeros((len(lags), y1 - y0, x1 - x0)), 0, None
    for k, o in enumerate(on):
        if o + lags[0] < 0 or o + lags[-1] >= n_frames:
            continue
        chunk = movie[o + lags[0]:o + lags[-1] + 1, y0:y1, x0:x1].astype(float)
        acc += chunk
        n += 1
        if k == example_rep:
            trial = chunk
    nb = len(lags) // bin_frames
    binned = lambda x: x[:nb * bin_frames].reshape(nb, bin_frames, *x.shape[1:]).mean(1)
    raw_b = binned(acc / n)
    cs = np.concatenate([np.zeros((1,) + trial.shape[1:]), np.cumsum(trial, 0)])
    k = np.arange(1, len(trial) + 1)
    lo = np.maximum(k - raw_avg_frames, 0)
    trailing = (cs[k] - cs[lo]) / (k - lo)[:, None, None]
    trial_b = trailing[np.arange(nb) * bin_frames + bin_frames - 1]      # trailing mean ending at each bin's last frame
    t = lags[:nb * bin_frames].reshape(nb, bin_frames).mean(1) / fs
    f0 = raw_b[t < 0].mean(0)
    f0 = np.maximum(f0, np.percentile(f0, 5))
    dff = (raw_b - f0) / f0

    out_dir = Path(out_dir) if out_dir else paths['analysis'] / 'stim_movies'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f'stim_movie_{source_tag}-{cell}'
    meta = {'axes': 'TYX', 'finterval': 1 / fs * bin_frames}
    tifffile.imwrite(f'{stem}.tif', dff.astype(np.float32), imagej=True, metadata=meta)
    tifffile.imwrite(f'{stem}_trial{example_rep}_raw.tif', trial_b.astype(np.float32), imagej=True, metadata=meta)

    train_end = np.median(off - on + 1) / fs
    outline = outline_overlay([{'ypix': np.asarray(s['ypix']) - y0, 'xpix': np.asarray(s['xpix']) - x0}],
                              (y1 - y0, x1 - x0), ROLE_COLOURS['other'], alpha=0.6)
    dff_s = ndimage.uniform_filter1d(ndimage.gaussian_filter(dff, (0, smooth_px, smooth_px)), smooth_bins, axis=0)
    lim = np.percentile(np.abs(dff_s[t > 0]), 99.5)
    tg = session['trials'].set_index('source_roi').loc[cell].iloc[0]
    trial_show = ndimage.gaussian_filter(trial_b, (0, raw_smooth_px, raw_smooth_px)) if raw_smooth_px else trial_b
    trial_lab = f"trial {example_rep}, {raw_avg_frames}-frame sliding avg" + (' (successful)' if from_success else '')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    ims = []
    for ax, data, cmap, clim, label in (
            (axes[0], trial_show, 'gray', display_range(trial_show, 1, 99.5), f'registered F, {trial_lab}'),
            (axes[1], dff_s, colormap(), (-lim / 3, lim), f'dF/F, average of {n} trials')):
        ims.append(ax.imshow(data[0], cmap=cmap, vmin=clim[0], vmax=clim[1]))
        ax.imshow(outline)
        ax.plot(tg.target_x_holo - x0, tg.target_y_holo - y0, '+', color='w', ms=8)
        fig.colorbar(ims[-1], ax=ax, fraction=0.046, label=label)
        ax.set_axis_off()
    title = fig.suptitle('')
    def draw(i):
        ims[0].set_data(trial_show[i])
        ims[1].set_data(dff_s[i])
        stim = '  STIM' if 0 <= t[i] <= train_end else ''
        title.set_text(f'{source_tag} cell {cell} ({c.status}), t = {t[i]:+.2f} s{stim}')
        return ims + [title]
    anim = animation.FuncAnimation(fig, draw, frames=len(t), blit=False)
    anim.save(f'{stem}.mp4', writer=animation.FFMpegWriter(fps=fps), dpi=150)
    plt.close(fig)
    print(f'Saved {stem}.mp4 and .tifs ({n} trials, example {trial_lab}, {len(t)} binned frames, '
          f'crop {y1 - y0}x{x1 - x0} px)')
    return t, dff, raw_b, trial_b


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

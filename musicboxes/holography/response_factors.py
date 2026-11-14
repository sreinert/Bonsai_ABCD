"""What makes a stimulated cell respond, or fail to? Tests four explanations on the reliability results.

Builds on target_responses.csv / target_trial_responses.csv (holo_functions.target_responses, written
by session_overview_holography.py; recomputed here if missing). Run from the repo root:

    python -m musicboxes.holography.response_factors 001 holo-001 seq3-007

Hypotheses and what is plotted (saved as rf_*.png in analysis/sub-<mouse>/<holo label>/):
  rf_1_red_intensity   out of plane -> dimmer in the red (structural) channel -> weaker stimulation.
                       Per cell, mean meanImg_chan2 over its mask vs its mean response (dF/F); also red
                       contrast (mask / surrounding annulus, independent of local illumination) and green
                       brightness. Colour = successful trials.
  rf_2_fov_position    cells near the FOV border get less laser power. Mean image with every stimulated
                       cell's mask filled by its number of successful trials, and success / response vs the
                       target's distance from the FOV centre and from a fitted response hotspot (2D
                       Gaussian of response vs target position, fit_power_centre).
  rf_3_motion          brain motion in failed trials moves the cell away from the fixed spiral target.
                       Per own trial: running speed (|wheel|, -1..+2 s), rigid registration displacement
                       during the train (|(xoff, yoff) - session median|, suite2p ops; registration was
                       rigid only and there is no z-stack, zcorr.npy is empty), registration quality
                       (corrXY during the train / session median) and the cell's red-channel F in the 1 s
                       before the train / its session median (F_chan2, matched cells only: a drop means the
                       cell had moved in x, y or z). Not measured during the train: there red F rises with
                       the cell's own response (GCaMP bleed-through into the red PMT) and with stimulation
                       light (holo-001: train/median 1.12 in successful vs 1.07 in failed trials), so it is
                       not a motion readout. Successful vs failed trials, pooled over
                       'mixed' cells (1-9 successes) and paired within cells (Wilcoxon on per-cell means).
  rf_4_adaptation      responsiveness decays over repeats. Response vs repeat (0-9) per cell (thin) and
                       mean +- sem over cells with >= 1 success; also z-scored within each cell, and
                       the success fraction per repeat. Per-cell slopes tested against 0 (Wilcoxon). Note:
                       repeat number is confounded with time in session (running, bleaching).
  rf_5_cell_factors    Spearman correlation of every per-cell factor with n_success and amp_mean.
Tables (output folder): response_factors_cells.csv (one row per stimulated cell) and
response_factors_trials.csv (one row per own trial, with the motion metrics).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats

from ceph_paths import ceph_path
import musicboxes.holography.holo_io as hio
import musicboxes.holography.holo_functions as holo

WINDOW, BASELINE, RESPONSE = (-2, 6), (-2, 0), (0.5, 2)
COL = holo.ROLE_COLOURS
SUCC_COL, FAIL_COL = COL['matched'], COL['extra']


def _despine(ax):
    ax.spines[['top', 'right']].set_visible(False)


def _save(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def load_responses(session):
    out = session['paths']['output']
    if (out / 'target_responses.csv').exists() and (out / 'target_trial_responses.csv').exists():
        return pd.read_csv(out / 'target_responses.csv'), pd.read_csv(out / 'target_trial_responses.csv')
    print('target_responses.csv missing, computing ...')
    resp, trials = holo.target_responses(session, WINDOW, BASELINE, RESPONSE, return_trials=True)
    resp.to_csv(out / 'target_responses.csv', index=False)
    trials.to_csv(out / 'target_trial_responses.csv', index=False)
    return resp, trials


def cell_mask(session, source_roi):
    """stat entry (ypix, xpix) of a stimulated cell in the holo FOV: holo ROI or projected mask."""
    c = holo.stimulated_cells(session).set_index('source_roi').loc[source_roi]
    if c.status == 'matched':
        return session['stat'][c.holo_roi]
    return next(p for p in session['stat_projected'] if int(p['source_roi']) == source_roi)


def _all_mask_pixels(session):
    m = np.zeros(session['mean_img'].shape, bool)
    for s in list(session['stat']) + list(session['stat_projected']):
        m[s['ypix'], s['xpix']] = True
    return m


def cell_features(session, resp, annulus_px=(2, 8)):
    """Per stimulated cell: brightness (red, green, contrast to the surrounding annulus) and position."""
    shape = session['mean_img'].shape
    occupied = _all_mask_pixels(session)
    red, green = session['mean_img2'], session['mean_img']
    match = session['match'].set_index('source_roi')
    rows = []
    for c in holo.stimulated_cells(session).itertuples():
        s = cell_mask(session, c.source_roi)
        m = np.zeros(shape, bool)
        m[s['ypix'], s['xpix']] = True
        ring = ndimage.binary_dilation(m, iterations=annulus_px[1]) & ~ndimage.binary_dilation(m, iterations=annulus_px[0])
        ring &= ~occupied
        tg = session['trials'][session['trials'].source_roi == c.source_roi].iloc[0]
        cy, cx = shape[0] / 2, shape[1] / 2
        rows.append({'source_roi': c.source_roi,
                     'x': np.mean(s['xpix']), 'y': np.mean(s['ypix']), 'n_pix': int(m.sum()),
                     'red_mean': red[m].mean() if red is not None else np.nan,
                     'red_contrast': red[m].mean() / red[ring].mean() if red is not None and ring.any() else np.nan,
                     'green_mean': green[m].mean(),
                     'green_contrast': green[m].mean() / green[ring].mean() if ring.any() else np.nan,
                     'x_target': tg.target_x_holo, 'y_target': tg.target_y_holo,
                     'target_dist_centre_px': np.hypot(tg.target_x_holo - cx, tg.target_y_holo - cy),
                     'target_to_cell_px': np.hypot(tg.target_x_holo - np.mean(s['xpix']),
                                                   tg.target_y_holo - np.mean(s['ypix']))})
    f = pd.DataFrame(rows)
    return resp.merge(f, on='source_roi', how='left')


def trial_motion(session, trials_df, speed_window=(-1, 2), disp_window=(-0.5, 2), red_window=(-1, 0)):
    """Motion metrics for every own trial in trials_df (source_roi, trial, rep, amp, success)."""
    paths = session['paths']
    ops = hio.load_ops(paths['plane'])
    xoff, yoff, corr = (np.asarray(ops[k], float) for k in ('xoff', 'yoff', 'corrXY'))
    disp = np.hypot(xoff - np.median(xoff), yoff - np.median(yoff))
    corr_rel = corr / np.median(corr)
    bad = np.asarray(ops.get('badframes', np.zeros(len(xoff), bool)), bool)
    f_red = np.load(paths['plane'] / 'F_chan2.npy', mmap_mode='r') if (paths['plane'] / 'F_chan2.npy').exists() else None
    speed = np.abs(np.asarray(session['wheel_speed'], float))
    fs, n = session['fs'], len(xoff)
    t = session['trials'].set_index('trial')
    cells = holo.stimulated_cells(session).set_index('source_roi')
    red_med = {}
    rows = []
    for r in trials_df.itertuples():
        on, off = int(t.loc[r.trial, 'onset_frame']), int(t.loc[r.trial, 'offset_frame'])
        sw = slice(max(on + int(speed_window[0] * fs), 0), min(on + int(speed_window[1] * fs), n))
        dw = slice(max(on + int(disp_window[0] * fs), 0), min(on + int(disp_window[1] * fs), n))
        train = slice(on, off + 1)
        rw = slice(max(on + int(red_window[0] * fs), 0), on + int(red_window[1] * fs))
        red_ratio = np.nan
        c = cells.loc[r.source_roi]
        if f_red is not None and c.status == 'matched':
            h = int(c.holo_roi)
            if h not in red_med:
                red_med[h] = np.median(np.asarray(f_red[h]))
            red_ratio = np.asarray(f_red[h, rw]).mean() / red_med[h]
        rows.append({'trial': r.trial, 'source_roi': r.source_roi, 'speed': np.nanmean(speed[sw]),
                     'disp_train': disp[train].mean(), 'disp_max': disp[dw].max(),
                     'corr_train': corr_rel[train].mean(), 'red_ratio': red_ratio,
                     'bad_frames': int(bad[dw].sum()), 'onset_time_min': t.loc[r.trial, 'onset_time'] / 60})
    return trials_df.merge(pd.DataFrame(rows), on=['trial', 'source_roi'], how='left')


# ---------------------------------------------------------------- plots

def _succ_norm(n_max=10):
    from matplotlib.colors import Normalize
    return Normalize(0, n_max)


def plot_red_intensity(cells, save_path=None):
    ok = cells[cells.n_trials > 0]
    cmap, norm = holo.colormap(), _succ_norm(int(ok.n_trials.max()))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, col, lab in zip(axes, ('red_mean', 'red_contrast', 'green_mean'),
                            ('mean red intensity in mask (meanImg_chan2)', 'red contrast (mask / annulus)',
                             'mean green intensity in mask (meanImg)')):
        sc = ax.scatter(ok[col], ok.amp_mean, c=ok.n_success, cmap=cmap, norm=norm, s=22,
                        edgecolor=np.where(ok.status == 'projected', 'k', 'none'), lw=0.7)
        rho, p = stats.spearmanr(ok[col], ok.amp_mean, nan_policy='omit')
        rho2, p2 = stats.spearmanr(ok[col], ok.n_success, nan_policy='omit')
        ax.set_title(f'Spearman vs response: rho={rho:.2f}, p={p:.2g}\nvs successful trials: rho={rho2:.2f}, p={p2:.2g}',
                     fontsize=9)
        ax.set_xlabel(lab)
        ax.set_ylabel('mean own response (dF/F)')
        _despine(ax)
    fig.colorbar(sc, ax=axes, fraction=0.02, label='successful trials')
    fig.suptitle('Out of plane? Red (structural) brightness of each stimulated cell vs its response; '
                 'black edge = projected mask', y=0.99)
    return _save(fig, save_path)


def fit_power_centre(cells, col='amp_mean'):
    """Isotropic 2D Gaussian fit of a response measure against target position, c + A exp(-r^2 / 2 sigma^2)
    with r the distance to (x0, y0): where does stimulation work best? Returns dict(x0, y0, sigma, A, c).
    In-sample, so the correlation with distance to this centre is optimistic; use it to locate the
    hotspot, not to test it."""
    from scipy.optimize import curve_fit
    ok = cells[(cells.n_trials > 0) & cells[col].notna()]
    xy = np.c_[ok.x_target, ok.y_target].T
    g = lambda xy, x0, y0, sig, A, c: c + A * np.exp(-((xy[0] - x0) ** 2 + (xy[1] - y0) ** 2) / (2 * sig ** 2))
    w = np.clip(ok[col].values, 0, None)
    p0 = [np.average(xy[0], weights=w + 1e-9), np.average(xy[1], weights=w + 1e-9), 150, ok[col].max(), 0]
    popt, _ = curve_fit(g, xy, ok[col].values, p0=p0, maxfev=20000)
    return dict(zip(['x0', 'y0', 'sigma', 'A', 'c'], popt))


def plot_fov_position(session, cells, save_path=None):
    ok = cells[cells.n_trials > 0].copy()
    fit = fit_power_centre(ok)
    ok['dist_fit_centre'] = np.hypot(ok.x_target - fit['x0'], ok.y_target - fit['y0'])
    shape = session['mean_img'].shape
    cmap, norm = holo.colormap(), _succ_norm(int(ok.n_trials.max()))
    fill = np.full(shape + (4,), 0.0)
    for c in ok.itertuples():
        s = cell_mask(session, c.source_roi)
        fill[s['ypix'], s['xpix']] = cmap(norm(c.n_success), alpha=0.85)
    fig = plt.figure(figsize=(17, 7))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(session['mean_img'], cmap='gray', clim=holo.display_range(session['mean_img']))
    ax.imshow(fill)
    for c in ok.itertuples():
        ax.text(c.x, c.y, str(c.source_roi), fontsize=4, color='w', ha='center', va='center')
    ax.plot(shape[1] / 2, shape[0] / 2, 'x', color='w', ms=8)
    ax.plot(fit['x0'], fit['y0'], '*', color='w', ms=12)
    ax.add_patch(plt.Circle((fit['x0'], fit['y0']), abs(fit['sigma']), fill=False, color='w', ls='--', lw=0.8))
    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0], 0)
    ax.set_axis_off()
    fig.colorbar(plt.cm.ScalarMappable(norm, cmap), ax=ax, fraction=0.04, label='successful trials')
    ax.set_title('stimulated cells coloured by successful trials\n(x = FOV centre, * and dashed circle = fitted '
                 'response hotspot, centre +- sigma)', fontsize=10)
    for i, (col, lab) in enumerate((('n_success', 'successful trials'), ('amp_mean', 'mean own response (dF/F)'))):
        ax = fig.add_subplot(gs[i, 1])
        ax.scatter(ok.target_dist_centre_px, ok[col], s=14, c=ok.n_success, cmap=cmap, norm=norm)
        rho, p = stats.spearmanr(ok.target_dist_centre_px, ok[col])
        ax.set_title(f'Spearman rho={rho:.2f}, p={p:.2g}', fontsize=9)
        ax.set_xlabel('target distance from FOV centre (px)')
        ax.set_ylabel(lab)
        _despine(ax)
    # success by radial bin and by x / y separately (power can fall off anisotropically)
    ax = fig.add_subplot(gs[0, 2])
    bins = np.arange(0, ok.target_dist_centre_px.max() + 40, 40)
    which = np.digitize(ok.target_dist_centre_px, bins) - 1
    frac = [ok.n_success[which == i].mean() if (which == i).any() else np.nan for i in range(len(bins) - 1)]
    cnt = [(which == i).sum() for i in range(len(bins) - 1)]
    ax.bar(bins[:-1] + 20, frac, width=36, color=SUCC_COL)
    for x, f, n in zip(bins[:-1] + 20, frac, cnt):
        ax.text(x, (f if np.isfinite(f) else 0) + 0.1, f'n={n}', ha='center', fontsize=7)
    ax.set_xlabel('target distance from FOV centre (px)')
    ax.set_ylabel('mean successful trials')
    _despine(ax)
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(ok.dist_fit_centre, ok.n_success, s=14, c=ok.n_success, cmap=cmap, norm=norm)
    rho, p = stats.spearmanr(ok.dist_fit_centre, ok.n_success)
    ax.set_title(f"hotspot ({fit['x0']:.0f}, {fit['y0']:.0f}), sigma {abs(fit['sigma']):.0f} px; "
                 f"rho={rho:.2f} (in-sample)", fontsize=9)
    ax.set_xlabel('target distance from fitted hotspot (px)')
    ax.set_ylabel('successful trials')
    _despine(ax)
    fig.tight_layout()
    return _save(fig, save_path)


MOTION_METRICS = [('speed', 'running speed, -1..+2 s (deg/s)'),
                  ('disp_train', 'registration displacement during train (px)'),
                  ('disp_max', 'max displacement, -0.5..+2 s (px)'),
                  ('corr_train', 'registration corrXY during train (/ median)'),
                  ('red_ratio', 'cell red F, 1 s before train (/ cell median)')]


def motion_stats(trials):
    """Successful vs failed trials of mixed cells (1..n-1 successes): pooled Mann-Whitney and paired
    within-cell Wilcoxon on per-cell means. One row per metric."""
    n_succ = trials.groupby('source_roi').success.transform('sum')
    n_tr = trials.groupby('source_roi').success.transform('size')
    mixed = trials[(n_succ > 0) & (n_succ < n_tr)]
    rows = []
    for m, _ in MOTION_METRICS:
        a, b = mixed[mixed.success][m].dropna(), mixed[~mixed.success][m].dropna()
        per = mixed.groupby(['source_roi', 'success'])[m].mean().unstack().dropna()
        rows.append({'metric': m, 'n_cells': mixed.source_roi.nunique(), 'n_success_trials': len(a),
                     'n_failed_trials': len(b), 'median_success': a.median(), 'median_failed': b.median(),
                     'p_pooled': stats.mannwhitneyu(a, b).pvalue if len(a) and len(b) else np.nan,
                     'n_paired_cells': len(per),
                     'p_paired': stats.wilcoxon(per[True], per[False]).pvalue if len(per) > 5 else np.nan})
    return pd.DataFrame(rows), mixed


def plot_motion(trials, save_path=None):
    st, mixed = motion_stats(trials)
    st = st.set_index('metric')
    fig, axes = plt.subplots(2, len(MOTION_METRICS), figsize=(4 * len(MOTION_METRICS), 7.5))
    for j, (m, lab) in enumerate(MOTION_METRICS):
        a, b = mixed[mixed.success][m].dropna(), mixed[~mixed.success][m].dropna()
        ax = axes[0, j]
        if len(a) and len(b):
            lo, hi = np.nanpercentile(np.r_[a, b], [0.5, 99.5])
            bins = np.linspace(lo, hi, 30)
            ax.hist(b, bins, density=True, color=FAIL_COL, alpha=0.6, label=f'failed (n={len(b)})')
            ax.hist(a, bins, density=True, color=SUCC_COL, alpha=0.6, label=f'successful (n={len(a)})')
        ax.set_xlabel(lab, fontsize=9)
        ax.set_title(f"pooled Mann-Whitney p={st.loc[m, 'p_pooled']:.2g}", fontsize=9)
        ax.legend(frameon=False, fontsize=7)
        _despine(ax)
        ax = axes[1, j]
        per = mixed.groupby(['source_roi', 'success'])[m].mean().unstack().dropna()
        for _, r in per.iterrows():
            ax.plot([0, 1], [r[False], r[True]], color='0.7', lw=0.6)
        if len(per):
            ax.plot([0, 1], [per[False].median(), per[True].median()], 'o-', color=SUCC_COL, lw=2)
        ax.set_xticks([0, 1], ['failed', 'successful'])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylabel(lab, fontsize=9)
        ax.set_title(f"per-cell means, Wilcoxon p={st.loc[m, 'p_paired']:.2g} (n={int(st.loc[m, 'n_paired_cells'])})",
                     fontsize=9)
        _despine(ax)
    fig.suptitle(f"Motion in successful vs failed trials of cells with 1-9 successes "
                 f"({mixed.source_roi.nunique()} cells); top: pooled trials, bottom: paired per cell")
    fig.tight_layout()
    return _save(fig, save_path), st


def adaptation_stats(trials):
    """Cells with >= 1 success: per-cell linear slope of amp vs rep, Wilcoxon of slopes vs 0."""
    ever = trials.groupby('source_roi').success.transform('any')
    a = trials[ever].copy()
    a['amp_norm'] = (a.amp - a.groupby('source_roi').amp.transform('mean')) / a.groupby('source_roi').amp.transform('std')
    slopes = a.groupby('source_roi').apply(lambda d: stats.linregress(d.rep, d.amp).slope, include_groups=False)
    p = stats.wilcoxon(slopes).pvalue if len(slopes) > 5 else np.nan
    return a, slopes, p


def plot_adaptation(trials, save_path=None):
    a, slopes, p = adaptation_stats(trials)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    for ax, col, lab in ((axes[0], 'amp', 'response (dF/F)'), (axes[1], 'amp_norm', 'response, z-scored within cell')):
        for _, d in a.groupby('source_roi'):
            ax.plot(d.rep, d[col], color='0.8', lw=0.5)
        g = a.groupby('rep')[col]
        ax.errorbar(g.mean().index, g.mean(), yerr=g.sem(), color=SUCC_COL, lw=2, marker='o', ms=4)
        ax.set_xlabel('repeat (trial number of the cell)')
        ax.set_ylabel(lab)
        _despine(ax)
    axes[0].set_title(f'{a.source_roi.nunique()} cells with >= 1 successful trial (grey) and mean +- sem', fontsize=9)
    ax = axes[2]
    g = a.groupby('rep').success
    ax.plot(g.mean().index, g.mean(), 'o-', color=SUCC_COL)
    ax.set_ylim(0, 1)
    ax.set_xlabel('repeat')
    ax.set_ylabel('fraction successful')
    rho, pr = stats.spearmanr(a.rep, a.success)
    ax.set_title(f'Spearman (all trials) rho={rho:.2f}, p={pr:.2g}', fontsize=9)
    _despine(ax)
    ax = axes[3]
    ax.hist(slopes, bins=25, color=SUCC_COL)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('per-cell slope (dF/F per repeat)')
    ax.set_ylabel('cells')
    ax.set_title(f'median slope {np.median(slopes):.4f}, Wilcoxon vs 0 p={p:.2g}', fontsize=9)
    _despine(ax)
    fig.suptitle('Adaptation: response over repeats (repeat number is confounded with time in session)')
    fig.tight_layout()
    return _save(fig, save_path), slopes, p


CELL_FACTORS = ['red_mean', 'red_contrast', 'green_mean', 'green_contrast', 'n_pix', 'target_dist_centre_px',
                'target_to_cell_px', 'speed', 'disp_train', 'disp_max', 'corr_train', 'red_ratio']


def cell_factor_table(cells, trials):
    per = trials.groupby('source_roi')[[m for m, _ in MOTION_METRICS]].mean()
    c = cells.merge(per, left_on='source_roi', right_index=True, how='left')
    ok = c[c.n_trials > 0]
    rows = []
    for f in CELL_FACTORS:
        for y in ('n_success', 'amp_mean'):
            rho, p = stats.spearmanr(ok[f], ok[y], nan_policy='omit')
            rows.append({'factor': f, 'outcome': y, 'rho': rho, 'p': p, 'n': int(ok[f].notna().sum())})
    t = pd.DataFrame(rows)
    t['q'] = holo.bh_fdr(t.p)
    return c, t


def plot_cell_factors(table, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, y in zip(axes, ('n_success', 'amp_mean')):
        d = table[table.outcome == y].set_index('factor').loc[CELL_FACTORS]
        cols = [SUCC_COL if q < 0.05 else '0.7' for q in d.q]
        ax.barh(np.arange(len(d)), d.rho, color=cols)
        for i, (r, p) in enumerate(zip(d.rho, d.p)):
            ax.text(r + np.sign(r) * 0.01, i, f'p={p:.2g}', va='center', ha='left' if r >= 0 else 'right', fontsize=7)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_yticks(np.arange(len(d)), d.index)
        ax.invert_yaxis()
        ax.set_xlabel(f'Spearman rho with {y}')
        ax.set_xlim(-1, 1)
        _despine(ax)
    fig.suptitle('Per-cell factors vs stimulation outcome (coloured: BH-FDR q < 0.05 over all factor x outcome tests); '
                 'motion factors are means over the cell\'s own trials')
    fig.tight_layout()
    return _save(fig, save_path)


def run(mouse, holo_tag, source_tag, project_root=hio.PROJECT_ROOT, holo_suffix='SR', show=False):
    session = holo.load_holo_session(mouse, holo_tag, source_tag, project_root, holo_suffix)
    resp, trials = load_responses(session)
    save_dir = session['paths']['analysis']
    save_dir.mkdir(parents=True, exist_ok=True)
    _p = lambda name: save_dir / f'{name}.png'
    out = session['paths']['output']

    print('Cell features (brightness, position) ...')
    cells = cell_features(session, resp)
    print('Trial motion (wheel, registration offsets, red channel) ...')
    trials = trial_motion(session, trials)
    cells, table = cell_factor_table(cells, trials)
    cells.to_csv(out / 'response_factors_cells.csv', index=False)
    trials.to_csv(out / 'response_factors_trials.csv', index=False)
    table.to_csv(out / 'response_factors_correlations.csv', index=False)

    figs = {'red': plot_red_intensity(cells, _p('rf_1_red_intensity')),
            'fov': plot_fov_position(session, cells, _p('rf_2_fov_position'))}
    figs['motion'], mst = plot_motion(trials, _p('rf_3_motion'))
    figs['adapt'], slopes, p_adapt = plot_adaptation(trials, _p('rf_4_adaptation'))
    figs['factors'] = plot_cell_factors(table, _p('rf_5_cell_factors'))

    print('\nPer-cell factors vs outcome (Spearman, BH-FDR across the table):')
    print(table.pivot(index='factor', columns='outcome', values=['rho', 'p', 'q']).round(3).loc[CELL_FACTORS].to_string())
    print('\nMotion, successful vs failed trials of mixed cells:')
    print(mst.round(4).to_string())
    print(f'\nAdaptation: {len(slopes)} cells with >= 1 success, median slope {np.median(slopes):.4f} dF/F per repeat, '
          f'{(slopes < 0).sum()} negative, Wilcoxon vs 0 p={p_adapt:.2g}')
    print(f'Saved rf_*.png to {save_dir} and response_factors_*.csv to {out}')
    if show:
        plt.show()
    else:
        for f in figs.values():
            plt.close(f)
    return cells, trials, table


def _cli():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('mouse')
    p.add_argument('holo_tag')
    p.add_argument('source_tag')
    p.add_argument('--project-root', default=str(hio.PROJECT_ROOT))
    p.add_argument('--holo-suffix', default='SR')
    p.add_argument('--show', action='store_true')
    a = p.parse_args()
    run(a.mouse, a.holo_tag, a.source_tag, ceph_path(a.project_root), a.holo_suffix, show=a.show)


if __name__ == '__main__':
    _cli()

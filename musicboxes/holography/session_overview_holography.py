"""Overview plots for a holographic photostimulation session.

Produces (saved to analysis/sub-<mouse>/<holo label>/, e.g. .../analysis/sub-001/holo-001/):
  1_fov_masks             mean images (chan1, chan2) with the masks of stimulated cells that have a
                          holo ROI, stimulated cells with only a projected source mask, and all other
                          holo ROIs in three colours; + marks the stimulation targets
  2_psth_stim_cells       one panel per stimulated cell: its dF/F around its own stimulations (mean +-
                          sem over trials) and around the blank trials (control); * = responsive
                          (own > far-target null, Mann-Whitney, BH-FDR q < 0.05 across cells)
  3_psth_heatmap          trial-averaged dF/F of all stimulated cells to their own stimulation, sorted by
                          response amplitude, next to the same cells on blank trials
  4_response_summary      own vs blank amplitude per cell, z-score against responses to other cells'
                          stimulation, off-target responses of all holo ROIs vs distance to the target,
                          and response vs target-to-ROI distance with and without the FOV-shift correction
  5_alignment_qc          barcode fit residuals, frame and train timing, wheel speed
  6_phase_sorted          responses of stimulated cells ordered by their preferred goal-progress phase in
                          the source session (holo_functions.stim_cells_by_phase)
  7_reliability           histogram of successful trials per cell (trial amplitude > 95th percentile of
                          the cell's far-target null) against the binomial chance expectation, success
                          vs mean amplitude, and trial-by-trial heatmaps of example cells across the
                          reliability range
  8_trial_heatmaps        trial-by-trial heatmaps (one row per trial) of every stimulated cell
plus target_responses.csv and target_trial_responses.csv (holo_functions.target_responses) in the
output folder.

Same structure as musicboxes/session_overview.py: plot_xxx(..., save_path) returns a figure,
run_overview runs everything, _cli parses the command line. Run from the repo root (bonsai_abcd env)
after match_rois.py, nwb_holography.py and stim_table.py:

    python -m musicboxes.holography.session_overview_holography 001 holo-001 seq3-007 [--show]

Traces are dF/F (holo_functions.ensure_dfof: neuropil-corrected dF over the raw-F baseline).
Windows (seconds from train onset): PSTH -2..6, baseline -2..0, response 0.5..2. The spiral train
lasts 0.475 s; it is shaded in the PSTHs and its frames are kept (no light artefact in holo-001).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ceph_paths import ceph_path
import musicboxes.holography.holo_io as hio
import musicboxes.holography.holo_functions as holo

WINDOW, BASELINE, RESPONSE = (-2, 6), (-2, 0), (0.5, 2)
COL = holo.ROLE_COLOURS


def _save(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def _despine(ax):
    ax.spines[['top', 'right']].set_visible(False)


def _train_len(session):
    t = session['trials']
    return float(np.median(t.offset_time - t.onset_time))


def plot_fov_masks(session, save_path=None):
    shape = session['mean_img'].shape
    cells = holo.stimulated_cells(session)
    stim_holo = set(cells.holo_roi[cells.status == 'matched'])
    stat = session['stat']
    other = [s for i, s in enumerate(stat) if i not in stim_holo]
    matched = [stat[i] for i in sorted(stim_holo)]
    tg = session['trials'].drop_duplicates('condition')
    imgs = [('chan1 mean', session['mean_img'])] + ([('chan2 mean', session['mean_img2'])]
                                                     if session['mean_img2'] is not None else [])
    fig, axes = plt.subplots(1, len(imgs), figsize=(7.5 * len(imgs), 8))
    for ax, (name, img) in zip(np.atleast_1d(axes), imgs):
        ax.imshow(img, cmap='gray', clim=holo.display_range(img))
        ax.imshow(holo.outline_overlay(other, shape, COL['other'], alpha=0.8))
        ax.imshow(holo.outline_overlay(matched, shape, COL['matched']))
        ax.imshow(holo.outline_overlay(session['stat_projected'], shape, COL['projected']))
        ax.plot(tg.target_x_holo, tg.target_y_holo, '+', color='w', ms=4, mew=0.6)
        ax.set_title(name)
        ax.set_axis_off()
    fig.suptitle(f"sub-{session['mouse']} {session['holo_tag']}: stimulated with holo ROI "
                 f"({len(matched)}, pink), stimulated with projected {session['source_tag']} mask "
                 f"({len(session['stat_projected'])}, blue), other holo ROIs ({len(other)}, yellow); + = target")
    fig.tight_layout()
    return _save(fig, save_path)


def _cell_psths(session, source_roi):
    tr = holo.trace_for(session, source_roi)
    if tr is None:
        return None
    t, own = holo.psth(tr, holo.onsets(session, source_roi)[0], session['fs'], WINDOW)
    _, blank = holo.psth(tr, holo.onsets(session, blank=True)[0], session['fs'], WINDOW)
    return t, own, blank


def plot_psth_grid(session, resp, save_path=None):
    cells = holo.stimulated_cells(session)
    n = len(cells)
    ncol = 10
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 1.6 * nrow), sharex=True)
    train = _train_len(session)
    resp = resp.set_index('source_roi')
    for ax, c in zip(axes.flat, cells.itertuples()):
        p = _cell_psths(session, c.source_roi)
        ax.axvspan(0, train, color=COL['other'], alpha=0.5, lw=0)
        if p is not None:
            t, own, blank = p
            for m, col in ((blank, COL['extra']), (own, COL['matched'] if c.status == 'matched' else COL['projected'])):
                mu, se = np.nanmean(m, 0), np.nanstd(m, 0) / np.sqrt(len(m))
                ax.fill_between(t, mu - se, mu + se, color=col, alpha=0.3, lw=0)
                ax.plot(t, mu, color=col, lw=0.8)
        star = '*' if c.source_roi in resp.index and bool(resp.loc[c.source_roi].get('responsive', False)) else ''
        ax.set_title(f'{c.source_roi}{star}' + ('' if c.status == 'matched' else ' (proj)'), fontsize=7)
        ax.tick_params(labelsize=6)
        _despine(ax)
    for ax in list(axes.flat)[n:]:
        ax.set_axis_off()
    fig.suptitle(f"dF/F around own stimulation (coloured) and blank trials (brown), mean +- sem; "
                 f"{session['source_tag']} ROI id, * = responsive (own > far-target null, FDR q < 0.05)", fontsize=10)
    fig.supxlabel('time from train onset (s)')
    fig.tight_layout()
    return _save(fig, save_path)


def plot_psth_heatmap(session, resp, order=None, ylabel='stimulated cells (sorted by response)', save_path=None):
    cells = holo.stimulated_cells(session)
    if order is None:
        order = resp.sort_values('amp_mean', ascending=False, na_position='last').source_roi.values
    mats = {'own stimulation': [], 'blank trials': []}
    t = None
    for r in order:
        p = _cell_psths(session, r)
        if p is None:
            continue
        t, own, blank = p
        mats['own stimulation'].append(np.nanmean(own, 0))
        mats['blank trials'].append(np.nanmean(blank, 0))
    lim = np.nanpercentile(np.abs(mats['own stimulation']), 99)
    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True)
    for ax, (name, m) in zip(axes, mats.items()):
        im = ax.imshow(np.array(m), aspect='auto', cmap=holo.colormap(), vmin=-lim / 3, vmax=lim,
                       extent=[t[0], t[-1], len(m), 0], interpolation='none')
        ax.axvline(0, color='w', lw=0.8)
        ax.axvline(_train_len(session), color='w', lw=0.8, ls='--')
        ax.set_title(name)
        ax.set_xlabel('time from train onset (s)')
    axes[0].set_ylabel(ylabel)
    fig.colorbar(im, ax=axes, fraction=0.03, label=f"{session['units']} (trial mean)")
    return _save(fig, save_path)


def plot_response_summary(session, resp, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    ok = resp[resp.n_trials > 0]
    ax = axes[0]
    for status, col in (('matched', COL['matched']), ('projected', COL['projected'])):
        r = ok[ok.status == status]
        ax.errorbar(r.blank_mean, r.amp_mean, yerr=r.amp_sem, fmt='o', ms=3, color=col, alpha=0.8, lw=0.6,
                    label=f'{status} ({int(r.responsive.sum())}/{len(r)} responsive, FDR)')
    lims = [np.nanmin([ok.blank_mean.min(), ok.amp_mean.min()]), np.nanmax([ok.blank_mean.max(), ok.amp_mean.max()])]
    ax.plot(lims, lims, 'k--', lw=0.8)
    ax.set_xlabel('blank-trial response (dF/F)')
    ax.set_ylabel('own-stimulation response (dF/F)')
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.hist(ok.z_vs_other.dropna(), bins=30, color=COL['matched'])
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('own response, z vs other cells\' stimulation')
    ax.set_ylabel('stimulated cells')

    ax = axes[2]
    amp, conds, dist = holo.response_matrix(session, WINDOW, BASELINE, RESPONSE)
    cond_src = session['trials'].drop_duplicates('condition').set_index('condition').loc[conds]
    own_holo = cond_src.holo_roi.values
    is_cell = ~cond_src.is_blank.values
    self_mask = np.zeros_like(amp, bool)
    for j, h in enumerate(own_holo):
        if h >= 0:
            self_mask[h, j] = True
    d, a = dist[:, is_cell][~self_mask[:, is_cell]], amp[:, is_cell][~self_mask[:, is_cell]]
    bins = np.arange(0, 260, 10)
    which = np.digitize(d, bins) - 1
    med = [np.nanmean(a[which == i]) for i in range(len(bins) - 1)]
    sem = [np.nanstd(a[which == i]) / np.sqrt(max((which == i).sum(), 1)) for i in range(len(bins) - 1)]
    centres = bins[:-1] + 5
    ax.errorbar(centres, med, yerr=sem, color=COL['matched'], marker='o', ms=3, lw=1)
    ax.axhline(np.nanmean(amp[:, ~is_cell]), color=COL['extra'], ls='--', lw=1, label='blank trials')
    ax.set_xlabel('distance of non-target ROI to target (px)')
    ax.set_ylabel('response (dF/F)')
    ax.set_title('off-target responses (all holo ROIs)', fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[3]
    m = session['match'].set_index('source_roi')
    r = ok[ok.status == 'matched'].set_index('source_roi')
    for col_name, col, lab in (('target_to_holo_px', COL['matched'], 'FOV shift applied'),
                               ('target_to_holo_px_unshifted', COL['projected'], 'source coordinates')):
        x = m.loc[r.index, col_name]
        rho = pd.Series(x.values).corr(pd.Series(r.amp_mean.values), method='spearman')
        ax.plot(x, r.amp_mean, 'o', ms=3, color=col, alpha=0.8, label=f'{lab} (Spearman {rho:.2f})')
    ax.set_xlabel('target to holo ROI centroid (px)')
    ax.set_ylabel('own-stimulation response (dF/F)')
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        _despine(ax)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_alignment_qc(session, save_path=None):
    a = np.load(session['paths']['output'] / 'holo_alignment.npz')
    t = session['trials']
    protocol, _ = hio.load_stim_protocol(session['paths']['raw_funcimg'])
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    ax = axes[0]
    ax.plot(a['fit_residual_ms'], '.', ms=2, color=COL['matched'])
    ax.set_xlabel('shared barcode')
    ax.set_ylabel('fit residual (ms)')
    ax.set_title(f"vDAQ vs behav DAQ: slope {float(a['fit_slope']):.7f}, {int(a['fit_n_shared'])} barcodes", fontsize=10)
    ax = axes[1]
    iri = 1000 * np.diff(a['frame_times'][:int(a['n_frame_ttl'])])
    ax.hist(iri, bins=50, color=COL['matched'])
    ax.set_yscale('log')
    ax.set_xlabel('FrameTTL interval (ms)')
    ax.set_title(f"{int(a['n_frame_ttl'])} TTLs, {len(a['frame_times'])} frames", fontsize=10)
    ax = axes[2]
    ax.plot(protocol.isi_s.values[:-1], np.diff(t.onset_time.values), '.', ms=3, color=COL['matched'])
    ax.set_xlabel('csv isi_s (s)')
    ax.set_ylabel('measured train-onset interval (s)')
    ax = axes[3]
    ft = a['frame_times']
    ax.plot(ft / 60, a['wheel_speed_frames'], lw=0.3, color=COL['projected'])
    ax.plot(t.onset_time / 60, np.full(len(t), np.nanmax(a['wheel_speed_frames']) * 1.05), '|', color=COL['matched'], ms=4)
    ax.set_xlabel('time (min)')
    ax.set_ylabel('wheel speed (deg/s)')
    ax.set_title('wheel speed; | = stimulation trains', fontsize=10)
    for ax in axes:
        _despine(ax)
    fig.tight_layout()
    return _save(fig, save_path)


def _trial_heatmap(ax, session, source_roi, trials_df, lim=None, smooth_frames=5):
    """One row per own trial (rep order), black ticks on the right mark successful trials. Rows are
    smoothed with a smooth_frames boxcar for display only (single frames are too noisy to read)."""
    from scipy.ndimage import uniform_filter1d
    p = _cell_psths(session, source_roi)
    if p is None:
        ax.set_axis_off()
        return None
    t, own, _ = p
    if smooth_frames > 1:
        own = uniform_filter1d(own, smooth_frames, axis=1, mode='nearest')
    lim = lim or np.nanpercentile(np.abs(own), 99)
    im = ax.imshow(own, aspect='auto', cmap=holo.colormap(), vmin=-lim / 3, vmax=lim,
                   extent=[t[0], t[-1], len(own), 0], interpolation='none')
    ax.axvline(0, color='w', lw=0.6)
    ax.axvline(_train_len(session), color='w', lw=0.6, ls='--')
    ok = trials_df[trials_df.source_roi == source_roi].sort_values('rep').success.values
    for k in np.flatnonzero(ok):
        ax.plot(t[-1], k + 0.5, '<', color='k', ms=3, clip_on=False)
    return im


def plot_reliability(session, resp, trials_df, n_examples=8, save_path=None):
    from scipy.stats import binom
    ok = resp[resp.n_trials > 0]
    n_trials = int(ok.n_trials.max())
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, n_examples // 2 + 2, width_ratios=[1.3, 1.3] + [1] * (n_examples // 2))
    ax = fig.add_subplot(gs[0, 0])
    k = np.arange(n_trials + 1)
    counts = [np.bincount(ok.n_success[ok.status == st], minlength=n_trials + 1)[:n_trials + 1]
              for st in ('matched', 'projected')]
    ax.bar(k, counts[0], color=COL['matched'], label='matched')
    ax.bar(k, counts[1], bottom=counts[0], color=COL['projected'], label='projected')
    trial_alpha = 0.05
    ax.plot(k, len(ok) * binom.pmf(k, n_trials, trial_alpha), 'o--', color=COL['extra'], ms=4, lw=1,
            label=f'chance (binomial, p={trial_alpha})')
    ax.set_xlabel(f'successful trials (of {n_trials})')
    ax.set_ylabel('stimulated cells')
    ax.set_xticks(k)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f'{int((ok.n_success >= 3).sum())}/{len(ok)} cells succeed in >= 3 trials', fontsize=10)
    _despine(ax)

    ax = fig.add_subplot(gs[1, 0])
    for st in ('matched', 'projected'):
        r = ok[ok.status == st]
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(r))
        ax.scatter(r.n_success + jitter, r.amp_mean, s=12, color=COL[st], alpha=0.8,
                   edgecolor=np.where(r.responsive, 'k', 'none'), lw=0.6)
    ax.set_xlabel(f'successful trials (of {n_trials})')
    ax.set_ylabel(f"mean own response ({session['units']})")
    ax.set_title('black edge = responsive (FDR q < 0.05)', fontsize=10)
    _despine(ax)

    ax = fig.add_subplot(gs[:, 1])
    order = ok.sort_values(['n_success', 'amp_mean'])
    ax.barh(np.arange(len(order)), order.frac_success, color=[COL[s] for s in order.status], height=0.8)
    ax.set_yticks(np.arange(len(order)), order.source_roi, fontsize=4)
    ax.set_xlabel('fraction of successful trials')
    ax.set_ylabel(f"{session['source_tag']} ROI")
    ax.axvline(trial_alpha, color=COL['extra'], ls='--', lw=1)
    _despine(ax)

    # examples spread over the reliability range (highest amplitude within each pick)
    ranked = ok.sort_values(['n_success', 'amp_mean'], ascending=False).reset_index(drop=True)
    picks = ranked.loc[np.unique(np.linspace(0, len(ranked) - 1, n_examples).round().astype(int))]
    for i, c in enumerate(picks.itertuples()):
        ax = fig.add_subplot(gs[i % 2, 2 + i // 2])
        im = _trial_heatmap(ax, session, c.source_roi, trials_df)
        ax.set_title(f'{c.source_roi} ({c.status}): {c.n_success}/{c.n_trials}, q={c.q_vs_null:.1g}', fontsize=9)
        ax.set_xlabel('time from train onset (s)', fontsize=8)
        ax.set_ylabel('trial', fontsize=8)
        if im is not None:
            fig.colorbar(im, ax=ax, fraction=0.05, label=session['units'])
    fig.suptitle('Reliability: a trial succeeds if its amplitude exceeds the 95th percentile of the cell\'s '
                 'responses to far-target (>= 50 px) and blank trials; < = successful trial')
    fig.tight_layout()
    return _save(fig, save_path)


def plot_trial_heatmaps(session, resp, trials_df, ncol=10, save_path=None):
    ok = resp.sort_values(['n_success', 'amp_mean'], ascending=False, na_position='last')
    n = len(ok)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 1.5 * nrow), sharex=True)
    for ax, c in zip(axes.flat, ok.itertuples()):
        _trial_heatmap(ax, session, c.source_roi, trials_df)
        lab = f'{c.source_roi}' + ('' if c.status == 'matched' else ' (proj)')
        if c.n_trials > 0:
            lab += f': {c.n_success}/{c.n_trials}' + ('*' if c.responsive else '')
        ax.set_title(lab, fontsize=7)
        ax.tick_params(labelsize=5)
        ax.set_yticks([])
    for ax in list(axes.flat)[n:]:
        ax.set_axis_off()
    fig.suptitle(f"{session['units']} per trial around own stimulation (rows = trials, colour scale per cell); "
                 'title: successful trials, * = responsive (FDR q < 0.05); < = successful trial', fontsize=10)
    fig.supxlabel('time from train onset (s)')
    fig.tight_layout()
    return _save(fig, save_path)


def run_overview(mouse, holo_tag, source_tag, project_root=hio.PROJECT_ROOT, holo_suffix='SR', save_dir=None,
                 show=False):
    session = holo.load_holo_session(mouse, holo_tag, source_tag, project_root, holo_suffix)
    save_dir = Path(save_dir) if save_dir else session['paths']['analysis']
    save_dir.mkdir(parents=True, exist_ok=True)
    _p = lambda name: save_dir / f'{name}.png'
    figs = {}

    print('Responses of stimulated cells ...')
    resp, trials_df = holo.target_responses(session, WINDOW, BASELINE, RESPONSE, return_trials=True)
    resp.to_csv(session['paths']['output'] / 'target_responses.csv', index=False)
    trials_df.to_csv(session['paths']['output'] / 'target_trial_responses.csv', index=False)
    ok = resp[resp.n_trials > 0]
    print(f"  {int(ok.responsive.sum())}/{len(ok)} stimulated cells with a trace respond (own > far-target null, "
          f"BH-FDR q < 0.05): matched {int(ok.responsive[ok.status == 'matched'].sum())}/"
          f"{int((ok.status == 'matched').sum())}, projected {int(ok.responsive[ok.status == 'projected'].sum())}/"
          f"{int((ok.status == 'projected').sum())}")
    print(f"  uncorrected p < 0.05: vs null {int((ok.p_vs_null < 0.05).sum())}, vs blank {int((ok.p_vs_blank < 0.05).sum())}; "
          f"FDR q < 0.05: vs blank {int((ok.q_vs_blank < 0.05).sum())}, reliability (binomial) {int((ok.q_binom < 0.05).sum())}")
    print(f"  successful trials per cell: median {ok.n_success.median():.0f}/{int(ok.n_trials.max())}, "
          f"distribution {np.bincount(ok.n_success, minlength=int(ok.n_trials.max()) + 1).tolist()} (0..n); "
          f"null trials per cell {int(ok.n_null.min())}-{int(ok.n_null.max())}")
    print(f"  median own response {ok.amp_mean.median():.3f} {session['units']} vs blank {ok.blank_mean.median():.3f}")

    print('Plot 1/8: FOV masks')
    figs['fov'] = plot_fov_masks(session, _p('1_fov_masks'))
    print('Plot 2/8: PSTH grid')
    figs['psth_grid'] = plot_psth_grid(session, resp, _p('2_psth_stim_cells'))
    print('Plot 3/8: PSTH heatmap')
    figs['psth_heatmap'] = plot_psth_heatmap(session, resp, save_path=_p('3_psth_heatmap'))
    print('Plot 4/8: response summary')
    figs['summary'] = plot_response_summary(session, resp, _p('4_response_summary'))
    print('Plot 5/8: alignment QC')
    figs['qc'] = plot_alignment_qc(session, _p('5_alignment_qc'))
    print('Plot 6/8: phase-sorted responses')
    phase = holo.stim_cells_by_phase(mouse, holo_tag, source_tag, project_root, holo_suffix)
    figs['phase'] = plot_psth_heatmap(session, resp, order=phase.source_roi.values,
                                      ylabel='stimulated cells (sorted by preferred GP phase)',
                                      save_path=_p('6_phase_sorted'))
    print('Plot 7/8: reliability')
    figs['reliability'] = plot_reliability(session, resp, trials_df, save_path=_p('7_reliability'))
    print('Plot 8/8: trial heatmaps')
    figs['trial_heatmaps'] = plot_trial_heatmaps(session, resp, trials_df, save_path=_p('8_trial_heatmaps'))
    print(f'Saved to {save_dir}')
    if show:
        plt.show()
    else:
        for f in figs.values():
            plt.close(f)
    return session, resp, figs


def _cli():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('mouse')
    p.add_argument('holo_tag')
    p.add_argument('source_tag')
    p.add_argument('--project-root', default=str(hio.PROJECT_ROOT))
    p.add_argument('--holo-suffix', default='SR')
    p.add_argument('--show', action='store_true')
    a = p.parse_args()
    run_overview(a.mouse, a.holo_tag, a.source_tag, ceph_path(a.project_root), a.holo_suffix, show=a.show)


if __name__ == '__main__':
    _cli()

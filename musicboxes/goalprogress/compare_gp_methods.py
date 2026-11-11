"""Compare binned GP tuning (gp_learning_helpers.gp_tuning) with the von Mises fit (gp_circular_fit.vm_tuning).

Run from the repo root:

    python -m musicboxes.goalprogress.compare_gp_methods synthetic [--n-shuffles 500]
    python -m musicboxes.goalprogress.compare_gp_methods real 001 seq3-007 [--n-shuffles 1000] [--cache-dir <local>]

Outputs go to --out-dir (default ~/Desktop/gp_method_comparison), in synthetic/ and sub-<mouse>_<session>/.
Run `real` first if the synthetic segments should use that session's reward intervals: it saves them as
reward_intervals_s.npy, which `synthetic` picks up (otherwise gamma-distributed intervals, mean 12 s).

The fit is called through gp_circular_fit.vm_tuning directly (gp_learning_helpers.gp_tuning(method='vm')
wraps the same function) so the null runs can skip the refinement and the split halves. Both methods run
on the same reward -> reward segments with the same seed, so their circular-shift shuffles
are identical (see gp_circular_fit). The binned method uses BINS = 30 as gp_cell_table.py.

Synthetic cells
---------------
Event rate R0 * (1 + gain * vM(phi; mu, kappa)) per frame (Poisson, lognormal amplitudes), convolved with an
exponential GCaMP-like kernel (TAU_S) plus Gaussian noise, so the traces have calcium-like autocorrelation.
The kernel delays the response, so phase errors are measured against mu_response, the peak of each
cell's noise-free expected trace (mu itself is the peak of the event rate). gain = 0 cells are untuned (the false-positive test), and a few bimodal cells (two fields half a segment
apart) show a case the unimodal fit cannot describe.

False positives / negatives
---------------------------
Synthetic: exact (FP = significant at gain 0; FN = not significant at gain > 0).
Real data has no ground truth, so two proxies are used:
  - Misaligned-clock null: the reward train is circularly shifted by >= MIN_SHIFT_S and both methods rerun.
    The fraction of significant cells is the FP rate under real noise. Cells locked to position or to the
    lap can stay partly aligned with a shifted clock, so this is an upper bound.
  - Split-half replication: a method calls a cell on the EVEN segments; the cell replicates if the odd-half
    30-bin curve correlates with the even-half curve (r > REP_R) and the odd-half peak lies within REP_TOL
    of the method's even-half phase. Probable FP = called but does not replicate. Probable FN = not called,
    but called by the other method and replicates. The same labels are computed for the synthetic cells,
    where they can be scored against the truth.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import lfilter

import era_brewer
import musicboxes.goalprogress.gp_learning_helpers as h
import musicboxes.goalprogress.gp_circular_fit as vm

OUT_ROOT = os.path.expanduser('~/Desktop/gp_method_comparison')
BINS = 30
ALPHA = 0.05
METHODS = ['binned', 'vM fit']
METHOD_COLORS = dict(zip(METHODS, list(era_brewer.era_brew('Lover1', n=5, brew_type='discrete'))[:2]))
REP_R, REP_TOL = 0.3, 0.1
MIN_SHIFT_S = 60

# synthetic
FRAME_RATE = 45.0
R0, TAU_S, NOISE_SD = 0.5, 0.7, 0.3
GAINS = [0, 0.25, 0.5, 1, 2, 4]
SYN_KAPPAS = [1, 3, 10]
N_PER_CONDITION = 20
N_BIMODAL = 20


def _category_colors():
    lover = era_brewer.era_brew('Lover1', n=5, brew_type='discrete')
    return {'both': lover[3], 'binned only': METHOD_COLORS['binned'], 'vM only': METHOD_COLORS['vM fit'], 'neither': '0.8'}


CATEGORY_COLORS = _category_colors()


# ---------------------------------------------------------------------------
# Running both methods
# ---------------------------------------------------------------------------

def run_both(loaded, n_shuffles, seed=0, alpha=ALPHA, segments=None, halves=True, refine=True):
    segments = loaded['segments']['reward'] if segments is None else np.asarray(segments)
    gb = h.gp_tuning(loaded, bins=BINS, n_shuffles=n_shuffles, alpha=alpha, seed=seed, segments=segments,
                     method='binned')
    gv = vm.vm_tuning(loaded['dF'], segments, n_shuffles=n_shuffles, alpha=alpha, seed=seed, refine=refine,
                      halves=halves)
    gv['segments'] = segments
    return gb, gv


def significance_category(sig_b, sig_v):
    return np.select([sig_b & sig_v, sig_b, sig_v], ['both', 'binned only', 'vM only'], 'neither')


def cell_table(gb, gv):
    return pd.DataFrame(dict(
        phase_binned=gb['phase'], phase_vm=gv['phase'], p_binned=gb['p'], p_vm=gv['p'],
        sig_binned=gb['significant'], sig_vm=gv['significant'],
        category=significance_category(gb['significant'], gv['significant']),
        tuning_index=gb['tuning'], vm_r2=gv['r2'], vm_kappa=gv['kappa'], vm_fwhm=gv['fwhm'], vm_amp=gv['amp'],
        phase_even_binned=gb['phase_even'], phase_odd_binned=gb['phase_odd'],
        phase_even_vm=gv['phase_even'], phase_odd_vm=gv['phase_odd'], split_r=gb['split_r']))


def split_half_calls(loaded, segments, n_shuffles, seed=0, alpha=ALPHA):
    """Call cells on even segments with each method, validate on odd segments (see module docstring)."""
    even, odd = segments[0::2], segments[1::2]
    gb_e, gv_e = run_both(loaded, n_shuffles, seed, alpha, segments=even, halves=False)
    even_curve = h.mean_curve(gb_e['mat'])
    odd_curve = h.mean_curve(h.binned_matrix(loaded['dFn'], odd, BINS))
    r = h.rowwise_corr(even_curve, odd_curve)
    odd_peak = h.preferred_phase(odd_curve)
    called = {'binned': gb_e['significant'], 'vM fit': gv_e['significant']}
    replicates = {'binned': (r > REP_R) & (h.circular_distance(odd_peak, gb_e['phase']) <= REP_TOL),
                  'vM fit': (r > REP_R) & (h.circular_distance(odd_peak, gv_e['phase']) <= REP_TOL)}
    out = dict(r=r, odd_peak=odd_peak, even_curve=even_curve, odd_curve=odd_curve,
               phase_even={'binned': gb_e['phase'], 'vM fit': gv_e['phase']}, called=called, replicates=replicates)
    for m, other in zip(METHODS, METHODS[::-1]):
        out[f'FP {m}'] = called[m] & ~replicates[m]
        out[f'FN {m}'] = ~called[m] & called[other] & replicates[other]
    return out


def proxy_rates(sh):
    """FP rate = probable FP / called; FN rate = probable FN / cells called by either method that replicate."""
    pool = (sh['called']['binned'] & sh['replicates']['binned']) | (sh['called']['vM fit'] & sh['replicates']['vM fit'])
    rows = []
    for m in METHODS:
        rows.append(dict(method=m, n_called=int(sh['called'][m].sum()), n_FP=int(sh[f'FP {m}'].sum()),
                         n_FN=int(sh[f'FN {m}'].sum()), n_replicating_pool=int(pool.sum()),
                         FP_rate=sh[f'FP {m}'].sum() / max(sh['called'][m].sum(), 1),
                         FN_rate=sh[f'FN {m}'].sum() / max(pool.sum(), 1)))
    return pd.DataFrame(rows)


def misaligned_segments(segments, rng, min_shift):
    """Reward train circularly shifted by a random offset in [min_shift, span - min_shift] frames."""
    rewards = np.r_[segments[:, 0], segments[-1, 1]]
    r0, span = rewards[0], rewards[-1] - rewards[0]
    off = rng.integers(min_shift, span - min_shift)
    shifted = np.unique((rewards - r0 + off) % span + r0)
    return np.stack([shifted[:-1], shifted[1:]], axis=1)


def misaligned_null(loaded, segments, n_offsets, n_shuffles, seed=0, alpha=ALPHA):
    """(n_offsets, 2) fraction of cells significant per method with a shifted reward clock."""
    rng = np.random.default_rng(seed + 1)
    min_shift = int(MIN_SHIFT_S * loaded['frame_rate'])
    out = []
    for i in range(n_offsets):
        seg = misaligned_segments(segments, rng, min_shift)
        gb, gv = run_both(loaded, n_shuffles, seed + i, alpha, segments=seg, halves=False, refine=False)
        out.append([gb['significant'].mean(), gv['significant'].mean()])
        print(f'  misaligned clock {i + 1}/{n_offsets}: binned {out[-1][0]:.3f}, vM {out[-1][1]:.3f}')
    return np.array(out)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def synthetic_loaded(durations_s, rng):
    """Fake `loaded` dict (dF, dFn, segments, n_frames, frame_rate) plus the per-cell ground truth table."""
    seg_len = np.maximum(np.round(np.asarray(durations_s) * FRAME_RATE).astype(int), 2)
    pad = int(5 * FRAME_RATE)
    bounds = pad + np.r_[0, np.cumsum(seg_len)]
    segments = np.stack([bounds[:-1], bounds[1:]], axis=1)
    n_frames = bounds[-1] + pad
    phase, _ = h.frame_phase(n_frames, segments)

    truth = [dict(kind='unimodal', gain=g, kappa=k, mu=rng.random())
             for g in GAINS for k in SYN_KAPPAS for _ in range(N_PER_CONDITION)]
    truth += [dict(kind='bimodal', gain=2.0, kappa=5, mu=rng.random()) for _ in range(N_BIMODAL)]
    truth = pd.DataFrame(truth)

    inside = np.isfinite(phase)
    dt = 1 / FRAME_RATE
    decay = np.exp(-dt / TAU_S)
    dF = np.empty((len(truth), n_frames), dtype=np.float32)
    fine = np.minimum((phase[inside] * 200).astype(int), 199)
    mu_response = np.full(len(truth), np.nan)
    for i, c in truth.iterrows():
        shape = np.zeros(n_frames)
        shape[inside] = vm.vm_shape(phase[inside], c.mu, c.kappa)
        if c.kind == 'bimodal':
            shape[inside] = 0.5 * (shape[inside] + vm.vm_shape(phase[inside], c.mu + 0.5, c.kappa))
        rate = R0 * (1 + c.gain * shape)
        events = rng.poisson(rate * dt) * rng.lognormal(0, 0.3, n_frames)
        dF[i] = lfilter([1], [1, -decay], events) + rng.normal(0, NOISE_SD, n_frames)
        if c.gain > 0:   # peak of the noise-free response, which lags mu by the kernel's decay
            expected = lfilter([1], [1, -decay], rate * dt)[inside]
            mu_response[i] = (np.argmax(np.bincount(fine, expected, 200) / np.bincount(fine, minlength=200)) + 0.5) / 200
    dF = dF.astype(float)
    truth['mu_response'] = mu_response
    med = np.median(dF, axis=1, keepdims=True)
    noise = 1.4826 * np.median(np.abs(dF - med), axis=1, keepdims=True)
    loaded = dict(dF=dF, dFn=(dF - med) / noise, segments={'reward': segments}, n_frames=n_frames,
                  frame_rate=FRAME_RATE)
    truth['tuned'] = truth.gain > 0
    truth['fwhm'] = vm.vm_fwhm(truth.kappa)
    return loaded, truth


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def cv_sorted(curves_odd, phase_even, rows):
    rows = np.flatnonzero(rows)
    order = rows[np.argsort(phase_even[rows])]
    return h.zscore_rows(curves_odd[order])


def plot_cv_heatmaps(gb, gv, title, path):
    """Significant cells per method, sorted by even-half phase, showing z-scored odd-half 30-bin curves."""
    odd = h.mean_curve(gb['mat'][:, 1::2])
    panels = {f"binned: {gb['significant'].sum()} cells, sorted by binned peak (even)":
              cv_sorted(odd, gb['phase_even'], gb['significant']),
              f"vM fit: {gv['significant'].sum()} cells, sorted by vM mu (even)":
              cv_sorted(odd, gv['phase_even'], gv['significant'])}
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5), sharex=True)
    im = h.heatmap_row(axes, panels, lambda t: t, vmin=-2, vmax=3)
    axes[0].set_ylabel('cell (sorted on even segments)')
    for ax in axes:
        ax.set_xlabel('GP phase (odd segments)')
        ax.plot([0, 1], [0, ax.get_ylim()[0]], color='w', lw=0.6, ls='--')
    fig.colorbar(im, ax=axes, shrink=0.6, label='z-scored dF (odd segments)')
    fig.suptitle(f'{title}: cross-validated sorting', fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_example_cells(loaded, gb, gv, cells, labels, title, path, n_fine=100):
    """Per-frame dF averaged over all frames in each of n_fine phase bins (grey; the raw per-frame scatter
    spans far more than the tuning and hides it), 30-bin mean curve and fitted von Mises, with both phases."""
    segments = gv['segments']
    layout = vm.frame_layout(segments)
    raw_curves = h.mean_curve(h.binned_matrix(loaded['dF'], segments, BINS))
    centres = (np.arange(BINS) + 0.5) / BINS
    fine = np.minimum((layout['phase'] * n_fine).astype(int), n_fine - 1)
    fine_counts = np.bincount(fine, minlength=n_fine)
    grid = np.linspace(0, 1, 200)
    ncol = 4
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), squeeze=False)
    for ax, c, lab in zip(axes.ravel(), cells, labels):
        fine_mean = np.bincount(fine, loaded['dF'][c, layout['frames']], n_fine) / fine_counts
        fit = vm.vm_model(grid, gv['baseline'][c], gv['amp'][c], gv['phase'][c], gv['kappa'][c])
        ax.scatter((np.arange(n_fine) + 0.5) / n_fine, fine_mean, s=5, color='0.65', lw=0, label=f'{n_fine}-bin mean')
        ax.plot(centres, raw_curves[c], color=METHOD_COLORS['binned'], lw=1.6, label='30-bin mean')
        ax.plot(grid, fit, color=METHOD_COLORS['vM fit'], lw=1.6, label='vM fit')
        ax.axvline(gb['phase'][c], color=METHOD_COLORS['binned'], ls='--', lw=1)
        ax.axvline(gv['phase'][c], color=METHOD_COLORS['vM fit'], ls=':', lw=1.4)
        lo, hi = np.nanmin(np.r_[fine_mean, fit]), np.nanmax(np.r_[fine_mean, fit])
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        ax.set_xlim(0, 1)
        ax.set_title(f"{lab}cell {c}: p_bin {gb['p'][c]:.3g}, p_vM {gv['p'][c]:.3g}\n"
                     f"phase bin {gb['phase'][c]:.2f} / vM {gv['phase'][c]:.2f}, FWHM {gv['fwhm'][c]:.2f}",
                     fontsize=7.5)
        h.style_axes(ax)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(cells):]:
        ax.axis('off')
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[-1, 0].set_xlabel('GP phase')
    axes[-1, 0].set_ylabel('dF')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_split_examples(sh, cells_by_cat, title, path):
    """Even- and odd-half 30-bin curves of example probable FP / FN cells, with each method's even phase."""
    cats = [k for k, v in cells_by_cat.items() if len(v)]
    if not cats:
        return
    ncol = max(len(v) for v in cells_by_cat.values())
    fig, axes = plt.subplots(len(cats), ncol, figsize=(3 * ncol, 2.3 * len(cats)), squeeze=False)
    centres = (np.arange(BINS) + 0.5) / BINS
    for row, cat in zip(axes, cats):
        for ax, c in zip(row, cells_by_cat[cat]):
            ax.plot(centres, sh['even_curve'][c], color='0.3', lw=1.4, label='even')
            ax.plot(centres, sh['odd_curve'][c], color='0.3', lw=1.4, ls='--', label='odd')
            for m in METHODS:
                ax.axvline(sh['phase_even'][m][c], color=METHOD_COLORS[m], lw=1.2,
                           ls='-' if sh['called'][m][c] else ':')
            ax.set_title(f'{cat}: cell {c}\nr(even, odd) = {sh["r"][c]:.2f}', fontsize=7.5)
            ax.set_xlim(0, 1)
            h.style_axes(ax)
            ax.tick_params(labelsize=7)
        for ax in row[len(cells_by_cat[cat]):]:
            ax.axis('off')
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[-1, 0].set_xlabel('GP phase')
    axes[-1, 0].set_ylabel('dF (noise units)')
    fig.suptitle(f'{title}\nvertical lines: even-half phase (binned / vM), solid = called on even half, '
                 f'dotted = not called', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_overview(table, gb, gv, title, path):
    """(a) phase vs phase, (b) p vs p, (c) split-half phase consistency."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    ax = axes[0]
    for cat, col in CATEGORY_COLORS.items():
        sel = table.category == cat
        ax.scatter(table.phase_binned[sel], table.phase_vm[sel], s=12, color=col, label=f'{cat} ({sel.sum()})',
                   zorder=1 if cat == 'neither' else 2)
    sig = table.sig_binned | table.sig_vm
    ax.plot([0, 1], [0, 1], color='0.5', lw=0.8, ls='--')
    ax.set(xlabel='binned peak phase', ylabel='vM fit mu', xlim=(0, 1), ylim=(0, 1), aspect='equal',
           title=f'preferred phase (circ r = {h.circular_corr(table.phase_binned[sig], table.phase_vm[sig]):.2f} '
                 f'on significant)')
    ax.legend(fontsize=7, frameon=False, title='significant', title_fontsize=7)

    ax = axes[1]
    for cat, col in CATEGORY_COLORS.items():
        sel = table.category == cat
        ax.scatter(table.p_binned[sel], table.p_vm[sel], s=12, color=col)
    ax.set(xscale='log', yscale='log', xlabel='p binned', ylabel='p vM fit', title='shuffle p-values')
    ax.axvline(ALPHA, color='0.5', lw=0.8, ls='--')
    ax.axhline(ALPHA, color='0.5', lw=0.8, ls='--')

    ax = axes[2]
    edges = np.linspace(0, 0.5, 21)
    for m, (e, o) in {'binned': ('phase_even_binned', 'phase_odd_binned'),
                      'vM fit': ('phase_even_vm', 'phase_odd_vm')}.items():
        d = h.circular_distance(table[e][sig], table[o][sig])
        ax.hist(d, bins=edges, histtype='step', lw=1.8, color=METHOD_COLORS[m],
                label=f'{m}: median {np.median(d):.3f}')
    ax.set(xlabel='|phase even - phase odd|', ylabel='cells', title='split-half phase stability\n(significant by either)')
    ax.legend(fontsize=7, frameon=False)
    for ax in axes:
        h.style_axes(ax)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_phase_distribution(table, title, path, n_bins=20):
    """Preferred-phase histograms of each method's significant cells, plus the cells significant by both."""
    edges = np.linspace(0, 1, n_bins + 1)
    panels = [('significant per method', {'binned': table.sig_binned, 'vM fit': table.sig_vm}),
              ('significant by both', {'binned': table.sig_binned & table.sig_vm,
                                       'vM fit': table.sig_binned & table.sig_vm})]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (name, sel) in zip(axes, panels):
        for m, col in [('binned', 'phase_binned'), ('vM fit', 'phase_vm')]:
            ax.hist(table[col][sel[m]], bins=edges, histtype='step', lw=1.8, color=METHOD_COLORS[m],
                    label=f'{m} (n = {sel[m].sum()})')
        ax.set(xlabel='preferred GP phase (0 = reward)', xlim=(0, 1), title=name)
        ax.legend(fontsize=7, frameon=False)
        h.style_axes(ax)
    axes[0].set_ylabel('cells')
    fig.suptitle(f'{title}: preferred phase distribution ({n_bins} histogram bins)', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_fp_fn_bars(rates, path, title):
    """rates: DataFrame(source, method, FP_rate, FN_rate)."""
    sources = list(dict.fromkeys(rates.source))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(sources))
    for ax, col in zip(axes, ['FP_rate', 'FN_rate']):
        for j, m in enumerate(METHODS):
            vals = [rates[(rates.source == s) & (rates.method == m)][col].mean() for s in sources]
            ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=METHOD_COLORS[m], label=m)
        ax.set_xticks(x, sources, fontsize=7, rotation=15)
        ax.set_ylabel(col.replace('_', ' '))
        h.style_axes(ax)
    axes[0].axhline(ALPHA, color='0.5', ls='--', lw=0.8)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def example_cells(table, n=4):
    """Cells significant by either method where the two phases agree most (strongest first) and disagree most."""
    sig = table[table.sig_binned | table.sig_vm].copy()
    sig['dist'] = h.circular_distance(sig.phase_binned, sig.phase_vm)
    agree = sig[sig.dist <= 0.05].sort_values('p_vm').index[:n]
    disagree = sig.sort_values('dist', ascending=False).index[:n]
    return list(agree) + list(disagree), ['agree: '] * len(agree) + ['disagree: '] * len(disagree)


def split_examples(sh, n=4, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for m in METHODS:
        for kind in ['FP', 'FN']:
            idx = np.flatnonzero(sh[f'{kind} {m}'])
            out[f'probable {kind} {m}'] = rng.choice(idx, min(n, len(idx)), replace=False) if len(idx) else []
    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_synthetic(out_dir, n_shuffles, seed=0):
    out_dir = Path(out_dir) / 'synthetic'
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    real_intervals = sorted(Path(out_dir).parent.glob('sub-*/reward_intervals_s.npy'))
    if real_intervals:
        durations = np.load(real_intervals[0])
        source = f'reward intervals of {real_intervals[0].parent.name}'
    else:
        durations = rng.gamma(4, 3, 100)
        source = 'gamma(4, 3 s) intervals'
    loaded, truth = synthetic_loaded(durations, rng)
    print(f'synthetic: {len(truth)} cells, {len(durations)} segments ({source}), {loaded["n_frames"]} frames')
    segments = loaded['segments']['reward']
    gb, gv = run_both(loaded, n_shuffles, seed)
    table = pd.concat([truth, cell_table(gb, gv)], axis=1)
    for m, col in [('binned', 'phase_binned'), ('vm', 'phase_vm')]:
        table[f'err_{m}'] = h.circular_distance(table[col], table.mu_response)
    sh = split_half_calls(loaded, segments, n_shuffles, seed)
    for m in METHODS:
        table[f'probable_FP_{m}'] = sh[f'FP {m}']
        table[f'probable_FN_{m}'] = sh[f'FN {m}']
    table.to_csv(out_dir / 'synthetic_cells.csv', index=False)

    uni = table[table.kind == 'unimodal']
    # detection, phase error, phase scatter, width recovery
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for j, k in enumerate(SYN_KAPPAS):
        ax = axes[0, j]
        sub = uni[uni.kappa == k]
        for m, col in [('binned', 'sig_binned'), ('vM fit', 'sig_vm')]:
            ax.plot(range(len(GAINS)), sub.groupby('gain')[col].mean().reindex(GAINS), 'o-', color=METHOD_COLORS[m], label=m)
        ax.axhline(ALPHA, color='0.5', ls='--', lw=0.8)
        ax.set_xticks(range(len(GAINS)), [f'{g:g}' for g in GAINS])
        ax.set(xlabel='gain (peak rate / baseline - 1)', ylabel='fraction significant', ylim=(-0.02, 1.02),
               title=f'detection, kappa = {k} (FWHM {vm.vm_fwhm(k):.2f})')
        ax2 = axes[1, j]
        for m, col in [('binned', 'err_binned'), ('vM fit', 'err_vm')]:
            g = sub[sub.gain > 0].groupby('gain')[col]
            gains = [g_ for g_ in GAINS if g_ > 0]
            med = g.median().reindex(gains)
            x = np.arange(1, len(GAINS))
            ax2.plot(x, med, 'o-', color=METHOD_COLORS[m], label=m)
            ax2.fill_between(x, g.quantile(0.25).reindex(gains), g.quantile(0.75).reindex(gains),
                             color=METHOD_COLORS[m], alpha=0.2, lw=0)
        ax2.axhline(0.25, color='0.5', ls=':', lw=0.8)
        ax2.set_xticks(range(len(GAINS)), [f'{g:g}' for g in GAINS])
        ax2.set(xlim=(0.5, len(GAINS) - 0.5), xlabel='gain', ylabel='|phase error| (median, IQR)', ylim=(0, 0.3),
                title=f'phase error, all tuned cells, kappa = {k}')
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax in axes.ravel():
        h.style_axes(ax)
    fig.suptitle(f'Synthetic cells ({N_PER_CONDITION} per gain x kappa, {len(segments)} segments, '
                 f'{n_shuffles} shuffles, alpha {ALPHA}); dashed = alpha, dotted = chance error', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / '1_detection_and_phase_error.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    tuned = uni[uni.gain >= 1]
    for ax, (m, col) in zip(axes[:2], [('binned', 'phase_binned'), ('vM fit', 'phase_vm')]):
        sc = ax.scatter(tuned.mu_response, tuned[col], c=np.log2(tuned.gain), s=10,
                        cmap=h.PHASE_CMAP)
        ax.set(xlabel='true response peak', ylabel=f'{m} phase', title=f'{m}, gain >= 1', aspect='equal', xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(sc, ax=axes[1], label='log2 gain')
    ax = axes[2]
    for k in SYN_KAPPAS:
        sub = tuned[tuned.kappa == k]
        ax.scatter(np.full(len(sub), vm.vm_fwhm(k)) + rng.normal(0, 0.008, len(sub)), sub.vm_fwhm, s=10,
                   color=METHOD_COLORS['vM fit'], alpha=0.6)
    ax.plot([0, 0.6], [0, 0.6], color='0.5', ls='--', lw=0.8)
    ax.set(xlabel='true FWHM', ylabel='fitted FWHM', title='vM width recovery (gain >= 1)')
    for ax in axes:
        h.style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / '2_phase_and_width_recovery.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    plot_cv_heatmaps(gb, gv, 'Synthetic', out_dir / '3_cv_sorted_heatmaps.png')

    # FP / FN: truth and proxy
    rows = []
    for m, col in [('binned', 'sig_binned'), ('vM fit', 'sig_vm')]:
        rows.append(dict(source='synthetic truth', method=m, FP_rate=uni[~uni.tuned][col].mean(),
                         FN_rate=1 - uni[uni.tuned][col].mean()))
    proxy = proxy_rates(sh)
    for _, r in proxy.iterrows():
        rows.append(dict(source='synthetic split-half proxy', method=r.method, FP_rate=r.FP_rate, FN_rate=r.FN_rate))
    rates = pd.DataFrame(rows)
    plot_fp_fn_bars(rates, out_dir / '4_fp_fn_rates.png', 'Synthetic: FP rate = significant among untuned (truth) / '
                    'probable FP among called (proxy); FN rate = missed tuned cells (truth) / probable FN (proxy)')
    plot_split_examples(sh, split_examples(sh), 'Synthetic: split-half proxy examples', out_dir / '5_fp_fn_examples.png')

    lines = [f'Synthetic run: {len(truth)} cells, {len(segments)} segments ({source}), {n_shuffles} shuffles', '',
             'Detection (fraction significant) per gain x kappa:']
    det = uni.groupby(['kappa', 'gain'])[['sig_binned', 'sig_vm']].mean().round(2)
    lines += [det.to_string(), '', 'Median |phase error| for tuned cells per gain x kappa:',
              uni[uni.gain > 0].groupby(['kappa', 'gain'])[['err_binned', 'err_vm']].median().round(3).to_string(), '',
              'Bimodal cells (2 fields 0.5 apart), fraction significant: '
              f"binned {table[table.kind == 'bimodal'].sig_binned.mean():.2f}, "
              f"vM {table[table.kind == 'bimodal'].sig_vm.mean():.2f}", '',
              'FP / FN rates:', rates.round(3).to_string(index=False), '']
    # how good is the proxy? among probable FPs, fraction truly untuned; among probable FNs, fraction truly tuned
    for m in METHODS:
        fp, fn = table[f'probable_FP_{m}'], table[f'probable_FN_{m}']
        lines.append(f'proxy check {m}: {fp.sum()} probable FP, {(~table.tuned[fp]).mean() if fp.any() else np.nan:.2f} '
                     f'truly untuned; {fn.sum()} probable FN, {table.tuned[fn].mean() if fn.any() else np.nan:.2f} truly tuned')
    lines.append(proxy.to_string(index=False))
    (out_dir / 'summary.txt').write_text('\n'.join(lines))
    print('\n'.join(lines))
    print(f'Saved to {out_dir}')


def run_real(mouse, session, out_dir, n_shuffles, cache_dir=None, n_offsets=20, null_shuffles=200, seed=0):
    out_dir = Path(out_dir) / f'sub-{mouse}_{session}'
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = h.load_session(mouse, session, behav_root=h.BEHAV_ROOT, cache_dir=cache_dir)
    segments = loaded['segments']['reward']
    np.save(out_dir / 'reward_intervals_s.npy', np.diff(segments, axis=1).ravel() / loaded['frame_rate'])
    title = f'sub-{mouse} {session}'
    print(f'{title}: {loaded["dF"].shape[0]} cells, {len(segments)} reward segments')

    gb, gv = run_both(loaded, n_shuffles, seed)
    table = cell_table(gb, gv)
    table.insert(0, 'neuron', loaded['neurons'])
    sh = split_half_calls(loaded, segments, n_shuffles, seed)
    for m in METHODS:
        table[f'probable_FP_{m}'] = sh[f'FP {m}']
        table[f'probable_FN_{m}'] = sh[f'FN {m}']
    table.to_csv(out_dir / 'gp_method_comparison_cells.csv', index=False)

    plot_overview(table, gb, gv, f'{title}: binned (30 bins) vs von Mises fit, {n_shuffles} shuffles, alpha {ALPHA}',
                  out_dir / '1_overview.png')
    plot_cv_heatmaps(gb, gv, title, out_dir / '2_cv_sorted_heatmaps.png')
    plot_phase_distribution(table, title, out_dir / '6_phase_distribution.png')
    cells, labels = example_cells(table)
    plot_example_cells(loaded, gb, gv, cells, labels, f'{title}: examples (row index into accepted cells)',
                       out_dir / '3_example_cells.png')

    print('misaligned-clock null ...')
    null = misaligned_null(loaded, segments, n_offsets, null_shuffles, seed)
    np.save(out_dir / 'misaligned_null_fraction_significant.npy', null)
    proxy = proxy_rates(sh)
    rows = [dict(source='real misaligned clock', method=m, FP_rate=null[:, j].mean(), FN_rate=np.nan)
            for j, m in enumerate(METHODS)]
    rows += [dict(source='real split-half proxy', method=r.method, FP_rate=r.FP_rate, FN_rate=r.FN_rate)
             for _, r in proxy.iterrows()]
    rates = pd.DataFrame(rows)
    plot_fp_fn_bars(rates, out_dir / '4_fp_fn_rates.png', f'{title}: estimated FP / FN rates')
    plot_split_examples(sh, split_examples(sh), f'{title}: probable FP / FN examples (split-half)',
                        out_dir / '5_fp_fn_examples.png')

    sig = table.sig_binned | table.sig_vm
    ct = pd.crosstab(table.sig_binned, table.sig_vm, rownames=['binned'], colnames=['vM fit'])
    lines = [f'{title}: {len(table)} cells, {len(segments)} segments, {n_shuffles} shuffles, alpha {ALPHA}', '',
             ct.to_string(), '',
             f'circular corr of phases (significant by either): {h.circular_corr(table.phase_binned[sig], table.phase_vm[sig]):.3f}',
             f'median |phase_binned - phase_vm| (significant by either): '
             f'{np.median(h.circular_distance(table.phase_binned[sig], table.phase_vm[sig])):.3f}',
             'median split-half |even - odd| (significant by either): '
             f'binned {np.median(h.circular_distance(table.phase_even_binned[sig], table.phase_odd_binned[sig])):.3f}, '
             f'vM {np.median(h.circular_distance(table.phase_even_vm[sig], table.phase_odd_vm[sig])):.3f}', '',
             f'misaligned-clock null ({n_offsets} offsets x {null_shuffles} shuffles), fraction significant: '
             f'binned {null[:, 0].mean():.3f} +/- {null[:, 0].std():.3f}, vM {null[:, 1].mean():.3f} +/- {null[:, 1].std():.3f}',
             '', 'split-half proxy (called on even segments, validated on odd):', proxy.to_string(index=False), '']
    for k, v in split_examples(sh).items():
        lines.append(f'{k}: rows {[int(i) for i in v]} (neurons {[int(loaded["neurons"][i]) for i in v]})')
    (out_dir / 'summary.txt').write_text('\n'.join(lines))
    print('\n'.join(lines))
    print(f'Saved to {out_dir}')


def _cli():
    parser = argparse.ArgumentParser(description='Compare binned and von Mises-fit goal-progress tuning.')
    sub = parser.add_subparsers(dest='mode', required=True)
    p_syn = sub.add_parser('synthetic')
    p_syn.add_argument('--n-shuffles', type=int, default=500)
    p_real = sub.add_parser('real')
    p_real.add_argument('mouse')
    p_real.add_argument('session')
    p_real.add_argument('--n-shuffles', type=int, default=1000)
    p_real.add_argument('--cache-dir', default=None, help='LOCAL folder to pickle the loaded session')
    p_real.add_argument('--n-offsets', type=int, default=20, help='misaligned-clock null: number of shifted clocks')
    p_real.add_argument('--null-shuffles', type=int, default=200)
    for p in (p_syn, p_real):
        p.add_argument('--out-dir', default=OUT_ROOT)
        p.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.mode == 'synthetic':
        run_synthetic(args.out_dir, args.n_shuffles, args.seed)
    else:
        run_real(args.mouse, args.session, args.out_dir, args.n_shuffles, args.cache_dir, args.n_offsets,
                 args.null_shuffles, args.seed)


if __name__ == '__main__':
    _cli()

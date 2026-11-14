"""Follower cells: non-targeted cells that respond to another cell's photostimulation.

Population: holo suite2p ROIs with iscell, plus every stimulated cell (its holo ROI, or its projected
source mask when it has no holo ROI). Run from the repo root after session_overview_holography.py
(which writes target_trial_responses.csv, the per-trial success of every target):

    python -m musicboxes.holography.followers 001 holo-001 seq3-007

Trial sets. A target's photostimulation can only recruit followers on trials in which the target
fired, so every analysis uses each target's SUCCESSFUL trials (holo_functions.target_responses: trial
amplitude > 95th percentile of the target's null). Controls: the same targets' FAILED trials (same
spiral, same light, but no target spike) and the blank condition (no spiral). Only targets with at
least min_trials successful (or failed, for that control) trials are tested.

Amplitude per trial and unit = mean 0.5..2 s minus mean -2..0 s (as in target_responses). A unit's null
= its amplitudes on the BLANK trials only (no spiral, no light: the only clean control; 10 trials),
leaving out blank trials starting within null_exclude_s after one of the tested condition's trials.
(null='far_and_blank' restores the earlier, larger null: blank trials plus trials whose target is
>= null_dist_px from the unit, ~900 trials.) Because the null mean now comes from only 10 trials, its
uncertainty is part of the effect size: z_se is a Welch t,
    (mean own - mean blank) / sqrt(var_own / n_own + var_blank / n_blank),
approximately t-distributed under the null (df = Welch-Satterthwaite).

1. Per-pair test (fol_1_overview, fol_2_examples). For every target x unit pair: amplitudes on the
   target's successful trials vs the unit's null, two-sided Mann-Whitney, BH-FDR over all pairs of the
   successful set (self pairs excluded). q < 0.05 -> 'follower_exc' / 'follower_inh' if the unit is >=
   direct_px from the target, else 'near_target' (possibly stimulated by the spiral directly). Failed
   trials form their own family and are also counted at the successful set's p cut-off. With the
   blank-only null (10 trials) the smallest attainable two-sided p is ~1e-5, so this test has little
   power across ~10^4 pairs; the pooled and phase analyses are the ones to read.
   Latency: peak (0..3 s) of the 5-frame smoothed trial-averaged PSTH.
2. Pooled test (fol_3_pooled). Followers can be too weak to detect pair by pair. Every pair's response
   z_se (mean amplitude minus unit null mean, in standard errors: N(0, 1) under the null whatever the
   number of trials, which differs between successful and failed sets) averaged per distance band
   (DIST_BANDS_UM, converted with the ScanImage pixel size, holo_io.um_per_pixel): successful vs failed trials
   of the same pairs (paired Wilcoxon per band), and successful net z_se vs 0. Net z_se (excitation -
   inhibition) and |z_se| (any modulation; null expectation = mean E|t_df| over the pairs).
3. Within-trial correlation with trial shuffling (fol_4_trial_correlation). Targets with >= min_corr_trials
   successful trials (and the same analysis on failed trials and on the blank trials, as controls: the
   shift predictor removes stimulus-locked activity but not trial-by-trial shared fluctuations such as
   brain state or running, which are present without stimulation too): in each successful trial, lagged correlation (+-max_lag_s; positive lag = unit
   lags the target) between the target's and the unit's dF/F in corr_window (0..4 s after onset), each
   z-scored within the trial; averaged over trials. The shift predictor (same, but target trial i paired
   with unit trial sigma(i), sigma a random derangement) keeps everything stimulus-locked and removes
   trial-specific coupling. Statistic: peak |real - mean shuffled| over lags vs the same statistic for
   n_perm derangements (z, normal p, BH-FDR over pairs; also the permutation p).
4. Goal-progress phase (fol_5_phase). Only GP-tuned cells of the source session have a preferred
   phase; in sub-001 these are exactly the 103 targets, so the responders are the other targets
   (matched holo ROI or projected mask). Phase bins: n_phase_bins over 0..1. For each tested target and
   each tuned unit >= direct_px away: delta dF/F (mean amplitude on the target's successful trials minus
   the unit's null mean) by circular bin difference (unit bin - target bin; -3 = +3 = opposite phase),
   averaged per target first, then mean +- sem over targets; failed trials as control; Wilcoxon of the
   target means vs 0 per bin (BH over bins) and Kruskal-Wallis across bins. Also target bin x unit bin.

5. Phase asymmetry (fol_6_phase_asymmetry). Targeted test of the hypothesis (raised after seeing
   fol_5, so exploratory until replicated) that stimulating a cell enhances cells whose preferred phase
   LAGS its own and suppresses cells that LEAD it, within a narrow phase window. Continuous phase
   difference dphi = unit phase - target phase, wrapped to (-0.5, 0.5]. Per target: A = mean delta dF/F
   of units with 0 < dphi <= w minus that of units with -w <= dphi < 0; statistic = mean A over
   targets, for successful and failed trials and their paired difference. Null: the units' phases
   permuted (same permutation for both trial sets), two-sided. Several widths w (1/6 = one 6-bin bin is
   the primary one). Also a kernel-smoothed delta dF/F vs dphi curve with a permutation band, and the
   distance to the target of lagging vs leading units (a spatial confound check).

Outputs (output folder): follower_pairs.csv (one row per trial set x condition x unit),
follower_trial_corr.csv, follower_phase.csv, follower_phase_asymmetry.csv, follower_units.csv; plots
fol_*.png in the analysis folder.
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
EXC_COL, INH_COL, NEAR_COL = COL['matched'], COL['projected'], COL['extra']
SET_COL = {'successful': COL['matched'], 'failed': COL['projected'], 'blank': COL['extra']}
DIST_BANDS = np.array([25, 50, 100, 150, 200, 300, 800])                 # px, per-pair plots
DIST_BANDS_UM = np.array([21, 40, 80, 120, 160, 240, 320, 600])          # um, pooled test (21 um = 25 px)


def _despine(ax):
    ax.spines[['top', 'right']].set_visible(False)


def _save(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------- population and trials

def population(session, source_tag=None):
    """Units table (unit, kind 'holo'/'projected', holo_roi, source_roi if stimulated, src = source-session
    ROI if matched/projected else -1, x, y, iscell) and their traces (n_units, n_frames) float32."""
    stim = holo.stimulated_cells(session)
    stim_holo = dict(zip(stim.holo_roi[stim.status == 'matched'], stim.source_roi[stim.status == 'matched']))
    match = session['match']
    holo_to_src = dict(zip(match.holo_roi[match.holo_roi >= 0], match.source_roi[match.holo_roi >= 0]))
    xy = hio.roi_centroids(session['stat'])
    rows, traces = [], []
    dff = session['dff']
    for h in range(len(session['stat'])):
        if session['iscell'][h] or h in stim_holo:
            rows.append({'unit': f'h{h}', 'kind': 'holo', 'holo_roi': h, 'source_roi': stim_holo.get(h, -1),
                         'src': holo_to_src.get(h, -1), 'x': xy[h, 0], 'y': xy[h, 1],
                         'iscell': bool(session['iscell'][h])})
            traces.append(np.asarray(dff[h], np.float32))
    for c in stim[stim.status == 'projected'].itertuples():
        tr = holo.trace_for(session, c.source_roi)
        if tr is None:
            continue
        x, y = holo.cell_centroid(session, c.source_roi)
        rows.append({'unit': f'p{c.source_roi}', 'kind': 'projected', 'holo_roi': -1, 'source_roi': c.source_roi,
                     'src': c.source_roi, 'x': x, 'y': y, 'iscell': True})
        traces.append(np.asarray(tr, np.float32))
    return pd.DataFrame(rows), np.vstack(traces)


def trial_success(session):
    """{trial id: success} for every target trial, from target_trial_responses.csv."""
    tr = pd.read_csv(session['paths']['output'] / 'target_trial_responses.csv')
    return dict(zip(tr.trial, tr.success.astype(bool)))


def amplitude_matrix(traces, onset_frames, fs, baseline=BASELINE, response=RESPONSE):
    """(n_units, n_trials) response amplitudes, chunked over units to bound memory."""
    out = np.empty((len(traces), len(onset_frames)), np.float32)
    for i in range(0, len(traces), 32):
        t, m = holo.psth(traces[i:i + 32], onset_frames, fs, (baseline[0], response[1]))
        out[i:i + 32] = holo.response_amplitude(t, m, baseline, response)
    return out


# ---------------------------------------------------------------- 1. per-pair test

def follower_stats(session, units, traces, null='blank', null_dist_px=50, null_exclude_s=8, direct_px=25,
                   alpha=0.05, min_trials=3, min_null=8, latency_window=(0, 3), smooth_frames=5):
    """One row per trial set ('successful', 'failed'; plus 'blank' when null='far_and_blank') x
    condition x unit. null: 'blank' (default) or 'far_and_blank'. Returns (pairs, A)."""
    t_all = session['trials']
    fs = session['fs']
    succ = trial_success(session)
    A = amplitude_matrix(traces, t_all.onset_frame.values, fs)
    tx, ty = t_all.target_x_holo.values, t_all.target_y_holo.values
    if null == 'blank':
        far = np.repeat(t_all.is_blank.values[None], len(units), 0)
    else:
        far = (np.hypot(units.x.values[:, None] - tx[None], units.y.values[:, None] - ty[None]) >= null_dist_px) \
            | t_all.is_blank.values[None]
    onset_t = t_all.onset_time.values
    trial_ok = t_all.trial.map(succ)
    unit_of_source = dict(zip(units.source_roi, units.index))
    conds = t_all.drop_duplicates('condition').sort_values(['is_blank', 'source_roi'])
    rows = []
    for c in conds.itertuples():
        if c.is_blank and null == 'blank':
            continue                                  # the blank trials are the null itself
        in_cond = (t_all.condition == c.condition).values
        dt = onset_t[:, None] - onset_t[in_cond][None]
        null_m = far & ~in_cond[None] & ~((dt > 0) & (dt <= null_exclude_s)).any(1)[None]
        target_unit = -1 if c.is_blank else unit_of_source.get(c.source_roi, -1)
        d_target = (np.full(len(units), np.nan) if c.is_blank else
                    np.hypot(units.x.values - c.target_x_holo, units.y.values - c.target_y_holo))
        sets = {'blank': in_cond} if c.is_blank else {
            'successful': in_cond & trial_ok.eq(True).values, 'failed': in_cond & trial_ok.eq(False).values}
        for name, own in sets.items():
            if own.sum() < min_trials:
                continue
            for i in range(len(units)):
                a_own, a_null = A[i, own], A[i, null_m[i]]
                a_own, a_null = a_own[~np.isnan(a_own)], a_null[~np.isnan(a_null)]
                if len(a_own) < min_trials or len(a_null) < min_null:
                    continue
                vo, vn = a_own.var(ddof=1) / len(a_own), a_null.var(ddof=1) / len(a_null)
                rows.append({'trial_set': name, 'condition': c.condition, 'target_source_roi': c.source_roi,
                             'unit': units.unit[i], 'unit_idx': i, 'self': i == target_unit, 'dist_px': d_target[i],
                             'n_trials': len(a_own), 'n_null': len(a_null), 'amp_mean': a_own.mean(),
                             'null_mean': a_null.mean(), 'delta': a_own.mean() - a_null.mean(),
                             'z': (a_own.mean() - a_null.mean()) / a_null.std(ddof=1),
                             'z_se': (a_own.mean() - a_null.mean()) / np.sqrt(vo + vn),
                             'df': (vo + vn) ** 2 / (vo ** 2 / (len(a_own) - 1) + vn ** 2 / (len(a_null) - 1)),
                             'p': stats.mannwhitneyu(a_own, a_null, alternative='two-sided').pvalue})
    pairs = pd.DataFrame(rows)
    pairs['q'] = np.nan
    for name in pairs.trial_set.unique():
        fam = (pairs.trial_set == name) & ~pairs.self
        pairs.loc[fam, 'q'] = holo.bh_fdr(pairs.p[fam])
    main = (pairs.trial_set == 'successful') & ~pairs.self
    p_cut = pairs.p[main & (pairs.q < alpha)].max() if (main & (pairs.q < alpha)).any() else 0.0
    pairs['sig_at_main_cutoff'] = pairs.p <= p_cut
    cls = np.where(pairs.self, 'self', 'none').astype(object)
    hit = (pairs.q < alpha) & ~pairs.self
    cls[hit & (pairs.dist_px < direct_px)] = 'near_target'
    cls[hit & ~(pairs.dist_px < direct_px) & (pairs.z > 0)] = 'follower_exc'
    cls[hit & ~(pairs.dist_px < direct_px) & (pairs.z < 0)] = 'follower_inh'
    pairs['class'] = cls
    pairs.attrs['p_cut'] = p_cut

    pairs['peak_s'] = np.nan
    need = pairs[(pairs['class'] != 'none') & (pairs.trial_set == 'successful')]
    for cond, g in need.groupby('condition'):
        on = t_all.onset_frame[(t_all.condition == cond) & trial_ok.eq(True)].values
        tt, m = holo.psth(traces[g.unit_idx.values], on, fs, WINDOW)
        mu = np.nanmean(m, 1)
        mu = ndimage.uniform_filter1d(mu - np.nanmean(mu[:, tt < 0], 1, keepdims=True), smooth_frames, axis=1)
        w = (tt >= latency_window[0]) & (tt <= latency_window[1])
        pairs.loc[g.index, 'peak_s'] = tt[w][np.argmax(mu[:, w] * np.sign(g.z.values)[:, None], 1)]
    return pairs, A


# ---------------------------------------------------------------- 2. pooled test

def pooled_by_distance(pairs, bands=DIST_BANDS):
    """Per distance band: mean net z and |z| per trial set, and paired successful-vs-failed Wilcoxon.
    z here is z_se = (mean own - null mean) / (null SD / sqrt(n own trials)), N(0, 1) under the null for
    any n (E|z_se| ~ 0.80). The plain z (in null SDs) is not used: its spread shrinks with n, and
    successful and failed trial counts differ, which biases |z| towards the smaller set."""
    p = pairs[~pairs.self].copy()
    p['band'] = pd.cut(p.dist_px, bands)
    sf = p[p.trial_set.isin(['successful', 'failed'])].pivot_table(
        index=['condition', 'unit'], columns='trial_set', values='z_se').dropna()
    sf = sf.join(p[p.trial_set == 'successful'].set_index(['condition', 'unit']).band)
    blank = p[p.trial_set == 'blank']
    rows = []

    def e_abs_t(df):
        from scipy.special import gammaln
        df = np.asarray(df, float)
        return np.exp(np.log(2) + 0.5 * np.log(df) + gammaln((df + 1) / 2) - 0.5 * np.log(np.pi)
                      - np.log(df - 1) - gammaln(df / 2))

    for band, g in sf.groupby('band', observed=True):
        rows.append({'band': str(band), 'lo_px': band.left, 'hi_px': band.right, 'centre': band.mid, 'n_pairs': len(g),
                     'z_successful': g.successful.mean(), 'sem_successful': g.successful.sem(),
                     'z_failed': g.failed.mean(), 'sem_failed': g.failed.sem(),
                     'absz_successful': g.successful.abs().mean(), 'absz_failed': g.failed.abs().mean(),
                     'p_net': stats.wilcoxon(g.successful, g.failed).pvalue,
                     'p_abs': stats.wilcoxon(g.successful.abs(), g.failed.abs()).pvalue,
                     'p_succ_vs0': stats.wilcoxon(g.successful).pvalue,
                     'p_vs_blank': stats.mannwhitneyu(g.successful, blank.z_se).pvalue if len(blank) else np.nan})
    out = pd.DataFrame(rows)
    out['q_net'] = holo.bh_fdr(out.p_net)
    out['q_abs'] = holo.bh_fdr(out.p_abs)
    out['q_succ_vs0'] = holo.bh_fdr(out.p_succ_vs0)
    if len(blank):
        out.attrs['blank_z'], out.attrs['blank_absz'] = blank.z_se.mean(), blank.z_se.abs().mean()
    sf_df = p[p.trial_set.isin(['successful', 'failed'])].df
    out.attrs['null_absz'] = float(np.nanmean(e_abs_t(sf_df[sf_df > 1]))) if 'df' in p else np.sqrt(2 / np.pi)
    return out


# ---------------------------------------------------------------- 3. within-trial correlation, trial-shuffled

def _derangements(n, n_perm, rng):
    out = []
    while len(out) < n_perm:
        s = rng.permutation(n)
        if not (s == np.arange(n)).any():
            out.append(s)
    return np.array(out)


def trial_correlation(session, units, traces, trial_set='successful', corr_window=(0, 4), max_lag_s=1.0,
                      min_corr_trials=5, n_perm=500, seed=0):
    """trial_set 'successful' / 'failed': the target's own trials of that kind; 'blank': the blank trials
    (target and unit traces in the same windows, no stimulation), the control for trial-specific shared
    activity that the shift predictor does not remove."""
    from scipy import fft
    t_all = session['trials']
    fs = session['fs']
    succ = trial_success(session)
    rng = np.random.default_rng(seed)
    unit_of_source = dict(zip(units.source_roi, units.index))
    L = int(round((corr_window[1] - corr_window[0]) * fs))
    max_lag = int(round(max_lag_s * fs))
    nfft = fft.next_fast_len(2 * L)
    lag_idx = np.r_[nfft - max_lag:nfft, 0:max_lag + 1]
    lags = np.arange(-max_lag, max_lag + 1) / fs
    rows = []
    for src, tu in unit_of_source.items():
        if src < 0:
            continue
        if trial_set == 'blank':
            sel = t_all.is_blank
        else:
            sel = (t_all.source_roi == src) & t_all.trial.map(succ).eq(trial_set == 'successful')
        on = t_all.onset_frame[sel].values + int(round(corr_window[0] * fs))
        on = on[(on >= 0) & (on + L <= traces.shape[1])]
        n = len(on)
        if n < min_corr_trials:
            continue
        seg = traces[:, on[:, None] + np.arange(L)[None]]                      # units x trials x L
        seg = (seg - seg.mean(2, keepdims=True)) / (seg.std(2, keepdims=True) + 1e-9)
        F = fft.rfft(seg, nfft, axis=2)
        # C[u, i, j, lag] = sum_t x_target,i(t) y_u,j(t + lag) / L
        C = fft.irfft(np.conj(F[tu])[None, :, None, :] * F[:, None, :, :], nfft, axis=3)[..., lag_idx] / L
        perms = _derangements(n, n_perm, rng)
        ii = np.arange(n)
        real = C[:, ii, ii].mean(1)                                              # units x lags
        shuf = np.stack([C[:, ii, s].mean(1) for s in perms])                    # perms x units x lags
        pred = shuf.mean(0)
        stat = np.abs(real - pred).max(1)
        null = np.abs(shuf - pred[None]).max(2)                                  # perms x units
        k = np.argmax(np.abs(real - pred), 1)
        for u in range(len(units)):
            if u == tu:
                continue
            rows.append({'trial_set': trial_set, 'target_source_roi': src, 'unit': units.unit[u], 'n_trials': n,
                         'r_real_peak': real[u, k[u]], 'r_shift_peak': pred[u, k[u]],
                         'r_diff_peak': real[u, k[u]] - pred[u, k[u]], 'lag_s': lags[k[u]],
                         'r_real_0': real[u, max_lag], 'r_shift_0': pred[u, max_lag],
                         'z_corr': (stat[u] - null[:, u].mean()) / null[:, u].std(),
                         'p_perm': (1 + (null[:, u] >= stat[u]).sum()) / (1 + n_perm)})
    tc = pd.DataFrame(rows)
    if tc.empty:
        return tc
    tc['p_corr'] = stats.norm.sf(tc.z_corr)
    tc['q_corr'] = holo.bh_fdr(tc.p_corr)
    tc['q_perm'] = holo.bh_fdr(tc.p_perm)
    return tc


# ---------------------------------------------------------------- 4. goal-progress phase

def phase_table(session, units, source_tag, project_root=hio.PROJECT_ROOT, n_phase_bins=6):
    gp = pd.read_csv(hio.gp_table_path(session['mouse'], source_tag, project_root))
    ph = gp.set_index('cell_id').preferred_phase
    u = units[['unit', 'src']].copy()
    u['phase'] = u.src.map(ph)
    u['phase_bin'] = np.floor(u.phase * n_phase_bins).clip(0, n_phase_bins - 1)
    return u.set_index('unit'), ph


def phase_delta(pairs, units_phase, ph, n_phase_bins=6, direct_px=25):
    """Pairs of tuned target x tuned unit (>= direct_px): delta dF/F by circular phase-bin difference."""
    p = pairs[pairs.trial_set.isin(['successful', 'failed']) & ~pairs.self & (pairs.dist_px >= direct_px)].copy()
    p['unit_bin'] = p.unit.map(units_phase.phase_bin)
    p['target_bin'] = np.floor(p.target_source_roi.map(ph) * n_phase_bins).clip(0, n_phase_bins - 1)
    p = p.dropna(subset=['unit_bin', 'target_bin'])
    half = n_phase_bins // 2
    p['dbin'] = ((p.unit_bin - p.target_bin + half) % n_phase_bins - half).astype(int)
    per_target = p.groupby(['trial_set', 'target_source_roi', 'dbin'])[['delta', 'z']].mean().reset_index()
    rows = []
    for (name, d), g in per_target.groupby(['trial_set', 'dbin']):
        rows.append({'trial_set': name, 'dbin': d, 'n_targets': len(g), 'delta_mean': g.delta.mean(),
                     'delta_sem': g.delta.sem(), 'z_mean': g.z.mean(), 'z_sem': g.z.sem(),
                     'p_vs0': stats.wilcoxon(g.delta).pvalue if len(g) > 5 else np.nan})
    summ = pd.DataFrame(rows)
    summ['q_vs0'] = np.nan
    for name in summ.trial_set.unique():
        m = summ.trial_set == name
        summ.loc[m, 'q_vs0'] = holo.bh_fdr(summ.p_vs0[m])
    kw = {name: stats.kruskal(*[g.delta.values for _, g in per_target[per_target.trial_set == name].groupby('dbin')]).pvalue
          for name in per_target.trial_set.unique()}
    summ.attrs['kruskal'] = kw
    return p, per_target, summ


# ---------------------------------------------------------------- 5. phase asymmetry

def _wrap(x):
    return (x + 0.5) % 1 - 0.5


def phase_asymmetry(pairs, units_phase, ph, widths=(0.1, 1 / 6, 0.25, 1 / 3), n_perm=5000, direct_px=25,
                    kernel_bw=0.05, n_perm_curve=500, seed=0):
    """Lagging-minus-leading asymmetry of delta dF/F (see module docstring, item 5).

    Returns (table, curves, extra): table has one row per width x quantity ('successful', 'failed',
    'diff') with the observed statistic, the permutation p and the null 2.5/97.5 percentiles; curves
    holds the kernel curves; extra the per-target A values (primary width), null samples and the
    lag/lead distance check.
    """
    rng = np.random.default_rng(seed)
    p = pairs[pairs.trial_set.isin(['successful', 'failed']) & ~pairs.self & (pairs.dist_px >= direct_px)].copy()
    p['unit_phase'] = p.unit.map(units_phase.phase)
    p['target_phase'] = p.target_source_roi.map(ph)
    p = p.dropna(subset=['unit_phase', 'target_phase', 'delta']).reset_index(drop=True)
    unit_names = np.array(sorted(p.unit.unique()))
    uix = np.searchsorted(unit_names, p.unit.values)
    uph = units_phase.phase.reindex(unit_names).values
    targets = np.array(sorted(p.target_source_roi.unique()))
    tix = np.searchsorted(targets, p.target_source_roi.values)
    is_s = (p.trial_set == 'successful').values
    delta, tph = p.delta.values, p.target_phase.values
    nt = len(targets)

    def per_target(unit_phase, w):
        dphi = _wrap(unit_phase[uix] - tph)
        out = {}
        for name, m in (('successful', is_s), ('failed', ~is_s)):
            res = []
            for side in ((dphi > 0) & (dphi <= w), (dphi < 0) & (dphi >= -w)):
                k = m & side
                sm = np.bincount(tix[k], delta[k], nt)
                n = np.bincount(tix[k], minlength=nt)
                res.append(np.where(n > 0, sm / np.maximum(n, 1), np.nan))
            out[name] = res[0] - res[1]
        return out

    def summary(a):
        both = ~np.isnan(a['successful']) & ~np.isnan(a['failed'])
        return {'successful': np.nanmean(a['successful']), 'failed': np.nanmean(a['failed']),
                'diff': np.mean(a['successful'][both] - a['failed'][both])}

    perms = [rng.permutation(len(uph)) for _ in range(n_perm)]
    rows, extra = [], {}
    for w in widths:
        obs_t = per_target(uph, w)
        obs = summary(obs_t)
        null = {k: np.array([summary(per_target(uph[pi], w))[k] for pi in perms]) for k in obs}
        for k in obs:
            rows.append({'width': w, 'quantity': k, 'observed': obs[k],
                         'n_targets': int(np.sum(~np.isnan(obs_t['successful' if k == 'successful' else 'failed'])))
                         if k != 'diff' else int(np.sum(~np.isnan(obs_t['successful']) & ~np.isnan(obs_t['failed']))),
                         'p_perm': (1 + np.sum(np.abs(null[k]) >= abs(obs[k]))) / (1 + n_perm),
                         'null_lo': np.percentile(null[k], 2.5), 'null_hi': np.percentile(null[k], 97.5)})
        if np.isclose(w, 1 / 6):
            extra['per_target'] = pd.DataFrame({'target_source_roi': targets, 'A_successful': obs_t['successful'],
                                                'A_failed': obs_t['failed']})
            extra['null_diff'] = null['diff']
            extra['obs_diff'] = obs['diff']
            dphi = _wrap(uph[uix] - tph)
            lag, lead = is_s & (dphi > 0) & (dphi <= w), is_s & (dphi < 0) & (dphi >= -w)
            extra['dist'] = {'lag_median_px': np.median(p.dist_px[lag]), 'lead_median_px': np.median(p.dist_px[lead]),
                             'n_lag': int(lag.sum()), 'n_lead': int(lead.sum()),
                             'p_mw': stats.mannwhitneyu(p.dist_px[lag], p.dist_px[lead]).pvalue}
    table = pd.DataFrame(rows)

    grid = np.linspace(-0.5, 0.5, 101)

    def curve(unit_phase, m):
        dphi = _wrap(unit_phase[uix][m] - tph[m])
        k = np.exp(-_wrap(grid[:, None] - dphi[None]) ** 2 / (2 * kernel_bw ** 2))
        return (k @ delta[m]) / k.sum(1)

    curves = {'grid': grid, 'successful': curve(uph, is_s), 'failed': curve(uph, ~is_s)}
    nc = np.array([curve(uph[perms[i]], is_s) for i in range(min(n_perm_curve, n_perm))])
    curves['null_lo'], curves['null_hi'] = np.percentile(nc, [2.5, 97.5], axis=0)
    return table, curves, extra


# ---------------------------------------------------------------- plots

def plot_overview(pairs, resp, save_path=None):
    real = pairs[(pairs.trial_set == 'successful') & ~pairs.self]
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    ax = axes[0]
    cnt = real.groupby('target_source_roi')['class'].value_counts().unstack(fill_value=0)
    for k in ('follower_exc', 'follower_inh', 'near_target'):
        if k not in cnt:
            cnt[k] = 0
    r = resp.set_index('source_roi').loc[cnt.index]
    jit = np.random.default_rng(0).uniform(-0.2, 0.2, len(cnt))
    ax.scatter(r.n_success + jit, cnt.follower_exc, s=16, color=EXC_COL, label='excited followers')
    ax.scatter(r.n_success - jit, -cnt.follower_inh, s=16, color=INH_COL, label='inhibited followers (negative)')
    for name, ls in (('failed', ':'), ('blank', '--')):
        if not (pairs.trial_set == name).any():
            continue
        n = int((pairs.trial_set.eq(name) & ~pairs.self & pairs.sig_at_main_cutoff & ~(pairs.dist_px < 25)).sum())
        ax.axhline(n, color=SET_COL[name], ls=ls, lw=1, label=f'{name} trials, all targets ({n})')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('target: successful trials')
    ax.set_ylabel('followers per target (>= 25 px, FDR q < 0.05)')
    ax.legend(frameon=False, fontsize=7)
    _despine(ax)

    ax = axes[1]
    ok = real[np.isfinite(real.dist_px)]
    ax.scatter(ok.dist_px, ok.z, s=2, color='0.75', rasterized=True)
    for k, col in (('follower_exc', EXC_COL), ('follower_inh', INH_COL), ('near_target', NEAR_COL)):
        d = ok[ok['class'] == k]
        ax.scatter(d.dist_px, d.z, s=10, color=col, label=f'{k} ({len(d)})')
    ax.set_xlabel('unit distance to target (px)')
    ax.set_ylabel('response z vs blank trials (successful trials)')
    ax.legend(frameon=False, fontsize=7)
    _despine(ax)

    ax = axes[2]
    bins = np.r_[0, DIST_BANDS]
    for name in ('successful', 'failed', 'blank'):
        g = pairs[(pairs.trial_set == name) & ~pairs.self]
        if g.empty:
            continue
        if name == 'blank':
            ax.axhline(100 * g.sig_at_main_cutoff.mean(), color=SET_COL[name], ls='--', lw=1, label=name)
            continue
        which = np.digitize(g.dist_px, bins) - 1
        frac = [100 * g.sig_at_main_cutoff[which == i].mean() if (which == i).any() else np.nan
                for i in range(len(bins) - 1)]
        ax.plot(bins[:-1] + np.diff(bins) / 2, frac, 'o-', color=SET_COL[name], label=f'{name} trials')
    ax.set_xlabel('unit distance to target (px)')
    ax.set_ylabel('% pairs below the FDR p cut-off')
    ax.set_yscale('symlog', linthresh=1)
    ax.legend(frameon=False, fontsize=7)
    _despine(ax)

    ax = axes[3]
    bins_t = np.arange(0, 3.05, 0.1)
    own = pairs[pairs.self & (pairs.trial_set == 'successful')]
    for k, col in (('follower_exc', EXC_COL), ('follower_inh', INH_COL), ('near_target', NEAR_COL)):
        d = real[real['class'] == k].peak_s.dropna()
        if len(d):
            ax.hist(d, bins_t, color=col, alpha=0.6, label=f'{k} (median {d.median():.2f} s)')
    tpk = own.peak_s.dropna()
    if len(tpk):
        ax.hist(tpk, bins_t, histtype='step', color='k', lw=1.2, label=f'targets (median {tpk.median():.2f} s)')
    ax.set_xlabel('peak time after train onset (s)')
    ax.set_ylabel('pairs')
    ax.legend(frameon=False, fontsize=7)
    _despine(ax)
    n_t = real.target_source_roi.nunique()
    fig.suptitle(f"Per-pair test on successful trials ({n_t} targets with >= 3): "
                 f"{int((real['class'] == 'follower_exc').sum())} excited, {int((real['class'] == 'follower_inh').sum())} "
                 f"inhibited followers >= 25 px, {int((real['class'] == 'near_target').sum())} near-target pairs "
                 f"(BH-FDR q < 0.05, p cut-off {pairs.attrs.get('p_cut', np.nan):.1e})")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_examples(session, units, traces, pairs, n_targets=4, n_followers=5, save_path=None):
    real = pairs[(pairs.trial_set == 'successful') & ~pairs.self]
    fol = real[real['class'].isin(['follower_exc', 'follower_inh', 'near_target'])]
    if fol.empty:
        if save_path is not None and Path(save_path).exists():
            Path(save_path).unlink()                  # no followers: drop an example plot from an earlier run
        return None
    succ = trial_success(session)
    top = fol.groupby('target_source_roi').size().sort_values(ascending=False).index[:n_targets]
    fs = session['fs']
    t_all = session['trials']
    fig, axes = plt.subplots(len(top), 2, figsize=(13, 3.6 * len(top)), gridspec_kw={'width_ratios': [1, 1.6]},
                             squeeze=False)
    img = session['mean_img']
    for row, tsrc in enumerate(top):
        allf = fol[fol.target_source_roi == tsrc]
        g = allf.sort_values('p').head(n_followers)
        cond = g.condition.iloc[0]
        tg = t_all[t_all.condition == cond].iloc[0]
        ax = axes[row, 0]
        ax.imshow(img, cmap='gray', clim=holo.display_range(img))
        for k, col in (('follower_exc', EXC_COL), ('follower_inh', INH_COL), ('near_target', NEAR_COL)):
            u = units.loc[allf.unit_idx[allf['class'] == k]]
            ax.scatter(u.x, u.y, s=18, facecolor='none', edgecolor=col, lw=1.2)
        ax.plot(tg.target_x_holo, tg.target_y_holo, '*', color='w', ms=12)
        ax.set_title(f'target {tsrc}: {len(allf)} pairs (pink = excited, blue = inhibited, brown = near target)',
                     fontsize=9)
        ax.set_axis_off()
        ax = axes[row, 1]
        sel = (t_all.condition == cond) & t_all.trial.map(succ).eq(True)
        on = t_all.onset_frame[sel].values
        tu = units.index[units.source_roi == tsrc]
        idx = list(tu) + list(g.unit_idx)
        tt, m = holo.psth(traces[idx], on, fs, WINDOW)
        offs = 0
        for j, (i, mm) in enumerate(zip(idx, m)):
            mu = ndimage.uniform_filter1d(np.nanmean(mm, 0), 5)
            mu = mu - np.nanmean(mu[tt < 0])
            span = np.nanmax(np.abs(mu)) + 1e-9
            lab = f'target {tsrc}' if j == 0 else (f"{units.unit[i]} ({g.dist_px.iloc[j - 1]:.0f} px, "
                                                    f"peak {g.peak_s.iloc[j - 1]:.2f} s)")
            col = 'k' if j == 0 else (EXC_COL if g.z.iloc[j - 1] > 0 else INH_COL)
            ax.plot(tt, mu / span - offs, color=col, lw=1)
            ax.text(tt[-1] + 0.1, -offs, lab, fontsize=7, va='center')
            offs += 1.3
        ax.axvspan(0, float(tg.offset_time - tg.onset_time), color=COL['other'], alpha=0.5, lw=0)
        ax.set_yticks([])
        ax.set_xlim(tt[0], tt[-1] + 2.5)
        ax.set_xlabel('time from train onset (s)')
        ax.set_title(f'successful trials only (n={len(on)}), each trace scaled to its max |dF/F|', fontsize=9)
        _despine(ax)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_pooled(pooled, um_per_px=None, save_path=None):
    """Linear distance axis; in um when um_per_px is given (holo_io.um_per_pixel), else px. Points at
    band centres, horizontal bars = band extent."""
    scale, unit = (um_per_px, 'um') if um_per_px else (1.0, 'px')
    x = pooled.centre * scale
    xerr = (pooled.hi_px - pooled.lo_px) / 2 * scale
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (col, sem, lab, blank) in zip(axes, (('z', 'sem', 'mean net t vs blank (excitation > 0)', 'blank_z'),
                                                 ('absz', None, 'mean |t| vs blank (any modulation)', 'blank_absz'))):
        for name, off in (('successful', -0.01), ('failed', 0.01)):
            y = pooled[f'{col}_{name}']
            yerr = pooled[f'sem_{name}'] if sem else None
            ax.errorbar(x * (1 + off), y, yerr=yerr, xerr=xerr, marker='o', ms=4, capsize=0, elinewidth=0.8,
                        color=SET_COL[name], label=f'{name} trials')
        if blank in pooled.attrs:
            ax.axhline(pooled.attrs[blank], color=SET_COL['blank'], ls='--', lw=1, label='blank trials')
        if col == 'z':
            ax.axhline(0, color='k', lw=0.5)
        else:
            ax.axhline(pooled.attrs.get('null_absz', np.sqrt(2 / np.pi)), color='k', ls=':', lw=1,
                       label='null expectation (E|t|)')
        qcol = 'q_net' if col == 'z' else 'q_abs'
        for xi, yi, q in zip(x, pooled[f'{col}_successful'], pooled[qcol]):
            if q < 0.05:
                ax.text(xi, yi, '*', fontsize=14, ha='center', va='bottom', color='k')
        ax.set_xlim(0, (pooled.hi_px.max()) * scale)
        ax.set_xlabel(f'unit distance to target ({unit})')
        ax.set_ylabel(lab)
        ax.legend(frameon=False, fontsize=8)
        _despine(ax)
    ax.set_title('* = successful vs failed, paired Wilcoxon, BH-FDR q < 0.05', fontsize=9)
    bands = ', '.join(f'{lo * scale:.0f}-{hi * scale:.0f}: {n}' for lo, hi, n in
                      zip(pooled.lo_px, pooled.hi_px, pooled.n_pairs))
    fig.suptitle(f'Pooled test: response vs blank trials (Welch t) of all units by distance to the target; '
                 f'pairs per band ({unit}):\n'
                 + bands, fontsize=9)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_trial_correlation(tc, pairs, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    sets = [ts for ts in ('successful', 'failed', 'blank') if (tc.trial_set == ts).any()]
    ax = axes[0]
    bins = np.linspace(np.nanpercentile(tc.z_corr, 0.5), np.nanpercentile(tc.z_corr, 99.8), 50)
    for ts in sets:
        g = tc[tc.trial_set == ts]
        ax.hist(g.z_corr, bins, density=True, histtype='step', lw=1.5, color=SET_COL[ts],
                label=f'{ts} ({g.target_source_roi.nunique()} targets, median {g.z_corr.median():.2f})')
    ax.set_xlabel('z of peak |real - shift predictor|')
    ax.set_ylabel('density')
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)
    ax = axes[1]
    for ts in sets:
        g = tc[tc.trial_set == ts]
        h, e = np.histogram(g.p_perm, np.linspace(0, 1, 21))
        ax.step(e[:-1], h / len(g), where='post', color=SET_COL[ts], lw=1.5,
                label=f'{ts}: {100 * (g.p_perm < 0.01).mean():.1f}% p < 0.01')
    ax.axhline(1 / 20, color='k', ls='--', lw=1)
    ax.set_xlabel('permutation p')
    ax.set_ylabel('fraction of pairs')
    ax.set_title('uniform under the null (dashed)', fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)
    ax = axes[2]
    pv = tc.pivot_table(index=['target_source_roi', 'unit'], columns='trial_set', values='z_corr')
    if {'successful', 'blank'} <= set(pv.columns):
        d = pv[['successful', 'blank']].dropna()
        ax.scatter(d.blank, d.successful, s=3, color=SET_COL['successful'], alpha=0.5, rasterized=True)
        lim = np.nanpercentile(d.values, [0.5, 99.8])
        ax.plot(lim, lim, 'k--', lw=0.8)
        rho = stats.spearmanr(d.blank, d.successful).statistic
        ax.set_title(f'same pairs, Spearman rho={rho:.2f}', fontsize=9)
    ax.set_xlabel('z, blank trials (no stimulation)')
    ax.set_ylabel('z, successful trials')
    _despine(ax)
    ax = axes[3]
    m = tc[tc.trial_set == 'successful'].merge(
        pairs[pairs.trial_set == 'successful'][['target_source_roi', 'unit', 'dist_px']],
        on=['target_source_roi', 'unit'], how='left')
    ax.scatter(m.dist_px, m.z_corr, s=2, color='0.7', rasterized=True)
    bins_d = np.r_[0, DIST_BANDS]
    which = np.digitize(m.dist_px, bins_d) - 1
    ax.plot(bins_d[:-1] + np.diff(bins_d) / 2, [m.z_corr[which == i].median() for i in range(len(bins_d) - 1)],
            'o-', color=SET_COL['successful'], label='median per band')
    ax.set_xscale('log')
    ax.set_xlabel('unit distance to target (px)')
    ax.set_ylabel('z, successful trials')
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)
    fig.suptitle('Within-trial lagged correlation of target and unit (0..4 s after onset), corrected by the '
                 'trial-shuffled shift predictor; failed and blank trials as controls for shared trial-by-trial activity')
    fig.tight_layout()
    return _save(fig, save_path)


def plot_phase(p, per_target, summ, n_phase_bins=6, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    half = n_phase_bins // 2
    for ax, col in zip(axes[:2], ('delta', 'z')):
        for name, off in (('successful', -0.08), ('failed', 0.08)):
            s = summ[summ.trial_set == name].sort_values('dbin')
            ax.errorbar(s.dbin + off, s[f'{col}_mean'], yerr=s[f'{col}_sem'], marker='o', ms=5, capsize=2,
                        color=SET_COL[name], label=f"{name} trials (Kruskal p={summ.attrs['kruskal'].get(name, np.nan):.2g})")
            if col == 'delta':
                for x, y, q in zip(s.dbin, s.delta_mean, s.q_vs0):
                    if q < 0.05:
                        ax.text(x + off, y, '*', fontsize=14, ha='center', va='bottom')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(np.arange(-half, half), [f'{d:+d}' if d != -half else f'+-{half}' for d in range(-half, half)])
        ax.set_xlabel('unit phase bin - target phase bin (circular)')
        ax.set_ylabel('delta dF/F (mean over targets)' if col == 'delta' else 'response z (mean over targets)')
        ax.legend(frameon=False, fontsize=8)
        _despine(ax)
    axes[0].set_title('* = successful trials, Wilcoxon vs 0 over targets, BH q < 0.05', fontsize=9)
    ax = axes[2]
    s = p[p.trial_set == 'successful']
    mat = s.pivot_table(index='target_bin', columns='unit_bin', values='delta', aggfunc='mean').reindex(
        index=range(n_phase_bins), columns=range(n_phase_bins))
    lim = np.nanmax(np.abs(mat.values))
    im = ax.imshow(mat.values, cmap=holo.colormap(), vmin=-lim, vmax=lim)
    ax.set_xticks(range(n_phase_bins))
    ax.set_yticks(range(n_phase_bins))
    ax.set_xlabel('responding unit phase bin')
    ax.set_ylabel('target phase bin')
    fig.colorbar(im, ax=ax, label='delta dF/F (successful trials)')
    n_t = s.target_source_roi.nunique()
    fig.suptitle(f'Goal-progress phase: response of GP-tuned cells (>= 25 px) to {n_t} targets, by phase difference '
                 f'({n_phase_bins} bins; tuned cells = the targets in this session)')
    fig.tight_layout()
    return _save(fig, save_path)


def plot_phase_asymmetry(table, curves, extra, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    ax = axes[0]
    g = curves['grid']
    ax.fill_between(g, curves['null_lo'], curves['null_hi'], color=SET_COL['successful'], alpha=0.15, lw=0,
                    label='successful, phase-permuted 95%')
    for name in ('successful', 'failed'):
        ax.plot(g, curves[name], color=SET_COL[name], lw=1.8, label=f'{name} trials')
    ax.axvline(0, color='k', lw=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('phase difference, unit - target (> 0: unit lags target)')
    ax.set_ylabel('delta dF/F (kernel mean over pairs)')
    ax.legend(frameon=False, fontsize=7)
    _despine(ax)
    ax = axes[1]
    for k, col, off in (('successful', SET_COL['successful'], -0.004), ('failed', SET_COL['failed'], 0),
                        ('diff', 'k', 0.004)):
        d = table[table.quantity == k]
        ax.fill_between(d.width + off, d.null_lo, d.null_hi, color=col, alpha=0.12, lw=0)
        ax.plot(d.width + off, d.observed, 'o-', color=col, label=k if k != 'diff' else 'successful - failed')
        for x, y, pv in zip(d.width + off, d.observed, d.p_perm):
            ax.text(x, y, f' {pv:.2g}', fontsize=7, color=col, va='bottom')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('half-window w (phase)')
    ax.set_ylabel('lagging - leading delta dF/F (mean over targets)')
    ax.set_title('labels: permutation p; shading: null 95%', fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    _despine(ax)
    ax = axes[2]
    pt = extra['per_target']
    ax.scatter(pt.A_failed, pt.A_successful, s=16, color=SET_COL['successful'])
    lim = np.nanpercentile(np.abs(pt[['A_failed', 'A_successful']].values), 98) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('A, failed trials')
    ax.set_ylabel('A, successful trials')
    ax.set_title('per target, w = 1/6 (A = lagging - leading)', fontsize=9)
    _despine(ax)
    ax = axes[3]
    ax.hist(extra['null_diff'], bins=60, color='0.7')
    ax.axvline(extra['obs_diff'], color=SET_COL['successful'], lw=2)
    d = extra['dist']
    ax.set_xlabel('successful - failed asymmetry, w = 1/6')
    ax.set_ylabel('permutations')
    ax.set_title(f"distance to target: lagging {d['lag_median_px']:.0f} px vs leading {d['lead_median_px']:.0f} px "
                 f"(p={d['p_mw']:.2g})", fontsize=9)
    _despine(ax)
    fig.suptitle('Phase asymmetry: are cells lagging the target in goal-progress phase enhanced and leading cells '
                 'suppressed? (hypothesis from fol_5, exploratory)')
    fig.tight_layout()
    return _save(fig, save_path)


def run(mouse, holo_tag, source_tag, project_root=hio.PROJECT_ROOT, holo_suffix='SR', show=False):
    session = holo.load_holo_session(mouse, holo_tag, source_tag, project_root, holo_suffix)
    out, save_dir = session['paths']['output'], session['paths']['analysis']
    resp = pd.read_csv(out / 'target_responses.csv')
    print('Population ...')
    units, traces = population(session)
    print(f'  {len(units)} units ({(units.kind == "holo").sum()} holo, {(units.kind == "projected").sum()} projected)')
    print('1. Per-pair test (successful / failed trials vs blank trials) ...')
    pairs, _ = follower_stats(session, units, traces)
    print('2. Pooled test ...')
    um_px = hio.um_per_pixel(session['paths']['raw_funcimg'])
    pooled = pooled_by_distance(pairs, bands=DIST_BANDS_UM / um_px)
    print('3. Within-trial correlation, trial-shuffled (successful, failed, blank trials) ...')
    tc = pd.concat([trial_correlation(session, units, traces, ts) for ts in ('successful', 'failed', 'blank')],
                   ignore_index=True)
    print('4. Goal-progress phase ...')
    uph, ph = phase_table(session, units, source_tag, project_root)
    p_phase, per_target, summ = phase_delta(pairs, uph, ph)
    print('5. Phase asymmetry (permutations) ...')
    asym, curves, extra = phase_asymmetry(pairs, uph, ph)

    units.to_csv(out / 'follower_units.csv', index=False)
    pairs.to_csv(out / 'follower_pairs.csv', index=False)
    pooled.to_csv(out / 'follower_pooled.csv', index=False)
    tc.to_csv(out / 'follower_trial_corr.csv', index=False)
    summ.to_csv(out / 'follower_phase.csv', index=False)
    asym.to_csv(out / 'follower_phase_asymmetry.csv', index=False)

    figs = {'overview': plot_overview(pairs, resp, save_dir / 'fol_1_overview.png'),
            'examples': plot_examples(session, units, traces, pairs, save_path=save_dir / 'fol_2_examples.png'),
            'pooled': plot_pooled(pooled, um_px, save_dir / 'fol_3_pooled.png'),
            'corr': plot_trial_correlation(tc, pairs, save_dir / 'fol_4_trial_correlation.png'),
            'phase': plot_phase(p_phase, per_target, summ, save_path=save_dir / 'fol_5_phase.png'),
            'asym': plot_phase_asymmetry(asym, curves, extra, save_dir / 'fol_6_phase_asymmetry.png')}

    real = pairs[(pairs.trial_set == 'successful') & ~pairs.self]
    print(f"\n1. Successful trials: {real.target_source_roi.nunique()} targets x {real.unit.nunique()} units = {len(real)} "
          f"pairs; BH-FDR q < 0.05 at p <= {pairs.attrs['p_cut']:.2e}")
    print('   ' + ', '.join(f'{k} {v}' for k, v in real['class'].value_counts().items()))
    for name in pairs.trial_set.unique():
        g = pairs[(pairs.trial_set == name) & ~pairs.self]
        far = g[~(g.dist_px < 25)]
        print(f"   {name}: {len(g)} pairs, >= 25 px: {100 * (far.p < 0.01).mean():.2f}% p < 0.01, "
              f"{int(far.sig_at_main_cutoff.sum())} at the cut-off, own-family q < 0.05: {int((g.q < 0.05).sum())}")
    print('2. Pooled by distance (successful vs failed, paired Wilcoxon):')
    print(pooled[['band', 'n_pairs', 'z_successful', 'z_failed', 'absz_successful', 'absz_failed', 'p_net', 'q_net',
                  'p_abs', 'q_abs', 'p_succ_vs0', 'q_succ_vs0']].round(4).to_string(index=False))
    print(f"   null expectation of |t|: {pooled.attrs['null_absz']:.3f}")
    print('3. Trial correlation (shift-predictor corrected):')
    for ts, g in tc.groupby('trial_set', sort=False):
        print(f"   {ts}: {g.target_source_roi.nunique()} targets, {len(g)} pairs; p_perm < 0.01: {int((g.p_perm < 0.01).sum())} "
              f"({100 * (g.p_perm < 0.01).mean():.1f}%), q_corr < 0.05: {int((g.q_corr < 0.05).sum())}, median z {g.z_corr.median():.2f}")
    pv = tc.pivot_table(index=['target_source_roi', 'unit'], columns='trial_set', values='z_corr')
    for ctrl in ('failed', 'blank'):
        if ctrl in pv and 'successful' in pv:
            d = pv[['successful', ctrl]].dropna()
            print(f"   successful vs {ctrl}, same pairs (n={len(d)}): median z {d.successful.median():.2f} vs "
                  f"{d[ctrl].median():.2f}, Wilcoxon p={stats.wilcoxon(d.successful, d[ctrl]).pvalue:.2g}")
    print('4. Phase (delta dF/F by bin difference, mean over targets):')
    print(summ.round(4).to_string(index=False))
    print(f"   Kruskal-Wallis across bins: {summ.attrs['kruskal']}")
    print('5. Phase asymmetry (lagging - leading delta dF/F, mean over targets; permutation p):')
    print(asym.round(5).to_string(index=False))
    print(f"   distance check (w = 1/6): {extra['dist']}")
    print(f'Saved fol_*.png to {save_dir} and follower_*.csv to {out}')
    if show:
        plt.show()
    else:
        for f in figs.values():
            if f is not None:
                plt.close(f)
    return units, pairs, pooled, tc, summ


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

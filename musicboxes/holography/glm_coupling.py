"""Coupling GLM: predict every neuron's dF/F from all other neurons and its own stimulation, then ask
whether the coupling weights follow goal-progress phase or physical distance (RSA).

Run from the repo root after followers.py has been importable (uses its population and phase table)
and session_overview_holography.py (dF/F cache):

    python -m musicboxes.holography.glm_coupling 001 holo-001 seq3-007

Model, per receiver unit i (units = followers.population: holo iscell ROIs + projected masks):
    y_i(t) = sum_j beta_ij x_j(t) + own-stimulation kernel_i(t) + any-stimulation kernel(t)
             + gamma * running speed(t) + delta * acceleration(t) + noise
- Data: dF/F (holo_functions.ensure_dfof) averaged in bin_frames bins (4 frames, ~11 Hz), every
  trace z-scored, so beta_ij is in SDs of y_i per SD of x_j.
- Two coupling blocks, fitted as separate models:
    'instant'   x_j(t) = unit j in the same bin, j != i (shared fast activity, undirected in spirit).
    'directed'  x_j(t) = mean of unit j over bins t - past_bins[1] .. t - past_bins[0] (0.09-0.27 s
                before), for all j INCLUDING i (the receiver's own past: Granger-style, so beta_ij is
                the extra prediction of i from j's past beyond i's own past).
- Own-stimulation kernel (targets only): n_basis raised-cosine bumps spanning 0..kernel_s after the
  onsets of all the unit's own trials. Any-stimulation kernel (nuisance, all units): the same basis on
  all non-blank trials; it absorbs the response common to every stimulation (light, sensory) seen in
  followers.py. Running speed: |wheel speed|, binned; acceleration: its time derivative after Gaussian
  smoothing (accel_smooth_s = 0.2 s). Pupil diameter and its derivative (arousal), when pupil.py has
  written <output>/pupil/pupil_Camera2.npz (gaps interpolated, same smoothing).
- Ridge, penalty chosen per unit from alphas by 5-fold cross-validation over contiguous blocks
  (cv R^2 = 1 - SSE / SS around the training mean, summed over folds); the coupling block's
  contribution is the cv R^2 drop when it is removed (reduced model refitted at its own best alpha).
  Everything is solved from block Gram matrices, so one pass over the data per fold.

Coupling analyses (GP-tuned units only have a phase; in sub-001 these are the 103 targets):
- Phase difference dphi_ij = phase(receiver i) - phase(sender j), wrapped to (-0.5, 0.5]; > 0 means
  the receiver's preferred phase lags the sender's. Mean beta vs dphi (kernel and binned) with a band
  from permuting phase labels; sender-phase-bin x receiver-phase-bin mean beta; beta vs distance (um).
- RSA: off-diagonal tuned pairs; the beta matrix (rank-transformed) regressed on rank-transformed models
  phase similarity cos(2 pi dphi), proximity -distance_um and, for 'directed', the antisymmetric
  lag/lead model sin(2 pi dphi). Coefficients tested by permuting the phase labels (phase models) or
  the positions (distance) across units (n_perm). Also each model's Spearman r alone.

Time subsets (conditions; each refitted from scratch on its bins only, with traces and predictors
re-centred and re-scaled within the subset since the model has no intercept, and CV folds contiguous
within the subset; the directed model's past window is taken from the full timeline first):
  full          the whole imaging session
  peri_stim     bins within peri_window (-0.5..+3 s) of a photostimulation train onset (non-blank)
  post_stim     from post_gap_s (10 s) after the last train's offset to the end of imaging: spontaneous
                activity only (holo-001: ~12 min)
  stim_matched  peri-stimulus windows of trials spread evenly over the stimulation period, as many bins
                as post_stim, so that a condition difference is not a difference in data amount
  peri_stim_success, stim_matched_success
                the same, using only trials in which the target responded (target_trial_responses.csv,
                written by session_overview_holography.py)
The RSA permutations use the same label shuffles in every condition (same units, same seed), so the
difference of a coefficient between two conditions is tested against the paired null of differences.

Outputs: <output>/glm_coupling[_<condition>].npz (betas, cv R^2, alphas, unit ids per model),
glm_units[_<condition>].csv, glm_rsa_conditions.csv; plots glm_1_fit, glm_2_phase_<model>, glm_3_rsa
(raw and phase-binned beta matrices, RSA) with a _<condition> suffix for the subsets, and
glm_4_conditions comparing the conditions.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats

from ceph_paths import ceph_path
import musicboxes.holography.holo_io as hio
import musicboxes.holography.holo_functions as holo
import musicboxes.holography.followers as fol

COL = holo.ROLE_COLOURS
MODEL_COL = {'instant': COL['matched'], 'directed': COL['projected']}
ALPHAS = np.logspace(1, 6, 11)


def _despine(ax):
    ax.spines[['top', 'right']].set_visible(False)


def _save(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def _wrap(x):
    return (x + 0.5) % 1 - 0.5


# ---------------------------------------------------------------- design

def bin_traces(traces, bin_frames):
    n = traces.shape[1] // bin_frames
    x = traces[:, :n * bin_frames].reshape(len(traces), n, bin_frames).mean(2)
    return ((x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)).T.astype(np.float64)      # T x U


def raised_cosine(n_basis, length):
    """(length, n_basis) raised-cosine bumps tiling 0..length-1 bins (linear spacing)."""
    t = np.arange(length)
    centres = np.linspace(0, length - 1, n_basis)
    width = (centres[1] - centres[0]) * 2 if n_basis > 1 else length
    arg = np.clip((t[:, None] - centres[None]) * np.pi / width, -np.pi, np.pi)
    return (np.cos(arg) + 1) / 2


def event_kernel_regressors(onset_bins, T, basis):
    """(T, n_basis): sum over events of the basis placed at each onset."""
    x = np.zeros((T, basis.shape[1]))
    L = len(basis)
    for o in onset_bins:
        e = min(o + L, T)
        if o < T:
            x[o:e] += basis[:e - o]
    return x


def past_window(Z, past_bins):
    """Mean of Z over bins t - past_bins[1] .. t - past_bins[0] (zeros where undefined)."""
    lo, hi = past_bins
    out = np.zeros_like(Z)
    for k in range(lo, hi + 1):
        out[k:] += Z[:-k]
    return out / (hi - lo + 1)


# ---------------------------------------------------------------- ridge with blocked CV from Gram matrices

def _folds(T, n_folds):
    edges = np.linspace(0, T, n_folds + 1).astype(int)
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def fit_coupling(Z, C, nuis, own_stim, self_allowed, alphas=ALPHAS, n_folds=5):
    """Ridge GLM for every receiver.

    Z: T x U targets (z-scored traces); C: T x U coupling predictors (same units, possibly lagged);
    nuis: T x q shared nuisance; own_stim: list of (T x k) arrays or None per unit; self_allowed: whether
    unit i's own column of C may predict it (True for the directed model).
    Returns dict: beta (U x U, receiver x sender, NaN where excluded), cv_r2, cv_r2_reduced, alpha.
    """
    T, U = Z.shape
    M = np.hstack([C, nuis])
    P = M.shape[1]
    folds = _folds(T, n_folds)
    G_all, B_all = M.T @ M, M.T @ Z
    G_f = [M[f].T @ M[f] for f in folds]
    B_f = [M[f].T @ Z[f] for f in folds]
    yy_f = [(Z[f] ** 2).sum(0) for f in folds]
    ys_f = [Z[f].sum(0) for f in folds]
    n_f = [f.stop - f.start for f in folds]
    beta = np.full((U, U), np.nan)
    cv_r2, cv_r2_red, best_alpha = np.full(U, np.nan), np.full(U, np.nan), np.full(U, np.nan)

    def gram_parts(i, cols, S):
        """Per fold and full: (XtX, Xty, yty) for receiver i with columns cols of M plus S."""
        out = []
        for f, sl in enumerate(folds + [slice(0, T)]):
            if f < len(folds):
                G, B = G_f[f][np.ix_(cols, cols)], B_f[f][cols, i]
            else:
                G, B = G_all[np.ix_(cols, cols)], B_all[cols, i]
            if S is not None:
                SM = S[sl].T @ M[sl][:, cols]
                G = np.block([[G, SM.T], [SM, S[sl].T @ S[sl]]])
                B = np.r_[B, S[sl].T @ Z[sl, i]]
            out.append((G, B))
        return out

    def cv_fit(parts):
        G_tot, B_tot = parts[-1]
        sse = np.zeros(len(alphas))
        ss = 0.0
        for f in range(len(folds)):
            Gf, Bf = parts[f]
            Gt, Bt = G_tot - Gf, B_tot - Bf
            w, V = np.linalg.eigh(Gt)
            VB = V.T @ Bt
            for a, al in enumerate(alphas):
                b = V @ (VB / (w + al))
                sse[a] += yy_cur[f] - 2 * b @ Bf + b @ Gf @ b
            ss += yy_cur[f] - ys_cur[f] ** 2 / n_f[f]
        a = int(np.argmin(sse))
        r2 = 1 - sse[a] / ss
        w, V = np.linalg.eigh(G_tot)
        return V @ ((V.T @ B_tot) / (w + alphas[a])), r2, alphas[a]

    n_coup = C.shape[1]
    for i in range(U):
        yy_cur = [y[i] for y in yy_f]
        ys_cur = [y[i] for y in ys_f]
        coup = [j for j in range(n_coup) if self_allowed or j != i]
        nuis_cols = list(range(n_coup, P))
        own_prior = [i] if self_allowed else []                     # reduced model keeps the own past
        b, r2, al = cv_fit(gram_parts(i, coup + nuis_cols, own_stim[i]))
        beta[i, coup] = b[:len(coup)]
        cv_r2[i], best_alpha[i] = r2, al
        _, r2r, _ = cv_fit(gram_parts(i, own_prior + nuis_cols, own_stim[i]))
        cv_r2_red[i] = r2r
    return {'beta': beta, 'cv_r2': cv_r2, 'cv_r2_reduced': cv_r2_red, 'alpha': best_alpha}


def build_design(session, units, traces, bin_frames=4, past_bins=(1, 3), kernel_s=4.0, n_basis=5,
                 accel_smooth_s=0.2, pupil_camera='Camera2'):
    t_all = session['trials']
    Z = bin_traces(traces, bin_frames)
    T = len(Z)
    fs_b = session['fs'] / bin_frames
    basis = raised_cosine(n_basis, int(round(kernel_s * fs_b)))
    on_b = (t_all.onset_frame.values // bin_frames).astype(int)
    any_stim = event_kernel_regressors(on_b[~t_all.is_blank.values], T, basis)
    speed = np.abs(np.asarray(session['wheel_speed'], float))
    speed = np.nan_to_num(speed[:T * bin_frames].reshape(T, bin_frames).mean(1))
    # acceleration: derivative of the running speed (deg/s^2), after a sigma = accel_smooth_s Gaussian
    accel = np.gradient(ndimage.gaussian_filter1d(speed, accel_smooth_s * fs_b)) * fs_b
    nuis = [any_stim, speed, accel]
    pupil_path = session['paths']['output'] / 'pupil' / f'pupil_{pupil_camera}.npz'
    if pupil_camera and pupil_path.exists():
        # pupil diameter (pupil.py, resampled to imaging frames) and its rate of change, as arousal nuisance
        d = np.load(pupil_path)['diameter_frames'].astype(float)
        d = np.interp(np.arange(len(d)), np.flatnonzero(np.isfinite(d)), d[np.isfinite(d)])
        d = ndimage.gaussian_filter1d(d[:T * bin_frames].reshape(T, bin_frames).mean(1), accel_smooth_s * fs_b)
        nuis += [d, np.gradient(d) * fs_b]
        print(f'  pupil diameter and its derivative added as nuisance ({pupil_path.name})')
    nuis = np.column_stack(nuis)
    nuis = (nuis - nuis.mean(0)) / nuis.std(0)
    own = []
    for src in units.source_roi:
        if src < 0:
            own.append(None)
            continue
        S = event_kernel_regressors(on_b[(t_all.source_roi == src).values], T, basis)
        own.append((S - S.mean(0)) / S.std(0))
    return Z, past_window(Z, past_bins), nuis, own, fs_b


# ---------------------------------------------------------------- coupling vs phase and distance

def pair_table(beta, units, uph, um_per_px):
    """Long table of sender -> receiver weights with phase difference and distance."""
    U = len(units)
    i, j = np.where(~np.eye(U, dtype=bool) & np.isfinite(beta))
    ph = units.unit.map(uph.phase).values
    xy = units[['x', 'y']].values
    d = np.hypot(*(xy[i] - xy[j]).T) * um_per_px
    return pd.DataFrame({'receiver': units.unit.values[i], 'sender': units.unit.values[j], 'beta': beta[i, j],
                         'beta_rev': beta[j, i], 'dist_um': d, 'phase_r': ph[i], 'phase_s': ph[j],
                         'dphi': _wrap(ph[i] - ph[j])})


def phase_curve(pt, kernel_bw=0.05, n_perm=500, seed=0):
    tuned = pt.dropna(subset=['dphi'])
    grid = np.linspace(-0.5, 0.5, 101)

    def curve(dphi):
        k = np.exp(-_wrap(grid[:, None] - dphi[None]) ** 2 / (2 * kernel_bw ** 2))
        return (k @ tuned.beta.values) / k.sum(1)

    obs = curve(tuned.dphi.values)
    # null: permute phase labels across units
    rng = np.random.default_rng(seed)
    units_ph = pd.concat([tuned[['receiver', 'phase_r']].set_axis(['u', 'p'], axis=1),
                          tuned[['sender', 'phase_s']].set_axis(['u', 'p'], axis=1)]).drop_duplicates('u')
    names, phs = units_ph.u.values, units_ph.p.values
    ri = pd.Index(names).get_indexer(tuned.receiver)
    si = pd.Index(names).get_indexer(tuned.sender)
    null = np.array([curve(_wrap(p[ri] - p[si])) for p in (rng.permutation(phs) for _ in range(n_perm))])
    return grid, obs, np.percentile(null, [2.5, 97.5], axis=0)


def rsa(pt, directed, n_perm=2000, seed=0):
    """Rank regression of beta on phase / distance models over tuned off-diagonal pairs, with label
    permutations. Returns (DataFrame of coefficients, null samples)."""
    rng = np.random.default_rng(seed)
    t = pt.dropna(subset=['dphi']).reset_index(drop=True)
    names = np.unique(np.r_[t.receiver, t.sender])
    ri, si = np.searchsorted(names, t.receiver), np.searchsorted(names, t.sender)
    ph = pd.concat([t.set_index('receiver').phase_r, t.set_index('sender').phase_s])
    ph = ph[~ph.index.duplicated()].reindex(names).values
    d = t.dist_um.values
    y = stats.rankdata(t.beta.values)

    def models(p, dist):
        dphi = _wrap(p[ri] - p[si])
        m = {'phase_similarity': np.cos(2 * np.pi * dphi), 'proximity': -dist}
        if directed:
            m['lag_lead'] = np.sin(2 * np.pi * dphi)
        return m

    def fit(m):
        X = np.column_stack([stats.rankdata(v) for v in m.values()])
        X = (X - X.mean(0)) / X.std(0)
        yy = (y - y.mean()) / y.std()
        coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(yy)), X], yy, rcond=None)
        return dict(zip(m, coef[1:]))

    obs_m = models(ph, d)
    obs = fit(obs_m)
    alone = {k: stats.spearmanr(v, t.beta.values).statistic for k, v in obs_m.items()}
    # permutations: phase labels for phase models, positions (via distance matrix relabelling) for proximity
    D = np.full((len(names), len(names)), np.nan)
    D[ri, si] = d
    null = {k: [] for k in obs}
    for _ in range(n_perm):
        pp = rng.permutation(len(names))
        m_phase = models(ph[pp], d)
        pd_ = rng.permutation(len(names))
        m_dist = models(ph, D[pd_][:, pd_][ri, si])
        f_phase, f_dist = fit(m_phase), fit(m_dist)
        for k in obs:
            null[k].append(f_dist[k] if k == 'proximity' else f_phase[k])
    rows = [{'model': k, 'coef': obs[k], 'spearman_alone': alone[k],
             'p_perm': (1 + np.sum(np.abs(null[k]) >= abs(obs[k]))) / (1 + n_perm),
             'null_lo': np.percentile(null[k], 2.5), 'null_hi': np.percentile(null[k], 97.5),
             'n_pairs': len(t), 'n_units': len(names)} for k in obs]
    return pd.DataFrame(rows), {k: np.array(v) for k, v in null.items()}


# ---------------------------------------------------------------- plots

def plot_fit(fits, units, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    for name, f in fits.items():
        ax.hist(f['cv_r2'], bins=40, histtype='step', lw=1.5, color=MODEL_COL[name],
                label=f'{name} (median {np.nanmedian(f["cv_r2"]):.3f})')
    ax.set_xlabel('cross-validated R^2 (full model)')
    ax.set_ylabel('units')
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)
    ax = axes[1]
    for name, f in fits.items():
        d = f['cv_r2'] - f['cv_r2_reduced']
        ax.hist(d, bins=40, histtype='step', lw=1.5, color=MODEL_COL[name], label=f'{name} (median {np.nanmedian(d):.3f})')
    ax.set_xlabel('cv R^2 gained by the coupling block')
    ax.set_ylabel('units')
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)
    ax = axes[2]
    for name, f in fits.items():
        b = f['beta'][np.isfinite(f['beta']) & ~np.eye(len(units), dtype=bool)]
        ax.hist(b, bins=np.linspace(*np.percentile(b, [0.2, 99.8]), 80), histtype='step', lw=1.5,
                color=MODEL_COL[name], label=name, density=True)
    ax.set_yscale('log')
    ax.set_xlabel('coupling weight beta (receiver SD per sender SD)')
    ax.set_ylabel('density')
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)
    fig.suptitle('Coupling GLM fit: every unit predicted from all others (+ own and any-stimulation kernels, speed); '
                 'ridge, alpha by blocked 5-fold CV')
    fig.tight_layout()
    return _save(fig, save_path)


def plot_phase(pt, name, n_phase_bins=6, save_path=None):
    grid, obs, band = phase_curve(pt)
    tuned = pt.dropna(subset=['dphi'])
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    ax = axes[0]
    ax.fill_between(grid, band[0], band[1], color='0.85', lw=0, label='phase labels permuted, 95%')
    ax.plot(grid, obs, color=MODEL_COL[name], lw=2, label='mean beta')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('phase(receiver) - phase(sender)  (> 0: receiver lags)')
    ax.set_ylabel('coupling beta (kernel mean)')
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)
    ax = axes[1]
    half = n_phase_bins // 2
    db = (np.round(tuned.dphi * n_phase_bins)).astype(int).clip(-half, half - 1)
    g = tuned.groupby(db).beta
    ax.errorbar(g.mean().index, g.mean(), yerr=g.sem(), marker='o', color=MODEL_COL[name])
    ax.axhline(tuned.beta.mean(), color='k', ls=':', lw=1, label='mean over tuned pairs')
    ax.set_xticks(range(-half, half))
    ax.set_xlabel(f'phase difference ({n_phase_bins} bins, receiver - sender)')
    ax.set_ylabel('mean beta +- sem')
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)
    ax = axes[2]
    rb = np.floor(tuned.phase_r * n_phase_bins).clip(0, n_phase_bins - 1)
    sb = np.floor(tuned.phase_s * n_phase_bins).clip(0, n_phase_bins - 1)
    mat = tuned.assign(rb=rb, sb=sb).pivot_table(index='sb', columns='rb', values='beta', aggfunc='mean')
    lim = np.nanmax(np.abs(mat.values - tuned.beta.mean()))
    im = ax.imshow(mat.values, cmap=holo.colormap(), vmin=tuned.beta.mean() - lim, vmax=tuned.beta.mean() + lim)
    ax.set_xlabel('receiver phase bin')
    ax.set_ylabel('sender phase bin')
    fig.colorbar(im, ax=ax, label='mean beta')
    ax = axes[3]
    ax.scatter(pt.dist_um, pt.beta, s=1, color='0.7', rasterized=True)
    bins = np.r_[0, 20, 40, 60, 80, 120, 160, 240, 320, 600]
    w = np.digitize(pt.dist_um, bins) - 1
    ax.plot(bins[:-1] + np.diff(bins) / 2, [pt.beta[w == k].mean() for k in range(len(bins) - 1)], 'o-',
            color=MODEL_COL[name], label='mean per bin (all units)')
    ax.set_ylim(np.percentile(pt.beta, [0.5, 99.5]))
    ax.set_xlabel('distance sender - receiver (um)')
    ax.set_ylabel('coupling beta')
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)
    fig.suptitle(f"'{name}' coupling vs goal-progress phase (tuned units, {tuned.receiver.nunique()} cells) "
                 f"and physical distance (all {pt.receiver.nunique()} units)")
    fig.tight_layout()
    return _save(fig, save_path)


def _phase_sorted_matrix(pt):
    t = pt.dropna(subset=['dphi'])
    order = pd.concat([t.set_index('receiver').phase_r, t.set_index('sender').phase_s])
    order = order[~order.index.duplicated()].sort_values()
    idx = pd.Index(order.index)
    B = np.full((len(idx), len(idx)), np.nan)
    B[idx.get_indexer(t.receiver), idx.get_indexer(t.sender)] = t.beta
    return B, order.values


def binned_matrix(pt, n_bins=20):
    """Mean beta per sender-phase bin x receiver-phase bin (tuned pairs), minus the mean over tuned pairs."""
    t = pt.dropna(subset=['dphi'])
    rb = np.floor(t.phase_r * n_bins).clip(0, n_bins - 1).astype(int)
    sb = np.floor(t.phase_s * n_bins).clip(0, n_bins - 1).astype(int)
    sm = np.zeros((n_bins, n_bins))
    n = np.zeros((n_bins, n_bins))
    np.add.at(sm, (rb, sb), t.beta.values)
    np.add.at(n, (rb, sb), 1)
    with np.errstate(invalid='ignore'):
        return sm / n - t.beta.mean(), n


def plot_rsa(results, pts, n_bins=20, title='', save_path=None):
    """Top: per coupling model its beta matrix with units sorted by preferred phase, the phase-similarity
    model and per coupling model the RSA coefficients against their permutation nulls. Bottom: the same
    beta matrices averaged in n_bins x n_bins phase bins (receiver x sender, minus the mean), and the
    number of pairs per bin."""
    names = list(results)
    n = 2 * len(names) + 1
    fig, axes2 = plt.subplots(2, n, figsize=(5.2 * n, 10))
    axes = axes2[0]
    for k, name in enumerate(names):
        M, cnt = binned_matrix(pts[name], n_bins)
        ax = axes2[1, k]
        lim = np.nanpercentile(np.abs(M), 98)
        im = ax.imshow(M, cmap=holo.colormap(), vmin=-lim, vmax=lim, extent=[0, 1, 1, 0])
        ax.plot([0, 1], [0, 1], color='k', lw=0.5, ls=':')
        ax.set_title(f"'{name}' beta, {n_bins} x {n_bins} phase bins (minus mean)", fontsize=9)
        ax.set_xlabel('sender preferred phase')
        ax.set_ylabel('receiver preferred phase')
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes2[1, len(names)]
    im = ax.imshow(cnt, cmap='Greys', extent=[0, 1, 1, 0])
    ax.set_title('pairs per bin', fontsize=9)
    ax.set_xlabel('sender preferred phase')
    ax.set_ylabel('receiver preferred phase')
    fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes2[1, len(names) + 1:]:
        ax.set_axis_off()
    if title:
        fig.suptitle(title)
    for ax, name in zip(axes, names):
        B, ph = _phase_sorted_matrix(pts[name])
        lim = np.nanpercentile(np.abs(B), 98)
        im = ax.imshow(B, cmap=holo.colormap(), vmin=-lim, vmax=lim)
        ax.set_title(f"'{name}' beta, units sorted by phase", fontsize=9)
        ax.set_xlabel('sender')
        ax.set_ylabel('receiver')
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[len(names)]
    ax.imshow(np.cos(2 * np.pi * _wrap(ph[:, None] - ph[None])), cmap=holo.colormap())
    ax.set_title('phase-similarity model cos(2 pi dphi)', fontsize=9)
    ax.set_xlabel('sender')
    ax.set_ylabel('receiver')
    for ax, name in zip(axes[len(names) + 1:], names):
        r, null = results[name]
        x = np.arange(len(r))
        for k, row in r.iterrows():
            parts = ax.violinplot(null[row.model], positions=[k], showextrema=False, widths=0.7)
            for b in parts['bodies']:
                b.set_facecolor('0.8')
                b.set_alpha(1)
        ax.scatter(x, r.coef, color=MODEL_COL[name], zorder=3, s=40)
        for k, row in r.iterrows():
            ax.text(k, row.coef, f'  p={row.p_perm:.2g}', fontsize=8, va='center')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(x, r.model, rotation=20)
        ax.set_ylabel('standardised rank-regression coefficient')
        ax.set_title(f"RSA, '{name}' coupling ({int(r.n_pairs.iloc[0])} pairs); grey: permutation null", fontsize=9)
        _despine(ax)
    fig.tight_layout()
    return _save(fig, save_path)


# ---------------------------------------------------------------- time subsets

def condition_masks(session, T, bin_frames, peri_window=(-0.5, 3.0), post_gap_s=10.0):
    """Boolean masks over the T bins for the conditions (see module docstring)."""
    t_all = session['trials']
    bt = session['frame_times'][np.arange(T) * bin_frames]
    post = bt >= t_all.offset_time.max() + post_gap_s
    per_trial = int(round((peri_window[1] - peri_window[0]) * session['fs'] / bin_frames))

    def peri_of(on):
        m = np.zeros(T, bool)
        near = np.searchsorted(on, bt)
        for k in (near - 1, near):
            k = np.clip(k, 0, len(on) - 1)
            m |= (bt >= on[k] + peri_window[0]) & (bt <= on[k] + peri_window[1])
        return m

    def matched_of(on):
        # peri windows of evenly spaced trials until as many bins as post_stim
        n_tr = min(len(on), int(np.ceil(post.sum() / per_trial)))
        m = peri_of(on[np.linspace(0, len(on) - 1, n_tr).round().astype(int)])
        m[np.flatnonzero(m)[post.sum():]] = False
        return m

    stim = t_all[~t_all.is_blank]
    on_all = stim.onset_time.values
    succ = stim.trial.map(fol.trial_success(session)).eq(True).values
    return {'full': np.ones(T, bool), 'peri_stim': peri_of(on_all), 'post_stim': post,
            'stim_matched': matched_of(on_all), 'peri_stim_success': peri_of(on_all[succ]),
            'stim_matched_success': matched_of(on_all[succ])}


def subset(Z, Zpast, nuis, own, mask):
    """Rows of the design in mask, every column re-centred and re-scaled within the subset (all-zero
    columns, e.g. stimulation kernels in post_stim, stay zero)."""
    def st(x):
        x = x[mask]
        sd = x.std(0)
        return (x - x.mean(0)) / np.where(sd > 0, sd, 1)
    return st(Z), st(Zpast), st(nuis), [None if o is None else st(o) for o in own]


def plot_conditions(table, curves, save_path=None):
    conds = list(dict.fromkeys(table.condition))
    cmap = dict(zip(conds, era_cols(len(conds))))
    models = list(dict.fromkeys(table.coupling))
    fig, axes = plt.subplots(1, 2 * len(models), figsize=(6 * 2 * len(models), 4.6))
    for m, model in enumerate(models):
        ax = axes[2 * m]
        d = table[table.coupling == model]
        for j, rsa_model in enumerate(['phase_similarity', 'proximity']):
            for c, cond in enumerate(conds):
                r = d[(d.condition == cond) & (d.model == rsa_model)]
                if r.empty:
                    continue
                x = j + (c - (len(conds) - 1) / 2) * 0.18
                ax.plot([x, x], [r.null_lo.iloc[0], r.null_hi.iloc[0]], color='0.75', lw=6, solid_capstyle='butt')
                ax.plot(x, r.coef.iloc[0], 'o', color=cmap[cond], ms=7, label=cond if j == 0 else None)
                ax.text(x, r.coef.iloc[0], f' {r.p_perm.iloc[0]:.2g}', fontsize=7, va='bottom')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks([0, 1], ['phase similarity', 'proximity'])
        ax.set_ylabel('RSA coefficient (grey: null 95%)')
        ax.set_title(f"'{model}' coupling; labels: permutation p", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        _despine(ax)
        ax = axes[2 * m + 1]
        for cond in conds:
            if (model, cond) in curves:
                g, y = curves[(model, cond)]
                ax.plot(g, y, color=cmap[cond], lw=1.8, label=cond)
        ax.axvline(0, color='k', lw=0.6)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('phase(receiver) - phase(sender)')
        ax.set_ylabel('beta, z-scored within condition (kernel mean)')
        ax.legend(frameon=False, fontsize=8)
        _despine(ax)
    fig.suptitle('Phase coupling during stimulation vs spontaneous activity')
    fig.tight_layout()
    return _save(fig, save_path)


def era_cols(n):
    import era_brewer
    return era_brewer.era_brew('Lover1', n=max(n, 2), brew_type='discrete')[:n]


def zcurve(pt, kernel_bw=0.05):
    t = pt.dropna(subset=['dphi'])
    z = (t.beta - t.beta.mean()) / t.beta.std()
    grid = np.linspace(-0.5, 0.5, 101)
    k = np.exp(-_wrap(grid[:, None] - t.dphi.values[None]) ** 2 / (2 * kernel_bw ** 2))
    return grid, (k @ z.values) / k.sum(1)


# ---------------------------------------------------------------- run

def run(mouse, holo_tag, source_tag, project_root=hio.PROJECT_ROOT, holo_suffix='SR', bin_frames=4,
        past_bins=(1, 3), conditions=('full', 'peri_stim', 'post_stim', 'stim_matched', 'peri_stim_success',
                                      'stim_matched_success'), show=False):
    session = holo.load_holo_session(mouse, holo_tag, source_tag, project_root, holo_suffix)
    out, save_dir = session['paths']['output'], session['paths']['analysis']
    um_px = hio.um_per_pixel(session['paths']['raw_funcimg'])
    print('Population and design ...')
    units, traces = fol.population(session)
    Z, Zpast, nuis, own, fs_b = build_design(session, units, traces, bin_frames, past_bins)
    print(f'  {len(units)} units, {len(Z)} bins at {fs_b:.1f} Hz, {sum(o is not None for o in own)} with own-stim kernels')
    masks = condition_masks(session, len(Z), bin_frames)
    uph, _ = fol.phase_table(session, units, source_tag, project_root)
    rows, curves, figs, all_fits = [], {}, {}, {}
    for cond in conditions:
        m = masks[cond]
        sfx = '' if cond == 'full' else f'_{cond}'
        print(f"=== {cond}: {m.sum()} bins ({m.sum() / fs_b / 60:.1f} min) ===")
        z, zp, nu, ow = subset(Z, Zpast, nuis, own, m) if cond != 'full' else (Z, Zpast, nuis, own)
        fits = {'instant': fit_coupling(z, z, nu, ow, self_allowed=False),
                'directed': fit_coupling(z, zp, nu, ow, self_allowed=True)}
        all_fits[cond] = fits
        pts, results = {}, {}
        for name, f in fits.items():
            pt = pair_table(f['beta'], units, uph, um_px)
            if name == 'directed':                               # own-past (AR) weights are not coupling
                pt = pt[pt.receiver != pt.sender]
            pts[name] = pt
            results[name] = rsa(pt, directed=(name == 'directed'))
            curves[(name, cond)] = zcurve(pt)
            r, null = results[name]
            for _, row in r.iterrows():
                rows.append({**row.to_dict(), 'coupling': name, 'condition': cond, 'n_bins': int(m.sum()),
                             '_null': null[row.model]})
            figs[f'phase_{name}{sfx}'] = plot_phase(pt, name, save_path=save_dir / f'glm_2_phase_{name}{sfx}.png')
        figs[f'fit{sfx}'] = plot_fit(fits, units, save_dir / f'glm_1_fit{sfx}.png')
        figs[f'rsa{sfx}'] = plot_rsa(results, pts, title=f'{cond}: {m.sum() / fs_b / 60:.1f} min',
                                     save_path=save_dir / f'glm_3_rsa{sfx}.png')
        np.savez_compressed(out / f'glm_coupling{sfx}.npz', units=units.unit.values.astype(str), bin_frames=bin_frames,
                            past_bins=np.array(past_bins), mask=m,
                            **{f'{n}_{k}': v for n, f in fits.items() for k, v in f.items()})
        u = units.copy()
        for n, f in fits.items():
            u[f'cv_r2_{n}'], u[f'cv_r2_reduced_{n}'], u[f'alpha_{n}'] = f['cv_r2'], f['cv_r2_reduced'], f['alpha']
        u['phase'] = u.unit.map(uph.phase)
        u.to_csv(out / f'glm_units{sfx}.csv', index=False)
        for n, f in fits.items():
            print(f"  {n}: cv R^2 median {np.nanmedian(f['cv_r2']):.3f}, coupling gain "
                  f"{np.nanmedian(f['cv_r2'] - f['cv_r2_reduced']):.3f}; " +
                  ', '.join(f"{r.model} {r.coef:.4f} (p={r.p_perm:.2g})" for _, r in results[n][0].iterrows()))

    table = pd.DataFrame(rows)
    # paired differences between conditions (same permutations in every condition)
    diffs = []
    for name in table.coupling.unique():
        for model in table.model.unique():
            d = table[(table.coupling == name) & (table.model == model)].set_index('condition')
            for a_, b_ in (('peri_stim', 'post_stim'), ('stim_matched', 'post_stim'), ('full', 'post_stim'),
                           ('peri_stim_success', 'post_stim'), ('stim_matched_success', 'post_stim'),
                           ('peri_stim_success', 'peri_stim'), ('stim_matched_success', 'stim_matched')):
                if a_ in d.index and b_ in d.index:
                    obs = d.coef[a_] - d.coef[b_]
                    nd = d._null[a_] - d._null[b_]
                    diffs.append({'coupling': name, 'model': model, 'contrast': f'{a_} - {b_}', 'diff': obs,
                                  'p_perm': (1 + np.sum(np.abs(nd) >= abs(obs))) / (1 + len(nd))})
    diffs = pd.DataFrame(diffs)
    table.drop(columns='_null').to_csv(out / 'glm_rsa_conditions.csv', index=False)
    if 'full' in conditions:
        table.drop(columns='_null').query("condition == 'full'").to_csv(out / 'glm_rsa.csv', index=False)
    diffs.to_csv(out / 'glm_rsa_condition_diffs.csv', index=False)
    figs['conditions'] = plot_conditions(table, curves, save_dir / 'glm_4_conditions.png')
    print(table.drop(columns='_null')[['coupling', 'condition', 'n_bins', 'model', 'coef', 'p_perm']].round(4).to_string(index=False))
    print(diffs.round(4).to_string(index=False))
    print(f'Saved glm_*.png to {save_dir} and glm_*.{{npz,csv}} to {out}')
    if show:
        plt.show()
    else:
        for fg in figs.values():
            plt.close(fg)
    return units, all_fits, table, diffs


def _cli():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('mouse')
    p.add_argument('holo_tag')
    p.add_argument('source_tag')
    p.add_argument('--project-root', default=str(hio.PROJECT_ROOT))
    p.add_argument('--holo-suffix', default='SR')
    p.add_argument('--bin-frames', type=int, default=4)
    p.add_argument('--conditions', nargs='+', default=None, help='subset of the conditions (default: all)')
    p.add_argument('--show', action='store_true')
    a = p.parse_args()
    kw = {'conditions': tuple(a.conditions)} if a.conditions else {}
    run(a.mouse, a.holo_tag, a.source_tag, ceph_path(a.project_root), a.holo_suffix, a.bin_frames, show=a.show, **kw)


if __name__ == '__main__':
    _cli()

"""Goal-progress (GP) tuning from a circular-Gaussian (von Mises) fit to per-frame phases, without binning.

This is the default GP tuning method: gp_learning_helpers.gp_tuning(method='vm') and
musicboxes/session_overview.py (gp_method='vm') both call vm_tuning here. The binned method is kept as
method='binned' / gp_method='binned'. compare_gp_methods.py compares the two on synthetic cells and on
sub-001 seq3-007 (results on the Desktop, gp_method_comparison/): the fit has the same false-positive rate
(~alpha), detects weakly tuned cells more often and gives a more precise and more split-half-stable
phase.

Binned method: each reward -> reward segment is cut into `bins` equal parts, bin means are averaged over
segments, the preferred phase is the peak bin of the (smoothed) mean curve, and significance is the
(max-min)/mean tuning index against a circular-shift shuffle within segments. The phase estimate is
quantised to 1/bins and depends on the bin count, and the peak of a noisy curve is a noisy statistic.

Fit method: every frame inside a segment gets its phase phi = (frame - start) / (end - start), exactly as
gp_learning_helpers.frame_phase, and each cell's dF over all frames is regressed on phi with

    dF(phi) = b + A * exp(kappa * (cos 2 pi (phi - mu) - 1)),     A >= 0, kappa > 0, mu in [0, 1)

i.e. a von Mises bump on a baseline. mu is the preferred phase; kappa sets the width (vm_fwhm gives the
full width at half maximum in phase units). Fitting this regression is the same as fitting a von Mises to
the dF-weighted phase distribution, but it allows a baseline and negative dF.

Design decisions
----------------
  - Self-contained (numpy / scipy only), so gp_learning_helpers and session_overview can both import it
    without an import cycle. frame_layout and shuffled_labels reproduce gp_learning_helpers.segment_layout
    and shuffled_groups exactly (same frames, same rng draws).
  - Two-stage fit. For fixed (mu, kappa) the model is linear in (b, A), and the least-squares fit's R^2 is
    the squared correlation r(dF, g_{mu,kappa}). A grid of N_MU x len(KAPPAS) basis functions therefore
    gives the best (mu, kappa) for all cells at once from one matrix product; curve_fit then refines
    (b, A, mu, kappa) per cell on the exact per-frame phases (refine=True).
  - The grid stage uses sufficient statistics on N_FINE = 500 phase bins (per cell: sum of dF per fine
    phase bin; per bin: frame count), with the basis evaluated at the fine-bin centres. This is a numerical
    shortcut, not the analysis binning: it changes a basis value by at most a 1/1000 phase shift, and the
    reported mu comes from the refinement on exact frame phases. It makes each shuffle a sparse sum over
    frames plus a (cells x 500) @ (500 x grid) product.
  - Significance: test statistic = best grid R^2 restricted to positive correlations (a bump, not a dip).
    Null = the within-segment circular shift of gp_tuning. With the same seed and segments both methods
    draw identical shifts, so their nulls are paired shuffle for shuffle. The statistic is the grid R^2
    (not the refined one) so observed and shuffled values are computed the same way; refining thousands
    of shuffles per cell would be slow and changes R^2 very little.
  - Split halves: even and odd segments are fitted separately (phase_even / phase_odd).
  - Known limitation: a cell with two fields gets one bump (usually the larger field) or, if the fields
    are similar, one very broad bump between them (FWHM -> 1). A FWHM near 1 means "not a single bump":
    treat its mu with care.
"""

import numpy as np
from scipy import sparse
from scipy.optimize import curve_fit

N_FINE = 500
N_MU = 72
KAPPAS = np.array([0.3, 0.7, 1.5, 3, 5, 8, 12, 20, 35])


def vm_shape(phase, mu, kappa):
    return np.exp(kappa * (np.cos(2 * np.pi * (phase - mu)) - 1))


def vm_model(phase, b, a, mu, kappa):
    return b + a * vm_shape(phase, mu, kappa)


def vm_fwhm(kappa):
    """Full width at half maximum of the bump (in phase units, 1 = whole segment)."""
    c = 1 - np.log(2) / np.asarray(kappa, float)
    return np.where(c <= -1, 1.0, np.arccos(np.clip(c, -1, 1)) / np.pi)


def frame_layout(segments):
    """Frames inside the (start, end) segments, in order, with segment id, position in the segment, exact
    phase and fine phase bin ('group'). Same frames and fields as gp_learning_helpers.segment_layout."""
    frames, seg = [], []
    segments = np.asarray(segments, dtype=int).reshape(-1, 2)
    for i, (s, e) in enumerate(segments):
        if e > s:
            frames.append(np.arange(s, e))
            seg.append(np.full(e - s, i))
    n_seg = len(segments)
    frames, seg = np.concatenate(frames), np.concatenate(seg)
    seg_start = np.searchsorted(seg, np.arange(n_seg))
    seg_len = np.bincount(seg, minlength=n_seg)
    pos = np.arange(len(frames)) - seg_start[seg]
    phase = pos / seg_len[seg]
    return dict(frames=frames, seg=seg, pos=pos, seg_start=seg_start, seg_len=seg_len, n_seg=n_seg, phase=phase,
                group=np.minimum((phase * N_FINE).astype(int), N_FINE - 1))


def shuffled_labels(layout, rng):
    """Circularly shift each segment's frames by an independent random amount (as
    gp_learning_helpers.shuffled_groups, same rng draws)."""
    seg, pos, seg_start, seg_len = layout['seg'], layout['pos'], layout['seg_start'], layout['seg_len']
    shift = rng.integers(0, np.maximum(seg_len, 1))
    src = seg_start[seg] + (pos + shift[seg]) % seg_len[seg]
    new_group = np.empty_like(layout['group'])
    new_group[src] = layout['group']
    return new_group


def _grid():
    mus = np.arange(N_MU) / N_MU
    mu, kappa = np.meshgrid(mus, KAPPAS, indexing='ij')
    mu, kappa = mu.ravel(), kappa.ravel()
    centres = (np.arange(N_FINE) + 0.5) / N_FINE
    return mu, kappa, vm_shape(centres[:, None], mu[None], kappa[None])   # (N_FINE, n_grid)


GRID_MU, GRID_KAPPA, GRID_BASIS = _grid()


def fine_sums(Y, labels):
    """(n_cells, N_FINE) sum of Y over frames with each fine phase label."""
    ind = sparse.csr_matrix((np.ones(len(labels)), (np.arange(len(labels)), labels)), shape=(len(labels), N_FINE))
    return np.asarray((ind.T @ Y.T).T)


def grid_corr(S, counts, sum_y, var_y):
    """Correlation of every cell with every grid basis function, from fine-bin sums S and counts.
    Returns (r (n_cells, n_grid), gbar, var_g)."""
    n = counts.sum()
    gbar = counts @ GRID_BASIS / n
    var_g = counts @ GRID_BASIS ** 2 - n * gbar ** 2
    cov = S @ GRID_BASIS - sum_y[:, None] * gbar[None]
    return cov / np.sqrt(np.maximum(var_y[:, None] * var_g[None], 1e-300)), gbar, var_g


def best_grid_stat(r):
    """Best R^2 over the grid among positive correlations, and its grid index."""
    rp = np.maximum(np.nan_to_num(r), 0)
    k = np.argmax(rp, axis=1)
    return rp[np.arange(len(k)), k] ** 2, k


def fit_vm(y, phase, p0):
    """curve_fit of vm_model on exact frame phases from p0 = (b, a, mu, kappa); falls back to p0 on failure.
    Returns (b, a, mu mod 1, kappa, R^2)."""
    lo, hi = [-np.inf, 0, p0[2] - 0.5, 0.05], [np.inf, np.inf, p0[2] + 0.5, 200]
    try:
        p, _ = curve_fit(vm_model, phase, y, p0=np.clip(p0, np.add(lo, 1e-9), np.subtract(hi, 1e-9)),
                         bounds=(lo, hi), maxfev=2000)
    except (RuntimeError, ValueError):
        p = np.asarray(p0, float)
    resid = y - vm_model(phase, *p)
    r2 = 1 - np.sum(resid ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-300)
    return p[0], p[1], p[2] % 1, p[3], r2


def _moments(Y):
    sum_y = Y.sum(axis=1)
    return sum_y, (Y ** 2).sum(axis=1) - sum_y ** 2 / Y.shape[1]


def vm_fit_cells(Y, layout, refine=True):
    """Von Mises fit of every row of Y (n_cells, n_layout_frames). Returns dict of per-cell arrays:
    phase, kappa, fwhm, amp, baseline, r2 (refined fit), stat (best grid R^2)."""
    labels = layout['group']
    counts = np.bincount(labels, minlength=N_FINE).astype(float)
    sum_y, var_y = _moments(Y)
    r, gbar, var_g = grid_corr(fine_sums(Y, labels), counts, sum_y, var_y)
    stat, k = best_grid_stat(r)
    cov = r[np.arange(len(k)), k] * np.sqrt(var_y * var_g[k])
    amp = np.maximum(cov / var_g[k], 0)
    base = sum_y / Y.shape[1] - amp * gbar[k]
    out = dict(phase=GRID_MU[k].copy(), kappa=GRID_KAPPA[k].copy(), amp=amp, baseline=base, r2=stat.copy(), stat=stat)
    if refine:
        for i in range(Y.shape[0]):
            if amp[i] <= 0:
                continue
            b, a, mu, kap, r2 = fit_vm(Y[i], layout['phase'], (base[i], amp[i], GRID_MU[k[i]], GRID_KAPPA[k[i]]))
            out['baseline'][i], out['amp'][i], out['phase'][i], out['kappa'][i], out['r2'][i] = b, a, mu, kap, r2
    out['fwhm'] = vm_fwhm(out['kappa'])
    return out


def shuffle_stats(Y, layout, n_shuffles, seed=0):
    """(n_cells, n_shuffles) best grid R^2 after circular shifts within segments."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(layout['group'], minlength=N_FINE).astype(float)
    sum_y, var_y = _moments(Y)
    out = np.empty((Y.shape[0], n_shuffles))
    for j in range(n_shuffles):
        r, _, _ = grid_corr(fine_sums(Y, shuffled_labels(layout, rng)), counts, sum_y, var_y)
        out[:, j] = best_grid_stat(r)[0]
    return out


def vm_tuning(signal, segments, n_shuffles=500, alpha=0.05, seed=0, refine=True, halves=True):
    """Von Mises GP tuning of every row of `signal` (n_cells, n_frames) on (start, end) frame segments.

    Returns dict of per-cell arrays: phase (mu, 0-1), kappa, fwhm, amp, baseline (units of `signal`), r2
    (refined fit), stat (best grid R^2, the test statistic), p, significant, shuffle_z, shuffle_mean
    (mean shuffled stat; the last four NaN / False without shuffles), and with halves=True phase_even / phase_odd (fits on even / odd segments; NaN if a half has
    no frames)."""
    segments = np.asarray(segments, dtype=int).reshape(-1, 2)
    layout = frame_layout(segments)
    Y = np.asarray(signal[:, layout['frames']], float)
    out = vm_fit_cells(Y, layout, refine=refine)
    if n_shuffles:
        shuf = shuffle_stats(Y, layout, n_shuffles, seed)
        p = (1 + np.sum(shuf >= out['stat'][:, None], axis=1)) / (1 + n_shuffles)
        shuffle_mean = shuf.mean(1)
        shuffle_z = (out['stat'] - shuffle_mean) / shuf.std(1)
    else:
        p, shuffle_z, shuffle_mean = (np.full(len(Y), np.nan) for _ in range(3))
    out.update(p=p, significant=p < alpha, shuffle_z=shuffle_z, shuffle_mean=shuffle_mean)
    if halves:
        for name, half in (('even', segments[0::2]), ('odd', segments[1::2])):
            if np.any(half[:, 1] > half[:, 0]):
                hl = frame_layout(half)
                out[f'phase_{name}'] = vm_fit_cells(np.asarray(signal[:, hl['frames']], float), hl, refine=refine)['phase']
            else:
                out[f'phase_{name}'] = np.full(len(Y), np.nan)
    return out

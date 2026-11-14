"""Map the ROIs of a holography session onto the ROIs of the session its targets came from.

Photostimulation targets are ROIs of an earlier "source" session (seq3-007 for sub-001 holo-001):
condition 'ensembleN' aims at source ROI N (see holo_io). The holo session has its own cellpose/
suite2p segmentation, so its ROI indices are unrelated. This script finds, for every source ROI,
the holo ROI covering the same cell, and keeps the source mask of stimulated cells that have no holo
counterpart so their traces can still be extracted (extract_projected.py).

Run from the repo root (bonsai_abcd env):

    python -m musicboxes.holography.match_rois 001 holo-001 seq3-007 [--iou 0.3]

Method
------
1. Rigid FOV shift: skimage phase_cross_correlation(holo meanImg, source meanImg, upsample 10),
   i.e. the (dy, dx) that moves source pixels into the holo FOV. The chan2 mean images give an
   independent estimate, and the same is done per FOV quadrant to see non-rigid residuals; both are
   reported, only the full-FOV chan1 shift is applied.
2. One-to-one matching by pixel-mask IoU with Hungarian assignment, reusing
   preprocessing/quality_control/compare_detection_pipelines.match_rois (shift rounded to whole
   pixels, as there). Pairs with IoU >= --iou are 'matched'.
3. Stimulated source ROIs without a match get status 'projected': their source mask, shifted into
   the holo FOV, is written to stat_projected.npy (pixels outside the FOV dropped).

Holo ROIs come from suite2p stat.npy when it exists and otherwise from the cellpose
meanImg_chan2_seg.npy that stat.npy will be built from (same ROI order, see holo_io.masks_to_stat),
so the mapping can be made before extraction has finished. The ROI origin is saved with the
results; re-run once stat.npy exists (suite2p may drop ROIs, which would change the indices).

Outputs (in holo_io output_dir; figure in analysis_dir)
------------------------------------------------------
roi_match_<source>.csv, one row per source ROI:
    source_roi, source_iscell, stimulated, condition, holo_roi (-1 if none), iou, best_iou (largest
    IoU with any holo ROI, also below threshold), centroid_dist_px, holo_iscell, status
    ('matched' / 'projected' / 'unmatched'), target_x_holo, target_y_holo (the stimulation target in
    holo pixels, assuming the FOV shift was compensated when stimulating), target_to_holo_px (that
    target to the matched holo ROI centroid), target_to_holo_px_unshifted (the same if the source
    coordinates were used unchanged).
roi_match_<source>.json: shifts, thresholds, ROI origins, counts.
stat_projected.npy: list of stat dicts (ypix, xpix, lam, npix, med, source_roi) in holo pixels.
roi_match_<source>.png: overlay of both segmentations.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.registration import phase_cross_correlation

from ceph_paths import ceph_path
from preprocessing.quality_control.compare_detection_pipelines import match_rois as iou_match
import musicboxes.holography.holo_io as hio
import musicboxes.holography.holo_functions as holo


def fov_shift(holo_img, source_img):
    """(dy, dx) moving source pixels into the holo FOV, and the phase-correlation error."""
    shift, error, _ = phase_cross_correlation(holo_img, source_img, upsample_factor=10, normalization=None)
    return shift, float(error)


def quadrant_shifts(holo_img, source_img):
    """Shift estimated separately in each FOV quadrant (dict 'top-left' etc -> (dy, dx))."""
    Ly, Lx = holo_img.shape
    out = {}
    for name, (ys, xs) in {'top-left': (slice(0, Ly // 2), slice(0, Lx // 2)),
                           'top-right': (slice(0, Ly // 2), slice(Lx // 2, Lx)),
                           'bottom-left': (slice(Ly // 2, Ly), slice(0, Lx // 2)),
                           'bottom-right': (slice(Ly // 2, Ly), slice(Lx // 2, Lx))}.items():
        out[name] = fov_shift(holo_img[ys, xs], source_img[ys, xs])[0].tolist()
    return out


def shift_stat(stat, shift, shape):
    """Copies of stat entries moved by the whole-pixel (dy, dx), pixels outside the FOV dropped."""
    dy, dx = int(round(shift[0])), int(round(shift[1]))
    out = []
    for s in stat:
        y, x = np.asarray(s['ypix']) + dy, np.asarray(s['xpix']) + dx
        ok = (y >= 0) & (y < shape[0]) & (x >= 0) & (x < shape[1])
        lam = np.asarray(s['lam'], np.float32)[ok]
        out.append({'ypix': y[ok], 'xpix': x[ok], 'lam': lam, 'npix': int(ok.sum()),
                    'med': [float(np.median(y[ok])), float(np.median(x[ok]))]})
    return out


def match_sessions(mouse, holo_tag, source_tag, project_root=hio.PROJECT_ROOT, holo_suffix='SR',
                   iou_threshold=0.3):
    """Compute the mapping. Returns (table, info, stat_projected)."""
    paths = hio.session_paths(mouse, holo_tag, project_root, holo_suffix)
    src_plane = hio.source_plane(mouse, source_tag, project_root)
    holo_rois, src_rois = hio.load_rois(paths['plane']), hio.load_rois(src_plane)
    holo_img, holo_img2 = hio.load_mean_images(paths['plane'])
    src_img, src_img2 = hio.load_mean_images(src_plane)
    shape = holo_img.shape
    trials, conditions = hio.load_stim_protocol(paths['raw_funcimg'])

    shift, err = fov_shift(holo_img, src_img)
    shift2 = fov_shift(holo_img2, src_img2)[0] if holo_img2 is not None and src_img2 is not None else None
    quads = quadrant_shifts(holo_img, src_img)
    print(f'FOV shift source -> holo (dy, dx): chan1 {np.round(shift, 2)} (error {err:.3f})'
          + (f', chan2 {np.round(shift2, 2)}' if shift2 is not None else ''))
    print('  per quadrant: ' + ', '.join(f'{k} {np.round(v, 1)}' for k, v in quads.items()))

    pairs, iou = iou_match(holo_rois['stat'], src_rois['stat'], shape, shift, iou_threshold)
    n_src = len(src_rois['stat'])
    stim = conditions[~conditions.is_blank].set_index('source_roi')

    t = pd.DataFrame({'source_roi': np.arange(n_src), 'source_iscell': src_rois['iscell']})
    t['stimulated'] = t.source_roi.isin(stim.index)
    t['condition'] = t.source_roi.map(stim['condition'])
    by_src = pairs.set_index('roi_b')
    t['holo_roi'] = t.source_roi.map(by_src['roi_a']).fillna(-1).astype(int)
    t['iou'] = t.source_roi.map(by_src['iou']).fillna(0.0)
    t['best_iou'] = iou.max(axis=0) if len(holo_rois['stat']) else 0.0
    t['centroid_dist_px'] = t.source_roi.map(by_src['centroid_dist'])
    t['holo_iscell'] = [bool(holo_rois['iscell'][r]) if r >= 0 else False for r in t.holo_roi]
    t['status'] = np.where(t.holo_roi >= 0, 'matched', np.where(t.stimulated, 'projected', 'unmatched'))

    # stimulation targets (source px) moved into the holo FOV, and their distance to the matched holo ROI
    t['target_y_holo'] = t.source_roi.map(stim['target_y']) + shift[0]
    t['target_x_holo'] = t.source_roi.map(stim['target_x']) + shift[1]
    holo_xy = hio.roi_centroids(holo_rois['stat'])
    t['target_to_holo_px'] = [np.hypot(*(holo_xy[r] - (tx, ty))) if r >= 0 and tr else np.nan
                              for r, tx, ty, tr in zip(t.holo_roi, t.target_x_holo, t.target_y_holo, t.stimulated)]
    # The same distance if the spirals went to the source coordinates unchanged, i.e. if the FOV
    # shift between the sessions was not compensated at stimulation time. Which of the two applies
    # is not recorded in the Aimbot files; stimulation responses vs these distances can tell.
    t['target_to_holo_px_unshifted'] = [
        np.hypot(*(holo_xy[r] - (tx, ty))) if r >= 0 and tr else np.nan
        for r, tx, ty, tr in zip(t.holo_roi, t.source_roi.map(stim['target_x']), t.source_roi.map(stim['target_y']),
                                 t.stimulated)]
    src_xy = hio.roi_centroids(src_rois['stat'])
    target_to_source = np.hypot(src_xy[stim.index, 0] - stim['target_x'], src_xy[stim.index, 1] - stim['target_y'])

    proj_ids = t.source_roi[t.status == 'projected'].tolist()
    stat_projected = shift_stat(src_rois['stat'][proj_ids], shift, shape)
    for s, r in zip(stat_projected, proj_ids):
        s['source_roi'] = int(r)

    st = t[t.stimulated]
    info = {
        'mouse': str(mouse), 'holo_session': paths['raw'].name, 'holo_preprocessed': paths['preproc'].name,
        'source_session': src_plane.parents[2].name, 'holo_roi_origin': holo_rois['origin'],
        'source_roi_origin': src_rois['origin'], 'iou_threshold': iou_threshold,
        'shift_yx': shift.tolist(), 'shift_error': err,
        'shift_yx_chan2': None if shift2 is None else shift2.tolist(), 'quadrant_shifts_yx': quads,
        'aimbot_experiment': conditions.attrs['experiment'],
        'n_holo_rois': len(holo_rois['stat']), 'n_holo_cells': int(holo_rois['iscell'].sum()),
        'n_source_rois': n_src, 'n_source_cells': int(src_rois['iscell'].sum()),
        'n_matched': int((t.status == 'matched').sum()),
        'n_matched_source_cells': int(((t.status == 'matched') & t.source_iscell).sum()),
        'n_stimulated': int(t.stimulated.sum()),
        'n_stimulated_matched': int((st.status == 'matched').sum()),
        'n_stimulated_projected': int((st.status == 'projected').sum()),
        'stimulated_matched_by_iou': {f'>={th}': int((st.iou >= th).sum()) for th in (0.3, 0.5, 0.7)},
        'stimulated_median_iou': float(st.iou[st.status == 'matched'].median()),
        'target_to_source_centroid_px_max': float(target_to_source.max()),
        'target_to_holo_centroid_px_median': float(st.target_to_holo_px.median()),
        'target_to_holo_centroid_px_max': float(st.target_to_holo_px.max()),
        'target_unshifted_to_holo_centroid_px_median': float(st.target_to_holo_px_unshifted.median()),
    }
    return t, info, stat_projected


def plot_match(table, info, mouse, holo_tag, source_tag, project_root=hio.PROJECT_ROOT, holo_suffix='SR',
               save_path=None):
    """Holo mean image with holo ROIs, and the shifted source masks coloured by status."""
    paths = hio.session_paths(mouse, holo_tag, project_root, holo_suffix)
    holo_rois = hio.load_rois(paths['plane'])
    src_rois = hio.load_rois(hio.source_plane(mouse, source_tag, project_root))
    img = hio.load_mean_images(paths['plane'])[0]
    shape = img.shape
    src_shifted = shift_stat(src_rois['stat'], info['shift_yx'], shape)
    stim = table[table.stimulated]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    ax = axes[0]
    ax.imshow(img, cmap='gray', clim=holo.display_range(img))
    ax.imshow(holo.outline_overlay(holo_rois['stat'], shape, holo.ROLE_COLOURS['other']))
    ax.imshow(holo.outline_overlay(src_shifted, shape, holo.ROLE_COLOURS['projected'], alpha=0.8))
    ax.set_title(f'holo ROIs ({len(holo_rois["stat"])}, yellow) and shifted {source_tag} ROIs '
                 f'({len(src_shifted)}, blue)', fontsize=10)

    ax = axes[1]
    ax.imshow(img, cmap='gray', clim=holo.display_range(img))
    matched = [src_shifted[r] for r in stim.source_roi[stim.status == 'matched']]
    projected = [src_shifted[r] for r in stim.source_roi[stim.status == 'projected']]
    ax.imshow(holo.outline_overlay(matched, shape, holo.ROLE_COLOURS['matched']))
    ax.imshow(holo.outline_overlay(projected, shape, holo.ROLE_COLOURS['projected']))
    ax.plot(stim.target_x_holo, stim.target_y_holo, '+', color=holo.ROLE_COLOURS['other'], ms=5, mew=0.8)
    for r in stim.itertuples():
        if r.status == 'projected':
            ax.text(r.target_x_holo + 4, r.target_y_holo, str(r.source_roi), color=holo.ROLE_COLOURS['projected'],
                    fontsize=6)
    ax.set_title(f'stimulated: matched {info["n_stimulated_matched"]} (pink), '
                 f'projected {info["n_stimulated_projected"]} (blue), + = target', fontsize=10)
    for a in axes[:2]:
        a.set_axis_off()

    ax = axes[2]
    ax.hist(table.best_iou[table.source_iscell & ~table.stimulated], bins=np.linspace(0, 1, 21), alpha=0.6,
            color=holo.ROLE_COLOURS['other'], label='not stimulated')
    ax.hist(stim.best_iou, bins=np.linspace(0, 1, 21), alpha=0.8, color=holo.ROLE_COLOURS['matched'],
            label='stimulated')
    ax.axvline(info['iou_threshold'], color='k', ls='--', lw=1)
    ax.set_xlabel(f'best IoU with a holo ROI (source cells, {source_tag})')
    ax.set_ylabel('cells')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'sub-{mouse} {holo_tag} vs {source_tag}: shift (dy, dx) = '
                 f'{np.round(info["shift_yx"], 1).tolist()} px; holo ROIs from {info["holo_roi_origin"]}', fontsize=11)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def run(mouse, holo_tag, source_tag, project_root=hio.PROJECT_ROOT, holo_suffix='SR', iou_threshold=0.3,
        show=False):
    table, info, stat_projected = match_sessions(mouse, holo_tag, source_tag, project_root, holo_suffix,
                                                 iou_threshold)
    paths = hio.session_paths(mouse, holo_tag, project_root, holo_suffix)
    out, fig_dir = paths['output'], paths['analysis']
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f'roi_match_{source_tag}.csv', index=False)
    (out / f'roi_match_{source_tag}.json').write_text(json.dumps(info, indent=1, default=float))
    np.save(out / 'stat_projected.npy', np.array(stat_projected, dtype=object), allow_pickle=True)
    fig = plot_match(table, info, mouse, holo_tag, source_tag, project_root, holo_suffix,
                     save_path=fig_dir / f'roi_match_{source_tag}.png')
    if show:
        plt.show()
    plt.close(fig)

    print(f"holo ROIs: {info['n_holo_rois']} ({info['n_holo_cells']} iscell, from {info['holo_roi_origin']}); "
          f"source ROIs: {info['n_source_rois']} ({info['n_source_cells']} iscell)")
    print(f"matched (IoU >= {iou_threshold}): {info['n_matched']} source ROIs "
          f"({info['n_matched_source_cells']} of them source cells)")
    print(f"stimulated: {info['n_stimulated']}, matched {info['n_stimulated_matched']} "
          f"(by IoU: {info['stimulated_matched_by_iou']}, median {info['stimulated_median_iou']:.2f}), "
          f"projected {info['n_stimulated_projected']}: "
          f"{table.source_roi[table.status == 'projected'].tolist()}")
    print(f"target -> source centroid max {info['target_to_source_centroid_px_max']:.2f} px; "
          f"target (shifted) -> matched holo centroid median {info['target_to_holo_centroid_px_median']:.2f}, "
          f"max {info['target_to_holo_centroid_px_max']:.2f} px; unshifted target -> matched holo centroid "
          f"median {info['target_unshifted_to_holo_centroid_px_median']:.2f} px")
    print(f'Saved to {out} and {fig_dir}')
    return table, info


def _cli():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('mouse')
    p.add_argument('holo_tag', help="e.g. 'holo-001'")
    p.add_argument('source_tag', help="session whose ROIs were targeted, e.g. 'seq3-007'")
    p.add_argument('--project-root', default=str(hio.PROJECT_ROOT))
    p.add_argument('--holo-suffix', default='SR', help="preprocessed folder suffix (default 'SR')")
    p.add_argument('--iou', type=float, default=0.3)
    p.add_argument('--show', action='store_true')
    a = p.parse_args()
    run(a.mouse, a.holo_tag, a.source_tag, ceph_path(a.project_root), a.holo_suffix, a.iou, a.show)


if __name__ == '__main__':
    _cli()

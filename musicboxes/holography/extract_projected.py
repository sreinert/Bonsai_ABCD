"""Extract F/Fneu/dF/F for stimulated source cells that have no holo ROI (projected masks).

match_rois.py writes stat_projected.npy: the source-session masks of stimulated cells without a
holo match, shifted into the holo FOV (14 of 103 for sub-001 holo-001). This script extracts their
traces from the registered holo movie (suite2p plane0 data.bin) the way suite2p does, so they can be
analysed alongside the holo ROIs:

- cell and neuropil masks from suite2p.extraction.masks.create_masks on ALL ROIs together (holo
  stat.npy + projected), so each neuropil annulus excludes every cell's pixels, with the extraction
  settings saved with the holo run (inner_neuropil_radius, min_neuropil_pixels, lam_percentile,
  allow_overlap). Overlap flags are recomputed over the combined set.
- traces with suite2p.extraction.extract.extract_traces (weighted pixel sum / annulus mean).
- dF/F as in nwb_barcoding/compute_dff.py: F - 0.7 Fneu, then suite2p's dcnv.preprocess baseline.

Validation: the same is done for --n-validate holo ROIs (iscell, not overlapping any projected
mask) and their F/Fneu are compared with suite2p's own F.npy/Fneu.npy (Pearson r and median
ratio, saved in validation.json; r should be > 0.99). The holo run used suite2p 1.x and this
script uses the repo's 0.14.6, so small differences in mask handling would show up here.

data.bin is 141 GB for holo-001, so run this on the HPC (extract_projected_hpc.sh), not over SMB:

    python -m musicboxes.holography.extract_projected 001 holo-001

Writes to <output_dir>/funcimg/projected/: F_projected.npy, Fneu_projected.npy, dFF_projected.npy
(n_projected x n_frames, rows in stat_projected.npy order), source_roi.npy, validation.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ceph_paths import ceph_path
from nwb_barcoding.compute_dff import _get_baseline_params
import musicboxes.holography.holo_io as hio

NEUROPIL_COEFFICIENT = 0.7


def combined_masks(holo_stat, proj_stat, Ly, Lx, extraction):
    """suite2p cell + neuropil masks for holo ROIs followed by projected ROIs."""
    from suite2p.extraction.masks import create_masks
    stats = [dict(s) for s in holo_stat] + [dict(s) for s in proj_stat]
    count = np.zeros((Ly, Lx), int)
    for s in stats:
        count[s['ypix'], s['xpix']] += 1
    for s in stats:
        s['overlap'] = count[s['ypix'], s['xpix']] > 1
        s['lam'] = np.asarray(s['lam'], np.float32)
        s.setdefault('radius', np.sqrt(len(s['ypix']) / np.pi))
    ops = {'lam_percentile': extraction.get('lam_percentile', 50.0),
           'allow_overlap': extraction.get('allow_overlap', False),
           'neuropil_extract': True,
           'inner_neuropil_radius': extraction.get('inner_neuropil_radius', 2),
           'min_neuropil_pixels': extraction.get('min_neuropil_pixels', 350),
           'circular_neuropil': extraction.get('circular_neuropil', False)}
    return create_masks(stats, Ly, Lx, ops)


def pick_validation_rois(holo_stat, iscell, proj_stat, n, Ly, Lx, seed=0):
    """n iscell holo ROIs whose pixels do not touch any projected mask."""
    proj = np.zeros((Ly, Lx), bool)
    for s in proj_stat:
        proj[s['ypix'], s['xpix']] = True
    ok = [i for i, s in enumerate(holo_stat) if iscell[i] and not proj[s['ypix'], s['xpix']].any()]
    return sorted(np.random.default_rng(seed).choice(ok, size=min(n, len(ok)), replace=False).tolist())


def extract(mouse, holo_tag, project_root=hio.PROJECT_ROOT, holo_suffix='SR', n_validate=5, force=False):
    from suite2p.extraction import dcnv
    from suite2p.extraction.extract import extract_traces

    paths = hio.session_paths(mouse, holo_tag, project_root, holo_suffix)
    out = paths['output'] / 'funcimg' / 'projected'
    if (out / 'dFF_projected.npy').exists() and not force:
        print(f'Projected traces exist, skipping: {out}')
        return out
    proj_stat = list(np.load(paths['output'] / 'stat_projected.npy', allow_pickle=True))
    if not proj_stat:
        print('stat_projected.npy is empty: every stimulated cell has a holo ROI, nothing to extract')
        return None
    rois = hio.load_rois(paths['plane'])
    if rois['origin'] != 'stat.npy':
        raise FileNotFoundError(f"{paths['plane']} has no stat.npy yet; run suite2p extraction first")
    ops = hio.load_ops(paths['plane'])
    Ly, Lx, n_frames = int(ops['Ly']), int(ops['Lx']), int(ops['nframes'])
    extraction = ops.get('extraction') or ops

    cell_masks, neuropil_masks = combined_masks(rois['stat'], proj_stat, Ly, Lx, extraction)
    n_holo = len(rois['stat'])
    val = pick_validation_rois(rois['stat'], rois['iscell'], proj_stat, n_validate, Ly, Lx)
    sel = list(range(n_holo, n_holo + len(proj_stat))) + val
    print(f'{len(proj_stat)} projected ROIs + {len(val)} validation holo ROIs {val}; '
          f"neuropil pixels per projected ROI {[len(neuropil_masks[i]) for i in sel[:len(proj_stat)]]}")

    movie = np.memmap(paths['plane'] / 'data.bin', mode='r', dtype=np.int16, shape=(n_frames, Ly, Lx))
    F, Fneu = extract_traces(movie, [cell_masks[i] for i in sel], [neuropil_masks[i] for i in sel],
                             batch_size=int(extraction.get('batch_size', 500)))

    k = len(proj_stat)
    F_ref = np.load(paths['plane'] / 'F.npy', mmap_mode='r')
    Fneu_ref = np.load(paths['plane'] / 'Fneu.npy', mmap_mode='r')
    validation = []
    for j, roi in enumerate(val):
        f, fr = F[k + j], np.asarray(F_ref[roi])
        nf, nfr = Fneu[k + j], np.asarray(Fneu_ref[roi])
        validation.append({'holo_roi': roi, 'r_F': float(np.corrcoef(f, fr)[0, 1]),
                           'ratio_F': float(np.median(f / fr)), 'r_Fneu': float(np.corrcoef(nf, nfr)[0, 1]),
                           'ratio_Fneu': float(np.median(nf / nfr))})
        print(f"  validation ROI {roi}: r(F) {validation[-1]['r_F']:.4f} ratio {validation[-1]['ratio_F']:.3f}; "
              f"r(Fneu) {validation[-1]['r_Fneu']:.4f} ratio {validation[-1]['ratio_Fneu']:.3f}")
    if any(v['r_F'] < 0.99 for v in validation):
        print('[FLAG] validation r(F) < 0.99: projected traces are not extracted like suite2p F.npy')

    F, Fneu = F[:k], Fneu[:k]
    params = _get_baseline_params(ops)
    dff = dcnv.preprocess(F - NEUROPIL_COEFFICIENT * Fneu, *params[:3], ops['fs'], params[3])

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'F_projected.npy', F)
    np.save(out / 'Fneu_projected.npy', Fneu)
    np.save(out / 'dFF_projected.npy', dff.astype(np.float32))
    np.save(out / 'source_roi.npy', np.array([s['source_roi'] for s in proj_stat]))
    (out / 'validation.json').write_text(json.dumps(validation, indent=1))
    print(f'Saved projected traces to {out}')
    return out


def _cli():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('mouse')
    p.add_argument('holo_tag')
    p.add_argument('--project-root', default=str(hio.PROJECT_ROOT))
    p.add_argument('--holo-suffix', default='SR')
    p.add_argument('--n-validate', type=int, default=5)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    extract(a.mouse, a.holo_tag, ceph_path(a.project_root), a.holo_suffix, a.n_validate, a.force)


if __name__ == '__main__':
    _cli()

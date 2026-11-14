"""Paths and loaders shared by the holography scripts in musicboxes/holography/.

Written for the holography cohort (SaReSuKuMaRa_20260911_holography/cohort1), first used on
sub-001 ses-holo-001_date-20261111: single-cell photostimulation of cells that were segmented
in an earlier VR session (the "source" session, seq3-007 for sub-001), while imaging the same
FOV and logging behaviour on the WarpedVisualStim DAQ.

Layout of a holo session (inspected 2026-11-14 for sub-001 holo-001)
--------------------------------------------------------------------
rawdata/sub-XXX/ses-holo-NNN_date-YYYYMMDD/
    funcimg/<stem>.tif                 ScanImage movie (2 channels, 512x512, ~44.6 Hz)
    funcimg/<stem>.h5                  vDAQ recorder, 5 kHz: FrameTTL, SLMFeedbackTTL (one 100 ms
                                       pulse per trial), SatsumaGateTTL (laser gate, 20 x 2 ms
                                       pulses at 40 Hz per trial), SyncTTL (barcode),
                                       TriggerFromBehavDAQ (unused, constant)
    funcimg/photostim_loader_*.csv     trial, phase_mask ('ensembleN' or 'blank'), isi_s: row i
                                       is the i-th SatsumaGate train (ITIs agree with r = 0.9999)
    funcimg/phase_mask_trials_*.csv    the planned trial order (same phase_mask column)
    funcimg/experiment_*.h5            Aimbot protocol: /targets/<cond> (2, n) as (y, x) - 256 in
                                       SOURCE-session pixels, /power_Vs/<cond>, SLM offsets
    funcimg/others/                    test acquisitions, ignored
    behav/<stem>_ai0/ai3/ai4/ai13/ctr0.npy, _settings.json   WarpedVisualStim DAQ logger (5 kHz)
preprocessed/sub-XXX/ses-holo-NNN_date-YYYYMMDD-SR/funcimg/suite2p/plane0/   suite2p on the holo
    movie, ROIs from cellpose on the chan2 mean image (meanImg_chan2_seg.npy). A '-SK' sibling
    holds a colleague's independent processing and is never used here.

Ensemble ids: 'ensembleN' targets ROI N of the source session's suite2p stat.npy (its 'med',
minus 256), which is also cell_id N of the source session's gp_cell_table.csv. They are NOT
holo ROI indices; musicboxes/holography/match_rois.py makes that mapping.

Outputs never go into rawdata/: data products go to
BehaviourAligned_robust/sub-XXX/<raw session folder name>/ (output_dir) and figures to
analysis/sub-XXX/<session label>/ (analysis_dir), as for the VR sessions.
"""

import glob
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ceph_paths import ceph_path

PROJECT_ROOT = ceph_path('/Volumes/mrsic_flogel/public/projects/SaReSuKuMaRa_20260911_holography/cohort1')
FRAME_RATE = 45  # nominal; the measured FrameTTL rate (~44.6 Hz) is used wherever times matter
FOV_CENTRE = 256  # Aimbot target coordinates are pixel - 256


def _mouse_dir(tree_root, mouse):
    mouse = str(mouse)
    return Path(tree_root) / (mouse if mouse.startswith('sub-') else f'sub-{mouse}')


def find_session_dir(mouse, tag, tree_root, suffix=None):
    """The one session folder `ses-<tag>[_...]` under tree_root/sub-<mouse>, optionally ending in
    '-<suffix>' (e.g. 'SR'). Unlike parse_nwb_functions.find_base_path this matches the tag as a
    whole name component ('holo-001' does not match 'holo-0010') and raises unless exactly one
    folder matches (holo-001 has both -SK and -SR folders in preprocessed/)."""
    pattern = re.compile(rf'^ses-{re.escape(tag)}($|_)')
    hits = [d for d in sorted(_mouse_dir(tree_root, mouse).iterdir())
            if d.is_dir() and pattern.match(d.name) and (suffix is None or d.name.endswith(f'-{suffix}'))]
    if len(hits) != 1:
        raise FileNotFoundError(f'Expected one folder for sub-{mouse} ses-{tag} (suffix={suffix}) in '
                                f'{_mouse_dir(tree_root, mouse)}, found {[d.name for d in hits]}')
    return hits[0]


def session_label(folder_name):
    """'ses-holo-001_date-20261111-SR' -> 'holo-001' (the analysis/ folder name)."""
    m = re.match(r'ses-(.+?)(_date-|$)', Path(folder_name).name)
    return m.group(1) if m else Path(folder_name).name


def session_paths(mouse, holo_tag, project_root=PROJECT_ROOT, holo_suffix='SR'):
    """Dict of the folders one holo session uses (see the module docstring)."""
    project_root = Path(ceph_path(project_root))
    raw = find_session_dir(mouse, holo_tag, project_root / 'rawdata')
    preproc = find_session_dir(mouse, holo_tag, project_root / 'preprocessed', suffix=holo_suffix)
    mouse_dir = _mouse_dir('', mouse).name
    return {
        'project_root': project_root,
        'raw': raw,
        'raw_funcimg': raw / 'funcimg',
        'raw_behav': raw / 'behav',
        'preproc': preproc,
        'funcimg': preproc / 'funcimg',
        'plane': preproc / 'funcimg' / 'suite2p' / 'plane0',
        'dff': preproc / 'funcimg' / 'dff' / 'plane0' / 'dFF.npy',
        'output': project_root / 'BehaviourAligned_robust' / mouse_dir / raw.name,
        'analysis': project_root / 'analysis' / mouse_dir / session_label(raw.name),
    }


def source_plane(mouse, source_tag, project_root=PROJECT_ROOT):
    """suite2p plane0 folder of the source (targeting) session in preprocessed/."""
    d = find_session_dir(mouse, source_tag, Path(ceph_path(project_root)) / 'preprocessed')
    return d / 'funcimg' / 'suite2p' / 'plane0'


def gp_table_path(mouse, source_tag, project_root=PROJECT_ROOT):
    """analysis/sub-<mouse>/<source label>/gp_cell_table.csv (musicboxes/goalprogress/gp_cell_table.py)."""
    return Path(ceph_path(project_root)) / 'analysis' / _mouse_dir('', mouse).name / source_tag / 'gp_cell_table.csv'


# ---------------------------------------------------------------- ROIs and images

def masks_to_stat(masks):
    """Minimal suite2p-like stat list from a cellpose label image. ROI i is the i-th label in sorted
    order, which is how preprocessing/suite2p/Convert_seg_to_stat.py builds stat.npy (checked on
    sub-001 seq3-007: stat ROI i lies on cellpose label i+1 for 183/183 ROIs)."""
    stat = []
    for u in np.unique(masks)[1:]:
        ypix, xpix = np.nonzero(masks == u)
        stat.append({'ypix': ypix, 'xpix': xpix, 'npix': len(ypix), 'lam': np.ones(len(ypix), np.float32),
                     'med': [float(np.median(ypix)), float(np.median(xpix))]})
    return np.array(stat, dtype=object)


def load_rois(plane_dir):
    """ROIs of one suite2p plane: dict with stat, iscell (bool), and origin ('stat.npy' or
    'cellpose_seg'). Falls back to *_seg.npy (all ROIs counted as cells) while suite2p extraction
    has not written stat.npy yet."""
    plane_dir = Path(plane_dir)
    if (plane_dir / 'stat.npy').exists():
        stat = np.load(plane_dir / 'stat.npy', allow_pickle=True)
        iscell = np.load(plane_dir / 'iscell.npy')[:, 0].astype(bool)
        return {'stat': stat, 'iscell': iscell, 'origin': 'stat.npy'}
    segs = sorted(plane_dir.glob('*_seg.npy'))
    if len(segs) != 1:
        raise FileNotFoundError(f'No stat.npy and not exactly one *_seg.npy in {plane_dir}: {[s.name for s in segs]}')
    stat = masks_to_stat(np.load(segs[0], allow_pickle=True).item()['masks'])
    return {'stat': stat, 'iscell': np.ones(len(stat), bool), 'origin': f'cellpose_seg ({segs[0].name})'}


def roi_centroids(stat):
    """(n, 2) unweighted mask centroids as (x, y), the convention of gp_cell_table.csv."""
    return np.array([[np.mean(s['xpix']), np.mean(s['ypix'])] for s in stat])


def load_mean_images(plane_dir):
    """(meanImg, meanImg_chan2) of a suite2p plane (chan2 None if not recorded)."""
    ops = np.load(Path(plane_dir) / 'ops.npy', allow_pickle=True).item()
    return ops['meanImg'], ops.get('meanImg_chan2')


def load_ops(plane_dir):
    return np.load(Path(plane_dir) / 'ops.npy', allow_pickle=True).item()


# ---------------------------------------------------------------- stimulation protocol

def _one(pattern):
    hits = sorted(glob.glob(str(pattern)))
    if len(hits) != 1:
        raise FileNotFoundError(f'Expected one file matching {pattern}, found {hits}')
    return Path(hits[0])


def ensemble_to_source_roi(name):
    """'ensemble22' -> 22, 'blank' -> -1."""
    m = re.fullmatch(r'ensemble(\d+)', str(name))
    return int(m.group(1)) if m else -1


def load_stim_protocol(raw_funcimg):
    """Trial list and per-condition targets from the Aimbot files in raw funcimg/.

    Returns (trials, conditions):
      trials     : DataFrame trial, condition, source_roi (-1 = blank), isi_s, from photostim_loader_*.csv
      conditions : DataFrame condition, source_roi, is_blank, target_y, target_x (intended target in
                   SOURCE-session imaging pixels, NaN for blank), slm_target_y, slm_target_x (the
                   coordinates sent to the SLM), power_V, one row per phase mask in experiment_*.h5
    plus a dict of the experiment's scalar settings (SLM offsets etc) in conditions.attrs['experiment'].

    The experiment h5 targets already include Aimbot's SLM calibration offset (offset_x/offset_y, from
    the phase_masks_calibration run): for sub-001 holo-001 they are source med + (0, -5) px exactly,
    with offset_x = -5, while ensemble_*.npy holds the uncorrected med. The offset corrects for the
    SLM/imaging misregistration, so the imaging-space target is the h5 target minus the offset.
    """
    raw_funcimg = Path(raw_funcimg)
    trials = pd.read_csv(_one(raw_funcimg / 'photostim_loader_*.csv')).rename(columns={'phase_mask': 'condition'})
    trials['source_roi'] = trials['condition'].map(ensemble_to_source_roi)

    rows, settings = [], {}
    with h5py.File(_one(raw_funcimg / 'experiment_*.h5'), 'r') as h:
        for k in ['offset_x', 'offset_y', 'offset_z', 'z_base_um', 'photostim_mode', 'power_uniformity_applied']:
            if k in h:
                v = h[k][()]
                settings[k] = v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, 'item') else v)
        off_y, off_x = settings.get('offset_y', 0.0), settings.get('offset_x', 0.0)
        for cond in h['targets']:
            t = np.asarray(h['targets'][cond][()], float)
            is_blank = cond == 'blank'
            if not is_blank and t.shape[1] != 1:
                raise ValueError(f'{cond} has {t.shape[1]} targets; this pipeline assumes single-cell conditions')
            slm_y = np.nan if is_blank else t[0, 0] + FOV_CENTRE
            slm_x = np.nan if is_blank else t[1, 0] + FOV_CENTRE
            rows.append({'condition': cond, 'source_roi': ensemble_to_source_roi(cond), 'is_blank': is_blank,
                         'target_y': slm_y - off_y, 'target_x': slm_x - off_x,
                         'slm_target_y': slm_y, 'slm_target_x': slm_x,
                         'power_V': float(h['power_Vs'][cond][()])})
    conditions = pd.DataFrame(rows).sort_values(['is_blank', 'source_roi']).reset_index(drop=True)
    conditions.attrs['experiment'] = settings

    unknown = set(trials['condition']) - set(conditions['condition'])
    if unknown:
        raise ValueError(f'Trials use conditions missing from experiment h5: {sorted(unknown)}')
    return trials, conditions


def vdaq_h5_path(raw_funcimg):
    """The ScanImage vDAQ recorder file next to the (single) tif in raw funcimg/."""
    return _one(Path(raw_funcimg) / '*.tif').with_suffix('.h5')


def behav_channel_path(raw_behav, channel):
    """WarpedVisualStim DAQ-logger channel file <stem>_<channel>.npy."""
    return _one(Path(raw_behav) / f'*_{channel}.npy')

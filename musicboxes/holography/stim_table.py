"""Stimulation tables for indexing the holo traces: which cell was stimulated in which frames.

Combines photostim_trials.csv (nwb_holography.py: trial timing on the vDAQ clock and in frames),
roi_match_<source>.csv (match_rois.py: holo ROI of each targeted source cell) and the source
session's gp_cell_table.csv (goal-progress tuning). Run from the repo root:

    python -m musicboxes.holography.stim_table 001 holo-001 seq3-007

Outputs (holo_io output_dir)
---------------------------
stim_trials.csv, one row per trial:
    trial, condition, rep (k-th trial of this condition), is_blank, source_roi, holo_roi, status
    ('matched' / 'projected' / 'blank'), trace_source + trace_row (where the stimulated cell's
    trace lives: 'suite2p' -> row of F.npy/dFF.npy = holo_roi; 'projected' -> row of
    funcimg/projected/*_projected.npy), onset_time, offset_time, slm_time (s, vDAQ clock),
    onset_frame, offset_frame, n_stim_frames (frames overlapping the spiral train, inclusive),
    isi_s, target_x_holo, target_y_holo, gp_tuned, preferred_phase.
stim_matrix.npz, the "stimulated cells x imaging frames" table:
    stim_frames   bool (n_conditions, n_frames): True in the frames overlapping that condition's trains
    conditions, source_roi, holo_roi, status, trace_source, trace_row: per row of stim_frames
    onset_frames, offset_frames: int (n_conditions, n_reps)
    Blank is the last row (its frames are the no-target control).
stim_cells.csv: one row per condition with the same ids plus onset_frames and stim_frames as text
    ('a-b;c-d;...', inclusive), for reading by eye or in other tools.

Use F[:, stim_frames[i]] or the onset_frames for event-aligned windows; holo_functions.psth does the
latter.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ceph_paths import ceph_path
import musicboxes.holography.holo_io as hio


def build_stim_table(mouse, holo_tag, source_tag, project_root=hio.PROJECT_ROOT, holo_suffix='SR'):
    paths = hio.session_paths(mouse, holo_tag, project_root, holo_suffix)
    out = paths['output']
    trials = pd.read_csv(out / 'photostim_trials.csv')
    match = pd.read_csv(out / f'roi_match_{source_tag}.csv').set_index('source_roi')
    n_frames = len(np.load(out / 'holo_alignment.npz')['frame_times'])

    proj_path = out / 'funcimg' / 'projected' / 'source_roi.npy'
    proj_rows = {}
    if proj_path.exists():
        proj_rows = {int(r): i for i, r in enumerate(np.load(proj_path))}
    else:
        print('No projected traces yet (extract_projected.py): projected cells get trace_row -1')

    t = trials.copy()
    t['rep'] = t.groupby('condition').cumcount()
    t['holo_roi'] = t.source_roi.map(match['holo_roi']).fillna(-1).astype(int)
    t['status'] = np.where(t.is_blank, 'blank', t.source_roi.map(match['status']))
    t['trace_source'] = t.status.map({'matched': 'suite2p', 'projected': 'projected', 'blank': ''})
    t['trace_row'] = [h if s == 'matched' else proj_rows.get(r, -1) if s == 'projected' else -1
                      for h, s, r in zip(t.holo_roi, t.status, t.source_roi)]
    t['n_stim_frames'] = t.offset_frame - t.onset_frame + 1
    t['target_x_holo'] = t.source_roi.map(match['target_x_holo'])
    t['target_y_holo'] = t.source_roi.map(match['target_y_holo'])
    gp_path = hio.gp_table_path(mouse, source_tag, project_root)
    if gp_path.exists():
        gp = pd.read_csv(gp_path).set_index('cell_id')
        t['gp_tuned'] = t.source_roi.map(gp['gp_tuned'])
        t['preferred_phase'] = t.source_roi.map(gp['preferred_phase'])
    else:
        print(f'No {gp_path}; GP columns left out')

    cols = ['trial', 'condition', 'rep', 'is_blank', 'source_roi', 'holo_roi', 'status', 'trace_source',
            'trace_row', 'onset_time', 'offset_time', 'slm_time', 'onset_frame', 'offset_frame', 'n_stim_frames',
            'isi_s', 'target_x_holo', 'target_y_holo', 'gp_tuned', 'preferred_phase']
    t = t[[c for c in cols if c in t]]

    # per condition, cells first (by source ROI), blank last
    cond = (t.drop_duplicates('condition').sort_values(['is_blank', 'source_roi'])
            [['condition', 'source_roi', 'holo_roi', 'status', 'trace_source', 'trace_row']].reset_index(drop=True))
    n_rep = t.groupby('condition').size().max()
    stim = np.zeros((len(cond), n_frames), bool)
    on = np.full((len(cond), n_rep), -1)
    off = np.full((len(cond), n_rep), -1)
    texts = []
    for i, c in enumerate(cond.condition):
        tc = t[t.condition == c].sort_values('rep')
        for k, (a, b) in enumerate(zip(tc.onset_frame, tc.offset_frame)):
            stim[i, a:b + 1] = True
            on[i, k], off[i, k] = a, b
        texts.append((' '.join(map(str, tc.onset_frame)),
                      ';'.join(f'{a}-{b}' for a, b in zip(tc.onset_frame, tc.offset_frame))))
    cells = cond.copy()
    cells['n_trials'] = (on >= 0).sum(1)
    cells['onset_frames'], cells['stim_frames'] = zip(*texts)
    if 'gp_tuned' in t:
        g = t.drop_duplicates('condition').set_index('condition')
        cells['gp_tuned'] = cells.condition.map(g['gp_tuned'])
        cells['preferred_phase'] = cells.condition.map(g['preferred_phase'])

    t.to_csv(out / 'stim_trials.csv', index=False)
    cells.to_csv(out / 'stim_cells.csv', index=False)
    np.savez_compressed(out / 'stim_matrix.npz', stim_frames=stim, onset_frames=on, offset_frames=off,
                        **{k: cond[k].values.astype(str) if cond[k].dtype == object else cond[k].values
                           for k in cond.columns})
    report(t, stim, n_frames)
    print(f'Saved stim_trials.csv, stim_cells.csv, stim_matrix.npz to {out}')
    return t, cells


def report(t, stim, n_frames):
    per = t.groupby('condition').size()
    cells = t[~t.is_blank]
    print(f"{t.condition.nunique()} conditions ({cells.source_roi.nunique()} cells + "
          f"{int(t.is_blank.any())} blank), trials per condition {per.min()}-{per.max()}")
    print('cells by status: ' + ', '.join(f'{k} {v}' for k, v in
                                          cells.drop_duplicates('condition').status.value_counts().items()))
    if (cells.trace_row < 0).any():
        print(f'  {cells[cells.trace_row < 0].condition.nunique()} cells have no trace yet (projected, not extracted)')
    print(f'stim frames per trial {t.n_stim_frames.min()}-{t.n_stim_frames.max()}; '
          f'{stim.any(0).sum()} of {n_frames} frames ({100 * stim.any(0).mean():.1f}%) are stimulation frames; '
          f'max conditions in one frame {stim.sum(0).max()}')
    if 'gp_tuned' in t:
        c = cells.drop_duplicates('condition')
        print(f'stimulated cells GP-tuned in the source session: {int(c.gp_tuned.sum())}/{len(c)}')


def load_stim_table(mouse, holo_tag, project_root=hio.PROJECT_ROOT, holo_suffix='SR'):
    """(stim_trials DataFrame, stim_matrix dict) as written by build_stim_table."""
    out = hio.session_paths(mouse, holo_tag, project_root, holo_suffix)['output']
    return pd.read_csv(out / 'stim_trials.csv'), dict(np.load(out / 'stim_matrix.npz', allow_pickle=True))


def stim_frames_for(trials, source_roi=None, holo_roi=None, condition=None):
    """List of frame-index arrays, one per trial, of the stimulation frames of one cell/condition."""
    if condition is None:
        sel = trials.source_roi == source_roi if source_roi is not None else trials.holo_roi == holo_roi
    else:
        sel = trials.condition == condition
    return [np.arange(a, b + 1) for a, b in zip(trials.onset_frame[sel], trials.offset_frame[sel])]


def _cli():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('mouse')
    p.add_argument('holo_tag')
    p.add_argument('source_tag')
    p.add_argument('--project-root', default=str(hio.PROJECT_ROOT))
    p.add_argument('--holo-suffix', default='SR')
    a = p.parse_args()
    build_stim_table(a.mouse, a.holo_tag, a.source_tag, ceph_path(a.project_root), a.holo_suffix)


if __name__ == '__main__':
    _cli()

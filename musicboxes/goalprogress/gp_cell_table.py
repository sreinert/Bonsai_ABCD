"""Per-cell table of mask centroid and goal-progress (GP) tuning for one holography session.

One row per accepted cell (iscell.npy column 0 == 1), saved as
<project_root>/analysis/sub-<mouse>/<session>/gp_cell_table.csv. Run from the repo root:

    python -m musicboxes.goalprogress.gp_cell_table 001 seq3-007

Columns
-------
  cell_id         : suite2p ROI index, i.e. the row of this cell in F.npy / iscell.npy and its
                    entry in stat.npy. NOT the row of the accepted-only dF matrix.
  x, y            : unweighted mean of the mask's xpix / ypix in stat.npy (registered FOV pixels).
                    Not suite2p's 'med', which is a median and is stored as (y, x).
  gp_tuned        : significant GP tuning after Bonferroni correction (see below).
  preferred_phase : preferred GP phase, 0-1 with 0 = reward; NaN if not gp_tuned. With the default
                    --method vm the centre of the von Mises fit (gp_circular_fit), with --method
                    binned the peak of the mean 30-bin GP curve.
  p               : uncorrected shuffle p-value (all cells).
  split_r         : correlation between the mean GP curves of even and odd segments (all cells).

GP tuning comes from gp_learning_helpers.gp_tuning (method 'vm' by default, 'binned' with --method
binned) on the reward -> reward clock (--clock correct: only correct transitions) with circular-shift
shuffles within segments (seed 0), as in GP_learning_holography.ipynb, but stricter
than the notebook's uncorrected p < 0.05 and therefore selecting fewer cells: gp_tuned is
p < 0.05 / n_cells, over the accepted cells. That needs 10000 shuffles rather than the notebook's
500, because the smallest possible p is 1 / (1 + n_shuffles) and 1/501 can never pass the
threshold (0.05 / 177 for sub-001 seq3-007).

The tuning runs on the NWB's accepted ROIs while the IDs and masks come from the suite2p folder,
so the script stops if the two sets of accepted ROIs differ (e.g. iscell.npy curated after the NWB
was built): otherwise tuning results would silently land on the wrong cells.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import parse_nwb_functions
from ceph_paths import ceph_path
import musicboxes.goalprogress.gp_learning_helpers as h

GP_BINS = 30
N_SHUFFLES = 10000
ALPHA = 0.05


def suite2p_plane(mouse, session, project_root):
    """Session folder name and suite2p plane0 path in the preprocessed tree."""
    session_path = parse_nwb_functions.find_base_path(mouse, session, f'{project_root}/preprocessed')
    if session_path is None:
        raise FileNotFoundError(f'No preprocessed folder for sub-{mouse} matching {session!r}')
    return Path(session_path).name, Path(session_path) / 'funcimg' / 'suite2p' / 'plane0'


def load_suite2p(mouse, session, project_root):
    """accepted ROI indices and (n_rois, 2) mask centroids (x, y) from the suite2p folder, checking
    that F, iscell and stat describe the same ROIs."""
    folder_name, plane = suite2p_plane(mouse, session, project_root)
    iscell = np.load(plane / 'iscell.npy')
    stat = np.load(plane / 'stat.npy', allow_pickle=True)
    n_rois_F = np.load(plane / 'F.npy', mmap_mode='r').shape[0]
    if not len(iscell) == len(stat) == n_rois_F:
        raise ValueError(f'ROI counts differ: iscell {len(iscell)}, stat {len(stat)}, F {n_rois_F}')
    centroids = np.array([[np.mean(s['xpix']), np.mean(s['ypix'])] for s in stat])
    return np.where(iscell[:, 0] == 1)[0], centroids, folder_name


def session_label(folder_name):
    """Analysis-folder name of a session folder: ses-seq3-007_date-... -> seq3-007."""
    return re.match(r'ses-(.+?)_date-', folder_name).group(1)


def build_table(mouse, session, project_root=h.PROJECT_ROOT, method='vm', clock='reward'):
    """Returns (table, session label). `session` is any tag that identifies the folder ('seq3-007', 't9').
    method: 'vm' or 'binned' (see gp_learning_helpers.gp_tuning); clock: 'reward' or 'correct'."""
    accepted, centroids, folder_name = load_suite2p(mouse, session, project_root)
    loaded = h.load_session(mouse, session, behav_root=f'{project_root}/BehaviourAligned_robust')
    if not np.array_equal(accepted, loaded['neurons']):
        raise ValueError(f"iscell.npy accepts {len(accepted)} ROIs, the NWB {len(loaded['neurons'])}, and they are "
                         f"not the same set -- was iscell.npy curated after the NWB was built?")
    gp = h.gp_tuning(loaded, clock=clock, bins=GP_BINS, n_shuffles=N_SHUFFLES, alpha=ALPHA / len(accepted),
                     method=method)
    if gp is None:
        raise ValueError(f'sub-{mouse} {session}: fewer than 4 {clock} segments, cannot test GP tuning')
    print(f"sub-{mouse} {session} ({method}, {clock} clock): {len(gp['segments'])} segments, "
          f"{gp['significant'].sum()}/{len(accepted)} GP cells")
    table = pd.DataFrame(dict(cell_id=accepted, x=centroids[accepted, 0], y=centroids[accepted, 1],
                              gp_tuned=gp['significant'],
                              preferred_phase=np.where(gp['significant'], gp['phase'], np.nan),
                              p=gp['p'], split_r=gp['split_r']))
    return table, session_label(folder_name)


def _cli():
    parser = argparse.ArgumentParser(description='Per-cell table of mask centroid and GP tuning for one session.')
    parser.add_argument('mouse', help='e.g. "001"')
    parser.add_argument('session', help='session tag to match, e.g. "seq3-007"')
    parser.add_argument('--project-root', default=h.PROJECT_ROOT, help='defaults to the holography cohort1 folder')
    parser.add_argument('--method', default='vm', choices=['vm', 'binned'], help='GP phase / significance method')
    parser.add_argument('--clock', default='reward', choices=['reward', 'correct'],
                        help="'correct': only reward -> reward segments after which the next lick hit the next goal")
    args = parser.parse_args()
    project_root = ceph_path(args.project_root)
    table, label = build_table(args.mouse, args.session, project_root, method=args.method, clock=args.clock)
    out_dir = Path(project_root) / 'analysis' / f'sub-{args.mouse}' / label
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / 'gp_cell_table.csv', index=False)
    print(f'Saved {len(table)} cells to {out_dir / "gp_cell_table.csv"}')


if __name__ == '__main__':
    _cli()

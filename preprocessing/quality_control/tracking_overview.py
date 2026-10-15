"""Per-mouse overview plots of ROICaT cross-session cell tracking (cohort2, cohort3, GoalProgress,
holography).

Every analysis in this repo so far is per session, but each mouse has the same FOV imaged across
sessions and ROICaT (preprocessing/roicat_analysis/roicat_tracking.py, run on the HPC) has clustered
the suite2p ROIs of all sessions into cross-session cells ("UCIDs": one cluster ID = one cell). This
module inspects that tracking for one mouse/run before it is used for any multi-session analysis --
the multi-session counterpart of musicboxes/session_overview.py.

Produces (saved to <root>/analysis/sub-<mouse>/tracking[/<run tags>]):
  1. cluster_matrix.png -- cells x sessions presence matrix, sorted by number of sessions present
  2. tracking_counts.png -- per-session cell counts, pairwise s->s+1 overlap and survival from the
     first session, with outlier sessions (few cells refound) shaded
  3. tracked_cell_<id>.png -- for n_cells random cells found in >=min_presence of sessions: one
     native-frame mean-image crop + mask outline per session, plus the cell's position in ROICaT's
     registered frame across sessions (a large jump suggests misidentification)
  4. position_consistency.png -- registered-frame displacement from the cluster median, all cells
  5. tracking_summary.csv (per session) and tracked_cells_sampled.csv (the cells in 3, with their
     suite2p ROI index per session)
  6. cell_matrix.csv / cell_matrix.npz -- every cell (rows of the presence matrix) x session: suite2p
     ROI index (-1 = absent) and registered-frame displacement from the cell's median position (um,
     NaN = absent or single-session cell). Meant for selecting cells for multi-session analyses
     together with the ROICaT matching: load_cell_matrix() + select_tracked_cells().

Design decisions:
- Cohort-agnostic loading. Each ROICaT run is self-describing: <name>.tracking.params_used.json lists
  the exact suite2p plane0/stat.npy per session (as /ceph/... HPC paths, remapped here to the local
  mount), in tracking order. Only the default location of the ROICaT folder and of analysis/ differs
  per cohort (COHORT_PRESETS). Cohort2 runs sit flat in <mouse>/ROICaT and cohort3's all-session runs
  flat in preprocessed_funcimg_Nov2025/derivatives/sub-<id>/funcimg_tracked (the smaller pairwise
  runs in preprocessed_behav_Nov2025/.../ROICaT/<run> are still reachable with --roicat-dir/--run).
  GoalProgress runs are one subfolder per run inside sub-<id>/ROICaT, and subfolder-style runs get
  their own output subfolder named after the joined session tags (GoalProgress run folder names are
  whole session names joined). Holography (SaReSuKuMaRa_20260911_holography/cohort1) runs come from
  roicat_analysis/roicat_tracking_holography.py (roicat 1.7, same output layout) and sit flat in
  cohort1/analysis/sub-<id>/ROICaT; session names (ses-2LM001_date-..-t0) give tags t0..t3.
- Duplicate sessions dropped. In cohort3, suite2p was rerun for some recordings and every rerun went
  into the funcimg_tracked ROICaT run as its own "session" (e.g. ses-004/005/006_id-full010_...), so the
  same recording appeared several times and its cells were matched to themselves. load_tracking keeps,
  per session tag, only the copy whose stat.npy is newest (the latest rerun -- the ses-XXX number is not
  a reliable order: for sub-010 full037 the newest copy is ses-003) and discards the others' labels
  before anything is built, so every plot and count shows each recording once. The clustering itself
  was still done with the copies present; this can't be undone without rerunning ROICaT.
- No roicat dependency. The richfile outputs are plain folders of .json/.npy/.npz files, read here
  with numpy/scipy, so this runs in the bonsai_abcd env (which has no roicat/torch).
  preprocessing/roicat_analysis/roi_tracking_helpers.py imports roicat/torch/IPython at module level
  and is therefore not imported; its count definitions (get_neuron_count,
  visualize_tracked_clusters_per_session, visualize_pairwise_cluster_persistence) and its iscell
  masking (align_rois -> roicat.util.mask_UCIDs_with_iscell) are reimplemented instead. Checked
  against align_rois' saved output for TAA0000062 (227 clusters, identical per-session counts).
- iscell-masked. ROICaT clusters all suite2p ROIs; as in align_rois, cluster labels of ROIs with
  iscell==0 are set to -1, so all plots are about accepted cells. iscell-accepted ROIs left
  unassigned by ROICaT (label -1) count as cells found in a single session. tracking_counts.png also
  shows how many accepted cells ROICaT did match to *some* ROI in another session, so the cost of
  the iscell mask (suite2p rejecting the same cell in other sessions) is visible.
- ROI index = suite2p row index. For cohort2 the stat paths in params_used point at the
  SaReMaRa_20260519_NeuralAnalysis copy of suite2p, verified byte-identical (Oct 2026, mice 062/065)
  to the raw-tree suite2p that nwb_barcoding built the NWBs from, so cluster members index directly
  into Ca_traces / the NWB ROI table.
- Two coordinate frames. Crops are cut from each session's own mean image around the ROI's own
  footprint (native frame -- exact masks, but every panel is re-centred), so position consistency is
  judged separately from the ROI's centroid in ROICaT's common registered frame (ROIs_aligned), as a
  displacement from the cluster's median position.
- Crop background is meanImg, switching per crop to meanImgE when the cell barely stands out
  ((mean in mask - mean of a surrounding ring) / crop std < LOW_CONTRAST).
- Centroids are taken from stat ypix/xpix means (not stat['med']) so ROIs imported from a Cellpose
  segmentation (GoalProgress sub-002) work too.
- Displacement matrix. Saved for all cells, not only the sampled ones, so a displacement threshold can
  be applied when picking cells instead of rerunning ROICaT with a stricter cutoff. It is measured from
  the cell's median position across its sessions, so for a 2-session cell both members get half the
  pair distance; for >=3 sessions a single mismatched session stands out against the others. Rows with
  roicat_cell_id -1 are iscell ROIs ROICaT left unassigned (one session each).
- Outlier sessions: per-session refound fraction (accepted cells also found in any other session)
  below median - 3 * scaled MAD, or below half the median.

Run in the bonsai_abcd env, from the repo root:
  python preprocessing/quality_control/tracking_overview.py TAA0000062 --cohort 2
  python preprocessing/quality_control/tracking_overview.py 010 --cohort 3
  python preprocessing/quality_control/tracking_overview.py 001 --cohort holo
"""

import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage

try:
    import era_brewer
except ImportError:  # run as a script: put the repo root on the path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    import era_brewer

PROJECTS_ROOT = '/Volumes/mrsic_flogel/public/projects'
HPC_PROJECTS_ROOT = '/ceph/mrsic_flogel/public/projects'

# Default ROICaT folder and analysis/ root per cohort; {mouse} is the mouse ID as typed on the CLI
# (cohort2: "TAA0000062", cohort3/GP/holo: "004").
COHORT_PRESETS = {
    '2': dict(roicat_dir=f'{PROJECTS_ROOT}/SaReMaRa_20260519_NeuralAnalysis/Cohort2/{{mouse}}/ROICaT',
              root_path=f'{PROJECTS_ROOT}/AtApSuKuSaRe_20250129_HFScohort2'),
    '3': dict(roicat_dir=f'{PROJECTS_ROOT}/SuKuSaRe_20250923_HFScohort3/preprocessed_funcimg_Nov2025/derivatives/sub-{{mouse}}/funcimg_tracked',
              root_path=f'{PROJECTS_ROOT}/SuKuSaRe_20250923_HFScohort3'),
    'gp': dict(roicat_dir=f'{PROJECTS_ROOT}/SaReMaRa_20260629_GoalProgress/preprocessed/sub-{{mouse}}/ROICaT',
               root_path=f'{PROJECTS_ROOT}/SaReMaRa_20260629_GoalProgress'),
    'holo': dict(roicat_dir=f'{PROJECTS_ROOT}/SaReSuKuMaRa_20260911_holography/cohort1/analysis/sub-{{mouse}}/ROICaT',
                 root_path=f'{PROJECTS_ROOT}/SaReSuKuMaRa_20260911_holography/cohort1'),
}

# Session-tag patterns tried in order: cohort2 "..._protocol-t6", cohort3 "ses-005_id-full030_date-...",
# GoalProgress "ses-1LM001_date-20260701T132524-t0".
SESSION_TAG_PATTERNS = [r'protocol-(t\w+)$', r'_id-([^_]+)_', r'-(t\d+)$', r'^(ses-[^_]+)']

CROP_HALF = 20        # native crop is (2*CROP_HALF) px square, ~54 um at 1.35 um/px
LOW_CONTRAST = 1.0    # below this, the crop falls back to meanImgE

_MIDNIGHT = era_brewer.era_brew('Midnight1', n=7, brew_type='continuous')
PRESENT_COLOR = _MIDNIGHT[2]
PRESENCE_CMAP = ListedColormap(['#F2F2F2', PRESENT_COLOR])
FLAG_COLOR = era_brewer.era_brew('NineteenEightyNine', n=7, brew_type='continuous')[0]
MASK_COLOR = '#FAE4A6'


# ----------------------------------------------------------------------------------------------
# Locating and loading a ROICaT run
# ----------------------------------------------------------------------------------------------

def _local_path(path):
    """Map an HPC (/ceph/...) path stored in a ROICaT run to the local share mount."""
    return Path(str(path).replace(HPC_PROJECTS_ROOT, PROJECTS_ROOT))


def find_roicat_run(roicat_dir, run=None):
    """Return the folder holding one ROICaT run's *.tracking.* files.

    Flat layout (cohort2): the files sit directly in roicat_dir. Subfolder layout (cohort3/GP): one
    folder per run; picked automatically if there is only one, else `run` must name it.
    """
    roicat_dir = Path(roicat_dir)
    if run is None and list(roicat_dir.glob('*.tracking.params_used.json')):
        return roicat_dir
    runs = sorted(p for p in roicat_dir.iterdir()
                  if p.is_dir() and list(p.glob('*.tracking.params_used.json')))
    if run is not None:
        matches = [p for p in runs if p.name == run]
        if not matches:
            raise FileNotFoundError(f"No ROICaT run '{run}' in {roicat_dir}. Available: {[p.name for p in runs]}")
        return matches[0]
    if len(runs) == 1:
        return runs[0]
    if not runs:
        raise FileNotFoundError(f'No ROICaT run (*.tracking.params_used.json) found in {roicat_dir}')
    raise ValueError(f'{roicat_dir} holds several ROICaT runs, pass one with run=/--run: '
                     f'{[p.name for p in runs]}')


def session_tag(session_name):
    for pattern in SESSION_TAG_PATTERNS:
        m = re.search(pattern, session_name)
        if m:
            return m.group(1)
    return session_name


def _richfile_list(richfile_dir, *keys):
    """Load a list stored in a richfile folder (nested dict items -> value.list/<i>.npy|.npz)."""
    folder = Path(richfile_dir)
    for key in keys:
        folder = folder / f'{key}.dict_item' / 'value.dict'
    folder = folder.parent / 'value.list'
    files = sorted(folder.glob('[0-9]*.np[yz]'), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f'No list items under {folder}')
    if files[0].suffix == '.npz':
        import scipy.sparse
        return [scipy.sparse.load_npz(f) for f in files]
    return [np.load(f) for f in files]


def latest_per_tag(sessions, stat_paths):
    """Indices (tracking order) keeping one session per session tag: the copy with the newest stat.npy.
    Prints which copies were dropped."""
    by_tag = {}
    for i, s in enumerate(sessions):
        by_tag.setdefault(session_tag(s), []).append(i)
    keep = []
    for tag, idx in by_tag.items():
        if len(idx) == 1:
            keep += idx
            continue
        mtimes = {i: stat_paths[i].stat().st_mtime for i in idx}
        best = max(idx, key=mtimes.get)
        keep.append(best)
        fmt = lambda i: f"{sessions[i]} ({datetime.fromtimestamp(mtimes[i]):%Y-%m-%d %H:%M})"
        print(f"Duplicate session {tag}: keeping {fmt(best)}; dropping "
              + ', '.join(fmt(i) for i in idx if i != best))
    return sorted(keep)


def load_tracking(run_dir, drop_duplicates=True):
    """Load one ROICaT run plus the suite2p outputs of every tracked session.

    drop_duplicates: keep only the newest suite2p copy of sessions that were tracked more than once
    (same session tag, see latest_per_tag); the other copies' ROIs are removed from the run.

    Returns a dict with:
        name : ROICaT data name (file prefix), run_dir : Path
        sessions, tags : session folder names / short tags, in tracking order
        um_per_pixel : float
        labels : list (per session) of int arrays, ROICaT cluster ID per suite2p ROI (-1 unassigned)
        iscell : list of bool arrays
        stat_paths : local paths of the stat.npy files (the ROI indices index into these)
        stat, ops : list of suite2p stat arrays / ops dicts
        aligned_centroids : list of (n_roi, 2) arrays, (row, col) in ROICaT's registered frame
        registered_images : list of registered meanImgE (ROICaT's FOV images), or None if absent
    """
    run_dir = Path(run_dir)
    name = next(run_dir.glob('*.tracking.params_used.json')).name.split('.tracking.')[0]
    params = json.load(open(run_dir / f'{name}.tracking.params_used.json'))
    clusters = json.load(open(run_dir / f'{name}.tracking.results_clusters.json'))

    stat_paths = [_local_path(p) for p in params['data']['__init__']['paths_statFiles']]
    # Session folder = the nearest "ses-*" path component (cohort2 nests funcimg/Session/suite2p).
    sessions = [next(part for part in reversed(p.parts) if part.startswith('ses-')) for p in stat_paths]
    labels = [np.asarray(lb, dtype=int) for lb in clusters['labels_bySession']]
    keep = latest_per_tag(sessions, stat_paths) if drop_duplicates else list(range(len(sessions)))
    stat_paths, sessions, labels = ([x[i] for i in keep] for x in (stat_paths, sessions, labels))

    stat, ops, iscell = [], [], []
    for path, lb in zip(stat_paths, labels):
        plane0 = path.parent
        st = np.load(plane0 / 'stat.npy', allow_pickle=True)
        if len(st) != len(lb):
            raise ValueError(f'{plane0}: stat.npy has {len(st)} ROIs but ROICaT labels {len(lb)} -- '
                             'suite2p was rerun after tracking?')
        stat.append(st)
        ops.append(np.load(plane0 / 'ops.npy', allow_pickle=True).item())
        iscell.append(np.load(plane0 / 'iscell.npy')[:, 0].astype(bool))

    # Registered-frame centroids from ROICaT's aligned footprints (n_roi x H*W sparse).
    footprints = _richfile_list(run_dir / f'{name}.tracking.results_all.richfile', 'ROIs', 'ROIs_aligned')
    footprints = [footprints[i] for i in keep]
    with open(run_dir / f'{name}.tracking.results_all.richfile' / 'ROIs.dict_item' / 'value.dict'
              / 'frame_width.dict_item' / 'value.json') as f:
        width = int(json.load(f))
    aligned_centroids = []
    for fp in footprints:
        fp = fp.tocsr()
        pix = np.arange(fp.shape[1])
        w = np.asarray(fp.sum(1)).ravel()
        w[w == 0] = np.nan
        rows = np.asarray(fp @ (pix // width)).ravel() / w
        cols = np.asarray(fp @ (pix % width)).ravel() / w
        aligned_centroids.append(np.column_stack([rows, cols]))

    try:
        registered = _richfile_list(run_dir / f'{name}.tracking.run_data.richfile',
                                    'aligner', 'ims_registered_nonrigid')
        registered = [registered[i] for i in keep]
    except FileNotFoundError:
        registered = None

    um = params['data']['__init__'].get('um_per_pixel', 1.0)
    um = float(np.mean(um)) if isinstance(um, list) else float(um)

    return dict(run_dir=run_dir, name=name, sessions=sessions, tags=[session_tag(s) for s in sessions],
                stat_paths=stat_paths,
                um_per_pixel=um, labels=labels, iscell=iscell, stat=stat, ops=ops,
                aligned_centroids=aligned_centroids, registered_images=registered)


def presence_matrix(tracking):
    """Cells x sessions matrix of suite2p ROI indices (-1 = absent), iscell-masked.

    Rows are ROICaT clusters with at least one iscell member, followed by iscell ROIs ROICaT left
    unassigned (one row each, present in a single session). Returns (P, cell_ids) where cell_ids[i] is
    the ROICaT cluster ID, or -1 for an unassigned ROI.
    """
    n_ses = len(tracking['labels'])
    rows, unassigned = {}, []
    n_dup = 0
    for s, (lb, ok) in enumerate(zip(tracking['labels'], tracking['iscell'])):
        for roi in np.where(ok)[0]:
            c = lb[roi]
            if c < 0:
                row = np.full(n_ses, -1)
                row[s] = roi
                unassigned.append(row)
                continue
            row = rows.setdefault(c, np.full(n_ses, -1))
            if row[s] >= 0:
                n_dup += 1  # ROICaT should never put two ROIs of one session in a cluster
                continue
            row[s] = roi
    if n_dup:
        print(f'Warning: {n_dup} cluster(s) had >1 ROI in the same session; kept the first.')
    ids = sorted(rows)
    P = np.array([rows[c] for c in ids] + unassigned, dtype=int).reshape(-1, n_ses)
    cell_ids = np.array(ids + [-1] * len(unassigned), dtype=int)
    return P, cell_ids


def displacement_um(tracking, P):
    """Registered-frame distance (um) of each present entry of P from its row's median centroid.
    NaN where absent, and for rows present in a single session."""
    n_cells, n_ses = P.shape
    xy = np.full((n_cells, n_ses, 2), np.nan)
    for s in range(n_ses):
        here = P[:, s] >= 0
        xy[here, s] = tracking['aligned_centroids'][s][P[here, s]]
    with warnings.catch_warnings():  # rows whose aligned footprints are all empty
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(xy, axis=1, keepdims=True)
    d = np.linalg.norm(xy - med, axis=2) * tracking['um_per_pixel']
    d[(P >= 0).sum(1) < 2] = np.nan
    return d, xy


# ----------------------------------------------------------------------------------------------
# Per-session summary
# ----------------------------------------------------------------------------------------------

def session_summary(tracking, P):
    """Per-session counts (DataFrame) incl. pairwise s->s+1 overlap, survival and outlier flag."""
    present = P >= 0
    n_present = present.sum(1)
    n_ses = P.shape[1]
    n_iscell = present.sum(0)

    # Accepted cells whose ROICaT cluster has a member (accepted or not) in another session.
    matched_any = []
    for lb, ok in zip(tracking['labels'], tracking['iscell']):
        others = set()
        for lb2 in tracking['labels']:
            if lb2 is not lb:
                others.update(np.unique(lb2[lb2 >= 0]).tolist())
        matched_any.append(int(sum(1 for c in lb[ok] if c >= 0 and c in others)))

    refound = np.array([(present[:, s] & (n_present >= 2)).sum() for s in range(n_ses)])
    refound_frac = refound / np.maximum(n_iscell, 1)
    pair_n = np.array([(present[:, s] & present[:, s + 1]).sum() for s in range(n_ses - 1)] + [np.nan])
    survival = np.array([present[:, :k + 1].all(1).sum() for k in range(n_ses)])

    med = np.median(refound_frac)
    mad = 1.4826 * np.median(np.abs(refound_frac - med))
    outlier = (refound_frac < med - 3 * mad) | (refound_frac < 0.5 * med)

    return pd.DataFrame(dict(
        session=tracking['sessions'], tag=tracking['tags'],
        n_rois=[len(st) for st in tracking['stat']], n_iscell=n_iscell,
        n_refound=refound, refound_frac=refound_frac,
        n_matched_before_iscell_mask=matched_any,
        n_with_next=pair_n, frac_with_next=pair_n / np.maximum(n_iscell, 1),
        n_survival_from_first=survival, frac_survival_from_first=survival / max(n_iscell[0], 1),
        outlier=outlier))


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def _finish(fig, save_path, show):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_cluster_matrix(tracking, P, save_path=None, show=False):
    """Cells x sessions presence matrix, rows sorted by number of sessions present then first session,
    with a side histogram of how many cells are present in k sessions."""
    present = P >= 0
    n_present = present.sum(1)
    order = np.lexsort((np.argmax(present, axis=1), -n_present))
    present, n_present = present[order], n_present[order]
    n_cells, n_ses = present.shape

    height = float(np.clip(0.045 * n_cells, 7, 40))  # ~6 px per row at dpi 150
    fig, axes = plt.subplots(1, 3, figsize=(3 + 0.5 * n_ses, height), sharey=True,
                             gridspec_kw=dict(width_ratios=[n_ses, 0.6, 3], wspace=0.06))
    ax = axes[0]
    ax.imshow(present, aspect='auto', interpolation='none', cmap=PRESENCE_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(n_ses))
    ax.set_xticklabels(tracking['tags'], rotation=90)
    ax.set_ylabel('cell (sorted by sessions present)')
    ax.set_xlabel('session')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(axis='x', labeltop=True)

    boundaries = np.where(np.diff(n_present) != 0)[0] + 0.5
    for a in axes[:2]:
        for y in boundaries:
            a.axhline(y, color='w', lw=0.8)

    axes[1].imshow(n_present[:, None], aspect='auto', interpolation='none', cmap='Greys', vmin=0, vmax=n_ses)
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(['# ses'], rotation=90)

    # Side histogram, one bar per "present in k sessions" group, aligned to the group's rows. Labels
    # are pushed apart when groups are too thin to hold them.
    hist_ax = axes[2]
    min_gap = n_cells * 0.022
    last_label_y = -np.inf
    for k in range(n_ses, 0, -1):
        idx = np.where(n_present == k)[0]
        if not len(idx):
            continue
        y0, c = idx[0], len(idx)
        hist_ax.barh(y0 + c / 2 - 0.5, c, height=c, color=PRESENT_COLOR, alpha=0.3 + 0.7 * k / n_ses)
        label_y = max(y0 + c / 2 - 0.5, last_label_y + min_gap)
        hist_ax.annotate(f'{k} ses: {c}', xy=(c, y0 + c / 2 - 0.5), xytext=(c + 0.04 * n_cells, label_y),
                         va='center', fontsize=8,
                         arrowprops=dict(arrowstyle='-', lw=0.5, color='0.5') if label_y != y0 + c / 2 - 0.5 else None)
        last_label_y = label_y
    hist_ax.set_xlim(0, n_present.size and np.bincount(n_present).max() * 1.6)
    hist_ax.set_xlabel('# cells')
    hist_ax.spines[['top', 'right']].set_visible(False)
    hist_ax.set_ylim(n_cells - 0.5, -0.5)

    fig.suptitle(f"{tracking['name']}: {n_cells} iscell cells across {n_ses} sessions "
                 f"({int((n_present == n_ses).sum())} in all sessions, {int((n_present >= 2).sum())} in >=2)",
                 fontsize=11, y=1 - 0.2 / height, va='top')
    fig.subplots_adjust(top=1 - 1.1 / height)  # fixed inches for the title + top tick labels
    _finish(fig, save_path, show)


def _shade_outliers(ax, summary, x_offset=0.0):
    for s in np.where(summary['outlier'])[0]:
        ax.axvspan(s - 0.5 + x_offset, s + 0.5 + x_offset, color=FLAG_COLOR, alpha=0.15, lw=0)


def plot_tracking_counts(tracking, summary, save_path=None, show=False):
    """(a) per-session counts, (b) pairwise s->s+1 overlap, (c) survival from the first session."""
    n_ses = len(summary)
    x = np.arange(n_ses)
    tags = list(summary['tag'])
    fig, axes = plt.subplots(3, 1, figsize=(max(6, 0.55 * n_ses + 2), 9), sharex=True)

    ax = axes[0]
    ax.plot(x, summary['n_rois'], 'o-', color='0.75', label='suite2p ROIs (all)')
    ax.plot(x, summary['n_iscell'], 'o-', color=_MIDNIGHT[4], label='iscell cells')
    ax.plot(x, summary['n_matched_before_iscell_mask'], 'o--', color=_MIDNIGHT[3], ms=4,
            label='iscell, matched by ROICaT to any ROI elsewhere')
    ax.plot(x, summary['n_refound'], 'o-', color=PRESENT_COLOR, label='iscell, refound as iscell elsewhere')
    ax2 = ax.twinx()
    ax2.plot(x, summary['refound_frac'], 's:', color=FLAG_COLOR, ms=4, label='refound fraction')
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('refound / iscell', color=FLAG_COLOR)
    ax.set_ylabel('# cells')
    ax.set_title('a. cells per session', loc='left', fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(1.13, 1))
    _shade_outliers(ax, summary)

    ax = axes[1]
    pair_x = x[:-1] + 0.5
    ax.bar(pair_x, summary['n_with_next'][:-1], width=0.6, color=PRESENT_COLOR, alpha=0.7, label='# in both')
    ax2 = ax.twinx()
    ax2.plot(pair_x, summary['frac_with_next'][:-1], 'o-', color=FLAG_COLOR, label='fraction of session s cells')
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('fraction of s', color=FLAG_COLOR)
    ax.set_ylabel('# cells in s and s+1')
    ax.set_title('b. pairwise overlap, session s -> s+1 (bars between sessions)', loc='left', fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(1.13, 1))
    _shade_outliers(ax, summary)

    ax = axes[2]
    ax.plot(x, summary['n_survival_from_first'], 'o-', color=PRESENT_COLOR, label='# present in all sessions up to k')
    ax2 = ax.twinx()
    ax2.plot(x, summary['frac_survival_from_first'], 's:', color=FLAG_COLOR, ms=4,
             label=f'fraction of {tags[0]} cells')
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel(f'fraction of {tags[0]}', color=FLAG_COLOR)
    ax.set_ylabel('# cells')
    ax.set_title(f'c. survival: cells present in every session from {tags[0]} to k', loc='left', fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(1.13, 1))
    _shade_outliers(ax, summary)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=90)
    ax.set_xlim(-0.6, n_ses - 0.4)

    flagged = [t for t, o in zip(tags, summary['outlier']) if o]
    fig.suptitle(f"{tracking['name']}: tracking counts"
                 + (f"  --  outlier session(s), few cells refound: {', '.join(flagged)} (shaded)" if flagged else
                    '  --  no outlier sessions'), fontsize=11)
    fig.tight_layout()
    _finish(fig, save_path, show)


def pick_cells(P, cell_ids, n_cells=10, min_presence=0.9, seed=0):
    """Random clusters present in >= min_presence of sessions; if too few, filled up with the
    most-present remaining clusters (random among ties). Returns (row indices into P, note)."""
    rng = np.random.default_rng(seed)
    n_ses = P.shape[1]
    n_present = (P >= 0).sum(1)
    n_present[cell_ids < 0] = 0  # unassigned ROIs are never tracked cells
    threshold = int(np.ceil(min_presence * n_ses))
    eligible = np.where(n_present >= threshold)[0]
    picked = list(rng.choice(eligible, min(n_cells, len(eligible)), replace=False))
    note = ''
    if len(picked) < n_cells:
        rest = np.array([i for i in np.where(n_present >= 2)[0] if i not in picked])
        rest = rest[np.lexsort((rng.random(len(rest)), -n_present[rest]))]
        filled = list(rest[:n_cells - len(picked)])
        picked += filled
        if filled:
            note = (f'only {len(eligible)} cell(s) present in >={threshold}/{n_ses} sessions; filled up with '
                    f'the most-present cells ({n_present[filled].min()}-{n_present[filled].max()} sessions)')
    return np.array(picked, dtype=int), note


def _crop(img, center, half=CROP_HALF):
    """Square crop around (row, col), padded with NaN at the FOV edge. Returns crop, (row0, col0)."""
    r0, c0 = int(round(center[0])) - half, int(round(center[1])) - half
    out = np.full((2 * half, 2 * half), np.nan)
    rs, cs = max(r0, 0), max(c0, 0)
    re_, ce = min(r0 + 2 * half, img.shape[0]), min(c0 + 2 * half, img.shape[1])
    if re_ > rs and ce > cs:
        out[rs - r0:re_ - r0, cs - c0:ce - c0] = img[rs:re_, cs:ce]
    return out, (r0, c0)


def _native_crop(stat_roi, ops):
    """Crop of meanImg (or meanImgE when the cell barely stands out) around one ROI, plus its mask."""
    ypix, xpix = stat_roi['ypix'], stat_roi['xpix']
    center = (ypix.mean(), xpix.mean())
    mask_full = np.zeros((ops['Ly'], ops['Lx']), bool)
    mask_full[ypix, xpix] = True
    mask, _ = _crop(mask_full.astype(float), center)
    mask = np.nan_to_num(mask) > 0

    img_name = 'meanImg'
    crop, _ = _crop(np.asarray(ops['meanImg'], float), center)
    ring = ndimage.binary_dilation(mask, iterations=4) & ~ndimage.binary_dilation(mask, iterations=1)
    contrast = np.nan
    if mask.any() and ring.any() and np.nanstd(crop) > 0:
        contrast = (np.nanmean(crop[mask]) - np.nanmean(crop[ring])) / np.nanstd(crop)
    if not contrast >= LOW_CONTRAST and 'meanImgE' in ops:
        img_name = 'meanImgE'
        crop, _ = _crop(np.asarray(ops['meanImgE'], float), center)
    return crop, mask, img_name


def plot_tracked_cell(tracking, P, row, cell_id, disp, xy, max_shift_um=5.0, save_path=None, show=False):
    """One cell across sessions: native crops (top) and registered-frame position (bottom)."""
    n_ses = P.shape[1]
    tags = tracking['tags']
    colors = era_brewer.era_brew('Midnight2', n=max(n_ses, 2), brew_type='continuous')

    ncols = max(n_ses, 6)
    fig = plt.figure(figsize=(1.6 * ncols, 6.2))
    gs = fig.add_gridspec(2, ncols, height_ratios=[1, 1.9], hspace=0.35)

    for s in range(n_ses):
        ax = fig.add_subplot(gs[0, s])
        ax.set_xticks([])
        ax.set_yticks([])
        roi = P[row, s]
        if roi < 0:
            ax.set_facecolor('0.85')
            ax.text(0.5, 0.5, 'not found', ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_title(tags[s], fontsize=8)
            continue
        crop, mask, img_name = _native_crop(tracking['stat'][s][roi], tracking['ops'][s])
        vmin, vmax = np.nanpercentile(crop, [1, 99])
        ax.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax, interpolation='none')
        ax.contour(mask, levels=[0.5], colors=MASK_COLOR, linewidths=0.9)
        shift = disp[row, s]
        flag = shift > max_shift_um
        ax.set_title(f"{tags[s]}{' *E' if img_name == 'meanImgE' else ''}\n{shift:.1f} um",
                     fontsize=8, color=FLAG_COLOR if flag else 'k')
        for sp in ax.spines.values():
            sp.set_color(colors[s])
            sp.set_linewidth(2.5 if not flag else 3.5)

    pts = xy[row]
    ok = ~np.isnan(pts[:, 0])
    third = ncols // 3

    med = np.nanmedian(pts, axis=0)
    half = int(np.ceil(max(CROP_HALF, np.nanmax(np.abs(pts[ok] - med)) + 6)))

    # (i) whole registered FOV with this cell's per-session centroids and the zoom window
    ax = fig.add_subplot(gs[1, :third])
    bg = tracking['registered_images'][0] if tracking['registered_images'] else None
    if bg is not None:
        # nan-safe: roicat >= 1.7 leaves NaN borders in the non-rigid registered images (holography runs)
        ax.imshow(bg, cmap='gray', vmin=np.nanpercentile(bg, 1), vmax=np.nanpercentile(bg, 99.5))
    ax.scatter(pts[ok, 1], pts[ok, 0], s=14, c=[colors[s] for s in np.where(ok)[0]], edgecolors='w', linewidths=0.4)
    ax.add_patch(plt.Rectangle((med[1] - half, med[0] - half), 2 * half, 2 * half, fill=False,
                               color=FLAG_COLOR, lw=1.2))
    ax.set_title('registered FOV (first session), box = zoom', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # (ii) zoom around the cluster in the registered frame
    ax = fig.add_subplot(gs[1, third:2 * third])
    if bg is not None:
        zoom, (r0, c0) = _crop(bg, med, half=half)
        ax.imshow(zoom, cmap='gray', vmin=np.nanpercentile(zoom, 1), vmax=np.nanpercentile(zoom, 99.5),
                  extent=(c0 - 0.5, c0 + zoom.shape[1] - 0.5, r0 + zoom.shape[0] - 0.5, r0 - 0.5))
    else:
        ax.invert_yaxis()
    ax.plot(pts[ok, 1], pts[ok, 0], '-', color='w', lw=0.8, alpha=0.7)
    ax.scatter(pts[ok, 1], pts[ok, 0], s=30, c=[colors[s] for s in np.where(ok)[0]], edgecolors='w',
               linewidths=0.5, zorder=3)
    circle = plt.Circle((med[1], med[0]), max_shift_um / tracking['um_per_pixel'], fill=False, ls='--',
                        color=FLAG_COLOR, lw=0.8)
    ax.add_patch(circle)
    ax.set_title(f'zoom: centroid per session (circle = {max_shift_um:g} um)', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # (iii) displacement from the cluster median per session
    ax = fig.add_subplot(gs[1, 2 * third:])
    d = disp[row]
    ax.scatter(np.arange(n_ses), d, c=[FLAG_COLOR if v > max_shift_um else colors[s] for s, v in enumerate(d)],
               s=28, zorder=3)
    ax.plot(np.arange(n_ses), d, '-', color='0.7', lw=0.8)
    ax.axhline(max_shift_um, ls='--', color=FLAG_COLOR, lw=0.8)
    ax.set_xticks(range(n_ses))
    ax.set_xticklabels(tags, rotation=90, fontsize=8)
    ax.set_ylabel('distance from median position (um)')
    ax.set_ylim(0, max(max_shift_um * 1.5, np.nanmax(d) * 1.15 if np.any(~np.isnan(d)) else 1))
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('registered-frame displacement', fontsize=9)

    n_here = int((P[row] >= 0).sum())
    fig.suptitle(f"{tracking['name']} -- ROICaT cell {cell_id}: present in {n_here}/{n_ses} sessions  "
                 f"(top: native mean image, *E = meanImgE used; outline = suite2p mask)", fontsize=10)
    _finish(fig, save_path, show)


def plot_position_consistency(tracking, P, disp, max_shift_um=5.0, save_path=None, show=False):
    """Displacement from cluster median for every member of every >=2-session cell: overall histogram
    and per-session distribution (a whole session shifted would point to a registration problem)."""
    tags = tracking['tags']
    d_all = disp[~np.isnan(disp)]
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.5 * len(tags) + 6), 4),
                             gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = axes[0]
    upper = max(max_shift_um * 3, np.percentile(d_all, 99.5)) if d_all.size else max_shift_um * 3
    ax.hist(np.clip(d_all, 0, upper), bins=40, range=(0, upper), color=PRESENT_COLOR, alpha=0.8)
    ax.axvline(max_shift_um, ls='--', color=FLAG_COLOR)
    frac = (d_all > max_shift_um).mean() if d_all.size else np.nan
    ax.set_title(f'{d_all.size} ROIs in {int(((P >= 0).sum(1) >= 2).sum())} multi-session cells\n'
                 f'{100 * frac:.1f}% further than {max_shift_um:g} um from their cell\'s median', fontsize=9)
    ax.set_xlabel('distance from cell median position (um, registered frame)')
    ax.set_ylabel('# ROIs')
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    per_ses = [disp[~np.isnan(disp[:, s]), s] for s in range(len(tags))]
    ax.boxplot([p if p.size else [np.nan] for p in per_ses], positions=range(len(tags)), widths=0.6,
               showfliers=True, flierprops=dict(ms=2, alpha=0.5), medianprops=dict(color=FLAG_COLOR))
    ax.axhline(max_shift_um, ls='--', color=FLAG_COLOR, lw=0.8)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=90)
    ax.set_ylabel('distance from cell median (um)')
    ax.set_title('per session', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f"{tracking['name']}: position consistency of tracked cells", fontsize=11)
    fig.tight_layout()
    _finish(fig, save_path, show)


# ----------------------------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------------------------

def run_tracking_overview(run_dir, save_dir, n_cells=10, min_presence=0.9, max_shift_um=5.0, seed=0,
                          show=False):
    """Build every tracking plot for one ROICaT run. save_dir=None to only display."""
    tracking = load_tracking(run_dir)
    P, cell_ids = presence_matrix(tracking)
    summary = session_summary(tracking, P)
    disp, xy = displacement_um(tracking, P)
    out = Path(save_dir) if save_dir is not None else None
    path = (lambda name: out / name) if out is not None else (lambda name: None)

    print(f"{tracking['name']}: {len(tracking['sessions'])} sessions ({', '.join(tracking['tags'])}), "
          f"{len(P)} iscell cells")
    if all(ic.all() for ic in tracking['iscell']):
        print('Note: iscell is all ones in every session (e.g. Cellpose-imported ROIs) -- no masking applied.')
    flagged = list(summary.loc[summary['outlier'], 'tag'])
    print(f"Outlier sessions (few cells refound): {flagged if flagged else 'none'}")

    plot_cluster_matrix(tracking, P, save_path=path('cluster_matrix.png'), show=show)
    plot_tracking_counts(tracking, summary, save_path=path('tracking_counts.png'), show=show)
    plot_position_consistency(tracking, P, disp, max_shift_um, save_path=path('position_consistency.png'),
                              show=show)

    rows, note = pick_cells(P, cell_ids, n_cells=n_cells, min_presence=min_presence, seed=seed)
    if note:
        print('Note:', note)
    for row in rows:
        plot_tracked_cell(tracking, P, row, cell_ids[row], disp, xy, max_shift_um,
                          save_path=path(f'tracked_cell_{cell_ids[row]}.png'), show=show)

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / 'tracking_summary.csv', index=False)
        sampled = pd.DataFrame(P[rows], columns=[f'roi_{t}' for t in tracking['tags']])
        sampled.insert(0, 'n_sessions', (P[rows] >= 0).sum(1))
        sampled.insert(0, 'roicat_cell_id', cell_ids[rows])
        sampled['max_shift_um'] = np.nanmax(disp[rows], axis=1)
        sampled.to_csv(out / 'tracked_cells_sampled.csv', index=False)
        save_cell_matrix(tracking, P, cell_ids, disp, out)
        print(f'Saved to {out}')
    return dict(tracking=tracking, P=P, cell_ids=cell_ids, summary=summary, displacement=disp,
                sampled_rows=rows)


def save_cell_matrix(tracking, P, cell_ids, disp, out):
    """Write cell_matrix.csv (one row per cell: roi_<tag> and shift_um_<tag> per session) and
    cell_matrix.npz (same arrays, plus session names and the stat.npy paths the ROI indices refer to)."""
    tags = tracking['tags']
    df = pd.DataFrame({'roicat_cell_id': cell_ids, 'n_sessions': (P >= 0).sum(1)})
    for s, t in enumerate(tags):
        df[f'roi_{t}'] = P[:, s]
    for s, t in enumerate(tags):
        df[f'shift_um_{t}'] = disp[:, s]
    with warnings.catch_warnings():  # all-NaN rows (single-session cells)
        warnings.simplefilter('ignore', RuntimeWarning)
        df['max_shift_um'] = np.nanmax(disp, axis=1)
    df.to_csv(Path(out) / 'cell_matrix.csv', index=False)
    np.savez(Path(out) / 'cell_matrix.npz', roi=P, shift_um=disp, roicat_cell_id=cell_ids,
             tags=np.array(tags), sessions=np.array(tracking['sessions']),
             stat_paths=np.array([str(p) for p in tracking['stat_paths']]),
             um_per_pixel=tracking['um_per_pixel'], roicat_run=str(tracking['run_dir']))


def load_cell_matrix(save_dir):
    """Load cell_matrix.npz from a tracking_overview output folder as a dict of arrays
    (roi, shift_um: n_cells x n_sessions; roicat_cell_id; tags; sessions; stat_paths; ...)."""
    with np.load(Path(save_dir) / 'cell_matrix.npz') as f:
        return {k: f[k] for k in f.files}


def select_tracked_cells(cell_matrix, sessions=None, max_shift_um=5.0):
    """Cells present in all `sessions` (tags; default: every session) whose registered-frame displacement
    is <= max_shift_um in each of them (max_shift_um=None: no displacement filter).

    Returns (roi, cell_ids): roi is an (n_selected x len(sessions)) array of suite2p ROI indices, columns
    in the order of `sessions`, ready to index each session's traces; cell_ids the ROICaT cluster IDs."""
    tags = list(cell_matrix['tags'])
    sessions = tags if sessions is None else list(sessions)
    cols = [tags.index(t) for t in sessions]
    roi, shift = cell_matrix['roi'][:, cols], cell_matrix['shift_um'][:, cols]
    keep = (roi >= 0).all(1) & (cell_matrix['roicat_cell_id'] >= 0)
    if max_shift_um is not None:
        keep &= ~(shift > max_shift_um).any(1)  # NaN (cell present in a single session) passes
    return roi[keep], cell_matrix['roicat_cell_id'][keep]


def default_save_dir(root_path, mouse, run_dir, roicat_dir):
    """<root>/analysis/sub-<mouse>/tracking, plus /<joined session tags> for subfolder-style runs."""
    base = Path(root_path) / 'analysis' / f'sub-{mouse}' / 'tracking'
    if Path(run_dir).resolve() == Path(roicat_dir).resolve():
        return base
    params = json.load(open(next(Path(run_dir).glob('*.tracking.params_used.json'))))
    sessions = [next(part for part in reversed(_local_path(p).parts) if part.startswith('ses-'))
                for p in params['data']['__init__']['paths_statFiles']]
    return base / '_'.join(session_tag(s) for s in sessions)


def _cli():
    """Inspect one mouse's ROICaT tracking and save overview plots to
    <root>/analysis/sub-<mouse>/tracking (plain argparse, runs in the bonsai_abcd env)."""
    import argparse
    parser = argparse.ArgumentParser(description=_cli.__doc__)
    parser.add_argument('mouse', help='e.g. "TAA0000062" (cohort2) or "004" (cohort3/GoalProgress/holo)')
    parser.add_argument('--cohort', required=True, choices=sorted(COHORT_PRESETS))
    parser.add_argument('--run', default=None, help='ROICaT run subfolder, if the mouse has several')
    parser.add_argument('--roicat-dir', default=None, help='override the cohort default ROICaT folder')
    parser.add_argument('--root-path', default=None, help='override the cohort default project root (analysis/ goes here)')
    parser.add_argument('--n-cells', type=int, default=10)
    parser.add_argument('--min-presence', type=float, default=0.9,
                        help='fraction of sessions a sampled cell must be present in (default 0.9)')
    parser.add_argument('--max-shift-um', type=float, default=5.0,
                        help='registered-frame displacement above which a session is flagged (default 5 um)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    preset = COHORT_PRESETS[args.cohort]
    roicat_dir = args.roicat_dir or preset['roicat_dir'].format(mouse=args.mouse)
    root_path = args.root_path or preset['root_path']
    run_dir = find_roicat_run(roicat_dir, args.run)
    save_dir = default_save_dir(root_path, args.mouse, run_dir, roicat_dir)
    run_tracking_overview(run_dir, save_dir, n_cells=args.n_cells, min_presence=args.min_presence,
                          max_shift_um=args.max_shift_um, seed=args.seed, show=args.show)


if __name__ == '__main__':
    _cli()

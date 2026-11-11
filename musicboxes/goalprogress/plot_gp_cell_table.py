"""Overview figure of the table written by gp_cell_table.py, to check what its columns mean.

Saved as gp_cell_table_overview.png next to the table. Run from the repo root, after gp_cell_table:

    python -m musicboxes.goalprogress.plot_gp_cell_table 001 seq3-007

Left: channel 2 mean image with the outlines of all gp_tuned cells. N_HIGHLIGHT of them, spread
over preferred phase, are drawn in colour with "cell_id (x, y)" printed next to them and a cross at
the table's (x, y). Right: the mean GP curves of those cells.

Coordinate check: the image is plotted as meanImg_chan2[row, column] with the origin at the top
left, so the horizontal axis is suite2p's x and the vertical axis is y. If x and y in the table are
right, each cross sits in the middle of its outline and the printed numbers can be read off the axes.
The masks are drawn from stat.npy, so a cross inside its outline shows that the table agrees with
stat.npy; whether a second system (e.g. the holography software) uses the same convention has to be
checked there, against the printed coordinates.

The curves are recomputed with n_shuffles=0 (the table only stores the p-values), which gives the
same curves and preferred phases as the shuffled run: the shuffles do not affect them.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

import era_brewer
from ceph_paths import ceph_path
import musicboxes.goalprogress.gp_learning_helpers as h
import musicboxes.goalprogress.gp_cell_table as gct

N_HIGHLIGHT = 10
CONTRAST_PERCENTILES = (1, 99.9)
CONTRAST_GAMMA = 0.7   # < 1 lifts the dim half of the FOV without saturating the bright half


def pick_highlighted(table, n=N_HIGHLIGHT, exclude=()):
    """Rows of `n` gp_tuned cells spread over phase: for each of n equal phase bins the cell with the
    highest split_r, or the closest unused phase if the bin is empty. Sorted by phase. Cell IDs in
    `exclude` are never highlighted (their outlines stay white)."""
    tuned = table[table.gp_tuned & ~table.cell_id.isin(exclude)]
    picked = []
    for target in (np.arange(n) + 0.5) / n:
        free = tuned[~tuned.index.isin(picked)]
        if free.empty:
            break
        dist = h.circular_distance(free.preferred_phase.to_numpy(), target)
        in_bin = free[dist <= 0.5 / n]
        picked.append(in_bin.split_r.idxmax() if len(in_bin) else free.index[np.argmin(dist)])
    return table.loc[picked].sort_values('preferred_phase')


def fov_limits(image, ops):
    """Display range from percentiles inside the registered area; the blank border would skew them."""
    (y0, y1), (x0, x1) = ops['yrange'], ops['xrange']
    return np.percentile(image[y0:y1, x0:x1], CONTRAST_PERCENTILES)


def mask_image(stat_entry, shape):
    mask = np.zeros(shape, bool)
    mask[stat_entry['ypix'], stat_entry['xpix']] = True
    return mask


def plot_overview(table, ops, stat, curves, sem, title, save_path, exclude=()):
    highlighted = pick_highlighted(table, exclude=exclude)
    colors = era_brewer.era_brew('NineteenEightyNine', n=len(highlighted), brew_type='continuous')
    image = ops['meanImg_chan2']
    vmin, vmax = fov_limits(image, ops)

    fig = plt.figure(figsize=(16, 9.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.14)
    ax_fov = fig.add_subplot(outer[0])
    ax_fov.imshow(image, cmap='gray', norm=PowerNorm(CONTRAST_GAMMA, vmin=vmin, vmax=vmax), interpolation='none')
    for cell in table.cell_id[table.gp_tuned & ~table.cell_id.isin(highlighted.cell_id)]:
        ax_fov.contour(mask_image(stat[cell], image.shape), levels=[0.5], colors='white', linewidths=0.6, alpha=0.7)

    grid = outer[1].subgridspec(len(highlighted), 1, hspace=0.25)
    bin_centres = (np.arange(curves.shape[1]) + 0.5) / curves.shape[1]
    for i, (row, color) in enumerate(zip(highlighted.itertuples(), colors)):
        ax_fov.contour(mask_image(stat[row.cell_id], image.shape), levels=[0.5], colors=[color], linewidths=2)
        ax_fov.plot(row.x, row.y, '+', color=color, ms=9, mew=1.5)
        left_side = row.x > image.shape[1] * 0.75
        ax_fov.annotate(f'{row.cell_id} ({row.x:.1f}, {row.y:.1f})', (row.x, row.y),
                        xytext=(-11 if left_side else 11, 8), textcoords='offset points',
                        ha='right' if left_side else 'left', fontsize=9, color='black',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor=color, linewidth=1.5, pad=1.5))

        ax = fig.add_subplot(grid[i])
        curve_row = table.index[table.cell_id == row.cell_id][0]
        ax.plot(bin_centres, curves[curve_row], color=color, lw=1.8)
        ax.fill_between(bin_centres, curves[curve_row] - sem[curve_row], curves[curve_row] + sem[curve_row],
                        color=color, alpha=0.25, lw=0)
        ax.axvline(row.preferred_phase, color='0.3', ls='--', lw=0.9)
        ax.set_xlim(0, 1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.text(1.01, 0.5, f'cell {row.cell_id}\nphase {row.preferred_phase:.2f}\nr = {row.split_r:.2f}',
                transform=ax.transAxes, va='center', fontsize=8)
        if i < len(highlighted) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Goal progress (0 = reward)')
        if i == len(highlighted) // 2:
            ax.set_ylabel('dF (noise units), mean ± SEM')

    ax_fov.set_xlabel('x (pixels)')
    ax_fov.set_ylabel('y (pixels)')
    ax_fov.set_title(f'{title}: channel 2 mean image, {int(table.gp_tuned.sum())}/{len(table)} GP-tuned cells '
                     f'(Bonferroni)\nlabels: cell_id (x, y); + is the table centroid')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _cli():
    parser = argparse.ArgumentParser(description='Overview figure of the GP cell table of one session.')
    parser.add_argument('mouse', help='e.g. "001"')
    parser.add_argument('session', help='session tag to match, e.g. "seq3-007"')
    parser.add_argument('--project-root', default=h.PROJECT_ROOT, help='defaults to the holography cohort1 folder')
    parser.add_argument('--exclude', type=int, nargs='*', default=[], help='cell IDs not to highlight, e.g. --exclude 56')
    args = parser.parse_args()
    project_root = ceph_path(args.project_root)
    folder_name, plane = gct.suite2p_plane(args.mouse, args.session, project_root)
    out_dir = Path(project_root) / 'analysis' / f'sub-{args.mouse}' / gct.session_label(folder_name)
    table = pd.read_csv(out_dir / 'gp_cell_table.csv')
    ops = np.load(plane / 'ops.npy', allow_pickle=True).item()
    stat = np.load(plane / 'stat.npy', allow_pickle=True)
    loaded = h.load_session(args.mouse, args.session, behav_root=f'{project_root}/BehaviourAligned_robust')
    if not np.array_equal(table.cell_id.to_numpy(), loaded['neurons']):
        raise ValueError('Table cell_ids differ from the accepted ROIs in the NWB -- rerun gp_cell_table')
    mat = h.gp_tuning(loaded, clock='reward', bins=gct.GP_BINS, n_shuffles=0)['mat']
    curves = h.mean_curve(mat)
    sem = np.nanstd(mat, axis=1) / np.sqrt(np.sum(np.isfinite(mat), axis=1))
    plot_overview(table, ops, stat, curves, sem, f'sub-{args.mouse} {gct.session_label(folder_name)}',
                  out_dir / 'gp_cell_table_overview.png', exclude=args.exclude)
    print(f'Saved {out_dir / "gp_cell_table_overview.png"}')


if __name__ == '__main__':
    _cli()

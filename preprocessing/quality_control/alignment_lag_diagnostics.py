"""Read-only, alignment-agnostic check of behaviour <-> imaging alignment via reward-triggered lag scans.

Written while debugging the holography cohort (SaReSuKuMaRa_20260911_holography/cohort1), where most
sessions showed no tuning at all. Two independent alignment bugs turned out to be responsible:
 - suite2p had concatenated short ScanImage test acquisitions in front of the main one, so
   Ca_traces/dF were shifted by 2000-4000 frames (44-89 s) relative to valid_frames;
 - build_behaviour_data's `harp_seconds + first camera TTL` anchor put behaviour 0.15-10.5 s late
   (see nwb_barcoding/build_behaviour_data.py::fit_analog_clock).

Rather than trusting any pipeline bookkeeping, this shifts the reward vector against the
Ca traces from behaviour_data.npz and asks at which lag the population shows the strongest
reward-locked response. With correct alignment, that peak sits near 0 s; a pure frame
offset shows up as a peak at -offset. Rewards recur every lap, so smaller side peaks at
multiples of the lap period are expected -- it's the location of the main peak that matters.

Also plots a cross-validated peri-reward heatmap (sort on odd rewards, display even rewards)
at lag 0 and at the best lag, so a session that only shows structure at the best lag stands out.

Only reads behaviour_data.npz; writes PNGs + summary.csv to --out (never into the data tree).

Usage (bonsai_abcd env, from repo root):
    python -m preprocessing.quality_control.alignment_lag_diagnostics <session_dir> [<session_dir> ...] \\
        --out ~/Desktop/diagnostics/holography_alignment
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

import era_brewer

FRAME_RATE = 45
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    'NineteenEightyNine_r', era_brewer.era_brew('NineteenEightyNine', n=256, brew_type='continuous')[::-1])
LINE_COLORS = era_brewer.era_brew('Lover1', n=4, brew_type='discrete')


def _zscore_traces(ca, smooth_sigma_frames=2):
    ca = gaussian_filter1d(ca.astype(np.float32), smooth_sigma_frames, axis=1)
    sd = ca.std(axis=1, keepdims=True)
    sd[sd == 0] = 1
    return (ca - ca.mean(axis=1, keepdims=True)) / sd


def peri_event_average(z, event_frames, window_frames):
    """Mean trace per cell around each event: (cells x window) plus the (events x cells x window) stack."""
    offsets = np.arange(-window_frames, window_frames + 1)
    keep = (event_frames - window_frames >= 0) & (event_frames + window_frames < z.shape[1])
    idx = event_frames[keep][:, None] + offsets[None, :]
    stack = z[:, idx].transpose(1, 0, 2)
    return stack.mean(axis=0), stack


def lag_scan(z, reward_frames, max_lag_s=120, step_s=0.5, window_s=3):
    """Reward-locked modulation (mean over cells of the variance of the trial-averaged peri-reward
    trace) as a function of the lag added to the reward frames."""
    window = int(window_s * FRAME_RATE)
    lags = np.arange(-max_lag_s, max_lag_s + step_s / 2, step_s)
    # Same reward set at every lag: otherwise rewards drop out near the scan edges, the average
    # gets noisier, and its variance (the score) drifts up -- a weakly tuned session (GoalProgress
    # sub-002 t1) then "peaks" at the scan edge instead of at its real, sharp peak at 0.
    margin = int(max_lag_s * FRAME_RATE) + window
    reward_frames = reward_frames[(reward_frames >= margin) & (reward_frames < z.shape[1] - margin)]
    score = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        shifted = reward_frames + int(round(lag * FRAME_RATE))
        avg, stack = peri_event_average(z, shifted, window)
        if stack.shape[0] >= 5:
            score[i] = avg.var(axis=1).mean()
    return lags, score


def _cv_heatmap(ax, z, reward_frames, lag_s, window_s=5, title=''):
    window = int(window_s * FRAME_RATE)
    frames = reward_frames + int(round(lag_s * FRAME_RATE))
    _, stack = peri_event_average(z, frames, window)
    sort_avg, show_avg = stack[0::2].mean(axis=0), stack[1::2].mean(axis=0)
    order = np.argsort(np.argmax(sort_avg, axis=1))
    t = np.arange(-window, window + 1) / FRAME_RATE
    ax.imshow(show_avg[order], aspect='auto', cmap=HEATMAP_CMAP, vmin=-1, vmax=1.5,
              extent=[t[0], t[-1], len(order), 0], interpolation='none')
    ax.axvline(0, color='w', lw=0.8, ls='--')
    ax.set_xlabel('Time from reward (s)')
    ax.set_ylabel('Cell (sorted on odd rewards)')
    ax.set_title(title, fontsize=9)


def diagnose_session(session_dir, out_dir):
    session_dir = Path(session_dir)
    label = f"{session_dir.parent.name}_{session_dir.name.split('_')[0]}_{session_dir.name.split('-')[-1]}"
    data = np.load(session_dir / 'behaviour_data.npz')
    z = _zscore_traces(data['Ca_traces'])
    reward_frames = np.where(data['rewards'] > 0)[0]
    lags, score = lag_scan(z, reward_frames)
    best_lag = lags[np.nanargmax(score)]
    zero_score = score[np.argmin(np.abs(lags))]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    axes[0].plot(lags, score, color=LINE_COLORS[1], lw=1.2)
    axes[0].axvline(0, color='k', lw=0.8, ls='--')
    axes[0].axvline(best_lag, color=LINE_COLORS[0], lw=1.2)
    axes[0].set_xlabel('Lag added to reward frames (s)')
    axes[0].set_ylabel('Reward-locked modulation')
    axes[0].set_title(f'{label}: best lag {best_lag:+.1f} s, peak/zero-lag = {np.nanmax(score) / zero_score:.2f}',
                      fontsize=10)
    _cv_heatmap(axes[1], z, reward_frames, 0, title='Peri-reward, lag 0 (even rewards)')
    _cv_heatmap(axes[2], z, reward_frames, best_lag, title=f'Peri-reward, lag {best_lag:+.1f} s (even rewards)')
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f'{label}_lagscan.png', dpi=130)
    plt.close(fig)

    print(f"{label}: {z.shape[0]} cells, {len(reward_frames)} rewards, best lag {best_lag:+.1f} s, "
          f"peak/zero-lag {np.nanmax(score) / zero_score:.2f}")
    return dict(session=label, n_cells=z.shape[0], n_frames=z.shape[1], n_rewards=len(reward_frames),
                best_lag_s=best_lag, peak_over_zero_lag=np.nanmax(score) / zero_score)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dirs', nargs='+')
    parser.add_argument('--out', default=str(Path.home() / 'Desktop' / 'diagnostics' / 'holography_alignment'))
    args = parser.parse_args()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in args.session_dirs:
        if not (Path(s) / 'behaviour_data.npz').exists():
            print(f"skipping {s}: no behaviour_data.npz")
            continue
        rows.append(diagnose_session(s, out_dir))
    summary_path = out_dir / 'summary.csv'
    df = pd.DataFrame(rows)
    if summary_path.exists():  # accumulate across runs (one session per run on the share)
        old = pd.read_csv(summary_path)
        df = pd.concat([old[~old['session'].isin(df['session'])], df], ignore_index=True)
    df.sort_values('session').to_csv(summary_path, index=False)
    print(f"Saved plots + {summary_path}")


if __name__ == '__main__':
    main()

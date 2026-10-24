r"""Teaching figure: what assuming a nominal sampling rate does to behaviour <-> imaging alignment.

Made for the pipeline-overview slides (clock-syncing slide). The point: rates printed on the
hardware/config are nominal; measure them from the recorded timestamps instead.

Two nominal rates are compared on one real session:
 - Imaging frame rate. The repo's default is 45 Hz, but the vDAQ FrameTTLs show ~44.70 Hz
   (median interval 22.4 ms, no dropped frames; checked on GoalProgress sub-002 t2 and cohort3
   sub-010 full037). Converting frame index -> time as i / 45 is therefore ~0.7 % too fast,
   which accumulates to ~12 s by the end of a 30-min session.
 - NIDAQ sample rate. Aimbot_barcoding hardcodes 9000 Hz and aligns offset-only. The barcode fit
   (nwb_barcoding/barcode_alignment.py::fit_alignment) gives ~8999.97 Hz (-1 to -4 ppm) on the
   GoalProgress and cohort3 sessions checked in Oct 2026, i.e. only ~3-8 ms over a session. It is
   plotted alongside for scale. (nwb_barcoding's docstrings quote ~111 ppm / 110-270 ms; that did
   not reproduce in these fits.)

Panel A: timing error vs session time for both assumptions.
Panel B: event-triggered activity for early vs late events, with event -> frame mapping done
either from the real frame timestamps or as (t - t_first_frame) * 45. Events are rewards
(--event reward, default; taken from behaviour_data.npz, since rewards are not NIDAQ TTLs on this
rig) or landmark onsets (--event landmark; MODD1ON/MODD2ON TTLs put on the imaging clock with the
barcode fit). Cells are selected on early events with the real timestamps, where both mappings
still agree most, so the selection does not favour either.

Reuses barcode_alignment.read_syncttl/fit_alignment, verify_session._get_ttl and
alignment_lag_diagnostics' z-scoring/peri-event helpers. Reads the behav + funcimg NWBs of ONE
session from a BehaviourAligned_robust-style tree (read-only); writes PNG + SVG to --out.

Usage (2p_preprocess_extraction env, from repo root):
    python -m preprocessing.quality_control.plot_clock_drift 001 t3 \
        /Volumes/mrsic_flogel/public/projects/SaReSuKuMaRa_20260911_holography/cohort1/BehaviourAligned_robust \
        --out ~/Desktop/clock_drift
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO

import era_brewer
from nwb_barcoding import barcode_alignment as ba
from nwb_barcoding.verify_session import _get_ttl
from preprocessing.quality_control.alignment_lag_diagnostics import _zscore_traces, peri_event_average

NOMINAL_FRAME_RATE = 45.0
NOMINAL_NIDAQ_RATE = 9000.0
# Lover1 discrete: pink, blue, pale yellow, brown
COLORS = era_brewer.era_brew('Lover1', n=4, brew_type='discrete')
C_WRONG, C_RIGHT, C_NIDAQ = COLORS[0], COLORS[1], COLORS[3]


def find_session(mouse, session_tag, root):
    matches = sorted((Path(root) / f'sub-{mouse}').glob(f'ses-*{session_tag}*'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one session for sub-{mouse} {session_tag} in {root}, found {matches}')
    session_path = matches[0]
    behav_nwb = list((session_path / 'behav').glob('vrabcd_preprocessed_*.nwb'))[0]
    funcimg_nwb = list((session_path / 'funcimg').glob('2p_preprocessed_*.nwb'))[0]
    return session_path, behav_nwb, funcimg_nwb


def load_imaging(funcimg_nwb):
    """Frame timestamps (vDAQ clock, s) and dF/F as (cells x frames)."""
    with NWBHDF5IO(str(funcimg_nwb), mode='r') as io:
        nwb = io.read()
        types = nwb.acquisition['vDAQ_TTLTypesTable'][:]
        ttls = nwb.acquisition['vDAQ_TTLsTable'][:]
        frame_types = types[types['event_name'] == 'FrameTTL'].index.tolist()
        if frame_types:
            ttls = ttls.loc[ttls['ttl_type'].isin(frame_types)]
        frame_times = ttls['timestamp'].values
        dff_interface = nwb.processing['ophys']['DfOverF']
        try:
            series = dff_interface['DfOverFChan1Plane0']
        except KeyError:
            series = dff_interface['DfOverF']
        dff = np.asarray(series.data[:], dtype=np.float32).T
    n = min(len(frame_times), dff.shape[1])
    if len(frame_times) != dff.shape[1]:
        print(f'[note] {len(frame_times)} frame TTLs vs {dff.shape[1]} dF/F frames; using first {n}.')
    return frame_times[:n], np.nan_to_num(dff[:, :n])


def event_response(z, frames, window, rate):
    """Mean z in the 1 s after the event minus the 1 s before, per cell."""
    avg, _ = peri_event_average(z, frames, window)
    one_s = int(rate)
    return avg[:, window:window + one_s].mean(axis=1) - avg[:, window - one_s:window].mean(axis=1)


def make_figure(mouse, session_tag, root, out_dir, event='reward', window_s=3, top_frac=0.1):
    session_path, behav_nwb, funcimg_nwb = find_session(mouse, session_tag, root)
    print(f'Session: {session_path}')

    # --- Imaging clock: real frame timestamps vs i / 45 ---
    frame_times, dff = load_imaging(funcimg_nwb)
    t0 = frame_times[0]
    frame_idx = np.arange(len(frame_times))
    measured_rate = (len(frame_times) - 1) / (frame_times[-1] - t0)
    frame_err_s = (t0 + frame_idx / NOMINAL_FRAME_RATE) - frame_times
    print(f'Frame rate: measured {measured_rate:.3f} Hz vs nominal {NOMINAL_FRAME_RATE:.0f} Hz; '
          f'i/{NOMINAL_FRAME_RATE:.0f} error at end {frame_err_s[-1]:+.2f} s')

    # --- NIDAQ clock: offset-only (hardcoded 9000 Hz) vs offset + slope ---
    fit = ba.fit_alignment(main_decoded=ba.read_syncttl(funcimg_nwb),
                           secondary_decoded=ba.read_syncttl(behav_nwb),
                           secondary_nominal_rate=NOMINAL_NIDAQ_RATE)
    match, slope, delta_t = fit['match'], fit['slope'], fit['delta_t']
    nidaq_err_s = (match['secondary_times'] + delta_t) - match['main_times']
    print(f'NIDAQ: measured {fit["corrected_secondary_rate"]:.2f} Hz ({(1 / slope - 1) * 1e6:+.1f} ppm); '
          f'offset-only error up to {1000 * np.abs(nidaq_err_s).max():.1f} ms')

    # --- Events on the imaging clock ---
    if event == 'reward':
        # Rewards are not NIDAQ TTLs on this rig; take them from behaviour_data.npz, where
        # build_behaviour_data already binned them onto imaging frames (rewards[i] = 1 means a
        # reward between valid_frames[i] and valid_frames[i + 1]).
        rewards = np.load(session_path / 'behaviour_data.npz')['rewards']
        valid_frames = np.load(session_path / 'valid_frames.npz')['valid_frames']
        ev_frames = valid_frames[np.nonzero(rewards)[0]]
        ev_t = frame_times[ev_frames[ev_frames < len(frame_times)]]
    else:
        with NWBHDF5IO(str(behav_nwb), mode='r') as io:
            nwb = io.read()
            lm_native = np.sort(np.concatenate([_get_ttl(nwb, 'MODD1ONTTL'), _get_ttl(nwb, 'MODD2ONTTL')]))
        ev_t = lm_native * slope + delta_t
    ev_t = ev_t[(ev_t > t0) & (ev_t < frame_times[-1])]
    f_right = np.clip(np.searchsorted(frame_times, ev_t), 0, len(frame_times) - 1)
    f_wrong = np.round((ev_t - t0) * NOMINAL_FRAME_RATE).astype(int)
    n_ev = len(ev_t)
    early, late = slice(0, n_ev // 3), slice(n_ev - n_ev // 3, n_ev)
    late_shift_s = np.mean((f_wrong - f_right)[late]) / measured_rate
    print(f'{n_ev} {event} events; late-third events misplaced by {late_shift_s:.1f} s on average with i/45')

    z = _zscore_traces(dff)
    window = int(round(window_s * measured_rate))
    resp = event_response(z, f_right[early], window, measured_rate)
    n_top = max(5, int(top_frac * len(resp)))
    cells = np.argsort(resp)[::-1][:n_top]
    print(f'Using top {n_top}/{len(resp)} {event}-responsive cells (selected on early {event}s)')
    t_win = np.arange(-window, window + 1) / measured_rate

    # --- Plot ---
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.35)

    ax_a = fig.add_subplot(gs[0])
    t_min = (frame_times - t0) / 60
    ax_a.plot(t_min, frame_err_s, color=C_WRONG, lw=2.5,
              label=f'index / {NOMINAL_FRAME_RATE:.0f} Hz (true {measured_rate:.2f} Hz)')
    ax_a.plot((match['main_times'] - t0) / 60, nidaq_err_s, color=C_NIDAQ, lw=2.5,
              label=f'NIDAQ {NOMINAL_NIDAQ_RATE:.0f} Hz (true {fit["corrected_secondary_rate"]:.2f} Hz)')
    ax_a.axhline(0, color=C_RIGHT, lw=1.5, ls='--', label='measured timestamps')
    ax_a.annotate(f'{frame_err_s[-1]:+.1f} s\n≈ {abs(frame_err_s[-1]) * measured_rate:.0f} frames',
                  xy=(t_min[-1], frame_err_s[-1]), xytext=(-8, 0), textcoords='offset points',
                  ha='right', va='center', color=C_WRONG, fontsize=11, fontweight='bold')
    ax_a.annotate(f'{1000 * nidaq_err_s[-1]:+.0f} ms', xy=((match['main_times'][-1] - t0) / 60, nidaq_err_s[-1]),
                  xytext=(-8, 8), textcoords='offset points', ha='right', color=C_NIDAQ, fontsize=10)
    ax_a.set_xlabel('Time in session (min)')
    ax_a.set_ylabel('Timing error (s)')
    ax_a.set_title('A   Assuming nominal rates: error accumulates', loc='left', fontsize=12)
    ax_a.legend(frameon=False, fontsize=9, loc='lower left')

    axes_b = [fig.add_subplot(gs[1])]
    axes_b.append(fig.add_subplot(gs[2], sharey=axes_b[0]))
    for ax, frames, label, color in [(axes_b[0], f_right, 'measured frame times', C_RIGHT),
                                      (axes_b[1], f_wrong, f'index / {NOMINAL_FRAME_RATE:.0f} Hz', C_WRONG)]:
        for sl, name, ls in [(early, 'first third', '-'), (late, 'last third', ':')]:
            _, stack = peri_event_average(z[cells], frames[sl], window)
            per_event = stack.mean(axis=1)
            mean = per_event.mean(axis=0)
            sem = per_event.std(axis=0) / np.sqrt(len(per_event))
            ax.fill_between(t_win, mean - sem, mean + sem, color=color, alpha=0.25, lw=0)
            ax.plot(t_win, mean, color=color, lw=2, ls=ls, label=f'{name} of session (n={len(per_event)})')
            print(f'  {label:>22}, {name}: peak {mean.max():.2f} z at {t_win[np.argmax(mean)]:+.2f} s')
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.set_xlabel(f'Time from {event} (s)')
        ax.set_title(label, fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc='upper left')
    axes_b[0].set_ylabel(f'{event.capitalize()} cells, z-scored dF/F (n={n_top})')
    axes_b[1].text(0.98, 0.02, f'late {event}s misplaced\nby {late_shift_s:.1f} s on average',
                   transform=axes_b[1].transAxes, ha='right', va='bottom', fontsize=9, color=C_WRONG)
    plt.setp(axes_b[1].get_yticklabels(), visible=False)
    fig.text(0.475, 0.95, 'B   Consequence for neural data', fontsize=12)

    for ax in [ax_a, *axes_b]:
        ax.spines[['top', 'right']].set_visible(False)

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f'clock_drift_sub-{mouse}_{session_tag}_{event}'
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight')
    print(f'Saved {stem}.png/.svg')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mouse', help="e.g. 002 (without 'sub-')")
    p.add_argument('session_tag', help='e.g. t2 or full037')
    p.add_argument('root', help='BehaviourAligned_robust-style tree containing sub-<mouse>/<session>')
    p.add_argument('--event', choices=['reward', 'landmark'], default='reward')
    p.add_argument('--out', default='~/Desktop/clock_drift')
    args = p.parse_args()
    make_figure(args.mouse, args.session_tag, args.root, args.out, event=args.event)


if __name__ == '__main__':
    main()

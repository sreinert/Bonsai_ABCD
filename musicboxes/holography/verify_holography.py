"""Sanity checks for a holography session built by nwb_holography.py.

Same style as nwb_barcoding/verify_session.py (whose barcode checks it reuses): each check prints
its numbers and adds '[FLAG] ...' lines for anything suspicious; verify() returns the flag list.
Reads only holo_alignment.npz, photostim_trials.csv, the raw protocol files and (optionally) the NWB.

    python -m musicboxes.holography.verify_holography 001 holo-001
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ceph_paths import ceph_path
from nwb_barcoding.verify_session import _flag, check_barcode_quality, check_fit_quality
import musicboxes.holography.holo_io as hio


def check_frames(a, n_frames):
    flags = []
    ft = a['frame_times']
    ttl_iri = np.diff(ft[:int(a['n_frame_ttl'])])
    med = np.median(ttl_iri)
    print(f"Frames: {n_frames} imaging frames, {int(a['n_frame_ttl'])} FrameTTL rises, "
          f"{int(a['n_frames_extrapolated'])} extrapolated at the end; rate {1 / med:.3f} Hz, "
          f"IRI {1000 * ttl_iri.min():.2f}-{1000 * ttl_iri.max():.2f} ms; vDAQ recording ends "
          f"{1000 * float(a['recording_end_after_last_frame_s']):.1f} ms after the last FrameTTL")
    if len(ft) != n_frames:
        _flag(flags, f'{len(ft)} frame times for {n_frames} frames')
    if np.any(np.abs(ttl_iri - med) > 0.05 * med):
        _flag(flags, f'{np.sum(np.abs(ttl_iri - med) > 0.05 * med)} FrameTTL intervals deviate > 5% from the median '
                     f'(dropped TTLs or pauses)')
    if int(a['n_frames_extrapolated']) > 0:
        _flag(flags, f"{int(a['n_frames_extrapolated'])} frames have no FrameTTL and were assumed to be the last "
                     f"ones (recorder stopped {1000 * float(a['recording_end_after_last_frame_s']):.1f} ms after the "
                     f"last TTL). If they were at the start instead, all frame times shift by that many frames.")
    return flags


def check_stimulation(a, trials, protocol):
    flags = []
    n = len(trials)
    npulse = trials.n_pulses.values
    dur = trials.offset_time.values - trials.onset_time.values
    slm_lag = 1000 * (trials.slm_time.values - trials.onset_time.values)
    print(f'Stimulation: {n} trains (csv {len(protocol)} rows); pulses per train {np.unique(npulse).tolist()}; '
          f'train duration {dur.min():.4f}-{dur.max():.4f} s; SLM pulse {np.nanmin(slm_lag):.1f}-'
          f'{np.nanmax(slm_lag):.1f} ms after the first gate pulse ({np.isnan(slm_lag).sum()} trains without one)')
    if n != len(protocol):
        _flag(flags, f'{n} trains vs {len(protocol)} csv rows')
    if len(np.unique(npulse)) > 1:
        _flag(flags, f'Trains have different pulse counts: {np.unique(npulse, return_counts=True)}')
    if np.isnan(slm_lag).any():
        _flag(flags, f'{np.isnan(slm_lag).sum()} trains have no SLMFeedbackTTL pulse within 10 ms')
    if len(a['slm_times']) != n:
        _flag(flags, f"{len(a['slm_times'])} SLMFeedbackTTL pulses for {n} trains")

    iti = np.diff(trials.onset_time.values)
    r = np.corrcoef(iti, protocol.isi_s.values[:-1])[0, 1]
    r_shift = np.corrcoef(iti[1:], protocol.isi_s.values[:-2])[0, 1]
    over = np.median(iti - protocol.isi_s.values[:-1])
    print(f'  ITI vs csv isi_s: r = {r:.4f} (shifted by one trial: {r_shift:.3f}), median overhead {1000 * over:.0f} ms')
    if r < 0.99:
        _flag(flags, f'Measured ITIs do not follow the csv isi_s (r = {r:.3f}): trial order may not match the TTL trains')

    per = trials.groupby('condition').size()
    print(f'  {len(per)} conditions ({int(trials.is_blank.any())} blank), trials per condition '
          f'{per.min()}-{per.max()} (median {per.median():.0f}); consecutive repeats '
          f'{int((trials.condition.values[1:] == trials.condition.values[:-1]).sum())}')
    if per.min() != per.max():
        _flag(flags, f'Unequal trial counts per condition: {per.value_counts().to_dict()}')

    nfr = trials.offset_frame.values - trials.onset_frame.values + 1
    print(f'  frames per train {nfr.min()}-{nfr.max()}; onset frames {trials.onset_frame.min()}-'
          f'{trials.onset_frame.max()} of {len(a["frame_times"])}')
    if (trials.onset_frame < 0).any() or (trials.offset_frame >= len(a['frame_times']) - 1).any():
        _flag(flags, 'Some trains fall outside the imaging period')
    gaps = trials.onset_frame.values[1:] - trials.offset_frame.values[:-1]
    print(f'  gap between consecutive trains: min {gaps.min()} frames')
    if gaps.min() < 45:
        _flag(flags, f'Trains closer than 1 s ({gaps.min()} frames): post-stim windows overlap the next train')
    return flags


def check_behaviour(a):
    flags = []
    ft, bt = a['frame_times'], a['behav_t']
    print(f'Behaviour DAQ covers {bt[0]:.2f} to {bt[-1]:.2f} s on the vDAQ clock; imaging {ft[0]:.2f} to {ft[-1]:.2f} s')
    if bt[0] > ft[0] or bt[-1] < ft[-1]:
        _flag(flags, 'Behaviour recording does not cover the whole imaging period')
    slope = float(a['fit_slope'])
    print(f"  barcode fit slope {slope:.8f} ({1e6 * (slope - 1):.1f} ppm), corrected behav rate "
          f"{float(a['fit_corrected_behav_rate']):.4f} Hz")
    if abs(slope - 1) > 1e-3:
        _flag(flags, f'Clock slope {slope:.6f} is > 0.1% from 1: check barcode decoding')

    cam = a['camera_times']
    in_img = cam[(cam >= ft[0]) & (cam <= ft[-1])]
    iri = np.diff(cam)
    med = np.median(iri)
    n_gap = int((iri > 1.5 * med).sum())
    print(f'  camera: {len(cam)} frames ({len(in_img)} during imaging), {1 / med:.2f} Hz, {n_gap} gaps > 1.5 periods')
    if n_gap > 0:
        _flag(flags, f'{n_gap} camera inter-frame intervals > 1.5x the median (dropped camera TTLs?)')

    sp = a['wheel_speed_frames']
    run = np.abs(sp) > 30
    print(f"  wheel: {float(a['wheel_deg'][-1] - a['wheel_deg'][0]) / 360:.0f} net revolutions; |speed| > 30 deg/s "
          f"in {100 * run.mean():.1f}% of frames; median |speed| {np.median(np.abs(sp)):.1f} deg/s")
    pd_ = a['photodiode']
    print(f'  photodiode {pd_.min():.2f}-{pd_.max():.2f} V; ai4 {a["ai4"].min():.3f}-{a["ai4"].max():.3f} V')
    return flags


def check_nwb(nwb_path, a, trials):
    flags = []
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(nwb_path), 'r') as io:
        nwb = io.read()
        F = nwb.processing['ophys']['Fluorescence']['F']
        ts = F.timestamps[:]
        ti = nwb.intervals['photostim_trials'].to_dataframe()
        ps = nwb.processing['ophys']['ImageSegmentation']
        print(f'NWB: F {F.data.shape}, {len(ti)} photostim trials, segmentations {list(ps.plane_segmentations)}, '
              f"dFF {nwb.processing['ophys']['DfOverF']['dFF'].data.shape}")
    if not np.allclose(ts, a['frame_times']):
        _flag(flags, 'NWB frame timestamps differ from holo_alignment.npz')
    if len(ti) != len(trials) or not np.allclose(ti.start_time.values, trials.onset_time.values):
        _flag(flags, 'NWB photostim_trials differ from photostim_trials.csv')
    return flags


def verify(mouse, holo_tag, project_root=hio.PROJECT_ROOT, holo_suffix='SR', nwb_path=None):
    paths = hio.session_paths(mouse, holo_tag, project_root, holo_suffix)
    a = dict(np.load(paths['output'] / 'holo_alignment.npz', allow_pickle=True))
    trials = pd.read_csv(paths['output'] / 'photostim_trials.csv')
    protocol, _ = hio.load_stim_protocol(paths['raw_funcimg'])
    n_frames = int(hio.load_ops(paths['plane'])['nframes'])

    flags = []
    print('--- Frames ---')
    flags += check_frames(a, n_frames)
    print('\n--- Stimulation ---')
    flags += check_stimulation(a, trials, protocol)
    print('\n--- Barcodes ---')
    vd = {'values': a['vdaq_barcode_values'], 'n_skipped': 0}
    bh = {'values': a['behav_barcode_values'], 'n_skipped': 0}
    flags += check_barcode_quality(vd, 'vDAQ', int(a['vdaq_n_samples']) / float(a['vdaq_rate']))
    flags += check_barcode_quality(bh, 'behav', int(a['behav_n_samples']) / float(a['behav_nominal_rate']))
    fit = {'match': {'shared_values': np.arange(int(a['fit_n_shared'])), 'residual_ms': a['fit_residual_ms'],
                     'n_outliers_excluded': int(a['fit_n_outliers'])},
           'delta_t': float(a['fit_delta_t']), 'corrected_secondary_rate': float(a['fit_corrected_behav_rate']),
           'slope': float(a['fit_slope'])}
    flags += check_fit_quality(fit)
    print('\n--- Behaviour ---')
    flags += check_behaviour(a)
    nwb_path = nwb_path or next(iter(sorted(paths['output'].glob('holo_2p_*.nwb'))), None)
    if nwb_path is not None:
        print('\n--- NWB ---')
        flags += check_nwb(nwb_path, a, trials)
    print(f'\n{len(flags)} issue(s) flagged.' if flags else '\nNo issues flagged.')
    return flags


def _cli():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('mouse')
    p.add_argument('holo_tag')
    p.add_argument('--project-root', default=str(hio.PROJECT_ROOT))
    p.add_argument('--holo-suffix', default='SR')
    a = p.parse_args()
    verify(a.mouse, a.holo_tag, ceph_path(a.project_root), a.holo_suffix)


if __name__ == '__main__':
    _cli()

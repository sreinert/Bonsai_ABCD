"""Build the aligned NWB for a holographic photostimulation session (imaging + stimulation + behaviour).

Run from the repo root in the bonsai_abcd env, one session per run (the share drops connections):

    python -m musicboxes.holography.nwb_holography 001 holo-001 seq3-007 [--force]

The third argument is the source session whose ROIs were targeted; its roi_match_<source>.csv
(match_rois.py) is used to add holo ROI ids to the trial table when it exists.

Why a standalone writer
-----------------------
The VR sessions go through nwb_barcoding/, which wraps Aimbot_barcoding's converters. For this
session those do not fit: Aimbot's WarpedVisualStim converter has no 'UniformContrast' stimulus,
its AimbotDataInterface requires exactly one phase_mask*.csv (holo-001 has two), and its photostim
table stores no stimulation times. Everything here is read with h5py/numpy and written with pynwb,
so it runs in bonsai_abcd and on the HPC venv alike.

Clocks
------
Everything is on the vDAQ (ScanImage recorder) clock, t = sample / 5000, with t = 0 at the start of
the <tif stem>.h5 recording:
- Imaging frames: FrameTTL rising edges. For holo-001 the TTL count (269,997) is 4 short of the
  frames suite2p registered (270,001). The vDAQ recording starts 5.7 s before the first frame but
  ends 22.6 ms (one frame period) after the last FrameTTL rise, so the untagged frames are taken to
  be the last ones, acquired after the recorder had stopped. Their times are extrapolated with the
  median frame period, and n_frames_extrapolated is saved. There is no light artefact from the
  spirals in the ROI traces that could test this independently. The alternative (missing frames at
  the start) would shift every frame by 4 frames, 90 ms.
- Stimulation: SatsumaGateTTL rising edges (laser gate, 2 ms pulses) grouped into trains by gaps
  > 1 s. Each train must have one SLMFeedbackTTL pulse (phase-mask update) within 10 ms of its
  first gate pulse, and train i is row i of photostim_loader_*.csv, which is checked via the ITIs.
  A trial's onset is the first gate pulse, its offset the end of the last one.
- Behaviour: the WarpedVisualStim DAQ logger (5 kHz nominal) shares the barcode with the vDAQ
  (vDAQ SyncTTL vs behav ai0). nwb_barcoding.barcode_alignment.fit_alignment gives slope + offset,
  t_vdaq = slope * t_behav_nominal + delta_t. The slope is applied, not only the offset, for the
  reason given in that module.

Behav channels (identified from the signals; the settings file only lists names):
    ai0  barcode            ai3  photodiode (stays near one level during UniformContrast)
    ai4  unidentified analog signal, stored raw
    ai13 camera frame TTL (pupil cameras)
    ctr0 cumulative wheel angle in degrees (steps of 360/4096). Speed is in deg/s; no cm conversion.

Steps (each skipped if its output exists, unless --force)
-------------------------------------------------------
1. dF/F: nwb_barcoding.compute_dff.ensure_dff on the holo suite2p folder (F - 0.7 Fneu, suite2p
   baseline), writing <preprocessed>/funcimg/dff/plane0/dFF.npy as for the VR sessions.
2. Alignment: parse the vDAQ and behav DAQ, fit the barcodes, and write holo_alignment.npz
   (everything an analysis needs without pynwb; see build_alignment for the keys).
3. NWB: holo_2p_<sub>_<ses>.nwb in the output folder: mean images, both segmentations (holo ROIs,
   projected source masks if extract_projected.py has run), F/Fneu/dF/F with frame timestamps,
   the photostim_trials table, and behaviour TimeSeries.
4. verify_holography.verify: sanity checks, printed as [FLAG] lines.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import pandas as pd

from ceph_paths import ceph_path
from nwb_barcoding import barcode_alignment as ba
from nwb_barcoding.compute_dff import ensure_dff
import musicboxes.holography.holo_io as hio

VDAQ_RATE = 5000.0
BEHAV_NOMINAL_RATE = 5000.0
TRAIN_GAP_S = 1.0
SLM_MAX_LAG_S = 0.010
BEHAV_DOWNSAMPLE = 5            # behav analog channels stored at 1 kHz
WHEEL_SMOOTH_S = 0.1            # boxcar for wheel speed


def _rising(x, threshold):
    b = np.asarray(x) > threshold
    return np.flatnonzero(np.diff(b.astype(np.int8)) == 1) + 1


def _falling(x, threshold):
    b = np.asarray(x) > threshold
    return np.flatnonzero(np.diff(b.astype(np.int8)) == -1) + 1


def _threshold(x):
    lo, hi = np.percentile(x, [1, 99.9])
    return (lo + hi) / 2


# ---------------------------------------------------------------- vDAQ

def parse_vdaq(h5_path, n_frames):
    """Frame times, stimulation trains and barcodes from the ScanImage vDAQ recorder."""
    with h5py.File(h5_path, 'r') as h:
        rate = float(h.attrs['samplerate'])
        frame = h['FrameTTL'][:]
        gate = h['SatsumaGateTTL'][:]
        slm = h['SLMFeedbackTTL'][:]
        sync = h['SyncTTL'][:]
    n_samples = len(frame)

    fr = _rising(frame, _threshold(frame))
    period = np.median(np.diff(fr)) / rate
    n_ttl = len(fr)
    if n_ttl > n_frames:
        raise ValueError(f'{n_ttl} FrameTTL rises but only {n_frames} imaging frames')
    frame_times = np.concatenate([fr / rate, fr[-1] / rate + period * np.arange(1, n_frames - n_ttl + 1)])

    th = _threshold(gate)
    g_on, g_off = _rising(gate, th), _falling(gate, th)
    g_off = g_off[g_off > g_on[0]][:len(g_on)]
    split = np.flatnonzero(np.diff(g_on) > TRAIN_GAP_S * rate) + 1
    trains_on, trains_off = np.split(g_on, split), np.split(g_off, split)
    slm_on = _rising(slm, _threshold(slm))

    sync_digital = (sync > _threshold(sync)).astype(np.float64)
    barcodes = ba.decode_barcode(sync_digital, rate, height=0.5)
    barcodes['sample_rate'] = rate
    barcodes['n_samples'] = n_samples

    return {
        'rate': rate, 'n_samples': n_samples, 'frame_ttl_samples': fr, 'frame_times': frame_times,
        'frame_period': period, 'n_frame_ttl': n_ttl, 'n_frames_extrapolated': n_frames - n_ttl,
        'recording_end_after_last_frame_s': (n_samples - fr[-1]) / rate,
        'gate_on_times': g_on / rate, 'gate_off_times': g_off / rate,
        'train_onset': np.array([t[0] for t in trains_on]) / rate,
        'train_offset': np.array([t[-1] for t in trains_off]) / rate,
        'train_n_pulses': np.array([len(t) for t in trains_on]),
        'slm_times': slm_on / rate, 'barcodes': barcodes,
    }


def stim_trials(vdaq, trials_csv, conditions, frame_times):
    """Trial table on the vDAQ clock: CSV row i = gate train i (checked in verify_holography)."""
    n = len(vdaq['train_onset'])
    if n != len(trials_csv):
        raise ValueError(f'{n} stimulation trains in FrameTTL h5 but {len(trials_csv)} rows in photostim_loader csv')
    on, off = vdaq['train_onset'], vdaq['train_offset']
    slm = vdaq['slm_times']
    j = np.clip(np.searchsorted(slm, on - SLM_MAX_LAG_S), 0, len(slm) - 1)
    slm_time = np.where(np.abs(slm[j] - on) <= SLM_MAX_LAG_S, slm[j], np.nan)

    t = trials_csv.copy()
    t['onset_time'], t['offset_time'] = on, off
    t['n_pulses'] = vdaq['train_n_pulses']
    t['slm_time'] = slm_time
    # frame i spans [frame_times[i], frame_times[i+1]); stim frames are those overlapping the train
    t['onset_frame'] = np.searchsorted(frame_times, on, side='right') - 1
    t['offset_frame'] = np.searchsorted(frame_times, off, side='right') - 1
    cond = conditions.set_index('condition')
    t['is_blank'] = t.condition.map(cond['is_blank'])
    for c in ['target_x', 'target_y', 'power_V']:
        t[c] = t.condition.map(cond[c])
    return t


# ---------------------------------------------------------------- behaviour DAQ

def parse_behav(raw_behav):
    """Barcodes, camera frames, photodiode, ai4 and wheel from the WarpedVisualStim DAQ logger,
    on its own nominal clock (seconds = sample / 5000)."""
    settings = json.loads(hio._one(Path(raw_behav) / '*_settings.json').read_text())
    rate = float(settings.get('ds_daqlogger_sample_rate', BEHAV_NOMINAL_RATE))
    load = lambda ch: np.load(hio.behav_channel_path(raw_behav, ch), mmap_mode='r')

    ai0 = np.asarray(load('ai0'))
    barcodes = ba.decode_barcode((ai0 > _threshold(ai0)).astype(np.float64), rate, height=0.5)
    barcodes['sample_rate'] = rate
    barcodes['n_samples'] = len(ai0)
    del ai0

    ai13 = np.asarray(load('ai13'))
    camera = _rising(ai13, _threshold(ai13)) / rate
    del ai13

    ds = BEHAV_DOWNSAMPLE
    def downsample(x):
        x = np.asarray(x)
        n = len(x) // ds * ds
        return x[:n].reshape(-1, ds).mean(1).astype(np.float32)
    photodiode = downsample(load('ai3'))
    ai4 = downsample(load('ai4'))
    wheel = np.asarray(load('ctr0'))[::ds].astype(np.float64)[:len(ai4)]

    return {'rate': rate, 'n_samples': barcodes['n_samples'], 'settings': settings, 'barcodes': barcodes,
            'camera_times': camera, 'ds_rate': rate / ds, 'photodiode': photodiode, 'ai4': ai4,
            'wheel_deg': wheel}


def wheel_speed(wheel_deg, rate, smooth_s=WHEEL_SMOOTH_S):
    """deg/s from cumulative angle: gradient of the boxcar-smoothed angle."""
    k = max(int(round(smooth_s * rate)), 1)
    sm = np.convolve(wheel_deg, np.ones(k) / k, mode='same')
    return np.gradient(sm) * rate


# ---------------------------------------------------------------- step 2

def build_alignment(paths, force=False):
    """Parse both DAQs, fit the barcodes and save holo_alignment.npz + photostim_trials.csv.

    npz keys: frame_times, frame_period, n_frame_ttl, n_frames_extrapolated, frame_ttl_offset,
    trial_* (onset/offset time and frame, slm_time, n_pulses), gate_on_times, gate_off_times,
    slm_times, camera_times, behav_t (aligned times of the 1 kHz behav samples), photodiode, ai4,
    wheel_deg, wheel_speed_deg_s, and *_frames versions of wheel/ai4/photodiode at imaging
    frames; barcode fit: fit_slope, fit_delta_t, fit_residual_ms, fit_corrected_behav_rate.
    """
    out = paths['output']
    npz_path, csv_path = out / 'holo_alignment.npz', out / 'photostim_trials.csv'
    if npz_path.exists() and csv_path.exists() and not force:
        print(f'Alignment exists, skipping: {npz_path}')
        return dict(np.load(npz_path, allow_pickle=True)), pd.read_csv(csv_path)

    ops = hio.load_ops(paths['plane'])
    n_frames = int(ops['nframes'])
    trials_csv, conditions = hio.load_stim_protocol(paths['raw_funcimg'])

    print('Parsing vDAQ ...')
    vdaq = parse_vdaq(hio.vdaq_h5_path(paths['raw_funcimg']), n_frames)
    print(f"  {vdaq['n_frame_ttl']} FrameTTL rises for {n_frames} frames ({vdaq['n_frames_extrapolated']} "
          f"extrapolated at the end), period {1000 * vdaq['frame_period']:.3f} ms; "
          f"{len(vdaq['train_onset'])} stim trains; {len(vdaq['barcodes']['values'])} barcodes")
    trials = stim_trials(vdaq, trials_csv, conditions, vdaq['frame_times'])

    print('Parsing behaviour DAQ (5 channels over the share, takes a few minutes) ...')
    behav = parse_behav(paths['raw_behav'])
    fit = ba.fit_alignment(main_decoded=vdaq['barcodes'], secondary_decoded=behav['barcodes'],
                           secondary_nominal_rate=behav['rate'])
    m = fit['match']
    print(f"  {len(behav['barcodes']['values'])} barcodes; fit on {len(m['shared_values'])} shared: "
          f"slope {fit['slope']:.8f}, delta_t {fit['delta_t']:.4f} s, max|residual| "
          f"{np.abs(m['residual_ms']).max():.2f} ms")
    to_vdaq = lambda t_behav: t_behav * fit['slope'] + fit['delta_t']

    behav_t = to_vdaq(np.arange(len(behav['ai4'])) / behav['ds_rate'])
    speed = wheel_speed(behav['wheel_deg'], behav['ds_rate'])
    at_frames = lambda x: np.interp(vdaq['frame_times'], behav_t, x).astype(np.float32)

    out.mkdir(parents=True, exist_ok=True)
    data = {
        'frame_times': vdaq['frame_times'], 'frame_period': vdaq['frame_period'],
        'n_frame_ttl': vdaq['n_frame_ttl'], 'n_frames_extrapolated': vdaq['n_frames_extrapolated'],
        'frame_ttl_offset': 0, 'recording_end_after_last_frame_s': vdaq['recording_end_after_last_frame_s'],
        'vdaq_n_samples': vdaq['n_samples'], 'vdaq_rate': vdaq['rate'],
        'trial_condition': trials.condition.values.astype(str), 'trial_source_roi': trials.source_roi.values,
        'trial_onset_time': trials.onset_time.values, 'trial_offset_time': trials.offset_time.values,
        'trial_onset_frame': trials.onset_frame.values, 'trial_offset_frame': trials.offset_frame.values,
        'trial_slm_time': trials.slm_time.values, 'trial_n_pulses': trials.n_pulses.values,
        'gate_on_times': vdaq['gate_on_times'], 'gate_off_times': vdaq['gate_off_times'],
        'slm_times': vdaq['slm_times'],
        'vdaq_barcode_times': vdaq['barcodes']['start_sample'] / vdaq['rate'],
        'vdaq_barcode_values': vdaq['barcodes']['values'],
        'behav_barcode_times': behav['barcodes']['start_sample'] / behav['rate'],
        'behav_barcode_values': behav['barcodes']['values'],
        'behav_n_samples': behav['n_samples'], 'behav_nominal_rate': behav['rate'],
        'fit_slope': fit['slope'], 'fit_delta_t': fit['delta_t'], 'fit_residual_ms': m['residual_ms'],
        'fit_corrected_behav_rate': fit['corrected_secondary_rate'],
        'fit_n_shared': len(m['shared_values']), 'fit_n_outliers': m['n_outliers_excluded'],
        'camera_times': to_vdaq(behav['camera_times']),
        'behav_t': behav_t, 'photodiode': behav['photodiode'], 'ai4': behav['ai4'],
        'wheel_deg': behav['wheel_deg'].astype(np.float32), 'wheel_speed_deg_s': speed.astype(np.float32),
        'wheel_deg_frames': at_frames(behav['wheel_deg']), 'wheel_speed_frames': at_frames(speed),
        'ai4_frames': at_frames(behav['ai4']), 'photodiode_frames': at_frames(behav['photodiode']),
    }
    np.savez(npz_path, **data)
    trials.to_csv(csv_path, index=False)
    print(f'Saved {npz_path} and {csv_path}')
    return data, trials


# ---------------------------------------------------------------- step 3

def _add_match(trials, out_dir, source_tag):
    match_csv = Path(out_dir) / f'roi_match_{source_tag}.csv'
    if not match_csv.exists():
        print(f'No {match_csv.name}; trial table gets no holo ROI ids (run match_rois.py first)')
        return trials, None
    m = pd.read_csv(match_csv).set_index('source_roi')
    t = trials.copy()
    t['holo_roi'] = t.source_roi.map(m['holo_roi']).fillna(-1).astype(int)
    t['status'] = np.where(t.is_blank, 'blank', t.source_roi.map(m['status']))
    t['target_x_holo'] = t.source_roi.map(m['target_x_holo'])
    t['target_y_holo'] = t.source_roi.map(m['target_y_holo'])
    return t, m


def write_nwb(paths, mouse, source_tag, data, trials, force=False):
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.base import Images
    from pynwb.behavior import BehavioralTimeSeries
    from pynwb.epoch import TimeIntervals
    from pynwb.image import GrayscaleImage
    from pynwb.ophys import (DfOverF, Fluorescence, ImageSegmentation, OpticalChannel,
                             RoiResponseSeries)

    out = paths['output']
    nwb_path = out / f'holo_2p_sub-{mouse}_{paths["raw"].name}.nwb'
    if nwb_path.exists() and not force:
        print(f'NWB exists, skipping: {nwb_path}')
        return nwb_path

    plane = paths['plane']
    rois = hio.load_rois(plane)
    ops = hio.load_ops(plane)
    F = np.load(plane / 'F.npy', mmap_mode='r')
    Fneu = np.load(plane / 'Fneu.npy', mmap_mode='r')
    dff = np.load(paths['dff'], mmap_mode='r')
    ts = data['frame_times']
    trials, _ = _add_match(trials, out, source_tag)

    tif = hio.vdaq_h5_path(paths['raw_funcimg']).stem
    start = datetime.strptime(tif[:15], '%Y%m%d_%H%M%S').replace(tzinfo=ZoneInfo('Europe/London'))
    nwb = NWBFile(session_description=f'Holographic single-cell photostimulation of {source_tag} ROIs '
                                     f'(UniformContrast screen), sub-{mouse}',
                  identifier=f'sub-{mouse}_{paths["raw"].name}_holo', session_start_time=start,
                  session_id=paths['raw'].name, lab='Mrsic-Flogel', institution='Sainsbury Wellcome Centre',
                  source_script='Bonsai_ABCD musicboxes/holography/nwb_holography.py',
                  notes=json.dumps({'n_frames_extrapolated': int(data['n_frames_extrapolated']),
                                    'fit_slope': float(data['fit_slope']), 'fit_delta_t': float(data['fit_delta_t']),
                                    'preprocessed': paths['preproc'].name}))

    device = nwb.create_device(name='2p-313', description='Two-photon microscope with SLM photostimulation')
    plane_meta = nwb.create_imaging_plane(
        name='ImagingPlane', optical_channel=OpticalChannel(name='Green', description='GCaMP', emission_lambda=525.),
        description='single plane, 512 x 512', device=device, excitation_lambda=920., indicator='GCaMP',
        location='cortex', imaging_rate=1 / float(data['frame_period']))

    ophys = nwb.create_processing_module('ophys', 'suite2p on the holo session (cellpose chan2 ROIs)')
    images = Images('SegmentationImages')
    images.add_image(GrayscaleImage(name='MeanImageChan1', data=ops['meanImg']))
    if ops.get('meanImg_chan2') is not None:
        images.add_image(GrayscaleImage(name='MeanImageChan2', data=ops['meanImg_chan2']))
    ophys.add(images)

    seg = ImageSegmentation()
    ps = seg.create_plane_segmentation(name='PlaneSegmentation', description=f'holo ROIs ({rois["origin"]})',
                                       imaging_plane=plane_meta)
    ps.add_column('iscell', 'suite2p classifier')
    for s, ic in zip(rois['stat'], rois['iscell']):
        ps.add_roi(pixel_mask=[(int(x), int(y), float(l)) for x, y, l in zip(s['xpix'], s['ypix'], s['lam'])],
                   iscell=bool(ic))
    proj_dir = out / 'funcimg' / 'projected'
    proj_stat_path = out / 'stat_projected.npy'
    has_proj = (proj_dir / 'F_projected.npy').exists() and proj_stat_path.exists()
    if has_proj:
        pstat = np.load(proj_stat_path, allow_pickle=True)
        pps = seg.create_plane_segmentation(name='ProjectedSourceMasks', imaging_plane=plane_meta,
                                            description=f'{source_tag} masks of stimulated cells without a holo '
                                                        f'ROI, shifted into the holo FOV (match_rois.py)')
        pps.add_column('source_roi', f'{source_tag} suite2p ROI index')
        for s in pstat:
            pps.add_roi(pixel_mask=[(int(x), int(y), float(l)) for x, y, l in zip(s['xpix'], s['ypix'], s['lam'])],
                        source_roi=int(s['source_roi']))
    else:
        print('No projected traces yet (extract_projected.py); NWB has holo ROIs only')
    ophys.add(seg)

    region = ps.create_roi_table_region('all holo ROIs', region=list(range(len(rois['stat']))))
    comp = lambda x: H5DataIO(np.asarray(x, np.float32).T, compression='gzip', compression_opts=4)
    fl = Fluorescence(name='Fluorescence')
    fl.create_roi_response_series(name='F', data=comp(F), rois=region, unit='a.u.', timestamps=ts)
    fl.add_roi_response_series(RoiResponseSeries(name='Fneu', data=comp(Fneu), rois=region, unit='a.u.',
                                                 timestamps=fl['F']))
    df = DfOverF(name='DfOverF')
    df.create_roi_response_series(name='dFF', data=comp(dff), rois=region, unit='dF/F', timestamps=fl['F'],
                                  description='(F - 0.7 Fneu) with suite2p baseline subtraction (compute_dff)')
    if has_proj:
        pregion = pps.create_roi_table_region('projected masks', region=list(range(len(pstat))))
        fl.add_roi_response_series(RoiResponseSeries(name='F_projected', rois=pregion, unit='a.u.', timestamps=fl['F'],
                                                     data=comp(np.load(proj_dir / 'F_projected.npy'))))
        fl.add_roi_response_series(RoiResponseSeries(name='Fneu_projected', rois=pregion, unit='a.u.',
                                                     timestamps=fl['F'],
                                                     data=comp(np.load(proj_dir / 'Fneu_projected.npy'))))
        df.add_roi_response_series(RoiResponseSeries(name='dFF_projected', rois=pregion, unit='dF/F',
                                                     timestamps=fl['F'],
                                                     data=comp(np.load(proj_dir / 'dFF_projected.npy'))))
    ophys.add(fl)
    ophys.add(df)

    ti = TimeIntervals(name='photostim_trials', description='One row per stimulation trial: SatsumaGate train '
                       '(20 x 2 ms spirals at 40 Hz); start/stop = first gate rise / last gate fall (vDAQ clock)')
    cols = {'trial': 'trial number in photostim_loader csv', 'condition': "'ensembleN' or 'blank'",
            'source_roi': f'{source_tag} ROI targeted (-1 blank)', 'is_blank': 'blank (no target) trial',
            'isi_s': 'planned ISI after this trial', 'slm_time': 'SLMFeedbackTTL rise',
            'n_pulses': 'gate pulses in the train', 'onset_frame': 'frame containing the first pulse',
            'offset_frame': 'frame containing the end of the last pulse', 'power_V': 'Aimbot power setting',
            'target_x': f'target x in {source_tag} pixels', 'target_y': f'target y in {source_tag} pixels'}
    if 'holo_roi' in trials:
        cols.update({'holo_roi': 'matched holo ROI (-1 none)', 'status': 'matched / projected / blank',
                     'target_x_holo': 'target x in holo pixels', 'target_y_holo': 'target y in holo pixels'})
    for c, d in cols.items():
        ti.add_column(c, d)
    for r in trials.itertuples(index=False):
        row = {c: getattr(r, c) for c in cols}
        row['condition'], row['is_blank'] = str(row['condition']), bool(row['is_blank'])
        if 'status' in row:
            row['status'] = str(row['status'])
        ti.add_interval(start_time=float(r.onset_time), stop_time=float(r.offset_time), **row)
    nwb.add_time_intervals(ti)

    events = lambda name, t, desc: TimeSeries(name=name, data=np.ones(len(t), np.uint8), timestamps=t,
                                              unit='n.a.', description=desc)
    nwb.add_acquisition(events('PhotostimGatePulses', data['gate_on_times'], 'SatsumaGateTTL rising edges'))
    nwb.add_acquisition(events('SLMFeedback', data['slm_times'], 'SLMFeedbackTTL rising edges'))
    nwb.add_acquisition(events('CameraFrames', data['camera_times'], 'behav ai13 rising edges (pupil cameras)'))
    nwb.add_acquisition(TimeSeries(name='BarcodeVDAQ', data=data['vdaq_barcode_values'],
                                   timestamps=data['vdaq_barcode_times'], unit='n.a.'))

    behav_rate = float(data['fit_corrected_behav_rate']) / BEHAV_DOWNSAMPLE
    behav_start = float(data['behav_t'][0])
    bts = BehavioralTimeSeries(name='BehavioralTimeSeries')
    for name, key, unit, desc in [('WheelAngle', 'wheel_deg', 'degrees', 'behav ctr0, cumulative'),
                                  ('WheelSpeed', 'wheel_speed_deg_s', 'degrees/s', f'{WHEEL_SMOOTH_S}s boxcar'),
                                  ('Photodiode', 'photodiode', 'V', 'behav ai3'),
                                  ('AI4', 'ai4', 'V', 'behav ai4, unidentified')]:
        bts.add_timeseries(TimeSeries(name=name, data=H5DataIO(data[key], compression='gzip'), unit=unit,
                                      starting_time=behav_start, rate=behav_rate,
                                      description=f'{desc}; 1 kHz, aligned by barcode (slope + offset)'))
    bts.add_timeseries(TimeSeries(name='WheelSpeedAtFrames', data=data['wheel_speed_frames'], unit='degrees/s',
                                  timestamps=fl['F']))
    nwb.create_processing_module('behavior', 'behaviour DAQ, on the vDAQ clock').add(bts)

    print(f'Writing {nwb_path} ...')
    tmp = nwb_path.with_suffix('.nwb.part')
    with NWBHDF5IO(str(tmp), 'w') as io:
        io.write(nwb)
    tmp.rename(nwb_path)
    print(f'Saved {nwb_path}')
    return nwb_path


# ---------------------------------------------------------------- entry point

def build_session(mouse, holo_tag, source_tag, project_root=hio.PROJECT_ROOT, holo_suffix='SR', force=False):
    paths = hio.session_paths(mouse, holo_tag, project_root, holo_suffix)
    print(f"=== Step 1/4: dF/F ({paths['funcimg']}) ===")
    ensure_dff(paths['funcimg'], neuropil_coefficient=0.7, force=force)
    print('=== Step 2/4: DAQ parsing and barcode alignment ===')
    data, trials = build_alignment(paths, force=force)
    print('=== Step 3/4: NWB ===')
    nwb_path = write_nwb(paths, mouse, source_tag, data, trials, force=force)
    print('=== Step 4/4: verify ===')
    from musicboxes.holography.verify_holography import verify
    flags = verify(mouse, holo_tag, project_root, holo_suffix, nwb_path=nwb_path)
    return nwb_path, flags


def _cli():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('mouse')
    p.add_argument('holo_tag')
    p.add_argument('source_tag')
    p.add_argument('--project-root', default=str(hio.PROJECT_ROOT))
    p.add_argument('--holo-suffix', default='SR')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    build_session(a.mouse, a.holo_tag, a.source_tag, ceph_path(a.project_root), a.holo_suffix, a.force)


if __name__ == '__main__':
    _cli()

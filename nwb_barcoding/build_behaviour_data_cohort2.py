"""Build behaviour_data.npz/.csv + valid_frames.npz for a cohort2 session.

Scriptable port of preprocessing/read_align_timestamps.ipynb (the cohort2 notebook), with
the same two alignment upgrades the cohort3 pipeline got:
  - NIDAQ -> vDAQ time uses the barcode slope + offset fit (barcode_alignment.fit_alignment)
    computed from the freshly built NWBs, instead of the notebook's older barcode code;
  - VR -> NIDAQ time uses the fit over all matched rewards (cohort2_layout.fit_vr_to_nidaq),
    instead of anchoring on the first reward only.

Output conventions follow build_behaviour_data.py (cohort3), so downstream code can treat
the two cohorts alike:
  - frames are those whose vDAQ FrameTTL falls inside the VR session, padded by one frame
    on each side (unless imaging stopped before the VR); behaviour arrays are binned
    between consecutive valid frames, so they have len(valid_frames) - 1 entries;
  - Ca_traces = spks.npy[:, valid_frames];
  - keys: Ca_traces, rewards, licks, modd1, modd2, position, speed -- plus `distance`
    (VR TotalRunDistance minus the config's flip_tunnel margin_start, as in the old cohort2
    files). Cohort2 has no photodiode
    channels, so there's no pd1/p2. t0 sessions have no MODD channels; modd1/modd2 are
    all-zero there.

Speed is the VR-log speed: its 'Speed' column when logged, otherwise the centred derivative
of TotalRunDistance over VR time. (Not create_session_struct()'s forward 10-sample difference:
that leads the position trace by ~6 imaging frames. The centred derivative lines up with
position and with the speed in the existing cohort2 behaviour_data.npz files.) Cohort2 has no
Treadmill column.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from pynwb import NWBHDF5IO

from . import barcode_alignment as ba
from . import cohort2_layout as layout


def _frame_times(funcimg_nwb_path):
    with NWBHDF5IO(str(funcimg_nwb_path), mode='r') as io:
        nwb = io.read()
        types = nwb.acquisition['vDAQ_TTLTypesTable'][:]
        ttls = nwb.acquisition['vDAQ_TTLsTable'][:]
    frame_type = types[types['event_name'] == 'FrameTTL'].index.tolist()
    return ttls.loc[ttls['ttl_type'].isin(frame_type), 'timestamp'].values


def _clean(values):
    """VR logs -1 when a value wasn't logged; interpolate over those like create_session_struct()."""
    values = np.asarray(values, dtype=float).copy()
    values[values == -1] = np.nan
    return pd.Series(values).interpolate(method='linear', limit_direction='both').to_numpy()


def _vr_speed(vr_log, total_dist):
    if 'Speed' in vr_log:
        return vr_log['Speed'].to_numpy(dtype=float)
    # The VR log repeats timestamps now and then; differentiate over unique times only.
    time = vr_log['Time'].to_numpy(dtype=float)
    unique_time, first = np.unique(time, return_index=True)
    return np.interp(time, unique_time, np.gradient(total_dist[first], unique_time))


def _unwrap(position):
    """Undo the lap wrap-around so interpolation/averaging across a lap boundary is valid.
    Returns (unwrapped, period, minimum); period is None if the mouse never completed a lap."""
    pmin, pmax = np.nanmin(position), np.nanmax(position)
    period = np.round(pmax - pmin)
    jumps = np.diff(position)
    if period <= 0 or not np.any(np.abs(jumps) > period / 2):
        return position, None, pmin
    offsets = np.concatenate([[0], np.cumsum(np.where(jumps < -period / 2, period,
                                                      np.where(jumps > period / 2, -period, 0)))])
    return position + offsets, period, pmin


def build_behaviour_data(files, output_session_path, vr_fit=None, save_npz=True, save_csv=True,
                         event_refractory_samples=5, nidaq_barcode_height=0.5, vdaq_barcode_height=0.5):
    """Build behaviour_data.npz/.csv + valid_frames.npz for one cohort2 session.

    :param files: cohort2_layout.find_session_files() dict for the raw session
    :param output_session_path: nwb_barcoding output session dir holding the funcimg/behav NWBs;
        outputs are written here
    :param vr_fit: cohort2_layout.fit_vr_to_nidaq() result (recomputed if None)
    :param event_refractory_samples: see cohort2_layout.detect_rising_edges

    Returns a dict of intermediates, reused by verify_behaviour_data_cohort2.
    """
    out = Path(output_session_path)
    funcimg_nwb_path = list((out / 'funcimg').glob('2p_preprocessed_*.nwb'))[0]
    behav_nwb_path = list((out / 'behav').glob('vrabcd_preprocessed_*.nwb'))[0]

    # Same fit the merged NWB was built with (deterministic, so recomputing it is equivalent).
    fit = ba.fit_alignment(main_decoded=ba.read_syncttl(funcimg_nwb_path, height=vdaq_barcode_height),
                           secondary_decoded=ba.read_syncttl(behav_nwb_path, height=nidaq_barcode_height),
                           secondary_nominal_rate=layout.NIDAQ_NOMINAL_RATE)
    if vr_fit is None:
        vr_fit = layout.fit_session_vr_clock(files)
    to_aligned = lambda t_nidaq: t_nidaq * fit['slope'] + fit['delta_t']

    funcimg_timestamps = _frame_times(funcimg_nwb_path)
    n_channels = layout.ai_channel_count(files['ai_bin'], files['ci_bin'])
    ai = layout.load_ai(files['ai_bin'], n_channels)
    vr_log = layout.load_vr_log(files['position_log'])

    # --- Behaviour window: the VR session, on the NIDAQ sample grid ---
    vr_nidaq_t = vr_log['Time'].to_numpy(dtype=float) * vr_fit['slope'] + vr_fit['intercept']
    rate = layout.NIDAQ_NOMINAL_RATE
    i0 = max(int(np.ceil(vr_nidaq_t[0] * rate)), 0)
    i1 = min(int(np.floor(vr_nidaq_t[-1] * rate)) + 1, ai.shape[0])
    behav_times = to_aligned(np.arange(i0, i1) / rate)
    vr_times = to_aligned(vr_nidaq_t)
    print(f"VR session on the aligned clock: {behav_times[0]:.2f} s - {behav_times[-1]:.2f} s "
          f"(NIDAQ samples {i0}-{i1})")

    # --- Valid frames (same logic as build_behaviour_data.py) ---
    if funcimg_timestamps[-1] < behav_times[-1]:
        print('Funcimg stopped before behaviour!')
        valid_frames_idx = np.where(funcimg_timestamps >= behav_times[0])[0]
    else:
        valid_frames_idx = np.where((funcimg_timestamps >= behav_times[0]) &
                                    (funcimg_timestamps <= behav_times[-1]))[0]
        lo, hi = valid_frames_idx[0] - 1, valid_frames_idx[-1] + 1
        valid_frames_idx = np.concatenate([[lo] if lo >= 0 else [], valid_frames_idx,
                                           [hi] if hi < len(funcimg_timestamps) else []]).astype(int)
    frame_edges = funcimg_timestamps[valid_frames_idx]
    print(f"{len(valid_frames_idx)} valid frames ({frame_edges[0]:.2f} s - {frame_edges[-1]:.2f} s)")

    # --- Events, binned per frame ---
    def _events(name):
        col = next((c for c, n in layout.AI_CHANNELS.items() if n == name), None)
        if col is None or col >= n_channels:
            print(f"  no {name} channel in this session ({n_channels} analog channels) -- all zeros")
            return np.array([], dtype=int), np.zeros(len(frame_edges) - 1)
        idx = layout.detect_rising_edges(ai[i0:i1, col], min_refractory_samples=event_refractory_samples)
        idx = idx[idx + 1 < len(behav_times)]
        counts, _ = np.histogram(behav_times[idx + 1], bins=frame_edges)
        return idx, counts.astype(float)

    raw_rewards_idx, reward_array = _events('RewardTTL')
    raw_licks_idx, lick_array = _events('LickTTL')
    raw_modd1_idx, modd1_array = _events('MODD1ONTTL')
    raw_modd2_idx, modd2_array = _events('MODD2ONTTL')

    # --- Position / distance / speed, from the VR log ---
    position_unwrapped, period, pmin = _unwrap(_clean(vr_log['Position']))
    total_dist = _clean(vr_log['TotalRunDistance'])
    speed = _vr_speed(vr_log, total_dist)
    margin_start = layout.load_config(files['config'])['flip_tunnel'].get('margin_start', 0)

    def _binned_mean(values):
        mean, _, _ = binned_statistic(behav_times, np.interp(behav_times, vr_times, values),
                                      statistic='mean', bins=frame_edges)
        return mean

    position_array = _binned_mean(position_unwrapped)
    if period is not None:
        position_array = (position_array - pmin) % period + pmin
    distance_array = _binned_mean(total_dist) - margin_start
    speed_array = _binned_mean(speed)

    # --- Calcium traces ---
    spikes = np.load(files['suite2p_dir'] / 'plane0' / 'spks.npy')
    behav_spikes = spikes[:, valid_frames_idx]

    if save_npz:
        np.savez_compressed(out / 'behaviour_data.npz', Ca_traces=behav_spikes, rewards=reward_array,
                            licks=lick_array, modd1=modd1_array, modd2=modd2_array,
                            position=position_array, speed=speed_array, distance=distance_array)
        print(f"Saved {out / 'behaviour_data.npz'}")
    if save_csv:
        pd.DataFrame({"Rewards": reward_array, "Licks": lick_array, "MODD1": modd1_array,
                      "MODD2": modd2_array, "Position": position_array, "Speed": speed_array,
                      "Distance": distance_array}).to_csv(out / 'behaviour_data.csv', index=False)
        print(f"Saved {out / 'behaviour_data.csv'}")
    np.savez_compressed(out / 'valid_frames.npz', valid_frames=valid_frames_idx)
    print(f"Saved {out / 'valid_frames.npz'}")

    return dict(
        vr_log=vr_log, vr_fit=vr_fit, fit=fit, funcimg_timestamps=funcimg_timestamps,
        behav_times=behav_times, valid_frames=valid_frames_idx,
        raw_rewards_idx=raw_rewards_idx, raw_licks_idx=raw_licks_idx,
        raw_modd1_idx=raw_modd1_idx, raw_modd2_idx=raw_modd2_idx,
        reward_array=reward_array, lick_array=lick_array, modd1_array=modd1_array,
        modd2_array=modd2_array, position_array=position_array, speed_array=speed_array,
        distance_array=distance_array, n_channels=n_channels,
    )

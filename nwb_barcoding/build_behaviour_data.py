"""Build behaviour_data.npz/.csv + valid_frames.npz from a behav/funcimg/merged NWB triple.

Standalone/scriptable port of Bonsai_ABCD's preprocessing/read_align_timestamps_bonsai.ipynb,
so it can run as part of the nwb_barcoding pipeline without the notebook being run by hand.
Reuses the `parsed_bonsai_data.pkl` that build_session_nwb.py's behav-NWB step already
writes (same file the original notebook reads), and the merged/funcimg NWBs nwb_barcoding
just built.

Inlines the handful of Bonsai_ABCD functions actually needed (load_settings,
load_analog_data, align_analog_to_events -- ported line-for-line from
preprocessing/parse_session_functions_cohort3.py) instead of importing that module,
since it pulls in seaborn/palettes/etc. that aren't installed in the
`2p_preprocess_extraction` conda env this pipeline runs in.

Must run in the `2p_preprocess_extraction` conda env (same as the rest of nwb_barcoding).

Deliberate departure from the notebook: analog-sample timing comes from fit_analog_clock()
(analog camera edges paired with the barcode-aligned CameraFrameTTLs), not the notebook's
`harp_seconds + first camera TTL` anchor, which put behaviour 0.15-10.5 s late -- see
fit_analog_clock's docstring.
"""

import json
import pickle
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from pynwb import NWBHDF5IO

from . import _aimbot_path
_aimbot_path.ensure_on_path()

from aeon.io.reader import Reader  # noqa: E402
import aeon.io.api as aeon_api  # noqa: E402


def load_settings(raw_session_path):
    """Port of preprocessing/parse_session_functions_cohort3.py::load_settings."""
    raw_session_path = Path(raw_session_path)
    settings_path = raw_session_path / "behav" / "session-settings"
    json_files = list(settings_path.glob("*.json"))
    if json_files:
        settings_file = json_files[0]
    else:
        matches = list(settings_path.glob("*.csv"))  # early sessions saved settings as csv
        if not matches:
            raise FileNotFoundError(f"No valid session settings found in {settings_path}")
        settings_file = matches[0]
    with open(settings_file, 'r') as f:
        ses_settings = json.load(f)["value"]

    rig_path = raw_session_path / "behav" / "rig-settings"
    rig_file = list(rig_path.glob("*.json"))[0]
    with open(rig_file, 'r') as f:
        rig_settings = json.load(f)["value"]
    return ses_settings, rig_settings


class _AnalogData(Reader):
    def __init__(self, pattern, columns, channels, extension="bin"):
        super().__init__(pattern, columns, extension)
        self.channels = channels

    def read(self, file):
        data = np.fromfile(file, dtype=np.float64)
        data = np.reshape(data, (-1, self.channels))
        return pd.DataFrame(data, columns=self.columns)


def load_analog_data(raw_session_path, rig_settings):
    """Port of preprocessing/parse_session_functions_cohort3.py::load_analog_data."""
    channel_names = [c['alias'] for c in rig_settings['analogInputChannels']]
    print(f"Analog channels found: {channel_names}")
    reader = _AnalogData("behav/analog-data/*", channel_names, len(channel_names))
    analog_data = aeon_api.load(Path(raw_session_path), reader)
    analog_data = analog_data.reset_index().drop(columns='time')
    return analog_data


def align_analog_to_events(analog_data, sess_dataframe):
    """Port of preprocessing/parse_session_functions_cohort3.py::align_analog_to_events."""
    buffer_data = sess_dataframe[['Buffer']].dropna().drop_duplicates()
    buffer_size = int(analog_data.shape[0] / buffer_data.shape[0])
    print(f'{len(analog_data)} analog samples were recorded.')
    print(f'{len(buffer_data)} buffer samples were recorded.')
    print(f'Buffer size: {buffer_size} samples')

    buffer_seconds = (buffer_data.index - dt.datetime(1904, 1, 1)).total_seconds()
    sliced_index = np.array(analog_data.index)[::buffer_size]
    o_m, o_b = np.polyfit(sliced_index, buffer_seconds, 1)
    index_to_timestamp = lambda x: x * o_m + o_b

    remapped_analog_index = aeon_api.aeon(index_to_timestamp(analog_data["rewards"].index))
    return analog_data.set_index(remapped_analog_index)


def detect_rising_edges(signal, min_refractory_samples=5, height=1):
    """Detect rising-edge events via np.diff(signal) > height, merging any two detected
    edges that are <= min_refractory_samples samples apart into a single event (keeping
    the first).

    Guards against a single physical TTL transition that spans multiple analog samples
    (e.g. a slow/staircased rise) being double-counted as separate events -- confirmed to
    happen in practice: a reward TTL line rising 0.10V -> 3.77V -> 5.08V over 2 consecutive
    samples (~110us at ~9000Hz) was otherwise detected as two reward pulses 1 sample apart,
    since both steps individually exceeded the 1V diff threshold. A real second event
    (another reward, lick, etc.) happening within 5 samples (<1ms at typical analog rates)
    is not physically plausible for any of these channels, so this merge is safe.
    """
    idx = np.where(np.diff(signal) > height)[0]
    if len(idx) < 2:
        return idx

    keep = [idx[0]]
    for i in idx[1:]:
        if i - keep[-1] > min_refractory_samples:
            keep.append(i)
    n_merged = len(idx) - len(keep)
    if n_merged > 0:
        print(f"  merged {n_merged} event(s) detected within {min_refractory_samples} "
              f"sample(s) of a preceding one (likely a multi-sample TTL rise, not a real "
              f"second event)")
    return np.array(keep)


def fit_analog_clock(camera_signal, aligned_camera_timestamps, threshold=1.0, max_residual_ms=1.0):
    """Map every analog sample index to a time on the funcimg (vDAQ) clock.

    The Bonsai analog-data stream IS the NIDAQ recording (Dev2, barcode on ai0, camera on ai13),
    and the merged NWB's CameraFrameTTL timestamps are those same camera edges, already put on
    the funcimg clock by the robust barcode fit (offset + corrected rate). So a linear fit of
    analog edge sample index -> NWB camera timestamp gives each analog sample's funcimg time
    directly, with no assumption about when the camera started.

    This replaces the read_align_timestamps_bonsai.ipynb anchor
    `behav_times = harp_seconds + aligned_camera_timestamps[0]`, which assumed analog sample 0
    coincides with the first camera frame. It doesn't: the camera starts 0.15-10.5 s after
    sample 0 (holography cohort1 and cohort3 measured), so behaviour landed that much late
    relative to imaging, plus ~0.2 s/30 min drift from using the Harp clock instead of the
    corrected NIDAQ rate.
    """
    camera_signal = np.asarray(camera_signal)
    edge_idx = np.where(np.diff((camera_signal > threshold).astype(np.int8)) == 1)[0] + 1
    if len(edge_idx) != len(aligned_camera_timestamps):
        raise ValueError(f"Analog camera channel has {len(edge_idx)} rising edges but the merged NWB "
                         f"has {len(aligned_camera_timestamps)} CameraFrameTTLs -- can't pair them "
                         f"one-to-one for the clock fit.")
    slope, intercept = np.polyfit(edge_idx, aligned_camera_timestamps, 1)
    residual_ms = 1e3 * (aligned_camera_timestamps - (slope * edge_idx + intercept))
    if np.abs(residual_ms).max() > max_residual_ms:
        raise ValueError(f"Analog-sample -> funcimg-clock fit residual up to {np.abs(residual_ms).max():.2f} ms "
                         f"(> {max_residual_ms} ms) -- camera edges don't line up with the NWB TTLs.")
    return dict(sample_times=slope * np.arange(len(camera_signal)) + intercept,
                slope=slope, intercept=intercept, rate_hz=1 / slope, n_edges=len(edge_idx),
                max_residual_ms=float(np.abs(residual_ms).max()))


def get_ttl(nwb_obj, event_name):
    types = nwb_obj.acquisition['NIDAQ_TTLTypesTable'][:]
    ttls = nwb_obj.acquisition['NIDAQ_TTLsTable'][:]
    type_idx = types[types['event_name'] == event_name].index.tolist()
    if not type_idx:
        return np.array([])
    return ttls.loc[ttls['ttl_type'].isin(type_idx), 'timestamp'].values


def build_behaviour_data(raw_session_path, output_session_path, suite2p_plane_path,
                          save_npz=True, save_csv=True, event_refractory_samples=5):
    """Build behaviour_data.npz/.csv + valid_frames.npz for one session.

    :param raw_session_path: true rawdata session path (raw analog .bin, behav CSVs).
    :param output_session_path: an nwb_barcoding output session dir -- must contain
        behav/parsed_bonsai_data.pkl and the merged (vrabcd_2p_preprocessed_*.nwb) and
        funcimg (funcimg/2p_preprocessed_*.nwb) NWBs. Outputs are written here too.
    :param suite2p_plane_path: suite2p plane folder holding spks.npy -- the SAME one
        used to build the funcimg NWB (see build_session_nwb.py's funcimg_source_session).
    :param event_refractory_samples: minimum analog-sample gap required between two
        detected reward/lick/MODD1/MODD2 events for both to count (see
        detect_rising_edges()) -- guards against a single slow/staircased TTL rise
        being double-counted as two events. Default 5 samples (<1ms at ~9000Hz),
        well under any physically real re-triggering interval for these channels.

    Returns a dict of the key intermediates, reused by verify_behaviour_data.py so it
    doesn't have to redo this work.
    """
    raw_session_path = Path(raw_session_path)
    output_session_path = Path(output_session_path)
    suite2p_plane_path = Path(suite2p_plane_path)

    ses_settings, rig_settings = load_settings(raw_session_path)

    with open(list((output_session_path / 'behav').glob('*.pkl'))[0], 'rb') as f:
        sess_dataframe = pickle.load(f)
    print(f'There are {len(sess_dataframe)} behav events.')

    analog_data = load_analog_data(raw_session_path, rig_settings)
    remapped_analog_data = align_analog_to_events(analog_data, sess_dataframe)
    remapped_analog_index = remapped_analog_data.index
    remapped_analog_times = (remapped_analog_index - dt.datetime(1904, 1, 1)).total_seconds().values
    bonsai_times = sess_dataframe.index

    merged_nwb_path = list(output_session_path.glob("vrabcd_2p_preprocessed_*.nwb"))[0]
    with NWBHDF5IO(str(merged_nwb_path), mode='r') as io:
        nwb = io.read()
        aligned_camera_timestamps = get_ttl(nwb, 'CameraFrameTTL')
        aligned_modd1_timestamps = get_ttl(nwb, 'MODD1ONTTL')
        aligned_modd2_timestamps = get_ttl(nwb, 'MODD2ONTTL')

    funcimg_nwb_path = list((output_session_path / 'funcimg').glob("2p_preprocessed_*.nwb"))[0]
    with NWBHDF5IO(str(funcimg_nwb_path), mode='r') as io:
        funcimg_nwb = io.read()
        funcimg_timestamps = funcimg_nwb.acquisition['vDAQ_TTLsTable'][:]['timestamp'].values

    # --- Analog sample -> funcimg-clock time (see fit_analog_clock's docstring) ---
    behav_end_ts = aligned_camera_timestamps[-1]
    clock_fit = fit_analog_clock(remapped_analog_data['camera'].values, aligned_camera_timestamps)
    behav_times = clock_fit['sample_times']
    old_anchor_offset = (remapped_analog_times[0] + aligned_camera_timestamps[0]) - behav_times[0]
    print(f"Analog clock: {clock_fit['n_edges']} camera edges matched, max residual "
          f"{clock_fit['max_residual_ms']:.3f} ms, effective NIDAQ rate {clock_fit['rate_hz']:.4f} Hz. "
          f"(The old harp+first-camera-TTL anchor would have put behaviour {old_anchor_offset:.3f} s late.)")
    print(f'Timestamps aligned to funcimg data\n Start: {behav_times[0]}\n End: {behav_times[-1]}')

    if funcimg_timestamps[-1] < behav_times[-1]:
        print('Funcimg stopped before behaviour!')
        valid_frames_behav_idx = np.where(funcimg_timestamps >= behav_times[0])[0]
        valid_frames_behav = funcimg_timestamps[valid_frames_behav_idx]
    else:
        if behav_end_ts < behav_times[-1]:
            print('Face camera stopped before behaviour!')
        valid_frames_behav_idx = np.where(
            (funcimg_timestamps >= behav_times[0]) & (funcimg_timestamps <= behav_times[-1]))[0]
        # Pad by one frame each side, clipped to the recording: with the fit_analog_clock timing,
        # behaviour can start before the first imaging frame (camera started after analog
        # sample 0), and an unclipped `0 - 1` index wrapped to the LAST frame, making the
        # frame bins non-monotonic.
        expanded_idx = np.arange(max(valid_frames_behav_idx[0] - 1, 0),
                                 min(valid_frames_behav_idx[-1] + 1, len(funcimg_timestamps) - 1) + 1)
        valid_frames_behav = funcimg_timestamps[expanded_idx]
        valid_frames_behav_idx = expanded_idx

    # --- Event arrays, binned to funcimg frames ---
    print("Detecting reward events...")
    raw_rewards_idx = detect_rising_edges(remapped_analog_data['rewards'].values,
                                           min_refractory_samples=event_refractory_samples)
    print("Detecting lick events...")
    raw_licks_idx = detect_rising_edges(remapped_analog_data['licks'].values,
                                         min_refractory_samples=event_refractory_samples)
    print("Detecting MODD1 events...")
    raw_modd1_idx = detect_rising_edges(remapped_analog_data['modd1'].values,
                                         min_refractory_samples=event_refractory_samples)
    print("Detecting MODD2 events...")
    raw_modd2_idx = detect_rising_edges(remapped_analog_data['modd2'].values,
                                         min_refractory_samples=event_refractory_samples)
    raw_ttl1_idx = np.where((remapped_analog_data['pd1']) > 1)[0]
    raw_ttl2_idx = np.where((remapped_analog_data['pd2']) > 1)[0]

    n = len(remapped_analog_data)
    bin_reward, bin_licks, bin_modd1, bin_modd2 = (np.zeros(n) for _ in range(4))
    bin_ttl1, bin_ttl2 = np.zeros(n), np.zeros(n)
    bin_reward[raw_rewards_idx + 1] = 1
    bin_licks[raw_licks_idx + 1] = 1
    bin_modd1[raw_modd1_idx + 1] = 1
    bin_modd2[raw_modd2_idx + 1] = 1
    valid = (raw_ttl1_idx + 1) < bin_ttl1.shape[0]
    bin_ttl1[raw_ttl1_idx[valid] + 1] = 1
    valid = (raw_ttl2_idx + 1) < bin_ttl2.shape[0]
    bin_ttl2[raw_ttl2_idx[valid] + 1] = 1

    reward_array, _, _ = binned_statistic(behav_times, bin_reward, statistic='sum', bins=valid_frames_behav)
    lick_array, _, _ = binned_statistic(behav_times, bin_licks, statistic='sum', bins=valid_frames_behav)
    modd1_array, _, _ = binned_statistic(behav_times, bin_modd1, statistic='sum', bins=valid_frames_behav)
    modd2_array, _, _ = binned_statistic(behav_times, bin_modd2, statistic='sum', bins=valid_frames_behav)
    ttl1_array, _, _ = binned_statistic(behav_times, bin_ttl1, statistic='sum', bins=valid_frames_behav)
    ttl2_array, _, _ = binned_statistic(behav_times, bin_ttl2, statistic='sum', bins=valid_frames_behav)

    # --- Position / speed ---
    # Speed comes from the raw Treadmill channel (the same signal Bonsai's own
    # velocityThreshold gating reads live), not from differencing position -- diff(position)/dt
    # is a different signal on a different numeric scale (no shared calibration with
    # velocityThreshold), which made downstream speed-thresholding (threshold_licks) far
    # stricter than intended and silently dropped ~1 in 5 real licks in the sequence session
    # this was diagnosed on.
    bonsai_position = sess_dataframe['Position']
    interp_position = np.interp(remapped_analog_index, bonsai_times, bonsai_position)
    bonsai_speed = sess_dataframe['Treadmill']
    interp_speed = np.interp(remapped_analog_index, bonsai_times, bonsai_speed)

    speed_array, _, _ = binned_statistic(behav_times, interp_speed, statistic='mean', bins=valid_frames_behav)
    position_array, _, _ = binned_statistic(behav_times, interp_position, statistic='mean', bins=valid_frames_behav)

    # --- Calcium traces ---
    spikes = np.load(suite2p_plane_path / 'spks.npy')
    behav_spikes = spikes[:, valid_frames_behav_idx]

    if save_npz:
        np.savez_compressed(output_session_path / 'behaviour_data.npz',
                             Ca_traces=behav_spikes, rewards=reward_array, licks=lick_array,
                             modd1=modd1_array, modd2=modd2_array, pd1=ttl1_array, p2=ttl2_array,
                             position=position_array, speed=speed_array)
        print(f"Saved {output_session_path / 'behaviour_data.npz'}")

    if save_csv:
        df = pd.DataFrame({"Rewards": reward_array, "Licks": lick_array, "MODD1": modd1_array,
                            "MODD2": modd2_array, "PD1": ttl1_array, "PD2": ttl2_array,
                            "Position": position_array, "Speed": speed_array})
        df.to_csv(output_session_path / 'behaviour_data.csv', index=False)
        print(f"Saved {output_session_path / 'behaviour_data.csv'}")

    np.savez_compressed(output_session_path / 'valid_frames.npz', valid_frames=valid_frames_behav_idx)
    print(f"Saved {output_session_path / 'valid_frames.npz'}")

    return dict(
        sess_dataframe=sess_dataframe, ses_settings=ses_settings, remapped_analog_data=remapped_analog_data,
        funcimg_timestamps=funcimg_timestamps, behav_times=behav_times,
        valid_frames=valid_frames_behav_idx, behav_spikes=behav_spikes,
        raw_rewards_idx=raw_rewards_idx, raw_licks_idx=raw_licks_idx,
        raw_modd1_idx=raw_modd1_idx, raw_modd2_idx=raw_modd2_idx,
        aligned_modd1_timestamps=aligned_modd1_timestamps, aligned_modd2_timestamps=aligned_modd2_timestamps,
        clock_fit={k: v for k, v in clock_fit.items() if k != 'sample_times'},
        reward_array=reward_array, lick_array=lick_array, modd1_array=modd1_array, modd2_array=modd2_array,
        position_array=position_array, speed_array=speed_array,
        merged_nwb_path=merged_nwb_path, funcimg_nwb_path=funcimg_nwb_path,
    )


if __name__ == "__main__":
    import defopt
    defopt.run(build_behaviour_data)

"""Sanity checks for behaviour_data.npz/valid_frames.npz.

Function-ized version of preprocessing/quality_control/session_sanity_checking.ipynb's Steps 2, 4
and 5 (shape/frame-count consistency, raw-vs-npz event counts and reward positions,
key behavioural metrics, Ca_traces identity), so build_session_nwb.py can run them
automatically right after build_behaviour_data.py, without the notebook.
"""

from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO

MAX_MISSING_END_FRAME_TTLS = 5


def _flag(flags, message):
    flags.append(message)
    print(f"[FLAG] {message}")


def _finite_position_span(values, label, flags):
    """Return last-minus-first finite position without hiding all-NaN data."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        _flag(flags, f"{label} position contains no finite values.")
        return np.nan
    n_missing = values.size - finite.size
    if n_missing:
        print(f"[INFO] {label} position has {n_missing}/{values.size} non-finite sample(s); "
              "using the first and last finite samples for distance.")
    return finite[-1] - finite[0]


def check_shapes(output_session_path, suite2p_plane_path):
    """Disk-based (doesn't need `built`): reloads valid_frames.npz/behaviour_data.npz and
    finds the merged NWB by globbing, so this still works when the build step was skipped."""
    flags = []
    output_session_path = Path(output_session_path)
    valid_frames = np.load(output_session_path / 'valid_frames.npz')['valid_frames']
    with np.load(output_session_path / 'behaviour_data.npz') as behav_data:
        expected_keys = {
            'Ca_traces', 'rewards', 'licks', 'modd1', 'modd2', 'pd1', 'pd2',
            'position', 'speed',
        }
        missing_keys = sorted(expected_keys.difference(behav_data.files))
        if missing_keys:
            _flag(flags, f"behaviour_data.npz is missing keys: {missing_keys}")
        ca_traces = behav_data['Ca_traces']
        expected_frames = ca_traces.shape[1]
        for key in sorted(expected_keys.difference({'Ca_traces'}).intersection(behav_data.files)):
            if behav_data[key].shape != (expected_frames,):
                _flag(
                    flags,
                    f"behaviour_data.npz '{key}' has shape {behav_data[key].shape}; "
                    f"expected ({expected_frames},) to match Ca_traces.",
                )

    if valid_frames.ndim != 1 or len(valid_frames) == 0:
        _flag(flags, f"valid_frames must be a non-empty 1D array, got shape {valid_frames.shape}.")
        return flags
    if len(valid_frames) != ca_traces.shape[1]:
        _flag(flags, f"valid_frames has {len(valid_frames)} entries but Ca_traces has "
              f"{ca_traces.shape[1]} frames.")
    if np.any(np.diff(valid_frames) != 1):
        _flag(flags, "valid_frames is not strictly contiguous.")

    ops = np.load(Path(suite2p_plane_path) / 'ops.npy', allow_pickle=True).item()
    files_processed = ops.get('file_list', ops.get('filelist'))
    frames_per_file = ops['frames_per_file']
    for i, file in enumerate(files_processed):
        print(f"File processed by suite2p: {file} with {frames_per_file[i]} frames")
    print(f"Valid frames detected: {valid_frames.shape}")
    print(f"Ca_traces shape - frames: {ca_traces.shape[1]} - cells: {ca_traces.shape[0]}")

    merged_nwb_path = list(output_session_path.glob("vrabcd_2p_preprocessed_*.nwb"))[0]
    with NWBHDF5IO(str(merged_nwb_path), mode='r') as io:
        nwb = io.read()
        dfoverf = nwb.processing['ophys']['DfOverF']
        try:
            dF_shape = dfoverf['DfOverFChan1Plane0'].data.shape
        except KeyError:
            dF_shape = dfoverf['DfOverF'].data.shape

    print(f"dF matrix shape - frames: {dF_shape[0]} - cells: {dF_shape[1]}")

    # valid_frames indexes the vDAQ frame TTLs of ONE ScanImage acquisition (the one whose .h5
    # the funcimg NWB was built from), starting at 0. If suite2p concatenated extra (e.g. short
    # test) acquisitions in front of it, spks/dF have more frames than that acquisition and every
    # trace is silently shifted by the prepended frame count -- this is what killed tuning in most
    # holography-cohort sessions (2000-4000 prepended frames = 44-89 s), and the old
    # `dF_frames > valid_frames.max()` test below can't catch it. So check frame counts directly.
    spks_frames = np.load(Path(suite2p_plane_path) / 'spks.npy', mmap_mode='r').shape[1]
    if len(files_processed) != 1:
        _flag(flags, f"suite2p ran on {len(files_processed)} files ({list(frames_per_file)} frames) -- "
              f"valid_frames only indexes one acquisition, so traces are offset by the frames of "
              f"the files before it. Rerun suite2p on the main acquisition only.")
    if not (int(frames_per_file[0]) == int(ops['nframes']) == spks_frames == dF_shape[0]):
        _flag(flags, f"Frame counts disagree: frames_per_file[0]={frames_per_file[0]}, "
              f"ops nframes={ops['nframes']}, spks.npy={spks_frames}, NWB dF={dF_shape[0]} -- "
              f"stale suite2p output or stale NWB.")
    funcimg_nwb_path = list((output_session_path / 'funcimg').glob("2p_preprocessed_*.nwb"))[0]
    with NWBHDF5IO(str(funcimg_nwb_path), mode='r') as io:
        n_frame_ttls = len(io.read().acquisition['vDAQ_TTLsTable'][:])
    n_missing = spks_frames - n_frame_ttls
    print(f"vDAQ frame TTLs: {n_frame_ttls} (suite2p frames: {spks_frames}, difference {n_missing})")
    # A few missing TTLs at the very end are expected (acquisition stopped while the frame line was
    # high -- seen as 80999/81000 and 53997/54000, no gaps mid-session); anything else isn't.
    if not 0 <= n_missing <= MAX_MISSING_END_FRAME_TTLS:
        _flag(flags, f"suite2p has {spks_frames} frames but the funcimg NWB has {n_frame_ttls} vDAQ "
              f"frame TTLs -- the NWB was likely built from a different acquisition's .h5 than "
              f"suite2p processed.")

    if dF_shape[1] != ca_traces.shape[0]:
        _flag(flags, f"Cell count mismatch: merged NWB DfOverF has {dF_shape[1]} cells but "
              f"Ca_traces has {ca_traces.shape[0]} cells -- check for a stale suite2p run "
              f"relative to when the funcimg NWB was built.")
    if dF_shape[0] <= valid_frames.max():
        _flag(flags, f"dF matrix has only {dF_shape[0]} frames but valid_frames indexes up to "
              f"{valid_frames.max()} -- frame numbering mismatch between the NWB and valid_frames.")
    return flags


def check_frame_overlap(built):
    flags = []
    funcimg_timestamps = built['funcimg_timestamps']
    behav_times = built['behav_times']
    valid_frames = built['valid_frames']
    print(f"funcimg session: {funcimg_timestamps[0]:.2f}s - {funcimg_timestamps[-1]:.2f}s "
          f"({len(funcimg_timestamps)} frames)")
    print(f"behaviour:       {behav_times[0]:.2f}s - {behav_times[-1]:.2f}s")
    print(f"valid_frames:    {funcimg_timestamps[valid_frames[0]]:.2f}s - "
          f"{funcimg_timestamps[valid_frames[-1]]:.2f}s ({len(valid_frames)} frames)")
    return flags


def check_event_alignment(built, max_error_s=1 / 45):
    """MODD1/MODD2 edges exist both in the raw analog stream (timed via behav_times, i.e. what
    goes into behaviour_data.npz) and as barcode-aligned NIDAQ TTLs in the merged NWB -- an
    independent check that behaviour lands on the right imaging frame (< 1 frame error)."""
    flags = []
    behav_times = built['behav_times']
    for name in ['modd1', 'modd2']:
        raw_idx, nwb_ts = built[f'raw_{name}_idx'], built[f'aligned_{name}_timestamps']
        if len(raw_idx) == 0 or len(nwb_ts) == 0:
            print(f"{name}: no events, skipped")
            continue
        if len(raw_idx) != len(nwb_ts):
            _flag(flags, f"{name}: {len(raw_idx)} analog edges vs {len(nwb_ts)} NWB TTLs -- can't pair them.")
            continue
        err = behav_times[raw_idx + 1] - nwb_ts
        print(f"{name}: {len(err)} events, median error {1e3 * np.median(err):.2f} ms, "
              f"max |error| {1e3 * np.abs(err).max():.2f} ms")
        if np.abs(err).max() > max_error_s:
            _flag(flags, f"{name}: behaviour timing differs from barcode-aligned TTLs by up to "
                  f"{np.abs(err).max():.3f} s (> 1 frame).")
    return flags


def check_event_counts(built, behav_data):
    flags = []
    print(f"{'event':<8}{'raw analog count':<20}{'behaviour_data.npz sum':<25}")
    for name, raw_idx, npz_key in [
        ('rewards', built['raw_rewards_idx'], 'rewards'), ('licks', built['raw_licks_idx'], 'licks'),
        ('modd1', built['raw_modd1_idx'], 'modd1'), ('modd2', built['raw_modd2_idx'], 'modd2'),
    ]:
        npz_sum = np.nansum(behav_data[npz_key])
        print(f"{name:<8}{len(raw_idx):<20}{npz_sum:<25.0f}")
        if len(raw_idx) != npz_sum:
            _flag(flags, f"Raw analog '{name}' count ({len(raw_idx)}) != behaviour_data.npz '{npz_key}' sum "
                  f"({npz_sum:.0f}) -- event(s) lost or double-counted during binning.")
    return flags


def check_reward_positions_and_metrics(built, behav_data):
    flags = []
    sess_dataframe = built['sess_dataframe']

    # Trivial reward/lick extraction -- same lines as
    # parse_session_functions_cohort3.py::get_event_parsed, minus the odor/landmark
    # event estimation that function also does (not needed for this check).
    reward_positions = sess_dataframe['Position'].values[sess_dataframe['Rewards'].notna()]

    npz_reward_frames = np.where(behav_data['rewards'] > 0)[0]
    npz_reward_positions = behav_data['position'][npz_reward_frames]
    npz_rewards_sum = int(np.nansum(behav_data['rewards']))

    print(f"Raw Bonsai: {len(reward_positions)} rewards; npz: {len(npz_reward_frames)} reward frames, "
          f"sum {npz_rewards_sum}")
    if len(reward_positions) != len(npz_reward_frames):
        _flag(flags, f"Raw reward count ({len(reward_positions)}) != npz reward-frame count "
              f"({len(npz_reward_frames)}).")
    if len(reward_positions) != npz_rewards_sum:
        _flag(flags, f"Raw reward count ({len(reward_positions)}) != npz reward sum ({npz_rewards_sum}) "
              f"-- likely two rewards landing in the same imaging frame.")
    if len(reward_positions) == len(npz_reward_frames) and len(reward_positions) > 0:
        max_pos_diff = float(np.max(np.abs(np.sort(reward_positions) - np.sort(npz_reward_positions))))
        if max_pos_diff > 5:
            _flag(flags, f"Raw vs. npz reward positions differ by up to {max_pos_diff:.1f} cm (sorted, paired).")

    raw_distance = _finite_position_span(sess_dataframe['Position'].values, 'Raw Bonsai', flags)
    npz_distance = _finite_position_span(behav_data['position'], 'behaviour_data.npz', flags)
    raw_licks_logged = int(sess_dataframe['Licks'].sum())
    npz_licks = int(np.nansum(behav_data['licks']))
    raw_mean_speed = sess_dataframe['Treadmill'].mean()
    npz_mean_speed = np.nanmean(behav_data['speed'])

    print(f"{'metric':<22}{'raw (Bonsai)':<20}{'behaviour_data.npz':<20}")
    print(f"{'Total distance (cm)':<22}{raw_distance:<20.1f}{npz_distance:<20.1f}")
    print(f"{'Total licks':<22}{raw_licks_logged:<20d}{npz_licks:<20d}")
    print(f"{'Mean speed (cm/s)':<22}{raw_mean_speed:<20.2f}{npz_mean_speed:<20.2f}")

    if abs(raw_distance - npz_distance) > 0.05 * max(abs(raw_distance), 1):
        _flag(flags, f"Total distance differs by more than 5% between raw ({raw_distance:.1f} cm) and "
              f"npz ({npz_distance:.1f} cm).")
    if raw_licks_logged != npz_licks:
        _flag(flags, f"Total licks differ between the raw Bonsai 'Licks' column ({raw_licks_logged}) and "
              f"npz ({npz_licks}) -- in a mouse 002/t3 case this turned out to be the raw-analog "
              f"edge detector over-counting a multi-sample TTL rise as two events (now guarded "
              f"against via detect_rising_edges()'s refractory merge in build_behaviour_data.py, "
              f"see event_refractory_samples), not a Bonsai logging issue -- check that first.")
    if npz_mean_speed > 0 and abs(raw_mean_speed - npz_mean_speed) / npz_mean_speed > 0.25:
        _flag(flags, f"Mean speed differs substantially between raw Treadmill ({raw_mean_speed:.2f} cm/s) "
              f"and npz derived speed ({npz_mean_speed:.2f} cm/s) -- confirm which is the intended "
              f"ground truth before using either for analysis.")
    return flags


def check_ca_traces_identity(output_session_path, suite2p_plane_path):
    """Re-reads behaviour_data.npz/valid_frames.npz and spks.npy fresh from disk, so this
    check stays meaningful even if called well after build_behaviour_data() ran (e.g. if
    suite2p has been rerun since)."""
    flags = []
    output_session_path = Path(output_session_path)
    behav_data = np.load(output_session_path / 'behaviour_data.npz')
    valid_frames = np.load(output_session_path / 'valid_frames.npz')['valid_frames']
    ca_traces = behav_data['Ca_traces']
    spikes_full = np.load(Path(suite2p_plane_path) / 'spks.npy')

    if spikes_full.shape[0] != ca_traces.shape[0] or spikes_full.shape[1] <= valid_frames.max():
        _flag(flags, f"Cannot compare: spks.npy shape {spikes_full.shape} vs Ca_traces shape "
              f"{ca_traces.shape} (valid_frames max {valid_frames.max()}) -- suite2p output has "
              f"changed since this behaviour_data.npz was built; it needs to be regenerated.")
    else:
        identical = np.allclose(spikes_full[:, valid_frames], ca_traces)
        print(f"Ca_traces matches spks.npy[:, valid_frames]: {identical}")
        if not identical:
            _flag(flags, "Ca_traces does not exactly match spks.npy[:, valid_frames] -- "
                  "check for an off-by-one/transpose bug.")
    return flags


def verify(output_session_path, suite2p_plane_path, built=None):
    """Run all behaviour_data.npz sanity checks. Returns a list of flag strings.

    :param built: the dict returned by build_behaviour_data.build_behaviour_data() for
        this session, needed for the checks that depend on the raw analog re-derivation
        (frame overlap, raw-vs-npz event counts, reward positions/metrics) -- that
        re-derivation isn't persisted to disk on its own. If None (e.g. because
        build_session_nwb.py skipped an already-built behaviour_data.npz), those
        checks are skipped and only the disk-based ones run (shapes, Ca_traces identity).
    """
    output_session_path = Path(output_session_path)

    print("--- Shapes ---")
    flags = check_shapes(output_session_path, suite2p_plane_path)

    if built is not None:
        behav_data = np.load(output_session_path / 'behaviour_data.npz')
        print("\n--- Frame overlap ---")
        flags += check_frame_overlap(built)
        print("\n--- Behaviour vs barcode-aligned TTL timing ---")
        flags += check_event_alignment(built)
        print("\n--- Raw vs npz event counts ---")
        flags += check_event_counts(built, behav_data)
        print("\n--- Reward positions + key metrics ---")
        flags += check_reward_positions_and_metrics(built, behav_data)
    else:
        print("\n(skipping frame-overlap/event-count/reward-position checks -- "
              "behaviour_data.npz build was skipped, so the raw analog re-derivation "
              "needed for those isn't available; pass force=True to redo it.)")

    print("\n--- Ca_traces identity ---")
    flags += check_ca_traces_identity(output_session_path, suite2p_plane_path)

    print(f"\n{len(flags)} issue(s) flagged." if flags else "\nNo issues flagged.")
    return flags

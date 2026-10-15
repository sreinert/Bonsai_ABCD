"""Barcode decoding + robust cross-DAQ alignment, operating on already-built NWB files.

This mirrors the data flow of the Aimbot_barcoding repo's
preprocess_alignment/barcode/{extract,align}.py (decode from an NWB's SyncTTL
acquisition TimeSeries, match by exact barcode value), but fixes a specific bug
found there: `calculate_temporal_difference` fits a full slope+intercept
regression across all shared barcodes, then DISCARDS the slope and keeps only
the intercept as a constant time offset (`delta_t`). Separately,
NIDAQDataInterFace.sampling_frequency (9000.0) and
AeonDataInterFace.sampling_frequency (9000/200) are hardcoded nominal rates,
not measured ones.

Two independent DAQs (behavior NIDAQ, imaging vDAQ) never tick at exactly their
nominal rate -- in every GoalProgress/cohort3 session sanity-checked so far,
the true NIDAQ rate was ~8999.0 Hz, not 9000.0 (~111 ppm off). A delta_t-only
correction is exact only at the single barcode anchor point used to compute
it, and grows increasingly wrong away from that point -- this is exactly the
~110-270 ms growing discrepancy found in preprocessing/quality_control/session_sanity_checking.ipynb
across three real sessions.

The fix: the discarded slope IS the measurement of that rate error. Use it to
derive the true secondary-DAQ sample rate instead of assuming a hardcoded
nominal one, and keep using the intercept as the anchor offset exactly as
before (see nwb_barcoding/robust_converter.py for where this gets applied).

No files are read from or written to `rawdata` here -- this module only reads
already-built NWB files (read-only) and returns numbers.
"""

import numpy as np
from pynwb import NWBHDF5IO
from scipy.signal import find_peaks


def decode_barcode(raw_signal, sample_rate, height=0.5, global_tolerance=0.2,
                    nbits=16, ind_wrap_duration=10, ind_bar_duration=30):
    """Decode Arduino-style barcodes from a raw digital/analog TTL trace.

    Same algorithm as preprocessing/quality_control/barcode_diagnostics.py (itself a port of
    preprocessing/barcode/extract_barcodes.py and Aimbot_barcoding's
    preprocess_alignment/barcode/extract.py -- the three are functionally
    identical decode implementations). `height=0.5` is appropriate for an
    already-digitized 0/1 SyncTTL signal (Aimbot_barcoding's own default is
    0.4-0.8 for this reason); a raw analog channel typically needs a higher
    value -- inspect decoded counts/plots before trusting the result, see
    nwb_barcoding/verify_session.py's barcode-count sanity check.

    Returns a dict with:
        start_sample : int64 array, sample index of each decoded barcode's
            wrapper start (same index space as raw_signal).
        values : int64 array, decoded 16-bit barcode value at each start_sample.
        transition_sample : int array, every detected TTL transition.
        n_skipped : int, number of candidate wrapper windows that could not be
            decoded (missing on/off times).
    """
    sample_conversion = 1000 / sample_rate  # ms per sample
    wrap_duration = 3 * ind_wrap_duration
    total_barcode_duration = nbits * ind_bar_duration + 2 * wrap_duration
    min_wrap_duration = ind_wrap_duration * (1 - global_tolerance)
    max_wrap_duration = ind_wrap_duration * (1 + global_tolerance)

    # float first: SyncTTL is stored as uint8, where np.diff wraps a 1->0 edge to 255
    event_index, _ = find_peaks(np.abs(np.diff(np.asarray(raw_signal, dtype=np.float64))), height=height)
    indexed_times = event_index

    if len(indexed_times) < 2:
        return {'start_sample': np.array([], dtype=np.int64),
                'values': np.array([], dtype=np.int64),
                'transition_sample': indexed_times,
                'n_skipped': 0}

    events_time_diff = np.diff(indexed_times) * sample_conversion
    wrapper_array = indexed_times[np.where(
        (min_wrap_duration < events_time_diff) & (events_time_diff < max_wrap_duration))[0]]

    if len(wrapper_array) < 2:
        return {'start_sample': np.array([], dtype=np.int64),
                'values': np.array([], dtype=np.int64),
                'transition_sample': indexed_times,
                'n_skipped': 0}

    false_wrapper_check = np.diff(wrapper_array) * sample_conversion
    nexttoeachother = np.where(np.abs(np.diff(false_wrapper_check)) < ind_wrap_duration * global_tolerance)[0]
    false_wrapper_check[nexttoeachother + 1] = np.nan
    with np.errstate(invalid='ignore'):
        false_wrappers = np.where(false_wrapper_check < max_wrap_duration)[0]
    wrapper_array = wrapper_array[false_wrappers]

    # A barcode can't start while the previous one is still being sent (2 wrappers + nbits bars =
    # total_barcode_duration, 540 ms by default). In some sessions/rigs (holography cohort1 t0/t3,
    # some t2) each barcode's TRAILING wrapper also passes the wrapper checks above and was decoded
    # as a phantom barcode 0.52 s after the real one -- always value 32767, from the idle line --
    # which doubled the decoded count, made every step "non-consecutive" in verify_session, and
    # gave one cross-stream 32767<->32767 match seconds off that the robust fit had to exclude.
    # The `height` threshold can't fix this (SyncTTL is clean 0/1, so any height < 1 finds the
    # same edges). Timing alone can't pick which of two overlapping starts is real (phantom is
    # 521 ms after a real start, the next real one 479 ms after the phantom -- both < 540 ms), so
    # keep the one with more TTL transitions inside its data window: a real barcode's bits toggle
    # the line there, a phantom sits on the idle line until the next real wrapper.
    def _n_data_transitions(start):
        lo = start + wrap_duration / sample_conversion
        hi = start + (total_barcode_duration - wrap_duration) / sample_conversion
        return np.count_nonzero((indexed_times > lo) & (indexed_times < hi))

    kept = [wrapper_array[0]]
    for start in wrapper_array[1:]:
        if (start - kept[-1]) * sample_conversion >= total_barcode_duration:
            kept.append(start)
        elif _n_data_transitions(start) > _n_data_transitions(kept[-1]):
            kept[-1] = start
    wrapper_array = np.array(kept)

    wrapper_start_times = wrapper_array
    on_times, off_times = np.array([]), np.array([])
    for idx, ts in enumerate(indexed_times):
        if ts == wrapper_start_times[0]:
            on_times = indexed_times[idx::2]
            off_times = indexed_times[idx + 1::2]
            break

    wrapper_start_ms = wrapper_start_times * sample_conversion
    on_ms = on_times * sample_conversion
    off_ms = off_times * sample_conversion

    barcodes = []
    start_samples = []
    n_skipped = 0
    for start_time, start_sample in zip(wrapper_start_ms, wrapper_start_times):
        oncode = on_ms[(on_ms > start_time) & (on_ms < start_time + total_barcode_duration)]
        offcode = off_ms[(off_ms > start_time) & (off_ms < start_time + total_barcode_duration)]

        if len(offcode) == 0 or len(oncode) == 0:
            n_skipped += 1
            continue

        if offcode[0] < oncode[0]:
            curr_time = offcode[0] + 2 * ind_wrap_duration
        else:
            offcode, oncode = oncode.copy(), offcode.copy()
            curr_time = offcode[0] + 2 * ind_wrap_duration

        bits = np.zeros(nbits)
        interbit_off = False
        for bit in range(nbits):
            next_on_idx = np.where(oncode >= (curr_time - ind_bar_duration * global_tolerance))[0]
            next_off_idx = np.where(offcode >= (curr_time - ind_bar_duration * global_tolerance))[0]
            next_on = oncode[next_on_idx[0]] if next_on_idx.size > 0 else curr_time + ind_bar_duration
            next_off = offcode[next_off_idx[0]] if next_off_idx.size > 0 else curr_time + ind_bar_duration

            min_bar = curr_time - ind_bar_duration * global_tolerance
            max_bar = curr_time + ind_bar_duration * global_tolerance

            if min_bar <= next_on <= max_bar:
                bits[bit] = 1
                interbit_off = False
            elif min_bar <= next_off <= max_bar:
                interbit_off = True
            elif not interbit_off:
                bits[bit] = 1

            curr_time += ind_bar_duration

        barcodes.append(sum(bits[b] * 2 ** b for b in range(nbits)))
        start_samples.append(start_sample)

    start_samples = np.array(start_samples)
    barcodes = np.array(barcodes)
    valid = (barcodes >= 1) & (barcodes < 65535)

    return {
        'start_sample': start_samples[valid].astype(np.int64),
        'values': barcodes[valid].astype(np.int64),
        'transition_sample': indexed_times,
        'n_skipped': n_skipped,
    }


def find_dropped_barcodes(values):
    """Indices i where values[i+1] does not equal values[i] + 1 (dropped/misdecoded barcode)."""
    steps = np.diff(values)
    return np.where(steps != 1)[0]


def read_syncttl(nwbfile_path, height=0.5):
    """Read+decode the SyncTTL barcode channel straight out of an NWB file (read-only).

    Mirrors Aimbot_barcoding's preprocess_alignment/barcode/extract.py::extract_syncttl_barcode,
    but returns the full decode dict (including transition samples and skip count for
    quality-checking) rather than just the bare (timestamp, value) array.
    """
    with NWBHDF5IO(str(nwbfile_path), mode='r') as io:
        nwbfile = io.read()
        sync = nwbfile.acquisition['SyncTTL']
        data = sync.data[:]
        sample_rate = float(sync.rate)
    decoded = decode_barcode(data, sample_rate=sample_rate, height=height)
    decoded['sample_rate'] = sample_rate
    decoded['n_samples'] = len(data)
    return decoded


def match_barcodes(main_times, main_values, secondary_times, secondary_values,
                    robust=True, outlier_threshold_ms=50.0):
    """Match two decoded barcode streams by exact value; fit secondary->main time.

    Parameters
    ----------
    main_times, secondary_times : arrays of timestamps in SECONDS (any common,
        internally-consistent reference is fine -- see fit_alignment for how
        this module derives them from raw sample indices + nominal rates).
    main_values, secondary_values : decoded integer barcode values.
    robust : bool
        If True (default), fit slope/intercept by least-squares regression over
        ALL shared barcodes -- unlike a 2-point (first/last shared barcode
        only) fit, this is not dominated by timing jitter at either endpoint.
        One round of residual-based outlier rejection (> outlier_threshold_ms)
        is then applied and the fit is redone.

    Returns a dict with shared_values, main_times, secondary_times (matched
    subset), slope, intercept, residual_ms, n_outliers_excluded, such that
    main_time ~= secondary_time * slope + intercept.
    """
    # A value seen more than once in a stream (e.g. a leftover phantom 32767, see decode_barcode)
    # can't be paired to one instant -- intersect1d would silently take its first occurrence.
    def _unique_only(times, values):
        vals, counts = np.unique(values, return_counts=True)
        keep = np.isin(values, vals[counts == 1])
        return times[keep], values[keep]
    main_times, main_values = _unique_only(np.asarray(main_times), np.asarray(main_values))
    secondary_times, secondary_values = _unique_only(np.asarray(secondary_times), np.asarray(secondary_values))

    shared_values, main_idx, secondary_idx = np.intersect1d(
        main_values, secondary_values, return_indices=True)

    if len(shared_values) < 2:
        raise ValueError(
            f"Only {len(shared_values)} shared barcode(s) found between the two streams; "
            "need at least 2 to fit an alignment. Check decode height/quality first.")

    main_shared_times = main_times[main_idx]
    secondary_shared_times = secondary_times[secondary_idx]

    n_outliers = 0
    if robust and len(shared_values) > 2:
        m, b = np.polyfit(secondary_shared_times, main_shared_times, 1)
        residual_ms = 1000 * (main_shared_times - (secondary_shared_times * m + b))
        keep = np.abs(residual_ms) <= outlier_threshold_ms
        n_outliers = int(np.sum(~keep))
        if n_outliers > 0 and keep.sum() >= 2:
            m, b = np.polyfit(secondary_shared_times[keep], main_shared_times[keep], 1)
            residual_ms = 1000 * (main_shared_times - (secondary_shared_times * m + b))
    else:
        m = ((main_shared_times[-1] - main_shared_times[0])
             / (secondary_shared_times[-1] - secondary_shared_times[0]))
        b = main_shared_times[0] - secondary_shared_times[0] * m
        residual_ms = 1000 * (main_shared_times - (secondary_shared_times * m + b))

    return {
        'shared_values': shared_values,
        'main_times': main_shared_times,
        'secondary_times': secondary_shared_times,
        'slope': m,
        'intercept': b,
        'residual_ms': residual_ms,
        'n_outliers_excluded': n_outliers,
    }


def fit_alignment(main_decoded, secondary_decoded, secondary_nominal_rate):
    """Compute the corrected secondary-DAQ sample rate + anchor offset from two decoded streams.

    `main_decoded`/`secondary_decoded` are dicts as returned by `read_syncttl`/`decode_barcode`
    (main = funcimg/vDAQ, secondary = behav/NIDAQ, matching Aimbot_barcoding's
    align_barcode_between_nwb convention exactly, so `delta_t` here is a drop-in
    replacement for its `calculate_temporal_difference` return value).

    Derivation: let t_main = sample_main / main_rate (vDAQ's declared rate is
    treated as accurate -- it's what everything else gets referenced to) and
    t_secondary_nominal = sample_secondary / secondary_nominal_rate. Barcode
    matching gives t_main ~= slope * t_secondary_nominal + intercept for the
    SAME real instants. Since this must hold for every matched instant (not
    just the anchor), 1/secondary_true_rate = slope / secondary_nominal_rate,
    i.e. secondary_true_rate = secondary_nominal_rate / slope. `delta_t` is
    just `intercept`, in seconds -- structurally identical to Aimbot's
    existing `intercept / main_sample_rate` (here `main_times` are already in
    seconds, so no extra division is needed).

    Returns a dict with delta_t, corrected_secondary_rate, slope, and the
    full `match_barcodes` output (for quality checking/flagging).
    """
    main_times = main_decoded['start_sample'] / main_decoded['sample_rate']
    secondary_times = secondary_decoded['start_sample'] / secondary_nominal_rate

    match = match_barcodes(
        main_times=main_times, main_values=main_decoded['values'],
        secondary_times=secondary_times, secondary_values=secondary_decoded['values'],
    )

    return {
        'delta_t': match['intercept'],
        'corrected_secondary_rate': secondary_nominal_rate / match['slope'],
        'slope': match['slope'],
        'match': match,
    }

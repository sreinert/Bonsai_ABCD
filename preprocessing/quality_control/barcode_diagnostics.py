"""Read-only barcode decoding/matching for alignment sanity-checking.

This is a pure-function port of the decode/match algorithm in
preprocessing/barcode/extract_barcodes.py and preprocessing/barcode/align.py.
Unlike those scripts, nothing here reads or writes files (no np.save, no
np.fromfile) -- every function takes arrays already in memory and returns
arrays. This makes it safe to run against rawdata for sanity-checking without
risk of ever writing into it.
"""

import numpy as np
from scipy.signal import find_peaks


def decode_barcode(raw_signal, sample_rate, height=2.0, global_tolerance=0.2,
                    nbits=16, ind_wrap_duration=10, ind_bar_duration=30):
    """Decode Arduino-style barcodes from a raw analog/digital TTL trace.

    Parameters
    ----------
    raw_signal : 1D array
        Raw analog or digital trace containing the barcode pulse train.
    sample_rate : float
        Sampling rate of raw_signal, in Hz.
    height : float
        Peak-detection threshold on the absolute derivative of raw_signal.
        NOTE: decoding is very sensitive to this value -- inspect the
        transition plot before trusting the decoded values.
    global_tolerance : float
        Fractional tolerance (0.2 = 20%) allowed on wrapper/bit durations.
    nbits, ind_wrap_duration, ind_bar_duration : Arduino barcode format
        parameters (16-bit code, 10 ms wrapper pulses, 30 ms per bit),
        must match preprocessing/barcode/arduino_barcodes.ino.

    Returns
    -------
    dict with:
        start_sample : int64 array, sample index of each decoded barcode's
            wrapper start (same index space as raw_signal).
        values : int64 array, decoded 16-bit barcode value at each start_sample.
        transition_sample : int array, every detected TTL transition
            (for plotting raw trace + detected events).
        n_skipped : int, number of candidate wrapper windows that could not
            be decoded (missing on/off times) -- large values suggest `height`
            is wrong or the signal is noisy.
    """
    sample_conversion = 1000 / sample_rate  # ms per sample
    wrap_duration = 3 * ind_wrap_duration
    total_barcode_duration = nbits * ind_bar_duration + 2 * wrap_duration
    min_wrap_duration = ind_wrap_duration * (1 - global_tolerance)
    max_wrap_duration = ind_wrap_duration * (1 + global_tolerance)

    event_index, _ = find_peaks(np.abs(np.diff(raw_signal)), height=height)
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
    """Indices i where values[i+1] does not equal values[i] + 1.

    A correctly decoded, gap-free barcode stream increments by exactly 1
    every ~1000 ms. Non-1 steps indicate a dropped/duplicated/misdecoded
    barcode at that point in the recording.
    """
    steps = np.diff(values)
    return np.where(steps != 1)[0]


def match_barcodes(main_times, main_values, secondary_times, secondary_values,
                    robust=True, outlier_threshold_ms=50.0):
    """Match two decoded barcode streams by exact value; fit secondary->main time.

    Takes already-decoded (time, value) arrays instead of re-reading files, and
    returns the fit instead of saving it.

    Parameters
    ----------
    main_times, secondary_times : arrays of timestamps (any consistent unit,
        e.g. seconds) for each decoded barcode in the "main"/"secondary" stream.
    main_values, secondary_values : decoded integer barcode values, same
        length/order as the corresponding *_times array.
    robust : bool
        If True (default), fit slope/intercept by least-squares regression over
        ALL shared barcodes -- unlike preprocessing/barcode/align.py's original
        method, which only uses the first and last shared barcode and is
        therefore very sensitive to timing jitter/misdecodes at either endpoint.
        After the initial fit, points whose residual exceeds
        `outlier_threshold_ms` are dropped and the fit is redone once.
        If False, reproduces align.py's original 2-point method exactly.

    Returns
    -------
    dict with shared_values, main_times, secondary_times (subset actually
    matched, in shared-value order), slope, intercept, residual_ms (main_time -
    predicted, in ms, for every shared barcode, including any excluded as
    outliers), and n_outliers_excluded, such that
    main_time ~= secondary_time * slope + intercept.
    """
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

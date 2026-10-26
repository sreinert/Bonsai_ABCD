"""Standalone sanity checks for a behav/funcimg/merged NWB triple.

Function-ized version of the barcode-alignment checks in
preprocessing/quality_control/session_sanity_checking.ipynb (Step 3), adapted to run directly
against freshly-built NWBs (no behaviour_data.npz/valid_frames.npz dependency,
since those don't exist until the separate read_align_timestamps_bonsai.ipynb
step runs downstream of NWB creation). Can be called standalone against any
already-built session, or automatically by build_session_nwb.py right after
building one.
"""

from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO

from . import barcode_alignment as ba


def _flag(flags, message):
    flags.append(message)
    print(f"[FLAG] {message}")


def _get_ttl(nwb_obj, event_name):
    types = nwb_obj.acquisition['NIDAQ_TTLTypesTable'][:]
    ttls = nwb_obj.acquisition['NIDAQ_TTLsTable'][:]
    type_idx = types[types['event_name'] == event_name].index.tolist()
    return ttls.loc[ttls['ttl_type'].isin(type_idx), 'timestamp'].values


def check_structure(nwb_path, required, label):
    """Confirm the expected acquisition/processing objects exist."""
    flags = []
    with NWBHDF5IO(str(nwb_path), mode='r') as io:
        nwb = io.read()
        for name in required.get('acquisition', []):
            if name not in nwb.acquisition:
                _flag(flags, f"{label} NWB is missing expected acquisition object '{name}'.")
        if required.get('ophys_dfoverf'):
            try:
                nwb.processing['ophys']['DfOverF']
            except KeyError:
                _flag(flags, f"{label} NWB has no processing/ophys/DfOverF.")
        if required.get('ophys_segmentation'):
            try:
                nwb.processing['ophys']['ImageSegmentation']
            except KeyError:
                _flag(flags, f"{label} NWB has no processing/ophys/ImageSegmentation.")
    return flags


def check_barcode_quality(decoded, label, expected_duration_s):
    """Barcode-count sanity check + dropped-barcode (non-consecutive-value) check.

    Mirrors preprocessing/quality_control/session_sanity_checking.ipynb's Step 3 checks: barcodes
    repeat every ~1000 ms, so decoded count should track session duration, and a
    correctly decoded stream increments by exactly 1 every barcode.
    """
    flags = []
    n = len(decoded['values'])
    ratio = n / expected_duration_s if expected_duration_s > 0 else np.nan
    print(f"{label}: decoded {n} barcodes ({decoded['n_skipped']} windows skipped), "
          f"session ~{expected_duration_s:.0f}s (ratio {ratio:.2f}x)")
    if ratio < 0.7 or ratio > 1.3:
        _flag(flags, f"{label} decoded {n} barcodes but the session is ~{expected_duration_s:.0f}s long "
              f"(expect ~1/s) -- ratio {ratio:.2f}x. The barcode `height` parameter is likely wrong "
              f"for this session (duplicate or missed transitions).")

    dropped = ba.find_dropped_barcodes(decoded['values'])
    if len(dropped) > 0:
        shown = dropped[:10].tolist()
        suffix = f" ... (+{len(dropped) - 10} more)" if len(dropped) > 10 else ""
        if not ba.barcode_discontinuities_are_sparse(decoded['values']):
            _flag(flags, f"{label} barcode stream: {len(dropped)}/{max(n-1,0)} steps are non-consecutive -- "
                  f"decode is unreliable for this session, not just a handful of dropped barcodes.")
        else:
            boundary = [int(i) for i in dropped if i in (0, n - 2)]
            detail = " (recording-boundary decode)" if len(boundary) == len(dropped) else ""
            print(f"[INFO] {label} barcode stream has {len(dropped)} isolated non-consecutive step(s) "
                  f"at decoded-barcode index {shown}{suffix}{detail}; exact-value matching excludes them.")
    return flags


def check_fit_quality(fit):
    """Flag a poor barcode fit (few shared barcodes, large residuals, many outliers)."""
    flags = []
    match = fit['match']
    n_shared = len(match['shared_values'])
    mean_abs = float(np.mean(np.abs(match['residual_ms'])))
    max_abs = float(np.max(np.abs(match['residual_ms'])))
    print(f"Alignment fit: {n_shared} shared barcodes, mean|residual|={mean_abs:.2f} ms, "
          f"max|residual|={max_abs:.2f} ms, {match['n_outliers_excluded']} outlier(s) excluded")
    print(f"  delta_t={fit['delta_t']:.6f} s, corrected secondary rate={fit['corrected_secondary_rate']:.4f} Hz "
          f"(slope={fit['slope']:.8f})")
    if n_shared < 10:
        _flag(flags, f"Only {n_shared} shared barcodes found -- alignment fit is based on very little data.")
    if match['n_outliers_excluded'] > 0:
        _flag(flags, f"Barcode fit excluded {match['n_outliers_excluded']} outlier barcode(s) out of "
              f"{n_shared} shared barcodes -- check decode quality.")
    if max_abs > 20:
        _flag(flags, f"Barcode fit residual reaches {max_abs:.1f} ms even after the robust fit -- "
              f"decode quality may be marginal for this session (see barcode-count checks above).")
    return flags


def check_merged_reproduces_fit(merged_nwb_path, behav_nwb_path, fit):
    """Cross-check: does the merged NWB's aligned NIDAQ_TTLsTable actually match
    `native_timestamp * slope + intercept`, i.e. was the corrected rate/delta_t
    actually applied correctly when the merged NWB was written?

    Unlike the equivalent check in session_sanity_checking.ipynb (which
    independently re-derives the fit from raw analog data as an external sanity
    check on the *upstream* pipeline's own, separately-computed alignment), this
    reuses the exact same `fit` used to build the file -- so a large discrepancy
    here means a wiring bug in THIS pipeline (e.g. the corrected rate not
    actually reaching the interface before conversion), not a decode-quality
    issue.
    """
    flags = []
    with NWBHDF5IO(str(behav_nwb_path), mode='r') as io:
        native_camera = _get_ttl(io.read(), 'CameraFrameTTL')
    with NWBHDF5IO(str(merged_nwb_path), mode='r') as io:
        aligned_camera = _get_ttl(io.read(), 'CameraFrameTTL')

    predicted = native_camera * fit['slope'] + fit['delta_t']
    n = min(len(predicted), len(aligned_camera))
    diff_ms = 1000 * (predicted[:n] - aligned_camera[:n])
    mean_abs, max_abs = float(np.mean(np.abs(diff_ms))), float(np.max(np.abs(diff_ms)))
    print(f"Merged-NWB reproduction check: mean|diff|={mean_abs:.3f} ms, max|diff|={max_abs:.3f} ms "
          f"(n={n} camera frames)")
    if max_abs > 1.0:
        _flag(flags, f"Merged NWB's aligned camera timestamps differ from the fit's own prediction by "
              f"up to {max_abs:.2f} ms -- the corrected rate/delta_t may not have been applied correctly "
              f"when building the merged NWB (this indicates a bug in nwb_barcoding itself, not a "
              f"decode-quality issue).")
    return flags


def verify(merged_nwb_path, behav_nwb_path, funcimg_nwb_path, fit=None,
           nidaq_nominal_rate=9000.0, nidaq_barcode_height=0.5, vdaq_barcode_height=0.5):
    """Run all checks against a behav/funcimg/merged NWB triple. Returns a list of flag strings."""
    flags = []

    print("--- Structure ---")
    flags += check_structure(behav_nwb_path, dict(acquisition=[
        'NIDAQ_TTLTypesTable', 'NIDAQ_TTLsTable', 'Position', 'Licks', 'Treadmill', 'SyncTTL',
    ]), 'behav')
    flags += check_structure(funcimg_nwb_path, dict(acquisition=[
        'vDAQ_TTLTypesTable', 'vDAQ_TTLsTable', 'SyncTTL',
    ], ophys_dfoverf=True, ophys_segmentation=True), 'funcimg')
    flags += check_structure(merged_nwb_path, dict(acquisition=[
        'NIDAQ_TTLTypesTable', 'NIDAQ_TTLsTable', 'vDAQ_TTLTypesTable', 'vDAQ_TTLsTable',
        'Position', 'Licks', 'SyncTTL',
    ], ophys_dfoverf=True, ophys_segmentation=True), 'merged')

    print("\n--- Barcode decode + fit ---")
    barcode_funcimg = ba.read_syncttl(funcimg_nwb_path, height=vdaq_barcode_height)
    barcode_behav = ba.read_syncttl(behav_nwb_path, height=nidaq_barcode_height)
    session_duration_s = barcode_behav['n_samples'] / nidaq_nominal_rate
    flags += check_barcode_quality(barcode_funcimg, 'funcimg', session_duration_s)
    flags += check_barcode_quality(barcode_behav, 'behav', session_duration_s)

    if fit is None:
        fit = ba.fit_alignment(main_decoded=barcode_funcimg, secondary_decoded=barcode_behav,
                                secondary_nominal_rate=nidaq_nominal_rate)
    flags += check_fit_quality(fit)

    print("\n--- Merged NWB reproduces the fit ---")
    flags += check_merged_reproduces_fit(merged_nwb_path, behav_nwb_path, fit)

    print(f"\n{len(flags)} issue(s) flagged." if flags else "\nNo issues flagged.")
    return flags


if __name__ == "__main__":
    import defopt
    defopt.run(verify)

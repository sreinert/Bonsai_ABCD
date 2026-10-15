"""Sanity checks for a cohort2 behav/funcimg/merged NWB triple.

Reuses verify_session.py's generic checks (structure, barcode count/consecutiveness, fit
quality) with cohort2's object names. Two checks are cohort2-specific:
  - the merged-NWB reproduction check uses the reward TTL (verify_session's version needs
    CameraFrameTTL, which cohort2 doesn't have under that name);
  - the VR-clock -> NIDAQ-clock reward fit gets the same kind of quality flags as the
    barcode fit, since every VR-logged value in the NWBs depends on it.
"""

import numpy as np
from pynwb import NWBHDF5IO

from . import barcode_alignment as ba
from . import cohort2_layout as layout
from .verify_session import _flag, _get_ttl, check_structure, check_barcode_quality, check_fit_quality


def check_vr_fit(vr_fit):
    flags = []
    residual = np.abs(vr_fit['residual_ms'])
    print(f"VR->NIDAQ fit: {vr_fit['n_matched']} matched rewards (VR logged {vr_fit['n_vr']}, NIDAQ "
          f"recorded {vr_fit['n_nidaq']}), mean|residual|={residual.mean():.1f} ms, "
          f"max|residual|={residual.max():.1f} ms, {vr_fit['n_outliers_excluded']} outlier(s) excluded, "
          f"VR clock {1e6 * (vr_fit['slope'] - 1):.0f} ppm off NIDAQ")
    if vr_fit['n_matched'] < min(vr_fit['n_vr'], vr_fit['n_nidaq']):
        _flag(flags, f"Only {vr_fit['n_matched']} of {min(vr_fit['n_vr'], vr_fit['n_nidaq'])} rewards could be "
              f"paired between the VR log and the NIDAQ reward channel.")
    if vr_fit['n_vr'] != vr_fit['n_nidaq']:
        _flag(flags, f"VR log has {vr_fit['n_vr']} rewards but the NIDAQ reward channel has "
              f"{vr_fit['n_nidaq']} pulses.")
    if residual.max() > 50:
        _flag(flags, f"VR->NIDAQ reward fit residual reaches {residual.max():.1f} ms -- VR-logged "
              f"position/events may be misaligned by that much.")
    return flags


def check_merged_reproduces_fit(merged_nwb_path, behav_nwb_path, fit, ttl_name='RewardTTL'):
    """Merged NWB's aligned TTLs must equal native TTLs * slope + delta_t (a wiring check
    on this pipeline, not on decode quality -- see verify_session.check_merged_reproduces_fit)."""
    flags = []
    with NWBHDF5IO(str(behav_nwb_path), mode='r') as io:
        native = _get_ttl(io.read(), ttl_name)
    with NWBHDF5IO(str(merged_nwb_path), mode='r') as io:
        aligned = _get_ttl(io.read(), ttl_name)
    if len(native) == 0 or len(native) != len(aligned):
        _flag(flags, f"Cannot run the reproduction check: {len(native)} native vs {len(aligned)} "
              f"merged '{ttl_name}' events.")
        return flags
    diff_ms = 1000 * (native * fit['slope'] + fit['delta_t'] - aligned)
    max_abs = float(np.max(np.abs(diff_ms)))
    print(f"Merged-NWB reproduction check ({ttl_name}, n={len(native)}): max|diff|={max_abs:.3f} ms")
    if max_abs > 1.0:
        _flag(flags, f"Merged NWB's aligned {ttl_name} times differ from the fit's prediction by up to "
              f"{max_abs:.2f} ms -- the alignment wasn't applied correctly (a bug in this pipeline).")
    return flags


def verify(merged_nwb_path, behav_nwb_path, funcimg_nwb_path, fit=None, vr_fit=None,
           nidaq_barcode_height=0.5, vdaq_barcode_height=0.5):
    """Run all checks against a cohort2 NWB triple. Returns a list of flag strings."""
    flags = []

    print("--- Structure ---")
    flags += check_structure(behav_nwb_path, dict(acquisition=[
        'NIDAQ_TTLTypesTable', 'NIDAQ_TTLsTable', 'Position', 'TotalRunDistance', 'SyncTTL',
    ]), 'behav')
    flags += check_structure(funcimg_nwb_path, dict(acquisition=[
        'vDAQ_TTLTypesTable', 'vDAQ_TTLsTable', 'SyncTTL',
    ], ophys_dfoverf=True, ophys_segmentation=True), 'funcimg')
    flags += check_structure(merged_nwb_path, dict(acquisition=[
        'NIDAQ_TTLTypesTable', 'NIDAQ_TTLsTable', 'vDAQ_TTLTypesTable', 'vDAQ_TTLsTable',
        'Position', 'TotalRunDistance', 'SyncTTL',
    ], ophys_dfoverf=True, ophys_segmentation=True), 'merged')

    print("\n--- Barcode decode + fit ---")
    barcode_funcimg = ba.read_syncttl(funcimg_nwb_path, height=vdaq_barcode_height)
    barcode_behav = ba.read_syncttl(behav_nwb_path, height=nidaq_barcode_height)
    session_duration_s = barcode_behav['n_samples'] / layout.NIDAQ_NOMINAL_RATE
    flags += check_barcode_quality(barcode_funcimg, 'funcimg', session_duration_s)
    flags += check_barcode_quality(barcode_behav, 'behav', session_duration_s)
    if fit is None:
        fit = ba.fit_alignment(main_decoded=barcode_funcimg, secondary_decoded=barcode_behav,
                               secondary_nominal_rate=layout.NIDAQ_NOMINAL_RATE)
    flags += check_fit_quality(fit)

    if vr_fit is not None:
        print("\n--- VR clock fit ---")
        flags += check_vr_fit(vr_fit)

    print("\n--- Merged NWB reproduces the fit ---")
    flags += check_merged_reproduces_fit(merged_nwb_path, behav_nwb_path, fit)

    print(f"\n{len(flags)} issue(s) flagged." if flags else "\nNo issues flagged.")
    return flags

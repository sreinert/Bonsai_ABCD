"""Cohort2 (HFScohort2, pre-Bonsai) session layout, raw-file loading and VR-clock fitting.

Cohort2 predates the Bonsai rig, so nothing about its on-disk layout matches what
build_session_nwb.py (cohort3/GoalProgress) expects:

  <root>/<mouse>/ses-012_date-20250318_protocol-t6/
      behav/
          20250318_M8_ai.bin          NIDAQ analog input, float64, interleaved channels @ 9000 Hz
          20250318_M8_ci.bin          NIDAQ rotary counter, float64, one value per ai sample
          250318/position_log.csv     VR log (Index, Time, Position, TotalRunDistance, Event),
                                      on the VR PC's own clock -- NOT the NIDAQ clock
          250318/config.yaml          VR task config (flip_tunnel landmarks/goals, sequence_task...)
      funcimg/
          <short pre-session test recording, ignored>
          Session/
              <YYYYMMDD_HHMMSS>__<mouse>_00001.h5   vDAQ recording (FrameTTL, SyncTTL, ...) @ 5000 Hz
              suite2p/plane0/                        suite2p output, written straight into this tree

Everything cohort2-specific about *where things are and how to read them* lives here,
so the NWB/behaviour-data builders only deal with already-loaded arrays. Only numpy/
pandas/yaml are needed (no pynwb), so this module also imports fine in the
`bonsai_abcd` env if you want to poke at a session interactively.

No file is ever written by this module.
"""

import re
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

NIDAQ_NOMINAL_RATE = 9000.0

# Column -> signal in *_ai.bin, as used by preprocessing/read_align_timestamps.ipynb
# (barcode: extract_barcodes.py signals_column=0; licks col 1; rewards col 2; MODDs cols 4/5).
# Column 3 is a ~38 Hz pulse train: the behaviour camera frame strobe (the only camera
# channel on this rig; named as in the cohort3 NWBs).
# t0 sessions were recorded with only the first 4 channels (no MODD1/MODD2).
AI_CHANNELS = {
    0: 'SyncTTL',
    1: 'LickTTL',
    2: 'RewardTTL',
    3: 'CameraFrameTTL',
    4: 'MODD1ONTTL',
    5: 'MODD2ONTTL',
}

# Volts. The raw analog pulses peak at ~4.3-5 V; the original cohort2 notebook detects events
# at > 1 V, which is kept here so event counts stay comparable with the existing
# behaviour_data.npz files in the raw tree.
TTL_THRESHOLD_V = 1.0

VR_REWARD_EVENTS = ('rewarded', 'assist-rewarded', 'manually-rewarded')


# --------------------------------------------------------------------------------------
# Session discovery
# --------------------------------------------------------------------------------------

def resolve_session(mouse_dir, session_tag):
    """Find a session folder under `mouse_dir` from a tag.

    Accepts, in order of precedence: the full folder name
    ("ses-012_date-20250318_protocol-t6"), a "ses-XXX" prefix ("ses-012"), or a protocol
    tag ("t6"). Protocol tags are matched as a *suffix* ("..._protocol-t6"), so "t1" never
    matches "t10"/"t11". Some protocols were run on two days (e.g. TAA0000066 has two t4
    sessions); a protocol tag that matches more than one folder raises instead of silently
    picking one -- pass the ses-XXX prefix to choose.
    """
    mouse_dir = Path(mouse_dir)
    folders = sorted(p.name for p in mouse_dir.iterdir() if p.is_dir() and p.name.startswith('ses-'))

    if session_tag in folders:
        return session_tag

    prefix_matches = [f for f in folders if f.split('_')[0] == session_tag]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    suffix_matches = [f for f in folders if f.endswith(f'protocol-{session_tag}')]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        raise ValueError(f"Protocol tag '{session_tag}' matches {len(suffix_matches)} sessions under "
                         f"{mouse_dir}: {suffix_matches} -- pass the ses-XXX prefix instead.")
    raise FileNotFoundError(f"No session folder under {mouse_dir} matches '{session_tag}'.")


def list_vr_imaging_sessions(mouse_dir):
    """All session folders that have both VR/NIDAQ behaviour and a suite2p'd imaging session."""
    mouse_dir = Path(mouse_dir)
    sessions = []
    for p in sorted(mouse_dir.iterdir()):
        if not (p.is_dir() and p.name.startswith('ses-')):
            continue
        try:
            find_session_files(p)
        except FileNotFoundError:
            continue
        sessions.append(p.name)
    return sessions


def find_session_files(session_path):
    """Locate every raw file a cohort2 session build needs. Raises FileNotFoundError on any gap."""
    session_path = Path(session_path)
    behav = session_path / 'behav'
    funcimg = session_path / 'funcimg' / 'Session'

    def _one(pattern_dir, pattern, what):
        matches = sorted(pattern_dir.glob(pattern)) if pattern_dir.exists() else []
        matches = [m for m in matches if not m.name.startswith('._')]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one {what} ({pattern_dir}/{pattern}), found "
                                    f"{len(matches)}: {[m.name for m in matches]}")
        return matches[0]

    vr_dirs = [d for d in behav.iterdir() if d.is_dir() and re.fullmatch(r'\d{6}', d.name)] \
        if behav.exists() else []
    if len(vr_dirs) != 1:
        raise FileNotFoundError(f"Expected exactly one YYMMDD VR folder under {behav}, found "
                                f"{[d.name for d in vr_dirs]}")
    vr_dir = vr_dirs[0]

    files = dict(
        session_path=session_path,
        ai_bin=_one(behav, '*_ai.bin', 'NIDAQ analog file'),
        ci_bin=_one(behav, '*_ci.bin', 'NIDAQ counter file'),
        position_log=_one(vr_dir, 'position_log.csv', 'VR position log'),
        config=_one(vr_dir, 'config.yaml', 'VR config'),
        vdaq_h5=_one(funcimg, '*.h5', 'vDAQ .h5 recording'),
        suite2p_dir=funcimg / 'suite2p',
        funcimg_dir=funcimg,
    )
    if not (files['suite2p_dir'] / 'plane0' / 'stat.npy').exists():
        raise FileNotFoundError(f"No suite2p output at {files['suite2p_dir'] / 'plane0'}")
    return files


def session_start_time(vdaq_h5):
    """Naive datetime of the imaging session start, parsed from the vDAQ file name
    ("20250318_152251__TAA0000066_00001.h5"). Used for both NWBs' session_start_time."""
    import datetime as dt
    match = re.match(r'(\d{8}_\d{6})__', Path(vdaq_h5).name)
    if match is None:
        raise ValueError(f"Could not parse a start time from vDAQ file name {Path(vdaq_h5).name}")
    return dt.datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')


# --------------------------------------------------------------------------------------
# Raw loading
# --------------------------------------------------------------------------------------

def ai_channel_count(ai_bin, ci_bin):
    """Number of interleaved channels in *_ai.bin.

    The counter file holds one float64 per analog sample, so the byte-size ratio of the two
    files is the analog channel count -- no need to infer it from the protocol (the original
    notebook hardcoded 4 for t0 and 6 otherwise). The two files can differ by an acquisition
    buffer (e.g. t2 of TAA0000066: ai has exactly 1000 more samples than ci), so the ratio
    only has to be close; the ai file must still split into whole samples.
    """
    ai_size, ci_size = Path(ai_bin).stat().st_size, Path(ci_bin).stat().st_size
    n = round(ai_size / ci_size)
    sample_mismatch = abs(ai_size / (8 * n) - ci_size / 8) / (ci_size / 8) if n else np.inf
    if n not in (4, 6) or ai_size % (8 * n) or sample_mismatch > 0.01:
        raise ValueError(f"Unexpected ai/ci size ratio {ai_size / ci_size:.4f} for {ai_bin} -- "
                         f"expected 4 or 6 analog channels.")
    return int(n)


@functools.lru_cache(maxsize=1)
def load_ai(ai_bin, n_channels):
    """(n_samples, n_channels) float64 array from *_ai.bin. Cached (one session at a time),
    since it's ~1 GB on the network share and several build steps need it."""
    data = np.fromfile(str(ai_bin), dtype=np.float64)
    return data.reshape(-1, n_channels)


def load_vr_log(position_log):
    return pd.read_csv(position_log)


def load_config(config):
    with open(config) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def detect_rising_edges(signal, threshold=TTL_THRESHOLD_V, min_refractory_samples=5):
    """Sample indices where `signal` crosses `threshold` upwards, merging edges <=
    min_refractory_samples apart (same guard as build_behaviour_data.detect_rising_edges:
    a slow multi-sample TTL rise must not count as two events).

    Returns the index of the last sample *below* threshold, matching the notebook's
    np.where(np.diff(binary) == 1) convention.
    """
    above = np.asarray(signal) > threshold
    idx = np.flatnonzero(~above[:-1] & above[1:])
    if len(idx) < 2:
        return idx
    keep = np.ones(len(idx), dtype=bool)
    last = idx[0]
    for i in range(1, len(idx)):
        if idx[i] - last <= min_refractory_samples:
            keep[i] = False
        else:
            last = idx[i]
    return idx[keep]


# --------------------------------------------------------------------------------------
# VR clock -> NIDAQ clock
# --------------------------------------------------------------------------------------

def vr_reward_times(vr_log):
    return vr_log['Time'].values[vr_log['Event'].isin(VR_REWARD_EVENTS).values]


def fit_vr_to_nidaq(vr_rewards, nidaq_rewards, match_window_s=1.0, outlier_threshold_ms=50.0):
    """Map VR-log time -> NIDAQ time (seconds from NIDAQ sample 0, nominal 9000 Hz clock).

    The VR PC logs every reward it triggers (rewarded / assist-rewarded / manually-rewarded)
    and the NIDAQ records the same valve TTL, so rewards are shared events on both clocks.
    The original notebook anchored on the *first* reward only (a constant offset); this
    fits slope + intercept over *all* matched rewards instead, for the same reason
    barcode_alignment.py does for the two DAQs: two independent clocks drift apart, and an
    offset-only mapping is exact only at its anchor.

    Matching: shift VR rewards by the first-reward offset, pair each with the nearest NIDAQ
    reward within `match_window_s`, fit, then re-pair with the fit and fit again (with one
    round of > outlier_threshold_ms residual rejection). Works even if one side has a few
    extra/missing rewards (e.g. a manual flush not logged by the VR).

    Returns dict(slope, intercept, residual_ms, n_matched, n_vr, n_nidaq, n_outliers_excluded)
    such that nidaq_time ~= vr_time * slope + intercept.
    """
    vr_rewards = np.asarray(vr_rewards, dtype=float)
    nidaq_rewards = np.asarray(nidaq_rewards, dtype=float)
    if len(vr_rewards) < 2 or len(nidaq_rewards) < 2:
        raise ValueError(f"Need >= 2 rewards on both clocks to fit VR->NIDAQ time "
                         f"(VR: {len(vr_rewards)}, NIDAQ: {len(nidaq_rewards)}).")

    def _pair(predicted):
        j = np.clip(np.searchsorted(nidaq_rewards, predicted), 1, len(nidaq_rewards) - 1)
        left, right = nidaq_rewards[j - 1], nidaq_rewards[j]
        nearest = np.where(np.abs(predicted - left) <= np.abs(predicted - right), j - 1, j)
        ok = np.abs(nidaq_rewards[nearest] - predicted) <= match_window_s
        return np.flatnonzero(ok), nearest[ok]

    slope, intercept = 1.0, nidaq_rewards[0] - vr_rewards[0]
    for _ in range(2):
        vr_i, ni_i = _pair(vr_rewards * slope + intercept)
        if len(vr_i) < 2:
            raise ValueError("Fewer than 2 VR rewards could be paired with a NIDAQ reward -- "
                             "check the reward channel / VR log.")
        slope, intercept = np.polyfit(vr_rewards[vr_i], nidaq_rewards[ni_i], 1)

    residual_ms = 1000 * (nidaq_rewards[ni_i] - (vr_rewards[vr_i] * slope + intercept))
    keep = np.abs(residual_ms) <= outlier_threshold_ms
    n_outliers = int(np.sum(~keep))
    if n_outliers and keep.sum() >= 2:
        slope, intercept = np.polyfit(vr_rewards[vr_i][keep], nidaq_rewards[ni_i][keep], 1)
        residual_ms = 1000 * (nidaq_rewards[ni_i] - (vr_rewards[vr_i] * slope + intercept))

    return dict(slope=float(slope), intercept=float(intercept), residual_ms=residual_ms,
                n_matched=len(vr_i), n_vr=len(vr_rewards), n_nidaq=len(nidaq_rewards),
                n_outliers_excluded=n_outliers)


def fit_session_vr_clock(files):
    """Convenience: load what's needed from a find_session_files() dict and fit VR->NIDAQ."""
    n_channels = ai_channel_count(files['ai_bin'], files['ci_bin'])
    ai = load_ai(files['ai_bin'], n_channels)
    nidaq_rewards = detect_rising_edges(ai[:, 2]) / NIDAQ_NOMINAL_RATE
    vr_log = load_vr_log(files['position_log'])
    return fit_vr_to_nidaq(vr_reward_times(vr_log), nidaq_rewards)

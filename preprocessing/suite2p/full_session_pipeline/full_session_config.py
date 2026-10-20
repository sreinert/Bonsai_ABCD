"""Locked configuration for the single-session registration and ROI pilot."""

from __future__ import annotations

import re


TIFF_SUFFIXES = {".tif", ".tiff"}
SESSION_PATTERN = re.compile(r"^ses-(?P<processed>.+?)_date-\d{8}T\d{6}(?:_|$)")


def processed_session_id(raw_session_id: str) -> str:
    match = SESSION_PATTERN.match(raw_session_id)
    if match is None:
        raise ValueError(
            f"Session {raw_session_id!r} does not match "
            "ses-<processed-id>_date-YYYYMMDDTHHMMSS"
        )
    return match.group("processed")


SUITE2P_VERSION = "1.1.0"
FS = 45.0
TAU = 0.4
NPLANES = 1
NCHANNELS = 2
FUNCTIONAL_CHAN = 1

# Every unlisted setting is inherited from suite2p.default_settings().
REGISTRATION = {
    "align_by_chan2": True,
    "smooth_sigma": 3.0,
    "smooth_sigma_time": 0,
    "norm_frames": True,
    "nimg_init": 1000,
    "do_bidiphase": False,
    "bidiphase": 0.0,
    "nonrigid": True,
    "block_size": [128, 128],
    "two_step_registration": True,
}

DETECTION = {
    "algorithm": "cellpose",
    "effective_model": "cpsam_v2",
    # This is the detection input validated in the earlier cpsam-meanimg
    # experiment.  Do not inherit Suite2p's "max_proj / meanImg" default.
    "input_image": "meanImg",
    # The default of 5000 materializes several GiB for a 512x512 movie even
    # though this workflow feeds the temporal mean (not max projection) to
    # CPSAM. 500 keeps the same mean-image method with a bounded peak.
    "nbins": 500,
    # Match the visually validated Cellpose GUI configuration. Suite2p's
    # generic [12, 12] default would otherwise rescale the image by 2.5x.
    "diameter": [30.0, 30.0],
    "flow_threshold": 1.0,
    "cellprob_threshold": 0.0,
    "denoise": True,
}

"""Locked configuration for the single-session registration and ROI pilot."""

from __future__ import annotations


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
    "effective_model": "cpsam",
    # This is the detection input validated in the earlier cpsam-meanimg
    # experiment.  Do not inherit Suite2p's "max_proj / meanImg" default.
    "input_image": "meanImg",
    "diameter": [12.0, 12.0],
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
}

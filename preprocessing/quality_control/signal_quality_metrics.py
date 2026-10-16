"""
Raw-signal quality metrics for suite2p output, computed directly from F.npy/Fneu.npy.

These metrics are meant to be comparable across mice/cohorts even when different
suite2p versions or ROI-detection pipelines (functional vs. anatomical/Cellpose)
were used, since they are computed independently from the raw fluorescence traces
rather than relying on suite2p's own per-ROI stat fields (which are not populated
consistently across pipelines, e.g. 'skew'/'std' are absent when ROIs are imported
from a Cellpose segmentation instead of being detected by suite2p itself).
"""
import os
import numpy as np
from scipy import stats


def load_percell_traces(plane0_dir, baseline_percentile=8, event_sigma=3):
    """Load raw traces and compute per-cell dF/F and quality metrics for one
    suite2p plane0 folder. Shared by compute_session_metrics (summary stats)
    and trace-plotting scripts (which need the actual per-frame dF/F).

    Returns a dict with 'error' on failure, otherwise:
        cell_roi_idx : indices into the original (all-ROI) F.npy rows that
            are iscell==1 (or all ROIs, if none are marked as cells)
        F, dff : (n_cells, n_frames) raw and baseline-normalized traces
        noise_sd, peak_dff, skew, snr, event_frac, neu_corr : (n_cells,) arrays
        fs, tau, n_frames : session-level scalars from ops.npy
    """
    plane0_dir = os.fspath(plane0_dir)
    try:
        F = np.load(os.path.join(plane0_dir, "F.npy")).astype(np.float64)
        Fneu = np.load(os.path.join(plane0_dir, "Fneu.npy")).astype(np.float64)
        iscell = np.load(os.path.join(plane0_dir, "iscell.npy"))
        ops = np.load(os.path.join(plane0_dir, "ops.npy"), allow_pickle=True).item()
    except Exception as e:
        return {"error": str(e), "plane0_dir": plane0_dir}

    cell_mask = iscell[:, 0].astype(bool)
    if cell_mask.sum() == 0:
        cell_mask = np.ones(F.shape[0], dtype=bool)

    Fc = F[cell_mask]
    Fneu_c = Fneu[cell_mask]

    F0 = np.percentile(Fc, baseline_percentile, axis=1, keepdims=True)
    F0[F0 <= 0] = np.nan
    dff = (Fc - F0) / F0

    # frame-to-frame diff MAD: robust noise estimate insensitive to slow drift/transients
    diffs = np.diff(dff, axis=1)
    noise_sd = 1.4826 * np.median(np.abs(diffs - np.median(diffs, axis=1, keepdims=True)), axis=1) / np.sqrt(2)

    peak_dff = np.nanpercentile(dff, 99, axis=1)
    skew = stats.skew(dff, axis=1, nan_policy="omit")
    snr = peak_dff / noise_sd
    event_frac = np.mean(dff > event_sigma * noise_sd[:, None], axis=1)
    neu_corr = np.array([np.corrcoef(Fc[i], Fneu_c[i])[0, 1] for i in range(Fc.shape[0])])

    return {
        "plane0_dir": plane0_dir,
        "fs": ops.get("fs"),
        "tau": ops.get("tau"),
        "ops": ops,
        "n_frames": F.shape[1],
        "n_roi_total": F.shape[0],
        "cell_roi_idx": np.nonzero(cell_mask)[0],
        "F": Fc,
        "F0": F0[:, 0],
        "dff": dff,
        "noise_sd": noise_sd,
        "peak_dff": peak_dff,
        "skew": skew,
        "snr": snr,
        "event_frac": event_frac,
        "neu_corr": neu_corr,
    }


def compute_session_metrics(plane0_dir, neuropil_coeff=0.7, baseline_percentile=8,
                             event_sigma=3, snr_threshold=5, skew_threshold=1):
    """Compute raw-signal quality summary metrics for one suite2p plane0 folder.

    Parameters
    ----------
    plane0_dir : str
        Path to a suite2p .../funcimg/suite2p/plane0 folder containing
        F.npy, Fneu.npy, iscell.npy and ops.npy.
    neuropil_coeff : float
        Unused directly (raw F/Fneu correlation is reported instead); kept
        for interface stability.
    baseline_percentile : float
        Percentile of the raw trace used as each cell's F0 for dF/F.
    event_sigma : float
        Number of noise SDs above baseline used to flag a frame as "event-like".
    snr_threshold, skew_threshold : float
        Thresholds used to report the fraction of cells that look like they
        have genuine calcium transients.

    Returns
    -------
    dict of scalar summary metrics, plus 'error' key if loading failed.
    """
    traces = load_percell_traces(plane0_dir, baseline_percentile=baseline_percentile,
                                  event_sigma=event_sigma)
    if "error" in traces:
        return traces
    return summarise_traces(traces, snr_threshold=snr_threshold, skew_threshold=skew_threshold)


def summarise_traces(traces, snr_threshold=5, skew_threshold=1):
    """Session-level summary metrics from an already-loaded load_percell_traces()
    dict. Split out of compute_session_metrics so callers that also need the
    per-cell traces (signal_quality_cache.py) only load F.npy/Fneu.npy once.
    """
    plane0_dir = traces["plane0_dir"]
    mean_img = traces["ops"].get("meanImg")
    mean_img_stats = {}
    if mean_img is not None:
        mean_img_stats = {
            "mean_img_mean": float(np.mean(mean_img)),
            "mean_img_median": float(np.median(mean_img)),
            "mean_img_p99": float(np.percentile(mean_img, 99)),
        }

    return {
        "plane0_dir": plane0_dir,
        "fs": traces["fs"],
        "tau": traces["tau"],
        "n_frames": traces["n_frames"],
        "n_cells": len(traces["cell_roi_idx"]),
        "n_roi_total": traces["n_roi_total"],
        "baseline_F_median": float(np.nanmedian(traces["F0"])),
        "median_skew_dff": float(np.nanmedian(traces["skew"])),
        "median_peak_dff_pct99": float(np.nanmedian(traces["peak_dff"])),
        "median_snr": float(np.nanmedian(traces["snr"])),
        "frac_cells_snr_gt_thresh": float(np.mean(traces["snr"] > snr_threshold)),
        "frac_cells_skew_gt_thresh": float(np.mean(traces["skew"] > skew_threshold)),
        "median_event_frac": float(np.nanmedian(traces["event_frac"])),
        "median_neu_corr": float(np.nanmedian(traces["neu_corr"])),
        **mean_img_stats,
    }


def compute_channel2_metrics(plane0_dir):
    """Quality-control metrics for the structural (e.g. red, activity-independent)
    channel 2, meant as a control for whether a functional-channel brightness
    deficit reflects a general imaging problem (dim in both channels) or an
    indicator/expression-specific one (dim only in channel 1).

    Channel-2 ROI brightness is computed by applying each ROI's suite2p mask
    (from stat.npy, lambda-weighted as suite2p does for F) directly to the
    static mean image of channel 2 (ops['meanImg_chan2']), rather than relying
    on F_chan2.npy/Fneu_chan2.npy, since those are not always extracted (e.g.
    when ROIs are imported from an anatomical Cellpose segmentation rather than
    suite2p's own detection). This keeps the method identical across pipelines.
    Where F_chan2.npy does exist, its time-median and skew are also reported,
    both as a sanity check on the mean-image-based estimate and to confirm
    channel 2 shows no activity-like transients (skew should stay near 0).
    """
    plane0_dir = os.fspath(plane0_dir)
    try:
        ops = np.load(os.path.join(plane0_dir, "ops.npy"), allow_pickle=True).item()
        stat = np.load(os.path.join(plane0_dir, "stat.npy"), allow_pickle=True)
        iscell = np.load(os.path.join(plane0_dir, "iscell.npy"))
    except Exception as e:
        return {"error": str(e), "plane0_dir": plane0_dir}

    mean_img2 = ops.get("meanImg_chan2")
    if mean_img2 is None:
        return {"error": "no meanImg_chan2 in ops", "plane0_dir": plane0_dir}

    cell_mask = iscell[:, 0].astype(bool)
    if cell_mask.sum() == 0:
        cell_mask = np.ones(len(stat), dtype=bool)

    roi_vals = []
    for s in stat[cell_mask]:
        ypix, xpix, lam = s["ypix"], s["xpix"], s["lam"]
        lam_norm = lam / lam.sum()
        roi_vals.append(float(np.sum(lam_norm * mean_img2[ypix, xpix])))
    roi_vals = np.array(roi_vals)

    result = {
        "plane0_dir": plane0_dir,
        "fov_chan2_mean": float(np.mean(mean_img2)),
        "fov_chan2_median": float(np.median(mean_img2)),
        "fov_chan2_p99": float(np.percentile(mean_img2, 99)),
        "roi_chan2_median_brightness": float(np.median(roi_vals)),
    }

    f2_path = os.path.join(plane0_dir, "F_chan2.npy")
    if os.path.exists(f2_path):
        F2 = np.load(f2_path).astype(np.float64)[cell_mask]
        time_median = np.median(F2, axis=1)
        time_skew = stats.skew(F2, axis=1, nan_policy="omit")
        result["roi_chan2_F_time_median"] = float(np.median(time_median))
        result["roi_chan2_F_time_skew_median"] = float(np.nanmedian(time_skew))

    return result

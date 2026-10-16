"""
Per-session caching and settings loading shared by compare_signal_quality.py and
plot_high_snr_traces.py.

Both scripts used to reload F.npy/Fneu.npy from the network share and recompute
everything on every run, which got slow as more sessions were compared. This
module computes each session once and stores two small files:

    signal_quality_metrics.json      scalar summary metrics (compute_session_metrics
                                     + compute_channel2_metrics), one dict
    signal_quality_top_traces.npz    dF/F window of the top-N cells by SNR, plus the
                                     per-cell noise SD/SNR/ROI index needed to plot it
                                     (~0.6 MB for 8 cells x 20000 frames, float32)

By default they are written into the session's suite2p plane0 folder, next to the
files they are derived from. A mouse can set `cache_dir` in the settings file to
put them elsewhere (named `<session>_signal_quality_*`); this is needed for
cohort2, whose suite2p output lives inside the raw data tree, which must not be
written into.

Cache validity: each file stores a `_cache_info` block with METRICS_VERSION, every
parameter that affects the result, and the mtimes of F.npy/Fneu.npy/iscell.npy/
stat.npy. A file is reused only if all of these match, so re-running suite2p,
re-curating iscell or changing a threshold invalidates it automatically. Bump
METRICS_VERSION whenever signal_quality_metrics.py or select_top_cells() change
in a way that alters the output, or pass --recompute to the scripts.

If either file is missing or stale, the session is loaded once and both files are
rebuilt from that single load, so running either script fills the cache for both.
Sessions that fail to load are not cached (so a share dropout is not stored), and
a failed write only prints a warning.
"""
import os
import json
import numpy as np
import yaml

try:  # Package import used by the full-session pipeline.
    from .signal_quality_metrics import (
        compute_channel2_metrics,
        load_percell_traces,
        summarise_traces,
    )
except ImportError:  # Direct execution from preprocessing/quality_control.
    from signal_quality_metrics import (
        compute_channel2_metrics,
        load_percell_traces,
        summarise_traces,
    )

METRICS_VERSION = 1

DEFAULT_SUITE2P_SUBPATH = os.path.join("funcimg", "suite2p", "plane0")
DEFAULT_PARAMS = {
    "baseline_percentile": 8,
    "event_sigma": 3,
    "snr_threshold": 5,
    "skew_threshold": 1,
}
DEFAULT_TRACE_PARAMS = {
    "n_cells": 8,
    "window_frames": 20000,
}
SOURCE_FILES = ("F.npy", "Fneu.npy", "iscell.npy", "stat.npy")


# ---------------------------------------------------------------- settings

def load_settings(path):
    """Read a comparison settings YAML (see settings/ for examples), fill
    defaults and check the required per-mouse fields.
    """
    with open(path) as f:
        settings = yaml.safe_load(f)

    if not settings or "mice" not in settings:
        raise ValueError(f"{path}: settings file needs a 'mice' list")

    settings["output_dir"] = os.path.expanduser(settings.get("output_dir", "~/Desktop"))
    settings.setdefault("output_name", os.path.splitext(os.path.basename(path))[0])
    settings.setdefault("highlight_session", None)
    settings.setdefault("title", None)
    settings.setdefault("chan2_title", None)
    settings["params"] = {**DEFAULT_PARAMS, **(settings.get("params") or {})}
    settings["traces"] = {**DEFAULT_TRACE_PARAMS, **(settings.get("traces") or {})}

    for i, mouse in enumerate(settings["mice"]):
        missing = [k for k in ("label", "base", "color", "sessions") if k not in mouse]
        if missing:
            raise ValueError(f"{path}: mouse #{i} is missing {missing}")
        mouse["base"] = os.path.expanduser(mouse["base"])
        mouse.setdefault("suite2p_subpath", DEFAULT_SUITE2P_SUBPATH)
        if mouse.get("cache_dir"):
            mouse["cache_dir"] = os.path.expanduser(mouse["cache_dir"])
        else:
            mouse["cache_dir"] = None
    return settings


def iter_sessions(settings):
    """Yield (mouse_cfg, session, plane0_dir) for every session in the settings."""
    for mouse in settings["mice"]:
        for ses in mouse["sessions"]:
            yield mouse, ses, os.path.join(mouse["base"], ses, mouse["suite2p_subpath"])


# ---------------------------------------------------------------- top-cell selection

def select_top_cells(traces, n_cells=8, window_frames=20000, snr_threshold=5):
    """Pick the top-n_cells cells by SNR and cut a window_frames-long window from
    the middle of the recording. Returns a dict of plain arrays/scalars (savable
    with np.savez), or {'error': ...} if no cell has a finite SNR.
    """
    snr = traces["snr"]
    n_frames = traces["n_frames"]

    # NaN SNR occurs when a cell's baseline (8th pct. of raw F) is <= 0 (very
    # dim/noisy ROIs); push these to the back rather than letting np.argsort's
    # NaN-last-ascending behaviour put them first after a naive [::-1] reverse.
    finite = np.isfinite(snr)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return {"error": "no cells with finite SNR"}
    order = np.argsort(np.where(finite, snr, -np.inf))[::-1]
    n_passing = int(np.sum(snr > snr_threshold))
    n_show = min(n_cells, n_finite)
    top_idx = order[:n_show]

    half_win = window_frames // 2
    mid = n_frames // 2
    start = max(0, mid - half_win)
    end = min(n_frames, mid + half_win)

    return {
        "dff": traces["dff"][top_idx, start:end].astype(np.float32),
        "noise_sd": traces["noise_sd"][top_idx],
        "snr": snr[top_idx],
        "roi_idx": traces["cell_roi_idx"][top_idx],
        "start": start,
        "end": end,
        "n_frames": n_frames,
        "fs": traces["fs"] if traces["fs"] is not None else np.nan,
        "n_total": traces["dff"].shape[0],
        "n_finite": n_finite,
        "n_passing": n_passing,
        "used_fallback": n_passing == 0,
    }


# ---------------------------------------------------------------- cache

def _cache_paths(plane0_dir, session, cache_dir):
    if cache_dir:
        prefix = os.path.join(cache_dir, f"{session}_signal_quality")
    else:
        prefix = os.path.join(plane0_dir, "signal_quality")
    return prefix + "_metrics.json", prefix + "_top_traces.npz"


def _source_mtimes(plane0_dir):
    mtimes = {}
    for name in SOURCE_FILES:
        p = os.path.join(plane0_dir, name)
        mtimes[name] = os.path.getmtime(p) if os.path.exists(p) else None
    return mtimes


def _cache_info(plane0_dir, params):
    return {
        "metrics_version": METRICS_VERSION,
        "params": params,
        "source_mtimes": _source_mtimes(plane0_dir),
    }


def _read_metrics(path, expected_info):
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None
    if data.pop("_cache_info", None) != expected_info:
        return None
    return data


def _read_top_traces(path, expected_info):
    try:
        with np.load(path, allow_pickle=False) as z:
            info = json.loads(str(z["_cache_info"]))
            if info != expected_info:
                return None
            out = {k: z[k] for k in z.files if k != "_cache_info"}
    except Exception:
        return None
    # 0-d arrays back to Python scalars
    return {k: (v.item() if v.ndim == 0 else v) for k, v in out.items()}


def _write(path, writer):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        writer(tmp)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[warn] could not write cache {path}: {e}")


def _to_json_scalar(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def get_session_results(plane0_dir, session, cache_dir=None, recompute=False,
                        params=None, trace_params=None):
    """Return (metrics, top_traces, source) for one session, from cache if valid.

    metrics : dict of scalars (as compute_session_metrics + compute_channel2_metrics),
        containing 'error' if the session could not be loaded.
    top_traces : dict from select_top_cells, or {'error': ...}.
    source : 'cache' or 'compute'.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    trace_params = {**DEFAULT_TRACE_PARAMS, **(trace_params or {})}
    metrics_path, traces_path = _cache_paths(plane0_dir, session, cache_dir)
    metrics_info = _cache_info(plane0_dir, params)
    traces_info = _cache_info(plane0_dir, {**params, **trace_params})

    if not recompute:
        metrics = _read_metrics(metrics_path, metrics_info)
        top = _read_top_traces(traces_path, traces_info)
        if metrics is not None and top is not None:
            return metrics, top, "cache"

    traces = load_percell_traces(plane0_dir, baseline_percentile=params["baseline_percentile"],
                                 event_sigma=params["event_sigma"])
    if "error" in traces:
        return traces, {"error": traces["error"]}, "compute"

    metrics = summarise_traces(traces, snr_threshold=params["snr_threshold"],
                               skew_threshold=params["skew_threshold"])
    chan2 = compute_channel2_metrics(plane0_dir)
    if "error" in chan2:
        metrics["chan2_error"] = chan2["error"]
    else:
        chan2.pop("plane0_dir", None)
        metrics.update(chan2)
    metrics = {k: _to_json_scalar(v) for k, v in metrics.items()}

    top = select_top_cells(traces, n_cells=trace_params["n_cells"],
                           window_frames=trace_params["window_frames"],
                           snr_threshold=params["snr_threshold"])

    def write_metrics(p):
        with open(p, "w") as f:
            json.dump({**metrics, "_cache_info": metrics_info}, f, indent=1)
    _write(metrics_path, write_metrics)

    # also cached when top has 'error' (no finite-SNR cells): that is a property
    # of the data, not a loading failure
    def write_traces(p):
        with open(p, "wb") as f:
            np.savez(f, _cache_info=json.dumps(traces_info), **top)
    _write(traces_path, write_traces)

    return metrics, top, "compute"

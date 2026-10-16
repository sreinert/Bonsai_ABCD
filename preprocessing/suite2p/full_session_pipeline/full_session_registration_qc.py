"""Small helpers shared by registration and the final QC report."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def array_from(*mappings: dict[str, Any], key: str) -> np.ndarray:
    for mapping in mappings:
        value = mapping.get(key)
        if value is not None:
            return np.asarray(value)
    return np.asarray([])


def finite(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values, dtype=float).ravel()
    return flattened[np.isfinite(flattened)]


def load_ops(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ops = np.load(path, allow_pickle=True).item()
    nested = ops.get("reg_outputs")
    return ops, nested if isinstance(nested, dict) else ops


def normalized_sharpness(image: np.ndarray) -> float:
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not np.isfinite(image).any():
        return math.nan
    low, high = np.nanpercentile(image, [1, 99])
    if high <= low:
        return math.nan
    normalized = np.clip((image - low) / (high - low), 0, 1)
    gradient_y, gradient_x = np.gradient(normalized)
    return float(np.nanmean(gradient_x * gradient_x + gradient_y * gradient_y))


def compute_metrics(ops: dict[str, Any], outputs: dict[str, Any]) -> dict[str, float]:
    yoff = finite(array_from(outputs, ops, key="yoff"))
    xoff = finite(array_from(outputs, ops, key="xoff"))
    corr = finite(array_from(outputs, ops, key="corrXY"))
    bad = array_from(outputs, ops, key="badframes").astype(bool).ravel()
    mean_image = array_from(outputs, ops, key="meanImg")
    nshift = min(yoff.size, xoff.size)
    displacement = np.hypot(yoff[:nshift], xoff[:nshift]) if nshift else np.asarray([])
    jitter = (
        np.hypot(np.diff(yoff[:nshift]), np.diff(xoff[:nshift]))
        if nshift > 1
        else np.asarray([])
    )
    settings = ops.get("settings")
    registration = (
        settings["registration"]
        if isinstance(settings, dict) and isinstance(settings.get("registration"), dict)
        else ops
    )
    ly = int(ops.get("Ly", mean_image.shape[0] if mean_image.ndim == 2 else 0))
    lx = int(ops.get("Lx", mean_image.shape[1] if mean_image.ndim == 2 else 0))
    search_limit = float(registration.get("maxregshift", 0.1)) * min(ly, lx)
    boundary_fraction = (
        float(
            np.mean(
                np.maximum(np.abs(yoff[:nshift]), np.abs(xoff[:nshift]))
                >= 0.95 * search_limit
            )
        )
        if nshift and search_limit > 0
        else math.nan
    )
    yoff1 = finite(array_from(outputs, ops, key="yoff1"))
    xoff1 = finite(array_from(outputs, ops, key="xoff1"))
    nonrigid_size = min(yoff1.size, xoff1.size)
    reg_dx = np.asarray(array_from(outputs, ops, key="regDX"), dtype=float)
    residual_values = finite(reg_dx[:, 3] if reg_dx.ndim == 2 and reg_dx.shape[1] >= 4 else reg_dx)
    return {
        "n_frames": float(max(yoff.size, corr.size, bad.size)),
        "corr_median": float(np.median(corr)) if corr.size else math.nan,
        "corr_p05": float(np.percentile(corr, 5)) if corr.size else math.nan,
        "sharpness": normalized_sharpness(mean_image),
        "bad_frame_fraction": float(np.mean(bad)) if bad.size else math.nan,
        "shift_p95": float(np.percentile(displacement, 95)) if displacement.size else math.nan,
        "shift_jitter_median": float(np.median(jitter)) if jitter.size else math.nan,
        "boundary_hit_fraction": boundary_fraction,
        "nonrigid_shift_p95": (
            float(np.percentile(np.hypot(yoff1[:nonrigid_size], xoff1[:nonrigid_size]), 95))
            if nonrigid_size
            else math.nan
        ),
        "reg_pc_residual_mean": (
            float(np.mean(residual_values)) if residual_values.size else math.nan
        ),
    }


def show_image(axis: plt.Axes, image: np.ndarray, title: str) -> None:
    image = np.asarray(image)
    if image.ndim != 2 or not image.size:
        axis.text(0.5, 0.5, "not available", ha="center", va="center")
    else:
        low, high = np.nanpercentile(image, [1, 99.5])
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
    axis.set_title(title)
    axis.set_axis_off()


def find_registered_binary(
    ops_path: Path, ops: dict[str, Any], channel: int
) -> Path | None:
    db = ops.get("db") if isinstance(ops.get("db"), dict) else {}
    key = "reg_file" if channel == 1 else "reg_file_chan2"
    candidates = [
        *[Path(str(mapping[key])) for mapping in (ops, db) if mapping.get(key)],
        ops_path.parent / ("data.bin" if channel == 1 else "data_chan2.bin"),
    ]
    return next((path for path in candidates if path.is_file()), None)


def make_registered_montage(
    ops_path: Path,
    ops: dict[str, Any],
    outputs: dict[str, Any],
    channel: int = 1,
) -> Path | None:
    if channel not in (1, 2):
        raise ValueError(f"channel must be 1 or 2, got {channel}")
    output_name = "registered_frames.png" if channel == 1 else "registered_frames_chan2.png"
    output_path = ops_path.parents[2] / output_name
    if output_path.is_file():
        return output_path
    binary = find_registered_binary(ops_path, ops, channel)
    mean_key = "meanImg" if channel == 1 else "meanImg_chan2"
    mean_image = array_from(outputs, ops, key=mean_key)
    if mean_image.ndim != 2 or binary is None:
        return None
    ly, lx = mean_image.shape
    bytes_per_frame = np.dtype(np.int16).itemsize * ly * lx
    nframes = binary.stat().st_size // bytes_per_frame
    if nframes <= 0:
        return None
    frames = np.memmap(binary, dtype=np.int16, mode="r", shape=(nframes, ly, lx))
    indices = np.unique(np.linspace(0, nframes - 1, min(12, nframes), dtype=int))
    low, high = np.percentile(mean_image, [1, 99.5])
    figure, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    for axis in axes.ravel():
        axis.set_axis_off()
    for axis, frame_index in zip(axes.ravel(), indices):
        axis.imshow(frames[frame_index], cmap="gray", vmin=low, vmax=high)
        axis.set_title(f"frame {frame_index}")
    figure.suptitle(f"Channel {channel}: registered frames sampled across the stack")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    del frames
    return output_path

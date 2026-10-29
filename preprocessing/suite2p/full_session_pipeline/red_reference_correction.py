"""Two-channel fluorescence calculations with explicit safety diagnostics.

The functions in this module deliberately do not alter CellTV's historical G/R
functions. They provide separately testable implementations of the same
baseline-subtracted fluorescence calculation used by ``nwb_barcoding``, the
CellTV ``get_dff_GR`` calculation, and the earlier opt-in robust
reference-channel regression.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    maximum_filter1d,
    median_filter,
    minimum_filter1d,
)
from scipy.signal import medfilt


@dataclass(frozen=True)
class ReferenceCorrectionResult:
    """Per-ROI red-reference correction outputs and diagnostics."""

    corrected: np.ndarray
    fitted_reference: np.ndarray
    smoothed_red: np.ndarray
    beta: np.ndarray
    raw_standardized_coupling: np.ndarray
    applied_standardized_coupling: np.ndarray
    green_center: np.ndarray
    red_center: np.ndarray
    green_scale: np.ndarray
    red_scale: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    status: np.ndarray
    smoothing_kernel_frames: int
    tuning_constant: float
    max_iterations: int
    tolerance: float
    scale_epsilon: float


@dataclass(frozen=True)
class SafeRatioResult:
    """CellTV baseline-subtracted G/R plus denominator-floor diagnostics."""

    delta_ratio: np.ndarray
    ratio: np.ndarray
    smoothed_red: np.ndarray
    denominator_floor: np.ndarray
    floor_fraction: np.ndarray
    typical_red: np.ndarray
    status: np.ndarray
    smoothing_kernel_frames: int
    floor_percentile: float
    absolute_floor: float


def _validate_trace_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Fluorescence arrays must be two-dimensional (ROI, frame)")
    if left.shape != right.shape:
        raise ValueError(
            f"Fluorescence and neuropil arrays must have equal shapes; got "
            f"{left.shape} and {right.shape}"
        )
    if left.shape[1] < 2:
        raise ValueError("Fluorescence arrays must contain at least two frames")
    return left, right


def _neuropil_coefficient(ops: dict[str, Any], override: float | None) -> float:
    if override is not None:
        coefficient = float(override)
    elif "neucoeff" in ops:
        coefficient = float(ops["neucoeff"])
    else:
        extraction = ops.get("extraction", {})
        if not isinstance(extraction, dict) or "neuropil_coefficient" not in extraction:
            raise KeyError(
                "ops has no 'neucoeff' or extraction['neuropil_coefficient']; "
                "cannot reproduce Suite2p/CellTV neuropil correction"
            )
        coefficient = float(extraction["neuropil_coefficient"])
    if not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError("Neuropil coefficient must be finite and non-negative")
    return coefficient


def _baseline_parameters(ops: dict[str, Any]) -> tuple[str, float, float, float, float]:
    nested = ops.get("dcnv_preprocess", {})
    if not isinstance(nested, dict):
        nested = {}

    def required(name: str) -> Any:
        if name in ops and ops[name] is not None:
            return ops[name]
        if name in nested and nested[name] is not None:
            return nested[name]
        raise KeyError(f"Suite2p ops has no preprocessing setting {name!r}")

    return (
        str(required("baseline")),
        float(required("win_baseline")),
        float(required("sig_baseline")),
        float(required("fs")),
        float(required("prctile_baseline")),
    )


def _suite2p_preprocess(
    fluorescence: np.ndarray,
    baseline: str,
    win_baseline: float,
    sig_baseline: float,
    fs: float,
    prctile_baseline: float,
) -> np.ndarray:
    try:
        from suite2p.extraction import dcnv
    except ModuleNotFoundError:
        # Keep report generation usable in the lightweight QC environment.  This
        # is Suite2p's baseline algorithm; production Suite2p environments take
        # the branch above and call the library implementation directly.
        window = max(1, int(win_baseline * fs))
        if baseline == "maximin":
            low = gaussian_filter(fluorescence, sigma=(0.0, sig_baseline))
            low = minimum_filter1d(low, size=window, axis=1)
            low = maximum_filter1d(low, size=window, axis=1)
        elif baseline == "constant":
            smoothed = gaussian_filter(fluorescence, sigma=(0.0, sig_baseline))
            low = np.broadcast_to(
                np.min(smoothed, axis=1, keepdims=True), fluorescence.shape
            )
        elif baseline == "constant_prctile":
            low = np.broadcast_to(
                np.percentile(
                    fluorescence, prctile_baseline, axis=1, keepdims=True
                ),
                fluorescence.shape,
            )
        elif baseline in {"none", "None", ""}:
            low = np.zeros_like(fluorescence)
        else:
            raise ValueError(f"Unsupported Suite2p baseline method: {baseline!r}")
        return fluorescence - low
    arguments = (
        fluorescence,
        baseline,
        win_baseline,
        sig_baseline,
        fs,
        prctile_baseline,
    )
    if "device" in inspect.signature(dcnv.preprocess).parameters:
        import torch

        return dcnv.preprocess(*arguments, device=torch.device("cpu"))
    return dcnv.preprocess(*arguments)


def compute_channel_delta(
    fluorescence: np.ndarray,
    neuropil: np.ndarray,
    ops: dict[str, Any],
    neuropil_coefficient: float | None = None,
) -> np.ndarray:
    """Return Suite2p/CellTV-style baseline-subtracted fluorescence.

    This intentionally does not divide by F0.  It matches the algorithm used in
    :mod:`nwb_barcoding.compute_dff`: neuropil correction followed by Suite2p's
    ``dcnv.preprocess``.
    """

    fluorescence, neuropil = _validate_trace_pair(fluorescence, neuropil)
    coefficient = _neuropil_coefficient(ops, neuropil_coefficient)
    baseline, window, sigma, fs, percentile = _baseline_parameters(ops)
    corrected = (
        np.asarray(fluorescence, dtype=np.float32)
        - coefficient * np.asarray(neuropil, dtype=np.float32)
    )
    delta = _suite2p_preprocess(
        corrected, baseline, window, sigma, fs, percentile
    )
    delta = np.asarray(delta)
    if delta.shape != corrected.shape:
        raise RuntimeError(
            f"Suite2p preprocessing changed trace shape from {corrected.shape} to {delta.shape}"
        )
    return delta.astype(np.float32, copy=False)


def compute_celltv_ratio_delta(
    green_fluorescence: np.ndarray,
    green_neuropil: np.ndarray,
    red_fluorescence: np.ndarray,
    red_neuropil: np.ndarray,
    ops: dict[str, Any],
    neuropil_coefficient: float | None = 0.7,
    *,
    smoothing_seconds: float = 0.25,
    floor_percentile: float = 5.0,
    absolute_floor: float = 1e-3,
) -> SafeRatioResult:
    """Reproduce CellTV ``get_dff_GR`` without interpolation or ROI rejection.

    CellTV neuropil-corrects both channels, median-filters red with
    :func:`scipy.signal.medfilt`, floors red at the larger of its per-ROI fifth
    percentile and ``1e-3``, divides green by that floored denominator, and
    applies Suite2p baseline preprocessing to G/R. The default neuropil
    coefficient is CellTV's hard-coded ``0.7``.
    """

    green_fluorescence, green_neuropil = _validate_trace_pair(
        green_fluorescence, green_neuropil
    )
    red_fluorescence, red_neuropil = _validate_trace_pair(
        red_fluorescence, red_neuropil
    )
    if green_fluorescence.shape != red_fluorescence.shape:
        raise ValueError(
            "Green and red fluorescence arrays must have equal shapes; got "
            f"{green_fluorescence.shape} and {red_fluorescence.shape}"
        )
    fs = float(_baseline_parameters(ops)[3])
    if not np.isfinite(smoothing_seconds) or smoothing_seconds < 0:
        raise ValueError("Smoothing duration must be finite and non-negative")
    if not 0 <= floor_percentile <= 100:
        raise ValueError("Denominator floor percentile must be in [0, 100]")
    if not np.isfinite(absolute_floor) or absolute_floor <= 0:
        raise ValueError("Absolute denominator floor must be finite and positive")

    coefficient = _neuropil_coefficient(ops, neuropil_coefficient)
    green = np.asarray(green_fluorescence, dtype=np.float32) - coefficient * np.asarray(
        green_neuropil, dtype=np.float32
    )
    red = np.asarray(red_fluorescence, dtype=np.float32) - coefficient * np.asarray(
        red_neuropil, dtype=np.float32
    )
    kernel = max(1, int(fs * smoothing_seconds) | 1)
    smoothed_red = medfilt(red, kernel_size=(1, kernel))
    typical_red = np.median(np.abs(smoothed_red), axis=1)
    denominator_floor = np.maximum(
        np.percentile(smoothed_red, floor_percentile, axis=1),
        absolute_floor,
    )
    low_red = smoothed_red < denominator_floor[:, None]
    floor_fraction = np.mean(low_red, axis=1)
    guarded_red = np.maximum(smoothed_red, denominator_floor[:, None])
    ratio = green / guarded_red
    status = np.full(green.shape[0], "ok", dtype="U32")
    nonfinite_input = ~(
        np.all(np.isfinite(green), axis=1)
        & np.all(np.isfinite(smoothed_red), axis=1)
    )
    status[nonfinite_input] = "nonfinite_input"

    baseline_method, window, sigma, _, percentile = _baseline_parameters(ops)
    delta_ratio = np.asarray(
        _suite2p_preprocess(
            ratio, baseline_method, window, sigma, fs, percentile
        )
    ).astype(np.float32, copy=False)
    nonfinite_output = ~np.all(np.isfinite(delta_ratio), axis=1)
    status[nonfinite_output] = "nonfinite_output"

    return SafeRatioResult(
        delta_ratio=delta_ratio,
        ratio=ratio,
        smoothed_red=smoothed_red.astype(np.float32, copy=False),
        denominator_floor=denominator_floor,
        floor_fraction=floor_fraction,
        typical_red=typical_red,
        status=status,
        smoothing_kernel_frames=kernel,
        floor_percentile=float(floor_percentile),
        absolute_floor=float(absolute_floor),
    )


# Backward-compatible import for code written while this calculation included
# additional interpolation/rejection guards. It now returns exact CellTV G/R.
compute_safe_ratio_delta = compute_celltv_ratio_delta


def _robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=1)
    q16, q84 = np.percentile(values, (16, 84), axis=1)
    return center, (q84 - q16) / 2.0


def _tukey_irls(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tuning_constant: float,
    max_iterations: int,
    tolerance: float,
    scale_epsilon: float,
) -> tuple[np.ndarray, bool, int]:
    design = np.column_stack((np.ones_like(x), x))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    if max_iterations == 0:
        return coefficients, False, 0

    for iteration in range(1, max_iterations + 1):
        residual = y - design @ coefficients
        residual_center = np.median(residual)
        residual_scale = 1.4826 * np.median(np.abs(residual - residual_center))
        if not np.isfinite(residual_scale):
            return coefficients, False, iteration
        if residual_scale <= scale_epsilon:
            return coefficients, True, iteration
        u = (residual - residual_center) / (tuning_constant * residual_scale)
        weights = np.zeros_like(u)
        inside = np.abs(u) < 1.0
        weights[inside] = (1.0 - u[inside] ** 2) ** 2
        if np.count_nonzero(weights > 0) < 2:
            return coefficients, False, iteration
        root_weight = np.sqrt(weights)
        updated = np.linalg.lstsq(
            design * root_weight[:, None], y * root_weight, rcond=None
        )[0]
        if not np.all(np.isfinite(updated)):
            return coefficients, False, iteration
        change = np.linalg.norm(updated - coefficients)
        threshold = tolerance * (1.0 + np.linalg.norm(coefficients))
        coefficients = updated
        if change <= threshold:
            return coefficients, True, iteration
    return coefficients, False, max_iterations


def fit_reference_correction(
    green_delta: np.ndarray,
    red_delta: np.ndarray,
    fs: float,
    *,
    smoothing_seconds: float = 0.25,
    tuning_constant: float = 3.0,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    scale_epsilon: float = 1e-6,
) -> ReferenceCorrectionResult:
    """Regress a stable red reference from green using Tukey-bisquare IRLS.

    The unconstrained standardized slope is retained for diagnostics.  The
    applied slope is clipped to ``[0, 1]``; non-positive, invalid, flat, or
    non-convergent fits apply no correction.
    """

    green_delta, red_delta = _validate_trace_pair(green_delta, red_delta)
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be finite and positive")
    if not np.isfinite(smoothing_seconds) or smoothing_seconds < 0:
        raise ValueError("Smoothing duration must be finite and non-negative")
    if not np.isfinite(tuning_constant) or tuning_constant <= 0:
        raise ValueError("Tukey tuning constant must be finite and positive")
    if max_iterations < 0:
        raise ValueError("Maximum iterations cannot be negative")
    if tolerance <= 0 or scale_epsilon <= 0:
        raise ValueError("Tolerance and scale epsilon must be positive")

    green = np.asarray(green_delta, dtype=np.float64)
    red = np.asarray(red_delta, dtype=np.float64)
    kernel = max(1, int(fs * smoothing_seconds) | 1)
    smoothed_red = median_filter(red, size=(1, kernel), mode="nearest")
    green_center, green_scale = _robust_center_scale(green)
    red_center, red_scale = _robust_center_scale(smoothed_red)

    n_rois = green.shape[0]
    corrected = green.copy()
    fitted_reference = np.zeros_like(green)
    beta = np.zeros(n_rois, dtype=float)
    raw_coupling = np.full(n_rois, np.nan, dtype=float)
    applied_coupling = np.zeros(n_rois, dtype=float)
    converged = np.zeros(n_rois, dtype=bool)
    iterations = np.zeros(n_rois, dtype=int)
    status = np.full(n_rois, "not_fitted", dtype="U32")

    for roi in range(n_rois):
        if not np.all(np.isfinite(red[roi])):
            status[roi] = "nonfinite_red"
            continue
        if not np.all(np.isfinite(green[roi])):
            status[roi] = "nonfinite_green"
            continue
        if red_scale[roi] <= scale_epsilon:
            status[roi] = "flat_red"
            continue
        if green_scale[roi] <= scale_epsilon:
            status[roi] = "flat_green"
            continue

        x = (smoothed_red[roi] - red_center[roi]) / red_scale[roi]
        y = (green[roi] - green_center[roi]) / green_scale[roi]
        coefficients, did_converge, count = _tukey_irls(
            x,
            y,
            tuning_constant=tuning_constant,
            max_iterations=max_iterations,
            tolerance=tolerance,
            scale_epsilon=scale_epsilon,
        )
        iterations[roi] = count
        converged[roi] = did_converge
        raw_coupling[roi] = float(coefficients[1])
        if not did_converge:
            status[roi] = "nonconvergent"
            continue
        if raw_coupling[roi] <= 0:
            status[roi] = "nonpositive_coupling"
            continue

        applied_coupling[roi] = min(raw_coupling[roi], 1.0)
        beta[roi] = applied_coupling[roi] * green_scale[roi] / red_scale[roi]
        fitted_reference[roi] = beta[roi] * (smoothed_red[roi] - red_center[roi])
        corrected[roi] = green[roi] - fitted_reference[roi]
        status[roi] = "clipped_high" if raw_coupling[roi] > 1.0 else "applied"

    return ReferenceCorrectionResult(
        corrected=corrected.astype(np.float32),
        fitted_reference=fitted_reference.astype(np.float32),
        smoothed_red=smoothed_red.astype(np.float32),
        beta=beta,
        raw_standardized_coupling=raw_coupling,
        applied_standardized_coupling=applied_coupling,
        green_center=green_center,
        red_center=red_center,
        green_scale=green_scale,
        red_scale=red_scale,
        converged=converged,
        iterations=iterations,
        status=status,
        smoothing_kernel_frames=kernel,
        tuning_constant=float(tuning_constant),
        max_iterations=int(max_iterations),
        tolerance=float(tolerance),
        scale_epsilon=float(scale_epsilon),
    )

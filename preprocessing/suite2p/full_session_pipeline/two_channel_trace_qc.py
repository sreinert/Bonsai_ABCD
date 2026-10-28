#!/usr/bin/env python3
"""Generate the supplementary two-channel/CellTV Suite2p trace QC report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d
from scipy.signal import medfilt

try:
    from .red_reference_correction import (
        SafeRatioResult,
        compute_channel_delta,
        compute_safe_ratio_delta,
    )
except ImportError:  # Direct script execution on the HPC.
    from red_reference_correction import (
        SafeRatioResult,
        compute_channel_delta,
        compute_safe_ratio_delta,
    )


REQUIRED_FILES = ("F.npy", "Fneu.npy", "iscell.npy", "ops.npy", "stat.npy")
CHANNEL2_FILES = ("F_chan2.npy", "Fneu_chan2.npy")
COLORS = {"green": "#16853b", "red": "#d43f3a", "ratio": "#272727"}


def neuropil_coefficient(ops: dict[str, Any], override: float | None = None) -> float:
    """Return Suite2p's stored neuropil coefficient unless explicitly overridden."""
    if override is not None:
        return float(override)
    if "neucoeff" in ops:
        return float(ops["neucoeff"])
    extraction = ops.get("extraction", {})
    if isinstance(extraction, dict) and "neuropil_coefficient" in extraction:
        return float(extraction["neuropil_coefficient"])
    raise KeyError(
        "ops.npy has no 'neucoeff' or extraction['neuropil_coefficient']; "
        "cannot reproduce CellTV neuropil correction"
    )


def baseline(signal: np.ndarray, ops: dict[str, Any]) -> np.ndarray:
    """Return Suite2p's baseline without subtracting it from the signal."""
    dcnv = ops.get("dcnv_preprocess", {})
    if not isinstance(dcnv, dict):
        dcnv = {}
    method = str(ops.get("baseline", dcnv.get("baseline", "maximin")))
    sigma = float(ops.get("sig_baseline", dcnv.get("sig_baseline", 10.0)))
    fs = float(ops["fs"])
    window = max(
        1, int(float(ops.get("win_baseline", dcnv.get("win_baseline", 60.0))) * fs)
    )
    percentile = float(
        ops.get("prctile_baseline", dcnv.get("prctile_baseline", 8.0))
    )

    if method == "maximin":
        result = gaussian_filter(signal, sigma=(0.0, sigma))
        result = minimum_filter1d(result, size=window, axis=1)
        return maximum_filter1d(result, size=window, axis=1)
    if method == "constant":
        smoothed = gaussian_filter(signal, sigma=(0.0, sigma))
        return np.broadcast_to(
            np.min(smoothed, axis=1, keepdims=True), signal.shape
        )
    if method == "constant_prctile":
        return np.broadcast_to(
            np.percentile(signal, percentile, axis=1, keepdims=True), signal.shape
        )
    if method in {"none", "None", ""}:
        return np.zeros_like(signal)
    raise ValueError(f"Unsupported Suite2p baseline method: {method!r}")


def normalized_dff(
    signal: np.ndarray, ops: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = baseline(signal, ops)
    # A relative epsilon prevents division by exact zero without masking a genuinely
    # weak/unstable baseline. The latter remains visible in metrics and plot limits.
    scale = np.maximum(np.median(np.abs(base), axis=1, keepdims=True), 1.0)
    denominator = np.maximum(np.abs(base), scale * 1e-6)
    dff = (signal - base) / denominator
    clipped = np.abs(base) < scale * 1e-6
    return dff.astype(np.float32), base, clipped


def channel_signals(
    F: np.ndarray,
    Fneu: np.ndarray,
    F2: np.ndarray,
    Fneu2: np.ndarray,
    ops: dict[str, Any],
    neucoeff: float,
) -> dict[str, np.ndarray]:
    green = np.asarray(F, dtype=np.float32) - neucoeff * np.asarray(Fneu, dtype=np.float32)
    red = np.asarray(F2, dtype=np.float32) - neucoeff * np.asarray(Fneu2, dtype=np.float32)

    # Match get_dff_GR in cellTV_functions_cohort3.py: smooth only red,
    # then floor each ROI's denominator at its fifth percentile (and 1e-3).
    kernel = max(1, int(float(ops["fs"]) * 0.25) | 1)
    red_smoothed = medfilt(red, kernel_size=(1, kernel)).astype(np.float32)
    red_floor = np.maximum(
        np.percentile(red_smoothed, 5, axis=1, keepdims=True), 1e-3
    ).astype(np.float32)
    ratio_floor_mask = red_smoothed < red_floor
    ratio = green / np.maximum(red_smoothed, red_floor)
    return {
        "green": green,
        "red": red,
        "red_smoothed": red_smoothed,
        "red_floor": red_floor,
        "ratio_floor_mask": ratio_floor_mask,
        "ratio": ratio.astype(np.float32),
        "red_median_kernel_frames": np.asarray(kernel),
    }


def robust_sigma(values: np.ndarray, axis: int = 1) -> np.ndarray:
    q16, q84 = np.percentile(values, (16, 84), axis=axis)
    return (q84 - q16) / 2.0


def row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - np.mean(left, axis=1, keepdims=True)
    right_centered = right - np.mean(right, axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(left.shape[0], np.nan, dtype=float),
        where=denominator > 0,
    )


def select_rois(iscell: np.ndarray, green_dff: np.ndarray, count: int) -> np.ndarray:
    candidates = np.flatnonzero(iscell[:, 0] > 0.5)
    if candidates.size == 0:
        candidates = np.arange(len(iscell))
    activity = robust_sigma(green_dff[candidates])
    probability = iscell[candidates, 1]
    # Classifier probability is primary; activity resolves ties. ROI index is the
    # final stable tiebreaker. This avoids hand-picked example cells.
    order = np.lexsort((candidates, -activity, -probability))
    return candidates[order[: min(count, candidates.size)]]


def decimated(values: np.ndarray, max_points: int = 1500) -> tuple[np.ndarray, int]:
    step = max(1, int(np.ceil(values.shape[1] / max_points)))
    if step == 1:
        return values, step
    usable = values.shape[1] - values.shape[1] % step
    return values[:, :usable].reshape(values.shape[0], -1, step).mean(axis=2), step


def display_limits(trace: np.ndarray) -> tuple[float, float]:
    finite = trace[np.isfinite(trace)]
    if finite.size == 0:
        return -1.0, 1.0
    low, high = np.percentile(finite, (0.5, 99.5))
    if high <= low:
        padding = max(abs(float(low)) * 0.05, 1e-3)
        return float(low - padding), float(high + padding)
    padding = 0.06 * (high - low)
    return float(low - padding), float(high + padding)


def roi_centroid(roi: dict[str, Any]) -> tuple[float, float]:
    return float(np.mean(roi["xpix"])), float(np.mean(roi["ypix"]))


def plot_selected(
    path: Path,
    selected: np.ndarray,
    traces: dict[str, np.ndarray],
    iscell: np.ndarray,
    fs: float,
    trace_kind: str,
    frame_slice: slice = slice(None),
    title: str = "Deterministically selected classified cells; y-axis robustly scaled per panel",
) -> None:
    if trace_kind == "celltv":
        column_titles = {
            "green": "Green ΔF (channel 1; CellTV)",
            "red": "Red ΔF (channel 2; CellTV)",
            "ratio": "Δ(G/R) (CellTV)",
        }
    elif trace_kind == "safe_ratio":
        column_titles = {
            "green": "Green ΔF (channel 1)",
            "red": "Red ΔF (channel 2)",
            "ratio": "Safe Δ(G/R)",
        }
    else:
        raise ValueError(f"Unknown trace kind: {trace_kind!r}")
    figure, axes = plt.subplots(
        len(selected), 3, figsize=(16, max(3.0, 2.35 * len(selected))), sharex=True
    )
    axes = np.asarray(axes).reshape(len(selected), 3)
    start = int(frame_slice.start or 0)
    for column, name in enumerate(("green", "red", "ratio")):
        sampled, step = decimated(traces[name][selected, frame_slice])
        time_minutes = (
            start + np.arange(sampled.shape[1]) * step + (step - 1) / 2
        ) / fs / 60.0
        for row, roi in enumerate(selected):
            axes[row, column].plot(
                time_minutes, sampled[row], color=COLORS[name], linewidth=0.55
            )
            axes[row, column].set_ylim(display_limits(sampled[row]))
            axes[row, column].axhline(0, color="#999999", linewidth=0.4)
            if row == 0:
                axes[row, column].set_title(column_titles[name])
            if column == 0:
                axes[row, column].set_ylabel(
                    f"ROI {roi}\np(cell)={iscell[roi, 1]:.2f}", fontsize=8
                )
            if row == len(selected) - 1:
                axes[row, column].set_xlabel("Time (minutes)")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_summary(
    path: Path,
    ops: dict[str, Any],
    stat: np.ndarray,
    selected: np.ndarray,
    iscell: np.ndarray,
    metrics: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    image = np.asarray(ops.get("meanImg", np.zeros((int(ops["Ly"]), int(ops["Lx"])))))
    low, high = np.percentile(image[np.isfinite(image)], (1, 99))
    axes[0, 0].imshow(image, cmap="gray", vmin=low, vmax=high)
    for roi in selected:
        x, y = roi_centroid(stat[roi])
        axes[0, 0].plot(x, y, "o", ms=6, mfc="none", mec="#ffd54f", mew=1.2)
        axes[0, 0].text(x + 3, y, str(roi), color="#ffd54f", fontsize=7)
    axes[0, 0].set_title("Selected ROIs on channel-1 mean image")
    axes[0, 0].axis("off")

    axes[0, 1].hist(iscell[:, 1], bins=np.linspace(0, 1, 21), color="#5f6f7a")
    axes[0, 1].axvline(0.5, color="#d43f3a", linestyle="--", linewidth=1)
    axes[0, 1].set(xlabel="Suite2p cell probability", ylabel="ROIs", title="Classification")

    correlations = np.asarray([row["green_red_correlation"] for row in metrics])
    classified = iscell[:, 0] > 0.5
    axes[1, 0].hist(correlations[np.isfinite(correlations)], bins=20, color="#7e57c2")
    axes[1, 0].set(
        xlabel="Correlation of neuropil-corrected G and R",
        ylabel="ROIs",
        title=f"Channel coupling ({classified.sum()} classified cells)",
    )

    floor_fraction = np.asarray(
        [row["safe_ratio_red_floor_fraction"] for row in metrics]
    )
    axes[1, 1].hist(floor_fraction, bins=np.linspace(0, 1, 21), color="#e07a2f")
    axes[1, 1].set(
        xlabel="Fraction of frames using red denominator floor",
        ylabel="ROIs",
        title="Guarded G/R denominator stability",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def metric_rows(
    signals: dict[str, np.ndarray],
    celltv_delta: dict[str, np.ndarray],
    safe_ratio: SafeRatioResult,
    iscell: np.ndarray,
) -> list[dict[str, Any]]:
    correlation_before = row_correlation(celltv_delta["green"], celltv_delta["red"])
    ratio_red_correlation = row_correlation(
        safe_ratio.delta_ratio, celltv_delta["red"]
    )
    rows: list[dict[str, Any]] = []
    for roi in range(len(iscell)):
        row: dict[str, Any] = {
            "roi": roi,
            "iscell": int(iscell[roi, 0] > 0.5),
            "cell_probability": float(iscell[roi, 1]),
            "green_red_correlation": float(correlation_before[roi]),
            "safe_ratio_red_correlation": float(ratio_red_correlation[roi]),
            "ratio_red_floor": float(signals["red_floor"][roi, 0]),
            "ratio_red_floor_fraction": float(np.mean(signals["ratio_floor_mask"][roi])),
            "safe_ratio_red_floor": float(safe_ratio.denominator_floor[roi]),
            "safe_ratio_red_floor_fraction": float(safe_ratio.floor_fraction[roi]),
            "safe_ratio_typical_red": float(safe_ratio.typical_red[roi]),
            "safe_ratio_status": str(safe_ratio.status[roi]),
            "safe_ratio_robust_sigma": float(
                robust_sigma(safe_ratio.delta_ratio[roi : roi + 1])[0]
            ),
            "safe_ratio_p99": float(np.percentile(safe_ratio.delta_ratio[roi], 99)),
        }
        for name in ("green", "red", "ratio"):
            row[f"{name}_corrected_median"] = float(np.median(signals[name][roi]))
            row[f"{name}_celltv_delta_robust_sigma"] = float(
                robust_sigma(celltv_delta[name][roi : roi + 1])[0]
            )
            row[f"{name}_celltv_delta_p99"] = float(
                np.percentile(celltv_delta[name][roi], 99)
            )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_html(
    trace_dir: Path,
    selected: np.ndarray,
    metrics: list[dict[str, Any]],
    settings: dict[str, Any],
) -> str:
    selected_rows = {int(row["roi"]): row for row in metrics}
    table_rows = "\n".join(
        "<tr>"
        f"<td>{roi}</td>"
        f"<td>{selected_rows[int(roi)]['cell_probability']:.3f}</td>"
        f"<td>{selected_rows[int(roi)]['green_red_correlation']:.3f}</td>"
        f"<td>{selected_rows[int(roi)]['ratio_red_floor_fraction']:.3f}</td>"
        f"<td>{selected_rows[int(roi)]['green_celltv_delta_robust_sigma']:.3g}</td>"
        f"<td>{selected_rows[int(roi)]['red_celltv_delta_robust_sigma']:.3g}</td>"
        f"<td>{selected_rows[int(roi)]['ratio_celltv_delta_robust_sigma']:.3g}</td>"
        f"<td>{selected_rows[int(roi)]['safe_ratio_red_floor']:.3g}</td>"
        f"<td>{selected_rows[int(roi)]['safe_ratio_red_floor_fraction']:.3g}</td>"
        f"<td>{selected_rows[int(roi)]['safe_ratio_robust_sigma']:.3g}</td>"
        f"<td>{html.escape(str(selected_rows[int(roi)]['safe_ratio_status']))}</td>"
        "</tr>"
        for roi in selected
    )
    n_cells = sum(row["iscell"] for row in metrics)
    classified_rows = [row for row in metrics if row["iscell"]]
    summary_rows = classified_rows or metrics
    median_correlation = float(np.median([row["green_red_correlation"] for row in summary_rows]))
    safe_count = sum(
        row["safe_ratio_status"] in {"ok", "frequent_flooring"}
        for row in summary_rows
    )
    frequent_floor_count = sum(
        row["safe_ratio_status"] == "frequent_flooring" for row in summary_rows
    )
    unstable_ratio_count = sum(row["ratio_red_floor_fraction"] > 0.05 for row in summary_rows)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Trace extraction QC</title>
<style>body{{font:15px system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#222}}
img{{width:100%;height:auto;border:1px solid #ddd}} table{{border-collapse:collapse}}
th,td{{padding:.4rem .65rem;border:1px solid #ccc;text-align:right}} th:first-child,td:first-child{{text-align:left}}
code{{background:#f2f2f2;padding:.1rem .3rem}} .note{{background:#eef6ff;padding:.8rem 1rem;border-left:4px solid #3974a8}}
.warning{{background:#fff3e8;padding:.8rem 1rem;border-left:4px solid #d76a00;margin-top:1rem}}</style></head>
<body><h1>Trace extraction QC</h1>
<p><b>Input:</b> <code>{html.escape(str(trace_dir))}</code><br>
<b>ROIs:</b> {len(metrics)}; <b>classified cells:</b> {n_cells};
<b>frames:</b> {settings['n_frames']}; <b>sampling:</b> {settings['fs']:.3g} Hz</p>
<div class="note"><b>Calculation.</b> Green and red are corrected as <code>F − {settings['neucoeff']:.3g} × Fneu</code>.
The red trace alone is median-filtered over {settings['red_kernel_frames']} frames before G/R, matching
<code>get_dff_GR</code>. Its denominator is floored at each ROI's fifth percentile or 1e−3, whichever is larger.
The <b>CellTV-style</b> plots show <code>signal − F0</code>, exactly matching the baseline-subtracted output
of Suite2p <code>preprocess</code> used by <code>get_dff</code> and <code>get_dff_GR</code>; despite historical
naming, these are ΔF and are not divided by F0. The safe ratio follows the original CellTV order:
neuropil-correct G and R, smooth R, divide G by a guarded R denominator, then baseline-subtract G/R.
The denominator floor is the largest of the fifth percentile, 5% of typical absolute red fluorescence,
and 1e−3. Weak or non-finite red references produce a zero ratio with an explicit status.</div>
<h2>Overview</h2><img src="summary.png" alt="QC summary">
<h2>Example classified cells</h2><p>Selected reproducibly by descending Suite2p cell probability,
then green-trace robust activity—not by manual choice. Plot limits use each panel's 0.5–99.5th percentiles;
traces are non-overlapping mean-binned to at most 1,500 points for display, and the CSV metrics use the
unbinned traces. Classified-cell median G/R channel correlation is
{median_correlation:.3f}. A finite guarded ratio was produced for {safe_count}/{len(summary_rows)}
classified cells; {frequent_floor_count}/{len(summary_rows)} required frequent safe-denominator flooring.
The legacy ratio used its denominator floor in more than 5% of frames for
{unstable_ratio_count}/{len(summary_rows)} cells.</p>
<h3>Exact CellTV-style baseline-subtracted traces</h3>
<p>These reproduce what the current CellTV functions return, although those functions call the result dF/F.</p>
<img src="selected_cells_celltv_zoom.png" alt="CellTV-style selected cell trace zoom">
<details><summary>Show full CellTV-style traces</summary><img src="selected_cells_celltv.png" alt="Full CellTV-style selected cell traces"></details>
<h3>Guarded CellTV-style G/R</h3>
<p>Green and red use the Suite2p baseline-subtracted ΔF calculation for comparison. The third column is
the safely floored, baseline-subtracted G/R ratio.</p>
<img src="selected_cells_safe_ratio_zoom.png" alt="Safe G/R selected cell trace zoom">
<details><summary>Show full safe G/R traces</summary><img src="selected_cells_safe_ratio.png" alt="Full safe G/R selected cell traces"></details>
<h2>Selected-cell metrics</h2><table><thead><tr><th>ROI</th><th>p(cell)</th><th>corr(G,R)</th>
<th>red floor fraction</th><th>CellTV green σ</th><th>CellTV red σ</th><th>CellTV ratio σ</th>
<th>safe red floor</th><th>safe floor fraction</th><th>safe ratio σ</th><th>status</th></tr></thead><tbody>
{table_rows}</tbody></table>
<h2>Files</h2><ul><li><a href="trace_metrics.csv">trace_metrics.csv</a>: metrics for every ROI</li>
<li><a href="calculation.json">calculation.json</a>: exact calculation settings and selected ROI indices</li></ul>
</body></html>"""


def validate_inputs(trace_dir: Path) -> bool:
    for name in REQUIRED_FILES:
        if not (trace_dir / name).is_file():
            raise FileNotFoundError(f"Required trace output is missing: {trace_dir / name}")
    channel2 = [(trace_dir / name).is_file() for name in CHANNEL2_FILES]
    if any(channel2) and not all(channel2):
        raise FileNotFoundError("Channel 2 is incomplete: both F_chan2.npy and Fneu_chan2.npy are required")
    return all(channel2)


def generate_two_channel_qc(
    trace_dir: Path,
    output_dir: Path,
    *,
    n_cells: int = 6,
    neucoeff_override: float | None = None,
) -> dict[str, Any]:
    """Write the supplementary two-channel report and return its settings."""
    if n_cells < 1:
        raise ValueError("n_cells must be positive")
    trace_dir = trace_dir.resolve()
    if not validate_inputs(trace_dir):
        raise RuntimeError("This QC report currently requires a complete second channel")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing QC output: {output_dir}")

    F = np.load(trace_dir / "F.npy", mmap_mode="r")
    Fneu = np.load(trace_dir / "Fneu.npy", mmap_mode="r")
    F2 = np.load(trace_dir / "F_chan2.npy", mmap_mode="r")
    Fneu2 = np.load(trace_dir / "Fneu_chan2.npy", mmap_mode="r")
    iscell = np.load(trace_dir / "iscell.npy")
    stat = np.load(trace_dir / "stat.npy", allow_pickle=True)
    ops = np.load(trace_dir / "ops.npy", allow_pickle=True).item()
    shapes = {array.shape for array in (F, Fneu, F2, Fneu2)}
    if len(shapes) != 1 or F.ndim != 2 or F.shape[0] != len(iscell) or len(stat) != len(iscell):
        raise ValueError("Trace, classification, and ROI array shapes are inconsistent")
    neucoeff = neuropil_coefficient(ops, override=neucoeff_override)
    dcnv = ops.get("dcnv_preprocess", {})
    if not isinstance(dcnv, dict):
        dcnv = {}

    signals = channel_signals(F, Fneu, F2, Fneu2, ops, neucoeff)
    celltv_delta: dict[str, np.ndarray] = {
        "green": compute_channel_delta(F, Fneu, ops, neucoeff),
        "red": compute_channel_delta(F2, Fneu2, ops, neucoeff),
    }
    ratio_baseline = baseline(signals["ratio"], ops)
    celltv_delta["ratio"] = (signals["ratio"] - ratio_baseline).astype(np.float32)
    for name in ("green", "red", "ratio"):
        if not np.all(np.isfinite(celltv_delta[name])):
            raise RuntimeError(f"Non-finite values found in CellTV-style {name} ΔF")
    safe_ratio = compute_safe_ratio_delta(
        F, Fneu, F2, Fneu2, ops, neuropil_coefficient=neucoeff
    )
    if not np.all(np.isfinite(safe_ratio.delta_ratio)):
        raise RuntimeError("Non-finite values found in guarded G/R ΔF")
    safe_ratio_traces = {
        "green": celltv_delta["green"],
        "red": celltv_delta["red"],
        "ratio": safe_ratio.delta_ratio,
    }
    selected = select_rois(iscell, celltv_delta["green"], n_cells)
    metrics = metric_rows(signals, celltv_delta, safe_ratio, iscell)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        zoom_frames = min(F.shape[1], int(round(2.0 * 60.0 * float(ops["fs"]))))
        zoom_start = max(0, (F.shape[1] - zoom_frames) // 2)
        zoom_stop = zoom_start + zoom_frames
        for trace_kind, traces in (("celltv", celltv_delta), ("safe_ratio", safe_ratio_traces)):
            stem = "celltv" if trace_kind == "celltv" else "safe_ratio"
            label = (
                "CellTV-style ΔF"
                if trace_kind == "celltv"
                else "guarded CellTV-style G/R"
            )
            plot_selected(
                temporary / f"selected_cells_{stem}.png",
                selected,
                traces,
                iscell,
                float(ops["fs"]),
                trace_kind,
                title=f"Full session: {label}; y-axis robustly scaled per panel",
            )
            plot_selected(
                temporary / f"selected_cells_{stem}_zoom.png",
                selected,
                traces,
                iscell,
                float(ops["fs"]),
                trace_kind,
                frame_slice=slice(zoom_start, zoom_stop),
                title=f"Central two-minute window: {label}; y-axis robustly scaled per panel",
            )
        plot_summary(temporary / "summary.png", ops, stat, selected, iscell, metrics)
        write_csv(temporary / "trace_metrics.csv", metrics)
        settings = {
            "trace_dir": str(trace_dir),
            "n_rois": int(F.shape[0]),
            "n_frames": int(F.shape[1]),
            "fs": float(ops["fs"]),
            "neucoeff": neucoeff,
            "baseline": str(ops.get("baseline", dcnv.get("baseline", "maximin"))),
            "win_baseline": float(ops.get("win_baseline", dcnv.get("win_baseline", 60.0))),
            "sig_baseline": float(ops.get("sig_baseline", dcnv.get("sig_baseline", 10.0))),
            "prctile_baseline": float(ops.get("prctile_baseline", dcnv.get("prctile_baseline", 8.0))),
            "red_kernel_frames": int(signals["red_median_kernel_frames"]),
            "red_ratio_floor_percentile": 5.0,
            "red_ratio_absolute_floor": 1e-3,
            "selected_rois": selected.tolist(),
            "zoom_frame_range": [zoom_start, zoom_stop],
            "selection_rule": "iscell first; descending cell probability, green robust sigma, ROI index",
            "celltv_formula": "signal - Suite2p baseline (matches dcnv.preprocess output)",
            "safe_ratio": {
                "method": "CellTV-style G/R with adaptive denominator guard",
                "formula": "baseline_subtract((G - alpha*Gneu) / guarded_median_filter(R - alpha*Rneu))",
                "red_smoothing_seconds": 0.25,
                "red_smoothing_kernel_frames": safe_ratio.smoothing_kernel_frames,
                "red_smoothing_boundary": "nearest",
                "floor_percentile": safe_ratio.floor_percentile,
                "relative_floor": safe_ratio.relative_floor,
                "absolute_floor": safe_ratio.absolute_floor,
                "maximum_floor_fraction": safe_ratio.maximum_floor_fraction,
                "status_counts": {
                    str(name): int(np.sum(safe_ratio.status == name))
                    for name in np.unique(safe_ratio.status)
                },
            },
        }
        (temporary / "calculation.json").write_text(
            json.dumps(settings, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "index.html").write_text(
            report_html(trace_dir, selected, metrics, settings), encoding="utf-8"
        )
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Default: TRACE_DIR/qc")
    parser.add_argument("--n-cells", type=int, default=6)
    parser.add_argument("--neucoeff", type=float, help="Default: coefficient stored in ops.npy")
    args = parser.parse_args()
    output_dir = (args.output_dir or args.trace_dir / "qc").resolve()
    generate_two_channel_qc(
        args.trace_dir,
        output_dir,
        n_cells=args.n_cells,
        neucoeff_override=args.neucoeff,
    )
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()

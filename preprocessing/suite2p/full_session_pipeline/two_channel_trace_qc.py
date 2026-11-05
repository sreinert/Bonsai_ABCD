#!/usr/bin/env python3
"""Generate the supplementary independent-channel Suite2p trace QC report."""

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

try:
    from .red_reference_correction import compute_channel_delta
except ImportError:  # Direct script execution on the HPC.
    from red_reference_correction import compute_channel_delta


REQUIRED_FILES = ("F.npy", "Fneu.npy", "iscell.npy", "ops.npy", "stat.npy")
CHANNEL2_FILES = ("F_chan2.npy", "Fneu_chan2.npy")
COLORS = {"green": "#16853b", "red": "#d43f3a"}


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
        "cannot reproduce Suite2p neuropil correction"
    )


def channel_signals(
    F: np.ndarray,
    Fneu: np.ndarray,
    F2: np.ndarray,
    Fneu2: np.ndarray,
    neucoeff: float,
) -> dict[str, np.ndarray]:
    green = np.asarray(F, dtype=np.float32) - neucoeff * np.asarray(Fneu, dtype=np.float32)
    red = np.asarray(F2, dtype=np.float32) - neucoeff * np.asarray(Fneu2, dtype=np.float32)
    return {"green": green, "red": red}


def robust_sigma(values: np.ndarray, axis: int = 1) -> np.ndarray:
    q16, q84 = np.percentile(values, (16, 84), axis=axis)
    return (q84 - q16) / 2.0


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
    frame_slice: slice = slice(None),
    title: str = "Deterministically selected classified cells; y-axis robustly scaled per panel",
) -> None:
    column_titles = {
        "green": "Green dF (channel 1; Suite2p baseline)",
        "red": "Red dF (channel 2; Suite2p baseline)",
    }
    figure, axes = plt.subplots(
        len(selected), 2, figsize=(12, max(3.0, 2.35 * len(selected))), sharex=True
    )
    axes = np.asarray(axes).reshape(len(selected), 2)
    start = int(frame_slice.start or 0)
    for column, name in enumerate(("green", "red")):
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

    classified = iscell[:, 0] > 0.5
    for axis, name in zip(axes[1], ("green", "red")):
        activity = np.asarray(
            [row[f"{name}_dff_robust_sigma"] for row in metrics]
        )
        axis.hist(activity[np.isfinite(activity)], bins=20, color=COLORS[name])
        axis.set(
            xlabel=f"{name.title()} dF robust sigma",
            ylabel="ROIs",
            title=f"{name.title()} activity ({classified.sum()} classified cells)",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def metric_rows(
    signals: dict[str, np.ndarray],
    channel_dff: dict[str, np.ndarray],
    iscell: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for roi in range(len(iscell)):
        row: dict[str, Any] = {
            "roi": roi,
            "iscell": int(iscell[roi, 0] > 0.5),
            "cell_probability": float(iscell[roi, 1]),
        }
        for name in ("green", "red"):
            row[f"{name}_corrected_median"] = float(np.median(signals[name][roi]))
            row[f"{name}_dff_robust_sigma"] = float(
                robust_sigma(channel_dff[name][roi : roi + 1])[0]
            )
            row[f"{name}_dff_p99"] = float(
                np.percentile(channel_dff[name][roi], 99)
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
        f"<td>{selected_rows[int(roi)]['green_dff_robust_sigma']:.3g}</td>"
        f"<td>{selected_rows[int(roi)]['green_dff_p99']:.3g}</td>"
        f"<td>{selected_rows[int(roi)]['red_dff_robust_sigma']:.3g}</td>"
        f"<td>{selected_rows[int(roi)]['red_dff_p99']:.3g}</td>"
        "</tr>"
        for roi in selected
    )
    n_cells = sum(row["iscell"] for row in metrics)
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
Each channel is then passed independently through Suite2p <code>dcnv.preprocess</code> using
the baseline settings stored in <code>ops.npy</code>. The result is baseline-subtracted dF;
no green/red ratio or reference-channel subtraction is calculated.</div>
<h2>Overview</h2><img src="summary.png" alt="QC summary">
<h2>Example classified cells</h2><p>Selected reproducibly by descending Suite2p cell probability,
then green-trace robust activity—not by manual choice. Plot limits use each panel's 0.5–99.5th percentiles;
traces are non-overlapping mean-binned to at most 1,500 points for display, and the CSV metrics use the
unbinned traces.</p>
<h3>Independent channel baseline-subtracted traces</h3>
<img src="selected_cells_channels_zoom.png" alt="Selected green and red trace zoom">
<details><summary>Show full channel traces</summary><img src="selected_cells_channels.png" alt="Full green and red traces"></details>
<h2>Selected-cell metrics</h2><table><thead><tr><th>ROI</th><th>p(cell)</th>
<th>green dF σ</th><th>green dF p99</th><th>red dF σ</th><th>red dF p99</th></tr></thead><tbody>
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

    signals = channel_signals(F, Fneu, F2, Fneu2, neucoeff)
    channel_dff: dict[str, np.ndarray] = {
        "green": compute_channel_delta(F, Fneu, ops, neucoeff),
        "red": compute_channel_delta(F2, Fneu2, ops, neucoeff),
    }
    for name in ("green", "red"):
        if not np.all(np.isfinite(channel_dff[name])):
            raise RuntimeError(f"Non-finite values found in {name} dF")
    selected = select_rois(iscell, channel_dff["green"], n_cells)
    metrics = metric_rows(signals, channel_dff, iscell)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        zoom_frames = min(F.shape[1], int(round(2.0 * 60.0 * float(ops["fs"]))))
        zoom_start = max(0, (F.shape[1] - zoom_frames) // 2)
        zoom_stop = zoom_start + zoom_frames
        plot_selected(
            temporary / "selected_cells_channels.png",
            selected,
            channel_dff,
            iscell,
            float(ops["fs"]),
            title="Full session: independent green/red dF; y-axis robustly scaled per panel",
        )
        plot_selected(
            temporary / "selected_cells_channels_zoom.png",
            selected,
            channel_dff,
            iscell,
            float(ops["fs"]),
            frame_slice=slice(zoom_start, zoom_stop),
            title="Central two-minute window: independent green/red dF; y-axis robustly scaled per panel",
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
            "selected_rois": selected.tolist(),
            "zoom_frame_range": [zoom_start, zoom_stop],
            "selection_rule": "iscell first; descending cell probability, green robust sigma, ROI index",
            "channel_formula": "dcnv.preprocess(F - neucoeff * Fneu) independently per channel",
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

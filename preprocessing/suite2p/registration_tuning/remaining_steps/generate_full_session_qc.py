#!/usr/bin/env python3
"""Generate one combined registration, detection, and trace QC report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from common import write_json_atomic  # noqa: E402
from evaluate import array_from, compute_metrics, load_ops, show_image  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required workflow metadata is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("state") != "complete":
        raise RuntimeError(f"Stage is not complete according to {path}: {value.get('state')!r}")
    return value


def load_trace_metrics(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                key: float(value) if key not in {"roi", "iscell"} else int(float(value))
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def save_registration_qc(
    path: Path, ops: dict[str, Any], outputs: dict[str, Any], metrics: dict[str, float]
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    show_image(axes[0, 0], array_from(outputs, ops, key="refImg"), "Reference image")
    show_image(axes[0, 1], array_from(outputs, ops, key="meanImg"), "Registered mean image")
    corr = array_from(outputs, ops, key="corrXY").ravel()
    axes[0, 2].plot(corr, linewidth=0.6)
    axes[0, 2].set(title="Frame-to-reference correlation", xlabel="frame", ylabel="corrXY")
    yoff = array_from(outputs, ops, key="yoff").ravel()
    xoff = array_from(outputs, ops, key="xoff").ravel()
    axes[1, 0].plot(yoff, linewidth=0.6, label="y")
    axes[1, 0].plot(xoff, linewidth=0.6, label="x")
    axes[1, 0].set(title="Rigid offsets", xlabel="frame", ylabel="pixels")
    axes[1, 0].legend()
    if min(yoff.size, xoff.size):
        axes[1, 1].hist(np.hypot(yoff[: min(yoff.size, xoff.size)], xoff[: min(yoff.size, xoff.size)]), bins=40)
    axes[1, 1].set(title="Rigid displacement", xlabel="pixels", ylabel="frames")
    axes[1, 2].text(
        0,
        1,
        "\n".join(
            f"{key}: {value:.5g}" if np.isfinite(value) else f"{key}: n/a"
            for key, value in metrics.items()
        ),
        va="top",
        family="monospace",
        fontsize=9,
    )
    axes[1, 2].axis("off")
    figure.suptitle("Suite2p registration QC")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_summary(
    path: Path,
    ops: dict[str, Any],
    outputs: dict[str, Any],
    masks: np.ndarray,
    iscell: np.ndarray,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    image = array_from(outputs, ops, key="meanImg")
    low, high = np.percentile(image[np.isfinite(image)], (1, 99.5))
    axes[0, 0].imshow(image, cmap="gray", vmin=low, vmax=high)
    outline = np.zeros(masks.shape, dtype=bool)
    outline[1:] |= (masks[1:] > 0) & (masks[1:] != masks[:-1])
    outline[:-1] |= (masks[:-1] > 0) & (masks[:-1] != masks[1:])
    outline[:, 1:] |= (masks[:, 1:] > 0) & (masks[:, 1:] != masks[:, :-1])
    outline[:, :-1] |= (masks[:, :-1] > 0) & (masks[:, :-1] != masks[:, 1:])
    axes[0, 0].contour(outline, levels=[0.5], colors="lime", linewidths=0.4)
    axes[0, 0].set_title(f"Registered mean image with {len(iscell)} ROIs")
    axes[0, 0].axis("off")
    corr = array_from(outputs, ops, key="corrXY").ravel()
    axes[0, 1].plot(corr, linewidth=0.5)
    axes[0, 1].set(title="Registration correlation across full session", xlabel="frame", ylabel="corrXY")
    axes[1, 0].hist(iscell[:, 1], bins=np.linspace(0, 1, 21), color="#5f6f7a")
    axes[1, 0].axvline(0.5, color="#d43f3a", linestyle="--")
    axes[1, 0].set(title="Suite2p cell probabilities", xlabel="p(cell)", ylabel="ROIs")
    yoff = array_from(outputs, ops, key="yoff").ravel()
    xoff = array_from(outputs, ops, key="xoff").ravel()
    axes[1, 1].plot(yoff, linewidth=0.5, label="y")
    axes[1, 1].plot(xoff, linewidth=0.5, label="x")
    axes[1, 1].set(title="Rigid offsets across full session", xlabel="frame", ylabel="pixels")
    axes[1, 1].legend()
    figure.suptitle("Full-session Suite2p QC")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--script-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    run_dir = args.run_dir.resolve()
    suite2p_dir = run_dir / "suite2p"
    plane_dir = suite2p_dir / "plane0"
    qc_dir = suite2p_dir / "qc"
    workflow_dir = suite2p_dir / ".workflow"
    detection_dir = qc_dir / "detection"
    trace_metadata_dir = workflow_dir / "trace_extraction"
    read_json(workflow_dir / "registration_status.json")
    detection_status = read_json(detection_dir / "status.json")
    trace_status = read_json(trace_metadata_dir / "status.json")
    if (qc_dir / "index.html").exists():
        raise FileExistsError(f"Refusing to overwrite completed QC report: {qc_dir}")

    trace_qc_dir = qc_dir / "traces"
    subprocess.run(
        [
            sys.executable,
            str(args.script_dir / "generate_trace_qc.py"),
            "--trace-dir",
            str(plane_dir),
            "--output-dir",
            str(trace_qc_dir),
        ],
        check=True,
    )

    ops, outputs = load_ops(plane_dir / "ops.npy")
    reg_metrics = compute_metrics(ops, outputs)
    iscell = np.load(plane_dir / "iscell.npy")
    masks = np.load(detection_dir / "roi_masks.npy")
    trace_metrics = load_trace_metrics(trace_qc_dir / "trace_metrics.csv")
    classified = [row for row in trace_metrics if row["iscell"]]
    selected = classified or trace_metrics
    areas = np.load(detection_dir / "roi_areas.npy")
    diameters = np.load(detection_dir / "roi_diameters.npy")

    registration_dir = qc_dir / "registration"
    registration_dir.mkdir(parents=True, exist_ok=True)
    save_registration_qc(registration_dir / "registration_qc.png", ops, outputs, reg_metrics)
    save_summary(qc_dir / "summary.png", ops, outputs, masks, iscell)

    session_metrics: dict[str, Any] = {
        **reg_metrics,
        "n_rois": int(len(iscell)),
        "n_cells": int(np.count_nonzero(iscell[:, 0] > 0.5)),
        "cell_fraction": float(np.mean(iscell[:, 0] > 0.5)),
        "cell_probability_median": float(np.median(iscell[:, 1])),
        "roi_area_median_pixels": float(np.median(areas)),
        "roi_diameter_median_pixels": float(np.median(diameters)),
        "green_red_correlation_median": float(np.median([row["green_red_correlation"] for row in selected])),
        "ratio_red_floor_fraction_median": float(np.median([row["ratio_red_floor_fraction"] for row in selected])),
        "green_dff_robust_sigma_median": float(np.median([row["green_dff_robust_sigma"] for row in selected])),
        "elapsed_registration_seconds": float(read_json(workflow_dir / "registration_status.json")["elapsed_seconds"]),
        "elapsed_detection_seconds": float(detection_status["elapsed_seconds"]),
        "elapsed_trace_seconds": float(trace_status["elapsed_seconds"]),
    }
    write_metrics(qc_dir / "session_metrics.csv", session_metrics)
    trace_calculation = json.loads((trace_qc_dir / "calculation.json").read_text(encoding="utf-8"))
    calculation = {
        "session_metrics": session_metrics,
        "trace_calculation": trace_calculation,
        "registration_status": str(workflow_dir / "registration_status.json"),
        "detection_status": str(detection_dir / "status.json"),
        "trace_status": str(trace_metadata_dir / "status.json"),
    }
    (qc_dir / "calculation.json").write_text(json.dumps(calculation, indent=2) + "\n", encoding="utf-8")

    metric_rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{value:.5g}</td></tr>"
        for key, value in session_metrics.items()
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Full-session Suite2p QC</title><style>
body{{font:15px system-ui,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1rem;color:#222}}
img{{max-width:100%;height:auto;border:1px solid #ddd}} table{{border-collapse:collapse}}
th,td{{padding:.35rem .6rem;border:1px solid #ccc;text-align:right}} th{{text-align:left}}
</style></head><body><h1>Full-session Suite2p QC</h1>
<p><b>Output:</b> <code>{html.escape(str(plane_dir))}</code></p>
<h2>Combined summary</h2><img src="summary.png" alt="Combined Suite2p QC">
<h2>Metrics</h2><table>{metric_rows}</table>
<h2>Registration</h2><img src="registration/registration_qc.png" alt="Registration QC">
<h2>ROI detection</h2><img src="detection/qc.png" alt="Detection QC">
<h2>Trace extraction</h2><img src="traces/summary.png" alt="Trace summary">
<h3>Full-session CellTV-style traces</h3><img src="traces/selected_cells_celltv.png" alt="Full-session CellTV traces">
<h3>Full-session normalized traces</h3><img src="traces/selected_cells_dff.png" alt="Full-session normalized traces">
<p><a href="session_metrics.csv">Session metrics CSV</a> · <a href="traces/trace_metrics.csv">Per-ROI metrics CSV</a> · <a href="calculation.json">Calculation settings</a></p>
</body></html>"""
    (qc_dir / "index.html").write_text(document, encoding="utf-8")
    finished = time.time()
    status = {
        "state": "complete",
        "started_unix": started,
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "report": str(qc_dir / "index.html"),
        "summary_png": str(qc_dir / "summary.png"),
    }
    write_json_atomic(workflow_dir / "status.json", status)
    write_json_atomic(workflow_dir / "qc_provenance.json", status)
    print(qc_dir / "index.html")


if __name__ == "__main__":
    main()

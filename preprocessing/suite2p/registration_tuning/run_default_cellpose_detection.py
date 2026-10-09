#!/usr/bin/env python3
"""Run default Suite2p Cellpose ROI detection without extraction or deconvolution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import write_json_atomic


FORBIDDEN_DOWNSTREAM_OUTPUTS = ("F.npy", "Fneu.npy", "iscell.npy", "spks.npy")


def atomic_save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def one_plane(run_dir: Path) -> Path:
    planes = sorted(run_dir.glob("suite2p/plane*"))
    if len(planes) != 1:
        raise ValueError(f"Expected exactly one Suite2p plane in {run_dir}, found {len(planes)}")
    return planes[0]


def masks_from_stat(stat: np.ndarray, Ly: int, Lx: int) -> np.ndarray:
    masks = np.zeros((Ly, Lx), dtype=np.int32)
    for label, roi in enumerate(stat, start=1):
        ypix = np.asarray(roi["ypix"], dtype=int)
        xpix = np.asarray(roi["xpix"], dtype=int)
        masks[ypix, xpix] = label
    return masks


def boundaries(masks: np.ndarray) -> np.ndarray:
    outline = np.zeros(masks.shape, dtype=bool)
    outline[1:, :] |= (masks[1:, :] > 0) & (masks[1:, :] != masks[:-1, :])
    outline[:-1, :] |= (masks[:-1, :] > 0) & (masks[:-1, :] != masks[1:, :])
    outline[:, 1:] |= (masks[:, 1:] > 0) & (masks[:, 1:] != masks[:, :-1])
    outline[:, :-1] |= (masks[:, :-1] > 0) & (masks[:, :-1] != masks[:, 1:])
    outline[[0, -1], :] |= masks[[0, -1], :] > 0
    outline[:, [0, -1]] |= masks[:, [0, -1]] > 0
    return outline


def model_provenance() -> dict[str, Any]:
    from cellpose import models

    candidates: list[Path] = []
    for attribute in ("MODEL_DIR", "MODEL_DIR_DEFAULT", "MODELS_DIR"):
        value = getattr(models, attribute, None)
        if value:
            candidates.append(Path(value) / "cpsam")
    candidates.append(Path.home() / ".cellpose" / "models" / "cpsam")
    for path in candidates:
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return {"name": "cpsam", "path": str(path), "sha256": digest.hexdigest()}
    return {"name": "cpsam", "path": None, "sha256": None}


def configure_default_cellpose(settings: dict[str, Any], device: str) -> dict[str, Any]:
    """Set only recording metadata, device, and the requested detection algorithm."""
    settings["fs"] = 45.0
    settings["tau"] = 0.4
    settings["torch_device"] = device
    settings["detection"]["algorithm"] = "cellpose"
    return settings


def save_qc(
    output_dir: Path,
    cellpose_input: np.ndarray,
    mean_image: np.ndarray,
    masks: np.ndarray,
    stat: np.ndarray,
    yrange: list[int] | np.ndarray,
    xrange: list[int] | np.ndarray,
) -> tuple[Path, Path, Path, Path, np.ndarray, np.ndarray]:
    outline = boundaries(masks)
    cropped_outline = outline[slice(*yrange), slice(*xrange)]
    if cropped_outline.shape != cellpose_input.shape:
        raise ValueError(
            f"Cropped ROI outline {cropped_outline.shape} does not match "
            f"Cellpose input {cellpose_input.shape}"
        )

    areas = np.asarray([len(roi["ypix"]) for roi in stat], dtype=float)
    diameters = 2 * np.sqrt(areas / np.pi)

    input_overlay_path = output_dir / "cellpose_input_overlay.png"
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.imshow(cellpose_input, cmap="gray")
    axis.contour(cropped_outline, levels=[0.5], colors="lime", linewidths=0.5)
    axis.set_title(f"Default Cellpose input with {len(stat)} ROI outlines")
    axis.axis("off")
    figure.savefig(input_overlay_path, dpi=160)
    plt.close(figure)

    mean_overlay_path = output_dir / "mean_image_overlay.png"
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.imshow(mean_image, cmap="gray")
    axis.contour(outline, levels=[0.5], colors="magenta", linewidths=0.5)
    axis.set_title("Registered channel-1 mean image with ROI outlines")
    axis.axis("off")
    figure.savefig(mean_overlay_path, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes[0, 0].imshow(cellpose_input, cmap="gray")
    axes[0, 0].set_title("Default Cellpose input: log(max_proj / meanImg)")
    axes[0, 1].imshow(cellpose_input, cmap="gray")
    axes[0, 1].contour(cropped_outline, levels=[0.5], colors="lime", linewidths=0.5)
    axes[0, 1].set_title(f"Cellpose input with {len(stat)} ROI outlines")
    axes[1, 0].imshow(mean_image, cmap="gray")
    axes[1, 0].contour(outline, levels=[0.5], colors="magenta", linewidths=0.5)
    axes[1, 0].set_title("Registered mean image with ROI outlines")
    axes[1, 1].hist(diameters, bins=min(40, max(10, len(diameters) // 10)))
    axes[1, 1].axvline(np.median(diameters), color="black", linestyle="--")
    axes[1, 1].set(xlabel="Equivalent ROI diameter (pixels)", ylabel="ROI count",
                   title=f"Median diameter: {np.median(diameters):.1f} px")
    for axis in axes.flat[:3]:
        axis.axis("off")
    qc_path = output_dir / "qc.png"
    figure.savefig(qc_path, dpi=160)
    plt.close(figure)

    html_path = output_dir / "index.html"
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Default Cellpose pilot</title>"
        f"<h1>Default Cellpose pilot</h1><p>Detected {len(stat)} ROIs; "
        f"median equivalent diameter {np.median(diameters):.1f} pixels.</p>"
        "<p>Model: Suite2p 1.1.0 default Cellpose-SAM (<code>cpsam</code>).</p>"
        "<p><a href='cellpose_input_overlay.png'>Cellpose-input overlay</a> · "
        "<a href='mean_image_overlay.png'>mean-image overlay</a></p>"
        "<img src='qc.png' style='max-width:100%;height:auto' alt='Cellpose QC'>",
        encoding="utf-8",
    )
    return (
        qc_path,
        html_path,
        input_overlay_path,
        mean_overlay_path,
        areas,
        diameters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    import suite2p
    import torch

    suite2p_version = importlib.metadata.version("suite2p")
    if suite2p_version != "1.1.0":
        raise RuntimeError(f"Pilot requires suite2p==1.1.0, found {suite2p_version}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    run_dir = args.run_dir.resolve()
    plane_dir = one_plane(run_dir)
    for name in FORBIDDEN_DOWNSTREAM_OUTPUTS:
        if (plane_dir / name).exists():
            raise FileExistsError(
                f"Detection-only pilot requires a clean registration output; found {plane_dir / name}"
            )
    output_dir = run_dir / "cellpose_exploration" / "default_cpsam"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite Cellpose pilot output: {output_dir}")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    started = time.time()
    write_json_atomic(status_path, {"state": "running", "started_unix": started})

    try:
        db_path = plane_dir / "db.npy"
        reg_outputs_path = plane_dir / "reg_outputs.npy"
        data_path = plane_dir / "data.bin"
        data_chan2_path = plane_dir / "data_chan2.bin"
        for required in (db_path, reg_outputs_path, data_path, data_chan2_path):
            if not required.is_file():
                raise FileNotFoundError(f"Required registered input is missing: {required}")
        db = np.load(db_path, allow_pickle=True).item()
        reg_outputs = np.load(reg_outputs_path, allow_pickle=True).item()
        settings = configure_default_cellpose(suite2p.default_settings(), args.device)
        if settings["diameter"] != [12.0, 12.0]:
            raise RuntimeError(f"Unexpected Suite2p default diameter: {settings['diameter']}")
        expected = settings["detection"]["cellpose_settings"]
        expected_defaults = {
            "cellpose_model": "cpsam",
            "img": "max_proj / meanImg",
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
        }
        for key, value in expected_defaults.items():
            if expected[key] != value:
                raise RuntimeError(
                    f"Unexpected Suite2p Cellpose default {key}={expected[key]!r}; expected {value!r}"
                )

        device = torch.device(args.device)
        with suite2p.io.BinaryFile(
            Ly=int(db["Ly"]), Lx=int(db["Lx"]), filename=data_path,
            n_frames=int(db["nframes"]), write=False,
        ) as f_reg:
            detect_outputs, stat, redcell = suite2p.detection.detection_wrapper(
                f_reg,
                # Channel 2 is retained and used for registration alignment, but
                # is intentionally omitted here to avoid red-cell classification.
                meanImg_chan2=None,
                yrange=reg_outputs["yrange"],
                xrange=reg_outputs["xrange"],
                tau=settings["tau"],
                fs=settings["fs"],
                diameter=settings["diameter"],
                settings=settings["detection"],
                classifier_path=None,
                badframes=np.asarray(reg_outputs["badframes"], dtype=bool),
                preclassify=settings["classification"]["preclassify"],
                device=device,
            )

        if len(stat) == 0:
            raise RuntimeError("Default Cellpose detection returned no ROIs")
        masks = masks_from_stat(stat, int(db["Ly"]), int(db["Lx"]))
        cellpose_input = np.asarray(detect_outputs["Vcorr"])
        mean_image = np.asarray(reg_outputs["meanImg"])
        atomic_save(output_dir / "stat.npy", stat)
        atomic_save(output_dir / "roi_masks.npy", masks)
        atomic_save(output_dir / "detect_outputs.npy", detect_outputs)
        atomic_save(output_dir / "cellpose_input.npy", cellpose_input)
        plt.imsave(output_dir / "cellpose_input.png", cellpose_input, cmap="gray")
        if redcell is not None:
            raise RuntimeError("Detection-only pilot unexpectedly returned red-cell labels")
        (
            qc_path,
            html_path,
            input_overlay_path,
            mean_overlay_path,
            areas,
            diameters,
        ) = save_qc(
            output_dir,
            cellpose_input,
            mean_image,
            masks,
            stat,
            reg_outputs["yrange"],
            reg_outputs["xrange"],
        )
        atomic_save(output_dir / "roi_areas.npy", areas)
        atomic_save(output_dir / "roi_diameters.npy", diameters)

        for name in FORBIDDEN_DOWNSTREAM_OUTPUTS:
            if (plane_dir / name).exists() or (output_dir / name).exists():
                raise RuntimeError(f"Detection-only pilot unexpectedly created {name}")

        cellpose_version = importlib.metadata.version("cellpose")
        torch_version = importlib.metadata.version("torch")
        finished = time.time()
        provenance = {
            "state": "complete",
            "started_unix": started,
            "finished_unix": finished,
            "elapsed_seconds": finished - started,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "suite2p_version": suite2p_version,
            "cellpose_version": cellpose_version,
            "torch_version": torch_version,
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "model": model_provenance(),
            "settings": {
                "fs": settings["fs"],
                "tau": settings["tau"],
                "diameter": settings["diameter"],
                "detection": settings["detection"],
            },
            "n_rois": int(len(stat)),
            "roi_area_pixels": {
                "minimum": float(areas.min()),
                "median": float(np.median(areas)),
                "maximum": float(areas.max()),
            },
            "roi_equivalent_diameter_pixels": {
                "minimum": float(diameters.min()),
                "median": float(np.median(diameters)),
                "maximum": float(diameters.max()),
            },
            "outputs": {
                "stat": str(output_dir / "stat.npy"),
                "masks": str(output_dir / "roi_masks.npy"),
                "cellpose_input": str(output_dir / "cellpose_input.npy"),
                "cellpose_input_png": str(output_dir / "cellpose_input.png"),
                "roi_areas": str(output_dir / "roi_areas.npy"),
                "roi_diameters": str(output_dir / "roi_diameters.npy"),
                "cellpose_input_overlay": str(input_overlay_path),
                "mean_image_overlay": str(mean_overlay_path),
                "qc": str(qc_path),
                "html": str(html_path),
            },
        }
        write_json_atomic(output_dir / "provenance.json", provenance)
        write_json_atomic(status_path, provenance)
        print(f"Default cpsam detection completed with {len(stat)} ROIs")
        print(f"QC report: {html_path}")
    except Exception as exc:
        write_json_atomic(
            status_path,
            {
                "state": "failed",
                "started_unix": started,
                "finished_unix": time.time(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()

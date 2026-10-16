#!/usr/bin/env python3
"""Run one named Cellpose experiment on an existing registered Suite2p movie."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
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
from scipy.ndimage import gaussian_filter


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
SUITE2P_DIR = WORKFLOW_DIR.parent
PIPELINE_DIR = SUITE2P_DIR / "full_session_pipeline"
sys.path.insert(0, str(WORKFLOW_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

from common import write_json_atomic

try:  # Support direct script execution and package imports in tests.
    from .config import FS, SUITE2P_VERSION, TAU
    from .run_roi_detection import FORBIDDEN_DOWNSTREAM_OUTPUTS, boundaries
except ImportError:
    from full_session_config import FS, SUITE2P_VERSION, TAU
    from run_roi_detection import FORBIDDEN_DOWNSTREAM_OUTPUTS, boundaries


INPUT_IMAGE_CHOICES = (
    "max_proj / meanImg",
    "meanImg",
    "max_proj",
    "registration_meanImg",
)
RESERVED_EVAL_PARAMS = {"diameter", "flow_threshold", "cellprob_threshold"}


def atomic_save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    if not result:
        raise ValueError("Experiment name must contain a letter or number")
    return result


def one_plane(run_dir: Path) -> Path:
    planes = sorted(run_dir.glob("suite2p/plane*"))
    if len(planes) != 1:
        raise ValueError(f"Expected exactly one Suite2p plane in {run_dir}, found {len(planes)}")
    return planes[0]


def parse_cellpose_params(values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Cellpose parameter must be NAME=JSON_VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Cellpose parameter has an empty name: {value!r}")
        if key in RESERVED_EVAL_PARAMS:
            raise ValueError(f"Use the dedicated --{key.replace('_', '-')} option for {key}")
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Value for {key} must be valid JSON (examples: 5, true, [1, 2]), got {raw!r}"
            ) from exc
    return parsed


def build_cellpose_input(
    mean_img: np.ndarray,
    max_proj: np.ndarray,
    input_image: str,
    diameter: list[float],
    highpass_spatial: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match Suite2p 1.1.0 anatomical.select_rois image preparation."""
    if input_image == "max_proj / meanImg":
        image = np.log(np.maximum(1e-3, max_proj / np.maximum(1e-3, mean_img)))
        weights = max_proj
    elif input_image == "meanImg":
        image = mean_img.copy()
        p1, p99 = np.percentile(mean_img, (1, 99))
        scale = max(float(p99 - p1), np.finfo(float).eps)
        weights = 0.1 + np.clip((mean_img - p1) / scale, 0, 1)
    elif input_image == "max_proj":
        image = max_proj.copy()
        weights = max_proj
    else:
        raise ValueError(f"Unsupported Cellpose input image: {input_image!r}")

    if highpass_spatial:
        from cellpose.transforms import normalize99

        image = np.clip(normalize99(image), 0, 1)
        # Suite2p 1.1.0 applies this filter subtraction twice.
        sigma = diameter[1] * highpass_spatial
        image -= gaussian_filter(image, sigma)
        image -= gaussian_filter(image, sigma)
    return np.asarray(image), np.asarray(weights)


def validate_model(cellpose_version: str, model_name: str, models: Any) -> str:
    """Reject unsupported names instead of allowing Cellpose's silent fallback."""
    model_path = Path(model_name).expanduser()
    if model_path.is_file():
        return str(model_path.resolve())
    major = int(cellpose_version.split(".", 1)[0])
    available = set(getattr(models, "MODEL_NAMES", ()))
    if model_name not in available:
        if model_name == "cyto3" and major >= 4:
            raise RuntimeError(
                "cyto3 requires Cellpose 3 (recommended: 3.1.1.2). Cellpose 4 would "
                "silently substitute a Cellpose-4 model, so this run was stopped."
            )
        raise ValueError(
            f"Model {model_name!r} is not available in cellpose=={cellpose_version}; "
            f"available built-ins: {sorted(available)}"
        )
    return model_name


def create_model(models: Any, model_name: str, cellpose_version: str, device: Any) -> Any:
    major = int(cellpose_version.split(".", 1)[0])
    if major >= 4:
        return models.CellposeModel(
            pretrained_model=model_name,
            gpu=device.type == "cuda",
            device=device,
        )
    return models.CellposeModel(
        model_type=model_name,
        gpu=device.type == "cuda",
        device=device,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_provenance(model: Any, requested: str) -> dict[str, Any]:
    resolved = getattr(model, "pretrained_model", None)
    if isinstance(resolved, (list, tuple)) and len(resolved) == 1:
        resolved = resolved[0]
    path = Path(str(resolved)).expanduser() if resolved else None
    return {
        "requested": requested,
        "resolved": str(resolved) if resolved is not None else None,
        "path": str(path.resolve()) if path is not None and path.is_file() else None,
        "sha256": sha256_file(path) if path is not None and path.is_file() else None,
    }


def full_masks_from_stat(stat: np.ndarray, Ly: int, Lx: int) -> np.ndarray:
    masks = np.zeros((Ly, Lx), dtype=np.int32)
    for label, roi in enumerate(stat, start=1):
        masks[np.asarray(roi["ypix"], int), np.asarray(roi["xpix"], int)] = label
    return masks


def relabel_masks(masks: np.ndarray) -> np.ndarray:
    """Make labels contiguous, matching Suite2p anatomical.roi_detect."""
    shape = masks.shape
    _, relabelled = np.unique(np.asarray(masks, dtype=np.int32), return_inverse=True)
    return relabelled.reshape(shape).astype(np.int32, copy=False)


def robust_display_limits(image: np.ndarray) -> tuple[float, float]:
    """Return finite 1st/99th-percentile limits for QC rendering only."""
    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = (float(value) for value in np.percentile(finite, (1, 99)))
    if vmax <= vmin:
        padding = max(abs(vmin) * 0.01, 1.0)
        return vmin - padding, vmax + padding
    return vmin, vmax


def save_qc(
    output_dir: Path,
    image: np.ndarray,
    registered_mean: np.ndarray,
    crop_masks: np.ndarray,
    full_masks: np.ndarray,
    stat: np.ndarray,
    title: str,
) -> dict[str, str]:
    crop_outline = boundaries(crop_masks)
    full_outline = boundaries(full_masks)
    areas = np.asarray([len(roi["ypix"]) for roi in stat], dtype=float)
    diameters = 2 * np.sqrt(areas / np.pi) if len(areas) else np.asarray([], dtype=float)
    atomic_save(output_dir / "roi_areas.npy", areas)
    atomic_save(output_dir / "roi_diameters.npy", diameters)
    input_vmin, input_vmax = robust_display_limits(image)
    mean_vmin, mean_vmax = robust_display_limits(registered_mean)

    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=input_vmin, vmax=input_vmax)
    if crop_outline.any():
        axes[0].contour(crop_outline, levels=[0.5], colors="lime", linewidths=0.5)
    axes[0].set_title(f"Cellpose input: {title}\n{int(crop_masks.max())} raw masks")
    axes[1].imshow(
        registered_mean, cmap="gray", vmin=mean_vmin, vmax=mean_vmax
    )
    if full_outline.any():
        axes[1].contour(full_outline, levels=[0.5], colors="magenta", linewidths=0.5)
    axes[1].set_title("Registered mean image")
    if len(diameters):
        axes[2].hist(diameters, bins=min(40, max(10, len(diameters) // 10)))
        axes[2].axvline(np.median(diameters), color="black", linestyle="--")
        axes[2].set_title(f"Median diameter: {np.median(diameters):.1f} px")
    else:
        axes[2].text(0.5, 0.5, "No ROIs detected", ha="center", va="center")
        axes[2].set_title("ROI size distribution")
    axes[2].set(xlabel="Equivalent ROI diameter (pixels)", ylabel="ROI count")
    for axis in axes[:2]:
        axis.axis("off")
    qc_path = output_dir / "qc.png"
    figure.savefig(qc_path, dpi=160)
    plt.close(figure)

    input_path = output_dir / "cellpose_input.png"
    plt.imsave(
        input_path, image, cmap="gray", vmin=input_vmin, vmax=input_vmax
    )
    html_path = output_dir / "index.html"
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Cellpose experiment</title>"
        f"<h1>{title}</h1><p>Final ROI count: {len(stat)}</p>"
        "<p>The report is retained even when the model finds zero ROIs. "
        "Image panels use 1st–99th percentile display contrast; detection arrays "
        "are unchanged.</p>"
        "<img src='qc.png' style='max-width:100%;height:auto' alt='ROI QC'>",
        encoding="utf-8",
    )
    return {"qc": str(qc_path), "html": str(html_path), "input_png": str(input_path)}


def prepare_images(
    suite2p: Any,
    plane_dir: Path,
    db: dict[str, Any],
    reg_outputs: dict[str, Any],
    settings: dict[str, Any],
    nbins: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build or reuse the expensive binned-movie mean and max projection."""
    nframes = int(db["nframes"])
    bin_size = settings.get("bin_size") or int(
        max(1, nframes // nbins, np.round(TAU * FS))
    )
    cache_key = f"nbins-{nbins}_binsize-{bin_size}_hpt-{settings['highpass_time']}"
    cache_dir = plane_dir.parent.parent / "roi_detection" / "prepared_images" / safe_name(cache_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    status_path = cache_dir / "status.json"
    mean_path = cache_dir / "meanImg_binned.npy"
    max_path = cache_dir / "max_proj.npy"
    if status_path.is_file() and mean_path.is_file() and max_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "complete":
            return np.load(mean_path), np.load(max_path), status

    write_json_atomic(status_path, {"state": "running", "started_unix": time.time()})
    with suite2p.io.BinaryFile(
        Ly=int(db["Ly"]), Lx=int(db["Lx"]), filename=plane_dir / "data.bin",
        n_frames=nframes, write=False,
    ) as f_reg:
        mov = suite2p.detection.detect.bin_movie(
            f_reg,
            bin_size,
            yrange=reg_outputs["yrange"],
            xrange=reg_outputs["xrange"],
            badframes=np.asarray(reg_outputs["badframes"], dtype=bool),
            nbins=nbins,
        )
    if settings.get("inverted_activity", False):
        mov -= mov.min()
        mov *= -1
        mov -= mov.min()
    if settings.get("denoise", False):
        mov = suite2p.detection.denoise.pca_denoise(
            mov, block_size=settings["block_size"], n_comps_frac=0.5
        )
    mean_img = mov.mean(axis=0)
    mov = suite2p.detection.utils.temporal_high_pass_filter(
        mov=mov, width=settings["highpass_time"]
    )
    max_proj = mov.max(axis=0)
    del mov
    atomic_save(mean_path, mean_img)
    atomic_save(max_path, max_proj)
    cache_status = {
        "state": "complete",
        "finished_unix": time.time(),
        "nbins": nbins,
        "bin_size": bin_size,
        "highpass_time": settings["highpass_time"],
        "mean_image": str(mean_path),
        "max_projection": str(max_path),
    }
    write_json_atomic(status_path, cache_status)
    return mean_img, max_proj, cache_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model", default="cpsam")
    parser.add_argument("--input-image", choices=INPUT_IMAGE_CHOICES, default="max_proj / meanImg")
    parser.add_argument("--diameter-y", type=float, default=12.0)
    parser.add_argument("--diameter-x", type=float, default=12.0)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--highpass-spatial", type=float, default=0.0)
    parser.add_argument("--nbins", type=int, default=None)
    parser.add_argument(
        "--cellpose-param", action="append", default=[], metavar="NAME=JSON_VALUE",
        help="additional CellposeModel.eval option; repeat as needed",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    import suite2p
    import torch
    from cellpose import models
    from suite2p.detection.anatomical import masks_to_stats
    from suite2p.detection.stats import roi_stats

    suite2p_version = importlib.metadata.version("suite2p")
    cellpose_version = importlib.metadata.version("cellpose")
    if suite2p_version != SUITE2P_VERSION:
        raise RuntimeError(f"Requires suite2p=={SUITE2P_VERSION}, found {suite2p_version}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if args.nbins is not None and args.nbins < 1:
        raise ValueError("--nbins must be positive")
    diameter = [args.diameter_y, args.diameter_x]
    if min(diameter) <= 0:
        raise ValueError("Both diameters must be positive")
    extra_params = parse_cellpose_params(args.cellpose_param)
    validated_model = validate_model(cellpose_version, args.model, models)

    run_dir = args.run_dir.resolve()
    plane_dir = one_plane(run_dir)
    for required in ("db.npy", "reg_outputs.npy", "data.bin", "data_chan2.bin"):
        if not (plane_dir / required).is_file():
            raise FileNotFoundError(f"Required registered input is missing: {plane_dir / required}")
    for name in FORBIDDEN_DOWNSTREAM_OUTPUTS:
        if (plane_dir / name).exists():
            raise FileExistsError(f"Detection-only run found downstream output: {plane_dir / name}")

    output_dir = run_dir / "roi_detection" / "experiments" / safe_name(args.experiment_name)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experiment: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    started = time.time()
    requested = {
        "model": args.model,
        "input_image": args.input_image,
        "diameter": diameter,
        "flow_threshold": args.flow_threshold,
        "cellprob_threshold": args.cellprob_threshold,
        "highpass_spatial": args.highpass_spatial,
        "nbins": args.nbins,
        "cellpose_eval_params": extra_params,
        "device": args.device,
    }
    write_json_atomic(status_path, {"state": "running", "started_unix": started, "settings": requested})

    try:
        db = np.load(plane_dir / "db.npy", allow_pickle=True).item()
        reg_outputs = np.load(plane_dir / "reg_outputs.npy", allow_pickle=True).item()
        detection_settings = suite2p.default_settings()["detection"]
        nbins = args.nbins or int(detection_settings["nbins"])
        if args.input_image == "registration_meanImg":
            yrange = [int(value) for value in reg_outputs["yrange"]]
            xrange = [int(value) for value in reg_outputs["xrange"]]
            registration_mean = np.asarray(reg_outputs["meanImg"])
            registration_mean_crop = registration_mean[
                slice(*yrange), slice(*xrange)
            ]
            image, weights = build_cellpose_input(
                registration_mean_crop,
                registration_mean_crop,
                "meanImg",
                diameter,
                args.highpass_spatial,
            )
            mean_img = registration_mean_crop
            max_proj = None
            cache_status = {
                "used": False,
                "reason": "registration_meanImg uses reg_outputs.npy directly",
                "source": str(plane_dir / "reg_outputs.npy"),
                "yrange": yrange,
                "xrange": xrange,
            }
        else:
            mean_img, max_proj, cache_status = prepare_images(
                suite2p, plane_dir, db, reg_outputs, detection_settings, nbins
            )
            image, weights = build_cellpose_input(
                mean_img, max_proj, args.input_image, diameter, args.highpass_spatial
            )
        atomic_save(output_dir / "cellpose_input.npy", image)
        if args.input_image == "registration_meanImg":
            atomic_save(output_dir / "registration_meanImg_crop.npy", mean_img)
        else:
            atomic_save(output_dir / "meanImg_binned.npy", mean_img)
            atomic_save(output_dir / "max_proj.npy", max_proj)
        write_json_atomic(
            output_dir / "requested_settings.json",
            {**requested, "nbins": nbins, "suite2p_detection_defaults": detection_settings},
        )

        device = torch.device(args.device)
        model = create_model(models, validated_model, cellpose_version, device)
        model_info = model_provenance(model, args.model)
        model_image = image
        resize_factor = args.diameter_x / args.diameter_y
        if resize_factor != 1.0:
            import cv2

            model_image = cv2.resize(
                image, (image.shape[1], int(image.shape[0] * resize_factor))
            )
        evaluation = model.eval(
            model_image,
            diameter=args.diameter_x,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            **extra_params,
        )
        atomic_save(
            output_dir / "cellpose_model_output.npy",
            {"flows": evaluation[1], "styles": evaluation[2]},
        )
        masks_crop = np.asarray(evaluation[0], dtype=np.int32)
        if resize_factor != 1.0:
            masks_crop = cv2.resize(
                masks_crop,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        masks_crop = relabel_masks(masks_crop)
        if masks_crop.shape != image.shape:
            raise ValueError(f"Cellpose returned masks {masks_crop.shape}, expected {image.shape}")
        atomic_save(output_dir / "cellpose_masks_crop.npy", masks_crop)

        raw_stat = masks_to_stats(masks_crop, weights) if masks_crop.max() else np.asarray([], dtype=object)
        ymin, xmin = int(reg_outputs["yrange"][0]), int(reg_outputs["xrange"][0])
        for roi in raw_stat:
            roi["ypix"] += ymin
            roi["xpix"] += xmin
            roi["med"][0] += ymin
            roi["med"][1] += xmin
        atomic_save(output_dir / "stat_before_suite2p_filters.npy", raw_stat)
        if len(raw_stat):
            stat = roi_stats(
                raw_stat,
                int(db["Ly"]),
                int(db["Lx"]),
                diameter=diameter,
                max_overlap=detection_settings.get("max_overlap", 0.75),
                do_soma_crop=detection_settings.get("soma_crop", 1),
                npix_norm_min=detection_settings.get("npix_norm_min", 0.0),
                npix_norm_max=detection_settings.get("npix_norm_max", 100.0),
                median=True,
            )
        else:
            stat = raw_stat
        atomic_save(output_dir / "stat.npy", stat)
        full_masks = full_masks_from_stat(stat, int(db["Ly"]), int(db["Lx"]))
        atomic_save(output_dir / "roi_masks.npy", full_masks)
        qc_outputs = save_qc(
            output_dir,
            image,
            np.asarray(reg_outputs["meanImg"]),
            masks_crop,
            full_masks,
            stat,
            f"{args.experiment_name}: {args.model} on {args.input_image}",
        )

        for name in FORBIDDEN_DOWNSTREAM_OUTPUTS:
            if (plane_dir / name).exists() or (output_dir / name).exists():
                raise RuntimeError(f"Detection-only experiment unexpectedly created {name}")
        finished = time.time()
        provenance = {
            "state": "complete",
            "outcome": "rois_detected" if len(stat) else "zero_rois",
            "started_unix": started,
            "finished_unix": finished,
            "elapsed_seconds": finished - started,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "suite2p_version": suite2p_version,
            "cellpose_version": cellpose_version,
            "torch_version": importlib.metadata.version("torch"),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "settings": {**requested, "nbins": nbins},
            "model": model_info,
            "prepared_image_cache": cache_status,
            "n_masks_from_cellpose": int(masks_crop.max()),
            "n_rois_before_suite2p_filters": int(len(raw_stat)),
            "n_rois": int(len(stat)),
            "outputs": {
                "cellpose_input": str(output_dir / "cellpose_input.npy"),
                "registration_meanImg_crop": (
                    str(output_dir / "registration_meanImg_crop.npy")
                    if args.input_image == "registration_meanImg"
                    else None
                ),
                "cellpose_model_output": str(output_dir / "cellpose_model_output.npy"),
                "cellpose_masks_crop": str(output_dir / "cellpose_masks_crop.npy"),
                "stat_before_suite2p_filters": str(output_dir / "stat_before_suite2p_filters.npy"),
                "stat": str(output_dir / "stat.npy"),
                "roi_masks": str(output_dir / "roi_masks.npy"),
                **qc_outputs,
            },
        }
        write_json_atomic(output_dir / "provenance.json", provenance)
        write_json_atomic(status_path, provenance)
        print(f"Experiment completed: {len(stat)} ROIs ({masks_crop.max()} raw masks)")
        print(f"QC report: {output_dir / 'index.html'}")
    except Exception as exc:
        write_json_atomic(
            status_path,
            {
                "state": "failed",
                "started_unix": started,
                "finished_unix": time.time(),
                "settings": requested,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()

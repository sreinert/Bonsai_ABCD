#!/usr/bin/env python3
"""Export viewable channel mean images from Suite2p ops.npy files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


def array_from_ops(ops: dict[str, Any], key: str) -> np.ndarray:
    nested = ops.get("reg_outputs")
    for mapping in (nested, ops):
        if isinstance(mapping, dict) and mapping.get(key) is not None:
            return np.asarray(mapping[key])
    return np.asarray([])


def normalized_preview(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if image.ndim != 2 or not finite.size:
        raise ValueError(f"Expected a finite 2-D mean image, received {image.shape}")
    low, high = np.percentile(finite, [1, 99.5])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0, 1)


def export_plane(ops_path: Path, *, require_chan2: bool = True) -> list[Path]:
    ops = np.load(ops_path, allow_pickle=True).item()
    chan1 = array_from_ops(ops, "meanImg")
    chan2 = array_from_ops(ops, "meanImg_chan2")
    if chan1.ndim != 2 or not chan1.size:
        raise KeyError(f"meanImg is missing or invalid in {ops_path}")
    if require_chan2 and (chan2.ndim != 2 or not chan2.size):
        raise KeyError(f"meanImg_chan2 is missing or invalid in {ops_path}")

    images = [("chan1", chan1)]
    if chan2.ndim == 2 and chan2.size:
        images.append(("chan2", chan2))

    exported: list[Path] = []
    previews: list[tuple[str, np.ndarray]] = []
    for label, image in images:
        image = np.asarray(image, dtype=np.float32)
        preview = normalized_preview(image)
        tiff_path = ops_path.parent / f"meanImg_{label}.tiff"
        png_path = ops_path.parent / f"meanImg_{label}.png"
        tifffile.imwrite(tiff_path, image, photometric="minisblack")
        plt.imsave(png_path, preview, cmap="gray", vmin=0, vmax=1)
        exported.extend((tiff_path, png_path))
        previews.append((label, preview))

    figure, axes = plt.subplots(1, len(previews), figsize=(6 * len(previews), 6))
    axes_array = np.atleast_1d(axes)
    for axis, (label, preview) in zip(axes_array, previews):
        axis.imshow(preview, cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"Registered mean image — {label}")
        axis.set_axis_off()
    figure.tight_layout()
    comparison_path = ops_path.parent / "meanImgs.png"
    figure.savefig(comparison_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    exported.append(comparison_path)
    return exported


def export_mean_images(run_dir: Path, *, require_chan2: bool = True) -> list[Path]:
    ops_files = sorted(run_dir.glob("suite2p/plane*/ops.npy"))
    if not ops_files:
        raise FileNotFoundError(f"No suite2p/plane*/ops.npy found below {run_dir}")
    exported: list[Path] = []
    for ops_path in ops_files:
        exported.extend(export_plane(ops_path, require_chan2=require_chan2))
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-missing-chan2",
        action="store_true",
        help="Export channel 1 even if Suite2p did not produce meanImg_chan2",
    )
    args = parser.parse_args()

    paths = export_mean_images(
        args.run_dir.resolve(),
        require_chan2=not args.allow_missing_chan2,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

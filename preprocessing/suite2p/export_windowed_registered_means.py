#!/usr/bin/env python3
"""Export temporal-window means from an existing registered Suite2p movie."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


def resolve_plane_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "db.npy").is_file() and (path / "data.bin").is_file():
        return path
    direct = path / "suite2p" / "plane0"
    if (direct / "db.npy").is_file() and (direct / "data.bin").is_file():
        return direct
    planes = sorted(path.glob("suite2p/plane*"))
    valid = [plane for plane in planes if (plane / "db.npy").is_file()]
    if len(valid) != 1:
        raise FileNotFoundError(
            f"Could not find exactly one registered Suite2p plane below {path}"
        )
    return valid[0]


def display_image(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if not finite.size:
        raise ValueError("Window mean contains no finite pixels")
    low, high = np.percentile(finite, (1, 99.5))
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0, 1).astype(np.float32)


def normalized_sharpness(image: np.ndarray) -> float:
    preview = display_image(image)
    gradient_y, gradient_x = np.gradient(preview)
    return float(np.mean(gradient_x * gradient_x + gradient_y * gradient_y))


def window_mean(
    movie: np.memmap,
    start: int,
    stop: int,
    badframes: np.ndarray,
    *,
    chunk_size: int = 256,
) -> tuple[np.ndarray, int, int]:
    accumulator = np.zeros(movie.shape[1:], dtype=np.float64)
    used = 0
    bad = 0
    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(stop, chunk_start + chunk_size)
        valid = ~badframes[chunk_start:chunk_stop]
        bad += int(np.count_nonzero(~valid))
        if not np.any(valid):
            continue
        frames = np.asarray(movie[chunk_start:chunk_stop][valid], dtype=np.float32)
        accumulator += frames.sum(axis=0, dtype=np.float64)
        used += int(frames.shape[0])
    if used == 0:
        raise RuntimeError(f"Window {start}:{stop} contains no usable frames")
    return (accumulator / used).astype(np.float32), used, bad


def save_contact_sheet(
    path: Path,
    images: list[np.ndarray],
    rows: list[dict[str, object]],
    channel: int,
) -> None:
    columns = min(5, len(images))
    nrows = math.ceil(len(images) / columns)
    figure, axes = plt.subplots(
        nrows, columns, figsize=(4 * columns, 4 * nrows), constrained_layout=True
    )
    axes_array = np.atleast_1d(axes).ravel()
    for axis in axes_array:
        axis.axis("off")
    for axis, image, row in zip(axes_array, images, rows):
        axis.imshow(display_image(image), cmap="gray", vmin=0, vmax=1)
        axis.set_title(
            f"{row['start_frame']}:{row['stop_frame']}\n"
            f"sharpness={float(row['sharpness']):.4g}"
        )
    figure.suptitle(
        f"Registered channel {channel}: temporal-window means",
        fontsize=14,
    )
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Suite2p run directory, or its suite2p/plane0 directory",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2000,
        help="registered frames per mean image (default: 2000)",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="frames between window starts (default: window size)",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument(
        "--channel",
        choices=("1", "2", "both"),
        default="both",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="also export a final window shorter than --window-size",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.window_size < 1:
        parser.error("--window-size must be positive")
    step = args.step or args.window_size
    if step < 1:
        parser.error("--step must be positive")

    plane_dir = resolve_plane_dir(args.run_dir)
    db = np.load(plane_dir / "db.npy", allow_pickle=True).item()
    ly, lx, nframes = (int(db[key]) for key in ("Ly", "Lx", "nframes"))
    start = args.start_frame
    stop = nframes if args.stop_frame is None else args.stop_frame
    if not 0 <= start < stop <= nframes:
        parser.error(
            f"Require 0 <= start < stop <= {nframes}; received {start}:{stop}"
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else plane_dir.parent / "qc" / "windowed_means"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; choose a new --output-dir: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    reg_outputs_path = plane_dir / "reg_outputs.npy"
    reg_outputs = (
        np.load(reg_outputs_path, allow_pickle=True).item()
        if reg_outputs_path.is_file()
        else {}
    )
    badframes = np.asarray(
        reg_outputs.get("badframes", np.zeros(nframes, dtype=bool)), dtype=bool
    ).ravel()
    if badframes.size != nframes:
        raise ValueError(
            f"badframes has length {badframes.size}; expected {nframes}"
        )

    starts = list(range(start, stop, step))
    windows = [
        (window_start, min(window_start + args.window_size, stop))
        for window_start in starts
        if args.include_partial or window_start + args.window_size <= stop
    ]
    if not windows:
        raise ValueError("The selected interval does not contain one complete window")

    channels = (1, 2) if args.channel == "both" else (int(args.channel),)
    all_rows: list[dict[str, object]] = []
    for channel in channels:
        binary = plane_dir / ("data.bin" if channel == 1 else "data_chan2.bin")
        if not binary.is_file():
            raise FileNotFoundError(f"Registered channel-{channel} binary is missing: {binary}")
        expected_bytes = nframes * ly * lx * np.dtype(np.int16).itemsize
        if binary.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"{binary} has {binary.stat().st_size} bytes; expected {expected_bytes}"
            )
        movie = np.memmap(binary, dtype=np.int16, mode="r", shape=(nframes, ly, lx))
        channel_dir = output_dir / f"channel{channel}"
        channel_dir.mkdir()
        images: list[np.ndarray] = []
        channel_rows: list[dict[str, object]] = []
        for index, (window_start, window_stop) in enumerate(windows):
            image, used, bad = window_mean(
                movie, window_start, window_stop, badframes
            )
            stem = f"window_{index:03d}_frames_{window_start:06d}_{window_stop:06d}"
            tiff_path = channel_dir / f"{stem}.tiff"
            png_path = channel_dir / f"{stem}.png"
            tifffile.imwrite(tiff_path, image, photometric="minisblack")
            plt.imsave(png_path, display_image(image), cmap="gray", vmin=0, vmax=1)
            row = {
                "channel": channel,
                "window": index,
                "start_frame": window_start,
                "stop_frame": window_stop,
                "window_frames": window_stop - window_start,
                "used_frames": used,
                "bad_frames_excluded": bad,
                "sharpness": normalized_sharpness(image),
                "tiff": str(tiff_path),
                "png": str(png_path),
            }
            images.append(image)
            channel_rows.append(row)
            all_rows.append(row)
            print(
                f"channel {channel}, frames {window_start}:{window_stop}, "
                f"used {used}, sharpness {row['sharpness']:.5g}"
            )
        del movie
        save_contact_sheet(
            output_dir / f"channel{channel}_contact_sheet.png",
            images,
            channel_rows,
            channel,
        )

    metrics_path = output_dir / "window_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Windowed means: {output_dir}")
    print(f"Metrics:        {metrics_path}")


if __name__ == "__main__":
    main()

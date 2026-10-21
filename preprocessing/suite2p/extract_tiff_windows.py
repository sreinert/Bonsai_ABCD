#!/usr/bin/env python3
"""Extract start, middle, and end windows from sequential raw TIFF stacks.

Window coordinates are imaging timepoints, not raw TIFF pages. For example,
with two channels and one plane, a 2,000-timepoint window contains 4,000 TIFF
pages. Keeping this distinction here prevents windows from splitting channel
pairs.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile


TIFF_SUFFIXES = {".tif", ".tiff"}


def natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def resolve_inputs(values: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = value.expanduser().resolve()
        if path.is_dir():
            paths.extend(
                candidate.resolve()
                for candidate in path.iterdir()
                if candidate.is_file()
                and not candidate.name.startswith("._")
                and candidate.suffix.lower() in TIFF_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in TIFF_SUFFIXES:
            paths.append(path)
        else:
            raise FileNotFoundError(f"Input is not a TIFF file or directory: {path}")
    unique = sorted(set(paths), key=natural_key)
    if not unique:
        raise FileNotFoundError("No TIFF inputs were found")
    return unique


def page_count(path: Path) -> int:
    with tifffile.TiffFile(path) as tif:
        count = len(tif.pages)
        if count == 1 and tif.pages[0].ndim != 2:
            raise ValueError(
                f"{path} stores a multidimensional array in one TIFF page; "
                "this extractor requires ordinary 2D TIFF pages"
            )
        return count


@dataclass(frozen=True)
class Window:
    label: str
    start_timepoint: int
    stop_timepoint: int
    start_page: int
    stop_page: int
    output_tiff: str


def select_windows(total_timepoints: int, window_size: int) -> list[tuple[str, int, int]]:
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if total_timepoints < window_size:
        raise ValueError(
            f"Session contains {total_timepoints} timepoints, fewer than the "
            f"requested {window_size}-timepoint window"
        )
    starts = {
        "start": 0,
        "middle": (total_timepoints - window_size) // 2,
        "end": total_timepoints - window_size,
    }
    windows: list[tuple[str, int, int]] = []
    used: set[tuple[int, int]] = set()
    for label, start in starts.items():
        interval = (start, start + window_size)
        if interval not in used:
            windows.append((label, *interval))
            used.add(interval)
    return windows


def copy_page_interval(
    inputs: list[Path],
    counts: list[int],
    start_page: int,
    stop_page: int,
    output: Path,
) -> tuple[tuple[int, int], str]:
    expected = stop_page - start_page
    written = 0
    frame_shape: tuple[int, int] | None = None
    dtype: str | None = None
    global_start = 0
    with tifffile.TiffWriter(output, bigtiff=True) as writer:
        for path, count in zip(inputs, counts):
            global_stop = global_start + count
            local_start = max(start_page, global_start) - global_start
            local_stop = min(stop_page, global_stop) - global_start
            if local_start < local_stop:
                with tifffile.TiffFile(path) as tif:
                    for page_index in range(local_start, local_stop):
                        frame = np.asarray(tif.pages[page_index].asarray())
                        if frame.ndim != 2:
                            raise ValueError(
                                f"Expected a 2D page in {path}, found shape {frame.shape}"
                            )
                        shape = (int(frame.shape[0]), int(frame.shape[1]))
                        if frame_shape is None:
                            frame_shape = shape
                            dtype = str(frame.dtype)
                        elif shape != frame_shape or str(frame.dtype) != dtype:
                            raise ValueError(
                                f"Inconsistent TIFF page in {path}: shape={shape}, "
                                f"dtype={frame.dtype}; expected shape={frame_shape}, "
                                f"dtype={dtype}"
                            )
                        writer.write(
                            frame,
                            photometric="minisblack",
                            contiguous=True,
                            metadata=None,
                        )
                        written += 1
            global_start = global_stop
            if global_start >= stop_page:
                break
    if written != expected or frame_shape is None or dtype is None:
        raise RuntimeError(
            f"Wrote {written} pages to {output}; expected {expected}"
        )
    return frame_shape, dtype


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="raw TIFF files, or a directory containing the session TIFFs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--window-size",
        type=int,
        default=2000,
        help="imaging timepoints per output stack (default: 2000)",
    )
    parser.add_argument("--nchannels", type=int, default=2)
    parser.add_argument("--nplanes", type=int, default=1)
    args = parser.parse_args()

    if args.nchannels < 1 or args.nplanes < 1:
        parser.error("--nchannels and --nplanes must be positive")
    pages_per_timepoint = args.nchannels * args.nplanes

    inputs = resolve_inputs(args.inputs)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; choose a new directory: {output_dir}"
        )
    if any(output_dir == path.parent or output_dir in path.parents for path in inputs):
        raise ValueError("Output directory must not contain the raw TIFF inputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = [page_count(path) for path in inputs]
    total_pages = sum(counts)
    if total_pages % pages_per_timepoint:
        raise ValueError(
            f"The {total_pages} raw TIFF pages are not divisible by "
            f"nchannels*nplanes={pages_per_timepoint}; verify channel/plane ordering"
        )
    total_timepoints = total_pages // pages_per_timepoint
    selected = select_windows(total_timepoints, args.window_size)

    records: list[Window] = []
    frame_shape: tuple[int, int] | None = None
    dtype: str | None = None
    for label, start, stop in selected:
        start_page = start * pages_per_timepoint
        stop_page = stop * pages_per_timepoint
        output = output_dir / f"{label}_timepoints_{start:06d}_{stop:06d}.tif"
        current_shape, current_dtype = copy_page_interval(
            inputs, counts, start_page, stop_page, output
        )
        if frame_shape is None:
            frame_shape, dtype = current_shape, current_dtype
        records.append(
            Window(
                label=label,
                start_timepoint=start,
                stop_timepoint=stop,
                start_page=start_page,
                stop_page=stop_page,
                output_tiff=str(output),
            )
        )
        print(
            f"{label:>6}: timepoints {start}:{stop}, raw pages "
            f"{start_page}:{stop_page} -> {output}"
        )

    manifest = {
        "inputs": [str(path) for path in inputs],
        "input_page_counts": counts,
        "nchannels": args.nchannels,
        "nplanes": args.nplanes,
        "pages_per_timepoint": pages_per_timepoint,
        "total_pages": total_pages,
        "total_timepoints": total_timepoints,
        "window_size": args.window_size,
        "frame_shape": list(frame_shape) if frame_shape else None,
        "dtype": dtype,
        "windows": [asdict(record) for record in records],
    }
    manifest_path = output_dir / "windows.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

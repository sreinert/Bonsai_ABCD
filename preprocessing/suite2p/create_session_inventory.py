#!/usr/bin/env python3
"""Create a CSV inventory of imaging sessions and their Suite2p outputs."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_PROJECT_ROOT = Path(
    "/Volumes/mrsic_flogel/public/projects/"
    "AtAp_20260119_SequenceCompression"
)
OUTPUT_FILENAME = "suite2p_session_inventory.csv"
SESSION_PATTERN = re.compile(
    r"^ses-(?P<short_name>.+?)_date-(?P<date>\d{8})(?:T\d{6})?(?:_|$)",
    re.IGNORECASE,
)
REGISTRATION_FILES = ("ops.npy", "data.bin")
TRACE_FILES = ("F.npy", "Fneu.npy", "spks.npy", "iscell.npy")
FIELDNAMES = (
    "mouse_id",
    "session_full_name",
    "session_short_name",
    "date",
    "suite2p_path_created",
    "registration",
    "registration_status",
    "detection",
    "trace_extraction",
    "number_of_rois",
    "number_of_cells",
)


@dataclass(frozen=True)
class Suite2pStatus:
    path_created: bool
    registration: bool
    detection: bool
    trace_extraction: bool
    number_of_rois: int | None
    number_of_cells: int | None


def parse_session_name(name: str) -> tuple[str, str]:
    """Return the short session name and an ISO-formatted date."""
    match = SESSION_PATTERN.match(name)
    if match is None:
        return name.removeprefix("ses-"), ""
    date = match.group("date")
    return match.group("short_name"), f"{date[:4]}-{date[4:6]}-{date[6:]}"


def suite2p_path_candidates(
    raw_session_path: Path, processed_session_path: Path
) -> tuple[Path, ...]:
    """Return supported Suite2p output locations, current layout first."""
    return (
        processed_session_path / "suite2p",
        raw_session_path / "suite2p",
        raw_session_path / "funcimg" / "suite2p",
    )


def find_suite2p_path(
    raw_session_path: Path, processed_session_path: Path
) -> Path | None:
    """Choose the first supported Suite2p directory that exists."""
    return next(
        (
            path
            for path in suite2p_path_candidates(
                raw_session_path, processed_session_path
            )
            if path.is_dir()
        ),
        None,
    )


def all_nonempty_files(directory: Path, names: Iterable[str]) -> bool:
    return all(
        (path := directory / name).is_file() and path.stat().st_size > 0
        for name in names
    )


def array_length(path: Path, *, allow_pickle: bool = False) -> int:
    array = np.load(path, mmap_mode=None if allow_pickle else "r", allow_pickle=allow_pickle)
    if array.ndim == 0:
        raise ValueError(f"Expected an array with a ROI dimension: {path}")
    return int(array.shape[0])


def count_cells(path: Path) -> tuple[int, int]:
    iscell = np.load(path, mmap_mode="r", allow_pickle=False)
    if iscell.ndim == 1:
        labels = iscell
    elif iscell.ndim == 2 and iscell.shape[1] >= 1:
        labels = iscell[:, 0]
    else:
        raise ValueError(f"Unexpected iscell.npy shape {iscell.shape}: {path}")
    return int(iscell.shape[0]), int(np.count_nonzero(labels.astype(bool)))


def inspect_suite2p(
    raw_session_path: Path, processed_session_path: Path
) -> Suite2pStatus:
    suite2p_path = find_suite2p_path(raw_session_path, processed_session_path)
    if suite2p_path is None:
        return Suite2pStatus(False, False, False, False, None, None)

    plane = suite2p_path / "plane0"
    if not plane.is_dir():
        return Suite2pStatus(True, False, False, False, None, None)

    registration = all_nonempty_files(plane, REGISTRATION_FILES)

    number_of_rois: int | None = None
    stat_path = plane / "stat.npy"
    detection = stat_path.is_file() and stat_path.stat().st_size > 0
    if detection:
        try:
            number_of_rois = array_length(stat_path, allow_pickle=True)
        except (OSError, ValueError, EOFError):
            detection = False

    trace_extraction = detection and all_nonempty_files(plane, TRACE_FILES)
    number_of_cells: int | None = None
    if trace_extraction:
        try:
            iscell_rois, number_of_cells = count_cells(plane / "iscell.npy")
            trace_lengths = {
                array_length(plane / filename)
                for filename in ("F.npy", "Fneu.npy", "spks.npy")
            }
            if (
                len(trace_lengths) != 1
                or iscell_rois not in trace_lengths
                or (number_of_rois is not None and iscell_rois != number_of_rois)
            ):
                trace_extraction = False
                number_of_cells = None
        except (OSError, ValueError, EOFError):
            trace_extraction = False
            number_of_cells = None

    return Suite2pStatus(
        True,
        registration,
        detection,
        trace_extraction,
        number_of_rois,
        number_of_cells,
    )


def visible_session_directories(mouse_path: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in mouse_path.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=lambda path: path.name,
    )


def inventory_mouse(
    session_root: Path, suite2p_root: Path, mouse_id: str
) -> list[dict[str, object]]:
    if not mouse_id or mouse_id in {".", ".."} or Path(mouse_id).name != mouse_id:
        raise ValueError(f"Mouse ID must be one directory name, received {mouse_id!r}")
    mouse_path = session_root / mouse_id
    if not mouse_path.is_dir():
        raise NotADirectoryError(f"Mouse directory not found: {mouse_path}")

    rows: list[dict[str, object]] = []
    for session_path in visible_session_directories(mouse_path):
        if not (session_path / "funcimg").is_dir():
            continue
        short_name, date = parse_session_name(session_path.name)
        processed_session_path = suite2p_root / mouse_id / short_name
        status = inspect_suite2p(session_path, processed_session_path)
        registration_status = (
            "succeeded"
            if status.registration
            else "failed"
            if status.path_created
            else "not_started"
        )
        rows.append(
            {
                "mouse_id": mouse_id,
                "session_full_name": session_path.name,
                "session_short_name": short_name,
                "date": date,
                "suite2p_path_created": "yes" if status.path_created else "no",
                "registration": "yes" if status.registration else "no",
                "registration_status": registration_status,
                "detection": "yes" if status.detection else "no",
                "trace_extraction": "yes" if status.trace_extraction else "no",
                "number_of_rois": (
                    "" if status.number_of_rois is None else status.number_of_rois
                ),
                "number_of_cells": (
                    "" if status.number_of_cells is None else status.number_of_cells
                ),
            }
        )
    return rows


def write_inventory(
    path: Path, rows: list[dict[str, object]], *, overwrite: bool
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def mouse_output_path(
    suite2p_root: Path, mouse_id: str, output_dir: Path | None
) -> Path:
    if output_dir is not None:
        return output_dir / f"{mouse_id}_{OUTPUT_FILENAME}"
    return suite2p_root / mouse_id / OUTPUT_FILENAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mouse-ids",
        nargs="+",
        required=True,
        metavar="MOUSE_ID",
        help="Mouse directory names, for example: sub-01 sub-02",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help=f"SequenceCompression project directory (default: {DEFAULT_PROJECT_ROOT})",
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        help="Mouse/session input root (default: PROJECT_ROOT/rawdata/cohort2)",
    )
    parser.add_argument(
        "--suite2p-root",
        type=Path,
        help="Processed mouse/session root (default: PROJECT_ROOT/processed/cohort2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Optional common output directory. By default, each CSV is written "
            "inside PROCESSED_ROOT/MOUSE_ID."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output CSV"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    session_root = (
        args.session_root.expanduser().resolve()
        if args.session_root
        else project_root / "rawdata" / "cohort2"
    )
    suite2p_root = (
        args.suite2p_root.expanduser().resolve()
        if args.suite2p_root
        else project_root / "processed" / "cohort2"
    )
    for label, path in (
        ("Session root", session_root),
        ("Suite2p root", suite2p_root),
    ):
        if not path.is_dir():
            raise NotADirectoryError(f"{label} not found: {path}")

    mouse_ids = list(dict.fromkeys(args.mouse_ids))
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None
    for mouse_id in mouse_ids:
        try:
            rows = inventory_mouse(session_root, suite2p_root, mouse_id)
        except PermissionError as exc:
            raise SystemExit(
                f"ERROR: permission denied while reading {exc.filename}. "
                "On macOS, allow your terminal application to access mounted volumes, "
                "then run this command again."
            ) from None
        output = mouse_output_path(suite2p_root, mouse_id, output_dir)
        write_inventory(output, rows, overwrite=args.overwrite)
        print(f"Wrote {len(rows)} session(s) for {mouse_id} to {output}")
        if not rows:
            print(
                f"WARNING: no visible session folders containing funcimg were found in "
                f"{session_root / mouse_id}"
            )


if __name__ == "__main__":
    main()

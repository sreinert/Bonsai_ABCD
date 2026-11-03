"""Resolve small-stack and full-session TIFF inputs without counting pages."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


TIFF_SUFFIXES = {".tif", ".tiff"}
INPUT_EXPECTATIONS = {
    "small": (2_000, 4_000),
    "full": (81_000, 162_000),
}


@dataclass(frozen=True)
class SessionInput:
    mouse_id: str
    session_id: str
    session_path: Path
    input_path: Path
    tiff_files: tuple[Path, ...]
    input_kind: str
    input_selection: str
    expected_timepoints: int
    expected_raw_pages: int

    def manifest_row(self) -> dict[str, object]:
        return {
            "mouse_id": self.mouse_id,
            "session_id": self.session_id,
            "selection_role": "requested",
            "session_path": str(self.session_path),
            "input_path": str(self.input_path),
            "tiff_files_json": json.dumps([str(path) for path in self.tiff_files]),
            "input_kind": self.input_kind,
            "input_selection": self.input_selection,
            "expected_timepoints": self.expected_timepoints,
            "expected_raw_pages": self.expected_raw_pages,
            "actual_raw_pages": "",
            "frame_count": "",
            "fs": 45.0,
            "tau": 0.4,
            "nplanes": 1,
            "nchannels": 2,
        }


def natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def validate_tiff_header(path: Path) -> None:
    with path.open("rb") as stream:
        header = stream.read(4)
    if header not in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}:
        raise ValueError(f"Unrecognised TIFF header {header!r}: {path}")


def visible_tiffs(directory: Path) -> list[Path]:
    return sorted(
        (
            path.resolve()
            for path in directory.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in TIFF_SUFFIXES
        ),
        key=natural_key,
    )


def discover_session_input(session_dir: Path, input_kind: str) -> SessionInput:
    if input_kind not in INPUT_EXPECTATIONS:
        raise ValueError(f"Unsupported input kind: {input_kind}")
    session = session_dir.expanduser().resolve()
    if not session.is_dir():
        raise NotADirectoryError(f"Session directory does not exist: {session}")
    if session.parent == session:
        raise ValueError(f"Cannot infer a mouse ID from session directory: {session}")

    if input_kind == "small":
        input_path = session / "funcimg" / "others"
        if not input_path.is_dir():
            raise NotADirectoryError(f"Small-stack directory not found: {input_path}")
        available = visible_tiffs(input_path)
        if not available:
            raise FileNotFoundError(f"No direct TIFF files found in {input_path}")
        selected = max(
            available,
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        tiffs = (selected,)
        selection = "newest-direct-tiff"
    else:
        input_path = session / "funcimg"
        if not input_path.is_dir():
            raise NotADirectoryError(f"Full-session directory not found: {input_path}")
        available = visible_tiffs(input_path)
        if not available:
            raise FileNotFoundError(f"No direct TIFF files found in {input_path}")
        cropped = [
            path for path in available if path.stem.lower().endswith("_cropped")
        ]
        tiffs = tuple(cropped or available)
        selection = "cropped" if cropped else "all-direct-tiffs"

    for path in tiffs:
        validate_tiff_header(path)
    expected_timepoints, expected_raw_pages = INPUT_EXPECTATIONS[input_kind]
    return SessionInput(
        mouse_id=session.parent.name,
        session_id=session.name,
        session_path=session,
        input_path=input_path.resolve(),
        tiff_files=tiffs,
        input_kind=input_kind,
        input_selection=selection,
        expected_timepoints=expected_timepoints,
        expected_raw_pages=expected_raw_pages,
    )

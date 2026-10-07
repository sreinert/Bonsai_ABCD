#!/usr/bin/env python3
"""Select first, last, and one reproducible random middle session per mouse."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tifffile

from common import write_csv


TIFF_SUFFIXES = {".tif", ".tiff"}
DATE_PATTERN = re.compile(r"(?:^|[_-])date[-_]?([0-9]{8})(?:[_-]|$)", re.IGNORECASE)
SESSION_PATTERN = re.compile(r"(?:^|[_-])ses(?:sion)?[-_]?([0-9]+)(?:[_-]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class Session:
    mouse_id: str
    session_id: str
    session_path: Path
    input_path: Path
    tiff_files: tuple[Path, ...]
    frame_count: int
    date: datetime | None
    session_number: int | None


def parse_date(name: str) -> datetime | None:
    match = DATE_PATTERN.search(name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d")
    except ValueError:
        return None


def parse_session_number(name: str) -> int | None:
    match = SESSION_PATTERN.search(name)
    return int(match.group(1)) if match else None


def tiff_frame_count(path: Path) -> int:
    """Count TIFF frames from metadata without loading pixel data."""
    with tifffile.TiffFile(path) as tif:
        page_count = len(tif.pages)
        if page_count > 1:
            return page_count
        if not tif.series:
            return page_count
        shape = tuple(int(value) for value in tif.series[0].shape)
        axes = tif.series[0].axes
        if "T" in axes:
            return shape[axes.index("T")]
        if len(shape) > 2:
            count = 1
            for dimension in shape[:-2]:
                count *= dimension
            return count
        return page_count


def stable_mouse_seed(seed: int, mouse_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{mouse_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def ordering_method(sessions: list[Session]) -> str:
    if all(session.date is not None for session in sessions):
        return "date"
    if all(session.session_number is not None for session in sessions):
        return "session_number"
    return "directory_name"


def order_sessions(sessions: list[Session]) -> tuple[list[Session], str]:
    method = ordering_method(sessions)
    if method == "date":
        key = lambda session: (session.date, session.session_id)
    elif method == "session_number":
        key = lambda session: (session.session_number, session.session_id)
    else:
        key = lambda session: session.session_id
    return sorted(sessions, key=key), method


def choose_sessions(sessions: list[Session], seed: int) -> list[tuple[str, Session]]:
    if not sessions:
        return []
    if len(sessions) == 1:
        return [("only", sessions[0])]
    if len(sessions) == 2:
        return [("first", sessions[0]), ("last", sessions[-1])]
    middle = random.Random(seed).choice(sessions[1:-1])
    return [("first", sessions[0]), ("middle", middle), ("last", sessions[-1])]


def discover_mouse(
    mouse_path: Path,
    session_glob: str,
    others_relative: Path,
    expected_frames: int,
) -> tuple[list[Session], list[dict[str, object]]]:
    eligible: list[Session] = []
    audit: list[dict[str, object]] = []
    for session_path in sorted(path for path in mouse_path.glob(session_glob) if path.is_dir()):
        input_path = session_path / others_relative
        base = {
            "mouse_id": mouse_path.name,
            "session_id": session_path.name,
            "session_path": str(session_path.resolve()),
            "input_path": str(input_path.resolve()),
        }
        if not input_path.is_dir():
            audit.append({**base, "eligible": False, "frame_count": "", "reason": "missing input directory"})
            continue
        files = tuple(
            sorted(
                path.resolve()
                for path in input_path.rglob("*")
                if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
            )
        )
        if not files:
            audit.append({**base, "eligible": False, "frame_count": 0, "reason": "no TIFF files"})
            continue
        try:
            frame_count = sum(tiff_frame_count(path) for path in files)
        except Exception as exc:
            audit.append({**base, "eligible": False, "frame_count": "", "reason": f"TIFF metadata error: {exc}"})
            continue
        if frame_count != expected_frames:
            audit.append(
                {
                    **base,
                    "eligible": False,
                    "frame_count": frame_count,
                    "reason": f"expected {expected_frames} frames",
                }
            )
            continue
        session = Session(
            mouse_id=mouse_path.name,
            session_id=session_path.name,
            session_path=session_path.resolve(),
            input_path=input_path.resolve(),
            tiff_files=files,
            frame_count=frame_count,
            date=parse_date(session_path.name),
            session_number=parse_session_number(session_path.name),
        )
        eligible.append(session)
        audit.append({**base, "eligible": True, "frame_count": frame_count, "reason": ""})
    return eligible, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--mouse", action="append", help="Mouse directory name; repeat as needed")
    parser.add_argument("--mouse-glob", default="TAA*")
    parser.add_argument("--session-glob", default="ses-*")
    parser.add_argument("--others-relative", type=Path, default=Path("funcimg/others"))
    parser.add_argument("--expected-frames", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--fs", type=float, default=45.0)
    parser.add_argument("--tau", type=float, default=0.4)
    parser.add_argument("--nplanes", type=int, default=1)
    parser.add_argument("--nchannels", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root does not exist: {data_root}")
    if args.mouse:
        mouse_paths = [data_root / mouse_id for mouse_id in args.mouse]
        missing = [path for path in mouse_paths if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Mouse directories not found: {missing}")
    else:
        mouse_paths = sorted(path for path in data_root.glob(args.mouse_glob) if path.is_dir())
    if not mouse_paths:
        raise RuntimeError("No mouse directories matched")

    selected_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for mouse_path in mouse_paths:
        eligible, audit = discover_mouse(
            mouse_path,
            args.session_glob,
            args.others_relative,
            args.expected_frames,
        )
        if not audit:
            audit.append(
                {
                    "mouse_id": mouse_path.name,
                    "session_id": "",
                    "session_path": "",
                    "input_path": "",
                    "eligible": False,
                    "frame_count": "",
                    "reason": f"no session directories matched {args.session_glob!r}",
                }
            )
        audit_rows.extend(audit)
        ordered, method = order_sessions(eligible)
        choices = choose_sessions(ordered, stable_mouse_seed(args.seed, mouse_path.name))
        completeness = "complete" if len(ordered) >= 3 else f"only_{len(ordered)}_eligible"
        for role, session in choices:
            selected_rows.append(
                {
                    "mouse_id": session.mouse_id,
                    "session_id": session.session_id,
                    "selection_role": role,
                    "selection_seed": args.seed,
                    "selection_status": completeness,
                    "ordering_method": method,
                    "session_path": str(session.session_path),
                    "input_path": str(session.input_path),
                    "frame_count": session.frame_count,
                    "tiff_files_json": json.dumps([str(path) for path in session.tiff_files]),
                    "fs": args.fs,
                    "tau": args.tau,
                    "nplanes": args.nplanes,
                    "nchannels": args.nchannels,
                }
            )
    if not selected_rows:
        raise RuntimeError("No eligible 2,000-frame sessions were found")

    audit_path = args.audit or args.output.with_name(args.output.stem + "_audit.csv")
    write_csv(args.output, selected_rows, overwrite=args.overwrite)
    if audit_rows:
        write_csv(audit_path, audit_rows, overwrite=args.overwrite)
    print(f"Selected {len(selected_rows)} sessions across {len(mouse_paths)} mice")
    print(f"Manifest: {args.output}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()

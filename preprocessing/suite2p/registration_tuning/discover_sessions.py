#!/usr/bin/env python3
"""Select first, last, and one reproducible random middle session per mouse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import tifffile

from common import write_csv


TIFF_SUFFIXES = {".tif", ".tiff"}
DEFAULT_PROJECT_ROOTS = (
    Path("/ceph/mrsic_flogel/public/projects"),
    Path("/Volumes/mrsic_flogel/public/projects"),
)
DATE_PATTERN = re.compile(
    r"(?:^|[_-])date[-_]?([0-9]{8})(?:T([0-9]{6}))?(?=[_-]|$)",
    re.IGNORECASE,
)
SESSION_PATTERN = re.compile(r"(?:^|[_-])ses(?:sion)?[-_]?([0-9]+)(?:[_-]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class Session:
    mouse_id: str
    session_id: str
    session_path: Path
    input_path: Path
    tiff_files: tuple[Path, ...]
    frame_count: int | None
    date: datetime | None
    session_number: int | None


def resolve_data_root(
    value: Path,
    project_roots: tuple[Path, ...] = DEFAULT_PROJECT_ROOTS,
) -> Path:
    """Resolve an absolute path or a path relative to the mounted projects root."""
    if value.is_absolute():
        return value.resolve()

    override = os.environ.get("MRSIC_PROJECTS_ROOT")
    roots = (Path(override),) if override else project_roots
    attempted: list[Path] = []
    for root in roots:
        candidate = (root / value).resolve()
        attempted.append(candidate)
        if candidate.is_dir():
            return candidate
    locations = ", ".join(str(path) for path in attempted)
    raise NotADirectoryError(
        f"Could not resolve relative data root {value!s}; checked: {locations}"
    )


def parse_date(name: str) -> datetime | None:
    match = DATE_PATTERN.search(name)
    if not match:
        return None
    try:
        if match.group(2):
            return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
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


def validate_tiff_header(path: Path) -> None:
    """Quickly verify that a file is readable and starts with a TIFF header."""
    with path.open("rb") as stream:
        header = stream.read(4)
    if header not in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}:
        raise ValueError(f"unrecognised TIFF header {header!r}")


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
    expected_frames: int | None,
    count_frames: bool = False,
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
            "tiff_file_count": 0,
            "selected_tiff": "",
            "selected_tiff_mtime_utc": "",
            "frame_count_status": "not_checked",
        }
        if not input_path.is_dir():
            audit.append({**base, "eligible": False, "frame_count": "", "reason": "missing input directory"})
            continue
        files = tuple(
            sorted(
                path.resolve()
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
            )
        )
        if not files:
            audit.append({**base, "eligible": False, "frame_count": 0, "reason": "no TIFF files"})
            continue
        selected_file = max(
            files,
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        selected_mtime = datetime.fromtimestamp(
            selected_file.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
        file_details = {
            **base,
            "tiff_file_count": len(files),
            "selected_tiff": str(selected_file),
            "selected_tiff_mtime_utc": selected_mtime,
        }
        try:
            validate_tiff_header(selected_file)
        except Exception as exc:
            audit.append(
                {
                    **file_details,
                    "eligible": False,
                    "frame_count": "",
                    "frame_count_status": "not_checked",
                    "reason": f"TIFF read/header error: {exc}",
                }
            )
            continue

        frame_count: int | None = None
        frame_count_status = "not_counted"
        count_error = ""
        if count_frames or expected_frames is not None:
            try:
                frame_count = tiff_frame_count(selected_file)
                frame_count_status = "counted"
            except Exception as exc:
                frame_count_status = "count_failed"
                count_error = f"frame count failed: {exc}"
                if expected_frames is not None:
                    audit.append(
                        {
                            **file_details,
                            "eligible": False,
                            "frame_count": "",
                            "frame_count_status": frame_count_status,
                            "reason": count_error,
                        }
                    )
                    continue
        if expected_frames is not None and frame_count != expected_frames:
            audit.append(
                {
                    **file_details,
                    "eligible": False,
                    "frame_count": frame_count,
                    "frame_count_status": frame_count_status,
                    "reason": f"expected {expected_frames} frames",
                }
            )
            continue
        session = Session(
            mouse_id=mouse_path.name,
            session_id=session_path.name,
            session_path=session_path.resolve(),
            input_path=input_path.resolve(),
            tiff_files=(selected_file,),
            frame_count=frame_count,
            date=parse_date(session_path.name),
            session_number=parse_session_number(session_path.name),
        )
        eligible.append(session)
        reasons = []
        if len(files) > 1:
            reasons.append(f"selected newest of {len(files)} TIFF files")
        if count_error:
            reasons.append(count_error)
        audit.append(
            {
                **file_details,
                "eligible": True,
                "frame_count": frame_count if frame_count is not None else "",
                "frame_count_status": frame_count_status,
                "reason": "; ".join(reasons),
            }
        )
    return eligible, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Absolute path, or path relative to the MRSIC projects mount. "
            "Relative paths auto-detect /ceph or /Volumes."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--mouse", action="append", help="Mouse directory name; repeat as needed")
    parser.add_argument("--mouse-glob", default="TAA*")
    parser.add_argument("--session-glob", default="ses-*")
    parser.add_argument("--others-relative", type=Path, default=Path("funcimg/others"))
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=None,
        help="Optional exact frame-count filter; omitted by default",
    )
    parser.add_argument(
        "--count-frames",
        action="store_true",
        help=(
            "Count TIFF pages for reporting. This can be slow for large TIFFs on "
            "network storage and is not needed for discovery."
        ),
    )
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--fs", type=float, default=45.0)
    parser.add_argument("--tau", type=float, default=0.4)
    parser.add_argument("--nplanes", type=int, default=1)
    parser.add_argument(
        "--nchannels",
        type=int,
        default=2,
        choices=(2,),
        help="Number of interleaved imaging channels (fixed at 2 for this project)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
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
            args.count_frames,
        )
        if not audit:
            audit.append(
                {
                    "mouse_id": mouse_path.name,
                    "session_id": "",
                    "session_path": "",
                    "input_path": "",
                    "tiff_file_count": 0,
                    "selected_tiff": "",
                    "selected_tiff_mtime_utc": "",
                    "eligible": False,
                    "frame_count": "",
                    "frame_count_status": "not_checked",
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
                    "frame_count": session.frame_count if session.frame_count is not None else "",
                    "tiff_files_json": json.dumps([str(path) for path in session.tiff_files]),
                    "fs": args.fs,
                    "tau": args.tau,
                    "nplanes": args.nplanes,
                    "nchannels": args.nchannels,
                }
            )
    audit_path = args.audit or args.output.with_name(args.output.stem + "_audit.csv")
    if audit_rows:
        write_csv(audit_path, audit_rows, overwrite=args.overwrite)
    if not selected_rows:
        raise RuntimeError(
            "No eligible sessions containing a readable TIFF were found. "
            f"See the audit report: {audit_path}"
        )

    write_csv(args.output, selected_rows, overwrite=args.overwrite)
    print(f"Selected {len(selected_rows)} sessions across {len(mouse_paths)} mice")
    print(f"Manifest: {args.output}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()

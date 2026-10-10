#!/usr/bin/env python3
"""Combine restarted Bonsai behaviour recordings into one Aeon-readable session.

The source sessions are copied unchanged into ``original_sessions``.  Only the
behaviour streams consumed by ``session_functions/io.py`` are merged into the
top-level ``behav`` directory; camera and functional-imaging data remain in the
archived originals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SESSION_NAME = re.compile(
    r"^(?P<prefix>.+)_date-(?P<timestamp>\d{8}T\d{6})$"
)
CHUNK_STAMP = "1904-01-01T00-00-00"
JOIN_GAP_SECONDS = 1e-6

# These are exactly the raw streams loaded by session_functions/io.py.
STREAM_PREFIXES = {
    "analog-data": "analog-data",
    "current-position": "current-position",
    "current-landmark": "current-landmark",
    "experiment-events": "experiment-events",
    "licks": "licks",
    "reward": "reward",
    "treadmill-speed": "treadmill-speed",
}
REQUIRED_STREAMS = set(STREAM_PREFIXES) - {"current-landmark"}


class ConcatenationError(RuntimeError):
    """Raised when source sessions cannot be combined safely."""


@dataclass(frozen=True)
class SourceSession:
    path: Path
    prefix: str
    acquired_at: datetime
    time_min: float
    time_max: float
    position_first: float
    position_last: float
    buffer_first: float
    buffer_last: float
    schemas: dict[str, tuple[str, ...]]
    row_counts: dict[str, int]


@dataclass(frozen=True)
class SessionOffsets:
    seconds: float
    position: float
    buffer: float


def _finite_float(value: str, *, description: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConcatenationError(f"Invalid {description}: {value!r}") from exc
    if not math.isfinite(number):
        raise ConcatenationError(f"Non-finite {description}: {value!r}")
    return number


def _csv_files(session: Path, stream: str) -> list[Path]:
    return sorted((session / "behav" / stream).glob("*.csv"))


def _binary_files(session: Path) -> list[Path]:
    return sorted((session / "behav" / "analog-data").glob("*.bin"))


def _iter_dict_rows(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ConcatenationError(f"CSV has no header: {path}")
            for row in reader:
                yield path, reader.fieldnames, row


def _scan_column(paths: list[Path], column: str) -> tuple[float, float]:
    first: float | None = None
    last: float | None = None
    for path, fieldnames, row in _iter_dict_rows(paths):
        if column not in fieldnames:
            raise ConcatenationError(f"Missing column {column!r} in {path}")
        value = _finite_float(row[column], description=f"{column} in {path}")
        if first is None:
            first = value
        last = value
    if first is None or last is None:
        raise ConcatenationError(f"No data rows found in {paths[0]}")
    return first, last


def _inspect_session(path: Path) -> SourceSession:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ConcatenationError(f"Session directory does not exist: {path}")
    match = SESSION_NAME.fullmatch(path.name)
    if match is None:
        raise ConcatenationError(
            f"Session name must end in _date-YYYYMMDDTHHMMSS: {path.name}"
        )
    behav = path / "behav"
    if not behav.is_dir():
        raise ConcatenationError(f"Missing behav directory: {behav}")

    schemas: dict[str, tuple[str, ...]] = {}
    row_counts: dict[str, int] = {}
    time_min: float | None = None
    time_max: float | None = None

    for stream in STREAM_PREFIXES:
        files = _csv_files(path, stream)
        if not files:
            if stream in REQUIRED_STREAMS:
                raise ConcatenationError(f"Missing {stream} CSV data in {path}")
            continue

        schema: tuple[str, ...] | None = None
        count = 0
        for csv_path, fieldnames, row in _iter_dict_rows(files):
            current_schema = tuple(fieldnames)
            if schema is None:
                schema = current_schema
            elif current_schema != schema:
                raise ConcatenationError(
                    f"CSV schema changes within {stream} in {path}: {csv_path}"
                )
            if "Seconds" not in fieldnames:
                raise ConcatenationError(f"Missing 'Seconds' column in {csv_path}")
            timestamp = _finite_float(
                row["Seconds"], description=f"Seconds in {csv_path}"
            )
            time_min = timestamp if time_min is None else min(time_min, timestamp)
            time_max = timestamp if time_max is None else max(time_max, timestamp)
            count += 1
        if schema is None or count == 0:
            raise ConcatenationError(f"No data rows found for {stream} in {path}")
        schemas[stream] = schema
        row_counts[stream] = count

    binaries = _binary_files(path)
    if not binaries:
        raise ConcatenationError(f"Missing analog-data binary file in {path}")
    if time_min is None or time_max is None:
        raise ConcatenationError(f"No timestamped behaviour rows found in {path}")

    position_first, position_last = _scan_column(
        _csv_files(path, "current-position"), "Value.Length"
    )
    analog_schema = schemas["analog-data"]
    if len(analog_schema) < 2:
        raise ConcatenationError(f"Analog CSV needs two columns in {path}")
    buffer_first, buffer_last = _scan_column(
        _csv_files(path, "analog-data"), analog_schema[1]
    )

    return SourceSession(
        path=path,
        prefix=match.group("prefix"),
        acquired_at=datetime.strptime(match.group("timestamp"), "%Y%m%dT%H%M%S"),
        time_min=time_min,
        time_max=time_max,
        position_first=position_first,
        position_last=position_last,
        buffer_first=buffer_first,
        buffer_last=buffer_last,
        schemas=schemas,
        row_counts=row_counts,
    )


def _settings_file(session: Path, directory: str) -> Path:
    folder = session / "behav" / directory
    candidates = sorted(folder.glob("*.json")) or sorted(folder.glob("*.csv"))
    if len(candidates) != 1:
        raise ConcatenationError(
            f"Expected exactly one JSON-compatible settings file in {folder}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _read_settings(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConcatenationError(f"Cannot read settings file {path}: {exc}") from exc
    return document.get("value") if isinstance(document, dict) else document


def _first_difference(left: Any, right: Any, location: str = "value") -> str | None:
    if type(left) is not type(right):
        return f"{location}: {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            missing_left = sorted(right.keys() - left.keys())
            missing_right = sorted(left.keys() - right.keys())
            return (
                f"{location}: differing keys "
                f"(only first={missing_right}, only second={missing_left})"
            )
        for key in left:
            difference = _first_difference(left[key], right[key], f"{location}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{location}: list lengths {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_difference(
                left_item, right_item, f"{location}[{index}]"
            )
            if difference:
                return difference
        return None
    if left != right:
        return f"{location}: {left!r} != {right!r}"
    return None


def _validate_compatibility(sessions: list[SourceSession]) -> None:
    if len(sessions) < 2:
        raise ConcatenationError("Provide at least two session paths")
    if len({session.path for session in sessions}) != len(sessions):
        raise ConcatenationError("The same session path was supplied more than once")
    if len({session.prefix for session in sessions}) != 1:
        names = ", ".join(session.path.name for session in sessions)
        raise ConcatenationError(f"Session IDs do not match: {names}")

    reference = sessions[0]
    for session in sessions[1:]:
        if session.schemas != reference.schemas:
            raise ConcatenationError(
                f"Behaviour stream schemas differ between {reference.path.name} "
                f"and {session.path.name}"
            )
        for directory in ("session-settings", "rig-settings"):
            first_file = _settings_file(reference.path, directory)
            next_file = _settings_file(session.path, directory)
            difference = _first_difference(
                _read_settings(first_file), _read_settings(next_file)
            )
            if difference:
                raise ConcatenationError(
                    f"{directory} differ between {reference.path.name} and "
                    f"{session.path.name}: {difference}"
                )


def _calculate_offsets(sessions: list[SourceSession]) -> list[SessionOffsets]:
    offsets: list[SessionOffsets] = []
    previous_time_end: float | None = None
    previous_position_end: float | None = None
    previous_buffer_end: float | None = None

    for session in sessions:
        seconds_offset = 0.0
        if previous_time_end is not None:
            seconds_offset = previous_time_end + JOIN_GAP_SECONDS - session.time_min

        position_offset = 0.0
        if (
            previous_position_end is not None
            and session.position_first < previous_position_end
        ):
            position_offset = previous_position_end - session.position_first

        buffer_offset = 0.0
        if previous_buffer_end is not None and session.buffer_first <= previous_buffer_end:
            buffer_offset = previous_buffer_end + 1 - session.buffer_first

        offsets.append(
            SessionOffsets(
                seconds=seconds_offset,
                position=position_offset,
                buffer=buffer_offset,
            )
        )
        previous_time_end = session.time_max + seconds_offset
        previous_position_end = session.position_last + position_offset
        previous_buffer_end = session.buffer_last + buffer_offset

    return offsets


def _format_number(value: float) -> str:
    return format(value, ".17g")


def _merge_csv_stream(
    destination: Path,
    stream: str,
    sessions: list[SourceSession],
    offsets: list[SessionOffsets],
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    schema = sessions[0].schemas[stream]
    rows_written = 0
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(schema)
        for session, session_offsets in zip(sessions, offsets):
            for csv_path in _csv_files(session.path, stream):
                with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
                    reader = csv.reader(handle)
                    try:
                        fieldnames = tuple(next(reader))
                    except StopIteration as exc:
                        raise ConcatenationError(f"CSV has no header: {csv_path}") from exc
                    if fieldnames != schema:
                        raise ConcatenationError(f"Unexpected schema in {csv_path}")

                    seconds_index = schema.index("Seconds")
                    value_index: int | None = None
                    value_column: str | None = None
                    value_offset = 0.0
                    if stream == "analog-data":
                        value_index = 1
                        value_column = schema[value_index]
                        value_offset = session_offsets.buffer
                    elif stream == "current-position":
                        value_column = "Value.Length"
                        value_index = schema.index(value_column)
                        value_offset = session_offsets.position
                    elif stream == "current-landmark":
                        value_column = "Value.Value.Position"
                        if value_column not in schema:
                            raise ConcatenationError(
                                f"Missing {value_column!r} in {csv_path}"
                            )
                        value_index = schema.index(value_column)
                        value_offset = session_offsets.position

                    for row in reader:
                        if len(row) != len(schema):
                            raise ConcatenationError(
                                f"Expected {len(schema)} fields in {csv_path}, "
                                f"found {len(row)}"
                            )
                        row[seconds_index] = _format_number(
                            _finite_float(
                                row[seconds_index],
                                description=f"Seconds in {csv_path}",
                            )
                            + session_offsets.seconds
                        )
                        if value_index is not None and value_column is not None:
                            row[value_index] = _format_number(
                                _finite_float(
                                    row[value_index],
                                    description=f"{value_column} in {csv_path}",
                                )
                                + value_offset
                            )
                        writer.writerow(row)
                        rows_written += 1
    return rows_written


def _merge_analog_binary(destination: Path, sessions: list[SourceSession]) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    with destination.open("wb") as output:
        for session in sessions:
            for source in _binary_files(session.path):
                with source.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output, length=1024 * 1024)
                bytes_written += source.stat().st_size
    return bytes_written


def _copy_settings(destination_behav: Path, latest: SourceSession) -> None:
    for directory in ("session-settings", "rig-settings"):
        source = _settings_file(latest.path, directory)
        destination = destination_behav / directory / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _validate_with_existing_loaders(
    output_session: Path, *, expected_analog_bytes: int
) -> dict[str, int]:
    # Local import keeps --help usable in minimal environments.
    from analysis.sequence_compression.session_functions.io import (
        load_analog_data,
        load_data,
        load_settings,
    )

    session_settings, rig_settings = load_settings(output_session)
    dataframe = load_data(output_session)
    analog = load_analog_data(output_session, rig_settings)
    if dataframe.empty:
        raise ConcatenationError("Existing load_data() returned no rows")
    if analog.empty:
        raise ConcatenationError("Existing load_analog_data() returned no rows")
    if not isinstance(session_settings, dict):
        raise ConcatenationError("Existing load_settings() returned invalid settings")
    channel_count = len(rig_settings["analogInputChannels"])
    expected_analog_samples = expected_analog_bytes // (8 * channel_count)
    if len(analog) != expected_analog_samples:
        raise ConcatenationError(
            "Existing load_analog_data() returned "
            f"{len(analog)} samples; expected {expected_analog_samples}"
        )
    return {"behaviour_rows": len(dataframe), "analog_samples": len(analog)}


def concatenate_sessions(
    session_paths: Iterable[Path | str],
    *,
    output_root: Path | str | None = None,
    creation_time: datetime | None = None,
    overwrite: bool = False,
    validate_loaders: bool = True,
) -> Path:
    """Concatenate two or more restarted sessions and return the output path."""
    sessions = sorted(
        (_inspect_session(Path(path)) for path in session_paths),
        key=lambda session: session.acquired_at,
    )
    _validate_compatibility(sessions)
    offsets = _calculate_offsets(sessions)

    latest = sessions[-1]
    created = creation_time or datetime.now().astimezone()
    output_name = f"{latest.prefix}_date-{created.strftime('%Y%m%dT%H%M%S')}"
    root = Path(output_root).expanduser().resolve() if output_root else latest.path.parent
    root.mkdir(parents=True, exist_ok=True)
    output_session = root / output_name

    for session in sessions:
        if output_session == session.path or output_session in session.path.parents:
            raise ConcatenationError(
                f"Output path would replace or contain a source session: {output_session}"
            )
    if output_session.exists() and not overwrite:
        raise ConcatenationError(
            f"Output already exists: {output_session} (use --overwrite to replace it)"
        )

    staging = Path(tempfile.mkdtemp(prefix=f".{output_name}.incomplete-", dir=root))
    try:
        originals = staging / "original_sessions"
        originals.mkdir()
        for session in sessions:
            shutil.copytree(session.path, originals / session.path.name)

        destination_behav = staging / "behav"
        merged_counts: dict[str, int] = {}
        for stream, prefix in STREAM_PREFIXES.items():
            if stream not in sessions[0].schemas:
                continue
            destination = destination_behav / stream / f"{prefix}_{CHUNK_STAMP}.csv"
            merged_counts[stream] = _merge_csv_stream(
                destination, stream, sessions, offsets
            )

        analog_bytes = _merge_analog_binary(
            destination_behav / "analog-data" / f"analog-data_{CHUNK_STAMP}.bin",
            sessions,
        )
        _copy_settings(destination_behav, latest)

        validation: dict[str, int] | None = None
        if validate_loaders:
            validation = _validate_with_existing_loaders(
                staging, expected_analog_bytes=analog_bytes
            )

        manifest = {
            "created_at": created.isoformat(),
            "output_name": output_name,
            "sources": [
                {
                    "path": str(session.path),
                    "acquired_at": session.acquired_at.isoformat(),
                    "rows": session.row_counts,
                    "offsets": {
                        "seconds": offset.seconds,
                        "position": offset.position,
                        "buffer": offset.buffer,
                    },
                }
                for session, offset in zip(sessions, offsets)
            ],
            "merged_rows": merged_counts,
            "analog_bytes": analog_bytes,
            "existing_loader_validation": validation,
        }
        with (staging / "concatenation_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")

        if output_session.exists():
            shutil.rmtree(output_session)
        staging.rename(output_session)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return output_session


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate restarted Bonsai behaviour sessions into one session "
            "readable by analysis/sequence_compression/session_functions."
        )
    )
    parser.add_argument("sessions", nargs="+", type=Path, help="source session paths")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="parent directory for the output (default: parent of latest session)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output with the same generated name",
    )
    parser.add_argument(
        "--skip-loader-validation",
        action="store_true",
        help="skip the final read-back through the existing session loaders",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        output = concatenate_sessions(
            args.sessions,
            output_root=args.output_root,
            overwrite=args.overwrite,
            validate_loaders=not args.skip_loader_validation,
        )
    except ConcatenationError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

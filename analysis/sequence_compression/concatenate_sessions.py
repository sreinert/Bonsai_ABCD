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
DEFAULT_LANDMARK_GAP = 9.0

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


@dataclass
class SessionSlice:
    """Half-open timestamp interval retained from one source session."""

    start_seconds: float | None = None
    end_seconds: float | None = None


@dataclass(frozen=True)
class LogicalLandmark:
    seconds: float
    position: float
    size: float

    @property
    def entry(self) -> float:
        return self.position - self.size / 2

    @property
    def exit(self) -> float:
        return self.position + self.size / 2


@dataclass(frozen=True)
class BoundarySplice:
    source_index: int
    next_source_index: int
    landmarks_per_lap: int
    complete_laps_retained: int
    discarded_logical_landmarks: int
    last_complete_landmark_exit: float
    target_next_landmark_entry: float
    next_source_first_landmark_entry: float
    preserved_gap: float
    gap_source: str
    source_end_seconds: float | None
    next_source_start_seconds: float


@dataclass(frozen=True)
class RetainedStats:
    time_min: float
    time_max: float
    position_first: float
    position_last: float
    buffer_first: float
    buffer_last: float
    row_counts: dict[str, int]


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


def _is_retained(seconds: float, session_slice: SessionSlice) -> bool:
    if session_slice.start_seconds is not None and seconds < session_slice.start_seconds:
        return False
    if session_slice.end_seconds is not None and seconds >= session_slice.end_seconds:
        return False
    return True


def _trial_settings(session: SourceSession) -> dict[str, Any]:
    settings = _read_settings(_settings_file(session.path, "session-settings"))
    if not isinstance(settings, dict) or "trial" not in settings:
        raise ConcatenationError(f"No trial settings found in {session.path}")
    trial = settings["trial"]
    if isinstance(trial, list):
        if not trial or not isinstance(trial[0], dict) or "trial" not in trial[0]:
            raise ConcatenationError(f"Unsupported trial settings in {session.path}")
        trial = trial[0]["trial"]
    if not isinstance(trial, dict):
        raise ConcatenationError(f"Unsupported trial settings in {session.path}")
    return trial


def _logical_landmarks(
    session: SourceSession, session_slice: SessionSlice | None = None
) -> list[LogicalLandmark]:
    if "current-landmark" not in session.schemas:
        raise ConcatenationError(
            f"Cannot splice laps without current-landmark data in {session.path}"
        )
    schema = session.schemas["current-landmark"]
    required = {
        "Seconds",
        "Value.Value.Landmark.Size",
        "Value.Value.Position",
        "Value.Value.Visited",
    }
    missing = required - set(schema)
    if missing:
        raise ConcatenationError(
            f"Missing current-landmark columns in {session.path}: {sorted(missing)}"
        )
    seconds_index = schema.index("Seconds")
    size_index = schema.index("Value.Value.Landmark.Size")
    position_index = schema.index("Value.Value.Position")
    visited_index = schema.index("Value.Value.Visited")
    seen_positions: set[float] = set()
    landmarks: list[LogicalLandmark] = []

    for path in _csv_files(session.path, "current-landmark"):
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            try:
                header = tuple(next(reader))
            except StopIteration as exc:
                raise ConcatenationError(f"CSV has no header: {path}") from exc
            if header != schema:
                raise ConcatenationError(f"Unexpected schema in {path}")
            for row in reader:
                if len(row) != len(schema):
                    raise ConcatenationError(
                        f"Expected {len(schema)} fields in {path}, found {len(row)}"
                    )
                if row[visited_index].strip().lower() != "false":
                    continue
                seconds = _finite_float(
                    row[seconds_index], description=f"Seconds in {path}"
                )
                if session_slice is not None and not _is_retained(seconds, session_slice):
                    continue
                position = _finite_float(
                    row[position_index], description=f"landmark position in {path}"
                )
                if position in seen_positions:
                    continue
                seen_positions.add(position)
                landmarks.append(
                    LogicalLandmark(
                        seconds=seconds,
                        position=position,
                        size=_finite_float(
                            row[size_index], description=f"landmark size in {path}"
                        ),
                    )
                )
    if not landmarks:
        raise ConcatenationError(f"No logical landmarks found in {session.path}")
    return landmarks


def _position_samples(session: SourceSession) -> list[tuple[float, float]]:
    schema = session.schemas["current-position"]
    seconds_index = schema.index("Seconds")
    position_index = schema.index("Value.Length")
    samples: list[tuple[float, float]] = []
    for path in _csv_files(session.path, "current-position"):
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            try:
                header = tuple(next(reader))
            except StopIteration as exc:
                raise ConcatenationError(f"CSV has no header: {path}") from exc
            if header != schema:
                raise ConcatenationError(f"Unexpected schema in {path}")
            for row in reader:
                samples.append(
                    (
                        _finite_float(row[seconds_index], description=f"Seconds in {path}"),
                        _finite_float(
                            row[position_index], description=f"Value.Length in {path}"
                        ),
                    )
                )
    return samples


def _first_position_time(
    samples: list[tuple[float, float]],
    position: float,
    *,
    not_before: float | None = None,
) -> float | None:
    for seconds, value in samples:
        if not_before is not None and seconds < not_before:
            continue
        if value >= position:
            return seconds
    return None


def _configured_gap_or_fallback(
    trial: dict[str, Any], fallback_gap: float
) -> tuple[float, str]:
    offsets = trial.get("offsets", [])
    try:
        unique_offsets = sorted({float(value) for value in offsets})
    except (TypeError, ValueError) as exc:
        raise ConcatenationError(f"Invalid trial offsets: {offsets!r}") from exc
    if len(unique_offsets) == 1 and math.isfinite(unique_offsets[0]):
        return unique_offsets[0], "settings"
    return fallback_gap, "fallback"


def _plan_session_slices(
    sessions: list[SourceSession], fallback_gap: float
) -> tuple[list[SessionSlice], list[BoundarySplice]]:
    if not math.isfinite(fallback_gap) or fallback_gap < 0:
        raise ConcatenationError("Fallback landmark gap must be finite and non-negative")

    slices = [SessionSlice() for _ in sessions]
    boundaries: list[BoundarySplice] = []

    for index in range(len(sessions) - 1):
        session = sessions[index]
        next_session = sessions[index + 1]
        trial = _trial_settings(session)
        configured_landmarks = trial.get("landmarks")
        if not isinstance(configured_landmarks, list) or not configured_landmarks:
            raise ConcatenationError(f"No configured landmarks found in {session.path}")
        landmarks_per_lap = len(configured_landmarks)

        landmarks = _logical_landmarks(session, slices[index])
        source_position_samples = _position_samples(session)
        retained_positions = [
            position
            for seconds, position in source_position_samples
            if _is_retained(seconds, slices[index])
        ]
        if not retained_positions:
            raise ConcatenationError(f"No retained position data in {session.path}")
        final_reached_position = retained_positions[-1]
        complete_laps = len(landmarks) // landmarks_per_lap
        while (
            complete_laps > 0
            and final_reached_position
            < landmarks[complete_laps * landmarks_per_lap - 1].exit
        ):
            complete_laps -= 1
        if complete_laps == 0:
            raise ConcatenationError(
                f"No complete lap found before the restart in {session.path}"
            )
        complete_landmark_count = complete_laps * landmarks_per_lap
        last_complete = landmarks[complete_landmark_count - 1]
        incomplete = landmarks[complete_landmark_count:]

        if incomplete:
            target_entry = incomplete[0].entry
            gap = target_entry - last_complete.exit
            gap_source = "recorded"
        else:
            gap, gap_source = _configured_gap_or_fallback(trial, fallback_gap)
            target_entry = last_complete.exit + gap

        if gap < 0:
            raise ConcatenationError(
                f"Negative inter-landmark gap ({gap}) at the end of {session.path}"
            )

        source_end = _first_position_time(
            source_position_samples,
            target_entry,
            not_before=last_complete.seconds,
        )
        if source_end is not None:
            slices[index].end_seconds = source_end

        next_landmarks = _logical_landmarks(next_session)
        next_entry = next_landmarks[0].entry
        next_start = _first_position_time(
            _position_samples(next_session),
            next_entry,
        )
        if next_start is None:
            raise ConcatenationError(
                f"The first landmark entry was not reached in {next_session.path}"
            )
        slices[index + 1].start_seconds = next_start

        boundaries.append(
            BoundarySplice(
                source_index=index,
                next_source_index=index + 1,
                landmarks_per_lap=landmarks_per_lap,
                complete_laps_retained=complete_laps,
                discarded_logical_landmarks=len(incomplete),
                last_complete_landmark_exit=last_complete.exit,
                target_next_landmark_entry=target_entry,
                next_source_first_landmark_entry=next_entry,
                preserved_gap=gap,
                gap_source=gap_source,
                source_end_seconds=source_end,
                next_source_start_seconds=next_start,
            )
        )

    return slices, boundaries


def _retained_stats(
    session: SourceSession, session_slice: SessionSlice
) -> RetainedStats:
    time_min: float | None = None
    time_max: float | None = None
    position_first: float | None = None
    position_last: float | None = None
    buffer_first: float | None = None
    buffer_last: float | None = None
    row_counts: dict[str, int] = {}

    for stream in session.schemas:
        count = 0
        for path, fieldnames, row in _iter_dict_rows(_csv_files(session.path, stream)):
            seconds = _finite_float(row["Seconds"], description=f"Seconds in {path}")
            if not _is_retained(seconds, session_slice):
                continue
            time_min = seconds if time_min is None else min(time_min, seconds)
            time_max = seconds if time_max is None else max(time_max, seconds)
            if stream == "current-position":
                position = _finite_float(
                    row["Value.Length"], description=f"Value.Length in {path}"
                )
                if position_first is None:
                    position_first = position
                position_last = position
            elif stream == "analog-data":
                buffer_column = fieldnames[1]
                buffer = _finite_float(
                    row[buffer_column], description=f"{buffer_column} in {path}"
                )
                if buffer_first is None:
                    buffer_first = buffer
                buffer_last = buffer
            count += 1
        row_counts[stream] = count

    values = (
        time_min,
        time_max,
        position_first,
        position_last,
        buffer_first,
        buffer_last,
    )
    if any(value is None for value in values):
        raise ConcatenationError(f"Cropping removed required data from {session.path}")
    return RetainedStats(
        time_min=time_min,
        time_max=time_max,
        position_first=position_first,
        position_last=position_last,
        buffer_first=buffer_first,
        buffer_last=buffer_last,
        row_counts=row_counts,
    )


def _calculate_offsets(
    sessions: list[SourceSession],
    retained: list[RetainedStats],
    boundaries: list[BoundarySplice],
) -> list[SessionOffsets]:
    offsets: list[SessionOffsets] = []
    previous_time_end: float | None = None
    previous_buffer_end: float | None = None

    for index, (session, stats) in enumerate(zip(sessions, retained)):
        seconds_offset = 0.0
        if previous_time_end is not None:
            seconds_offset = previous_time_end + JOIN_GAP_SECONDS - stats.time_min

        position_offset = 0.0
        if index > 0:
            boundary = boundaries[index - 1]
            previous_offset = offsets[index - 1].position
            position_offset = (
                boundary.target_next_landmark_entry
                + previous_offset
                - boundary.next_source_first_landmark_entry
            )

        buffer_offset = 0.0
        if previous_buffer_end is not None and stats.buffer_first <= previous_buffer_end:
            buffer_offset = previous_buffer_end + 1 - stats.buffer_first

        offsets.append(
            SessionOffsets(
                seconds=seconds_offset,
                position=position_offset,
                buffer=buffer_offset,
            )
        )
        previous_time_end = stats.time_max + seconds_offset
        previous_buffer_end = stats.buffer_last + buffer_offset

    return offsets


def _format_number(value: float) -> str:
    return format(value, ".17g")


def _merge_csv_stream(
    destination: Path,
    stream: str,
    sessions: list[SourceSession],
    slices: list[SessionSlice],
    offsets: list[SessionOffsets],
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    schema = sessions[0].schemas[stream]
    rows_written = 0
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(schema)
        for session, session_slice, session_offsets in zip(sessions, slices, offsets):
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
                        raw_seconds = _finite_float(
                            row[seconds_index],
                            description=f"Seconds in {csv_path}",
                        )
                        if not _is_retained(raw_seconds, session_slice):
                            continue
                        row[seconds_index] = _format_number(
                            raw_seconds + session_offsets.seconds
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


def _analog_row_range(
    session: SourceSession, session_slice: SessionSlice
) -> tuple[int, int, int]:
    total = 0
    retained_indices: list[int] = []
    for path, _, row in _iter_dict_rows(_csv_files(session.path, "analog-data")):
        seconds = _finite_float(row["Seconds"], description=f"Seconds in {path}")
        if _is_retained(seconds, session_slice):
            retained_indices.append(total)
        total += 1
    if not retained_indices:
        raise ConcatenationError(f"Cropping removed all analog buffers from {session.path}")
    start = retained_indices[0]
    end = retained_indices[-1] + 1
    if retained_indices != list(range(start, end)):
        raise ConcatenationError(f"Retained analog buffers are not contiguous in {session.path}")
    return start, end, total


def _copy_binary_range(
    sources: list[Path], output, start_byte: int, end_byte: int
) -> None:
    cursor = 0
    for source in sources:
        size = source.stat().st_size
        source_start = max(start_byte - cursor, 0)
        source_end = min(end_byte - cursor, size)
        if source_start < source_end:
            with source.open("rb") as input_file:
                input_file.seek(source_start)
                remaining = source_end - source_start
                while remaining:
                    chunk = input_file.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ConcatenationError(
                            f"Unexpected end of analog binary file: {source}"
                        )
                    output.write(chunk)
                    remaining -= len(chunk)
        cursor += size


def _merge_analog_binary(
    destination: Path,
    sessions: list[SourceSession],
    slices: list[SessionSlice],
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    with destination.open("wb") as output:
        for session, session_slice in zip(sessions, slices):
            sources = _binary_files(session.path)
            total_bytes = sum(source.stat().st_size for source in sources)
            rig_settings = _read_settings(
                _settings_file(session.path, "rig-settings")
            )
            try:
                channel_count = len(rig_settings["analogInputChannels"])
            except (KeyError, TypeError) as exc:
                raise ConcatenationError(
                    f"No analogInputChannels found in {session.path}"
                ) from exc
            bytes_per_frame = 8 * channel_count  # float64 value per channel
            if total_bytes % bytes_per_frame:
                raise ConcatenationError(
                    f"Analog binary size is not divisible by its channel width in {session.path}"
                )
            start_row, end_row, total_rows = _analog_row_range(session, session_slice)
            total_samples = total_bytes // bytes_per_frame
            if total_samples % total_rows == 0:
                binary_buffer_count = total_rows
            elif total_rows > 1 and total_samples % (total_rows - 1) == 0:
                binary_buffer_count = total_rows - 1
            else:
                raise ConcatenationError(
                    "Analog samples do not match either the CSV buffer count or "
                    f"one fewer trailing buffer in {session.path}"
                )
            samples_per_buffer = total_samples // binary_buffer_count
            missing_trailing_buffers = total_rows - binary_buffer_count
            if missing_trailing_buffers not in (0, 1):
                raise ConcatenationError(
                    "Analog CSV/binary buffer counts differ by more than one "
                    f"trailing buffer in {session.path}: CSV={total_rows}, "
                    f"binary={binary_buffer_count}"
                )
            bytes_per_buffer = samples_per_buffer * bytes_per_frame
            start_byte = start_row * bytes_per_buffer
            end_byte = min(end_row, binary_buffer_count) * bytes_per_buffer
            if start_byte >= end_byte:
                raise ConcatenationError(
                    f"Cropping removed all analog binary samples from {session.path}"
                )
            _copy_binary_range(sources, output, start_byte, end_byte)
            bytes_written += end_byte - start_byte
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
    if __package__:
        # Used when imported as analysis.sequence_compression.concatenate_sessions.
        from .session_functions.io import load_analog_data, load_data, load_settings
    else:
        # Used when invoked directly:
        # python analysis/sequence_compression/concatenate_sessions.py ...
        from session_functions.io import load_analog_data, load_data, load_settings

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
    fallback_landmark_gap: float = DEFAULT_LANDMARK_GAP,
) -> Path:
    """Concatenate two or more restarted sessions and return the output path."""
    sessions = sorted(
        (_inspect_session(Path(path)) for path in session_paths),
        key=lambda session: session.acquired_at,
    )
    _validate_compatibility(sessions)
    slices, boundaries = _plan_session_slices(sessions, fallback_landmark_gap)
    retained = [
        _retained_stats(session, session_slice)
        for session, session_slice in zip(sessions, slices)
    ]
    offsets = _calculate_offsets(sessions, retained, boundaries)

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
                destination, stream, sessions, slices, offsets
            )

        analog_bytes = _merge_analog_binary(
            destination_behav / "analog-data" / f"analog-data_{CHUNK_STAMP}.bin",
            sessions,
            slices,
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
                    "retained_rows": stats.row_counts,
                    "discarded_rows": {
                        stream: session.row_counts[stream] - stats.row_counts.get(stream, 0)
                        for stream in session.row_counts
                    },
                    "slice": {
                        "start_seconds_inclusive": session_slice.start_seconds,
                        "end_seconds_exclusive": session_slice.end_seconds,
                    },
                    "offsets": {
                        "seconds": offset.seconds,
                        "position": offset.position,
                        "buffer": offset.buffer,
                    },
                }
                for session, session_slice, stats, offset in zip(
                    sessions, slices, retained, offsets
                )
            ],
            "splices": [
                {
                    "source": sessions[splice.source_index].path.name,
                    "next_source": sessions[splice.next_source_index].path.name,
                    "landmarks_per_lap": splice.landmarks_per_lap,
                    "complete_laps_retained": splice.complete_laps_retained,
                    "discarded_logical_landmarks": splice.discarded_logical_landmarks,
                    "last_complete_landmark_exit": splice.last_complete_landmark_exit,
                    "target_next_landmark_entry": splice.target_next_landmark_entry,
                    "next_source_first_landmark_entry_before_shift": (
                        splice.next_source_first_landmark_entry
                    ),
                    "preserved_gap": splice.preserved_gap,
                    "gap_source": splice.gap_source,
                    "source_end_seconds_exclusive": splice.source_end_seconds,
                    "next_source_start_seconds_inclusive": (
                        splice.next_source_start_seconds
                    ),
                }
                for splice in boundaries
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
    parser.add_argument(
        "--fallback-landmark-gap",
        type=float,
        default=DEFAULT_LANDMARK_GAP,
        help=(
            "edge-to-edge gap used when the next landmark was not logged and "
            "settings do not specify one unambiguous offset (default: 9)"
        ),
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
            fallback_landmark_gap=args.fallback_landmark_gap,
        )
    except ConcatenationError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Batch discovery and barcode alignment for sequence-compression sessions.

The Slurm wrapper runs one completed Suite2p session per array task. This stage builds
and verifies alignment products only; session overviews are handled by a separate batch.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from .sequence_compression import (
    build_sequence_compression_session,
    resolve_paths,
)


@dataclass(frozen=True)
class BatchSession:
    subject: str
    session: str
    suite2p_dir: Path


REQUIRED_PLANE0_FILES = ("ops.npy", "F.npy", "Fneu.npy", "spks.npy", "iscell.npy", "stat.npy")


def normalize_subject(subject: str) -> str:
    """Return the cohort directory form (for example, ``07`` -> ``sub-07``)."""
    value = subject.strip()
    if value.startswith("sub-"):
        value = value[4:]
    if value.isdigit():
        value = f"{int(value):02d}"
    if not value:
        raise ValueError("Subject cannot be empty")
    return f"sub-{value}"


def discover_sessions(
    project_root: str | Path, subjects: list[str] | tuple[str, ...] | None = None
) -> list[BatchSession]:
    """Find complete Suite2p outputs and resolve each to its exact raw session."""
    project_root = Path(project_root).expanduser().resolve()
    processed_root = project_root / "processed" / "cohort2"
    if not processed_root.is_dir():
        raise FileNotFoundError(f"Processed cohort-2 directory does not exist: {processed_root}")

    selected_subjects = (
        {normalize_subject(subject) for subject in subjects} if subjects else None
    )
    sessions = []
    for subject_dir in sorted(path for path in processed_root.iterdir() if path.is_dir()):
        if selected_subjects is not None and subject_dir.name not in selected_subjects:
            continue
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            raw_session = (
                project_root
                / "rawdata"
                / "cohort2"
                / subject_dir.name
                / session_dir.name
            )
            if not session_dir.name.startswith("ses-") or not raw_session.is_dir():
                continue
            suite2p_dir = session_dir / "suite2p"
            plane0 = suite2p_dir / "plane0"
            if not all((plane0 / filename).is_file() for filename in REQUIRED_PLANE0_FILES):
                continue
            sessions.append(
                BatchSession(
                    subject=subject_dir.name,
                    session=session_dir.name,
                    suite2p_dir=suite2p_dir,
                )
            )
    return sessions


def write_manifest(sessions: list[BatchSession], manifest_path: str | Path) -> Path:
    """Write a stable, zero-based CSV manifest for a Slurm array."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.partial")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["task_id", "subject", "session", "suite2p_dir"]
        )
        writer.writeheader()
        for task_id, item in enumerate(sessions):
            writer.writerow(
                {
                    "task_id": task_id,
                    "subject": item.subject,
                    "session": item.session,
                    "suite2p_dir": item.suite2p_dir,
                }
            )
    temporary_path.replace(manifest_path)
    return manifest_path


def read_manifest_task(manifest_path: str | Path, task_id: int) -> BatchSession:
    """Read exactly one zero-based task, rejecting malformed or duplicate IDs."""
    matches = []
    with Path(manifest_path).expanduser().resolve().open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            if int(row["task_id"]) == task_id:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one row for task_id={task_id}, found {len(matches)}"
        )
    row = matches[0]
    return BatchSession(
        subject=row["subject"],
        session=row["session"],
        suite2p_dir=Path(row["suite2p_dir"]),
    )


def run_session(
    project_root: str | Path,
    subject: str,
    session: str,
    *,
    force: bool = False,
) -> Path:
    """Build and verify one session's barcode-aligned outputs."""
    project_root = Path(project_root).expanduser().resolve()
    paths = resolve_paths(subject, session, project_root)
    print(f"=== {paths.subject} / {paths.full_session_name} ===", flush=True)
    print(f"Suite2p: {paths.suite2p_dir}", flush=True)

    merged_nwb, verification_flags = build_sequence_compression_session(
        subject=paths.subject,
        session=paths.full_session_name,
        project_root=str(project_root),
        force=force,
    )
    if verification_flags:
        joined = "\n  - ".join(str(flag) for flag in verification_flags)
        raise RuntimeError(
            "NWB/barcode verification returned issue(s):\n"
            f"  - {joined}"
        )

    print(f"Aligned output: {paths.output_session}", flush=True)
    return Path(merged_nwb)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--write-manifest")
    source.add_argument("--manifest")
    source.add_argument("--all", action="store_true")
    source.add_argument("--subject")
    parser.add_argument(
        "--include-subject",
        action="append",
        default=[],
        help="Limit discovery to this subject; repeat to include multiple subjects.",
    )
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--session")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.write_manifest:
        sessions = discover_sessions(args.project_root, args.include_subject)
        manifest = write_manifest(sessions, args.write_manifest)
        print(f"Wrote {len(sessions)} task(s) to {manifest}")
        return 0

    if args.manifest:
        if args.task_id is None:
            raise SystemExit("--manifest requires --task-id")
        task = read_manifest_task(args.manifest, args.task_id)
        run_session(
            args.project_root, task.subject, task.session, force=args.force
        )
        return 0

    if args.subject:
        if not args.session:
            raise SystemExit("--subject requires --session")
        run_session(args.project_root, args.subject, args.session, force=args.force)
        return 0

    sessions = discover_sessions(args.project_root, args.include_subject)
    failures = []
    for task in sessions:
        try:
            run_session(
                args.project_root, task.subject, task.session, force=args.force
            )
        except Exception as error:  # continue the serial batch, then return failure
            failures.append((task, error))
            print(
                f"FAILED {task.subject}/{task.session}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
    if failures:
        print(f"{len(failures)}/{len(sessions)} session(s) failed.", flush=True)
        return 1
    print(f"Completed {len(sessions)} session(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Batch discovery and plotting for aligned sequence-compression sessions.

This stage never builds or modifies NWBs. It discovers complete alignment products under
``BehaviourAligned_robust`` and runs the sequence-compression session overview for each.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from nwb_barcoding.batch_sequence_compression import normalize_subject


@dataclass(frozen=True)
class AlignedSession:
    subject: str
    session: str
    aligned_dir: Path


REQUIRED_ALIGNED_FILES = ("behaviour_data.npz", "valid_frames.npz")


def alignment_is_complete(session_dir: str | Path) -> bool:
    """Return whether a directory has every product needed by the overview."""
    session_dir = Path(session_dir)
    merged_nwbs = sorted(session_dir.glob("vrabcd_2p_preprocessed_*.nwb"))
    return (
        len(merged_nwbs) == 1
        and all((session_dir / name).is_file() for name in REQUIRED_ALIGNED_FILES)
    )


def discover_aligned_sessions(
    project_root: str | Path,
    subjects: list[str] | tuple[str, ...] | None = None,
) -> list[AlignedSession]:
    """Find complete aligned sessions, optionally restricted to selected subjects."""
    project_root = Path(project_root).expanduser().resolve()
    aligned_root = project_root / "BehaviourAligned_robust" / "cohort2"
    if not aligned_root.is_dir():
        raise FileNotFoundError(
            f"Aligned cohort-2 directory does not exist: {aligned_root}"
        )
    selected_subjects = (
        {normalize_subject(subject) for subject in subjects} if subjects else None
    )
    sessions = []
    for subject_dir in sorted(path for path in aligned_root.iterdir() if path.is_dir()):
        if selected_subjects is not None and subject_dir.name not in selected_subjects:
            continue
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            if alignment_is_complete(session_dir):
                sessions.append(
                    AlignedSession(
                        subject=subject_dir.name,
                        session=session_dir.name,
                        aligned_dir=session_dir,
                    )
                )
    return sessions


def write_manifest(sessions: list[AlignedSession], manifest_path: str | Path) -> Path:
    """Write an atomic, stable, zero-based CSV manifest."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.partial")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["task_id", "subject", "session", "aligned_dir"]
        )
        writer.writeheader()
        for task_id, item in enumerate(sessions):
            writer.writerow(
                {
                    "task_id": task_id,
                    "subject": item.subject,
                    "session": item.session,
                    "aligned_dir": item.aligned_dir,
                }
            )
    temporary_path.replace(manifest_path)
    return manifest_path


def read_manifest_task(manifest_path: str | Path, task_id: int) -> AlignedSession:
    """Read exactly one zero-based task from an overview manifest."""
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
    return AlignedSession(
        subject=row["subject"],
        session=row["session"],
        aligned_dir=Path(row["aligned_dir"]),
    )


def run_session(
    project_root: str | Path,
    subject: str,
    session: str,
) -> Path:
    """Generate the five-plot overview for one already-aligned session."""
    project_root = Path(project_root).expanduser().resolve()
    subject = normalize_subject(subject)
    aligned_subject = project_root / "BehaviourAligned_robust" / "cohort2" / subject
    from nwb_barcoding.sequence_compression import resolve_raw_session

    aligned_session = resolve_raw_session(aligned_subject, session)
    if not alignment_is_complete(aligned_session):
        raise FileNotFoundError(
            f"Alignment products are incomplete for {subject}/{aligned_session.name}"
        )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from analysis.sequence_compression.session_overview_sc import run_overview

    overview_dir = (
        project_root / "analysis" / "cohort2" / subject / aligned_session.name
    )
    _, figures = run_overview(
        mouse=subject,
        session_id=aligned_session.name,
        project_root=project_root,
        save_dir=overview_dir,
        show=False,
    )
    for figure in figures.values():
        if figure is not None:
            plt.close(figure)
    print(f"Overview output: {overview_dir}", flush=True)
    return overview_dir


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.write_manifest:
        sessions = discover_aligned_sessions(args.project_root, args.include_subject)
        manifest = write_manifest(sessions, args.write_manifest)
        print(f"Wrote {len(sessions)} task(s) to {manifest}")
        return 0

    if args.manifest:
        if args.task_id is None:
            raise SystemExit("--manifest requires --task-id")
        task = read_manifest_task(args.manifest, args.task_id)
        run_session(args.project_root, task.subject, task.session)
        return 0

    if args.subject:
        if not args.session:
            raise SystemExit("--subject requires --session")
        run_session(args.project_root, args.subject, args.session)
        return 0

    sessions = discover_aligned_sessions(args.project_root, args.include_subject)
    failures = []
    for task in sessions:
        try:
            run_session(args.project_root, task.subject, task.session)
        except Exception as error:
            failures.append((task, error))
            print(
                f"FAILED {task.subject}/{task.session}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
    if failures:
        print(f"{len(failures)}/{len(sessions)} session(s) failed.", flush=True)
        return 1
    print(f"Completed {len(sessions)} session overview(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

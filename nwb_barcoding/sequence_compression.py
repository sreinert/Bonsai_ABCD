"""Path adapter for Bonsai sequence-compression cohort 2 sessions.

This module contains no conversion logic.  It resolves the project's split raw and
processed layouts, then delegates to :func:`build_session_from_paths`.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SequenceCompressionPaths:
    subject: str
    full_session_name: str
    short_session_name: str
    raw_session_path: Path
    processed_session_path: Path
    suite2p_dir: Path
    output_session: Path


def short_session_name(full_session_name):
    """Return ``full002`` from ``ses-full002_date-...`` (and equivalent names)."""
    name = Path(full_session_name).name
    if name.startswith("ses-"):
        name = name[4:]
    return name.split("_date-", 1)[0]


def resolve_raw_session(subject_dir, session):
    """Resolve an exact full or short session identifier, rejecting ambiguity."""
    subject_dir = Path(subject_dir)
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"Subject directory does not exist: {subject_dir}")

    exact = subject_dir / session
    if exact.is_dir():
        return exact

    wanted = short_session_name(session)
    matches = sorted(
        path for path in subject_dir.iterdir()
        if path.is_dir()
        and path.name.startswith("ses-")
        and short_session_name(path.name) == wanted
    )
    if not matches:
        raise FileNotFoundError(
            f"No session under {subject_dir} has full or short identifier {session!r}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Session identifier {session!r} is ambiguous under {subject_dir}: "
            f"{[path.name for path in matches]}. Pass the full session name."
        )
    return matches[0]


def resolve_paths(subject, session, project_root, output_root=None):
    """Resolve the raw, Suite2p, and isolated output paths for one session."""
    project_root = Path(project_root).expanduser().resolve()
    raw_subject = project_root / "rawdata" / "cohort2" / subject
    raw_session = resolve_raw_session(raw_subject, session)
    short_name = short_session_name(raw_session.name)
    processed_session = project_root / "processed" / "cohort2" / subject / short_name
    suite2p_dir = processed_session / "suite2p"
    if not suite2p_dir.is_dir():
        raise FileNotFoundError(f"Suite2p directory does not exist: {suite2p_dir}")

    if output_root is None:
        output_root = project_root / "BehaviourAligned_robust" / "cohort2"
    else:
        output_root = Path(output_root).expanduser().resolve()

    return SequenceCompressionPaths(
        subject=subject,
        full_session_name=raw_session.name,
        short_session_name=short_name,
        raw_session_path=raw_session,
        processed_session_path=processed_session,
        suite2p_dir=suite2p_dir,
        output_session=output_root / subject / raw_session.name,
    )


def build_sequence_compression_session(subject: str, session: str, project_root: str,
                                       output_root: str | None = None,
                                       dff_neuropil_coefficient: float | None = None,
                                       build_behaviour_data_files: bool = True,
                                       force: bool = False):
    """Resolve sequence-compression paths and run the shared Bonsai builder."""
    paths = resolve_paths(subject, session, project_root, output_root)

    # Delay the heavyweight pynwb/neuroconv imports so path discovery and its tests work
    # in the repository's ordinary development environment.
    from .build_session_nwb import build_session_from_paths

    return build_session_from_paths(
        subject=paths.subject,
        full_session_name=paths.full_session_name,
        raw_session_path=paths.raw_session_path,
        suite2p_dir=paths.suite2p_dir,
        output_session=paths.output_session,
        source_session=paths.processed_session_path,
        dff_neuropil_coefficient=dff_neuropil_coefficient,
        build_behaviour_data_files=build_behaviour_data_files,
        force=force,
    )


if __name__ == "__main__":
    import defopt

    defopt.run(build_sequence_compression_session)

#!/usr/bin/env python3
"""Find the processed-data session associated with a Slurm job ID or name."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from full_session_pipeline.full_session_common import resolve_data_root  # noqa: E402


DEFAULT_PROJECT_ROOT = Path("AtAp_20260119_SequenceCompression")
JOB_ID_PATTERN = re.compile(r"^\d+(?:_[0-9]+)?$")


@dataclass(frozen=True)
class SessionMatch:
    processed_session: Path
    job_id: str | None = None
    job_name: str | None = None
    source: str = "pipeline provenance"


def _job_fields(payload: object) -> tuple[str | None, str | None]:
    if not isinstance(payload, dict):
        return None, None
    job = payload.get("job")
    job = job if isinstance(job, dict) else {}
    job_id = job.get("job_id") or payload.get("job_id")
    job_name = job.get("name") or payload.get("job_name")
    return (
        str(job_id) if job_id not in (None, "") else None,
        str(job_name) if job_name not in (None, "") else None,
    )


def provenance_matches(processed_root: Path, reference: str) -> list[SessionMatch]:
    """Read full-pipeline submission records beneath a processed cohort root."""
    matches: list[SessionMatch] = []
    pattern = "*/*/suite2p/.workflow/submission.json"
    for submission in sorted(processed_root.glob(pattern)):
        try:
            payload = json.loads(submission.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        job_id, job_name = _job_fields(payload)
        if reference not in {job_id, job_name}:
            continue
        matches.append(
            SessionMatch(
                processed_session=submission.parents[2],
                job_id=job_id,
                job_name=job_name,
            )
        )
    return matches


def processed_session_from_workdir(
    workdir: str, processed_root: Path
) -> Path | None:
    """Return the mouse/session directory containing a Slurm working directory."""
    if not workdir or workdir in {"Unknown", "(null)", "N/A"}:
        return None
    path = Path(workdir).expanduser()
    try:
        relative = path.resolve(strict=False).relative_to(
            processed_root.resolve(strict=False)
        )
    except ValueError:
        return None
    if len(relative.parts) < 2:
        return None
    return processed_root / relative.parts[0] / relative.parts[1]


def parse_slurm_output(text: str, processed_root: Path) -> list[SessionMatch]:
    """Parse ``job id|job name|working directory`` records from Slurm."""
    matches: list[SessionMatch] = []
    for line in text.splitlines():
        fields = line.strip().split("|", maxsplit=2)
        if len(fields) != 3:
            continue
        job_id, job_name, workdir = (field.strip() for field in fields)
        processed_session = processed_session_from_workdir(workdir, processed_root)
        if processed_session is None:
            continue
        matches.append(
            SessionMatch(
                processed_session=processed_session,
                job_id=job_id or None,
                job_name=job_name or None,
                source="Slurm working directory",
            )
        )
    return matches


def _run_slurm(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def scheduler_matches(processed_root: Path, reference: str) -> list[SessionMatch]:
    """Ask squeue and sacct for current/recent jobs and use their WorkDir fields."""
    by_id = JOB_ID_PATTERN.fullmatch(reference) is not None
    filter_option = f"--jobs={reference}" if by_id else f"--name={reference}"
    outputs: list[str] = []
    if shutil.which("squeue"):
        outputs.append(
            _run_slurm(
                ["squeue", "--noheader", filter_option, "--format=%i|%j|%Z"]
            )
        )
    if shutil.which("sacct"):
        outputs.append(
            _run_slurm(
                [
                    "sacct",
                    "--noheader",
                    "--parsable2",
                    "--allocations",
                    filter_option,
                    "--format=JobIDRaw,JobName,WorkDir",
                ]
            )
        )
    return [
        match
        for output in outputs
        for match in parse_slurm_output(output, processed_root)
    ]


def deduplicate(matches: list[SessionMatch]) -> list[SessionMatch]:
    result: list[SessionMatch] = []
    seen: set[tuple[Path, str | None]] = set()
    for match in matches:
        key = (match.processed_session.resolve(strict=False), match.job_id)
        if key not in seen:
            seen.add(key)
            result.append(match)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "job",
        help=(
            "Slurm job ID (recommended for older jobs) or exact job name, as shown "
            "by squeue/sacct"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help=(
            "SequenceCompression project directory or name. Relative paths "
            "auto-detect /ceph or /Volumes"
        ),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        help="Processed cohort root (default: PROJECT_ROOT/processed/cohort2)",
    )
    parser.add_argument(
        "--no-slurm",
        action="store_true",
        help="Only search saved pipeline provenance; do not call squeue or sacct",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.processed_root:
        processed_root = args.processed_root.expanduser().resolve()
    else:
        project_root = resolve_data_root(args.project_root.expanduser())
        processed_root = project_root / "processed" / "cohort2"
    if not processed_root.is_dir():
        raise NotADirectoryError(f"Processed cohort root not found: {processed_root}")

    matches = provenance_matches(processed_root, args.job)
    if not args.no_slurm:
        matches.extend(scheduler_matches(processed_root, args.job))
    matches = deduplicate(matches)

    if not matches:
        print(
            f"No processed session found for Slurm job {args.job!r} under "
            f"{processed_root}.",
            file=sys.stderr,
        )
        if not JOB_ID_PATTERN.fullmatch(args.job):
            print(
                "Older pipeline jobs shared one generic name; use the numeric job ID "
                "for those jobs.",
                file=sys.stderr,
            )
        return 1

    if len(matches) > 1:
        print(f"Found {len(matches)} matching sessions:")
    for index, match in enumerate(matches, start=1):
        prefix = f"[{index}] " if len(matches) > 1 else ""
        relative = match.processed_session.relative_to(processed_root)
        print(f"{prefix}Session:           {relative}")
        print(f"{prefix}Processed session: {match.processed_session}")
        print(f"{prefix}Suite2p output:    {match.processed_session / 'suite2p'}")
        print(
            f"{prefix}Slurm job:         "
            f"{match.job_id or '<unknown>'} ({match.job_name or '<unknown name>'})"
        )
        print(f"{prefix}Resolved from:      {match.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

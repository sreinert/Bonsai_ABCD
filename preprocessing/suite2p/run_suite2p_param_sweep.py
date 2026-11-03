#!/usr/bin/env python3
"""Deprecated adapter for the unified registration-tuning CLI.

New code should call registration_tuning/cli.py directly. Unlike the former
implementation, this adapter requires a safe output root outside raw data.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKFLOW_DIR = SCRIPT_DIR / "registration_tuning"


def resolve_session(args: argparse.Namespace) -> Path:
    if args.session_dir is not None:
        return args.session_dir.expanduser().resolve()
    projects_root = Path(
        os.environ.get(
            "MRSIC_PROJECTS_ROOT",
            (
                "/ceph/mrsic_flogel/public/projects"
                if Path("/ceph/mrsic_flogel/public/projects").is_dir()
                else "/Volumes/mrsic_flogel/public/projects"
            ),
        )
    )
    basepath = args.basepath
    base = basepath if basepath.is_absolute() else projects_root / basepath
    return (base / args.animal / args.session).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path)
    parser.add_argument("--animal", default="TAA0000061")
    parser.add_argument("--session", default="ses-007_date-20250307_protocol-t2")
    parser.add_argument(
        "--basepath", type=Path, default=Path("AtApSuKuSaRe_20250129_HFScohort2")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-kind", choices=("small", "full"), default="small")
    parser.add_argument(
        "--config", type=Path, default=WORKFLOW_DIR / "candidates.json"
    )
    parser.add_argument("--param", action="append")
    parser.add_argument(
        "--execution", choices=("direct", "slurm"), default="direct"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    parameters = args.param or ["smooth_sigma", "maxregshift", "maxregshiftNR"]
    command = [
        sys.executable,
        str(WORKFLOW_DIR / "cli.py"),
        "sweep",
        "--session-dir",
        str(resolve_session(args)),
        "--input-kind",
        args.input_kind,
        "--config",
        str(args.config),
        "--output-root",
        str(args.output_root),
        "--execution",
        args.execution,
        "--device",
        args.device,
    ]
    for parameter in parameters:
        command.extend(["--param", parameter])
    print(
        "DEPRECATED: use preprocessing/suite2p/registration_tuning/cli.py sweep",
        file=sys.stderr,
    )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

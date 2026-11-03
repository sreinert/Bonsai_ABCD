#!/usr/bin/env python3
"""Build and run Suite2p registration sweeps or one-off registrations."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import paths_overlap
from execution import (
    run_direct,
    submit_slurm,
    write_or_validate_experiment,
    write_or_validate_manifest,
)
from make_tasks import build_tasks
from session_inputs import discover_session_input
from tuning_config import (
    build_preset_candidate,
    build_sweep_candidates,
    load_config,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="mode", required=True)
    for mode in ("sweep", "single"):
        command = subparsers.add_parser(mode)
        command.add_argument(
            "--session-dir", type=Path, action="append", required=True
        )
        command.add_argument(
            "--input-kind", choices=("small", "full"), required=True
        )
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument(
            "--execution", choices=("direct", "slurm"), required=True
        )
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    subparsers.choices["sweep"].add_argument(
        "--param", action="append", required=True, dest="parameters"
    )
    subparsers.choices["single"].add_argument("--preset", required=True)
    return result


def prepare_experiment(args: argparse.Namespace) -> tuple[Path, int, Path]:
    if args.mode == "single" and len(args.session_dir) != 1:
        raise ValueError("single mode requires exactly one --session-dir")
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    sessions = [
        discover_session_input(path, args.input_kind) for path in args.session_dir
    ]
    session_keys = [(item.mouse_id, item.session_id) for item in sessions]
    if len(session_keys) != len(set(session_keys)):
        raise ValueError("The same inferred mouse/session was selected more than once")

    if args.mode == "sweep":
        candidates = build_sweep_candidates(config, args.parameters)
        selection = {"parameters": args.parameters}
    else:
        candidates = [build_preset_candidate(config, args.preset)]
        selection = {"preset": args.preset}

    output_root = args.output_root.expanduser().resolve()
    for session in sessions:
        if paths_overlap(output_root, session.input_path):
            raise ValueError(
                f"Output root {output_root} overlaps raw input {session.input_path}"
            )
    session_rows = [item.manifest_row() for item in sessions]
    tasks = build_tasks(
        session_rows,
        candidates,
        str(config["suite2p_version"]),
        output_root / "runs",
    )
    experiment = {
        "schema_version": 1,
        "mode": args.mode,
        "input_kind": args.input_kind,
        "sessions": [str(item.session_path) for item in sessions],
        **selection,
        "config": config,
    }
    experiment_path = output_root / "experiment.json"
    if experiment_path.exists():
        write_or_validate_experiment(experiment_path, experiment)
    manifest = output_root / "tasks.csv"
    write_or_validate_manifest(manifest, tasks)
    if not experiment_path.exists():
        write_or_validate_experiment(experiment_path, experiment)
    print(f"Experiment: {experiment_path}")
    print(f"Manifest:   {manifest}")
    print(f"Tasks:      {len(tasks)}")
    for item in sessions:
        print(
            f"Input:      {item.mouse_id}/{item.session_id} [{item.input_kind}; "
            f"{item.input_selection}] -> {len(item.tiff_files)} TIFF(s)"
        )
    return manifest, len(tasks), output_root


def main() -> None:
    args = parser().parse_args()
    manifest, task_count, output_root = prepare_experiment(args)
    if args.execution == "direct":
        run_direct(
            manifest,
            task_count,
            output_root,
            SCRIPT_DIR / "run_candidate.py",
            device=args.device,
        )
    else:
        submit_slurm(
            manifest,
            task_count,
            output_root,
            SCRIPT_DIR / "run_grid_array.sbatch",
            REPO_ROOT,
            input_kind=args.input_kind,
        )


if __name__ == "__main__":
    main()

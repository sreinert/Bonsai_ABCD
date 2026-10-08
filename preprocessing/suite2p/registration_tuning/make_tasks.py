#!/usr/bin/env python3
"""Create an immutable session-by-candidate task manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from common import paths_overlap, read_csv, require_columns, slug, write_csv


SESSION_COLUMNS = {
    "mouse_id",
    "session_id",
    "selection_role",
    "input_path",
    "tiff_files_json",
    "fs",
    "tau",
    "nplanes",
    "nchannels",
}


def load_candidates(path: Path) -> tuple[str, list[dict[str, object]]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    suite2p_version = str(config["suite2p_version"])
    fixed = config.get("fixed_registration", {})
    candidates = config.get("candidates", [])
    if not candidates:
        raise ValueError("Candidate configuration contains no candidates")
    names: set[str] = set()
    merged: list[dict[str, object]] = []
    for candidate in candidates:
        name = candidate.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Every candidate requires a non-empty name")
        if name in names:
            raise ValueError(f"Duplicate candidate name: {name}")
        names.add(name)
        registration = dict(fixed)
        registration.update(candidate.get("registration", {}))
        merged.append({"name": name, "registration": registration})
    return suite2p_version, merged


def build_tasks(
    sessions: list[dict[str, str]],
    candidates: list[dict[str, object]],
    suite2p_version: str,
    output_root: Path,
) -> list[dict[str, object]]:
    output_root = output_root.resolve()
    for session in sessions:
        if int(session["nchannels"]) != 2:
            raise ValueError(
                f"Session {session['mouse_id']}/{session['session_id']} has "
                f"nchannels={session['nchannels']}; this project requires 2. "
                "Regenerate selected_sessions.csv with the current discovery script."
            )
        input_path = Path(session["input_path"]).resolve()
        if paths_overlap(output_root, input_path):
            raise ValueError(
                f"Unsafe output root {output_root}: it overlaps raw input {input_path}"
            )
    rows: list[dict[str, object]] = []
    for session in sessions:
        for candidate in candidates:
            registration = candidate["registration"]
            digest = hashlib.sha256(
                json.dumps(registration, sort_keys=True).encode("utf-8")
            ).hexdigest()[:10]
            candidate_id = f"{slug(str(candidate['name']))}-{digest}"
            run_dir = (
                output_root
                / slug(session["mouse_id"])
                / slug(session["session_id"])
                / candidate_id
            )
            if paths_overlap(run_dir, Path(session["input_path"])):
                raise ValueError(f"Unsafe run directory: {run_dir}")
            rows.append(
                {
                    "task_id": len(rows),
                    "mouse_id": session["mouse_id"],
                    "session_id": session["session_id"],
                    "selection_role": session["selection_role"],
                    "candidate_name": candidate["name"],
                    "candidate_id": candidate_id,
                    "suite2p_version": suite2p_version,
                    "input_path": session["input_path"],
                    "tiff_files_json": session["tiff_files_json"],
                    "frame_count": session.get("frame_count", ""),
                    "fs": session["fs"],
                    "tau": session["tau"],
                    "nplanes": session["nplanes"],
                    "nchannels": session["nchannels"],
                    "registration_json": json.dumps(registration, sort_keys=True),
                    "run_dir": str(run_dir),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--mouse",
        action="append",
        help="Include only this mouse ID; repeat to include multiple mice",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sessions = require_columns(read_csv(args.sessions), SESSION_COLUMNS, "session manifest")
    if args.mouse:
        requested = set(args.mouse)
        available = {row["mouse_id"] for row in sessions}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Requested mice are absent from the session manifest: {missing}")
        sessions = [row for row in sessions if row["mouse_id"] in requested]
    suite2p_version, candidates = load_candidates(args.candidates)
    tasks = build_tasks(sessions, candidates, suite2p_version, args.output_root)
    write_csv(args.manifest, tasks, overwrite=args.overwrite)
    print(f"Wrote {len(tasks)} tasks to {args.manifest}")


if __name__ == "__main__":
    main()

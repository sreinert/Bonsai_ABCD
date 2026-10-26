import json
from pathlib import Path

from preprocessing.suite2p.find_slurm_session import (
    deduplicate,
    main,
    parse_slurm_output,
    provenance_matches,
)


def write_submission(
    processed_root: Path,
    mouse: str,
    session: str,
    *,
    job_id: str,
    job_name: str,
) -> Path:
    workflow = processed_root / mouse / session / "suite2p" / ".workflow"
    workflow.mkdir(parents=True)
    (workflow / "submission.json").write_text(
        json.dumps({"job_name": job_name, "job": {"job_id": job_id, "name": job_name}}),
        encoding="utf-8",
    )
    return workflow


def test_provenance_matches_job_id_and_name(tmp_path):
    processed_root = tmp_path / "processed" / "cohort2"
    write_submission(
        processed_root,
        "sub-02",
        "full002",
        job_id="12345",
        job_name="s2p__sub-02__full002",
    )

    by_id = provenance_matches(processed_root, "12345")
    by_name = provenance_matches(processed_root, "s2p__sub-02__full002")

    expected = processed_root / "sub-02" / "full002"
    assert [match.processed_session for match in by_id] == [expected]
    assert [match.processed_session for match in by_name] == [expected]


def test_parse_slurm_output_maps_workdir_to_processed_session(tmp_path):
    processed_root = tmp_path / "processed" / "cohort2"
    workdir = processed_root / "sub-03" / "full010" / "suite2p" / ".workflow"

    matches = parse_slurm_output(
        f"8123|s2p-full-session|{workdir}\n8123.batch|batch|Unknown\n",
        processed_root,
    )

    assert len(matches) == 1
    assert matches[0].processed_session == processed_root / "sub-03" / "full010"
    assert matches[0].job_id == "8123"


def test_deduplicate_same_provenance_and_scheduler_result(tmp_path):
    processed_root = tmp_path / "processed" / "cohort2"
    workflow = write_submission(
        processed_root,
        "sub-02",
        "full002",
        job_id="12345",
        job_name="s2p__sub-02__full002",
    )
    matches = provenance_matches(processed_root, "12345")
    matches += parse_slurm_output(
        f"12345|s2p__sub-02__full002|{workflow}\n", processed_root
    )

    assert len(deduplicate(matches)) == 1


def test_cli_prints_processed_session(tmp_path, capsys):
    processed_root = tmp_path / "processed" / "cohort2"
    write_submission(
        processed_root,
        "sub-02",
        "full002",
        job_id="12345",
        job_name="s2p__sub-02__full002",
    )

    result = main(["12345", "--processed-root", str(processed_root), "--no-slurm"])

    assert result == 0
    output = capsys.readouterr().out
    assert "Session:           sub-02/full002" in output
    assert str(processed_root / "sub-02" / "full002") in output

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from preprocessing.suite2p.full_session_pipeline.run_trace_extraction import (
    promote_output_bundle,
)


def test_promote_output_bundle_can_preserve_iscell_mtime(tmp_path: Path) -> None:
    output_dir = tmp_path / "plane0"
    expected_mtime = 1_700_000_000_123_456_789

    promote_output_bundle(
        output_dir,
        {"iscell.npy": np.asarray([[1, 0.9]])},
        preserved_mtimes_ns={"iscell.npy": expected_mtime},
    )

    assert (output_dir / "iscell.npy").stat().st_mtime_ns == expected_mtime


def test_promote_output_bundle_rejects_unknown_preserved_timestamp(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unknown outputs"):
        promote_output_bundle(
            tmp_path / "plane0",
            {"F.npy": np.ones((1, 1))},
            preserved_mtimes_ns={"iscell.npy": 1},
        )

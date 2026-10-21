from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import tifffile


SCRIPT = Path(__file__).parents[1] / "extract_tiff_windows.py"
SPEC = importlib.util.spec_from_file_location("extract_tiff_windows", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_pages(path: Path, values: list[int]) -> None:
    with tifffile.TiffWriter(path) as writer:
        for value in values:
            writer.write(
                np.full((4, 5), value, dtype=np.uint16),
                photometric="minisblack",
                contiguous=True,
                metadata=None,
            )


def test_select_windows() -> None:
    assert MODULE.select_windows(12, 4) == [
        ("start", 0, 4),
        ("middle", 4, 8),
        ("end", 8, 12),
    ]


def test_copy_page_interval_across_files(tmp_path: Path) -> None:
    first = tmp_path / "stack1.tif"
    second = tmp_path / "stack2.tif"
    write_pages(first, list(range(10)))
    write_pages(second, list(range(10, 24)))
    output = tmp_path / "window.tif"

    shape, dtype = MODULE.copy_page_interval(
        [first, second], [10, 14], 8, 16, output
    )

    movie = tifffile.imread(output)
    assert shape == (4, 5)
    assert dtype == "uint16"
    assert movie.shape == (8, 4, 5)
    assert movie[:, 0, 0].tolist() == list(range(8, 16))

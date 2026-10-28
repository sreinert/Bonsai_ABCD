from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from preprocessing.quality_control import compare_signal_quality
from preprocessing.quality_control import plot_high_snr_traces
from preprocessing.quality_control.signal_quality_cache import (
    get_session_results,
    select_top_cells,
)
from preprocessing.quality_control.signal_quality_metrics import (
    compute_channel2_metrics,
    load_percell_traces,
    summarise_traces,
)
from preprocessing.suite2p.full_session_pipeline.generate_trace_qc import generate_trace_qc


def make_plane(path: Path, *, neuropil_offset: float = 0.0) -> Path:
    path.mkdir(parents=True)
    frames = np.arange(120, dtype=float)
    F = np.vstack(
        [
            100 + np.sin(frames / 5) * 2 + (frames % 31 == 0) * 20,
            80 + np.cos(frames / 7) * 1.5 + (frames % 43 == 0) * 12,
            60 + np.sin(frames / 9),
        ]
    )
    Fneu = np.vstack(
        [10 + neuropil_offset + np.sin(frames / period) for period in (8, 10, 12)]
    )
    F2 = np.vstack([50 + np.sin(frames / 11), 45 + np.cos(frames / 13), 40 + np.sin(frames / 17)])
    stat = np.asarray(
        [
            {"ypix": np.array([i, i]), "xpix": np.array([i, i + 1]), "lam": np.array([0.5, 0.5])}
            for i in range(3)
        ],
        dtype=object,
    )
    np.save(path / "F.npy", F)
    np.save(path / "Fneu.npy", Fneu)
    np.save(path / "F_chan2.npy", F2)
    np.save(path / "Fneu_chan2.npy", np.ones_like(F2))
    np.save(path / "iscell.npy", np.asarray([[1, 0.95], [1, 0.8], [0, 0.2]]))
    np.save(path / "stat.npy", stat)
    np.save(
        path / "ops.npy",
        {
            "fs": 10.0,
            "tau": 0.4,
            "Ly": 5,
            "Lx": 5,
            "meanImg": np.arange(25, dtype=float).reshape(5, 5) + 10,
            "meanImg_chan2": np.arange(25, dtype=float).reshape(5, 5) + 20,
            "baseline": "constant_prctile",
            "win_baseline": 60.0,
            "sig_baseline": 10.0,
            "prctile_baseline": 8.0,
            "neucoeff": 0.7,
        },
    )
    return path


def test_raw_signal_metrics_are_independent_of_neuropil(tmp_path: Path) -> None:
    first = make_plane(tmp_path / "first", neuropil_offset=0)
    second = make_plane(tmp_path / "second", neuropil_offset=1000)

    first_metrics = summarise_traces(load_percell_traces(first))
    second_metrics = summarise_traces(load_percell_traces(second))

    for key in (
        "baseline_F_median",
        "median_skew_dff",
        "median_peak_dff_pct99",
        "median_snr",
        "frac_cells_snr_gt_thresh",
        "median_event_frac",
    ):
        assert np.allclose(first_metrics[key], second_metrics[key], equal_nan=True)


def test_channel2_metrics_use_mean_image_and_roi_masks(tmp_path: Path) -> None:
    plane = make_plane(tmp_path / "plane0")
    metrics = compute_channel2_metrics(plane)
    assert metrics["fov_chan2_mean"] == 32.0
    assert metrics["roi_chan2_median_brightness"] > 0
    assert "roi_chan2_F_time_median" in metrics


def test_top_cell_fallback_and_cache_invalidation(tmp_path: Path) -> None:
    traces = {
        "snr": np.asarray([2.0, 1.0, np.nan]),
        "n_frames": 10,
        "dff": np.ones((3, 10)),
        "noise_sd": np.ones(3),
        "cell_roi_idx": np.arange(3),
        "fs": 10.0,
    }
    top = select_top_cells(traces, n_cells=2, window_frames=6, snr_threshold=5)
    assert top["used_fallback"] is True
    assert np.array_equal(top["roi_idx"], [0, 1])

    plane = make_plane(tmp_path / "plane0")
    cache_dir = tmp_path / "cache"
    _, _, source = get_session_results(plane, "session", cache_dir=cache_dir)
    assert source == "compute"
    _, _, source = get_session_results(plane, "session", cache_dir=cache_dir)
    assert source == "cache"
    F = np.load(plane / "F.npy")
    np.save(plane / "F.npy", F + 1)
    _, _, source = get_session_results(plane, "session", cache_dir=cache_dir)
    assert source == "compute"


def test_package_api_and_direct_cli_help(tmp_path: Path) -> None:
    plane = make_plane(tmp_path / "plane0")
    metrics, top, source = compare_signal_quality.evaluate_session(
        plane, "session", "mouse", cache_dir=tmp_path / "cache"
    )
    assert metrics["session"] == "session"
    assert "median_snr" in metrics
    assert "snr" in top
    assert source == "compute"

    for module in (compare_signal_quality, plot_high_snr_traces):
        completed = subprocess.run(
            [sys.executable, str(Path(module.__file__)), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "YAML settings file" in completed.stdout


def test_integrated_trace_qc_outputs(tmp_path: Path) -> None:
    plane = make_plane(tmp_path / "run" / "suite2p" / "plane0")
    output = tmp_path / "qc" / "traces"

    provenance = generate_trace_qc(plane, output, n_cells=2)

    for relative in (
        "signal_quality_metrics.png",
        "signal_quality_metrics.csv",
        "signal_quality_metrics.json",
        "high_snr_traces.png",
        "additional_two_channel/summary.png",
        "additional_two_channel/selected_cells_celltv.png",
        "additional_two_channel/selected_cells_safe_ratio.png",
        "additional_two_channel/trace_metrics.csv",
        "calculation.json",
        "index.html",
    ):
        assert (output / relative).is_file()
    saved = json.loads((output / "calculation.json").read_text(encoding="utf-8"))
    assert saved["raw_signal_quality"]["source_commit"] == provenance["raw_signal_quality"]["source_commit"]
    assert "Additional two-channel/CellTV diagnostics" in (output / "index.html").read_text(encoding="utf-8")

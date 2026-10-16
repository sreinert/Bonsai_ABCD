#!/usr/bin/env python3
"""Generate integrated raw-signal and supplementary two-channel trace QC."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PIPELINE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for import_root in (REPOSITORY_ROOT, PIPELINE_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from preprocessing.quality_control.compare_signal_quality import (  # noqa: E402
    evaluate_session,
    make_session_figure,
)
from preprocessing.quality_control.plot_high_snr_traces import plot_session  # noqa: E402
from preprocessing.quality_control.signal_quality_cache import (  # noqa: E402
    DEFAULT_PARAMS,
    DEFAULT_TRACE_PARAMS,
)
from two_channel_trace_qc import generate_two_channel_qc  # noqa: E402


SANDRA_ANALYSIS_COMMIT = "ec188a3df008cfdc220383120250b020f68fa3e9"
SIGNAL_COLOR = "#2a78d6"


def write_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def write_unavailable_trace_figure(path: Path, message: str) -> None:
    figure, axis = plt.subplots(figsize=(10, 3))
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def trace_report_html(trace_dir: Path, metrics: dict[str, Any]) -> str:
    metric_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in metrics.items()
        if key not in {"plane0_dir", "mouse", "session"}
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Suite2p trace QC</title><style>
body{{font:15px system-ui,sans-serif;max-width:1250px;margin:2rem auto;padding:0 1rem;color:#222}}
img{{max-width:100%;height:auto;border:1px solid #ddd}} table{{border-collapse:collapse}}
th,td{{padding:.35rem .6rem;border:1px solid #ccc;text-align:right}} th{{text-align:left}}
.note{{background:#eef6ff;padding:.8rem 1rem;border-left:4px solid #3974a8}}
</style></head><body><h1>Suite2p trace QC</h1>
<p><b>Input:</b> <code>{html.escape(str(trace_dir))}</code></p>
<div class="note"><b>Primary metrics:</b> Sandra-analysis raw-F definitions: eighth-percentile
baseline, frame-difference MAD noise, p99 peak SNR, skew and event fractions. These are deliberately
independent of neuropil correction. Corrected green/red/G/R and CellTV-style calculations are retained
separately below.</div>
<h2>Raw signal-quality metrics</h2><img src="signal_quality_metrics.png" alt="Raw signal-quality metrics">
<table>{metric_rows}</table>
<h2>Highest-SNR raw traces</h2><img src="high_snr_traces.png" alt="Highest-SNR traces">
<h2>Additional two-channel/CellTV diagnostics</h2>
<img src="additional_two_channel/summary.png" alt="Additional two-channel summary">
<h3>CellTV-style traces</h3><img src="additional_two_channel/selected_cells_celltv.png" alt="CellTV traces">
<h3>Normalized traces</h3><img src="additional_two_channel/selected_cells_dff.png" alt="Normalized traces">
<p><a href="signal_quality_metrics.csv">Raw session metrics CSV</a> ·
<a href="additional_two_channel/trace_metrics.csv">Additional per-ROI metrics CSV</a> ·
<a href="calculation.json">Calculation settings and provenance</a></p>
</body></html>"""


def generate_trace_qc(
    trace_dir: Path,
    output_dir: Path,
    *,
    n_cells: int = 6,
    neucoeff: float | None = None,
) -> dict[str, Any]:
    """Generate both QC families atomically and return the provenance payload."""
    if n_cells < 1:
        raise ValueError("n_cells must be positive")
    trace_dir = trace_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing QC output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        session = trace_dir.parents[1].name
        cache_dir = temporary / "signal_quality_cache"
        trace_params = {**DEFAULT_TRACE_PARAMS, "n_cells": max(n_cells, 8)}
        metrics, top, cache_source = evaluate_session(
            trace_dir,
            session,
            session,
            cache_dir=cache_dir,
            recompute=False,
            params=DEFAULT_PARAMS,
            trace_params=trace_params,
        )
        if "error" in metrics:
            raise RuntimeError(f"Sandra-style signal QC failed: {metrics['error']}")

        write_single_row_csv(temporary / "signal_quality_metrics.csv", metrics)
        (temporary / "signal_quality_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        figure = make_session_figure(metrics)
        figure.savefig(temporary / "signal_quality_metrics.png", dpi=200,
                       facecolor=figure.get_facecolor())
        plt.close(figure)
        trace_path = plot_session(
            session,
            session,
            top,
            SIGNAL_COLOR,
            DEFAULT_PARAMS["snr_threshold"],
            out_path=temporary / "high_snr_traces.png",
        )
        if trace_path is None:
            write_unavailable_trace_figure(
                temporary / "high_snr_traces.png", top.get("error", "No finite-SNR cells")
            )

        additional_settings = generate_two_channel_qc(
            trace_dir,
            temporary / "additional_two_channel",
            n_cells=n_cells,
            neucoeff_override=neucoeff,
        )
        provenance = {
            "raw_signal_quality": {
                "source_branch": "origin/Sandra_analysis",
                "source_commit": SANDRA_ANALYSIS_COMMIT,
                "cache_source": cache_source,
                "cache_dir": str(output_dir / "signal_quality_cache"),
                "params": DEFAULT_PARAMS,
                "trace_params": trace_params,
                "signal_definition": "raw F.npy; Fneu.npy is used only for raw F/Fneu correlation",
                "metrics": metrics,
            },
            "additional_two_channel": additional_settings,
        }
        (temporary / "calculation.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "index.html").write_text(
            trace_report_html(trace_dir, metrics), encoding="utf-8"
        )
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Default: TRACE_DIR/qc")
    parser.add_argument("--n-cells", type=int, default=6)
    parser.add_argument("--neucoeff", type=float, help="Default: ops['neucoeff'], or 0.7")
    args = parser.parse_args()
    output_dir = (args.output_dir or args.trace_dir / "qc").resolve()
    generate_trace_qc(
        args.trace_dir,
        output_dir,
        n_cells=args.n_cells,
        neucoeff=args.neucoeff,
    )
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()

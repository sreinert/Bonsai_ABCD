#!/usr/bin/env python3
"""Generate per-run QC, visual comparisons, and cross-mouse rankings."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import read_csv, remap_mounted_path, require_columns, write_csv


TASK_COLUMNS = {
    "task_id",
    "mouse_id",
    "session_id",
    "candidate_name",
    "registration_json",
    "run_dir",
}

# Direction: +1 means larger is better, -1 means smaller is better.
METRIC_DIRECTIONS = {
    "corr_median": 1.0,
    "corr_p05": 1.0,
    "sharpness": 1.0,
    "bad_frame_fraction": -1.0,
    "shift_jitter_median": -1.0,
    "boundary_hit_fraction": -1.0,
    "reg_pc_residual_mean": -1.0,
}
METRIC_WEIGHTS = {
    "corr_median": 0.25,
    "corr_p05": 0.15,
    "sharpness": 0.15,
    "bad_frame_fraction": 0.15,
    "shift_jitter_median": 0.10,
    "boundary_hit_fraction": 0.10,
    "reg_pc_residual_mean": 0.10,
}


def array_from(*mappings: dict[str, Any], key: str) -> np.ndarray:
    for mapping in mappings:
        value = mapping.get(key)
        if value is not None:
            return np.asarray(value)
    return np.asarray([])


def finite(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values, dtype=float).ravel()
    return flattened[np.isfinite(flattened)]


def find_ops(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("suite2p/plane*/ops.npy"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one plane ops.npy below {run_dir}, found {len(matches)}"
        )
    return matches[0]


def load_ops(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ops = np.load(path, allow_pickle=True).item()
    nested = ops.get("reg_outputs")
    outputs = nested if isinstance(nested, dict) else ops
    return ops, outputs


def normalized_sharpness(image: np.ndarray) -> float:
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not np.isfinite(image).any():
        return math.nan
    low, high = np.nanpercentile(image, [1, 99])
    if high <= low:
        return math.nan
    normalized = np.clip((image - low) / (high - low), 0, 1)
    gradient_y, gradient_x = np.gradient(normalized)
    return float(np.nanmean(gradient_x * gradient_x + gradient_y * gradient_y))


def registration_settings(ops: dict[str, Any]) -> dict[str, Any]:
    settings = ops.get("settings")
    if isinstance(settings, dict) and isinstance(settings.get("registration"), dict):
        return settings["registration"]
    return ops


def compute_metrics(ops: dict[str, Any], outputs: dict[str, Any]) -> dict[str, float]:
    yoff = finite(array_from(outputs, ops, key="yoff"))
    xoff = finite(array_from(outputs, ops, key="xoff"))
    corr = finite(array_from(outputs, ops, key="corrXY"))
    bad = array_from(outputs, ops, key="badframes").astype(bool).ravel()
    mean_image = array_from(outputs, ops, key="meanImg")
    nshift = min(yoff.size, xoff.size)
    displacement = np.hypot(yoff[:nshift], xoff[:nshift]) if nshift else np.asarray([])
    jitter = (
        np.hypot(np.diff(yoff[:nshift]), np.diff(xoff[:nshift]))
        if nshift > 1
        else np.asarray([])
    )

    ly = int(ops.get("Ly", mean_image.shape[0] if mean_image.ndim == 2 else 0))
    lx = int(ops.get("Lx", mean_image.shape[1] if mean_image.ndim == 2 else 0))
    maxregshift = float(registration_settings(ops).get("maxregshift", 0.1))
    search_limit = maxregshift * min(ly, lx) if min(ly, lx) else math.nan
    boundary_fraction = (
        float(
            np.mean(
                np.maximum(np.abs(yoff[:nshift]), np.abs(xoff[:nshift]))
                >= 0.95 * search_limit
            )
        )
        if nshift and np.isfinite(search_limit) and search_limit > 0
        else math.nan
    )

    yoff1 = finite(array_from(outputs, ops, key="yoff1"))
    xoff1 = finite(array_from(outputs, ops, key="xoff1"))
    nonrigid_size = min(yoff1.size, xoff1.size)
    nonrigid_p95 = (
        float(np.percentile(np.hypot(yoff1[:nonrigid_size], xoff1[:nonrigid_size]), 95))
        if nonrigid_size
        else math.nan
    )

    reg_dx = np.asarray(array_from(outputs, ops, key="regDX"), dtype=float)
    if reg_dx.ndim == 2 and reg_dx.shape[1] >= 4:
        residual_values = finite(reg_dx[:, 3])
    else:
        residual_values = finite(reg_dx)

    return {
        "n_frames": float(max(yoff.size, corr.size, bad.size)),
        "corr_median": float(np.median(corr)) if corr.size else math.nan,
        "corr_p05": float(np.percentile(corr, 5)) if corr.size else math.nan,
        "sharpness": normalized_sharpness(mean_image),
        "bad_frame_fraction": float(np.mean(bad)) if bad.size else math.nan,
        "shift_p95": float(np.percentile(displacement, 95)) if displacement.size else math.nan,
        "shift_jitter_median": float(np.median(jitter)) if jitter.size else math.nan,
        "boundary_hit_fraction": boundary_fraction,
        "nonrigid_shift_p95": nonrigid_p95,
        "reg_pc_residual_mean": (
            float(np.mean(residual_values)) if residual_values.size else math.nan
        ),
    }


def show_image(axis: plt.Axes, image: np.ndarray, title: str) -> None:
    image = np.asarray(image)
    if image.ndim != 2 or not image.size:
        axis.text(0.5, 0.5, "not available", ha="center", va="center")
    else:
        low, high = np.nanpercentile(image, [1, 99.5])
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
    axis.set_title(title)
    axis.set_axis_off()


def make_qc_plot(
    task: dict[str, str], ops: dict[str, Any], outputs: dict[str, Any], metrics: dict[str, float]
) -> Path:
    run_dir = remap_mounted_path(Path(task["run_dir"]), must_exist=True)
    output_path = run_dir / "qc.png"
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    show_image(axes[0, 0], array_from(outputs, ops, key="refImg"), "Reference image")
    show_image(axes[0, 1], array_from(outputs, ops, key="meanImg"), "Registered mean image")

    corr = array_from(outputs, ops, key="corrXY").ravel()
    axes[0, 2].plot(corr, linewidth=0.6)
    axes[0, 2].set(title="Frame-to-reference correlation", xlabel="frame", ylabel="corrXY")

    yoff = array_from(outputs, ops, key="yoff").ravel()
    xoff = array_from(outputs, ops, key="xoff").ravel()
    axes[1, 0].plot(yoff, linewidth=0.6, label="y")
    axes[1, 0].plot(xoff, linewidth=0.6, label="x")
    axes[1, 0].set(title="Rigid offsets", xlabel="frame", ylabel="pixels")
    axes[1, 0].legend()

    nshift = min(yoff.size, xoff.size)
    if nshift:
        axes[1, 1].hist(np.hypot(yoff[:nshift], xoff[:nshift]), bins=40)
    axes[1, 1].set(title="Rigid displacement", xlabel="pixels", ylabel="frames")

    parameters = json.loads(task["registration_json"])
    lines = [
        f"{task['mouse_id']} / {task['session_id']}",
        f"candidate: {task['candidate_name']}",
        "",
        *[f"{key}: {value}" for key, value in sorted(parameters.items())],
        "",
        *[
            f"{key}: {value:.5g}" if np.isfinite(value) else f"{key}: n/a"
            for key, value in metrics.items()
        ],
    ]
    axes[1, 2].text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8)
    axes[1, 2].set_axis_off()
    figure.suptitle("Suite2p registration QC", fontsize=14)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def find_registered_binary(ops_path: Path, ops: dict[str, Any]) -> Path | None:
    db = ops.get("db") if isinstance(ops.get("db"), dict) else {}
    candidates = []
    for mapping in (ops, db):
        for key in ("reg_file", "reg_file_chan2"):
            if mapping.get(key):
                candidates.append(Path(str(mapping[key])))
    candidates.extend([ops_path.parent / "data.bin", ops_path.parent / "data_chan2.bin"])
    return next((path for path in candidates if path.is_file()), None)


def make_registered_montage(
    ops_path: Path, ops: dict[str, Any], outputs: dict[str, Any]
) -> Path | None:
    binary = find_registered_binary(ops_path, ops)
    mean_image = array_from(outputs, ops, key="meanImg")
    if mean_image.ndim != 2 or binary is None:
        return None
    ly, lx = mean_image.shape
    bytes_per_frame = np.dtype(np.int16).itemsize * ly * lx
    nframes = binary.stat().st_size // bytes_per_frame
    if nframes <= 0:
        return None
    frames = np.memmap(binary, dtype=np.int16, mode="r", shape=(nframes, ly, lx))
    indices = np.unique(np.linspace(0, nframes - 1, min(12, nframes), dtype=int))
    low, high = np.percentile(mean_image, [1, 99.5])
    figure, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    for axis in axes.ravel():
        axis.set_axis_off()
    for axis, frame_index in zip(axes.ravel(), indices):
        axis.imshow(frames[frame_index], cmap="gray", vmin=low, vmax=high)
        axis.set_title(f"frame {frame_index}")
    output_path = Path(ops_path).parents[2] / "registered_frames.png"
    figure.suptitle("Registered frames sampled across the stack")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    del frames
    return output_path


def robust_z(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return result
    center = np.median(values[valid])
    mad = np.median(np.abs(values[valid] - center))
    result[valid] = 0.0 if mad <= 1e-12 else (values[valid] - center) / (1.4826 * mad)
    return np.clip(result, -5, 5)


def add_session_scores(results: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault((result["mouse_id"], result["session_id"]), []).append(result)
    for group in groups.values():
        total = np.zeros(len(group), dtype=float)
        weights = np.zeros(len(group), dtype=float)
        for metric, direction in METRIC_DIRECTIONS.items():
            values = np.asarray([row.get(metric, math.nan) for row in group], dtype=float)
            zscores = robust_z(values) * direction
            weight = METRIC_WEIGHTS[metric]
            valid = np.isfinite(zscores)
            total[valid] += weight * zscores[valid]
            weights[valid] += weight
        for index, result in enumerate(group):
            result["within_session_score"] = (
                float(total[index] / weights[index]) if weights[index] else math.nan
            )


def summarize_candidates(
    results: list[dict[str, Any]], tasks: list[dict[str, str]]
) -> list[dict[str, Any]]:
    expected_sessions = {(row["mouse_id"], row["session_id"]) for row in tasks}
    candidate_names = sorted({row["candidate_name"] for row in tasks})
    summaries: list[dict[str, Any]] = []
    for candidate in candidate_names:
        selected = [row for row in results if row["candidate_name"] == candidate]
        mouse_scores: dict[str, list[float]] = {}
        for row in selected:
            score = float(row["within_session_score"])
            if np.isfinite(score):
                mouse_scores.setdefault(row["mouse_id"], []).append(score)
        per_mouse = np.asarray([np.median(value) for value in mouse_scores.values()])
        median = float(np.median(per_mouse)) if per_mouse.size else math.nan
        worst = float(np.min(per_mouse)) if per_mouse.size else math.nan
        robust = 0.7 * median + 0.3 * worst if per_mouse.size else math.nan
        completed_sessions = {(row["mouse_id"], row["session_id"]) for row in selected}
        summaries.append(
            {
                "candidate_name": candidate,
                "robust_score": robust,
                "median_mouse_score": median,
                "worst_mouse_score": worst,
                "mice_completed": len(mouse_scores),
                "sessions_completed": len(completed_sessions),
                "sessions_expected": len(expected_sessions),
                "complete": completed_sessions == expected_sessions,
            }
        )
    summaries.sort(
        key=lambda row: (
            bool(row["complete"]),
            float(row["robust_score"]) if np.isfinite(row["robust_score"]) else -math.inf,
        ),
        reverse=True,
    )
    for rank, row in enumerate(summaries, start=1):
        row["rank"] = rank
    return summaries


def format_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return "n/a" if not np.isfinite(value) else f"{value:.4g}"
    return html.escape(str(value))


def relative_link(path: str | Path, report_dir: Path) -> str:
    return os.path.relpath(Path(path), report_dir)


def write_report(
    path: Path, summaries: list[dict[str, Any]], results: list[dict[str, Any]]
) -> None:
    headers = list(summaries[0]) if summaries else []
    summary_rows = "".join(
        "<tr>" + "".join(f"<td>{format_value(row[key])}</td>" for key in headers) + "</tr>"
        for row in summaries
    )
    cards = []
    for row in sorted(
        results,
        key=lambda value: (value["mouse_id"], value["session_id"], value["candidate_name"]),
    ):
        qc_link = html.escape(relative_link(row["qc_path"], path.parent))
        montage = row.get("montage_path")
        montage_link = (
            f' · <a href="{html.escape(relative_link(montage, path.parent))}">registered frames</a>'
            if montage
            else ""
        )
        cards.append(
            "<article class='card'>"
            f"<h3>{html.escape(row['mouse_id'])} · {html.escape(row['session_id'])}</h3>"
            f"<p>{html.escape(row['candidate_name'])} · score "
            f"{format_value(row['within_session_score'])} · "
            f"<a href='{qc_link}'>full QC</a>{montage_link}</p>"
            f"<a href='{qc_link}'><img src='{qc_link}' alt='registration QC'></a>"
            f"<small>{html.escape(row['run_dir'])}</small>"
            "</article>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Suite2p registration QC</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1600px}}
table{{border-collapse:collapse}}th,td{{border:1px solid #ccc;padding:.4rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:1rem}}
.card{{border:1px solid #ccc;padding:1rem}}.card img{{width:100%;height:auto}}small{{display:block;overflow-wrap:anywhere}}
</style></head><body>
<h1>Suite2p registration QC</h1>
<p>Scores are robustly normalised within each session, aggregated within mouse, then combined as 70% median-mouse and 30% worst-mouse performance. Incomplete candidates rank below complete candidates. Always inspect the images and traces before choosing.</p>
<h2>Cross-mouse ranking</h2>
<table><thead><tr>{''.join(f'<th>{html.escape(header)}</th>' for header in headers)}</tr></thead><tbody>{summary_rows}</tbody></table>
<h2>Run QC</h2><div class="grid">{''.join(cards)}</div>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--strict", action="store_true", help="Fail if any task is incomplete")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tasks = require_columns(read_csv(args.manifest), TASK_COLUMNS, "task manifest")
    qc_dir = args.manifest.parent / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for task in tasks:
        try:
            run_dir = remap_mounted_path(Path(task["run_dir"]), must_exist=True)
            status_path = run_dir / "status.json"
            if not status_path.is_file():
                raise RuntimeError("missing status.json")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("state") != "complete":
                raise RuntimeError(f"run state is {status.get('state', 'unknown')}")
            ops_path = find_ops(run_dir)
            ops, outputs = load_ops(ops_path)
            metrics = compute_metrics(ops, outputs)
            qc_path = make_qc_plot(task, ops, outputs, metrics)
            montage_path = make_registered_montage(ops_path, ops, outputs)
            results.append(
                {
                    "task_id": int(task["task_id"]),
                    "mouse_id": task["mouse_id"],
                    "session_id": task["session_id"],
                    "candidate_name": task["candidate_name"],
                    **metrics,
                    "run_dir": str(run_dir),
                    "qc_path": str(qc_path),
                    "montage_path": str(montage_path) if montage_path else "",
                }
            )
        except Exception as exc:
            errors.append(f"task {task.get('task_id', '?')}: {exc}")

    if not results:
        raise RuntimeError("No completed runs could be evaluated: " + "; ".join(errors))
    add_session_scores(results)
    summaries = summarize_candidates(results, tasks)
    write_csv(qc_dir / "run_metrics.csv", results, overwrite=args.overwrite)
    write_csv(qc_dir / "candidate_summary.csv", summaries, overwrite=args.overwrite)
    write_report(qc_dir / "index.html", summaries, results)
    if errors:
        error_path = qc_dir / "evaluation_errors.txt"
        error_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"Skipped {len(errors)} task(s); see {error_path}")
        if args.strict:
            raise RuntimeError("Some tasks were incomplete or invalid")
    print(f"Evaluated {len(results)} runs; open {qc_dir / 'index.html'}")


if __name__ == "__main__":
    main()

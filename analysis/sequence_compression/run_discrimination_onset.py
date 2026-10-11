"""Batch analysis of selective-licking onset for even, odd, and shuffle.

The first run parses raw sessions and creates one appendable NPZ cache per
mouse. Later runs load the derived arrays and only parse new or stale sessions.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEQUENCE_COMPRESSION_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SEQUENCE_COMPRESSION_ROOT.parents[1]
for import_path in (REPOSITORY_ROOT, SEQUENCE_COMPRESSION_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from analysis.sequence_compression import learning_transition
from session_functions import io, session


DEFAULT_BEHAV_ROOT = Path(
    "/Volumes/mrsic_flogel/public/projects/"
    "AtAp_20260119_SequenceCompression/rawdata/cohort2"
)
DEFAULT_PROCESSED_ROOT = Path(
    "/Volumes/mrsic_flogel/public/projects/"
    "AtAp_20260119_SequenceCompression/processed/cohort2"
)
SEQUENCE_DATES = {
    "even": [
        "260916", "260917", "260919", "260921",
        "260922", "260923", "260924", "260925",
    ],
    "odd": ["260928", "260929", "260930", "261001"],
    "shuffle": [
        "261005", "261006", "261007",
        "261008", "261009", "261010",
    ],
}


def load_mouse_sequence(
    mouse: str,
    sequence: str,
    *,
    behav_root: Path,
    processed_root: Path,
    bins: int,
    stage: str = "t5",
    refresh_cache: bool = False,
) -> dict[str, object]:
    """Load cached records and calculate only missing or stale sessions."""

    cache_path = (
        processed_root / f"sub-{mouse}" / "analysis" / "lick_rate_cache.npz"
    )
    cache = learning_transition.load_lick_cache(cache_path)
    cache_updated = False
    records = []
    session_ids = []
    dates = []
    lap_counts = []
    seen_session_ids: set[str] = set()

    for date in SEQUENCE_DATES[sequence]:
        base_path = io.find_base_path(
            mouse, date, behav_root, prefer_concatenated=True
        )
        session_id = base_path.name
        if session_id in seen_session_ids:
            print(
                f"[DUPLICATE] sub-{mouse} {sequence} {session_id}; "
                "keeping its first date"
            )
            continue
        seen_session_ids.add(session_id)

        record = learning_transition.get_cached_record(
            cache,
            sequence=sequence,
            session_id=session_id,
            source_path=base_path,
            bins=bins,
        )
        if refresh_cache:
            record = None
        if record is None:
            print(f"[CACHE MISS] sub-{mouse} {sequence} {session_id}")
            settings, rig_settings = io.load_settings(base_path)
            dataframe = io.load_data(base_path)
            current_session = session.ABCD_Session(
                settings,
                rig_settings,
                dataframe,
                base_path,
                mouse,
                stage,
                session_id,
            )
            parsed = current_session.analyse_session_pre7_behav(plot=False)
            entries, exits = current_session.get_lm_entry_exit(parsed)
            record = learning_transition.build_session_lick_record(
                parsed, entries, exits, bins=bins
            )
            record.update({"mouse": mouse, "date": date})
            learning_transition.put_cached_record(
                cache,
                sequence=sequence,
                session_id=session_id,
                source_path=base_path,
                record=record,
            )
            cache_updated = True
        else:
            print(f"[CACHE HIT] sub-{mouse} {sequence} {session_id}")

        lick_map = record["full_lick_rate"]
        if lick_map.size == 0:
            print(f"[SKIP] No complete laps in {session_id}")
            continue
        records.append(record)
        session_ids.append(session_id)
        dates.append(date)
        lap_counts.append(int(lick_map.shape[0]))

    if cache_updated:
        learning_transition.save_lick_cache(cache_path, cache)
        print(f"[CACHE SAVED] {cache_path}")
    if not records:
        raise RuntimeError(
            f"No complete sessions for sub-{mouse}, sequence {sequence}"
        )

    lick_map = np.concatenate(
        [record["pre_reward_lick_rate"] for record in records], axis=0
    )
    full_lick_map = np.concatenate(
        [record["full_lick_rate"] for record in records], axis=0
    )
    full_binary_lick = np.concatenate(
        [record["full_binary_lick"] for record in records], axis=0
    )
    roles = np.concatenate(
        [record["role_codes"] for record in records], axis=0
    )
    rewarded = np.concatenate(
        [record["rewarded_goal"] for record in records], axis=0
    )
    category_rates = learning_transition.calculate_lap_category_rates(
        full_lick_map, roles, rewarded
    )
    binary_category_rates = learning_transition.calculate_lap_binary_rates(
        full_binary_lick, roles, rewarded
    )

    return {
        "pre_reward_lick_rate": lick_map,
        "full_lick_rate": full_lick_map,
        "full_binary_lick": full_binary_lick,
        "role_codes": roles,
        "rewarded_goal": rewarded,
        "category_rates": category_rates,
        "binary_category_rates": binary_category_rates,
        "session_ids": session_ids,
        "dates": dates,
        "lap_counts": lap_counts,
    }


def save_diagnostic_plot(
    mouse: str,
    sequence: str,
    data: dict[str, object],
    rates: dict[str, np.ndarray],
    result: dict[str, object],
    legacy: dict[str, object],
    *,
    min_effect: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rates_fig, rates_axis = plt.subplots(
        figsize=(8, 3.5), constrained_layout=True
    )
    rates_axis.plot(
        result["rolling_goal"], color="#1E2985", label="goal"
    )
    rates_axis.plot(
        result["rolling_non_goal"], color="#9D1DA3", label="non-goal"
    )
    rates_axis.plot(
        result["rolling_test"], color="darkorange", label="test"
    )
    rates_axis.set_ylabel("Rolling fraction licked")
    rates_axis.set_xlabel("Cumulative lap")
    rates_axis.legend(
        frameon=False,
        handlelength=0,
        handletextpad=0,
        labelcolor="linecolor",
    )

    selectivity_fig, selectivity_axis = plt.subplots(
        figsize=(8, 3.5), constrained_layout=True
    )
    selectivity_axis.plot(
        result["goal_vs_nongoal"],
        color="#9D1DA3",
        label="goal − non-goal",
    )
    selectivity_axis.plot(
        result["goal_vs_test"],
        color="darkorange",
        label="goal − test",
    )
    selectivity_axis.axhline(
        min_effect, color="black", linestyle="--", linewidth=1
    )
    selectivity_axis.set_ylabel("Rolling selectivity")
    selectivity_axis.set_xlabel("Cumulative lap")
    selectivity_axis.legend(
        frameon=False,
        handlelength=0,
        handletextpad=0,
        labelcolor="linecolor",
    )

    axes = (rates_axis, selectivity_axis)
    for boundary in np.cumsum(data["lap_counts"])[:-1]:
        for axis in axes:
            axis.axvline(boundary, color="grey", linestyle=":", alpha=0.5)
    if result["onset_index"] is not None:
        for axis in axes:
            axis.axvline(
                result["onset_index"], color="firebrick", linewidth=2
            )
    if legacy["discrimination_onset_index"] is not None:
        for axis in axes:
            axis.axvline(
                legacy["discrimination_onset_index"],
                color="black",
                linestyle="--",
            )

    for axis in axes:
        axis.set_title(f"sub-{mouse}, {sequence}")
        axis.spines[["right", "top"]].set_visible(False)

    rates_fig.savefig(output_dir / "rolling_fraction_licked.png", dpi=300)
    selectivity_fig.savefig(
        output_dir / "rolling_selectivity.png", dpi=300
    )
    plt.close(rates_fig)
    plt.close(selectivity_fig)


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_sequence_comparison(
    rows: list[dict[str, object]], path: Path
) -> None:
    sequence_order = list(SEQUENCE_DATES)
    mice = sorted({str(row["mouse"]) for row in rows})
    mouse_colors = dict(zip(mice, ("cornflowerblue", "#D97706")))
    fig, axis = plt.subplots(figsize=(3.4, 4))
    for mouse in mice:
        mouse_rows = {row["sequence"]: row for row in rows if row["mouse"] == mouse}
        values = [
            mouse_rows[sequence]["onset_lap"]
            if mouse_rows[sequence]["onset_lap"] is not None
            else np.nan
            for sequence in sequence_order
        ]
        axis.plot(
            sequence_order,
            values,
            marker="o",
            color=mouse_colors[mouse],
            label=f"sub-{mouse}",
        )
    axis.set_ylabel("Discrimination onset (cumulative lap)")
    axis.spines[["right", "top"]].set_visible(False)
    legend = axis.legend(
        frameon=False,
        handlelength=0,
        handletextpad=0,
        labelcolor="linecolor",
    )
    for handle in legend.legend_handles:
        handle.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mice", nargs="+", default=["05", "08"])
    parser.add_argument(
        "--sequences", nargs="+", choices=SEQUENCE_DATES, default=list(SEQUENCE_DATES)
    )
    parser.add_argument("--behav-root", type=Path, default=DEFAULT_BEHAV_ROOT)
    parser.add_argument(
        "--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT
    )
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument(
        "--min-effect",
        type=float,
        default=0.20,
        help="Minimum goal-minus-comparator lick-probability difference.",
    )
    parser.add_argument("--min-goal-fraction", type=float, default=0.25)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recalculate requested sessions even when cached records exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    comparison_dir = args.processed_root / "analysis" / "discrimination_onset"

    for sequence in args.sequences:
        for mouse in args.mice:
            data = load_mouse_sequence(
                mouse,
                sequence,
                behav_root=args.behav_root,
                processed_root=args.processed_root,
                bins=args.bins,
                refresh_cache=args.refresh_cache,
            )
            rates = data["binary_category_rates"]
            result = learning_transition.detect_discrimination_onset(
                rates,
                window=args.window,
                persistence=args.persistence,
                min_effect=args.min_effect,
                min_goal_fraction=args.min_goal_fraction,
            )
            legacy = learning_transition.detect_legacy_phase_boundaries(rates)
            sensitivity = {
                current_window: learning_transition.detect_discrimination_onset(
                    rates,
                    window=current_window,
                    persistence=args.persistence,
                    min_effect=args.min_effect,
                    min_goal_fraction=args.min_goal_fraction,
                )["onset_lap"]
                for current_window in (8, 10, 12)
            }
            rows.append(
                {
                    "mouse": mouse,
                    "sequence": sequence,
                    "readout": "full_landmark_binary",
                    "n_sessions": len(data["session_ids"]),
                    "n_laps": len(rates["goal"]),
                    "window": args.window,
                    "persistence": args.persistence,
                    "min_effect": args.min_effect,
                    "min_goal_fraction": args.min_goal_fraction,
                    "onset_lap": result["onset_lap"],
                    "legacy_exploration_end_lap": legacy[
                        "exploration_end_lap"
                    ],
                    "legacy_discrimination_onset_lap": legacy[
                        "discrimination_onset_lap"
                    ],
                    "onset_window_8": sensitivity[8],
                    "onset_window_10": sensitivity[10],
                    "onset_window_12": sensitivity[12],
                }
            )
            save_diagnostic_plot(
                mouse,
                sequence,
                data,
                rates,
                result,
                legacy,
                min_effect=args.min_effect,
                output_dir=(
                    args.processed_root
                    / f"sub-{mouse}"
                    / "analysis"
                    / f"seq-{sequence}"
                ),
            )

    write_summary(rows, comparison_dir / "discrimination_onset_summary.csv")
    if set(args.sequences) == set(SEQUENCE_DATES):
        plot_sequence_comparison(
            rows, comparison_dir / "sequence_onset_comparison.png"
        )
    print(f"Saved summary to {comparison_dir}")


if __name__ == "__main__":
    main()

"""Project-specific loading for cross-protocol A/B population decoding."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from analysis.sequence_compression.sc_neural_data.cross_session_decoding import (
    make_landmark_population,
    make_trial_population,
)
from analysis.sequence_compression.session_functions import io
from analysis.sequence_compression.session_functions import session


def _match_event_indices_to_visits(event_positions, landmark_centres):
    position_error = np.abs(
        np.asarray(event_positions, dtype=float)[:, None]
        - np.asarray(landmark_centres, dtype=float)[None, :]
    )
    event_visit = np.argmin(position_error, axis=1)
    matched = position_error[np.arange(len(event_visit)), event_visit] <= 1.0
    if np.any(~matched):
        warnings.warn(
            f"Ignoring {np.sum(~matched)} landmark event(s) without a "
            "position-matched completed visit",
            RuntimeWarning,
        )
    event_visit = event_visit[matched]
    if np.unique(event_visit).size != event_visit.size:
        raise ValueError("Multiple landmark events matched the same parsed visit")
    return event_visit, matched


def _match_events_to_visits(
    event_positions, event_labels, landmark_centres, *, return_matches=False
):
    event_visit, matched = _match_event_indices_to_visits(
        event_positions, landmark_centres
    )
    event_labels = np.asarray(event_labels)[matched]
    visit_labels = np.full(len(landmark_centres), "", dtype="<U2")
    visit_labels[event_visit] = event_labels
    if np.any(visit_labels == ""):
        missing = np.flatnonzero(visit_labels == "")
        raise ValueError(f"Unlabelled landmark visits: {missing.tolist()}")
    if return_matches:
        return visit_labels, event_visit, matched
    return visit_labels


def label_landmark_visits(
    sess_dataframe, landmark_centres, protocol, *, return_event_metadata=False
):
    """Return fine and pooled A/B labels for an ABAB, ABB, or AAB session."""
    event_mask = (
        sess_dataframe["LM_Count"].notna()
        & (sess_dataframe["LM_Count"] >= 0)
    )
    events = sess_dataframe.loc[
        event_mask, ["LM_Position", "Sequence_Position"]
    ].dropna().copy()
    sequence_position = events["Sequence_Position"].to_numpy(float)
    if not np.all(np.isin(sequence_position, [-1, 0])):
        raise ValueError("A/B labelling expects Sequence_Position -1/0 only")

    if protocol == "ABAB":
        event_labels = np.where(sequence_position == 0, "A", "B")
    elif protocol == "ABB":
        event_phase = np.arange(len(events)) % 3
        rewarded_counts = np.bincount(
            event_phase[sequence_position == 0], minlength=3
        )
        if rewarded_counts.max() == 0 or np.sum(
            rewarded_counts == rewarded_counts.max()
        ) != 1:
            raise ValueError(
                f"Cannot uniquely infer ABB A phase: {rewarded_counts.tolist()}"
            )
        a_phase = int(np.argmax(rewarded_counts))
        phase_labels = {
            a_phase: "A",
            (a_phase + 1) % 3: "B1",
            (a_phase + 2) % 3: "B2",
        }
        event_labels = np.asarray(
            [phase_labels[int(phase)] for phase in event_phase]
        )
        if np.any((sequence_position == 0) & (event_labels != "A")):
            raise ValueError("A rewarded event falls outside the inferred ABB A phase")
    elif protocol == "AAB":
        event_phase = np.arange(len(events)) % 3
        b_counts = np.bincount(
            event_phase[sequence_position == -1], minlength=3
        )
        if b_counts.max() == 0 or np.sum(b_counts == b_counts.max()) != 1:
            raise ValueError(
                f"Cannot uniquely infer AAB B phase: {b_counts.tolist()}"
            )
        b_phase = int(np.argmax(b_counts))
        phase_labels = {
            (b_phase + 1) % 3: "A1",
            (b_phase + 2) % 3: "A2",
            b_phase: "B",
        }
        event_labels = np.asarray(
            [phase_labels[int(phase)] for phase in event_phase]
        )
        if np.any((sequence_position == 0) & (event_labels == "B")):
            raise ValueError("An AAB rewarded event falls in the inferred B phase")
    else:
        raise ValueError("protocol must be 'ABAB', 'ABB', or 'AAB'")

    event_positions = events["LM_Position"].to_numpy(float)
    matched_result = _match_events_to_visits(
        event_positions,
        event_labels,
        landmark_centres,
        return_matches=return_event_metadata,
    )
    if return_event_metadata:
        fine_labels, event_visit, matched = matched_result
    else:
        fine_labels = matched_result
    pooled_labels = np.where(
        np.char.startswith(fine_labels.astype(str), "A"), "A", "B"
    )
    if not return_event_metadata:
        return fine_labels, pooled_labels

    metadata = {
        "event_visit": event_visit,
        "sequence_position": sequence_position[matched],
    }
    return fine_labels, pooled_labels, metadata


def find_abb_omission_triplet_visits(
    fine_labels, event_visit, sequence_position
):
    """Return ABB visit indices belonging to omitted-A triplets.

    An omitted A is an A-phase landmark whose ``Sequence_Position`` is -1.
    The omitted A and its immediately following B1 and B2 are excluded as one
    triplet. The label checks prevent an incomplete or shifted sequence from
    silently removing visits from the next triplet.
    """
    fine_labels = np.asarray(fine_labels, dtype=str)
    event_visit = np.asarray(event_visit, dtype=int)
    sequence_position = np.asarray(sequence_position, dtype=float)
    if event_visit.shape != sequence_position.shape:
        raise ValueError("Event visits and sequence positions must align")

    omitted_a_visits = event_visit[
        (sequence_position == -1) & (fine_labels[event_visit] == "A")
    ]
    excluded = []
    for omitted_a in omitted_a_visits:
        triplet = np.arange(omitted_a, omitted_a + 3, dtype=int)
        if triplet[-1] >= len(fine_labels):
            raise ValueError(
                f"Omitted A visit {omitted_a} lacks a complete following "
                "B1/B2 pair"
            )
        labels = fine_labels[triplet].tolist()
        if labels != ["A", "B1", "B2"]:
            raise ValueError(
                f"Omitted A visit {omitted_a} is not followed by B1/B2: "
                f"{labels}"
            )
        excluded.extend(triplet.tolist())
    return np.asarray(sorted(set(excluded)), dtype=int), omitted_a_visits


def prepare_protocol_session(
    session_id,
    alignment_row,
    *,
    mouse,
    rawdata_root,
    behaviour_aligned_root,
    dff_filename="dff.npy",
):
    """Load and parse one session without committing to a feature window."""
    if "-abab-" in session_id:
        protocol = "ABAB"
    elif "-abb-" in session_id:
        protocol = "ABB"
    elif "-aab-" in session_id:
        protocol = "AAB"
    else:
        raise ValueError(f"Unsupported session protocol: {session_id}")

    npz_path = (
        behaviour_aligned_root / f"sub-{mouse}" / session_id
        / "behaviour_data.npz"
    )
    with np.load(npz_path) as archive:
        position = np.asarray(archive["position"], dtype=float)
        rewards = np.asarray(archive["rewards"], dtype=float)
        licks = np.asarray(archive["licks"], dtype=float)
        speed = np.asarray(archive["speed"], dtype=float)
    dff_path = (
        behaviour_aligned_root / f"sub-{mouse}" / session_id
        / "funcimg" / dff_filename
    )
    dff_full = np.load(dff_path, mmap_mode="r")
    already_valid_frame_aligned = dff_full.shape[1] == position.size
    if already_valid_frame_aligned:
        # Some CellTV exports have already been restricted to valid frames.
        dff = dff_full
    else:
        valid_frames_path = (
            behaviour_aligned_root / f"sub-{mouse}" / session_id
            / "valid_frames.npz"
        )
        with np.load(valid_frames_path) as archive:
            valid_frames = np.asarray(archive["valid_frames"], dtype=int)
        if valid_frames.ndim != 1:
            raise ValueError(
                f"{session_id}: valid_frames must be one-dimensional"
            )
        if valid_frames.size and (
            valid_frames.min() < 0 or valid_frames.max() >= dff_full.shape[1]
        ):
            raise ValueError(f"{session_id}: invalid valid_frames indices")
        dff = dff_full[:, valid_frames]
    if dff.ndim != 2 or position.shape != (dff.shape[1],):
        raise ValueError(f"{session_id}: incompatible dF/F and position shapes")

    session_path = rawdata_root / f"sub-{mouse}" / session_id
    settings, rig_settings = io.load_settings(session_path)
    velocity_threshold = settings["velocityThreshold"]
    if velocity_threshold is None or not np.isfinite(velocity_threshold):
        raise ValueError(f"{session_id}: invalid velocityThreshold")
    dataframe = io.load_data(session_path)
    parsed = session.ABCD_Session(
        settings,
        rig_settings,
        dataframe,
        session_path,
        mouse,
        session_id,
        session_id,
    )
    structure = parsed.create_session_struct()
    structure.update(
        position=np.nan_to_num(position, nan=0.0),
        rewards=np.flatnonzero(rewards),
        mouse=mouse,
        stage=session_id,
        session_id=session_id,
        world=parsed.world,
        laps_needed=1,
    )
    structure = parsed.get_landmark_positions(structure, data="odour")
    structure = parsed.get_lap_idx(structure)
    structure = parsed.get_lms_visited(structure)
    entry_frames, exit_frames = parsed.get_lm_entry_exit(structure)
    entry_frames = np.asarray(entry_frames, dtype=int)
    exit_frames = np.asarray(exit_frames, dtype=int)
    landmark_centres = np.mean(structure["all_landmarks"], axis=1)
    fine_labels, pooled_labels, event_metadata = label_landmark_visits(
        dataframe,
        landmark_centres,
        protocol,
        return_event_metadata=True,
    )
    if not (
        len(fine_labels) == len(entry_frames) == len(exit_frames)
    ):
        raise ValueError(f"{session_id}: labels and landmark entries differ")

    source_visit_indices = np.arange(len(fine_labels), dtype=int)
    omission_triplet_visits = np.empty(0, dtype=int)
    omitted_a_visits = np.empty(0, dtype=int)
    if protocol == "ABB":
        omission_triplet_visits, omitted_a_visits = (
            find_abb_omission_triplet_visits(
                fine_labels,
                event_metadata["event_visit"],
                event_metadata["sequence_position"],
            )
        )
    keep_visit = np.ones(len(fine_labels), dtype=bool)
    keep_visit[omission_triplet_visits] = False
    source_visit_indices = source_visit_indices[keep_visit]
    entry_frames = entry_frames[keep_visit]
    exit_frames = exit_frames[keep_visit]
    fine_labels = fine_labels[keep_visit]
    pooled_labels = pooled_labels[keep_visit]

    if licks.shape != position.shape or speed.shape != position.shape:
        raise ValueError(f"{session_id}: incompatible lick/speed array shapes")
    any_lick_mask = licks > 0
    task_lick_mask = any_lick_mask & (speed < velocity_threshold)
    visit_was_licked = np.asarray(
        [
            np.any(task_lick_mask[entry:exit_ + 1])
            for entry, exit_ in zip(entry_frames, exit_frames)
        ],
        dtype=bool,
    )
    visit_had_any_lick = np.asarray(
        [
            np.any(any_lick_mask[entry:exit_ + 1])
            for entry, exit_ in zip(entry_frames, exit_frames)
        ],
        dtype=bool,
    )
    visit_mean_speed = np.asarray(
        [
            np.nanmean(speed[entry:exit_ + 1])
            for entry, exit_ in zip(entry_frames, exit_frames)
        ],
        dtype=float,
    )
    first_task_lick_frames = np.asarray(
        [
            (entry + hits[0] if len(hits) else -1)
            for entry, exit_ in zip(entry_frames, exit_frames)
            for hits in [np.flatnonzero(task_lick_mask[entry:exit_ + 1])]
        ],
        dtype=int,
    )
    first_any_lick_frames = np.asarray(
        [
            (entry + hits[0] if len(hits) else -1)
            for entry, exit_ in zip(entry_frames, exit_frames)
            for hits in [np.flatnonzero(any_lick_mask[entry:exit_ + 1])]
        ],
        dtype=int,
    )
    has_any_lick = first_any_lick_frames >= 0
    first_lick_below_speed_threshold = np.zeros(
        len(first_any_lick_frames), dtype=bool
    )
    if np.any(has_any_lick):
        first_lick_speeds = speed[first_any_lick_frames[has_any_lick]]
        if np.any(~np.isfinite(first_lick_speeds)):
            raise ValueError(
                f"{session_id}: non-finite speed at a first lick frame"
            )
        first_lick_below_speed_threshold[has_any_lick] = (
            first_lick_speeds < velocity_threshold
        )
    if alignment_row is None:
        local_roi_ids = np.arange(dff.shape[0], dtype=int)
        cluster_ids = local_roi_ids.copy()
    else:
        alignment_row = np.asarray(alignment_row, dtype=float)
        present = np.isfinite(alignment_row)
        cluster_ids = np.flatnonzero(present)
        local_roi_ids = alignment_row[present].astype(int)
    if local_roi_ids.size and (
        local_roi_ids.min() < 0 or local_roi_ids.max() >= dff.shape[0]
    ):
        raise IndexError(f"{session_id}: aligned ROI is outside dF/F array")

    return {
        "session_id": session_id,
        "protocol": protocol,
        "dff_filename": dff_filename,
        "already_valid_frame_aligned": already_valid_frame_aligned,
        "dff_shape": dff.shape,
        "selected_dff": np.asarray(dff[local_roi_ids]),
        "cluster_ids": cluster_ids,
        "entry_frames": entry_frames,
        "exit_frames": exit_frames,
        "fine_labels": fine_labels,
        "pooled_labels": pooled_labels,
        "source_visit_indices": source_visit_indices,
        "visit_was_licked": visit_was_licked,
        "visit_had_any_lick": visit_had_any_lick,
        "visit_mean_speed": visit_mean_speed,
        "first_task_lick_frames": first_task_lick_frames,
        "first_any_lick_frames": first_any_lick_frames,
        "first_lick_below_speed_threshold": (
            first_lick_below_speed_threshold
        ),
        "any_lick_mask": any_lick_mask,
        "task_lick_mask": task_lick_mask,
        "velocity_threshold": float(velocity_threshold),
        "omitted_a_visits": omitted_a_visits,
        "omission_triplet_visits": omission_triplet_visits,
    }


def build_protocol_population(
    prepared,
    *,
    frame_rate=45.0,
    feature_window="landmark",
    pre_entry_seconds=2.0,
    pre_lick_seconds=2.0,
    lick_search_pre_entry_seconds=0.0,
    lick_definition="task",
):
    """Build one feature population from an already parsed session.

    ``feature_window='landmark'`` uses each neuron's mean session-z-scored
    activity from landmark entry through exit. ``feature_window='pre_entry'``
    instead uses the mean over the ``pre_entry_seconds`` immediately before
    entry, excluding the entry frame and without subtracting another baseline.
    ``feature_window='pre_first_lick'`` uses the ``pre_lick_seconds`` before
    the first qualifying lick within each landmark, excluding the lick frame.
    ``lick_definition='task'`` requires speed below the task threshold;
    ``lick_definition='any'`` accepts every detected lick.
    ``feature_window='pre_lick_or_entry'`` searches from
    ``lick_search_pre_entry_seconds`` before entry through landmark exit. It
    anchors to the first lick below the task speed threshold when one occurs,
    to the first detected lick when all licks are at/above threshold, and
    otherwise to landmark entry.
    """
    if feature_window not in {
        "landmark", "pre_entry", "pre_first_lick", "pre_lick_or_entry"
    }:
        raise ValueError(
            "feature_window must be 'landmark', 'pre_entry', or "
            "'pre_first_lick', or 'pre_lick_or_entry'"
        )
    if (
        feature_window == "pre_entry"
        and (not np.isfinite(pre_entry_seconds) or pre_entry_seconds <= 0)
    ):
        raise ValueError("pre_entry_seconds must be finite and positive")
    if (
        feature_window in {"pre_first_lick", "pre_lick_or_entry"}
        and (not np.isfinite(pre_lick_seconds) or pre_lick_seconds <= 0)
    ):
        raise ValueError("pre_lick_seconds must be finite and positive")
    if (
        not np.isfinite(lick_search_pre_entry_seconds)
        or lick_search_pre_entry_seconds < 0
    ):
        raise ValueError(
            "lick_search_pre_entry_seconds must be finite and non-negative"
        )
    if lick_definition not in {"task", "any"}:
        raise ValueError("lick_definition must be 'task' or 'any'")

    selected_dff = prepared["selected_dff"]
    cluster_ids = prepared["cluster_ids"]
    entry_frames = prepared["entry_frames"]
    exit_frames = prepared["exit_frames"]
    fine_labels = prepared["fine_labels"]
    pooled_labels = prepared["pooled_labels"]
    source_visit_indices = prepared["source_visit_indices"]
    visit_was_licked = prepared["visit_was_licked"]
    visit_had_any_lick = prepared["visit_had_any_lick"]
    visit_mean_speed = prepared["visit_mean_speed"]
    first_task_lick_frames = prepared["first_task_lick_frames"]
    first_any_lick_frames = prepared["first_any_lick_frames"]
    first_lick_below_speed_threshold = prepared[
        "first_lick_below_speed_threshold"
    ]
    any_lick_mask = prepared["any_lick_mask"]
    task_lick_mask = prepared["task_lick_mask"]
    velocity_threshold = prepared["velocity_threshold"]
    lick_search_pre_frames = int(round(
        float(lick_search_pre_entry_seconds) * frame_rate
    ))
    lick_search_start_frames = np.maximum(
        0, entry_frames - lick_search_pre_frames
    )
    visit_was_licked_in_search_window = np.asarray(
        [
            np.any(task_lick_mask[start:exit_ + 1])
            for start, exit_ in zip(lick_search_start_frames, exit_frames)
        ],
        dtype=bool,
    )

    if feature_window == "landmark":
        population = make_landmark_population(
            selected_dff,
            entry_frames,
            exit_frames,
            pooled_labels,
            cluster_ids,
            classes=("A", "B"),
        )
        feature_description = "whole landmark"
    elif feature_window == "pre_entry":
        population = make_trial_population(
            selected_dff,
            entry_frames,
            pooled_labels,
            cluster_ids,
            frame_rate=frame_rate,
            response_window=(-float(pre_entry_seconds), 0.0),
            baseline_window=None,
            classes=("A", "B"),
        )
        retained_for_duration = population["trial_indices"]
        population["landmark_duration_frames"] = (
            exit_frames[retained_for_duration]
            - entry_frames[retained_for_duration]
            + 1
        )
        feature_description = f"{float(pre_entry_seconds):g} s pre-entry"
    elif feature_window == "pre_first_lick":
        first_lick_frames = (
            first_task_lick_frames
            if lick_definition == "task"
            else first_any_lick_frames
        )
        population = make_trial_population(
            selected_dff,
            first_lick_frames,
            pooled_labels,
            cluster_ids,
            frame_rate=frame_rate,
            response_window=(-float(pre_lick_seconds), 0.0),
            baseline_window=None,
            classes=("A", "B"),
        )
        retained_for_duration = population["trial_indices"]
        population["landmark_duration_frames"] = (
            exit_frames[retained_for_duration]
            - entry_frames[retained_for_duration]
            + 1
        )
        population["first_lick_frame"] = first_lick_frames[
            retained_for_duration
        ]
        population["lick_definition"] = lick_definition
        feature_description = (
            f"{float(pre_lick_seconds):g} s before first "
            f"{lick_definition} lick"
        )
    else:
        search_pre_frames = int(round(
            float(lick_search_pre_entry_seconds) * frame_rate
        ))
        search_start_frames = np.maximum(
            0, entry_frames - search_pre_frames
        )
        search_first_task_lick_frames = np.asarray(
            [
                (start + hits[0] if len(hits) else -1)
                for start, exit_ in zip(search_start_frames, exit_frames)
                for hits in [
                    np.flatnonzero(task_lick_mask[start:exit_ + 1])
                ]
            ],
            dtype=int,
        )
        search_first_any_lick_frames = np.asarray(
            [
                (start + hits[0] if len(hits) else -1)
                for start, exit_ in zip(search_start_frames, exit_frames)
                for hits in [
                    np.flatnonzero(any_lick_mask[start:exit_ + 1])
                ]
            ],
            dtype=int,
        )
        search_was_licked = search_first_task_lick_frames >= 0
        search_had_any_lick = search_first_any_lick_frames >= 0
        anchor_condition = np.full(
            len(entry_frames), "no detected lick", dtype="<U48"
        )
        anchor_condition[
            search_had_any_lick & ~search_was_licked
        ] = "licks only at/above speed threshold"
        anchor_condition[
            search_was_licked
        ] = "at least one lick below speed threshold"
        anchor_frames = np.where(
            search_was_licked,
            search_first_task_lick_frames,
            np.where(
                search_had_any_lick,
                search_first_any_lick_frames,
                entry_frames,
            ),
        )
        population = make_trial_population(
            selected_dff,
            anchor_frames,
            pooled_labels,
            cluster_ids,
            frame_rate=frame_rate,
            response_window=(-float(pre_lick_seconds), 0.0),
            baseline_window=None,
            classes=("A", "B"),
        )
        retained_for_duration = population["trial_indices"]
        population["landmark_duration_frames"] = (
            exit_frames[retained_for_duration]
            - entry_frames[retained_for_duration]
            + 1
        )
        population["anchor_frame"] = anchor_frames[retained_for_duration]
        population["anchor_condition"] = anchor_condition[
            retained_for_duration
        ]
        population["anchor_was_licked"] = search_was_licked[
            retained_for_duration
        ]
        population["anchor_had_any_lick"] = search_had_any_lick[
            retained_for_duration
        ]
        population["anchor_first_task_lick_frame"] = (
            search_first_task_lick_frames[retained_for_duration]
        )
        population["anchor_first_any_lick_frame"] = (
            search_first_any_lick_frames[retained_for_duration]
        )
        population["lick_search_start_frame"] = search_start_frames[
            retained_for_duration
        ]
        population["lick_search_pre_entry_seconds"] = float(
            lick_search_pre_entry_seconds
        )
        feature_description = (
            f"{float(pre_lick_seconds):g} s before the condition-specific "
            "first lick or entry; lick search begins "
            f"{float(lick_search_pre_entry_seconds):g} s pre-entry"
        )
    retained = population["trial_indices"]
    population["fine_labels"] = fine_labels[retained]
    population["source_visit_indices"] = source_visit_indices[retained]
    population["visit_was_licked"] = visit_was_licked[retained]
    population["visit_was_licked_in_search_window"] = (
        visit_was_licked_in_search_window[retained]
    )
    population["lick_search_pre_entry_seconds"] = float(
        lick_search_pre_entry_seconds
    )
    population["visit_had_any_lick"] = visit_had_any_lick[retained]
    population["visit_mean_speed"] = visit_mean_speed[retained]
    population["first_any_lick_frame"] = first_any_lick_frames[retained]
    population["first_task_lick_frame"] = first_task_lick_frames[retained]
    population["first_lick_below_speed_threshold"] = (
        first_lick_below_speed_threshold[retained]
    )
    population["velocity_threshold"] = float(velocity_threshold)
    overview = pd.Series(
        {
            "session": prepared["session_id"],
            "protocol": prepared["protocol"],
            "dF/F file": prepared["dff_filename"],
            "feature window": feature_description,
            "lick definition": (
                lick_definition
                if feature_window == "pre_first_lick"
                else None
            ),
            "lick search pre-entry (s)": (
                float(lick_search_pre_entry_seconds)
                if feature_window == "pre_lick_or_entry"
                else None
            ),
            "already valid-frame aligned": prepared[
                "already_valid_frame_aligned"
            ],
            "dF/F shape": prepared["dff_shape"],
            "aligned cells": len(cluster_ids),
            "omitted A triplets excluded": int(
                len(prepared["omitted_a_visits"])
            ),
            "omission-linked visits excluded": int(
                len(prepared["omission_triplet_visits"])
            ),
            "A trials": int(np.sum(population["labels"] == "A")),
            "B trials (pooled)": int(np.sum(population["labels"] == "B")),
            "B1 trials": int(np.sum(population["fine_labels"] == "B1")),
            "B2 trials": int(np.sum(population["fine_labels"] == "B2")),
            "A1 trials": int(np.sum(population["fine_labels"] == "A1")),
            "A2 trials": int(np.sum(population["fine_labels"] == "A2")),
            "B no-lick trials": int(np.sum(
                (population["labels"] == "B")
                & ~population["visit_was_licked"]
            )),
            "B any-lick trials": int(np.sum(
                (population["labels"] == "B")
                & population["visit_had_any_lick"]
            )),
            "median landmark duration (s)": float(
                np.median(population["landmark_duration_frames"]) / frame_rate
            ),
        }
    )
    return population, overview


def load_protocol_population(
    session_id,
    alignment_row,
    *,
    mouse,
    rawdata_root,
    behaviour_aligned_root,
    dff_filename="dff.npy",
    frame_rate=45.0,
    feature_window="landmark",
    pre_entry_seconds=2.0,
    pre_lick_seconds=2.0,
    lick_search_pre_entry_seconds=0.0,
    lick_definition="task",
):
    """Compatibility wrapper that prepares a session and builds one window."""
    prepared = prepare_protocol_session(
        session_id,
        alignment_row,
        mouse=mouse,
        rawdata_root=rawdata_root,
        behaviour_aligned_root=behaviour_aligned_root,
        dff_filename=dff_filename,
    )
    return build_protocol_population(
        prepared,
        frame_rate=frame_rate,
        feature_window=feature_window,
        pre_entry_seconds=pre_entry_seconds,
        pre_lick_seconds=pre_lick_seconds,
        lick_search_pre_entry_seconds=lick_search_pre_entry_seconds,
        lick_definition=lick_definition,
    )


def load_protocol_population_bundle(
    session_id,
    alignment_row,
    *,
    mouse,
    rawdata_root,
    behaviour_aligned_root,
    population_specs,
    dff_filename="dff.npy",
    frame_rate=45.0,
):
    """Prepare a session once and build several named feature populations.

    ``population_specs`` maps each output name to keyword arguments accepted by
    :func:`build_protocol_population`, excluding ``prepared`` and optionally
    ``frame_rate``. The expensive dF/F, behaviour, and landmark parsing occurs
    exactly once for the session.
    """
    prepared = prepare_protocol_session(
        session_id,
        alignment_row,
        mouse=mouse,
        rawdata_root=rawdata_root,
        behaviour_aligned_root=behaviour_aligned_root,
        dff_filename=dff_filename,
    )
    results = {}
    for name, spec in population_specs.items():
        options = dict(spec)
        options.setdefault("frame_rate", frame_rate)
        results[name] = build_protocol_population(prepared, **options)
    return results

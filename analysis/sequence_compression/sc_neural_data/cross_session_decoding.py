"""Cross-session population decoding for tracked calcium-imaging ROIs.

The public functions deliberately keep feature extraction separate from model
fitting.  This makes the two important safeguards explicit: features use the
same tracked cells in the same order, and all supervised preprocessing is fit
on the training session only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _frame_slice(window_seconds, frame_rate):
    if len(window_seconds) != 2:
        raise ValueError("A time window must contain (start, stop)")
    start, stop = (float(value) for value in window_seconds)
    if not np.isfinite([start, stop, frame_rate]).all() or frame_rate <= 0:
        raise ValueError("Time windows and frame_rate must be finite and valid")
    start_frame = int(np.floor(start * frame_rate))
    stop_frame = int(np.ceil(stop * frame_rate))
    if stop_frame <= start_frame:
        raise ValueError("A time window must have stop > start")
    return np.arange(start_frame, stop_frame, dtype=int)


def make_trial_population(
    dff,
    entry_frames,
    labels,
    cluster_ids,
    *,
    frame_rate,
    response_window=(0.0, 1.0),
    baseline_window=(-1.0, 0.0),
    classes=("A", "B"),
    zscore_traces=True,
):
    """Convert landmark visits into one population vector per trial.

    Each feature is a tracked cell's mean response in ``response_window``.  If
    ``baseline_window`` is not ``None``, its mean is subtracted trial by trial.
    Traces are z-scored within each session before windowing by default; this
    removes session-specific fluorescence scale without using class labels.

    Returns a dictionary accepted by :func:`decode_session_pair`. Trials whose
    full response/baseline windows do not fit in the recording are omitted.
    """
    dff = np.asarray(dff, dtype=float)
    entry_frames = np.asarray(entry_frames, dtype=int)
    labels = np.asarray(labels)
    cluster_ids = np.asarray(cluster_ids)
    classes = tuple(classes)

    if dff.ndim != 2:
        raise ValueError("dff must have shape (cells, frames)")
    if entry_frames.ndim != 1 or labels.ndim != 1:
        raise ValueError("entry_frames and labels must be one-dimensional")
    if len(entry_frames) != len(labels):
        raise ValueError("entry_frames and labels must have equal length")
    if cluster_ids.shape != (dff.shape[0],):
        raise ValueError("cluster_ids must contain one ID per dff row")
    if np.unique(cluster_ids).size != cluster_ids.size:
        raise ValueError("cluster_ids must be unique within a session")
    if len(classes) != 2 or classes[0] == classes[1]:
        raise ValueError("classes must contain two distinct labels")

    signal = dff.copy()
    if zscore_traces:
        means = np.nanmean(signal, axis=1, keepdims=True)
        scales = np.nanstd(signal, axis=1, keepdims=True)
        if np.any(~np.isfinite(scales)) or np.any(scales == 0):
            raise ValueError("Every cell must have a finite, non-zero trace variance")
        signal = (signal - means) / scales

    response_offsets = _frame_slice(response_window, frame_rate)
    baseline_offsets = (
        None if baseline_window is None
        else _frame_slice(baseline_window, frame_rate)
    )
    selected = np.flatnonzero(np.isin(labels, classes))
    features = []
    retained_trials = []
    n_frames = signal.shape[1]
    for trial in selected:
        response_frames = entry_frames[trial] + response_offsets
        all_frames = response_frames if baseline_offsets is None else np.concatenate(
            [response_frames, entry_frames[trial] + baseline_offsets]
        )
        if all_frames.min() < 0 or all_frames.max() >= n_frames:
            continue
        feature = np.mean(signal[:, response_frames], axis=1)
        if baseline_offsets is not None:
            feature -= np.mean(
                signal[:, entry_frames[trial] + baseline_offsets], axis=1
            )
        if np.isfinite(feature).all():
            features.append(feature)
            retained_trials.append(trial)

    if not features:
        raise ValueError("No trials have complete, finite feature windows")
    retained_trials = np.asarray(retained_trials, dtype=int)
    retained_labels = labels[retained_trials]
    missing = set(classes).difference(retained_labels)
    if missing:
        raise ValueError(f"No retained trials for class(es): {sorted(missing)}")
    y = (retained_labels == classes[1]).astype(int)
    return {
        "X": np.vstack(features),
        "y": y,
        "labels": retained_labels,
        "trial_indices": retained_trials,
        "cluster_ids": cluster_ids.copy(),
        "classes": classes,
    }


def make_landmark_population(
    dff,
    entry_frames,
    exit_frames,
    labels,
    cluster_ids,
    *,
    classes=("A", "B"),
    zscore_traces=True,
):
    """Convert each complete landmark visit into one population vector.

    A cell contributes its mean activity from landmark entry through landmark
    exit (both inclusive). No pre-entry baseline is subtracted. Traces are
    z-scored within each session by default so fluorescence scales are
    comparable across cells while preserving the visit's absolute activity
    relative to that session.
    """
    dff = np.asarray(dff, dtype=float)
    entry_frames = np.asarray(entry_frames, dtype=int)
    exit_frames = np.asarray(exit_frames, dtype=int)
    labels = np.asarray(labels)
    cluster_ids = np.asarray(cluster_ids)
    classes = tuple(classes)

    if dff.ndim != 2:
        raise ValueError("dff must have shape (cells, frames)")
    if entry_frames.ndim != 1 or exit_frames.ndim != 1 or labels.ndim != 1:
        raise ValueError("entry_frames, exit_frames, and labels must be one-dimensional")
    if not (len(entry_frames) == len(exit_frames) == len(labels)):
        raise ValueError("entry_frames, exit_frames, and labels must have equal length")
    if cluster_ids.shape != (dff.shape[0],):
        raise ValueError("cluster_ids must contain one ID per dff row")
    if np.unique(cluster_ids).size != cluster_ids.size:
        raise ValueError("cluster_ids must be unique within a session")
    if len(classes) != 2 or classes[0] == classes[1]:
        raise ValueError("classes must contain two distinct labels")

    signal = dff.copy()
    if zscore_traces:
        means = np.nanmean(signal, axis=1, keepdims=True)
        scales = np.nanstd(signal, axis=1, keepdims=True)
        if np.any(~np.isfinite(scales)) or np.any(scales == 0):
            raise ValueError("Every cell must have a finite, non-zero trace variance")
        signal = (signal - means) / scales

    features = []
    retained_trials = []
    durations = []
    for trial in np.flatnonzero(np.isin(labels, classes)):
        entry = int(entry_frames[trial])
        exit_ = int(exit_frames[trial])
        if entry < 0 or exit_ < entry or exit_ >= signal.shape[1]:
            raise ValueError(
                f"Trial {trial} has invalid landmark frames ({entry}, {exit_})"
            )
        feature = np.mean(signal[:, entry:exit_ + 1], axis=1)
        if np.isfinite(feature).all():
            features.append(feature)
            retained_trials.append(trial)
            durations.append(exit_ - entry + 1)

    if not features:
        raise ValueError("No landmark visits have finite population activity")
    retained_trials = np.asarray(retained_trials, dtype=int)
    retained_labels = labels[retained_trials]
    missing = set(classes).difference(retained_labels)
    if missing:
        raise ValueError(f"No retained trials for class(es): {sorted(missing)}")
    return {
        "X": np.vstack(features),
        "y": (retained_labels == classes[1]).astype(int),
        "labels": retained_labels,
        "trial_indices": retained_trials,
        "landmark_duration_frames": np.asarray(durations, dtype=int),
        "cluster_ids": cluster_ids.copy(),
        "classes": classes,
    }


def _aligned_features(train_population, test_population, minimum_cells):
    train_ids = np.asarray(train_population["cluster_ids"])
    test_ids = np.asarray(test_population["cluster_ids"])
    shared = np.intersect1d(train_ids, test_ids)
    if shared.size < minimum_cells:
        raise ValueError(
            f"Only {shared.size} shared cells; minimum_cells={minimum_cells}"
        )
    train_lookup = {value: index for index, value in enumerate(train_ids)}
    test_lookup = {value: index for index, value in enumerate(test_ids)}
    train_columns = [train_lookup[value] for value in shared]
    test_columns = [test_lookup[value] for value in shared]
    return (
        np.asarray(train_population["X"], dtype=float)[:, train_columns],
        np.asarray(test_population["X"], dtype=float)[:, test_columns],
        shared,
    )


def _classifier(C, random_state):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
    )


def decode_session_pair(
    train_population,
    test_population,
    *,
    minimum_cells=5,
    C=1.0,
    n_permutations=0,
    random_state=0,
):
    """Train an A/B decoder in one session and evaluate another session.

    Labels are never used to transform the held-out session.  The permutation
    null shuffles training labels only, preserving the independent test labels.
    """
    if tuple(train_population["classes"]) != tuple(test_population["classes"]):
        raise ValueError("Train and test populations use different classes")
    if minimum_cells < 1 or n_permutations < 0:
        raise ValueError("minimum_cells must be positive and n_permutations non-negative")
    X_train, X_test, shared = _aligned_features(
        train_population, test_population, minimum_cells
    )
    y_train = np.asarray(train_population["y"], dtype=int)
    y_test = np.asarray(test_population["y"], dtype=int)
    if np.unique(y_train).size != 2 or np.unique(y_test).size != 2:
        raise ValueError("Both sessions must contain both classes")

    model = _classifier(C, random_state)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    score = model.decision_function(X_test)
    result = {
        "n_cells": int(shared.size),
        "n_train_trials": int(y_train.size),
        "n_test_trials": int(y_test.size),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "roc_auc": float(roc_auc_score(y_test, score)),
        "shared_cluster_ids": shared,
        "prediction": prediction,
        "decision_score": score,
        "model": model,
    }

    rng = np.random.default_rng(random_state)
    null_scores = np.empty(n_permutations, dtype=float)
    permutation_groups = train_population.get("permutation_groups")
    if permutation_groups is not None:
        permutation_groups = np.asarray(permutation_groups)
        if permutation_groups.shape != y_train.shape:
            raise ValueError("permutation_groups must contain one value per training trial")
    for permutation in range(n_permutations):
        if permutation_groups is None:
            shuffled = rng.permutation(y_train)
        else:
            shuffled = y_train.copy()
            for group in np.unique(permutation_groups):
                group_mask = permutation_groups == group
                shuffled[group_mask] = rng.permutation(y_train[group_mask])
        null_model = _classifier(C, random_state)
        null_model.fit(X_train, shuffled)
        null_prediction = null_model.predict(X_test)
        null_scores[permutation] = balanced_accuracy_score(
            y_test, null_prediction
        )
    result["null_balanced_accuracy"] = null_scores
    result["permutation_p"] = (
        float((1 + np.sum(null_scores >= result["balanced_accuracy"]))
              / (n_permutations + 1))
        if n_permutations else np.nan
    )
    return result


def decode_all_session_pairs(
    populations: Mapping[str, Mapping],
    *,
    session_order: Sequence[str] | None = None,
    **decode_kwargs,
):
    """Decode every ordered, off-diagonal train/test session pair."""
    sessions = list(populations if session_order is None else session_order)
    unknown = set(sessions).difference(populations)
    if unknown:
        raise KeyError(f"Unknown sessions: {sorted(unknown)}")
    rows = []
    pair_results = {}
    for train_session in sessions:
        for test_session in sessions:
            if train_session == test_session:
                continue
            result = decode_session_pair(
                populations[train_session], populations[test_session],
                **decode_kwargs,
            )
            pair_results[(train_session, test_session)] = result
            rows.append({
                "train_session": train_session,
                "test_session": test_session,
                **{key: result[key] for key in (
                    "n_cells", "n_train_trials", "n_test_trials",
                    "balanced_accuracy", "accuracy", "roc_auc",
                    "permutation_p",
                )},
                "null_mean": (
                    float(np.mean(result["null_balanced_accuracy"]))
                    if result["null_balanced_accuracy"].size else np.nan
                ),
            })
    return pd.DataFrame(rows), pair_results

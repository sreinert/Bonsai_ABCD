"""Overview plots for an aligned sequence-compression imaging session.

Imaging metadata is loaded from the barcode-aligned NWB. Green and red delta-fluorescence
signals, plus Tukey red-reference-corrected green delta fluorescence, are loaded or
extracted from the matching two-channel Suite2p traces and saved under the aligned
session's ``funcimg`` folder. Landmark intervals are parsed only to keep reward-PSTH
windows from overlapping other landmarks; no position-tuning or goal-progress analysis
is performed. Raw VR behaviour is mapped onto the functional-imaging clock and compared
with ``behaviour_data.npz``. The established Cohort1SC speed/lick PSTH, transition-matrix,
and sliding-performance calculations are also shown side by side for raw and aligned data.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats

import era_brewer
from nwb_barcoding.sequence_compression import resolve_raw_session, short_session_name
from preprocessing.suite2p.full_session_pipeline.red_reference_correction import (
    compute_channel_delta,
    fit_reference_correction,
)


HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "NineteenEightyNine_r",
    era_brewer.era_brew("NineteenEightyNine", n=256, brew_type="continuous")[::-1],
)

LEGACY_OVERVIEW_PATTERNS = (
    "5_position_heatmap.png",
    "5b_position_heatmap_cv.png",
    "7_goal_progress*.png",
    "8_goal_progress*.png",
    "9_goal_progress*.png",
    "10_goal_progress*.png",
    "11_goal_progress*.png",
    "12_goal_progress*.png",
    "13_goal_progress*.png",
    "6_gr_raw_traces.png",
    "7_gr_reward_heatmap.png",
    "8_gr_reward_heatmap_cv.png",
    "9_gr_reward_psth_selected_cells.png",
)

BEHAVIOUR_OUTPUT_NAMES = (
    "behaviour_raw_vs_aligned.png",
    "behaviour_speed_lick_rate_psth_raw_vs_aligned.png",
    "behaviour_transition_matrix_raw_vs_aligned.png",
    "behaviour_sw_hit_fa_split_raw_vs_aligned.png",
)


def _orient_dff(dff_on_disk, n_rois):
    """Return an imaging-frames by ROIs dF/F array."""
    dff_on_disk = np.asarray(dff_on_disk)
    if dff_on_disk.ndim != 2:
        raise ValueError(f"Expected a 2-D DfOverF array, got {dff_on_disk.shape}")
    if dff_on_disk.shape[1] == n_rois:
        return dff_on_disk
    if dff_on_disk.shape[0] == n_rois:
        return dff_on_disk.T
    raise ValueError(
        f"Cannot match DfOverF shape {dff_on_disk.shape} to {n_rois} segmented ROIs"
    )


def _select_valid_frames(dff_frames_by_roi, valid_frames):
    """Select behavior-overlapping imaging frames after validating their indices."""
    valid_frames = np.asarray(valid_frames, dtype=int)
    if valid_frames.ndim != 1 or valid_frames.size == 0:
        raise ValueError("valid_frames must be a non-empty one-dimensional array")
    if valid_frames.min() < 0 or valid_frames.max() >= len(dff_frames_by_roi):
        raise IndexError(
            f"valid_frames range [{valid_frames.min()}, {valid_frames.max()}] is "
            f"incompatible with DfOverF shape {dff_frames_by_roi.shape}"
        )
    return dff_frames_by_roi[valid_frames]


def _empirical_frame_rate(timestamps):
    """Calculate frame rate from vDAQ timestamps rather than a nominal setting."""
    timestamps = np.asarray(timestamps, dtype=float)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("At least two one-dimensional vDAQ frame timestamps are required")
    intervals = np.diff(timestamps)
    if not np.all(np.isfinite(intervals)) or np.median(intervals) <= 0:
        raise ValueError("vDAQ frame timestamps do not have valid positive intervals")
    return float(1 / np.median(intervals))


def _accepted_roi_indices(plane_seg):
    """Return accepted ROI indices, or every ROI when no Accepted column exists."""
    if "Accepted" in plane_seg.columns:
        accepted = np.flatnonzero(np.asarray(plane_seg["Accepted"]).astype(bool))
    else:
        accepted = np.arange(len(plane_seg))
    if accepted.size == 0:
        raise ValueError("No accepted cells were found")
    return accepted


def _load_nwb_imaging(session_path):
    """Load accepted-cell imaging data from the session's single merged NWB."""
    from pynwb import NWBHDF5IO

    session_path = Path(session_path)
    nwb_paths = sorted(session_path.glob("*.nwb"))
    if len(nwb_paths) != 1:
        raise ValueError(
            f"Expected exactly one merged NWB in {session_path}, found {nwb_paths}"
        )

    with NWBHDF5IO(str(nwb_paths[0]), mode="r") as io:
        nwb = io.read()
        ophys = nwb.processing["ophys"].data_interfaces
        dff_series = next(iter(ophys["DfOverF"].roi_response_series.values()))
        dff_on_disk = np.asarray(dff_series.data[:])

        segmentation_images = ophys["SegmentationImages"].images
        try:
            mean_img = segmentation_images["MeanImageChan1Plane0"][:]
        except KeyError:
            mean_img = segmentation_images["MeanImage"][:]

        segmentations = ophys["ImageSegmentation"].plane_segmentations
        try:
            plane_seg = segmentations["PlaneSegmentationChan1Plane0"][:]
        except KeyError:
            plane_seg = segmentations["PlaneSegmentation"][:]

        n_rois = len(plane_seg)
        dff_frames_by_roi = _orient_dff(dff_on_disk, n_rois)
        timestamps = np.asarray(
            nwb.acquisition["vDAQ_TTLsTable"][:]["timestamp"].values,
            dtype=float,
        )
        frame_rate = _empirical_frame_rate(timestamps)

    with np.load(session_path / "valid_frames.npz") as valid_data:
        valid_frames = valid_data["valid_frames"]
    dff_valid = _select_valid_frames(dff_frames_by_roi, valid_frames)

    try:
        accepted = _accepted_roi_indices(plane_seg)
    except ValueError as error:
        raise ValueError(f"{error} in {nwb_paths[0]}") from error

    image_mask_all = plane_seg["image_mask"]
    dff = dff_valid[:, accepted].T
    image_mask = image_mask_all.iloc[accepted].reset_index(drop=True)
    return dff, accepted, mean_img, image_mask, frame_rate, timestamps


def _atomic_save_npy(path, values):
    """Atomically replace one NumPy output without exposing a partial file."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial.npy")
    try:
        np.save(temporary, values)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_npz(path, **values):
    """Atomically replace one compressed NumPy archive."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial.npz")
    try:
        np.savez_compressed(temporary, **values)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_or_extract_reference_signals(
    suite2p_plane_path, aligned_funcimg_path, accepted, valid_frames
):
    """Load or create aligned green, red, and red-reference-corrected traces."""
    suite2p_plane_path = Path(suite2p_plane_path)
    aligned_funcimg_path = Path(aligned_funcimg_path)
    output_paths = {
        "green_delta": aligned_funcimg_path / "reference_green_delta_v1.npy",
        "red_delta": aligned_funcimg_path / "reference_red_delta_v1.npy",
        "corrected_green_delta": (
            aligned_funcimg_path / "reference_corrected_green_delta_v1.npy"
        ),
    }
    diagnostics_path = aligned_funcimg_path / "reference_correction_diagnostics_v1.npz"
    valid_frames = np.asarray(valid_frames, dtype=int)
    if valid_frames.ndim != 1 or valid_frames.size == 0:
        raise ValueError("valid_frames must be a non-empty one-dimensional array")
    cache_is_complete = diagnostics_path.is_file() and all(
        path.is_file() for path in output_paths.values()
    )
    cache_matches_alignment = False
    if cache_is_complete:
        with np.load(diagnostics_path) as diagnostics:
            cache_matches_alignment = (
                "valid_frames" in diagnostics
                and np.array_equal(diagnostics["valid_frames"], valid_frames)
            )
    if cache_matches_alignment:
        outputs = {
            name: np.load(path, mmap_mode="r") for name, path in output_paths.items()
        }
    else:
        required = ("F.npy", "Fneu.npy", "F_chan2.npy", "Fneu_chan2.npy", "ops.npy")
        missing = [name for name in required if not (suite2p_plane_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Red-reference extraction requires {missing} under {suite2p_plane_path}"
            )
        arrays = [
            np.load(suite2p_plane_path / name, mmap_mode="r")
            for name in required[:4]
        ]
        shapes = {array.shape for array in arrays}
        if len(shapes) != 1:
            raise ValueError(
                f"Green/red Suite2p arrays have incompatible shapes: {shapes}"
            )
        if valid_frames.min() < 0 or valid_frames.max() >= arrays[0].shape[1]:
            raise IndexError(
                f"valid_frames range [{valid_frames.min()}, {valid_frames.max()}] is "
                f"incompatible with Suite2p trace shape {arrays[0].shape}"
            )
        ops = np.load(
            suite2p_plane_path / "ops.npy", allow_pickle=True
        ).item()
        green_delta_full = compute_channel_delta(arrays[0], arrays[1], ops)
        red_delta_full = compute_channel_delta(arrays[2], arrays[3], ops)
        correction = fit_reference_correction(
            green_delta_full, red_delta_full, fs=float(ops["fs"])
        )
        outputs = {
            "green_delta": green_delta_full[:, valid_frames],
            "red_delta": red_delta_full[:, valid_frames],
            "corrected_green_delta": correction.corrected[:, valid_frames],
        }
        aligned_funcimg_path.mkdir(parents=True, exist_ok=True)
        diagnostics_path.unlink(missing_ok=True)
        for name, path in output_paths.items():
            _atomic_save_npy(path, np.asarray(outputs[name], dtype=np.float32))
        _atomic_save_npz(
            diagnostics_path,
            beta=correction.beta,
            raw_standardized_coupling=correction.raw_standardized_coupling,
            applied_standardized_coupling=correction.applied_standardized_coupling,
            green_scale=correction.green_scale,
            red_scale=correction.red_scale,
            converged=correction.converged,
            iterations=correction.iterations,
            status=correction.status,
            smoothing_kernel_frames=correction.smoothing_kernel_frames,
            tuning_constant=correction.tuning_constant,
            max_iterations=correction.max_iterations,
            tolerance=correction.tolerance,
            scale_epsilon=correction.scale_epsilon,
            valid_frames=valid_frames,
        )
        print(
            "Saved aligned red-reference signals: "
            + ", ".join(
                str(path) for path in (*output_paths.values(), diagnostics_path)
            )
        )

    shapes = {name: tuple(values.shape) for name, values in outputs.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Saved red-reference signal shapes are incompatible: {shapes}")
    expected_frames = len(np.asarray(valid_frames))
    signal_shape = next(iter(shapes.values()))
    if signal_shape[1] != expected_frames:
        raise ValueError(
            f"Saved red-reference signals have {signal_shape[1]} frames, but "
            f"valid_frames contains {expected_frames}; remove the four "
            "reference_*_v1 files and rerun"
        )

    accepted = np.asarray(accepted, dtype=int)
    if accepted.ndim != 1 or accepted.size == 0:
        raise ValueError("Red-reference signals require at least one accepted ROI")
    if accepted.min() < 0 or accepted.max() >= signal_shape[0]:
        raise IndexError(
            f"Accepted ROI range [{accepted.min()}, {accepted.max()}] is incompatible "
            f"with red-reference signal shapes {shapes}"
        )
    return {name: np.asarray(values[accepted]) for name, values in outputs.items()}


def _load_aligned_behaviour(session_path, n_frames):
    """Load aligned behaviour, verifying every array against the imaging frames."""
    required = ("position", "speed", "licks", "rewards")
    with np.load(Path(session_path) / "behaviour_data.npz") as aligned:
        missing = [name for name in required if name not in aligned]
        if missing:
            raise KeyError(f"behaviour_data.npz is missing arrays: {missing}")
        values = {
            name: np.asarray(aligned[name], dtype=float) for name in required
        }
    for name, array in values.items():
        if array.ndim != 1:
            raise ValueError(
                f"Expected a one-dimensional {name} array, got {array.shape}"
            )
        if array.size != n_frames:
            raise ValueError(
                f"Aligned dF/F has {n_frames} frames but {name} has {array.size}"
            )
    values["position"] = np.nan_to_num(values["position"], nan=0.0)
    values["speed"] = np.nan_to_num(values["speed"], nan=0.0)
    return values


def _aligned_frame_midpoints(frame_timestamps, valid_frames):
    """Return the clock time at the midpoint of every aligned behaviour bin."""
    frame_timestamps = np.asarray(frame_timestamps, dtype=float)
    valid_frames = np.asarray(valid_frames, dtype=int)
    if frame_timestamps.ndim != 1 or frame_timestamps.size < 2:
        raise ValueError("Frame timestamps must be a one-dimensional array of length >= 2")
    if valid_frames.ndim != 1 or valid_frames.size == 0:
        raise ValueError("valid_frames must be a non-empty one-dimensional array")
    if valid_frames.min() < 0 or valid_frames.max() + 1 >= frame_timestamps.size:
        raise IndexError(
            "Every aligned behaviour bin requires its starting and following frame "
            "timestamp"
        )
    starts = frame_timestamps[valid_frames]
    ends = frame_timestamps[valid_frames + 1]
    if np.any(ends <= starts):
        raise ValueError("Aligned behaviour bins do not have positive durations")
    return starts + (ends - starts) / 2


def _datetime_seconds(index):
    """Convert a pandas datetime-like index to floating seconds without precision loss."""
    converted = pd.DatetimeIndex(index)
    if converted.hasnans:
        raise ValueError("Behaviour timestamps contain NaT values")
    return converted.asi8.astype(float) / 1e9


def _map_raw_times_to_imaging_clock(raw_index, analog_index, analog_imaging_times):
    """Map Bonsai/Harp timestamps onto the barcode-aligned imaging clock."""
    raw_seconds = _datetime_seconds(raw_index)
    analog_seconds = _datetime_seconds(analog_index)
    analog_imaging_times = np.asarray(analog_imaging_times, dtype=float)
    if analog_seconds.size != analog_imaging_times.size or analog_seconds.size < 2:
        raise ValueError(
            "Analog timestamps and imaging-clock samples must have equal length >= 2"
        )
    if np.any(np.diff(analog_seconds) <= 0):
        raise ValueError("Remapped analog timestamps must be strictly increasing")
    origin = analog_seconds[0]
    return np.interp(
        raw_seconds - origin,
        analog_seconds - origin,
        analog_imaging_times,
    )


def _nearest_frame_indices(event_times, frame_times):
    """Map ordered event times to unique nearest aligned imaging frames."""
    event_times = np.asarray(event_times, dtype=float)
    frame_times = np.asarray(frame_times, dtype=float)
    if event_times.ndim != 1 or frame_times.ndim != 1 or frame_times.size == 0:
        raise ValueError("Event and frame times must be one-dimensional")
    insertion = np.searchsorted(frame_times, event_times)
    insertion = np.clip(insertion, 0, frame_times.size - 1)
    previous = np.clip(insertion - 1, 0, frame_times.size - 1)
    choose_previous = np.abs(event_times - frame_times[previous]) <= np.abs(
        event_times - frame_times[insertion]
    )
    mapped = np.where(choose_previous, previous, insertion).astype(int)
    if mapped.size and (np.any(np.diff(mapped) <= 0) or np.unique(mapped).size != mapped.size):
        raise ValueError(
            "Raw landmark events do not map to distinct, strictly ordered imaging frames"
        )
    return mapped


def _build_aligned_task_dataframe(aligned, frame_times, raw_dataframe, raw_times):
    """Build the frame-indexed dataframe expected by Cohort1SC task metrics."""
    frame_times = np.asarray(frame_times, dtype=float)
    raw_times = np.asarray(raw_times, dtype=float)
    if len(frame_times) != len(aligned["position"]):
        raise ValueError("Aligned frame timestamps do not match behaviour arrays")
    index = pd.to_datetime(frame_times, unit="s", origin="unix")
    dataframe = pd.DataFrame(
        {
            "Position": aligned["position"],
            "Treadmill": aligned["speed"],
            "Licks": aligned["licks"],
            "Rewards": np.where(aligned["rewards"] > 0, aligned["rewards"], np.nan),
            "Buffer": np.arange(len(frame_times), dtype=float),
        },
        index=index,
    )

    if "LM_Count" in raw_dataframe.columns:
        event_mask = raw_dataframe["LM_Count"].notna() & (
            raw_dataframe["LM_Count"] >= 0
        )
        event_rows = raw_dataframe.loc[event_mask]
        mapped = _nearest_frame_indices(raw_times[event_mask.to_numpy()], frame_times)
        for column in ("LM_Count", "LM_Position", "LM_Odour", "Sequence_Position"):
            if column in raw_dataframe.columns:
                if pd.api.types.is_numeric_dtype(raw_dataframe[column].dtype):
                    dataframe[column] = np.nan
                else:
                    dataframe[column] = pd.Series(
                        [None] * len(dataframe), index=dataframe.index, dtype=object
                    )
                dataframe.iloc[
                    mapped, dataframe.columns.get_loc(column)
                ] = event_rows[column].to_numpy()
    else:
        if "Events" not in raw_dataframe.columns:
            raise KeyError("Raw behaviour has neither LM_Count nor Events")
        event_text = raw_dataframe["Events"].fillna("").astype(str)
        event_mask = event_text.str.startswith("release") & ~event_text.str.contains(
            "odour0"
        )
        mapped = _nearest_frame_indices(raw_times[event_mask.to_numpy()], frame_times)
        dataframe["Events"] = np.nan
        dataframe.iloc[mapped, dataframe.columns.get_loc("Events")] = event_text.loc[
            event_mask
        ].to_numpy()

    return dataframe


def _parse_landmark_intervals(raw_session, subject, position, reward_frames):
    """Use the current modular parser to locate landmark entry and exit frames."""
    from analysis.sequence_compression.session_functions import io as session_io
    from analysis.sequence_compression.session_functions import session as session_models

    settings, rig_settings = session_io.load_settings(raw_session)
    dataframe = session_io.load_data(raw_session)
    stage = short_session_name(raw_session.name)
    mouse_id = subject.removeprefix("sub-")
    parser = session_models.ABCD_Session(
        settings,
        rig_settings,
        dataframe,
        raw_session,
        mouse_id,
        stage,
        stage,
    )
    parsed = parser.create_session_struct()
    parsed.update(
        position=position,
        rewards=np.asarray(reward_frames, dtype=int),
        mouse=mouse_id,
        stage=stage,
        session_id=stage,
        world=parser.world,
        laps_needed=1,
    )
    parsed = parser.get_landmark_positions(parsed, data="odour")
    parsed = parser.get_lap_idx(parsed)
    parsed = parser.get_lms_visited(parsed)
    entries, exits = parser.get_lm_entry_exit(parsed)
    entries = np.asarray(entries, dtype=int)
    exits = np.asarray(exits, dtype=int)
    if entries.shape != exits.shape or entries.ndim != 1:
        raise ValueError(
            f"Landmark entry/exit arrays are incompatible: {entries.shape}, {exits.shape}"
        )
    if np.any(exits < entries):
        raise ValueError("A parsed landmark exits before its entry")
    return entries, exits


def _load_behaviour_analysis(raw_session, subject, aligned, frame_times, session_path):
    """Load raw VR data and build parallel raw/aligned task-analysis objects."""
    from pynwb import NWBHDF5IO

    from analysis.sequence_compression.session_functions import behaviour as behaviour_models
    from analysis.sequence_compression.session_functions import io as session_io
    from analysis.sequence_compression.session_functions import performance as performance_models
    from analysis.sequence_compression.session_functions import session as session_models
    from nwb_barcoding.build_behaviour_data import (
        align_analog_to_events,
        fit_analog_clock,
        get_ttl,
        load_analog_data,
    )

    settings, rig_settings = session_io.load_settings(raw_session)
    raw_dataframe = session_io.load_data(raw_session)
    analog_data = load_analog_data(raw_session, rig_settings)
    remapped_analog = align_analog_to_events(analog_data, raw_dataframe)

    nwb_paths = sorted(Path(session_path).glob("*.nwb"))
    if len(nwb_paths) != 1:
        raise ValueError(
            f"Expected exactly one merged NWB in {session_path}, found {nwb_paths}"
        )
    with NWBHDF5IO(str(nwb_paths[0]), mode="r") as io:
        camera_timestamps = get_ttl(io.read(), "CameraFrameTTL")
    clock_fit = fit_analog_clock(
        remapped_analog["camera"].to_numpy(), camera_timestamps
    )
    raw_times = _map_raw_times_to_imaging_clock(
        raw_dataframe.index,
        remapped_analog.index,
        clock_fit["sample_times"],
    )

    frame_times = np.asarray(frame_times, dtype=float)
    half_frame = np.median(np.diff(frame_times)) / 2
    overlap = (raw_times >= frame_times[0] - half_frame) & (
        raw_times <= frame_times[-1] + half_frame
    )
    if not np.any(overlap):
        raise ValueError("Raw VR data do not overlap the aligned imaging frames")
    raw_dataframe = raw_dataframe.loc[overlap].copy()
    raw_times = raw_times[overlap]
    # Session's landmark parser treats Buffer as a positional row index. Cropping to
    # the imaging overlap therefore requires re-indexing this column from zero.
    raw_dataframe["Buffer"] = np.arange(len(raw_dataframe), dtype=float)

    aligned_dataframe = _build_aligned_task_dataframe(
        aligned, frame_times, raw_dataframe, raw_times
    )
    mouse_id = subject.removeprefix("sub-")
    stage = short_session_name(Path(raw_session).name)
    raw_task_session = session_models.Session(
        settings, raw_dataframe, raw_session, mouse_id, 2, stage
    )
    aligned_task_session = session_models.Session(
        settings, aligned_dataframe, raw_session, mouse_id, 2, stage
    )
    raw_performance = performance_models.Performance(raw_task_session)
    aligned_performance = performance_models.Performance(aligned_task_session)
    return {
        "raw_dataframe": raw_dataframe,
        "raw_times": raw_times,
        "raw_session": raw_task_session,
        "aligned_session": aligned_task_session,
        "raw_behaviour": behaviour_models.Behaviour(
            raw_task_session, raw_performance
        ),
        "aligned_behaviour": behaviour_models.Behaviour(
            aligned_task_session, aligned_performance
        ),
        "raw_performance": raw_performance,
        "aligned_performance": aligned_performance,
    }


def load_session_data(mouse, session_id, project_root):
    """Load aligned green and robust red-reference signals for the overview."""
    project_root = Path(project_root).expanduser().resolve()
    subject = str(mouse) if str(mouse).startswith("sub-") else f"sub-{mouse}"
    aligned_subject = project_root / "BehaviourAligned_robust" / "cohort2" / subject
    session_path = resolve_raw_session(aligned_subject, session_id)

    (
        nwb_dff,
        accepted,
        mean_img,
        image_mask,
        frame_rate,
        frame_timestamps,
    ) = _load_nwb_imaging(session_path)
    with np.load(session_path / "valid_frames.npz") as valid_data:
        valid_frames = np.asarray(valid_data["valid_frames"], dtype=int)
    suite2p_plane = (
        project_root
        / "processed"
        / "cohort2"
        / subject
        / short_session_name(session_path.name)
        / "suite2p"
        / "plane0"
    )
    try:
        reference = _load_or_extract_reference_signals(
            suite2p_plane, session_path / "funcimg", accepted, valid_frames
        )
    except FileNotFoundError as error:
        print(
            "Red-reference signals unavailable; corrected plots will be skipped: "
            f"{error}"
        )
        reference = None
    if reference is None:
        green_delta = nwb_dff
        red_delta = None
        corrected_green_delta = None
    else:
        green_delta = reference["green_delta"]
        red_delta = reference["red_delta"]
        corrected_green_delta = reference["corrected_green_delta"]
        reference_shapes = {
            "NWB green": nwb_dff.shape,
            "computed green": green_delta.shape,
            "computed red": red_delta.shape,
            "corrected green": corrected_green_delta.shape,
        }
        if len(set(reference_shapes.values())) != 1:
            raise ValueError(
                f"Aligned imaging signal shapes do not match: {reference_shapes}"
            )
    aligned_behaviour = _load_aligned_behaviour(session_path, green_delta.shape[1])
    reward_frames = np.flatnonzero(aligned_behaviour["rewards"])
    position = aligned_behaviour["position"]
    frame_times = _aligned_frame_midpoints(frame_timestamps, valid_frames)
    raw_session = (
        project_root / "rawdata" / "cohort2" / subject / session_path.name
    )
    if not raw_session.is_dir():
        raise FileNotFoundError(f"Matching raw session does not exist: {raw_session}")
    landmark_entries, landmark_exits = _parse_landmark_intervals(
        raw_session, subject, position, reward_frames
    )
    behaviour_analysis = _load_behaviour_analysis(
        raw_session, subject, aligned_behaviour, frame_times, session_path
    )
    return {
        "dF": green_delta,
        "nwb_dF": nwb_dff,
        "red_dF": red_delta,
        "red_reference_corrected_dF": corrected_green_delta,
        "neurons": accepted,
        "mean_img": mean_img,
        "image_mask": image_mask,
        "frame_rate": frame_rate,
        "frame_times": frame_times,
        "aligned_behaviour": aligned_behaviour,
        "reward_frames": reward_frames,
        "landmark_entries": landmark_entries,
        "landmark_exits": landmark_exits,
        "session_path": session_path,
        "subject": subject,
        "behaviour_analysis": behaviour_analysis,
    }


def _concat_masks(image_mask):
    """Combine per-ROI boolean masks into one labelled image."""
    masks = [np.asarray(mask, dtype=bool) for mask in image_mask]
    if not masks:
        raise ValueError("No ROI masks were found")
    combined = np.zeros(masks[0].shape, dtype=int)
    for index, mask in enumerate(masks, start=1):
        combined[mask] = index
    return combined


def plot_fov_masks(mean_img, image_mask, title="Field of view", save_path=None):
    """Plot the mean image with accepted-cell masks overlaid."""
    masks = _concat_masks(image_mask)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(np.asarray(mean_img).T, cmap="gray")
    plt.imshow(masks.T, alpha=(masks != 0).astype(float).T, cmap="rainbow")
    plt.title(f"{title} (n={len(image_mask)} accepted cells)")
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_raw_traces(
    dff,
    n_cells=20,
    start_frame=0,
    n_frames=20000,
    cell_indices=None,
    cell_labels=None,
    frame_rate=None,
    offset_step=6.0,
    signal_label="dF/F",
    save_path=None,
):
    """Plot stacked traces for cells sampled across the accepted population."""
    n_total_cells, n_total_frames = dff.shape
    if n_total_cells == 0 or n_total_frames == 0:
        raise ValueError("Raw traces require at least one cell and one frame")
    if start_frame < 0 or start_frame >= n_total_frames:
        raise ValueError(
            f"start_frame={start_frame} is outside a recording with {n_total_frames} frames"
        )
    end_frame = min(start_frame + n_frames, n_total_frames)
    if cell_indices is None:
        cell_indices = select_trace_cells(n_total_cells, n_cells)
    cell_indices = np.asarray(cell_indices, dtype=int)
    labels = cell_indices if cell_labels is None else np.asarray(cell_labels)
    if labels.size != cell_indices.size:
        raise ValueError("cell_labels must match the number of selected cells")

    window = dff[cell_indices, start_frame:end_frame]
    scale = np.nanstd(window, axis=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale == 0)] = 1
    normalized = window / scale
    time = np.arange(start_frame, end_frame)
    xlabel = "Frame"
    if frame_rate:
        time = time / frame_rate
        xlabel = "Time (s)"

    fig = plt.figure(figsize=(14, 0.35 * len(cell_indices) + 2))
    for index, cell in enumerate(cell_indices):
        plt.plot(time, normalized[index] + index * offset_step, lw=0.6, color="k")
    plt.yticks(
        np.arange(len(cell_indices)) * offset_step,
        [f"ROI {label}" for label in labels],
    )
    plt.xlabel(xlabel)
    plt.title(
        f"Raw {signal_label}, {len(cell_indices)} cells, frames "
        f"{start_frame}-{end_frame}"
    )
    plt.xlim(time[0], time[-1])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def select_trace_cells(n_total_cells, n_cells=20):
    """Select evenly spaced accepted-cell indices for plots 2 and 5."""
    if n_total_cells <= 0:
        raise ValueError("At least one accepted cell is required")
    return np.unique(
        np.linspace(0, n_total_cells - 1, min(n_cells, n_total_cells)).astype(int)
    )


def _window_offsets(frame_rate, window_size):
    """Convert a time window to integer frame offsets."""
    frame_rate_int = int(round(frame_rate))
    if abs(frame_rate_int - frame_rate) > 0.1:
        print(
            f"  rounding frame_rate {frame_rate:.3f} -> {frame_rate_int} Hz "
            "for frame-index math"
        )
    offsets = np.arange(
        int(np.floor(window_size[0] * frame_rate_int)),
        int(np.ceil(window_size[1] * frame_rate_int)),
    )
    if offsets.size == 0 or not (window_size[0] <= 0 < window_size[1]):
        raise ValueError("reward window must contain time zero and at least one frame")
    return offsets


def _separated_complete_rewards(reward_frames, offsets, n_frames, frame_rate):
    """Deduplicate reward pulses and retain events with complete recording windows."""
    rewards = np.asarray(reward_frames, dtype=int).copy()
    if rewards.size > 1:
        keep = np.ones(rewards.size, dtype=bool)
        keep[1:][np.diff(rewards) < 2 * int(round(frame_rate))] = False
        rewards = rewards[keep]
    valid = rewards[
        (rewards + offsets[0] >= 0) & (rewards + offsets[-1] < n_frames)
    ]
    if valid.size == 0:
        raise ValueError("no rewards have a complete peri-event imaging window")
    return valid


def reward_tuning_matrix(dff, reward_frames, frame_rate, window_size=(-5, 5)):
    """Return cells by valid reward trials by peri-event imaging frames."""
    offsets = _window_offsets(frame_rate, window_size)
    valid = _separated_complete_rewards(
        reward_frames, offsets, dff.shape[1], frame_rate
    )
    return dff[:, valid[:, None] + offsets[None, :]]


def reward_psth_matrix(
    dff,
    cell_indices,
    reward_frames,
    landmark_entries,
    landmark_exits,
    frame_rate,
    window_size=(-2, 2),
):
    """Return PSTHs using the largest common window that avoids other landmarks.

    The requested window is shortened only when necessary. For every reward, the
    contiguous interval around time zero is bounded by the recording edges and the
    nearest *other* landmark. The intersection of those intervals is the maximum common
    window that can be averaged without any trial sampling another landmark.
    """
    offsets = _window_offsets(frame_rate, window_size)
    rewards = np.asarray(reward_frames, dtype=int).copy()
    if rewards.size > 1:
        keep = np.ones(rewards.size, dtype=bool)
        keep[1:][np.diff(rewards) < 2 * int(round(frame_rate))] = False
        rewards = rewards[keep]
    entries = np.asarray(landmark_entries, dtype=int)
    exits = np.asarray(landmark_exits, dtype=int)
    usable_rewards = []
    safe_ranges = []
    for reward in rewards:
        containing = np.flatnonzero((entries <= reward) & (exits >= reward))
        if containing.size != 1:
            continue
        other = np.arange(entries.size) != containing[0]
        sample_frames = reward + offsets
        safe = (sample_frames >= 0) & (sample_frames < dff.shape[1])
        for entry, exit_ in zip(entries[other], exits[other]):
            safe &= ~((sample_frames >= entry) & (sample_frames <= exit_))

        zero_candidates = np.flatnonzero(offsets == 0)
        if zero_candidates.size != 1 or not safe[zero_candidates[0]]:
            continue
        zero_index = zero_candidates[0]
        left = zero_index
        while left > 0 and safe[left - 1]:
            left -= 1
        right = zero_index
        while right + 1 < safe.size and safe[right + 1]:
            right += 1
        usable_rewards.append(reward)
        safe_ranges.append((offsets[left], offsets[right]))

    if not usable_rewards:
        raise ValueError(
            "no rewards lie in a uniquely identified landmark with a usable time window"
        )
    common_start = max(start for start, _ in safe_ranges)
    common_end = min(end for _, end in safe_ranges)
    if common_start > 0 or common_end < 0:
        raise ValueError("neighboring landmarks leave no common reward-centered window")
    common_offsets = np.arange(common_start, common_end + 1)
    usable_rewards = np.asarray(usable_rewards, dtype=int)
    matrix = dff[np.asarray(cell_indices, dtype=int)][
        :, usable_rewards[:, None] + common_offsets
    ]
    return (
        matrix,
        usable_rewards,
        common_offsets / float(int(round(frame_rate))),
    )


def _zscore_sort_plot(
    mat_for_sort, mat_for_display, x_values, title, signal_label="dF/F", save_path=None
):
    """Average trials, z-score cells, and display using the sorting half's peaks."""
    sort_curve = stats.zscore(
        np.nanmean(mat_for_sort, axis=1), axis=1, nan_policy="omit"
    )
    display_curve = stats.zscore(
        np.nanmean(mat_for_display, axis=1), axis=1, nan_policy="omit"
    )
    valid = (
        np.isfinite(sort_curve).all(axis=1)
        & np.isfinite(display_curve).all(axis=1)
    )
    if not np.any(valid):
        raise ValueError("no cells have finite peri-reward tuning curves")
    if np.any(~valid):
        print(f"  dropping {np.sum(~valid)} cell(s) with non-finite tuning curves")
    sort_curve = sort_curve[valid]
    display_curve = display_curve[valid]
    order = np.argsort(np.argmax(sort_curve, axis=1))

    fig = plt.figure(figsize=(8, 6))
    extent = [x_values[0], x_values[-1], sort_curve.shape[0], 0]
    plt.imshow(
        display_curve[order],
        aspect="auto",
        cmap=HEATMAP_CMAP,
        interpolation="none",
        extent=extent,
    )
    plt.colorbar(label=f"z-scored {signal_label}")
    plt.axvline(0, color="white", linestyle="--", lw=1)
    plt.xlabel("Time from reward (s)")
    plt.ylabel(f"Cells (sorted, n={sort_curve.shape[0]})")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_reward_heatmap(
    dff,
    reward_frames,
    frame_rate,
    window_size=(-5, 5),
    cross_validated=False,
    signal_label="dF/F",
    save_path=None,
):
    """Plot all-trial or even/odd cross-validated peri-reward activity."""
    matrix = reward_tuning_matrix(dff, reward_frames, frame_rate, window_size)
    n_window = matrix.shape[2]
    x_values = np.linspace(window_size[0], window_size[1], n_window)
    if cross_validated:
        if matrix.shape[1] < 2:
            raise ValueError(
                "cross-validated reward tuning requires at least two complete rewards"
            )
        sort_matrix, display_matrix = matrix[:, 0::2, :], matrix[:, 1::2, :]
        title = (
            f"{signal_label} peri-reward tuning "
            "(sorted on even trials, shown on odd trials)"
        )
    else:
        sort_matrix = display_matrix = matrix
        title = f"{signal_label} peri-reward tuning (all cells)"
    return _zscore_sort_plot(
        sort_matrix,
        display_matrix,
        x_values,
        title,
        signal_label=signal_label,
        save_path=save_path,
    )


def plot_selected_cell_reward_psths(
    dff,
    cell_indices,
    reward_frames,
    landmark_entries,
    landmark_exits,
    frame_rate,
    window_size=(-2, 2),
    cell_labels=None,
    signal_label="dF/F",
    save_path=None,
):
    """Plot mean reward PSTH +/- SEM for each selected cell in its own subplot."""
    matrix, usable_rewards, time = reward_psth_matrix(
        dff,
        cell_indices,
        reward_frames,
        landmark_entries,
        landmark_exits,
        frame_rate,
        window_size,
    )
    print(
        f"  using {len(usable_rewards)} reward(s); adaptive window "
        f"{time[0]:.2f} to {time[-1]:.2f} s"
    )
    cell_indices = np.asarray(cell_indices, dtype=int)
    labels = cell_indices if cell_labels is None else np.asarray(cell_labels)
    if labels.size != cell_indices.size:
        raise ValueError("cell_labels must match the number of selected cells")

    n_columns = min(4, len(cell_indices))
    n_rows = int(math.ceil(len(cell_indices) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4 * n_columns, 2.8 * n_rows),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()
    for index, (axis, label) in enumerate(zip(axes, labels)):
        mean = np.nanmean(matrix[index], axis=0)
        axis.plot(time, mean, color="black", linewidth=1.5)
        if matrix.shape[1] > 1:
            sem = stats.sem(matrix[index], axis=0, nan_policy="omit")
            axis.fill_between(time, mean - sem, mean + sem, color="black", alpha=0.2)
        axis.axvline(0, color="grey", linestyle="--", linewidth=1)
        axis.set_title(f"ROI {label}")
        axis.set_xlim(time[0], time[-1])
        axis.set_xticks([time[0], 0, time[-1]])
        axis.set_xticklabels([f"{time[0]:.2f}", "0", f"{time[-1]:.2f}"])
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[len(cell_indices):]:
        axis.set_visible(False)
    for row in range(n_rows):
        axes[row * n_columns].set_ylabel(signal_label)
    for axis in axes[-n_columns:]:
        if axis.get_visible():
            axis.set_xlabel("Time from reward (s)")
    fig.suptitle(
        f"{signal_label} reward PSTHs for raw-trace cells "
        f"(n={len(usable_rewards)} rewards; {time[0]:.2f} to {time[-1]:.2f} s)"
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_behaviour_alignment(data, save_path=None):
    """Compare raw VR signals with the frame-binned aligned NPZ arrays."""
    context = data["behaviour_analysis"]
    raw = context["raw_dataframe"]
    raw_times = np.asarray(context["raw_times"], dtype=float)
    aligned = data["aligned_behaviour"]
    frame_times = np.asarray(data["frame_times"], dtype=float)
    time_origin = frame_times[0]
    raw_elapsed = raw_times - time_origin
    aligned_elapsed = frame_times - time_origin

    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(raw_elapsed, raw["Position"], color="0.25", lw=0.8, label="Raw VR")
    axes[0].plot(
        aligned_elapsed,
        aligned["position"],
        color="#4478ac",
        lw=0.9,
        alpha=0.9,
        label="Aligned NPZ",
    )
    axes[0].set_ylabel("Position")
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(raw_elapsed, raw["Treadmill"], color="0.25", lw=0.8)
    axes[1].plot(
        aligned_elapsed,
        aligned["speed"],
        color="#4478ac",
        lw=0.9,
        alpha=0.9,
    )
    axes[1].set_ylabel("Speed")

    raw_lick_mask = raw["Licks"].to_numpy(dtype=float) > 0
    aligned_lick_mask = aligned["licks"] > 0
    raw_lick_count = int(np.nansum(raw["Licks"].to_numpy(dtype=float)))
    aligned_lick_count = int(np.nansum(aligned["licks"]))
    axes[2].scatter(
        raw_elapsed[raw_lick_mask],
        np.ones(np.sum(raw_lick_mask)),
        s=9,
        color="0.25",
        label=f"Raw VR (n={raw_lick_count})",
    )
    axes[2].scatter(
        aligned_elapsed[aligned_lick_mask],
        np.zeros(np.sum(aligned_lick_mask)),
        s=9,
        color="#d97706",
        label=f"Aligned NPZ (n={aligned_lick_count})",
    )
    axes[2].set_yticks([0, 1], labels=["Aligned", "Raw"])
    axes[2].set_ylabel("Licks")
    axes[2].legend(frameon=False, loc="upper right")

    raw_reward_mask = raw["Rewards"].notna().to_numpy()
    aligned_reward_mask = aligned["rewards"] > 0
    aligned_reward_count = int(np.nansum(aligned["rewards"]))
    axes[3].scatter(
        raw_elapsed[raw_reward_mask],
        np.ones(np.sum(raw_reward_mask)),
        s=12,
        color="0.25",
        label=f"Raw VR (n={int(np.sum(raw_reward_mask))})",
    )
    axes[3].scatter(
        aligned_elapsed[aligned_reward_mask],
        np.zeros(np.sum(aligned_reward_mask)),
        s=12,
        color="#2f855a",
        label=f"Aligned NPZ (n={aligned_reward_count})",
    )
    axes[3].set_yticks([0, 1], labels=["Aligned", "Raw"])
    axes[3].set_ylabel("Rewards")
    axes[3].set_xlabel("Time from first aligned frame (s)")
    axes[3].legend(frameon=False, loc="upper right")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(aligned_elapsed[0], aligned_elapsed[-1])
    fig.suptitle("Raw VR versus behaviour_data.npz")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_behaviour_psth_comparison(context, save_path=None):
    """Plot the established speed/lick PSTHs for raw and aligned behaviour."""
    from analysis.sequence_compression.session_functions import plotting

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for row, (label, session, behaviour) in enumerate(
        (
            ("Raw VR", context["raw_session"], context["raw_behaviour"]),
            (
                "Aligned NPZ",
                context["aligned_session"],
                context["aligned_behaviour"],
            ),
        )
    ):
        plotting.plot_speed_lick_rate_psth(
            session, behaviour, bins=None, axes=axes[row]
        )
        axes[row, 0].set_title(f"{label} — speed")
        axes[row, 1].set_title(f"{label} — lick rate")
    fig.suptitle("Landmark-aligned speed and lick-rate PSTHs")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_behaviour_transition_comparison(context, save_path=None):
    """Plot raw and aligned versions of the established transition matrices."""
    from analysis.sequence_compression.session_functions import plotting

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    for row, (label, session, performance) in enumerate(
        (
            ("Raw VR", context["raw_session"], context["raw_performance"]),
            (
                "Aligned NPZ",
                context["aligned_session"],
                context["aligned_performance"],
            ),
        )
    ):
        plotting.plot_transition_matrix(session, performance, axes=axes[row])
        for axis in axes[row]:
            axis.set_title(f"{label} — {axis.get_title()}")
    fig.suptitle("Task transition matrices")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_behaviour_sliding_performance(context, save_path=None):
    """Plot split landmark hit/false-alarm rates for raw and aligned behaviour."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for axis, (label, performance) in zip(
        axes,
        (
            ("Raw VR", context["raw_performance"]),
            ("Aligned NPZ", context["aligned_performance"]),
        ),
    ):
        performance.calc_sw_hit_fa(
            split_lms=True, window=24, plot=True, ax=axis
        )
        axis.set_title(label)
    fig.suptitle("Sliding task performance (24-landmark window)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def remove_legacy_overview_plots(save_dir):
    """Remove only known outputs from previous overview implementations."""
    save_dir = Path(save_dir)
    removed = []
    for pattern in LEGACY_OVERVIEW_PATTERNS:
        for path in sorted(save_dir.glob(pattern)):
            if path.is_file():
                path.unlink()
                removed.append(path)
    if removed:
        print(f"Removed {len(removed)} legacy overview plot(s).")
    return removed


def run_overview(
    mouse,
    session_id,
    project_root,
    n_trace_cells=20,
    trace_start_frame=0,
    trace_n_frames=20000,
    reward_window=(-5, 5),
    reward_psth_window=(-2, 2),
    save_dir=None,
    show=True,
):
    """Produce imaging plus strict raw/aligned behavioural overview plots."""
    if save_dir is None:
        project_root_path = Path(project_root).expanduser().resolve()
        subject = str(mouse) if str(mouse).startswith("sub-") else f"sub-{mouse}"
        aligned_subject = (
            project_root_path / "BehaviourAligned_robust" / "cohort2" / subject
        )
        resolved_session = resolve_raw_session(aligned_subject, session_id)
        save_dir = (
            project_root_path
            / "analysis"
            / "cohort2"
            / subject
            / resolved_session.name
        )
    elif save_dir is False:
        save_dir = None
    else:
        save_dir = Path(save_dir)
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        for filename in BEHAVIOUR_OUTPUT_NAMES:
            (save_dir / filename).unlink(missing_ok=True)

    data = load_session_data(mouse, session_id, project_root)
    dff = data["dF"]
    corrected = data.get("red_reference_corrected_dF")
    green_label = "Green ΔF"
    n_imaging_plots = 9 if corrected is not None else 5
    n_plots = n_imaging_plots + len(BEHAVIOUR_OUTPUT_NAMES)
    print(
        f"Loaded {dff.shape[0]} accepted cells, {dff.shape[1]} frames, "
        f"{data['frame_rate']:.2f} Hz"
    )

    def output_path(name):
        return str(save_dir / f"{name}.png") if save_dir else None

    trace_cell_indices = select_trace_cells(dff.shape[0], n_trace_cells)
    figures = {}
    print(f"Plot 1/{n_plots}: FOV with accepted-cell masks...")
    figures["fov"] = plot_fov_masks(
        data["mean_img"],
        data["image_mask"],
        title=f"{data['subject']} {data['session_path'].name}",
        save_path=output_path("1_fov_masks"),
    )
    print(f"Plot 2/{n_plots}: raw {green_label} traces...")
    figures["raw_traces"] = plot_raw_traces(
        dff,
        n_cells=n_trace_cells,
        start_frame=trace_start_frame,
        n_frames=trace_n_frames,
        cell_indices=trace_cell_indices,
        cell_labels=data["neurons"][trace_cell_indices],
        frame_rate=data["frame_rate"],
        signal_label=green_label,
        save_path=output_path("2_raw_traces"),
    )

    for key, filename, cross_validated, label in (
        ("reward", "3_reward_heatmap", False, "all trials"),
        ("reward_cv", "4_reward_heatmap_cv", True, "even/odd cross-validated"),
    ):
        print(
            f"Plot {'4' if cross_validated else '3'}/{n_plots}: "
            f"peri-reward heatmap ({label})..."
        )
        try:
            figures[key] = plot_reward_heatmap(
                dff,
                data["reward_frames"],
                data["frame_rate"],
                window_size=reward_window,
                cross_validated=cross_validated,
                signal_label=green_label,
                save_path=output_path(filename),
            )
        except ValueError as error:
            print(f"  skipping plot: {error}")
            figures[key] = None
            skipped_output = output_path(filename)
            if skipped_output:
                Path(skipped_output).unlink(missing_ok=True)

    print(
        f"Plot 5/{n_plots}: selected-cell reward PSTHs bounded by "
        "neighboring landmarks..."
    )
    try:
        figures["reward_psth_selected_cells"] = plot_selected_cell_reward_psths(
            dff,
            trace_cell_indices,
            data["reward_frames"],
            data["landmark_entries"],
            data["landmark_exits"],
            data["frame_rate"],
            window_size=reward_psth_window,
            cell_labels=data["neurons"][trace_cell_indices],
            signal_label=green_label,
            save_path=output_path("5_reward_psth_selected_cells"),
        )
    except ValueError as error:
        print(f"  skipping plot: {error}")
        figures["reward_psth_selected_cells"] = None
        skipped_output = output_path("5_reward_psth_selected_cells")
        if skipped_output:
            Path(skipped_output).unlink(missing_ok=True)

    corrected_outputs = (
        "6_reference_corrected_raw_traces",
        "7_reference_corrected_reward_heatmap",
        "8_reference_corrected_reward_heatmap_cv",
        "9_reference_corrected_reward_psth_selected_cells",
    )
    if corrected is None:
        for filename in corrected_outputs:
            skipped_output = output_path(filename)
            if skipped_output:
                Path(skipped_output).unlink(missing_ok=True)
    else:
        corrected_label = "Red-reference-corrected green ΔF"
        print(f"Plot 6/{n_plots}: raw {corrected_label} traces...")
        figures["reference_corrected_raw_traces"] = plot_raw_traces(
            corrected,
            n_cells=n_trace_cells,
            start_frame=trace_start_frame,
            n_frames=trace_n_frames,
            cell_indices=trace_cell_indices,
            cell_labels=data["neurons"][trace_cell_indices],
            frame_rate=data["frame_rate"],
            signal_label=corrected_label,
            save_path=output_path(corrected_outputs[0]),
        )

        for key, filename, cross_validated, plot_number, label in (
            (
                "reference_corrected_reward",
                corrected_outputs[1],
                False,
                7,
                "all trials",
            ),
            (
                "reference_corrected_reward_cv",
                corrected_outputs[2],
                True,
                8,
                "even/odd cross-validated",
            ),
        ):
            print(
                f"Plot {plot_number}/{n_plots}: {corrected_label} peri-reward "
                f"heatmap ({label})..."
            )
            try:
                figures[key] = plot_reward_heatmap(
                    corrected,
                    data["reward_frames"],
                    data["frame_rate"],
                    window_size=reward_window,
                    cross_validated=cross_validated,
                    signal_label=corrected_label,
                    save_path=output_path(filename),
                )
            except ValueError as error:
                print(f"  skipping plot: {error}")
                figures[key] = None
                skipped_output = output_path(filename)
                if skipped_output:
                    Path(skipped_output).unlink(missing_ok=True)

        print(
            f"Plot 9/{n_plots}: selected-cell {corrected_label} reward PSTHs bounded "
            "by neighboring landmarks..."
        )
        try:
            figures["reference_corrected_reward_psth_selected_cells"] = (
                plot_selected_cell_reward_psths(
                    corrected,
                    trace_cell_indices,
                    data["reward_frames"],
                    data["landmark_entries"],
                    data["landmark_exits"],
                    data["frame_rate"],
                    window_size=reward_psth_window,
                    cell_labels=data["neurons"][trace_cell_indices],
                    signal_label=corrected_label,
                    save_path=output_path(corrected_outputs[3]),
                )
            )
        except ValueError as error:
            print(f"  skipping plot: {error}")
            figures["reference_corrected_reward_psth_selected_cells"] = None
            skipped_output = output_path(corrected_outputs[3])
            if skipped_output:
                Path(skipped_output).unlink(missing_ok=True)

    behaviour_start = n_imaging_plots + 1
    context = data["behaviour_analysis"]
    print(
        f"Plot {behaviour_start}/{n_plots}: raw VR versus aligned behaviour QC..."
    )
    figures["behaviour_alignment"] = plot_behaviour_alignment(
        data, save_path=output_path(BEHAVIOUR_OUTPUT_NAMES[0][:-4])
    )
    print(
        f"Plot {behaviour_start + 1}/{n_plots}: "
        "raw/aligned speed and lick-rate PSTHs..."
    )
    figures["behaviour_speed_lick_psth"] = plot_behaviour_psth_comparison(
        context, save_path=output_path(BEHAVIOUR_OUTPUT_NAMES[1][:-4])
    )
    print(
        f"Plot {behaviour_start + 2}/{n_plots}: raw/aligned transition matrices..."
    )
    figures["behaviour_transition_matrix"] = (
        plot_behaviour_transition_comparison(
            context, save_path=output_path(BEHAVIOUR_OUTPUT_NAMES[2][:-4])
        )
    )
    print(
        f"Plot {behaviour_start + 3}/{n_plots}: raw/aligned sliding performance..."
    )
    figures["behaviour_sliding_performance"] = plot_behaviour_sliding_performance(
        context, save_path=output_path(BEHAVIOUR_OUTPUT_NAMES[3][:-4])
    )

    if save_dir:
        remove_legacy_overview_plots(save_dir)
    if show:
        plt.show()
    return data, figures


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mouse", help='e.g. "07" or "sub-07"')
    parser.add_argument("session_id", help='full or short identifier, e.g. "full001"')
    parser.add_argument("project_root", help="sequence-compression project root")
    parser.add_argument(
        "--show",
        action="store_true",
        help="display figures interactively (default off for unattended runs)",
    )
    args = parser.parse_args()
    run_overview(
        args.mouse,
        args.session_id,
        args.project_root,
        show=args.show,
    )


if __name__ == "__main__":
    _cli()

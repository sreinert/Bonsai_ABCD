# Single-session registration and ROI detection

This directory is the non-grid workflow for testing the steps after
registration. It selects exactly one session from an earlier registration
tuning manifest, registers the full session with the locked parameters in
`config.py`, runs default Suite2p Cellpose detection (`cpsam`), and creates a
visual ROI report. It does not run signal extraction, ROI classification, or
spike deconvolution.

Raw TIFFs are read-only. Registration keeps `data.bin` and `data_chan2.bin`;
only generated `data_raw*.bin` files may be removed after successful output
validation.

## 1. List the tested sessions

Use the manifest from the registration-parameter run you want to draw from:

```bash
python preprocessing/suite2p/registration_tuning/remaining_steps/prepare_session.py \
  --source-manifest /absolute/path/to/tasks.csv \
  --list
```

The command collapses multiple registration candidates for the same raw
session into one row and prints a stable `index`. Review the mouse and session
columns, then use that index in the submission command.

## 2. Submit one session

Choose a new processed output root that does not overlap the raw `funcimg`
directory:

```bash
bash preprocessing/suite2p/registration_tuning/remaining_steps/submit_single_session.sh \
  /absolute/path/to/tasks.csv \
  SESSION_INDEX \
  /absolute/path/to/processed/cohort2/suite2p_remaining_steps
```

This creates a one-row manifest and submits exactly two jobs:

1. full-session registration with the chosen parameters;
2. ROI detection, submitted with an `afterok` dependency on registration.

There is no job array and no parameter grid.

## 3. Review the detected ROIs

The submission command prints the exact report path:

```text
<run_dir>/roi_detection/cpsam/index.html
```

The most useful visual files are:

- `qc.png`: Cellpose input, ROI outlines, registered mean image, and ROI-size
  distribution;
- `cellpose_input_overlay.png`: outlines on the exact image given to Cellpose;
- `mean_image_overlay.png`: outlines on the registered channel-1 mean image.

The same directory contains `stat.npy`, `roi_masks.npy`, ROI area and diameter
arrays, exact software/model provenance, and `status.json`. The detection job
fails if it finds or creates `F.npy`, `Fneu.npy`, `iscell.npy`, or `spks.npy`.

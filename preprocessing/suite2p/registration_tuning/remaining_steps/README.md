# Single-session registration and ROI detection

This directory is the non-grid workflow for testing the steps after
registration. It uses an earlier registration-tuning manifest only to identify
one mouse/session. It then rediscovers every direct TIFF in that raw session's
`funcimg` directory, registers the full session from scratch with the locked
parameters in `config.py`, runs default Suite2p Cellpose detection (`cpsam`),
and creates a visual ROI report. It never reuses a tuning candidate's
registered output or candidate parameters. It does not run signal extraction,
ROI classification, or spike deconvolution.

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

The command collapses multiple registration-candidate rows for the same raw
session into one row and prints a stable `index`. Those candidates are not
separate input runs; review the mouse and session columns, then use that index
in the submission command.

## 2. Submit one session

By default, output is written to a `remaining_steps` directory beside the
source manifest. For example, a source manifest under
`_suite2p_registration_tuning/cohort2` produces output under
`_suite2p_registration_tuning/cohort2/remaining_steps`, not under `processed`:

```bash
bash preprocessing/suite2p/registration_tuning/remaining_steps/submit_single_session.sh \
  /absolute/path/to/tasks.csv \
  SESSION_INDEX
```

An optional third argument overrides the output root. It must not overlap the
raw `funcimg` directory. Because `data.bin` and `data_chan2.bin` are retained,
the selected location must have enough space for both full registered movies.

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

## Inspect a registration failure and its exact settings

Registration writes `resolved_suite2p_configuration.json` before calling
Suite2p. It contains the complete resolved settings, including defaults—not
just the explicit overrides. To display the failure and registration settings:

```bash
python preprocessing/suite2p/registration_tuning/remaining_steps/inspect_run.py \
  --run-dir /absolute/path/to/chosen-registration-cpsam
```

The same information remains available in `provenance.json`. Slurm stderr and
stdout are stored under `<output-root>/.workflow_manifests/logs`.

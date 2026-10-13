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

All registration and detection submissions exclude `gpu-350-02` and
`gpu-350-04` by default because those nodes have repeatedly failed jobs or
Ceph access. To exclude additional known-bad nodes, pass a
comma-separated Slurm node list:

```bash
EXCLUDE_NODES=gpu-350-02,gpu-350-04,gpu-OTHER-NODE \
  bash preprocessing/suite2p/registration_tuning/remaining_steps/submit_single_session.sh \
  /absolute/path/to/tasks.csv SESSION_INDEX
```

The launcher prints the applied exclusion list. Setting `EXCLUDE_NODES` to an
empty string explicitly disables the default exclusion.

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

## Repeat detection without registering again

After registration has succeeded, use `submit_detection_experiment.sh` for
each named detection attempt. This submits one detection job only. It reads the
existing `data.bin`; it neither submits registration nor changes the registered
binaries. Every experiment must have a new name because prior results are never
overwritten.

For example, rerun Cellpose-SAM on the binned mean image:

```bash
bash preprocessing/suite2p/registration_tuning/remaining_steps/submit_detection_experiment.sh \
  /absolute/path/to/chosen-registration-cpsam \
  cpsam-meanimg \
  --model cpsam \
  --input-image meanImg
```

`meanImg` above matches Suite2p's normal detection path: it is the mean of the
temporally binned movie, cropped to the valid registered field. To instead use
the full-session mean image saved by registration in `reg_outputs.npy` (cropped
to the same valid field), select `registration_meanImg`:

```bash
bash preprocessing/suite2p/registration_tuning/remaining_steps/submit_detection_experiment.sh \
  /absolute/path/to/chosen-registration-cpsam \
  cpsam-registration-meanimg \
  --model cpsam \
  --input-image registration_meanImg
```

This option reads the saved registration mean directly and skips temporal
binning of `data.bin`. Its exact input is saved as both `cellpose_input.npy`
and `registration_meanImg_crop.npy`.

Or make Cellpose more permissive and change the assumed cell diameter:

```bash
bash preprocessing/suite2p/registration_tuning/remaining_steps/submit_detection_experiment.sh \
  /absolute/path/to/chosen-registration-cpsam \
  cpsam-meanimg-d10-prob-minus1 \
  --model cpsam \
  --input-image meanImg \
  --diameter-y 10 --diameter-x 10 \
  --cellprob-threshold -1
```

Additional `CellposeModel.eval` settings can be supplied with repeatable
`--cellpose-param NAME=JSON_VALUE` options, for example
`--cellpose-param min_size=5`. The exact settings and model file/hash are saved
in the experiment directory.

The old Suite2p `anatomical_only=2` setting corresponds here to Cellpose
detection with `--input-image meanImg`. `connected` is a parameter of
Suite2p's functional sourcery detector and has no effect on Cellpose, so this
launcher deliberately does not pretend to apply it.

Cellpose 4.2.1.1 cannot run the legacy `cyto3` network. Use a separate
environment with Cellpose 3.1.1.2 so the existing registration environment is
not changed. One way to make it is:

```bash
conda create --clone suite2p-reg-1.1.0 -n suite2p-cellpose3
conda activate suite2p-cellpose3
python -m pip install 'cellpose==3.1.1.2'
python -c 'import importlib.metadata as m; print(m.version("suite2p"), m.version("cellpose"))'
```

Then submit with that environment's Python executable:

```bash
PYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  bash preprocessing/suite2p/registration_tuning/remaining_steps/submit_detection_experiment.sh \
  /absolute/path/to/chosen-registration-cpsam \
  cyto3-meanimg \
  --model cyto3 \
  --input-image meanImg
```

The runner checks the installed model list and stops instead of allowing
Cellpose to silently substitute another model. Expensive binned mean/max
images are cached under `roi_detection/prepared_images` and reused across
experiments with the same binning settings. Each attempt writes a report to:

```text
<run_dir>/roi_detection/experiments/<experiment-name>/index.html
```

A valid experiment that detects nothing completes with
`"outcome": "zero_rois"` and still saves its input image, empty masks, settings,
and QC report. It does not produce `F.npy`, `Fneu.npy`, `iscell.npy`, or
`spks.npy`.

# Full-session pipeline: one command, one Slurm job

The complete production workflow is submitted with only a mouse ID and the
exact raw session directory name:

```bash
bash preprocessing/suite2p/registration_tuning/remaining_steps/submit_full_session_pipeline.sh \
  sub-XX \
  ses-NAME_date-YYYYMMDDTHHMMSS
```

The launcher submits one 24-hour GPU job. That newly allocated job runs full
registration, default CPSAM ROI detection, two-channel trace extraction,
classification, deconvolution, and combined QC sequentially. The submission
shell does not run analysis on its current host. Slurm sends `BEGIN`, `END`,
and `FAIL` mail to `athina.apostolelli.24@ucl.ac.uk`; set
`SLURM_MAIL_USER` to override the address.

The default GPU exclusion is exactly `gpu-350-02,gpu-350-04`. Set
`EXCLUDE_NODES` to override it. `PROJECT_ROOT`, `MRSIC_PROJECTS_ROOT`, and
`PYTHON_EXECUTABLE` retain their existing meanings.

For a raw input under:

```text
<project>/rawdata/cohort2/<mouse>/<full-session-id>/funcimg
```

all standard Suite2p products are written directly to:

```text
<project>/processed/cohort2/<mouse>/<short-session-id>/suite2p/plane0
```

This includes `stat.npy`, `iscell.npy`, `spks.npy`, `F.npy`, `Fneu.npy`, both
channel-2 trace arrays, `ops.npy`, and the retained registered binaries. QC
HTML, PNGs, and supporting QC metrics are kept under `suite2p/qc`; manifests,
status, provenance, and Slurm logs are kept under `suite2p/.workflow`. The
pipeline refuses partial or inconsistent existing Suite2p output and treats a
fully completed run as a no-op.

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

QC images use 1st–99th percentile contrast for display only. This makes the
Cellpose input and registration mean easier to compare without changing any
values supplied to detection. To update an experiment completed with an older
version of the workflow, regenerate only its QC files:

```bash
python preprocessing/suite2p/registration_tuning/remaining_steps/refresh_detection_qc.py \
  --experiment-dir \
  /absolute/path/to/chosen-registration-cpsam/roi_detection/experiments/cpsam-registration-meanimg
```

This replaces only the generated `qc.png`, `cellpose_input.png`, `index.html`,
and ROI summary arrays in that experiment directory. It does not run Cellpose,
read raw TIFFs, or change registered binaries or detected masks.

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

## 4. Extract traces from the reviewed CPSAM ROIs

Once the default CPSAM report has been reviewed, submit trace extraction for
that already-registered run:

```bash
bash preprocessing/suite2p/registration_tuning/remaining_steps/submit_trace_extraction.sh \
  /absolute/path/to/chosen-registration-cpsam
```

The job reads `roi_detection/cpsam/stat.npy`, `data.bin`, and
`data_chan2.bin`. It does not rerun registration or ROI detection and opens
both registered binaries read-only. Results are written to the separate
directory:

```text
<run_dir>/trace_extraction/cpsam
```

That directory contains the standard Suite2p `F.npy`, `Fneu.npy`,
`F_chan2.npy`, `Fneu_chan2.npy`, `stat.npy`, `iscell.npy`, `spks.npy`, and
`ops.npy` outputs. The built-in Suite2p classifier is selected explicitly so
results do not depend on a user-specific classifier in the cluster home
directory. `status.json` and `provenance.json` record the exact detection
input hash, settings, software versions, ROI/cell/frame counts, timings, and
registered-binary metadata before and after extraction.

The extraction job refuses to overwrite either an earlier trace result or
standard trace files in the registered `suite2p/plane0` directory. To use a
different reviewed detection experiment, pass its directory as the second
argument. An optional third argument selects another isolated output folder:

```bash
bash preprocessing/suite2p/registration_tuning/remaining_steps/submit_trace_extraction.sh \
  /absolute/path/to/chosen-registration-cpsam \
  /absolute/path/to/chosen-registration-cpsam/roi_detection/experiments/NAME \
  /absolute/path/to/chosen-registration-cpsam/trace_extraction/NAME
```

## 5. Generate a two-channel trace QC report

After trace extraction completes, submit the report to the HPC CPU partition:

```bash
bash preprocessing/suite2p/registration_tuning/remaining_steps/submit_trace_qc.sh \
  /absolute/path/to/chosen-registration-cpsam/trace_extraction/cpsam-meanimg
```

The launcher checks that trace extraction has state `complete`, verifies both
channels, checks the Python environment, and prints the Slurm job ID and log
paths. The CPU job requests four cores, 16 GB RAM, and two hours. It does not
modify the extracted arrays. When it finishes, open:

```text
<trace-dir>/qc/index.html
```

As with the other launchers, set `PYTHON_EXECUTABLE` if the active shell is not
already using the intended environment:

```bash
PYTHON_EXECUTABLE=/path/to/environment/bin/python \
  bash preprocessing/suite2p/registration_tuning/remaining_steps/submit_trace_qc.sh \
  /absolute/path/to/chosen-registration-cpsam/trace_extraction/cpsam-meanimg
```

Options after the trace directory are passed to the report generator. For
example, select eight cells and write a new report name:

```bash
bash preprocessing/suite2p/registration_tuning/remaining_steps/submit_trace_qc.sh \
  /absolute/path/to/chosen-registration-cpsam/trace_extraction/cpsam-meanimg \
  --n-cells 8 \
  --output-dir \
  /absolute/path/to/chosen-registration-cpsam/trace_extraction/cpsam-meanimg/qc-eight-cells
```

The output directory must be new because the generator refuses to overwrite a
previous report. To run the same calculation interactively instead of through
Slurm:

```bash
python preprocessing/suite2p/registration_tuning/remaining_steps/generate_trace_qc.py \
  --trace-dir \
  /absolute/path/to/chosen-registration-cpsam/trace_extraction/cpsam-meanimg
```

The command writes `qc/index.html`, five PNG figures, per-ROI metrics as CSV,
and the exact calculation settings as JSON. It refuses to overwrite an
existing QC directory.

Green and red traces use `F - neucoeff * Fneu`. The G/R calculation matches
`cellTV/cellTV_functions_cohort3.py`: only red is median-filtered over 0.25 s,
and the denominator is floored at its per-ROI fifth percentile (with an
absolute floor of `1e-3`). The report shows both the exact CellTV-style
baseline-subtracted output (`signal - baseline`) and normalized
`(signal - baseline) / abs(baseline)` for green, red, and G/R. This makes clear
that the current CellTV helper labels a baseline-subtracted ΔF result as dF/F.
The report also shows how often the red denominator floor was used so
unreliable ratios can be identified.

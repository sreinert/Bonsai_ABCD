# Suite2p full-session pipeline

This pipeline runs registration, CPSAM cell detection, trace extraction, and
QC for one cohort-2 session. It expects `suite2p==1.1.0` and a CUDA-enabled
environment.

From the repository root, submit one Slurm job with:

```bash
bash preprocessing/suite2p/submit_full_session_pipeline.sh \
  sub-02 \
  ses-full002_date-20260818T134310
```

To submit one independent Slurm job for each of several sessions from the same
mouse, provide all full session names after the mouse ID:

```bash
bash preprocessing/suite2p/submit_full_session_pipeline.sh \
  sub-02 \
  ses-full005_date-20260821T131725 \
  ses-full006_date-20260822T130410 \
  ses-full007_date-20260823T125955
```

The launcher validates and submits every session separately, continues to the
remaining sessions if one submission fails, and prints a batch summary. Duplicate
session names are rejected to prevent accidental duplicate jobs. Options such as
`--resume` or `--overwrite` apply to every session in that command.

To run on the GPU node you already allocated, add `--current-node`:

```bash
bash preprocessing/suite2p/submit_full_session_pipeline.sh \
  --current-node \
  sub-02 \
  ses-full002_date-20260818T134310
```

If registration finished but a later stage failed, add `--resume`. To discard
the selected session's existing Suite2p output and start again, add
`--overwrite`. These two options cannot be combined.

By default the project is
`/ceph/mrsic_flogel/public/projects/AtAp_20260119_SequenceCompression`.
Override it with `PROJECT_ROOT`. The input and output layouts are:

```text
rawdata/cohort2/<mouse>/<full-session-id>/funcimg
processed/cohort2/<mouse>/<short-session-id>/suite2p
```

When one or more direct TIFF files whose names end in `_cropped.tif` or
`_cropped.tiff` are present in the raw `funcimg` directory, the pipeline uses
only those cropped files. If none are present, it retains the existing behavior
and uses all direct TIFF files in `funcimg`. Registration records the selected
files in its manifest, and every downstream stage uses the binaries produced
from that registration.

Standard Suite2p arrays are saved in `suite2p/plane0`. The combined HTML and
PNG report is saved in `suite2p/qc`, and logs and provenance are saved in
`suite2p/.workflow`.

Trace QC uses the raw-fluorescence signal-quality definitions imported from
`preprocessing/quality_control` (baseline fluorescence, SNR, skew and detected
event fraction) and plots the highest-SNR cells automatically. Neuropil-corrected
green/red/G/R and CellTV-style traces are retained as a separate supplementary
section. The original YAML-driven comparison commands remain available:

```bash
python preprocessing/quality_control/compare_signal_quality.py SETTINGS.yaml
python preprocessing/quality_control/plot_high_snr_traces.py SETTINGS.yaml
```

Normally only `submit_full_session_pipeline.sh` is called directly. Its
implementation is kept in the `full_session_pipeline` subfolder.

## Find the session for a Slurm job

Look up the processed-data session from either the numeric Slurm job ID or the
exact job name shown by `squeue`/`sacct`:

```bash
python preprocessing/suite2p/find_slurm_session.py 12345678
python preprocessing/suite2p/find_slurm_session.py s2p__sub-02__full002
```

The command prints the processed session directory and its Suite2p output
directory. It checks the pipeline's saved `submission.json` provenance and, when
available, the Slurm working directory reported by `squeue` or `sacct`. Future
full-session submissions use names such as `s2p__sub-02__full002`, which are
unique to a mouse/session. Older jobs used the shared name `s2p-full-session`,
so use their numeric job ID when that name matches more than one session.

## Session inventory

Create one CSV per selected cohort-2 mouse with:

```bash
python preprocessing/suite2p/create_session_inventory.py \
  --mouse-ids sub-02 sub-03
```

This creates `suite2p_session_inventory.csv` inside each processed mouse folder,
for example `processed/cohort2/sub-02/suite2p_session_inventory.csv`. The default
project is `AtAp_20260119_SequenceCompression`, resolved automatically under
`/ceph/mrsic_flogel/public/projects` on the cluster or
`/Volumes/mrsic_flogel/public/projects` on macOS. Set `MRSIC_PROJECTS_ROOT` to
override the projects mount.
Sessions containing `funcimg` are discovered under `rawdata/cohort2`, then their
Suite2p results are read from the matching short-name directory under
`processed/cohort2`. Use `--project-root PATH` for another project, or override
the two locations separately with `--session-root` and `--suite2p-root`. Use
`--output-dir PATH` only when all per-mouse CSVs should instead be placed in one
common directory. Only visible, immediate session directories containing
`funcimg` are included. The legacy `SESSION/funcimg/suite2p/plane0` layout is
also recognised.

Registration requires non-empty `ops.npy` and `data.bin`; detection requires a
readable `stat.npy`; and trace extraction requires readable, ROI-count-consistent
`F.npy`, `Fneu.npy`, `spks.npy`, and `iscell.npy`. ROI and cell counts are left
blank when their corresponding stage is incomplete. Existing CSVs are protected
unless `--overwrite` is supplied. `suite2p_path_created` reports whether the
session's Suite2p directory exists. `registration_status` is `succeeded` when
the registration outputs are complete, `failed` when the Suite2p directory
exists without those outputs, and `not_started` when no Suite2p directory exists.
`tiff_files_present` is `yes` when `funcimg` contains at least one direct,
non-hidden `.tif` or `.tiff` file and `no` when the `funcimg` directory is empty
or contains no TIFF files. Sessions without TIFF files remain in the inventory.
`h5_files_present` similarly reports whether `funcimg` contains at least one
direct, non-hidden `.h5` file (case-insensitive).
`meanImg_chan1_seg_created` reports whether a manual Cellpose
`registration/meanImg_chan1_seg.npy` exists (the current pipeline location is
`suite2p/qc/registration`). `iscell_updated` compares the modification times of
`iscell.npy` and `stat.npy`. It reports `yes` only when `iscell.npy` is
strictly newer than `stat.npy`, `no` when both exist but it is not newer, and is
blank when either file is missing.
`automatic_roi_detection_model` reads the recorded automatic detection
provenance and reports `cpsam` or `cpsam_v2`. It is blank when no supported
automatic model can be identified; manual GUI segmentation alone does not set
this field.

## Import a reviewed Cellpose segmentation

After saving Cellpose's reviewed segmentation as
`suite2p/qc/registration/meanImg_chan1_seg.npy`, replace Suite2p's detected ROIs
and regenerate all ROI-dependent traces with:

```bash
python preprocessing/suite2p/Convert_seg_to_stat.py \
  sub-02 \
  ses-full002_date-20260818T134310
```

The short processed session ID (for example `full002`) is also accepted. Use
`--dry-run` first to validate the resolved paths, mask dimensions, and ROI count
without changing files. The command reads the registered `data.bin` and, when
present, `data_chan2.bin`; it updates `stat.npy`, fluorescence/neuropil traces,
`iscell.npy`, and `spks.npy` together. Previous outputs are retained under
`suite2p/.workflow/manual_segmentation_backups`. On Slurm, the equivalent GPU
submission is:

```bash
sbatch preprocessing/suite2p/convert_seg_to_stat.sh \
  sub-02 \
  ses-full002_date-20260818T134310
```

By default, the script looks for the segmentation in `suite2p/qc/registration`.
For the previous layout, where the mean image and its Cellpose segmentation were
saved directly in the Suite2p plane output folder, use:

```bash
python preprocessing/suite2p/Convert_seg_to_stat.py \
  sub-02 \
  ses-full002_date-20260818T134310 \
  --meanimg-folder plane0
```

`--meanimg-folder` may also be an absolute directory. If the segmentation has a
nonstandard filename, pass its complete path with `--segmentation` instead.

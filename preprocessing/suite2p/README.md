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

## Session inventory

Create one CSV per selected cohort-2 mouse with:

```bash
python preprocessing/suite2p/create_session_inventory.py \
  --mouse-ids sub-02 sub-03
```

This creates `suite2p_session_inventory.csv` inside each processed mouse folder,
for example `processed/cohort2/sub-02/suite2p_session_inventory.csv`. The default
project is `/Volumes/mrsic_flogel/public/projects/AtAp_20260119_SequenceCompression`.
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

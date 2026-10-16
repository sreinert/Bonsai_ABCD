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

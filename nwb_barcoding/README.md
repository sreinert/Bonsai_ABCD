# Robust Bonsai-to-imaging alignment

This package ports the slope-corrected barcode alignment pipeline from
`Sandra_analysis`.  It builds separate behaviour and imaging NWBs, aligns the
behaviour clock to the vDAQ clock using shared barcodes, writes a merged NWB, and
selects the imaging frames that overlap behaviour.

The conversion interfaces are imported from the sibling `Aimbot_barcoding`
repository.  Use the pinned environment in `pyproject.toml`/`uv.lock` for production
runs.

## Sequence-compression cohort 2

The adapter understands the split layout used by this project:

```text
rawdata/cohort2/<subject>/<full-session>/
processed/cohort2/<subject>/<full-session>/suite2p/
BehaviourAligned_robust/cohort2/<subject>/<full-session>/
```

Run one session by full or short selector. A short selector is resolved against
`rawdata`; the resulting full folder name is then used under `processed`:

```bash
python -m nwb_barcoding.sequence_compression \
  sub-02 full002 \
  /Volumes/mrsic_flogel/public/projects/AtAp_20260119_SequenceCompression
```

Use `--output-root PATH` to override the isolated output root and `--force` to
rebuild existing products. Source raw data and Suite2p outputs are read-only.
Temporary dF/F input is created in a system temporary directory and removed
after the imaging NWB is complete, avoiding writes to both the source trees and
the network output tree. Its neuropil coefficient is read from each plane's
Suite2p `ops.npy`; no coefficient is assumed by the builder.

Each completed output session contains:

```text
funcimg/2p_preprocessed_*.nwb
behav/vrabcd_preprocessed_*.nwb
vrabcd_2p_preprocessed_*.nwb
behaviour_data.npz
behaviour_data.csv
valid_frames.npz
```

## Slurm stage 1: barcode alignment

The alignment launcher discovers completed Suite2p `plane0` outputs for the explicitly
selected subject and submits one independent CPU-array task per session. Partially
preprocessed sessions are left out. Each task builds the NWBs, aligns barcodes, creates the
behavior-frame products, runs verification, and then stops; it does not generate plots.

The launcher automatically uses
`/ceph/python_envs/$USER/nwb_barcoding/bin/python`; environment activation is not
required. After creating and syncing that environment, run:

```bash
bash nwb_barcoding/hpc/submit_sequence_compression_batch.sh \
  /ceph/mrsic_flogel/public/projects/AtAp_20260119_SequenceCompression \
  sub-07
```

The `Aimbot_barcoding` repository must be a sibling of this repository. The launcher
validates Python 3.12 and all required imports before submitting anything. The subject may
also be written as `07`. A different interpreter can still be supplied as the optional
third argument or through `PYTHON_EXECUTABLE`.

Alignment outputs are written to:

```text
BehaviourAligned_robust/cohort2/<subject>/<full-session>/
```

To deliberately submit multiple completed mice together, use a comma-separated argument:

```bash
bash nwb_barcoding/hpc/submit_sequence_compression_batch.sh \
  /ceph/mrsic_flogel/public/projects/AtAp_20260119_SequenceCompression \
  sub-02,sub-07
```

Useful submission environment variables:

- `MAX_CONCURRENT=4` controls the array concurrency limit.
- `TASK_RANGE=0-9%2` submits only a manifest subset.
- `SLURM_PARTITION=cpu` overrides the CPU partition.
- `SLURM_MAIL_USER=name@example.org` enables begin/end/failure email.
- `FORCE_REBUILD=true` rebuilds existing NWBs in each worker; the default is resumable.

For a non-Slurm serial alignment run, use:

```bash
python -m nwb_barcoding.batch_sequence_compression \
  --project-root /path/to/AtAp_20260119_SequenceCompression \
  --include-subject sub-07 \
  --all
```

## Slurm stage 2: session overviews

The separate overview launcher discovers sessions under `BehaviourAligned_robust` that
contain exactly one merged NWB plus `behaviour_data.npz` and `valid_frames.npz`. It does not
run NWB conversion or barcode alignment. Each array task runs only
`analysis/sequence_compression/session_overview_sc.py` and writes plots to:

```text
analysis/cohort2/<subject>/<full-session>/
```

The first overview run also performs the aligned two-channel extraction from the
matching Suite2p traces. The G/R trace reproduces CellTV `get_dff_GR`, including its
0.7 neuropil coefficient, red median filter, and fifth-percentile denominator floor.
It saves its caches under
`BehaviourAligned_robust/cohort2/<subject>/<full-session>/funcimg/`:

```text
celltv_green_delta_v4.npy
celltv_red_delta_v4.npy
celltv_green_red_ratio_delta_v4.npy
celltv_green_red_ratio_diagnostics_v4.npz
valid_neurons.npy
```

`valid_neurons.npy` contains original Suite2p ROI indices returned by CellTV's
`get_responsive_neurons`; both the high-green-SNR and reproducibly random selected-cell
plots are sampled only from ROIs that are also classified as cells. Existing complete
files are reused.

The overview stage uses the existing `suite2p-reg-1.1.0` conda environment rather
than the barcode-alignment uv environment. Install its small set of additional
analysis dependencies once:

```bash
module load mamba
source activate suite2p-reg-1.1.0
python -m pip install -r \
  analysis/sequence_compression/requirements-session-overview.txt
```

The submission launcher activates `suite2p-reg-1.1.0` automatically and validates
Python 3.11, Suite2p 1.1.0, and all overview imports before submitting jobs. Set
`SESSION_OVERVIEW_ENV` to override the conda environment name, or pass an explicit
Python executable as the optional third positional argument.

Submit overviews independently after alignment jobs have finished:

```bash
bash analysis/sequence_compression/submit_session_overview_sc_batch.sh \
  /ceph/mrsic_flogel/public/projects/AtAp_20260119_SequenceCompression \
  sub-07
```

The overview launcher accepts the same subject syntax, Python override,
`MAX_CONCURRENT`, `TASK_RANGE`, `SLURM_PARTITION`, and `SLURM_MAIL_USER` options as the
alignment launcher. Its manifests and logs are stored separately under
`analysis/cohort2/.session_overview_manifests`.

For a non-Slurm serial overview run, use:

```bash
python -m analysis.sequence_compression.batch_session_overview_sc \
  --project-root /path/to/AtAp_20260119_SequenceCompression \
  --include-subject sub-07 \
  --all
```

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
processed/cohort2/<subject>/<short-session>/suite2p/
BehaviourAligned_robust/cohort2/<subject>/<full-session>/
```

Run one session by full or short identifier:

```bash
python -m nwb_barcoding.sequence_compression \
  sub-02 full002 \
  /Volumes/mrsic_flogel/public/projects/AtAp_20260119_SequenceCompression
```

Use `--output-root PATH` to override the isolated output root and `--force` to
rebuild existing products. Source raw data and Suite2p outputs are read-only.
Temporary dF/F input is created in a system temporary directory and removed
after the imaging NWB is complete, avoiding writes to both the source trees and
the network output tree.

Each completed output session contains:

```text
funcimg/2p_preprocessed_*.nwb
behav/vrabcd_preprocessed_*.nwb
vrabcd_2p_preprocessed_*.nwb
behaviour_data.npz
behaviour_data.csv
valid_frames.npz
```

## Slurm batch: alignment followed by session overview

The batch launcher discovers completed Suite2p `plane0` outputs for the explicitly selected
subject and submits one independent CPU-array task per session. Partially preprocessed
sessions are left out. Within each task, barcode alignment and all verification checks must
succeed before `musicboxes/session_overview.py` runs.

The launcher automatically uses
`/ceph/python_envs/$USER/nwb_barcoding/bin/python`; environment activation is not
required. After creating and syncing that environment, run:

```bash
bash nwb_barcoding/hpc/submit_sequence_compression_batch.sh \
  /ceph/mrsic_flogel/public/projects/AtAp_20260119_SequenceCompression \
  sub-07
```

The `Aimbot_barcoding` repository must be a sibling of this repository, as for individual
NWB builds. The launcher validates Python 3.12 and all required imports before submitting
anything. The subject may also be written as `07`. A different interpreter can still be
supplied as the optional third argument or
through `PYTHON_EXECUTABLE`.

Outputs are written to:

```text
BehaviourAligned_robust/cohort2/<subject>/<full-session>/
analysis/cohort2/<subject>/<full-session>/
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

For a non-Slurm serial run, use:

```bash
python -m nwb_barcoding.batch_sequence_compression \
  --project-root /path/to/AtAp_20260119_SequenceCompression \
  --include-subject sub-07 \
  --all
```

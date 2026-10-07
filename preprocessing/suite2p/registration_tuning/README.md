# Suite2p registration tuning plan

This repository contains analysis and preprocessing code for two-photon calcium
imaging experiments in mice. The immediate development goal is to select one
robust set of Suite2p registration parameters that can be used across all mice,
while keeping the computational search practical on the HPC cluster.

## Suite2p registration optimisation plan

### Objectives

The registration evaluation should:

- use the same Suite2p version and parameter set for every mouse;
- screen parameter candidates on the available short TIFF stacks before committing
  resources to complete recordings of approximately 81,000 frames;
- run on the Slurm HPC cluster against data on network storage;
- combine quantitative diagnostics with visual inspection of registered data;
- favour settings that work consistently across mice rather than settings that
  only perform exceptionally well on one recording; and
- preserve complete provenance so that the selected configuration can be
  reproduced later.

### Suite2p version

Use the most recent stable Suite2p release and pin it exactly in a dedicated
HPC environment. At the time this plan was written (7 October 2026), the latest
stable PyPI release is [`suite2p==1.1.0`](https://pypi.org/project/suite2p/).

The new workflow should use the Suite2p 1.x API, which separates recording
information in `db` from nested pipeline `settings`. The existing
[`../run_suite2p.py`](../run_suite2p.py)
uses the older, flat `ops` API and should remain unchanged until the new
workflow has been validated. The runner must record the installed Suite2p,
Python, PyTorch, and CUDA versions for every job and fail if the Suite2p version
does not match the pinned version.

Because PyTorch GPU builds must be compatible with the cluster CUDA stack, the
cluster-recommended PyTorch installation should be installed before Suite2p.
Once the environment is working, its complete package lock should be saved.

### HPC setup

Create a dedicated environment on the cluster. Install the cluster-supported
CUDA build of PyTorch first, then install the pinned workflow requirements:

```bash
module load mamba
mamba create -n suite2p-reg-1.1.0 python=3.11 pip -y
source activate suite2p-reg-1.1.0

# Install the cluster-recommended CUDA/PyTorch build here, then:
python -m pip install -r preprocessing/suite2p/registration_tuning/requirements.txt
python -c "from importlib.metadata import version; import torch; print(version('suite2p'), torch.cuda.is_available())"
```

Do not proceed with GPU jobs unless this prints Suite2p `1.1.0` and CUDA
availability `True` on a GPU compute node.

## Running the screen

All commands below are run from the repository root. The example output path is
deliberately separate from every mouse/session input directory.

### 1. Discover and lock the selected sessions

```bash
if [[ -d /ceph/mrsic_flogel/public/projects ]]; then
  PROJECTS_ROOT=/ceph/mrsic_flogel/public/projects
else
  PROJECTS_ROOT=/Volumes/mrsic_flogel/public/projects
fi

TUNING_ROOT=${PROJECTS_ROOT}/AtApSuKuSaRe_20250129_HFScohort2/_suite2p_registration_tuning

python preprocessing/suite2p/registration_tuning/discover_sessions.py \
  --data-root AtApSuKuSaRe_20250129_HFScohort2 \
  --output "${TUNING_ROOT}/selected_sessions.csv" \
  --seed 20261007
```

This checks each selected TIFF's header without loading pixels or walking all
TIFF pages. It produces:

- `selected_sessions.csv`, the immutable first/random-middle/last selection;
- `selected_sessions_audit.csv`, including rejected sessions and reasons.

Review both files before submitting any registration jobs. To restrict the
scan to known mice, repeat `--mouse`, for example:

```bash
--mouse TAA0000061 --mouse TAA0000066
```

The defaults assume mouse directories named `TAA*`, session directories named
`ses-*`, and inputs below `funcimg/others`.

`--data-root` can be absolute or relative to the MRSIC projects directory. A
relative value uses `/ceph/mrsic_flogel/public/projects` when that mount exists,
otherwise `/Volumes/mrsic_flogel/public/projects`. Set `MRSIC_PROJECTS_ROOT` to
override both locations. This means the same relative `--data-root` argument
works locally and on the HPC.

Paths saved in manifests are portable between the two standard mounts. The
runner and evaluator preserve everything after `public/projects` and remap an
unavailable `/Volumes/...` prefix to `/ceph/...`, or vice versa. Therefore a
manifest created locally can be used on the HPC without editing its CSV path
fields. The path used to invoke the manifest itself must still be valid on the
current machine.

For Sequence Compression cohort 2, whose layout is
`rawdata/cohort2/sub-XX/ses-..._date-YYYYMMDDTHHMMSS/funcimg/others`, use:

```bash
DATA_ROOT=AtAp_20260119_SequenceCompression/rawdata/cohort2
TUNING_ROOT=${PROJECTS_ROOT}/AtAp_20260119_SequenceCompression/_suite2p_registration_tuning/cohort2

python preprocessing/suite2p/registration_tuning/discover_sessions.py \
  --data-root "${DATA_ROOT}" \
  --mouse-glob 'sub-*' \
  --session-glob 'ses-*' \
  --others-relative 'funcimg/others' \
  --nchannels 2 \
  --output "${TUNING_ROOT}/selected_sessions.csv" \
  --seed 20261007
```

The timestamp in names such as
`ses-abab-random-001_date-20260824T102425` is parsed directly and used for
chronological first/last selection.

All recordings in this project contain two interleaved channels, so discovery
defaults to `nchannels=2`. The explicit `--nchannels 2` above documents this in
the command and the value is also saved in the session and task manifests.
Task creation rejects older session manifests containing `nchannels=1`.

### 2. Create the task manifest

```bash
python preprocessing/suite2p/registration_tuning/make_tasks.py \
  --sessions "${TUNING_ROOT}/selected_sessions.csv" \
  --candidates preprocessing/suite2p/registration_tuning/candidates.json \
  --output-root "${TUNING_ROOT}/runs" \
  --manifest "${TUNING_ROOT}/tasks.csv"
```

This creates one task per selected session and parameter candidate. It refuses
an output root that overlaps any `funcimg/others` input directory.

### 3. Submit the Slurm array

Before launching the complete screen, submit task zero as a cluster/API pilot:

```bash
TASK_RANGE=0 bash preprocessing/suite2p/registration_tuning/submit_grid.sh \
  "${TUNING_ROOT}/tasks.csv"
```

The submission command prints the resolved manifest, working directory, and
absolute log directory before printing the Slurm job ID. This avoids dependence
on the shell directory from which `sbatch` was invoked.

Inspect its Slurm log, `status.json`, `provenance.json`, and `ops.npy`. After it
completes successfully, submit the full array. The completed pilot task will be
detected and skipped:

```bash
bash preprocessing/suite2p/registration_tuning/submit_grid.sh \
  "${TUNING_ROOT}/tasks.csv"
```

The default environment is `suite2p-reg-1.1.0` and at most six tasks run at
once. These can be overridden without editing the script:

```bash
SUITE2P_ENV=my-suite2p-env MAX_CONCURRENT=3 \
  bash preprocessing/suite2p/registration_tuning/submit_grid.sh \
  "${TUNING_ROOT}/tasks.csv"
```

`TASK_RANGE` accepts any Slurm array expression, such as `0-5%2`, for a larger
pilot subset.

Every task writes only beneath `${TUNING_ROOT}/runs`. Completed tasks are
skipped if the array is resubmitted. Failed tasks are forced through
registration again on resubmission but retain their previous Slurm logs.

### 4. Generate QC after the array finishes

This step is CPU-only:

```bash
python preprocessing/suite2p/registration_tuning/evaluate.py \
  --manifest "${TUNING_ROOT}/tasks.csv" \
  --strict
```

The report is written to `${TUNING_ROOT}/qc/index.html`. Individual run
directories also receive `qc.png` and `registered_frames.png`. Re-run with
`--overwrite` when intentionally regenerating an existing report.

## Session selection

### Eligible sessions

For each mouse, a session is eligible when:

1. the session contains a `funcimg` directory;
2. `funcimg/others` exists; and
3. `funcimg/others` directly contains at least one readable TIFF stack.

Nested TIFFs are ignored so that prior Suite2p outputs such as `meanImg.tiff`
cannot be mistaken for raw input. If several TIFF files are present directly
in `funcimg/others`, the discovery step selects the one with the
newest filesystem modification time (with filename as a deterministic
tie-breaker). By default it performs a fast TIFF-header read, rather than
walking thousands of TIFF page records over network storage. The audit CSV
records the selected file, its modification time, and the number of TIFF
alternatives. Use `--count-frames` to record the page count when desired, or
`--expected-frames N` to count and require an exact number.

Sessions with a missing directory, no TIFF, or an unreadable newest TIFF are
reported and excluded rather than silently accepted.

### Three sessions per mouse

Eligible sessions should be ordered chronologically using the date embedded in
the session directory name. If no valid date is available, the session number
should be used as a documented fallback.

Select three sessions for every mouse:

1. the first eligible session;
2. the last eligible session; and
3. one randomly selected session from the eligible sessions between the first
   and last.

The random selection must be reproducible. The selection script should accept
a random seed, use a stable per-mouse derivation of that seed, and save the
chosen sessions in a CSV manifest. Re-running an existing experiment should
use the saved manifest rather than draw another middle session.

Edge cases should be explicit:

- exactly three eligible sessions: use all three;
- two eligible sessions: use the first and last and flag the mouse as having
  incomplete sampling;
- one eligible session: use it once and flag the mouse as having incomplete
  sampling; and
- no eligible sessions: exclude the mouse and report the reason.

## Screening dataset

The newest TIFF stack in each selected session's `funcimg/others` directory is
used directly. The chosen path is locked into the session manifest, so the same
stack is used for every parameter candidate within that session even if a newer
file is added later.

For two-channel recordings, 4,000 TIFF pages represent approximately 2,000
timepoints per channel. Discovery defaults to `--nchannels 2`, and that value is
carried into every Suite2p task. Do not use `--expected-frames 2000` for these
files; no exact page count is required by default.

Suite2p registration metrics require at least 1,500 timepoints. Shorter TIFFs can
still be registered, but some built-in metrics may be unavailable and the
resulting ranking will use the remaining diagnostics. Short stacks do not
capture every long-timescale drift or rare motion event, so the screen is
intended to eliminate weak candidates and identify a shortlist rather than
replace full-session validation.

## Initial registration candidates

Begin with a small, interpretable candidate set rather than a large Cartesian
grid:

| Candidate | Non-rigid | Block size | Temporal smoothing |
|---|---:|---:|---:|
| Current-style rigid baseline | No | — | 1 |
| Rigid without temporal smoothing | No | — | 0 |
| Non-rigid default blocks | Yes | 128 × 128 | 0 |
| Non-rigid default blocks, low-SNR smoothing | Yes | 128 × 128 | 1 |
| Fine non-rigid blocks | Yes | 64 × 64 | 0 |
| Fine non-rigid blocks, low-SNR smoothing | Yes | 64 × 64 | 1 |

Initially hold the following settings constant:

| Parameter | Initial value |
|---|---:|
| `smooth_sigma` | `1.15` |
| `maxregshift` | `0.1` |
| `maxregshiftNR` | `5` |
| `snr_thresh` | `1.2` |
| `norm_frames` | `True` |
| `nimg_init` | `1000` |
| `two_step_registration` | `False` |

These values form a controlled first screen. Expand the search only in response
to a diagnosed failure:

- rigid offsets reaching the search boundary: examine `maxregshift`;
- non-rigid blocks reaching their limit: examine `maxregshiftNR`;
- noisy or implausible local deformation: increase `snr_thresh` or use larger
  blocks;
- a poor or blurred reference image: examine `nimg_init` and then test
  two-step registration; or
- consistently low-SNR registration: examine temporal smoothing.

The candidate file should store explicit values even when they equal Suite2p
defaults, so future changes in upstream defaults cannot alter the experiment.

## HPC execution design

The screen should run as a throttled Slurm job array, with one task for each
session/candidate pair. The workflow should contain:

1. a session-discovery command that creates the reproducible three-session
   manifest;
2. a candidate configuration file;
3. a command that creates the complete session-by-candidate task manifest;
4. a registration-only Suite2p runner using the Suite2p 1.x API;
5. a Slurm array submission wrapper; and
6. a separate CPU evaluation command that runs after registration jobs finish.

Every candidate must write to its own directory below a dedicated tuning output
root. The grid must never write into or overwrite a session's production
`suite2p` directory.

Each run directory should contain:

- the exact `db` and `settings` passed to Suite2p;
- the source mouse, session, and input TIFF list;
- Suite2p, Python, PyTorch, and CUDA versions;
- Slurm job identifiers and hostname;
- start time, finish time, and runtime;
- completion or failure status with an error traceback; and
- the Suite2p registration output and generated QC files.

Completed tasks should be restart-safe: resubmitting the array should skip a
successfully completed run and retry failed or absent runs.

## Quantitative evaluation

Calculate diagnostics for every completed candidate, including:

- median and fifth-percentile frame-to-reference correlation (`corrXY`);
- fraction of frames marked bad;
- 95th-percentile rigid displacement;
- median frame-to-frame shift jitter;
- fraction of rigid shifts close to the allowed search boundary;
- non-rigid displacement summaries when non-rigid registration is enabled;
- registered mean-image sharpness; and
- Suite2p's built-in registration principal-component metrics when available.

No individual metric should determine the winner. In particular, sharpness can
be increased by noise or implausible local warping and must be interpreted with
the visual output.

Metrics should first be normalised within each session because signal level,
imaging depth, expression, and field structure differ between recordings. The
session-level scores should then be aggregated within each mouse before
comparing candidates across mice.

A useful selection score is:

```text
robust score = 0.7 × median mouse score + 0.3 × worst mouse score
```

Candidates missing an expected session should be marked incomplete and should
not outrank candidates evaluated on the complete manifest.

## Visual evaluation

Create an HTML report that allows all candidates for the same session to be
compared side by side. For every run, show:

- the reference image;
- the registered mean image with consistent contrast scaling;
- rigid X and Y offsets over time;
- frame-to-reference correlation over time;
- rigid and non-rigid displacement distributions;
- bad-frame and boundary-hit diagnostics;
- a montage of registered frames sampled across the stack; and
- where practical, a short side-by-side raw-versus-registered movie using the
  same source frames and display scaling.

Visually reject candidates showing duplicated structures, local tearing or
warping, abrupt offset jumps, blurred cell boundaries, unstable borders, or
frequent shifts at the allowed limit, even if their scalar score is high.

## Full-session validation

After the short-stack screen:

1. inspect the quantitative ranking and visual report;
2. retain the best two or three candidates;
3. run those candidates on at least one full approximately 81,000-frame session
   per mouse;
4. inspect long-timescale drift, rare motion episodes, cropping, and non-rigid
   stability;
5. select one parameter set with acceptable performance for every mouse; and
6. apply the final pinned configuration to all production sessions.

The final decision, manifests, candidate configuration, environment lock, and
QC summaries should be retained together as the provenance record for the
production registration settings.

## Implementation boundary

The registration-tuning implementation is self-contained under:

```text
preprocessing/suite2p/registration_tuning/
```

The existing production runner will not be changed during the initial screen.
Migration of downstream detection, extraction, segmentation, and CellTV code to
the Suite2p 1.x output structure should be treated as a separate validation
step after the registration parameters and version have been accepted.

### Files

- `discover_sessions.py`: validates and selects the newest TIFF, optionally
  counts its pages, and performs reproducible three-session selection.
- `candidates.json`: explicit initial parameter candidates.
- `make_tasks.py`: creates the session-by-candidate task table and enforces the
  input/output safety boundary.
- `run_candidate.py`: runs registration and registration metrics only.
- `run_grid_array.sbatch` and `submit_grid.sh`: Slurm array execution.
- `evaluate.py`: metrics, registered-frame montages, per-run QC, and the HTML
  comparison/ranking report.
- `requirements.txt`: pinned Suite2p and QC dependencies.

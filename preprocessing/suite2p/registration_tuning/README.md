# Suite2p registration tuning plan

This repository contains analysis and preprocessing code for two-photon calcium
imaging experiments in mice. The immediate development goal is to select a
robust Suite2p registration configuration separately for each mouse, while
keeping the computational search practical on the HPC cluster.

## Suite2p registration optimisation plan

### Objectives

The registration evaluation should:

- use the same pinned Suite2p version while allowing the selected registration
  parameters to differ by mouse;
- screen parameter candidates on the available short TIFF stacks before committing
  resources to complete recordings of approximately 81,000 frames;
- run on the Slurm HPC cluster against data on network storage;
- combine quantitative diagnostics with visual inspection of registered data;
- favour settings that work consistently across the selected sessions for each
  mouse; and
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

Create a dedicated environment on the cluster. Its NVIDIA 580 driver supports
the official PyTorch 2.13 CUDA 12.6 wheel, which is pinned together with the
workflow requirements:

```bash
module load mamba
mamba create -n suite2p-reg-1.1.0 python=3.11 pip -y
source activate suite2p-reg-1.1.0

python -m pip install --upgrade pip
python -m pip install --no-cache-dir \
  -r preprocessing/suite2p/registration_tuning/requirements-hpc-cu126.txt

python - <<'PY'
from importlib.metadata import version
import torch

print("Suite2p:", version("suite2p"))
print("Torch:", version("torch"))
print("Torch module:", torch.__file__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
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

To tune one mouse independently, copy `candidates.json`, edit that copy for the
mouse, and create a mouse-specific manifest and output root:

```bash
MOUSE=sub-02
MOUSE_ROOT="${TUNING_ROOT}/mice/${MOUSE}"
mkdir -p "${MOUSE_ROOT}"
cp preprocessing/suite2p/registration_tuning/candidates.json \
  "${MOUSE_ROOT}/candidates.json"

# Review/edit ${MOUSE_ROOT}/candidates.json, then:
python preprocessing/suite2p/registration_tuning/make_tasks.py \
  --sessions "${TUNING_ROOT}/selected_sessions.csv" \
  --mouse "${MOUSE}" \
  --candidates "${MOUSE_ROOT}/candidates.json" \
  --output-root "${MOUSE_ROOT}/runs" \
  --manifest "${MOUSE_ROOT}/tasks.csv" \
  --overwrite
```

Repeat with a different candidate JSON for each mouse. Each task records the
complete merged registration settings, so different mouse-specific searches
remain reproducible.

### 3. Submit the Slurm array

Before launching the complete screen, test task zero directly from an allocated
GPU node. This bypasses Slurm log routing while checking the environment, CUDA,
manifest paths, TIFF access, Suite2p execution, and expected output files:

```bash
bash preprocessing/suite2p/registration_tuning/run_pilot.sh \
  "${TUNING_ROOT}/tasks.csv"
```

The wrapper writes timestamped `pilot_task0_*.out` and `pilot_task0_*.err`
files under `${TUNING_ROOT}/logs` while also displaying their contents in the
terminal. It defaults to `suite2p-reg-1.1.0`, task zero, and CUDA. Override these
only when needed with `SUITE2P_ENV`, `PILOT_TASK_ID`, or `PILOT_DEVICE`.

The final output should include `Completed task 0:` and `Pilot finished
successfully.` The task directory recorded in `tasks.csv` should contain
`status.json` with `state: complete` and `suite2p/plane0/ops.npy`. The runner
also exports `meanImg_chan1.tiff`, `meanImg_chan2.tiff`, individual PNG
previews, and a side-by-side `meanImgs.png` under `suite2p/plane0`. It also
exports registered-frame montages for both channels as `registered_frames.png`
and `registered_frames_chan2.png` in the candidate run directory.

After those visual assets have been created, the runner deletes Suite2p's
derived binary movie copies (`data*.bin`). These files account for almost all
of a run's disk usage and are not needed by the evaluator or final HTML report.
The original TIFF input is never modified or deleted. The removed paths and
sizes are recorded in `status.json`; the binary movies can be recreated by
rerunning Suite2p. To retain them temporarily for debugging, set
`KEEP_DERIVED_BINARIES=1` or pass `--keep-derived-binaries` directly to
`run_candidate.py`.

To add these files to a run that completed before mean-image export was added:

```bash
python preprocessing/suite2p/registration_tuning/export_mean_images.py \
  --run-dir /absolute/path/to/the/candidate/run
```

Alternatively, submit task zero as a Slurm pilot:

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

Run submission from the activated `suite2p-reg-1.1.0` environment. The submit
script validates Suite2p 1.1.0 and passes that exact absolute Python executable
to every Slurm task, avoiding environment-name resolution differences on
compute nodes. At most six tasks run at once. These settings can be overridden
without editing the script:

```bash
PYTHON_EXECUTABLE=/absolute/path/to/python MAX_CONCURRENT=3 \
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

The combined report is written to `${TUNING_ROOT}/qc/index.html`, and separate
reports are written to `${TUNING_ROOT}/qc/mice/<mouse-id>/index.html`.
Individual run directories also receive `qc.png`, `registered_frames.png`, and
`registered_frames_chan2.png`. The frame montages are generated by the runner
before its large binary movies are removed, so the evaluator can reuse the
saved PNGs without needing those movies.
Re-run with `--overwrite` when intentionally regenerating an existing report.

Each new run records peak Python-process RAM, peak CUDA memory allocated, and
peak CUDA memory reserved in `status.json`. These measurements are copied into
`qc/run_metrics.csv` and displayed on each HTML QC card. They measure the
registration Python process and PyTorch's CUDA allocator; Slurm's `MaxRSS`
remains the authoritative scheduler-level memory measurement and is available
after a job exits:

```bash
sacct -j JOB_ID --units=G \
  --format=JobID,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,AllocTRES%50
```

Runs completed before resource recording was added show `n/a` in the report.

For a mouse-specific manifest, the report is written below that manifest's
directory, for example `${TUNING_ROOT}/mice/sub-02/qc/index.html`. Candidate
ranking in this report is aggregated only across the selected sessions for
that mouse as `0.7 × median_session_score + 0.3 × worst_session_score`, and can
therefore be used to choose a different parameter set for each mouse.

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

All candidates use non-rigid registration, 128 × 128 blocks, and two-step
registration. The first pass uses an 11-candidate one-factor-at-a-time screen
rather than the 216 jobs per session required by the full Cartesian product:

| Candidate group | Values |
|---|---|
| Baseline | `maxregshift=0.2`, `maxregshiftNR=10`, channel 1 alignment, `smooth_sigma=1.15`, `snr_thresh=1.2` |
| `maxregshift` | `0.1`, `0.3`, `0.4` |
| `maxregshiftNR` | `5`, `15` |
| Alignment channel | channel 2 |
| `smooth_sigma` | `1.5`, `2.0` |
| `snr_thresh` | `1.0`, `1.5` |

Initially hold the following settings constant:

| Parameter | Initial value |
|---|---:|
| `smooth_sigma_time` | `0` |
| `norm_frames` | `True` |
| `nimg_init` | `1000` |
| `do_bidiphase` | `False` |
| `bidiphase` | `0.0` |
| `nonrigid` | `True` |
| `block_size` | `128 × 128` |
| `two_step_registration` | `True` |

Temporal smoothing is disabled for every candidate. In Suite2p 1.1.0,
`smooth_sigma_time > 0` sends a CUDA tensor into SciPy's NumPy-only Gaussian
filter and fails before registration. Bidirectional phase correction and its
automatic estimation are also disabled explicitly for this project. The runner
rejects candidate manifests that disable non-rigid or two-step registration.

These values form a controlled first screen. Run the first 11 tasks initially
to compare all candidates on one session, then use the resulting QC to select a
smaller combined shortlist for the full cross-mouse screen. Expand or combine
settings only in response to a diagnosed failure:

- rigid offsets reaching the search boundary: examine `maxregshift`;
- non-rigid blocks reaching their limit: examine `maxregshiftNR`;
- noisy or implausible local deformation: increase `snr_thresh` or use larger
  blocks;
- a poor or blurred reference image: examine `nimg_init` and then test
  two-step registration.

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

### How the scores are calculated

For each metric, candidates from the same session are robustly normalised as:

```text
robust z-score = (value - candidate median) / (1.4826 × median absolute deviation)
```

Scores are clipped to the range -5 to +5. The sign is reversed for metrics
where lower values are better. The weighted mean of the available robust
z-scores is the `within_session_score`:

| Metric | Weight | Better direction |
|---|---:|---|
| Median frame-to-reference correlation | 0.25 | Higher |
| Fifth-percentile frame-to-reference correlation | 0.15 | Higher |
| Registered mean-image sharpness | 0.15 | Higher |
| Bad-frame fraction | 0.15 | Lower |
| Median frame-to-frame shift jitter | 0.10 | Lower |
| Shift-boundary-hit fraction | 0.10 | Lower |
| Registration-PC residual mean | 0.10 | Lower |

If a metric is unavailable, its weight is omitted and the available weights are
renormalised. The reported rigid and non-rigid 95th-percentile displacements are
diagnostics but are not currently included in the combined score.

A score near zero is typical relative to the candidates run on that session;
positive is better and negative is worse. It is a relative ranking and has no
absolute biological or image-quality interpretation.

For each candidate, the median `within_session_score` across one mouse's
sessions becomes that mouse's score. `median_mouse_score` is the median of
these per-mouse scores; it does not refer to a particular "median mouse".
`worst_mouse_score` is the minimum per-mouse score. The final ranking score is:

```text
robust_score = 0.7 × median_mouse_score + 0.3 × worst_mouse_score
```

This rewards good typical performance while penalising a candidate that fails
for one mouse. During a partial screen containing only one session from one
mouse, the within-session, mouse, median-mouse, worst-mouse, and robust scores
are identical. Cross-mouse robustness cannot be interpreted until the same
candidates have completed multiple sessions and mice. Always check
`mice_completed`, `sessions_completed`, and the visual QC alongside the score.

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
- `export_mean_images.py`: viewable TIFF and PNG exports of both channel mean
  images from each `ops.npy`.
- `requirements.txt`: pinned Suite2p and QC dependencies.
- `requirements-hpc-cu126.txt`: the official CUDA 12.6 PyTorch wheel plus the
  pinned workflow dependencies for the cluster.

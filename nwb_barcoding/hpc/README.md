# Running nwb_barcoding on the HPC

These scripts run `python -m nwb_barcoding.build_session_nwb` as a SLURM array job, with one session per task. The job reads `/ceph/...` directly, so it doesn't go through the SMB mount on the laptop. The environment is built with uv from `../pyproject.toml` + `../uv.lock` and replaces the `2p_preprocess_extraction` conda env.

| File | Where it runs | What it does |
|---|---|---|
| `sync_to_cluster.sh` | laptop | copies `Aimbot_barcoding/preprocess_alignment/` to `hpc_files/code/` on the share and writes `CODE_VERSION.txt` (also `nwb_barcoding/`, but only while `hpc_files/code/Bonsai_ABCD` isn't a git clone yet) |
| `build_nwb_array.sh` | HPC | SLURM array job; task N builds the N-th session in the sessions file |
| `sessions_holography.txt` | HPC | one `mouse tag` per line |

## On the laptop

**1. Mount the share.** Check in Finder that `/Volumes/mrsic_flogel` exists.

**2. Get the code onto the share.** `hpc_files/code/Bonsai_ABCD` is a git clone of this repo (branch `Sandra_analysis`; see the top-level README). Commit and push on the laptop, then on the cluster:
```bash
cd /ceph/mrsic_flogel/public/projects/SaRe_20240219_hfs/hpc_files/code/Bonsai_ABCD
git pull
```
`Aimbot_barcoding` isn't a clone yet, so copy it from the laptop:
```bash
cd "/Users/sandrareinert/Library/Mobile Documents/com~apple~CloudDocs/Documents/Postdoc_SWC/Code/Bonsai_ABCD"
bash nwb_barcoding/hpc/sync_to_cluster.sh
```
The script copies Aimbot's working tree, including uncommitted edits, to `SaRe_20240219_hfs/hpc_files/code/Aimbot_barcoding`, and never deletes anything there. It skips `Bonsai_ABCD` because that's a git clone. At the end it prints `CODE_VERSION.txt`, which lists the Aimbot commit and local edits that were copied.

## On the cluster: one-time setup

**3. Log in** to the HPC login node as usual.

**4. Build the environment on a compute node.** Don't build it on the login node, where git runs into the per-user thread limit (`unable to create thread`). The SWC guidelines say venvs belong under `/ceph/python_envs/$USER`. The `uv` module already points the uv cache and uv's Python downloads (`UV_PYTHON_INSTALL_DIR`) there.
```bash
srun -p cpu --cpus-per-task=4 --mem=16G -t 1:00:00 --pty bash

module load uv
export UV_PROJECT_ENVIRONMENT=/ceph/python_envs/$USER/nwb_barcoding
# limit git and uv parallelism (git settings apply to this shell only)
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=pack.threads  GIT_CONFIG_VALUE_0=1
export GIT_CONFIG_KEY_1=index.threads GIT_CONFIG_VALUE_1=1
export UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=2

cd /ceph/mrsic_flogel/public/projects/SaRe_20240219_hfs/hpc_files/code/Bonsai_ABCD
uv sync --frozen --project nwb_barcoding --python 3.12
```
When the build works, the last lines read `Installed ~115 packages`. The env needs Python 3.12, not the module's default 3.13, because numpy 1.26.4 is pinned to match the existing outputs and has no 3.13 build. uv downloads 3.12 by itself.

**5. Check that it works**, in the same shell or any other one:
```bash
cd /ceph/mrsic_flogel/public/projects/SaRe_20240219_hfs/hpc_files/code/Bonsai_ABCD
source /ceph/python_envs/$USER/nwb_barcoding/bin/activate
python -c "import neuroconv, pynwb, ndx_events, roiextractors; import nwb_barcoding.build_session_nwb; print('ok')"
deactivate
exit
```
A few spikeinterface warnings are normal. From here on you don't need `module load uv` or the exports. Jobs only activate the env and never install anything.

## On the cluster: first test

**6. Run one session interactively** in a new `srun` shell. Use the larger allocation here: the build can be killed partway through if it runs out of memory or time.
```bash
srun -p cpu --mem=32G -t 2:00:00 --pty bash
cd /ceph/mrsic_flogel/public/projects/SaRe_20240219_hfs/hpc_files/code/Bonsai_ABCD
source /ceph/python_envs/$USER/nwb_barcoding/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
B=/ceph/mrsic_flogel/public/projects/SaReSuKuMaRa_20260911_holography/cohort1
python -m nwb_barcoding.build_session_nwb sub-001 2LM001 $B/rawdata $B/preprocessed $B/BehaviourAligned_robust
exit
```
Check that all 7 steps run and look at the verification flags at the end. Output goes to `cohort1/BehaviourAligned_robust/sub-001/...`. You can also see it from the laptop through the mount.

## On the cluster: batch run

**7. Submit two sessions first** (tasks 1 and 2 in the sessions file):
```bash
cd /ceph/mrsic_flogel/public/projects/SaRe_20240219_hfs/hpc_files/code/Bonsai_ABCD/nwb_barcoding/hpc
sbatch --array=1-2 build_nwb_array.sh sessions_holography.txt
squeue -u $USER                  # are they running?
tail -f logs/nwb_<jobid>_2.out   # follow a log; Ctrl+C stops watching, the job keeps running
```

**8. Run the rest** once those look good:
```bash
sbatch --array=3-16%4 build_nwb_array.sh sessions_holography.txt
```
`%4` means at most 4 sessions run at the same time. Steps that already have output are skipped. So if a session fails, fix the problem and resubmit just that task with `--array=<N>`. Outputs only go to `<cohort>/BehaviourAligned_robust/` and never into `rawdata/`.

## Later, when things change

- **Code only:** edit on the laptop, commit + push, and `git pull` on the cluster (step 2; rerun the sync script only for Aimbot changes). The env doesn't need rebuilding.
- **Dependencies:** edit `nwb_barcoding/pyproject.toml`, then run `uv lock --project nwb_barcoding` on the laptop. Commit + push, `git pull` on the cluster, then step 4. When you run it on the laptop, set `UV_PROJECT_ENVIRONMENT` to a folder outside iCloud, such as `~/envs/nwb_barcoding`.
- **New sessions:** add lines to `sessions_holography.txt` (or make a new sessions file), commit + push and `git pull`. Small edits can also be made directly in the clone on the cluster and committed from there.
- **Another cohort with the same layout** (`rawdata/` + `preprocessed/`): add `--export=ALL,BASE=/ceph/.../<cohort root>` to `sbatch`.

## Troubleshooting the env build

- **`Cache is currently in-use, waiting for other uv processes`:** the module's cache in `/ceph/python_envs/.uv_cache` is shared with other users. Never use `--force`, because it can break their installs. Check `squeue -u $USER` for an old interactive job of your own that's still running, and wait or `scancel` it.
- **git errors that mention `.uv_cache/git-v0/...`:** an earlier clone was interrupted and left broken files in the shared cache. Build once with your own cache on the same filesystem, so hardlinking still works:
  `export UV_CACHE_DIR=/ceph/python_envs/$USER/.uv_cache`, then rerun `uv sync`.
- **`ModuleNotFoundError` in step 5, but `which python` points into the env:** `uv sync` didn't finish. uv installs everything at the very end, so one failed download leaves the venv empty. Rerun step 4.

## Environment notes

- Package versions are pinned to the conda env that produced the existing outputs. The scientific packages match it exactly: numpy 1.26.4, scipy, pandas, h5py, pynwb 3.0.0, hdmf 3.14.6, neuroconv 0.7.2, spikeinterface 0.103.1, suite2p-NIU 0.1.1, and the ndx-events/roiextractors fork commits. A few small utility packages are a minor version newer.
- torch is only installed because suite2p-NIU depends on it (via cellpose), and the pipeline never uses it. On Linux it comes from the CPU-only wheel index, which avoids several GB of CUDA libraries.
- `uv.lock` was written with uv 0.13.0. The module's uv 0.12.11 reads it without problems.

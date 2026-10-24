# Running ROICaT on the HPC (holography cohort)

These scripts run `roicat_tracking_holography.py` as a SLURM array job on the GPU partition, one animal per task. Each animal is tracked across the sessions listed for it in `mice_holography.txt` (currently 2LM001 to 2LM004 plus seq1-001 and seq1-004, i.e. t0 to t5). The job reads `/ceph/...` directly. The environment is a uv venv built from `pyproject.toml` + `uv.lock` (roicat 1.7.12, torch 2.14.1, Python 3.12), and it replaces the `roicat` conda env used by `../run_roicat.sh`.

| File | Where it runs | What it does |
|---|---|---|
| `sync_to_cluster.sh` | laptop | legacy: copied `preprocessing/roicat_analysis/` to `hpc_files/code/Bonsai_ABCD/` before that folder became a git clone; now refuses to run |
| `run_roicat_array.sh` | HPC | SLURM array job; task N tracks the N-th animal in the mice file |
| `mice_holography.txt` | HPC | one `mouse tag1 tag2 ...` per line |

Inputs: `cohort1/preprocessed/sub-00X/ses-<tag>_*/funcimg/suite2p/plane0/{stat,ops}.npy`.
Outputs: `cohort1/analysis/sub-00X/ROICaT/` (results `.json`/`.richfile`, `sessions_used.json`, `visualization/`).

## On the laptop

**1. Get the code onto the cluster.** `hpc_files/code/Bonsai_ABCD` is a git clone of this repo (branch `Sandra_analysis`; see the top-level README). Commit and push on the laptop, then on the cluster:
```bash
cd /ceph/mrsic_flogel/public/projects/SaRe_20240219_hfs/hpc_files/code/Bonsai_ABCD
git pull
```

Optional check that the inputs resolve (this only prints paths):
```bash
python preprocessing/roicat_analysis/roicat_tracking_holography.py --animal sub-001 \
    --sessions 2LM001 2LM002 2LM003 2LM004 \
    --root /Volumes/mrsic_flogel/public/projects/SaReSuKuMaRa_20260911_holography/cohort1 --dry_run
```

## On the cluster: one-time setup

**2. Build the env on a GPU compute node**, not the login node. Following the SWC guidelines, the venv goes under `/ceph/python_envs/$USER`, and the `uv` module points the uv cache and Python downloads there.
```bash
srun -p gpu --gres gpu:1 --exclude=gpu-380-[10,12,14] --cpus-per-task=4 --mem=16G -t 1:00:00 --pty bash

module load uv
export UV_PROJECT_ENVIRONMENT=/ceph/python_envs/$USER/roicat
export UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=2

cd /ceph/mrsic_flogel/public/projects/SaRe_20240219_hfs/hpc_files/code/Bonsai_ABCD/preprocessing/roicat_analysis/hpc
uv sync --frozen --python 3.12
```
This installs torch with its CUDA libraries (a few GB), so expect it to take a while. Everything comes as prebuilt wheels and nothing is compiled. uv downloads Python 3.12 if the module's default (3.13) is the only one available.

**3. Check it**, in the same shell:
```bash
source /ceph/python_envs/$USER/roicat/bin/activate
nvidia-smi --query-gpu=name,driver_version --format=csv
python -c "import roicat, torch; print(roicat.__version__, torch.__version__, torch.cuda.is_available())"
deactivate
exit
```
The last line should print `1.7.12 2.14.1 True`. If it prints `False` and the driver version is below 580, the driver is too old for the CUDA 13 build. Uncomment the cu126 block at the end of `pyproject.toml`, run `uv lock --project preprocessing/roicat_analysis/hpc` on the laptop, commit + push, then repeat steps 1–3.

**GPU type:** torch 2.14.1+cu130 only supports compute capability 7.5 (Turing) and newer. The Quadro P5000 (Pascal, 6.1) fails with `cuDNN version 92400 is not compatible with devices with SM < 7.5`, so `run_roicat_array.sh` and the `srun` above exclude the P5000 nodes (`gpu-380-10`, `-12`, `-14`; `-12` and `-14` also have an RTX 2080, which is lost to the exclusion). All other GPUs in the partition (RTX 2080, Quadro RTX 5000, A4500, RTX 4000 Ada, A100, L40S) work. Check `sinfo -p gpu -o "%N %G"` if nodes are added.

From here on you don't need `module load uv`. Jobs only activate the env.

The first run downloads the ROInet model (from osf.io) into the node's temp dir, so compute nodes need internet access. They already have it for `uv sync`.

## On the cluster: run

**4. Test one animal:**
```bash
cd /ceph/mrsic_flogel/public/projects/SaRe_20240219_hfs/hpc_files/code/Bonsai_ABCD/preprocessing/roicat_analysis/hpc
sbatch --array=1 run_roicat_array.sh mice_holography.txt
squeue -u $USER
tail -f logs/roicat_<jobid>_1.out
```
Without `--array` the script runs the first animal (log `roicat_<jobid>_4294967294.out`).
Then look at `cohort1/analysis/sub-001/ROICaT/visualization/`:
- `FOV_images_aligned_geometric.gif` and `FOV_images_aligned_non-rigid.gif` should look still from frame to frame.
- `FOV_clusters.gif` shows the matched ROIs in the same colour across sessions.
- `quality_metrics.png`

The log ends with the number of clusters and the number of discarded ROIs.

**5. Run the other three:**
```bash
sbatch --array=2-4 run_roicat_array.sh mice_holography.txt
```
Rerunning a task overwrites that animal's `ROICaT/` results.

## Later

- **Code only:** edit on the laptop, commit + push, and `git pull` on the cluster (step 1). The env doesn't need rebuilding.
- **Dependencies:** edit `pyproject.toml`, run `uv lock --project preprocessing/roicat_analysis/hpc` on the laptop, then repeat steps 1–3.
- **Other sessions/animals:** edit `mice_holography.txt` (then commit + push and `git pull`), or copy it and pass the new file as the argument to `sbatch ... run_roicat_array.sh <file>`.
- **Parameters:** `--um_per_pixel` defaults to 0.834 (ScanImage `imagingFovUm` 426.996 µm / 512 px, the same in all 16 sessions). The alignment, blurring and embedding settings are the same as in the cohort2 script `../roicat_tracking.py`. The similarity and clustering steps use the roicat 1.7 API with its default mixing-parameter search, because the old calls no longer exist (see the script's docstring).

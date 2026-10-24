# Bonsai_ABCD

Small repo for parsing and plotting Bonsai VR behavioural sessions in the style of ABC/ABCD tasks.

## Installation (tested compatibility for use in Win and Mac OS)
I recommend using the environment.yml file to create a new conda environment 

```console
conda env create -f environment.yml
conda activate bonsai_abcd
pip install --no-deps swc-aeon==0.1.0
```

(`swc-aeon` is installed separately because it declares `numpy<2`, which pip can't override; it works fine on numpy 2.2.6.)

to activate the environment, run

```console
conda activate bonsai_abcd
```

### Alternatively, 
create a blank environment and install python 3.12:

```console
conda create --name bonsai_abcd python=3.12
```

activate it:

```console
conda activate bonsai_abcd
```

then use the requirements.txt file to pip install the necessary packages  

```console
pip install -r requirements.txt
pip install --no-deps swc-aeon==0.1.0
```


### With uv (local or SWC HPC)
`pyproject.toml` + `uv.lock` reproduce the `bonsai_abcd` env with uv (versions pinned to the
working conda env; torch is CPU-only on Linux). Locally:

```console
uv sync                # creates .venv in the repo
uv run jupyter lab
```

On the SWC HPC the repo is a git clone at `/ceph/mrsic_flogel/public/projects/SaRe_20240219_hfs/hpc_files/code/Bonsai_ABCD`
(branch `Sandra_analysis`). The `nwb_barcoding/hpc` and `preprocessing/roicat_analysis/hpc` jobs run from
the same clone, each with its own venv. Update it with `git pull` after pushing from the laptop.
Build the env on a compute node (the login node's thread limit breaks git/uv), and keep the venv
(and the Python uv downloads) on /ceph, not in home:

```console
srun -p cpu --cpus-per-task=4 --mem=16G -t 1:00:00 --pty bash
module load uv
export UV_PROJECT_ENVIRONMENT=/ceph/python_envs/$USER/bonsai_abcd
export UV_PYTHON_INSTALL_DIR=/ceph/python_envs/$USER/python
cd /ceph/mrsic_flogel/public/projects/SaRe_20240219_hfs/hpc_files/code/Bonsai_ABCD
uv sync --locked
uv run python -m musicboxes.session_overview ...   # or: source $UV_PROJECT_ENVIRONMENT/bin/activate
```

To use it as a Jupyter kernel (e.g. through a VS Code tunnel on a compute node), select the
interpreter at `/ceph/python_envs/$USER/bonsai_abcd/bin/python`.

If you add a dependency, use `uv add <pkg>` and also add it to `environment.yml` / `requirements.txt`.

## Use

The main output script is the jupyter notebook Cohort3_daily_summary.ipynb. 
To analyse the behaviour of one session, change [mouse_id] and [date] to the session you want, then run the cells. 

To keep the script tidy, it imports the central functions from parse_bonsai_functions.py, which is a collection of all parsing and plotting operations. 
To add new functionality, write a function in that script and call it in the notebook. 

There will be some analyses/plots that only make sense under certain task conditions (e.g. stable world vs. random world). 
As the scripts grow, let's try and make sure it is always indicated which conditions need to be met.

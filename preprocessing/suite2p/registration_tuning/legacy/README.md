# Legacy registration-tuning launchers

These scripts are retained to reproduce earlier manifest-based experiments.
They have been superseded by `../cli.py` and should not be used for new runs.

- `submit_grid.sh` and `run_pilot.sh` are replaced by `cli.py` with
  `--execution slurm` and `--execution direct`, respectively.
- `make_full_session_tasks.py` and `submit_full_sessions.sh` are replaced by
  `cli.py --input-kind full`. Their shared array worker remains in the parent
  directory because a downstream Cellpose pilot still uses it.

The scripts still resolve and call the current core runner in the parent
directory, so historical commands remain reproducible when their paths are
updated to include `legacy/`.

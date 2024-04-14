"""Runs the evaluation script for all runs that have not already been evaluated in the /wandb folder."""

import datetime
import subprocess

from pinc.data import REPO_ROOT


def eval_all_runs(start_date: datetime.datetime) -> None:
    """Downloads all files from a remote Weights & Biases run to the current machine."""
    eval_script = REPO_ROOT / "pinc" / "run_eval.py"
    runs = REPO_ROOT.glob("wandb/run-*_*-*")
    for run_dir in sorted(runs):
        _run, date_string, _run_id = str(run_dir).split("-")
        date = datetime.datetime.strptime(date_string, "%Y%m%d_%H%M%S")
        if date < start_date:
            print(f"Skipping {run_dir}, run too old")
            continue
        if (run_dir / "files/metrics").exists() and (run_dir / "files/reconstructions").exists():
            print(f"Skipping {run_dir}")
            continue
        command = ["python", str(eval_script), "--n-models", "1", "--run-path", str(run_dir)]
        print(f"Running {command}")
        subprocess.run(command)


if __name__ == "__main__":
    start_date = datetime.datetime(2024, 4, 1)
    eval_all_runs(start_date)

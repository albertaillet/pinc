"""Downloads all files from a remote Weights & Biases run to the current machine."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from pinc.data import REPO_ROOT


def get_args() -> str:
    """Checks the command line arguments and returns the run id."""
    args = sys.argv[1:]
    msg = "Usage: python download_from_wandb.py <run_id>"
    assert len(args) == 1, msg
    run_id = args[0]
    print(f"Parsed run id: {run_id}")
    assert isinstance(run_id, str), msg
    assert len(run_id) == 8, msg
    assert run_id.isalnum(), msg
    assert run_id.islower(), msg
    return run_id


def download_from_wandb(run_id: str) -> None:
    """Downloads all files from a remote Weights & Biases run to the current machine."""
    project = "pinc"
    entity = "reproducibility-challenge"
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(REPO_ROOT) / f"wandb/run-{date}-{run_id}/files"
    run_dir.mkdir(exist_ok=True, parents=True)
    print(f"Downloading run {run_id} to {run_dir}...")
    command = ["wandb", "pull", "--project", project, "--entity", entity, run_id]
    subprocess.run(command, cwd=run_dir, check=True)


if __name__ == "__main__":
    download_from_wandb(get_args())

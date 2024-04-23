"""Generates the plots for the paper."""

import datetime
import subprocess
from enum import StrEnum

from pinc.data import REPO_ROOT

PAPER_SRB_FILES = ["anchor", "daratech", "dc", "gargoyle", "lord_quas"]


class RunTypes(StrEnum):
    # TYPE_1 = "PINC (ours) $epsilon=1$"
    # TYPE_2 = "PINC (ours) $epsilon=0.1$"
    TYPE_3 = "$[$Re$]$ PINC $epsilon=1$"
    TYPE_4 = "$[$Re$]$ PINC $epsilon=0.1$"


RUN_IDS = {
    "anchor": {
        # RunTypes.TYPE_1: ["2024_02_15_13_02_27"],
        # RunTypes.TYPE_2: ["2024_02_12_11_31_23_lor_anchor_eps_01"],
        RunTypes.TYPE_3: ["7nsgf68t", "3pzxgbig", "zwohp52r"],
        RunTypes.TYPE_4: ["8lcvaqh4", "xw68upg6", "suobhxom"],
    },
    "daratech": {
        # RunTypes.TYPE_1: ["2024_02_12_22_17_49"],
        # RunTypes.TYPE_2: ["2024_02_13_10_44_11"],
        RunTypes.TYPE_3: ["teoj02mm", "ggtjf89r", "b5o23l1e"],
        RunTypes.TYPE_4: ["kt67i502", "vd7dkf40", "funmhwh3"],
    },
    "dc": {
        # RunTypes.TYPE_1: ["2024_02_14_09_06_06_lor_dc_eps_1"],
        # RunTypes.TYPE_2: ["2024_02_13_08_51_39_lor_dc_eps_01"],
        RunTypes.TYPE_3: ["v0l912vq", "4jsncb8b", "dqh60r92"],
        RunTypes.TYPE_4: ["1o79somx", "raimbjb8", "nq2dyh10"],
    },
    "gargoyle": {
        # RunTypes.TYPE_1: ["2024_02_12_11_27_25"],
        # RunTypes.TYPE_2: ["2024_02_09_19_17_27"],
        RunTypes.TYPE_3: ["ojbs5lgy", "q7e5btzg", "ljbi7xiv"],
        RunTypes.TYPE_4: ["6jfgejpa", "j7lwywt2", "mpcpqqo7"],
    },
    "lord_quas": {
        # RunTypes.TYPE_1: ["2024_02_14_17_35_11"],
        # RunTypes.TYPE_2: ["2024_02_13_22_30_18"],
        RunTypes.TYPE_3: ["1f5dvyoc", "s3z11lkx", "wzpaoy41"],
        RunTypes.TYPE_4: ["lc8cu813", "evqmf9oi", "pdrpsqow"],
    },
}


ALL_RUN_IDS = [
    run_id for file in PAPER_SRB_FILES for run_type in [RunTypes.TYPE_3, RunTypes.TYPE_4] for run_id in RUN_IDS[file][run_type]
]


def eval_all_runs(start_date: datetime.datetime) -> None:
    """Downloads all files from a remote Weights & Biases run to the current machine."""
    eval_script = REPO_ROOT / "pinc" / "run_eval.py"
    runs = REPO_ROOT.glob("wandb/run-*_*-*")
    for run_dir in sorted(runs):
        _run, date_string, run_id = str(run_dir).split("-")
        date = datetime.datetime.strptime(date_string, "%Y%m%d_%H%M%S")
        if run_id not in ALL_RUN_IDS:
            continue
        if date < start_date:
            print(f"Skipping {run_dir}, run too old")
            continue
        metrics_dir = run_dir / "files/metrics"
        if metrics_dir.exists() and len(list(metrics_dir.glob("*.json"))) > 1:
            print(f"Skipping {run_dir}, {len(list(metrics_dir.glob('*.json')))}")
            continue
        for i in range(1, 5):
            command = [
                "python",
                str(eval_script),
                "--n-models",
                "1",
                "--run-path",
                str(run_dir),
                "--seed",
                str(i),
                "--suffix",
                f"seed_{i}",
            ]
            print(f"Running {command}")
            subprocess.run(command)


if __name__ == "__main__":
    start_date = datetime.datetime(2024, 2, 1)
    eval_all_runs(start_date)

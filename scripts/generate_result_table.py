import json
from enum import StrEnum
from typing import NamedTuple

import pandas as pd

from pinc.data import REPO_ROOT


class Metrics(NamedTuple):
    gt_chamfer: float
    gt_hausdorff: float
    scan_directed_chamfer: float
    scan_directed_hausdorff: float


PAPER_SRB_FILES = ["anchor", "daratech", "dc", "gargoyle", "lord_quas"]
PAPER_TABLE = """
IGR 0.45 7.45 0.17 4.55 4.9 42.15 0.7 3.68 0.63 10.35 0.14 3.44 0.77 17.46 0.18 2.04 0.16 4.22 0.08 1.14
SIREN 0.72 10.98 0.11 1.27 0.21 4.37 0.09 1.78 0.34 6.27 0.06 2.71 0.46 7.76 0.08 0.68 0.35 8.96 0.06 0.65
SAL 0.42 7.21 0.17 4.67 0.62 13.21 0.11 2.15 0.18 3.06 0.08 2.82 0.45 9.74 0.21 3.84 0.13 414 0.07 4.04
PHASE 0.29 7.43 0.09 1.49 0.35 7.24 0.08 1.21 0.19 4.65 0.05 2.78 0.17 4.79 0.07 1.58 0.11 0.71 0.05 0.74
DiGS 0.29 7.19 0.11 1.17 0.20 3.72 0.09 1.80 0.15 1.70 0.07 2.75 0.17 4.10 0.09 0.92 0.12 0.91 0.06 0.70
PINC 0.29 7.54 0.09 1.20 0.37 7.24 0.11 1.88 0.14 2.56 0.04 2.73 0.16 4.78 0.05 0.80 0.10 0.92 0.04 0.67
"""


class RunTypes(StrEnum):
    TYPE_1 = "jax with epsilon=0.1"
    TYPE_2 = "jax with epsilon=1.0"


RUN_IDS = {
    "anchor": {
        RunTypes.TYPE_1: "8lcvaqh4",
        RunTypes.TYPE_2: None,
    },
    "daratech": {
        RunTypes.TYPE_1: "kt67i502",
        RunTypes.TYPE_2: "teoj02mm",
    },
    "dc": {
        RunTypes.TYPE_1: "1o79somx",
        RunTypes.TYPE_2: None,
    },
    "gargoyle": {
        RunTypes.TYPE_1: "6jfgejpa",
        RunTypes.TYPE_2: "ojbs5lgy",
    },
    "lord_quas": {
        RunTypes.TYPE_1: "lc8cu813",
        RunTypes.TYPE_2: "1f5dvyoc",
    },
}


def parse_table(table: str, columns: list[str]) -> dict[str, dict[str, Metrics]]:
    """Parses the table string and returns a dictionary of the metrics for each file and method."""
    metric_results = {}
    for line in table.split("\n")[1:-1]:  # skip the first and last empty lines
        chunks = line.split(" ")
        key = chunks[0].lower()
        values = list(map(float, chunks[1:]))
        metric_results[key] = {f: Metrics(*values[i : i + 4]) for f, i in zip(columns, range(0, 20, 4))}

    # invert the dictionary to have the file names as the keys
    return {file: {method: metric_results[method][file] for method in metric_results} for file in columns}


def flatten_metrics_for_df(metrics: dict[str, dict[str, Metrics]]) -> dict[str, dict[tuple[str, str], float]]:
    # flatten the dictionary to have a tuple of (file, metric_name) as the key and the spec as the subkey
    flat_metrics = {}
    for file, specs in metrics.items():
        for spec, metric in specs.items():
            for metric_name, value in metric._asdict().items():
                if (file, metric_name) not in flat_metrics:
                    flat_metrics[(file, metric_name)] = {}
                flat_metrics[(file, metric_name)][spec] = value
    return flat_metrics


def add_runs_to_reported_metrics(reported_metrics: dict[str, dict[str, Metrics]]) -> dict[str, dict[str, Metrics]]:
    for file, run_ids in RUN_IDS.items():
        for spec, run_id in run_ids.items():
            if run_id is None:
                continue
            matching_run_dirs = list((REPO_ROOT / "wandb").glob(f"run-*-{run_id}"))
            assert len(matching_run_dirs) == 1, f"Found {len(matching_run_dirs)} matching run directories for {run_id}"
            run_dir = matching_run_dirs[0]
            assert run_dir.is_dir(), f"{run_dir} is not a directory"

            # load the metrics from the run directory
            metrics_path = run_dir / "files" / "config.json"

            with (run_dir / "files" / "config.json").open("r") as f:
                config = json.load(f)

            # Check if the config file is correct
            if spec == RunTypes.TYPE_1:
                assert config["epsilon"] == 0.1
            elif spec == RunTypes.TYPE_2:
                assert config["epsilon"] == 1.0
            else:
                raise ValueError(f"Unknown spec: {spec}")
            assert config["data_filename"] == file
            assert len(config["loss_weights"]) == 4  # check for the right version of the config

            # load the metrics from the run directory
            metrics_path = run_dir / "files" / "metrics"
            key_fun = lambda p: int(p.stem.split("_")[1])
            metrics = sorted(metrics_path.glob("model_*_metrics.json"), key=key_fun)
            if len(metrics) == 0:
                continue
            metric_path = metrics[-1]
            assert key_fun(metric_path) == 100_000
            with metric_path.open("r") as f:
                metric_data = json.load(f)
            metric = Metrics(
                gt_chamfer=metric_data["distances"]["ground_truth"]["chamfer"],
                gt_hausdorff=metric_data["distances"]["ground_truth"]["hausdorff"],
                scan_directed_chamfer=metric_data["distances"]["scan"]["directed_chamfer"],
                scan_directed_hausdorff=metric_data["distances"]["scan"]["directed_hausdorff"],
            )

            reported_metrics[file][spec] = metric
    return reported_metrics


if __name__ == "__main__":
    reported_metrics = parse_table(PAPER_TABLE, PAPER_SRB_FILES)

    reported_metrics = add_runs_to_reported_metrics(reported_metrics)

    flat_metrics = flatten_metrics_for_df(reported_metrics)

    print(pd.DataFrame(flat_metrics))

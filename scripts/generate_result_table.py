"""Generates the result table with the reported distance metrics. This script also require pandas and jinja2.
to install them: pip install pandas jinja2
"""

import json
from enum import StrEnum
from typing import NamedTuple

import numpy as np
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
    TYPE_1 = "PINC (ours) $epsilon=1$"
    TYPE_2 = "PINC (ours) $epsilon=0.1$"
    TYPE_3 = "$[$Re$]$ PINC $epsilon=1$"
    TYPE_4 = "$[$Re$]$ PINC $epsilon=0.1$"


RUN_IDS = {
    "anchor": {
        RunTypes.TYPE_1: ["2024_02_15_13_02_27"],
        RunTypes.TYPE_2: ["2024_02_12_11_31_23_lor_anchor_eps_01"],
        RunTypes.TYPE_3: ["7nsgf68t", "3pzxgbig", "zwohp52r"],
        RunTypes.TYPE_4: ["8lcvaqh4", "xw68upg6", "suobhxom"],
    },
    "daratech": {
        RunTypes.TYPE_1: ["2024_02_12_22_17_49"],
        RunTypes.TYPE_2: ["2024_02_13_10_44_11"],
        RunTypes.TYPE_3: ["teoj02mm", "ggtjf89r", "b5o23l1e"],
        RunTypes.TYPE_4: ["kt67i502", "vd7dkf40", "funmhwh3"],
    },
    "dc": {
        RunTypes.TYPE_1: ["2024_02_14_09_06_06_lor_dc_eps_1"],
        RunTypes.TYPE_2: ["2024_02_13_08_51_39_lor_dc_eps_01"],
        RunTypes.TYPE_3: ["v0l912vq", "4jsncb8b", "dqh60r92"],
        RunTypes.TYPE_4: ["1o79somx", "raimbjb8", "nq2dyh10"],
    },
    "gargoyle": {
        RunTypes.TYPE_1: ["2024_02_12_11_27_25"],
        RunTypes.TYPE_2: ["2024_02_09_19_17_27"],
        RunTypes.TYPE_3: ["ojbs5lgy", "q7e5btzg", "ljbi7xiv"],
        RunTypes.TYPE_4: ["6jfgejpa", "j7lwywt2", "mpcpqqo7"],
    },
    "lord_quas": {
        RunTypes.TYPE_1: ["2024_02_14_17_35_11"],
        RunTypes.TYPE_2: ["2024_02_13_22_30_18"],
        RunTypes.TYPE_3: ["1f5dvyoc", "s3z11lkx", "wzpaoy41"],
        RunTypes.TYPE_4: ["lc8cu813", "evqmf9oi", "pdrpsqow"],
    },
}


def parse_table(table: str, columns: list[str]) -> dict[str, dict[str, Metrics]]:
    """Parses the table string and returns a dictionary of the metrics for each file and method."""
    metric_results = {}
    for line in table.split("\n")[1:-1]:  # skip the first and last empty lines
        chunks = line.split(" ")
        key = chunks[0]
        values = list(map(float, chunks[1:]))
        metric_results[key] = {f: Metrics(*values[i : i + 4]) for f, i in zip(columns, range(0, 20, 4))}

    # invert the dictionary to have the file names as the keys
    return {file: {method: metric_results[method][file] for method in metric_results} for file in columns}


def load_run_metrics(run_id: str, file: str, spec: RunTypes) -> Metrics:
    """Loads the metrics from a run."""
    assert spec in RunTypes, f"Unknown spec: {spec}"
    if spec in [RunTypes.TYPE_4, RunTypes.TYPE_3]:
        matching_run_dirs = list((REPO_ROOT / "wandb").glob(f"run-*-{run_id}"))
        assert len(matching_run_dirs) == 1, f"Found {len(matching_run_dirs)} matching run directories for {run_id}"
        run_dir = matching_run_dirs[0]
        assert run_dir.is_dir(), f"{run_dir} is not a directory"

        # load the metrics from the run directory
        metrics_path = run_dir / "files" / "config.json"

        with (run_dir / "files" / "config.json").open("r") as f:
            config = json.load(f)

        # Check if the config file is correct
        if spec == RunTypes.TYPE_4:
            assert config["epsilon"] == 0.1
        elif spec == RunTypes.TYPE_3:
            assert config["epsilon"] == 1.0
        else:
            raise ValueError(f"Unknown spec: {spec}")
        assert config["data_filename"] == file
        assert len(config["loss_weights"]) == 4  # check for the right version of the config

        # load the metrics from the run directory
        metrics_path = run_dir / "files" / "metrics"
        key_fun = lambda p: int(p.stem.split("_")[1])
        metrics = sorted(metrics_path.glob("model_*_metrics.json"), key=key_fun)
        assert len(metrics) > 0, f"No metrics found in {metrics_path}"
        metric_path = metrics[-1]
        assert key_fun(metric_path) == 100_000
        with metric_path.open("r") as f:
            metric_data = json.load(f)
    elif spec in [RunTypes.TYPE_2, RunTypes.TYPE_1]:
        # load the metrics from the pinc code
        metric_path = REPO_ROOT / "tmp" / "metrics" / run_id / f"{file}_{run_id}.json"
    else:
        raise ValueError(f"Unknown spec: {spec}")

    with metric_path.open("r") as f:
        metric_data = json.load(f)
    return Metrics(
        gt_chamfer=metric_data["distances"]["ground_truth"]["chamfer"],
        gt_hausdorff=metric_data["distances"]["ground_truth"]["hausdorff"],
        scan_directed_chamfer=metric_data["distances"]["scan"]["directed_chamfer"],
        scan_directed_hausdorff=metric_data["distances"]["scan"]["directed_hausdorff"],
    )


def add_runs_to_reported_metrics(reported_metrics: dict[str, dict[str, Metrics]]) -> dict[str, dict[str, Metrics]]:
    for file, file_run_ids in RUN_IDS.items():
        for spec, spec_run_ids in file_run_ids.items():
            if len(spec_run_ids) == 1:  # Only one run
                reported_metrics[file][spec] = load_run_metrics(spec_run_ids[0], file, spec)
            else:  # Multiple repeated runs
                spec_metrics: list[Metrics] = []
                for run_id in spec_run_ids:
                    try:
                        spec_metrics.append(load_run_metrics(run_id, file, spec))
                    except AssertionError as e:  # noqa: PERF203
                        print(e)
                print(f"Metrics for {spec}, {file}")
                f = lambda v: f"{v:>6.4f}"
                for attribute in Metrics._fields:
                    values = [getattr(m, attribute) for m in spec_metrics]
                    value_string = ", ".join(map(f, values))
                    print(value_string, f"mean: {np.mean(values):.5f}, std: {np.std(values):.5f}")
    return reported_metrics


def flatten_metrics_for_df(metrics: dict[str, dict[str, Metrics]]) -> dict[str, dict[tuple[str, str, str], float]]:
    # flatten the dictionary to have a tuple of (file_name, compared_point_cloud, distance_name) as the key
    # as the key and the spec as the subkey
    flat_metrics = {}
    for file, specs in metrics.items():
        file_name = file.replace("_", " ").capitalize()
        for spec, metric in specs.items():
            for compared_point_cloud, distance_name, value in [
                ("GT", "chamfer", metric.gt_chamfer),
                ("GT", "hausdorff", metric.gt_hausdorff),
                ("Scan", "directed_chamfer", metric.scan_directed_chamfer),
                ("Scan", "directed_hausdorff", metric.scan_directed_hausdorff),
            ]:
                tuple_key = (file_name, compared_point_cloud, distance_name)
                if tuple_key not in flat_metrics:
                    flat_metrics[tuple_key] = {}
                flat_metrics[tuple_key][spec] = value
    return flat_metrics


if __name__ == "__main__":
    reported_metrics = parse_table(PAPER_TABLE, PAPER_SRB_FILES)

    reported_metrics = add_runs_to_reported_metrics(reported_metrics)

    flat_metrics = flatten_metrics_for_df(reported_metrics)

    df = pd.DataFrame(flat_metrics)

    n_top = len(df.columns.levels[0])  # type: ignore
    column_format = "l" + "|".join(["cccc"] * n_top)

    latex_string = (
        df.to_latex(float_format="%.2f", multirow=True, column_format=column_format, multicolumn_format="c")
        .replace("directed_chamfer", r"$d_{\overrightarrow{C}}$")
        .replace("directed_hausdorff", r"$d_{\overrightarrow{H}}$")
        .replace("chamfer", "$d_C$")
        .replace("hausdorff", "$d_H$")
        .replace(r"\toprule", r"\hline")
        .replace(r"\midrule", r"\hline")
        .replace(r"\bottomrule", r"\hline")
        .replace("NaN", " ")
        .replace("epsilon", r"\varepsilon")
    )
    print(latex_string)

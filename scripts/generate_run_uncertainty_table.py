"""Generates the result table with the reported distance metrics. This script also require pandas and jinja2.
to install them: pip install pandas jinja2
"""

import json
from enum import StrEnum
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from pinc.data import REPO_ROOT


class Metric(NamedTuple):
    value: float
    uncertainty: float | None

    def __str__(self) -> str:
        if self.uncertainty is None:
            return f"{self.value:.3f}"
        return f"{self.value:.3f} ± {self.uncertainty:.3f}"


class Metrics(NamedTuple):
    gt_chamfer: Metric
    gt_hausdorff: Metric
    scan_directed_chamfer: Metric
    scan_directed_hausdorff: Metric
    normal_consistency: None | Metric


PAPER_SRB_FILES = ["anchor", "daratech", "dc", "gargoyle", "lord_quas"]
PAPER_TABLE = """
PINC 0.29 7.54 0.09 1.20 0.37 7.24 0.11 1.88 0.14 2.56 0.04 2.73 0.16 4.78 0.05 0.80 0.10 0.92 0.04 0.67
"""
PINC_NC_TABLE = dict(zip(PAPER_SRB_FILES, [0.9754, 0.9311, 0.9828, 0.9803, 0.9915]))


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
        metric_results[key] = {}
        for f, i in zip(columns, range(0, 20, 4)):
            metric_results[key][f] = Metrics(*[Metric(v, None) for v in values[i : i + 4]], Metric(PINC_NC_TABLE[f], None))

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
        gt_chamfer=Metric(metric_data["distances"]["ground_truth"]["chamfer"], None),
        gt_hausdorff=Metric(metric_data["distances"]["ground_truth"]["hausdorff"], None),
        scan_directed_chamfer=Metric(metric_data["distances"]["scan"]["directed_chamfer"], None),
        scan_directed_hausdorff=Metric(metric_data["distances"]["scan"]["directed_hausdorff"], None),
        normal_consistency=Metric(metric_data["normal_consistency"], None) if "normal_consistency" in metric_data else None,
    )


def load_uncertainty_metrics(spec_run_ids: list[str], file: str, spec: RunTypes) -> Metrics:
    spec_metrics = [load_run_metrics(run_id, file, spec) for run_id in spec_run_ids]
    uncertainty_metrics = {}
    for attribute in Metrics._fields:
        values = [getattr(m, attribute).value for m in spec_metrics]
        mean = np.mean(values)
        std = np.std(values)
        t_student_interval = stats.t.ppf(0.975, len(values)) * std
        uncertainty_metrics[attribute] = Metric(float(mean), float(t_student_interval))
    return Metrics(**uncertainty_metrics)


def add_runs_to_reported_metrics(reported_metrics: dict[str, dict[str, Metrics]]) -> dict[str, dict[str, Metrics]]:
    for file, run_ids in RUN_IDS.items():
        for spec, spec_run_ids in run_ids.items():
            if len(spec_run_ids) == 1:  # Only one run
                pass  # reported_metrics[file][spec] = load_run_metrics(spec_run_ids[0], file, spec)
            else:  # Multiple repeated runs
                reported_metrics[file][spec] = load_uncertainty_metrics(spec_run_ids, file, spec)
    return reported_metrics


def flatten_metrics_for_df(metrics: dict[str, dict[str, Metrics]]) -> dict[str, dict[tuple[str, str, str], float]]:
    # flatten the dictionary to have a tuple of (file_name, compared_point_cloud, distance_name) as the key
    # as the key and the spec as the subkey
    flat_metrics = {}
    for file, specs in metrics.items():
        file_name = file.replace("_", " ").capitalize()
        for spec, metric in specs.items():
            for compared_point_cloud, distance_name, value in [
                ("GT", "chamfer", str(metric.gt_chamfer)),
                ("GT", "hausdorff", str(metric.gt_hausdorff)),
                ("Scan", "directed_chamfer", str(metric.scan_directed_chamfer)),
                ("Scan", "directed_hausdorff", str(metric.scan_directed_hausdorff)),
                ("", "Normal Consistency", str(metric.normal_consistency)),
            ]:
                col_key = (compared_point_cloud, distance_name)
                row_key = (file_name, spec)
                if col_key not in flat_metrics:
                    flat_metrics[col_key] = {}
                flat_metrics[col_key][row_key] = value
    return flat_metrics


if __name__ == "__main__":
    reported_metrics = parse_table(PAPER_TABLE, PAPER_SRB_FILES)

    reported_metrics = add_runs_to_reported_metrics(reported_metrics)

    flat_metrics = flatten_metrics_for_df(reported_metrics)

    df = pd.DataFrame(flat_metrics)

    n_top = len(df.columns.levels[0])  # type: ignore
    column_format = "llcc|cc|c"

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
        .replace("$[$Re$]$ PINC ", "")
    )
    print(latex_string)

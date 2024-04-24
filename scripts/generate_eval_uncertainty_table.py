"""Generates the result table with the reported distance metrics. This script also require pandas and jinja2.
to install them: pip install pandas jinja2
"""

import json
from enum import StrEnum
from typing import NamedTuple

import numpy as np
import pandas as pd

from pinc.data import REPO_ROOT


class Metric(NamedTuple):
    value: float
    uncertainty: float | None

    def __str__(self) -> str:
        if self.uncertainty is None:
            return f"{self.value:.2f}"
        return f"{self.value:.5f} ± {self.uncertainty:.5f}"


class Metrics(NamedTuple):
    gt_chamfer: Metric
    gt_hausdorff: Metric
    scan_directed_chamfer: Metric
    scan_directed_hausdorff: Metric


PAPER_SRB_FILES = ["anchor", "daratech", "dc", "gargoyle", "lord_quas"]


class RunTypes(StrEnum):
    # TYPE_1 = "PINC (ours) $epsilon=1$"
    # TYPE_2 = "PINC (ours) $epsilon=0.1$"
    TYPE_3 = "$epsilon=1$"  # $[$Re$]$ PINC
    TYPE_4 = "$epsilon=0.1$"  # $[$Re$]$ PINC


RUN_IDS = {
    "anchor": {
        # RunTypes.TYPE_1: "2024_02_15_13_02_27",
        # RunTypes.TYPE_2: "2024_02_12_11_31_23_lor_anchor_eps_01",
        RunTypes.TYPE_3: "7nsgf68t",
        RunTypes.TYPE_4: "8lcvaqh4",
    },
    "daratech": {
        # RunTypes.TYPE_1: "2024_02_12_22_17_49",
        # RunTypes.TYPE_2: "2024_02_13_10_44_11",
        RunTypes.TYPE_3: "teoj02mm",
        RunTypes.TYPE_4: "kt67i502",
    },
    "dc": {
        # RunTypes.TYPE_1: "2024_02_14_09_06_06_lor_dc_eps_1",
        # RunTypes.TYPE_2: "2024_02_13_08_51_39_lor_dc_eps_01",
        RunTypes.TYPE_3: "v0l912vq",
        RunTypes.TYPE_4: "1o79somx",
    },
    "gargoyle": {
        # RunTypes.TYPE_1: "2024_02_12_11_27_25",
        # RunTypes.TYPE_2: "2024_02_09_19_17_27",
        RunTypes.TYPE_3: "ojbs5lgy",
        RunTypes.TYPE_4: "6jfgejpa",
    },
    "lord_quas": {
        # RunTypes.TYPE_1: "2024_02_14_17_35_11",
        # RunTypes.TYPE_2: "2024_02_13_22_30_18",
        RunTypes.TYPE_3: "1f5dvyoc",
        RunTypes.TYPE_4: "lc8cu813",
    },
}


def load_run_metrics(run_id: str, file: str, spec: RunTypes, suffix: str) -> Metrics:
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
        pattern = f"model_100000_metrics{suffix}.json"
        metrics = sorted(metrics_path.glob(pattern), key=key_fun)
        assert len(metrics) > 0, f"No metrics found in {metrics_path} for pattern {pattern}"
        metric_path = metrics[-1]
        assert key_fun(metric_path) == 100_000
        with metric_path.open("r") as f:
            metric_data = json.load(f)
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


def load_uncertainty_metrics(run_id: str, file: str, spec: RunTypes) -> Metrics:
    suffixes = [""] + [f"_seed_{i}" for i in range(1, 15)]
    spec_metrics = [load_run_metrics(run_id, file, spec, suffix) for suffix in suffixes]
    uncertainty_metrics = {}
    for attribute in Metrics._fields:
        values = [getattr(m, attribute) for m in spec_metrics]
        mean = np.mean(values)
        std = np.std(values)
        uncertainty_metrics[attribute] = Metric(float(mean), float(std))
    return Metrics(**uncertainty_metrics)


def parse_uncertainty() -> dict[str, dict[str, Metrics]]:
    reported_metrics = {file: {} for file in PAPER_SRB_FILES}
    for file, run_ids in RUN_IDS.items():
        for spec, run_id in run_ids.items():
            reported_metrics[file][spec] = load_uncertainty_metrics(run_id, file, spec)
    return reported_metrics


def flatten_metrics_for_df(metrics: dict[str, dict[str, Metrics]]) -> dict[str, dict[tuple[str, str, str], float]]:
    # flatten the dictionary to have a tuple of (file_name, compared_point_cloud, distance_name) as the key
    # as the key and the spec as the subkey
    flat_metrics = {}
    for file in PAPER_SRB_FILES:
        file_name = file.replace("_", " ").capitalize()
        for spec in RunTypes:
            metric = metrics[file][spec]
            for compared_point_cloud, distance_name, value in [
                ("GT", "chamfer", metric.gt_chamfer),
                ("GT", "hausdorff", metric.gt_hausdorff),
                ("Scan", "directed_chamfer", metric.scan_directed_chamfer),
                ("Scan", "directed_hausdorff", metric.scan_directed_hausdorff),
            ]:
                col_key = (compared_point_cloud, distance_name)
                row_key = (file_name, spec)
                if col_key not in flat_metrics:
                    flat_metrics[col_key] = {}
                flat_metrics[col_key][row_key] = value
    return flat_metrics


if __name__ == "__main__":
    uncertainty = parse_uncertainty()

    flat_metrics = flatten_metrics_for_df(uncertainty)

    df = pd.DataFrame(flat_metrics)

    n_top = len(df.columns.levels[0])  # type: ignore
    column_format = "ll" + "|".join(["cc"] * n_top)

    latex_string = (
        df.to_latex(float_format="%.2f", multirow=True, multicolumn_format="c", column_format=column_format)
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

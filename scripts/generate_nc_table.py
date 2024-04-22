"""Generates the result table with the reported normal consistency metrics. This script also require pandas and jinja2.
to install them: pip install pandas jinja2
"""

import json
from enum import StrEnum

import pandas as pd

from pinc.data import REPO_ROOT

PAPER_SRB_FILES = ["anchor", "daratech", "dc", "gargoyle", "lord_quas"]
PAPER_TABLE = """
IGR 0.9706 0.8526 0.9800 0.9765 0.9901
SIREN 0.9438 0.9682 0.9735 0.9392 0.9762
DiGS 0.9767 0.9680 0.9826 0.9788 0.9907
SAP 0.9750 0.9414 0.9636 0.9731 0.9838
PINC 0.9754 0.9311 0.9828 0.9803 0.9915
"""


class RunTypes(StrEnum):
    # TYPE_1 = "PINC (ours) $epsilon=1$"
    # TYPE_2 = "PINC (ours) $epsilon=0.1$"
    TYPE_3 = "$[$Re$]$ PINC $epsilon=1$"
    TYPE_4 = "$[$Re$]$ PINC $epsilon=0.1$"


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


def parse_table(table: str, columns: list[str]) -> dict[str, dict[str, float]]:
    """Parses the table string and returns a dictionary of the metrics for each file and method."""
    metric_results = {}
    for line in table.split("\n")[1:-1]:  # skip the first and last empty lines
        chunks = line.split(" ")
        key = chunks[0]
        values = list(map(float, chunks[1:]))
        metric_results[key] = dict(zip(columns, values))

    # invert the dictionary to have the file names as the keys
    return {file: {method: metric_results[method][file] for method in metric_results} for file in columns}


def load_run_metrics(run_id: str, file: str, spec: RunTypes) -> float:
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
    # elif spec in [RunTypes.TYPE_2, RunTypes.TYPE_1]:
    #     # load the metrics from the pinc code
    #     metric_path = REPO_ROOT / "tmp" / "metrics" / run_id / f"{file}_{run_id}.json"
    else:
        raise ValueError(f"Unknown spec: {spec}")

    with metric_path.open("r") as f:
        metric_data = json.load(f)
    return metric_data["normal_consistency"]


def add_runs_to_reported_metrics(reported_metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    for file, run_ids in RUN_IDS.items():
        for spec, run_id in run_ids.items():
            reported_metrics[file][spec] = load_run_metrics(run_id, file, spec)
    return reported_metrics


def flatten_metrics_for_df(metrics: dict[str, dict[str, float]]) -> dict[str, dict[tuple[str, str, str], float]]:
    # flatten the dictionary to have a tuple of (file_name, compared_point_cloud, distance_name) as the key
    # as the key and the spec as the subkey
    flat_metrics = {}
    for file, specs in metrics.items():
        file_name = file.replace("_", " ").capitalize()
        for spec, nc_value in specs.items():
            if file_name not in flat_metrics:
                flat_metrics[file_name] = {}
            flat_metrics[file_name][spec] = nc_value
    return flat_metrics


if __name__ == "__main__":
    reported_metrics = parse_table(PAPER_TABLE, PAPER_SRB_FILES)

    reported_metrics = add_runs_to_reported_metrics(reported_metrics)

    flat_metrics = flatten_metrics_for_df(reported_metrics)

    df = pd.DataFrame(flat_metrics)

    column_format = "l" + "c" * len(PAPER_SRB_FILES)

    latex_string = (
        df.to_latex(float_format="%.4f", multirow=True, column_format=column_format)
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

"""Generates the plots for the paper."""

from enum import StrEnum

import numpy as np
import trimesh
from plotly import graph_objects as go

# from plotly.subplots import make_subplots
from pinc.data import REPO_ROOT, process_points
from pinc.visualize import plot_mesh, plot_points

PAPER_SRB_FILES = ["anchor", "daratech", "dc", "gargoyle", "lord_quas"]


class RunTypes(StrEnum):
    TYPE_1 = "PINC (ours) $epsilon=1$"
    TYPE_2 = "PINC (ours) $epsilon=0.1$"
    TYPE_3 = "$[$Re$]$ PINC $epsilon=1$"
    TYPE_4 = "$[$Re$]$ PINC $epsilon=0.1$"


RUN_IDS = {
    "anchor": {
        RunTypes.TYPE_1: "2024_02_15_13_02_27",
        RunTypes.TYPE_2: "2024_02_12_11_31_23_lor_anchor_eps_01",
        RunTypes.TYPE_3: "7nsgf68t",
        RunTypes.TYPE_4: "8lcvaqh4",
    },
    "daratech": {
        RunTypes.TYPE_1: "2024_02_12_22_17_49",
        RunTypes.TYPE_2: "2024_02_13_10_44_11",
        RunTypes.TYPE_3: "teoj02mm",
        RunTypes.TYPE_4: "kt67i502",
    },
    "dc": {
        RunTypes.TYPE_1: "2024_02_14_09_06_06_lor_dc_eps_1",
        RunTypes.TYPE_2: "2024_02_13_08_51_39_lor_dc_eps_01",
        RunTypes.TYPE_3: "v0l912vq",
        RunTypes.TYPE_4: "1o79somx",
    },
    "gargoyle": {
        RunTypes.TYPE_1: "2024_02_12_11_27_25",
        RunTypes.TYPE_2: "2024_02_09_19_17_27",
        RunTypes.TYPE_3: "ojbs5lgy",
        RunTypes.TYPE_4: "6jfgejpa",
    },
    "lord_quas": {
        RunTypes.TYPE_1: "2024_02_14_17_35_11",
        RunTypes.TYPE_2: "2024_02_13_22_30_18",
        RunTypes.TYPE_3: "1f5dvyoc",
        RunTypes.TYPE_4: "lc8cu813",
    },
}


def load_mesh(run_id: str, spec: RunTypes, center_point: np.ndarray, max_coord: float) -> trimesh.Trimesh:
    """Loads the metrics from a run."""
    assert spec in RunTypes, f"Unknown spec: {spec}"
    if spec in [RunTypes.TYPE_4, RunTypes.TYPE_3]:
        matching_run_dirs = list((REPO_ROOT / "wandb").glob(f"run-*-{run_id}"))
        assert len(matching_run_dirs) == 1, f"Found {len(matching_run_dirs)} matching run directories for {run_id}"
        mesh_path = matching_run_dirs[0] / "files" / "reconstructions" / "model_100000.ply"
    elif spec in [RunTypes.TYPE_2, RunTypes.TYPE_1]:
        mesh_path = REPO_ROOT / "tmp" / "metrics" / run_id / "igr_100000_single_shape.ply"
    else:
        raise ValueError(f"Unknown spec: {spec}")
    assert mesh_path.is_file(), f"{mesh_path} is not a file"
    mesh: trimesh.Trimesh = trimesh.load_mesh(mesh_path)  # type: ignore
    if spec in [RunTypes.TYPE_4, RunTypes.TYPE_3]:
        return mesh
    mesh.vertices = mesh.vertices * max_coord + center_point
    return mesh


def plot_trimesh(mesh: trimesh.Trimesh, **kwargs) -> go.Mesh3d:
    """Plots the trimesh."""
    points, triangles = mesh.vertices, mesh.faces
    return plot_mesh(points, triangles, **kwargs)


CAMERAS = {
    "anchor": dict(
        eye=dict(x=1, y=-1.4, z=1),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
    ),
    "daratech": dict(
        eye=dict(x=0, y=-1.5, z=1),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0.07, y=0, z=0),
    ),
    "dc": dict(
        eye=dict(x=0, y=0.5, z=1.5),
        up=dict(x=0, y=1, z=0),
        center=dict(x=-0.05, y=0, z=0),
    ),
    "gargoyle": dict(
        eye=dict(x=1, y=-0.5, z=-1.2),
        up=dict(x=0, y=-1, z=0),
        center=dict(x=0, y=0, z=0),
    ),
    "lord_quas": dict(
        eye=dict(x=1.4, y=1, z=1.4),
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0.1, z=0),
    ),
}


def make_and_save_figure(trace, camera: dict[str, dict], name: str):
    fig = go.Figure(
        data=[trace],
        layout=go.Layout(
            scene=go.layout.Scene(
                xaxis_visible=False,
                yaxis_visible=False,
                zaxis_visible=False,
                camera=camera,
            ),
        ),
    )
    print(f"Writing {name}")
    fig.write_image(REPO_ROOT / f"tmp/figs/{name}", scale=6, width=1080, height=1080)
    print("Done writing file")


def main(file: str, color: str) -> None:
    assert file in PAPER_SRB_FILES, f"Unknown file: {file}"
    camera = CAMERAS[file]
    scan_mesh: trimesh.PointCloud = trimesh.load(REPO_ROOT / "data" / "scans" / f"{file}.ply")  # type: ignore

    points, max_coord, center_point = process_points(scan_mesh.vertices)

    marker = dict(
        size=0.3,  # Set marker size
        color=color,  # Set marker color
        symbol="circle",  # Set marker symbol ('circle' in this case)
        opacity=0.7,  # Set marker opacity
    )

    trace = plot_points(points, marker=marker)
    make_and_save_figure(trace, camera, f"{file}_scan.png")

    for type_index, spec in enumerate(RunTypes, 1):
        mesh = load_mesh(RUN_IDS[file][spec], spec, center_point, max_coord)
        trace = plot_trimesh(mesh, color=color)
        make_and_save_figure(trace, camera, f"{file}_{type_index}.png")


COLORS = ["lightblue", "lightgreen", "lightcoral", "lightgray", "orange"]

if __name__ == "__main__":
    for file, color in zip(PAPER_SRB_FILES, COLORS):
        main(file, color)

# %%

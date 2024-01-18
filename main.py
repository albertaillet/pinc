# %%
import numpy as np
import open3d as o3d
from pathlib import Path
from plotly import graph_objects as go
from functools import partial
from skimage.measure import marching_cubes
from typing import Callable


def process_point_cloud(point_cloud: o3d.geometry.PointCloud) -> tuple[np.ndarray, float, np.ndarray]:
    points = np.asarray(point_cloud.points).astype(np.float32)
    center_point = np.mean(points, axis=0)
    points = points - center_point
    max_coord = np.abs(points).max()
    points = points / max_coord
    assert points.shape[1] == 3
    return points, max_coord, center_point


def sdf_sphere(points: np.ndarray, radius: float) -> np.ndarray:
    return np.linalg.norm(points, axis=1) - radius


def sdf_torus(points: np.ndarray, radius: float, tube_radius: float) -> float:
    q = np.stack([np.linalg.norm(points[:, :2], axis=-1) - radius, points[:, 2]], axis=-1)
    return np.linalg.norm(q, axis=-1) - tube_radius


def mesh_from_sdf(f: Callable, max_pts: float, center: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-max_pts, max_pts, resolution)
    y = np.linspace(-max_pts, max_pts, resolution)
    z = np.linspace(-max_pts, max_pts, resolution)
    points = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
    points = points + center
    values = f(points)
    verts, faces, _, _ = marching_cubes(values.reshape(resolution, resolution, resolution), 0)
    return verts, faces


def figure(trace, title) -> go.Figure:
    return go.Figure(
        data=[trace],
        layout=go.Layout(
            title=title,
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
            ),
        ),
    )


def plot_points(points, title="") -> go.Figure:
    return figure(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=1, opacity=1, colorscale="Viridis", color=points[:, 2]),
        ),
        title,
    )


def plot_mesh(points, triangles, title="") -> go.Figure:
    return figure(
        go.Mesh3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            opacity=0.5,
        ),
        title,
    )


# %%
# repo_dir = Path(__file__).resolve().parent
# data_dir = repo_dir / "data"
# scans_dir = data_dir / "scans"
# scan = o3d.io.read_point_cloud(str(scans_dir / "gargoyle.ply"))
# points_array = np.asarray(scan.points)  # (95435, 3)
# point_set, max_pts, center = process_point_cloud(scan)
# plot_points(point_set, title="Gargoyle").show()


# %%
f = partial(sdf_torus, radius=0.2, tube_radius=0.1)
verts, faces = mesh_from_sdf(f, 1, np.zeros(3), resolution=100)
plot_mesh(verts, faces, title="Sphere").show()

# %%

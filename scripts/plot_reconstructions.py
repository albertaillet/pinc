# %%
from pathlib import Path
from pprint import pprint

import numpy as np
import trimesh

from pinc.distance import mesh_distances
from pinc.visualize import figure, plot_mesh, plot_points


def process_points(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Center and normalize points."""
    center_point = np.mean(points, axis=0)
    points = points - center_point
    max_coord = np.abs(points).max()
    points = points / max_coord
    return points, max_coord, center_point


REPO_ROOT = Path(__file__).resolve().parent.parent

# %%
ground_truth_point_cloud: trimesh.PointCloud = trimesh.load(REPO_ROOT / "data/ground_truth/gargoyle.xyz")  # type: ignore
scan_point_cloud: trimesh.PointCloud = trimesh.load(REPO_ROOT / "data/scans/gargoyle.ply")  # type: ignore
recon_mesh: trimesh.Trimesh = trimesh.load(REPO_ROOT / "temp/model_95000.ply")  # type: ignore
recon_mesh_pinc: trimesh.Trimesh = trimesh.load(REPO_ROOT / "temp/igr_100000_single_shape.ply")  # type: ignore

# %%
scan_center_point = scan_point_cloud.vertices.mean(axis=0)
scan_max_coord = np.abs(scan_point_cloud.vertices - scan_center_point).max()
rescaled_recon_mesh_pinc = recon_mesh_pinc.copy()
rescaled_recon_mesh_pinc.vertices = recon_mesh_pinc.vertices * scan_max_coord + scan_center_point


# %%
def compare_reconstructions(point_could: trimesh.PointCloud, recon_mesh: trimesh.Trimesh, *, normalize: bool, title: str):
    point_cloud_points = point_could.vertices
    if len(point_cloud_points) > 1e6:
        point_cloud_points = point_cloud_points[::10]

    recon_trace = plot_mesh(recon_mesh.vertices, recon_mesh.faces)
    point_cloud_trace = plot_points(point_cloud_points, marker=dict(size=1, opacity=1, color="red"))

    fig = figure(recon_trace, point_cloud_trace, title=title)
    fig.show(renderer="browser")


# %%
compare_reconstructions(ground_truth_point_cloud, recon_mesh, normalize=False, title="Ground truth vs. Reproduced")

# %%
compare_reconstructions(scan_point_cloud, recon_mesh, normalize=True, title="Scan vs. Reproduced")

# %%
compare_reconstructions(ground_truth_point_cloud, recon_mesh_pinc, normalize=True, title="Ground truth vs. PINC")

# %%
compare_reconstructions(scan_point_cloud, recon_mesh_pinc, normalize=True, title="Scan vs. PINC")

# %%
# Plot the ground truth point cloud vs the scan point cloud
ground_truth_trace = plot_points(ground_truth_point_cloud.vertices[::20], color="red")
scan_trace = plot_points(scan_point_cloud.vertices, color="blue")
fig = figure(ground_truth_trace, scan_trace, title="Ground truth vs. Scan")
fig.show(renderer="browser")

# %%
n_samples, seed, workers = 1000, 0, 5

# %%
dists = mesh_distances(recon_mesh, ground_truth_point_cloud, scan_point_cloud, n_samples=n_samples, seed=seed, workers=workers)
pprint(dists)

# %%
dists = mesh_distances(
    rescaled_recon_mesh_pinc, ground_truth_point_cloud, scan_point_cloud, n_samples=n_samples, seed=seed, workers=workers
)
pprint(dists)

# %%

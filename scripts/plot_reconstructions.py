# %%
from pprint import pprint

import numpy as np
import trimesh

from pinc.data import REPO_ROOT
from pinc.distance import mesh_distances
from pinc.visualize import figure, plot_mesh, plot_points


def process_points(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Center and normalize points."""
    center_point = np.mean(points, axis=0)
    points = points - center_point
    max_coord = np.abs(points).max()
    points = points / max_coord
    return points, max_coord, center_point


# %%
ground_truth_point_cloud: trimesh.PointCloud = trimesh.load(REPO_ROOT / "data/ground_truth/gargoyle.xyz")  # type: ignore
scan_point_cloud: trimesh.PointCloud = trimesh.load(REPO_ROOT / "data/scans/gargoyle.ply")  # type: ignore
recon_mesh: trimesh.Trimesh = trimesh.load(REPO_ROOT / "temp/model_100000.ply")  # type: ignore
recon_mesh_pinc: trimesh.Trimesh = trimesh.load(REPO_ROOT / "temp/igr_100000_single_shape.ply")  # type: ignore

# %%
scan_center_point = scan_point_cloud.vertices.mean(axis=0)
scan_max_coord = np.abs(scan_point_cloud.vertices - scan_center_point).max()
rescaled_recon_mesh_pinc = recon_mesh_pinc.copy()
rescaled_recon_mesh_pinc.vertices = recon_mesh_pinc.vertices * scan_max_coord + scan_center_point


# %%
def compare_meshes(point_could: trimesh.PointCloud, recon_mesh: trimesh.Trimesh, *, title: str):
    point_cloud_points = point_could.vertices
    if len(point_cloud_points) > 1e6:
        point_cloud_points = point_cloud_points[::10]

    recon_trace = plot_mesh(recon_mesh.vertices, recon_mesh.faces)
    point_cloud_trace = plot_points(point_cloud_points, marker=dict(size=1, opacity=1, color="red"))

    fig = figure(recon_trace, point_cloud_trace, title=title)
    fig.show(renderer="browser")


# %%
compare_meshes(ground_truth_point_cloud, recon_mesh, title="Ground truth vs. Reproduced")

# %%
compare_meshes(scan_point_cloud, recon_mesh, title="Scan vs. Reproduced")

# %%
compare_meshes(ground_truth_point_cloud, rescaled_recon_mesh_pinc, title="Ground truth vs. PINC")

# %%
compare_meshes(scan_point_cloud, rescaled_recon_mesh_pinc, title="Scan vs. PINC")

# %%
# Plot the ground truth point cloud vs the scan point cloud
ground_truth_trace = plot_points(ground_truth_point_cloud.vertices[::20], marker=dict(size=1, opacity=1, color="red"))
scan_trace = plot_points(scan_point_cloud.vertices, marker=dict(size=1, opacity=1, color="blue"))
fig = figure(ground_truth_trace, scan_trace, title="Ground truth vs. Scan")
fig.show(renderer="browser")

# %%
n_samples, seed, workers = 10_000_000, 0, 5

# %%
repro_dists = mesh_distances(
    recon_mesh, ground_truth_point_cloud, scan_point_cloud, n_samples=n_samples, seed=seed, workers=workers
)
pprint(repro_dists)

# %%
pinc_dists = mesh_distances(
    rescaled_recon_mesh_pinc, ground_truth_point_cloud, scan_point_cloud, n_samples=n_samples, seed=seed, workers=workers
)
pprint(pinc_dists)


# %%
def relevant_metrics(*distances: dict[str, dict[str, float]]) -> list[str]:
    """Return a string with the relevant metrics."""
    # we want the gt chamfer and hausdorff and the scan directed_chamfer and directed_hausdorff
    # cols = "GT chamfer, GT hausdorff, Scan chamfer, Scan hausdorff"
    return [
        "{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
            dist["ground_truth"]["chamfer"],
            dist["ground_truth"]["hausdorff"],
            dist["scan"]["directed_chamfer"],
            dist["scan"]["directed_hausdorff"],
        )
        for dist in distances
    ]


print(*relevant_metrics(repro_dists, pinc_dists), sep="\n")

# %%

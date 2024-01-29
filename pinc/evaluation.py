import jax.numpy as jnp
import trimesh

from pinc.data import REPO_ROOT
from pinc.distance import mesh_distances
from pinc.model import mlp_forward
from pinc.normal_consistency import compute_normal_consistency
from pinc.utils import mesh_from_sdf


def eval_step(params, points, normals, static, max_coord, center_point, data_filename, n_eval_samples):
    normal_consistency = compute_normal_consistency(points=points, normals=normals, params=params, static=static)

    def sdf(x: jnp.ndarray) -> jnp.ndarray:
        return mlp_forward(params, x, activation=static.activation, skip_layers=static.skip_layers)[0]

    recon_vertices, recon_faces = mesh_from_sdf(sdf, grid_range=1.5, resolution=10, level=0)  # TODO: resolution 256 in paper
    recon_vertices = recon_vertices * max_coord + center_point
    recon_mesh = trimesh.Trimesh(vertices=recon_vertices, faces=recon_faces)
    if data_filename != "sphere":
        ground_truth_mesh = trimesh.load(REPO_ROOT / f"data/ground_truth/{data_filename}.xyz")
        scan_mesh = trimesh.load(REPO_ROOT / f"data/scans/{data_filename}.ply")
        assert isinstance(ground_truth_mesh, trimesh.PointCloud) and isinstance(scan_mesh, trimesh.PointCloud)
        distances = mesh_distances(recon_mesh, ground_truth_mesh, scan_mesh, n_samples=n_eval_samples)
    else:
        distances = dict()

    return dict(normal_consistency=normal_consistency, **distances)

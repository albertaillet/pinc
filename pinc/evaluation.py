from functools import partial

import jax.numpy as jnp
import numpy as np
import trimesh
from jax import ShapeDtypeStruct, pure_callback  # type: ignore

from pinc.distance import mesh_distances
from pinc.model import mlp_forward
from pinc.normal_consistency import compute_normal_consistency
from pinc.utils import mesh_from_sdf_grid, sdf_grid_from_sdf

GRID_RANGE = 1.5
RESOLUTION = 10  # TODO(albertaillet): 256 in paper
LEVEL = 0


return_format = {
    "reconstruction": {
        "chamfer": ShapeDtypeStruct(tuple(), jnp.float32),
        "directed_chamfer": ShapeDtypeStruct(tuple(), jnp.float32),
        "hausdorff": ShapeDtypeStruct(tuple(), jnp.float32),
        "directed_hausdorff": ShapeDtypeStruct(tuple(), jnp.float32),
    },
    "scan": {
        "chamfer": ShapeDtypeStruct(tuple(), jnp.float32),
        "directed_chamfer": ShapeDtypeStruct(tuple(), jnp.float32),
        "hausdorff": ShapeDtypeStruct(tuple(), jnp.float32),
        "directed_hausdorff": ShapeDtypeStruct(tuple(), jnp.float32),
    },
}

dict_comprehenstion = {
    data_type: {
        metric: ShapeDtypeStruct(tuple(), jnp.float32)
        for metric in ["chamfer", "directed_chamfer", "hausdorff", "directed_hausdorff"]
    }
    for data_type in ["reconstruction", "scan"]
}


def cpu_function(
    grid: np.ndarray,
    max_coord: float,
    center_point: np.ndarray,
    ground_truth_mesh: trimesh.PointCloud,
    scan_mesh: trimesh.PointCloud,
    n_eval_samples: int,
) -> dict[str, dict[str, np.float32]]:
    recon_vertices, recon_faces = mesh_from_sdf_grid(grid, grid_range=GRID_RANGE, resolution=RESOLUTION, level=LEVEL)
    recon_vertices = recon_vertices * max_coord + center_point
    recon_mesh = trimesh.Trimesh(vertices=recon_vertices, faces=recon_faces)
    return mesh_distances(recon_mesh, ground_truth_mesh, scan_mesh, n_samples=n_eval_samples)


def eval_step(params, points, normals, static, max_coord, center_point, ground_truth_mesh, scan_mesh, n_eval_samples):
    normal_consistency = compute_normal_consistency(points=points, normals=normals, params=params, static=static)

    def sdf(x: jnp.ndarray) -> jnp.ndarray:
        return mlp_forward(params, x, activation=static.activation, skip_layers=static.skip_layers)[0]

    sdf_grid = sdf_grid_from_sdf(sdf, grid_range=GRID_RANGE, resolution=RESOLUTION)
    fun = partial(
        cpu_function,
        max_coord=max_coord,
        center_point=center_point,
        ground_truth_mesh=ground_truth_mesh,
        scan_mesh=scan_mesh,
        n_eval_samples=n_eval_samples,
    )

    distances = pure_callback(fun, return_format, sdf_grid)
    return dict(normal_consistency=normal_consistency, **distances)

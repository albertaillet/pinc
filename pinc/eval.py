import jax.numpy as jnp
import numpy as np
from jax import Array, vmap

from pinc.distance import chamfer, directed_chamfer, directed_hausdorff, hausdorff
from pinc.model import Params, StaticLossArgs, get_variables, mlp_forward
from pinc.utils import mesh_from_sdf


def normal_consistency(x: Array, y: Array) -> Array:
    """Normal consistency: NC(G, n)= 1 / N sum_{i=1}^N |G(x_i)^T n_i|."""
    return jnp.mean(jnp.abs(jnp.sum(x * y, axis=1)))


def computer_normal_consistency(points: Array, normals: Array, params: Params, static: StaticLossArgs) -> float:
    """Computes the normal consistency of a point cloud."""

    def get_G(x: Array) -> Array:
        return get_variables(params, x, activation=static.activation, F=static.F, skip_layers=static.skip_layers)[2]

    G = vmap(get_G)(points)
    return normal_consistency(G, normals).item()


def compute_distances(
    points: Array, params: Params, static: StaticLossArgs, grid_range: float, resolution: int, level: float
) -> dict[str, float]:
    """Computes the distance metrics between a the implicit function and a point cloud."""

    # Level set extraction
    def sdf(point: Array) -> Array:
        return mlp_forward(params, point, activation=static.activation, skip_layers=static.skip_layers)[0]

    level = 2  # TODO: this should be zero
    level_set_points, _, _, _ = mesh_from_sdf(sdf, grid_range=grid_range, resolution=resolution, level=level)

    # TODO: use trimesh.sample.sample_surface here

    # Calculate the distance metrics and return
    numpy_points = np.asarray(points)
    return {
        "chamfer": chamfer(level_set_points, numpy_points),
        "directed_chamfer": directed_chamfer(level_set_points, numpy_points),
        "hausdorff": hausdorff(level_set_points, numpy_points),
        "directed_hausdorff": directed_hausdorff(level_set_points, numpy_points),
    }


if __name__ == "__main__":
    from pathlib import Path

    from jax import nn, random

    from pinc.data import load_ply, process_points
    from pinc.model import init_mlp_params

    skip_layers = [1]
    activation = nn.relu
    params = init_mlp_params([3, 128, 128, 7], random.key(0), skip_layers)
    F = lambda x: x / 3

    repo_root = Path(__file__).resolve().parent.parent
    points, normals = load_ply(repo_root / "data/scans/gargoyle.ply")
    points, _, _ = process_points(points)
    points, normals = jnp.array(points), jnp.array(normals)

    static = StaticLossArgs(activation=activation, F=F, skip_layers=skip_layers, loss_weights=None, epsilon=None)  # type: ignore
    nc = computer_normal_consistency(points, normals, params, static)
    print({"normal_consistency": nc})

    dists = compute_distances(points, params, static, grid_range=1.5, resolution=40, level=0.0)
    print(dists)

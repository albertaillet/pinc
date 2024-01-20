from functools import partial

import numpy as np
from jax import Array, vmap
from skimage.measure import marching_cubes

from pinc.distance import chamfer, directed_chamfer, directed_hausdorff, hausdorff
from pinc.model import Params, StaticLossArgs, get_variables, mlp_forward


def normal_consistency(x: Array, y: Array) -> Array:
    """Normal consistency: NC(G, n)= 1 / N sum_{i=1}^N |G(x_i)^T n_i|."""
    return jnp.mean(jnp.abs(jnp.sum(x * y, axis=1)))


def evaluate(points: Array, normals: Array, params: Params, static: StaticLossArgs) -> dict[str, float]:
    # Calculate the normal consistency
    forward = partial(get_variables, params, activation=static.activation, F=static.F, skip_layers=static.skip_layers)
    (_, _, G, _, _) = vmap(forward)(points)
    nc = normal_consistency(G, normals).item()

    # Level set extraction
    forward = partial(mlp_forward, params, activation=static.activation, skip_layers=static.skip_layers)
    max_pts = 1.5  # TODO: set as arguments
    resolution = 40
    coord = jnp.linspace(-max_pts, max_pts, resolution)
    grid = jnp.stack(jnp.meshgrid(coord, coord, coord), axis=-1).reshape(-1, 3)
    out = vmap(forward)(grid)
    values = np.array(out[:, 0]).reshape(resolution, resolution, resolution)
    level_set_value = max(0.0, values.min())  # TODO: use this or a try except
    level_set_points, _, _, _ = marching_cubes(values.reshape(resolution, resolution, resolution), level_set_value)

    # Calculate the distance metrics and return
    numpy_points = np.array(points)
    return {
        "normal_consistency": nc,
        "chamfer": chamfer(level_set_points, numpy_points),
        "directed_chamfer": directed_chamfer(level_set_points, numpy_points),
        "hausdorff": hausdorff(level_set_points, numpy_points),
        "directed_hausdorff": directed_hausdorff(level_set_points, numpy_points),
    }


if __name__ == "__main__":
    from pathlib import Path

    import jax.numpy as jnp
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
    out = evaluate(points, normals, params, static)
    print(out)

from functools import partial, wraps
from typing import Callable

import numpy as np
from jax import Array, vmap

from pinc.model import StaticLossArgs


def input_to_cpu(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = [np.array(arg) if isinstance(arg, Array) else arg for arg in args]
        kwargs = {k: np.array(v) if isinstance(v, Array) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return wrapper


def normal_consistency(x: Array, y: Array) -> Array:
    """Normal consistency: NC(G, n)= 1 / N sum_{i=1}^N |G(x_i)^T n_i|."""
    return jnp.mean(jnp.abs(jnp.sum(x * y, axis=1)))


def evaluate(points: Array, normals: Array, params: list[tuple[Array, Array]], static: StaticLossArgs) -> Array:
    """Evaluate the SDF at the given points."""
    forward = partial(get_variables, params, activation=static.activation, F=static.F, skip_layers=static.skip_layers)
    (_, _, G, _, _) = vmap(forward)(points)
    nc = normal_consistency(G, normals)
    return nc


if __name__ == "__main__":
    from pathlib import Path

    import jax.numpy as jnp
    from jax import nn, random

    from pinc.data import load_ply, process_points
    from pinc.model import get_variables, init_mlp_params

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

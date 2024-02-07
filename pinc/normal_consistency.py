import jax.numpy as jnp
from jax import Array, vmap

from pinc.model import Params, StaticLossArgs, compute_variables


def normal_consistency(x: Array, y: Array) -> Array:
    """Normal consistency: NC(G, n)= 1 / N sum_{i=1}^N |G(x_i)^T n_i|."""
    return jnp.mean(jnp.abs(jnp.sum(x * y, axis=1)))


def compute_normal_consistency(points: Array, normals: Array, params: Params, static: StaticLossArgs) -> Array:
    """Computes the normal consistency of a point cloud."""

    def compute_G(x: Array) -> Array:
        return compute_variables(params, x, activation=static.activation, F=static.F, skip_layers=static.skip_layers)[2]

    G = vmap(compute_G)(points)
    return normal_consistency(G, normals)


if __name__ == "__main__":
    from jax import nn, random

    from pinc.data import load_data
    from pinc.model import init_mlp_params

    skip_layers = [1]
    params = init_mlp_params([3, 128, 128, 7], random.key(0), skip_layers)

    points, normals, _, _, _ = load_data("gargoyle")
    static = StaticLossArgs(activation=nn.relu, F=lambda x: x / 3, skip_layers=skip_layers, loss_weights=None, epsilon=None)  # type: ignore
    nc = compute_normal_consistency(points, normals, params, static)
    print({"normal_consistency": nc.item()})

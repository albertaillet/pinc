from math import sqrt, pi
from jax import numpy as np, nn
from jax.random import normal, split, PRNGKey

# typing
from jax import Array
from typing import Callable
from functools import partial


Params = list[tuple[Array, Array]]


def init_layer_params(layer: int, in_dim: int, out_dim: int, key: Array, n_layers: int) -> tuple[Array, Array]:
    def create_params(w_mean: float, w_std: float, b_const: float) -> tuple[Array, Array]:
        return w_mean + w_std * normal(key, (out_dim, in_dim)), b_const * np.ones(out_dim)

    # Geometric initialization according to "SAL: Sign Agnostic Learning of Shapes From Raw Data"
    # https://github.com/matanatz/SAL/blob/master/code/model/network.py#L112
    # PINC implementation:
    # https://github.com/Yebbi/PINC/blob/main/model/network.py#L46
    if layer == n_layers - 2:
        # Inconsitency between SAL and PINC (factor of 2) and p in denominator
        # in SAL: 2*sqrt(pi) / sqrt(in_dim*p)
        # in PINC:  sqrt(pi) / sqrt(in_dim)
        return create_params(w_mean=sqrt(pi) / sqrt(in_dim), w_std=0.00001, b_const=-0.1)
    else:
        return create_params(w_mean=0.0, w_std=sqrt(2) / sqrt(in_dim), b_const=0.0)


def init_mlp_params(dims: list[int], key: Array, skip_layers: list[int]) -> Params:
    input_dim, n_layers = dims[0], len(dims)
    assert all(0 <= layer < n_layers for layer in skip_layers)
    in_dims = dims[:-1]
    out_dims = [dim - input_dim if layer + 1 in skip_layers else dim for layer, dim in enumerate(dims[1:])]
    keys = split(key, n_layers)
    return [
        init_layer_params(layer, in_dim, out_dim, key, n_layers)
        for layer, (in_dim, out_dim, key) in enumerate(zip(in_dims, out_dims, keys))
    ]


def mlp_forward(params: Params, x: Array, activation: Callable, skip_layers: list[int]) -> Array:
    _in = x
    for layer, (w, b) in enumerate(params[:-1]):
        if layer in skip_layers:
            x = np.concatenate([x, _in]) / np.sqrt(2)  # sqrt(2) seems to not be explained in the paper
        x = activation(w @ x + b)
    w, b = params[-1]
    return w @ x + b


if __name__ == "__main__":
    layer_sizes = [3] + [512] * 7 + [7]
    skip_layers = [4]
    params = init_mlp_params(layer_sizes, key=PRNGKey(0), skip_layers=skip_layers)
    x = np.ones(3)
    out = mlp_forward(params, x, activation=nn.relu, skip_layers=skip_layers)
    print(out.shape)

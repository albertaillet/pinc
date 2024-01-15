from jax import numpy as np, nn
from jax.random import normal, split, PRNGKey

# typing
from jax import Array
from typing import Callable
from functools import partial


Params = list[tuple[Array, Array]]


def init_layer_params(layer_index: int, in_dim: int, out_dim: int, key: Array, n_layers: int) -> tuple[Array, Array]:
    w_shape, b_shape = (out_dim, in_dim), (out_dim,)

    # Geometric initialization according to "SAL: Sign Agnostic Learning of Shapes From Raw Data"
    # https://github.com/matanatz/SAL/blob/master/code/model/network.py#L112
    # PINC implementation:
    # https://github.com/Yebbi/PINC/blob/main/model/network.py#L46
    if layer_index == n_layers - 2:
        # Inconsitency between SAL and PINC (factor of 2)
        # in SAL: 2*np.sqrt(np.pi) / np.sqrt(p * dims[l])
        # in PINC: np.sqrt(np.pi) / np.sqrt(dims[layer])
        w_mean = np.sqrt(np.pi) / np.sqrt(in_dim)
        w_std = 0.00001
        b_const = -0.1
    else:
        w_mean = 0.0
        w_std = np.sqrt(2) / np.sqrt(in_dim)
        b_const = 0.0

    return w_mean + w_std * normal(key, w_shape), b_const * np.ones(b_shape)


def init_mlp_params(sizes: list[int], key: Array) -> Params:
    n_layers = len(sizes)
    keys = split(key, n_layers)
    init = partial(init_layer_params, n_layers=n_layers)
    return [init(i, in_dim, out_dim, key) for i, in_dim, out_dim, key in zip(range(n_layers), sizes[:-1], sizes[1:], keys)]


def mlp_forward(params: Params, x: Array, activation: Callable) -> Array:
    for w, b in params[:-1]:
        x = activation(w @ x + b)
    w, b = params[-1]
    return w @ x + b


if __name__ == "__main__":
    layer_sizes = [784, 512, 512, 10]
    params = init_mlp_params(layer_sizes, key=PRNGKey(0))
    x = np.ones(784)
    out = mlp_forward(params, x, activation=nn.relu)
    print(out.shape)

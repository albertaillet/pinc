from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array, jacfwd, nn
from jax.random import key, normal, split
from numpy import load, savez_compressed

Params = list[tuple[Array, Array]]


def beta_softplus(x: Array, beta: float) -> Array:
    return nn.softplus(x * beta) / beta


def init_layer_params(in_dim: int, out_dim: int, key: Array, last_layer: bool) -> tuple[Array, Array]:
    def create_params(w_mean: Array, w_std: Array, b_const: float) -> tuple[Array, Array]:
        return w_mean + w_std * normal(key, (out_dim, in_dim)), b_const * jnp.ones(out_dim)

    # Geometric initialization according to "SAL: Sign Agnostic Learning of Shapes From Raw Data"
    # https://github.com/matanatz/SAL/blob/master/code/model/network.py#L112
    # PINC implementation:
    # https://github.com/Yebbi/PINC/blob/main/model/network.py#L46
    # IGR implementation:
    # https://github.com/amosgropp/IGR/blob/master/code/model/network.py#L48
    if last_layer:
        # Inconsitency between SAL and PINC (factor of 2) and p in denominator
        # in SAL: 2*sqrt(pi) / sqrt(in_dim*p)
        # in PINC:  sqrt(pi) / sqrt(in_dim)
        return create_params(w_mean=jnp.sqrt(jnp.pi / in_dim), w_std=jnp.array(1e-5), b_const=-0.1)
    return create_params(w_mean=jnp.array(0), w_std=jnp.sqrt(2 / in_dim), b_const=0.0)


def init_mlp_params(dims: list[int], key: Array, skip_layers: list[int]) -> Params:
    input_dim, n_layers = dims[0], len(dims) - 1
    assert all(0 <= i < n_layers for i in skip_layers)
    in_dims = dims[:-1]
    out_dims = [dim - input_dim if i in skip_layers else dim for i, dim in enumerate(dims[1:], 1)]
    keys = split(key, n_layers)
    assert len(in_dims) == len(out_dims) == len(keys)
    return [
        init_layer_params(in_dim, out_dim, key, i == n_layers)
        for i, (in_dim, out_dim, key) in enumerate(zip(in_dims, out_dims, keys), 1)
    ]


def mlp_forward(params: Params, x: Array, activation: Callable[[Array], Array], skip_layers: list[int]) -> Array:
    _in = x
    for i, (w, b) in enumerate(params[:-1]):
        if i in skip_layers:
            x = jnp.concatenate([x, _in]) / jnp.sqrt(2)  # sqrt(2) seems to not be explained in the paper
        x = activation(w @ x + b)
    w, b = params[-1]
    return w @ x + b


def curl_from_jacobian(jacobian: Array) -> Array:
    return jnp.array([
        jacobian[2, 1] - jacobian[1, 2],
        jacobian[0, 2] - jacobian[2, 0],
        jacobian[1, 0] - jacobian[0, 1],
    ])


def get_variables(params: Params, x: Array, activation: Callable, F: Callable, skip_layers: list[int]) -> tuple[Array, ...]:
    def forward_and_aux(x: Array):
        out = mlp_forward(params, x, activation, skip_layers)
        sdf, phi, phi_tilde = out[0], out[1:4], out[4:7]
        G_tilde = phi_tilde / jnp.maximum(1, jnp.linalg.norm(phi_tilde))
        return (sdf, phi, G_tilde), (sdf, G_tilde)

    # shapes: grad_sdf(3,)  jac_phi(3, 3)  jac_G_tilde(3, 3)
    (grad_sdf, jac_phi, jac_G_tilde), (sdf, G_tilde) = jacfwd(forward_and_aux, has_aux=True)(x)

    curl_phi = curl_from_jacobian(jac_phi)
    curl_phi_minus_F = curl_phi - F(x)
    G = curl_phi_minus_F / (jnp.linalg.norm(curl_phi_minus_F) + 1e-6)  # when p approaches inf

    curl_G_tilde = curl_from_jacobian(jac_G_tilde)

    return sdf, grad_sdf, G, G_tilde, curl_G_tilde


def delta_e(x: Array, epsilon: float) -> Array:
    return 1 - jnp.tanh(x / epsilon) ** 2


class StaticLossArgs(NamedTuple):
    activation: Callable[[Array], Array]
    F: Callable[[Array], Array]
    skip_layers: list[int]
    loss_weights: Array
    epsilon: float


def compute_loss(
    params: Params,
    x: Array,
    boundary: bool,
    static: StaticLossArgs,
) -> Array:
    activation, F, skip_layers, loss_weights, epsilon = static
    sdf, grad_sdf, G, G_tilde, curl_G_tilde = get_variables(params, x, activation, F, skip_layers)
    loss = jnp.array([
        jnp.abs(sdf) * boundary,  # loss function for sdf
        jnp.square(grad_sdf - G).sum(),  # loss function for grad
        jnp.square(G - G_tilde).sum(),  # loss function for G
        jnp.square(curl_G_tilde).sum(),  # loss function for curl
        delta_e(sdf, epsilon) * jnp.linalg.norm(grad_sdf),  # loss function for area
    ])
    return loss @ loss_weights


def save_model(params: Params, path: Path):
    assert path.suffix == ".npz"
    param_dict = {}
    for i, (w, b) in enumerate(params):
        param_dict[f"w_{i}"] = w
        param_dict[f"b_{i}"] = b
    savez_compressed(path, **param_dict)


def load_model(path: Path) -> Params:
    assert path.suffix == ".npz"
    param_dict = load(path, allow_pickle=False)
    n_layers = len(param_dict) // 2
    params = [(param_dict[f"w_{i}"], param_dict[f"b_{i}"]) for i in range(n_layers)]
    return params


if __name__ == "__main__":
    layer_sizes = [3] + [512] * 7 + [7]
    skip_layers = [4]
    params = init_mlp_params(layer_sizes, key=key(0), skip_layers=skip_layers)
    path = Path(".") / "params.npz"
    save_model(params, path)
    params = load_model(path)
    loss_weights = jnp.array([1, 0.1, 1e-4, 1e-4, 0.1])
    x = jnp.arange(3, dtype=jnp.float32)
    out = mlp_forward(params, x, activation=nn.relu, skip_layers=skip_layers)
    assert out.shape == (7,)
    F = lambda x: x / 3
    static = StaticLossArgs(activation=nn.relu, F=F, skip_layers=skip_layers, loss_weights=loss_weights, epsilon=0.1)
    loss = compute_loss(params, x, boundary=True, static=static)
    print(loss)

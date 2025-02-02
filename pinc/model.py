from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array, jacfwd, nn
from jax.random import key, normal, split
from numpy import savez_compressed

Params = list[tuple[Array, Array]]


def beta_softplus(x: Array, beta: float) -> Array:
    """Compute the softplus activation function with a beta parameter."""
    return nn.softplus(x * beta) / beta


def init_layer_params(in_dim: int, out_dim: int, key: Array, last_layer: bool) -> tuple[Array, Array]:
    """Initialize the parameters of a single layer of a multi-layer perceptron using geometric initialization."""

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
    """Initialize the parameters of a multi-layer perceptron."""
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
    """Compute the output of the multi-layer perceptron on one data point."""
    _in = x
    for i, (w, b) in enumerate(params[:-1]):
        if i in skip_layers:
            x = jnp.concatenate([x, _in]) / jnp.sqrt(2)  # sqrt(2) seems to not be explained in the paper
        x = activation(w @ x + b)
    w, b = params[-1]
    return w @ x + b


def curl_from_jacobian(jacobian: Array) -> Array:
    """Compute the curl of a vector field from its Jacobian."""
    return jnp.array([
        jacobian[2, 1] - jacobian[1, 2],
        jacobian[0, 2] - jacobian[2, 0],
        jacobian[1, 0] - jacobian[0, 1],
    ])


def compute_variables(params: Params, x: Array, activation: Callable, F: Callable, skip_layers: list[int]) -> tuple[Array, ...]:
    """Compute the sdf and auxiliary variables of the PINC model."""

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
    static: StaticLossArgs,
) -> tuple[Array, Array]:
    """Compute the loss function on one data point."""
    activation, F, skip_layers, loss_weights, epsilon = static
    sdf, grad_sdf, G, G_tilde, curl_G_tilde = compute_variables(params, x, activation, F, skip_layers)
    loss_sdf = jnp.abs(sdf)  # loss function for sdf (should only be computed on the boundary)
    loss_terms = jnp.array([
        jnp.square(grad_sdf - G).sum(),  # loss function for grad
        jnp.square(G - G_tilde).sum(),  # loss function for G
        jnp.square(curl_G_tilde).sum(),  # loss function for curl
        delta_e(sdf, epsilon) * jnp.linalg.norm(grad_sdf),  # loss function for area
    ])
    return loss_sdf, loss_terms * loss_weights


def save_model(params: Params, path: Path) -> None:
    """Save model parameters to a compressed .npz file."""
    savez_compressed(path, *[jnp.concatenate([w, b[:, None]], axis=1) for w, b in params])


def load_model(path: Path) -> Params:
    """Load model parameters from a .npz file."""
    return [(w_b[:, :-1], w_b[:, -1]) for w_b in jnp.load(path, allow_pickle=False).values()]  # type: ignore


if __name__ == "__main__":
    layer_sizes = [3] + [512] * 7 + [7]
    skip_layers = [4]
    params = init_mlp_params(layer_sizes, key=key(0), skip_layers=skip_layers)
    loss_weights = jnp.array([0.1, 1e-4, 1e-4, 0.1])
    x = jnp.array([0.2, 0.1, 0.3])
    out = mlp_forward(params, x, activation=nn.relu, skip_layers=skip_layers)
    print(out)
    assert out.shape == (7,)
    F = lambda x: x / 3
    static = StaticLossArgs(activation=nn.relu, F=F, skip_layers=skip_layers, loss_weights=loss_weights, epsilon=0.1)
    loss = compute_loss(params, x, static=static)
    print("loss: ", loss)
    path = Path("test.npz")
    save_model(params, path)
    params = load_model(path)
    out_loaded = mlp_forward(params, x, activation=nn.relu, skip_layers=skip_layers)
    print(out_loaded)
    assert jnp.allclose(out, out_loaded)

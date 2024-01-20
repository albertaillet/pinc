import optax
import jax.numpy as np
from jax.random import key, split, choice, normal
from jax.lax import scan
from jax import value_and_grad, Array, vmap
from pinc.model import beta_softplus, init_mlp_params, compute_loss, Params
from functools import partial
from typing import Callable


def step(
    params: Params,
    boundary_points: Array,
    sample_points,
    opt_state: optax.OptState,
    optim: optax.GradientTransformation,
    activation: Callable,
    F: Callable,
    skip_layers: list[int],
    loss_weights: Array,
) -> tuple[Params, Array]:
    """Compute loss and update parameters"""
    compute_loss_applied = partial(
        compute_loss, params=params, activation=activation, F=F, skip_layers=skip_layers, loss_weights=loss_weights
    )

    def batch_loss(params: Params, boundary_points: Array, sample_points: Array) -> Array:
        boundary_loss = vmap(partial(compute_loss_applied, params=params, boundary=True))
        sample_loss = vmap(partial(compute_loss_applied, params=params, boundary=False))
        return boundary_loss(x=boundary_points).sum() + sample_loss(x=sample_points).sum() / (
            len(boundary_points) + len(sample_points)
        )

    loss, grad = value_and_grad(batch_loss)(params, boundary_points, sample_points)
    updates, opt_state = optim.update(grad, opt_state)
    params = optax.apply_updates(params, updates)  # type: ignore
    return params, loss


def sample_data(data: Array, data_std: Array, batch_size: int, key: Array) -> tuple[Array, Array]:
    """Get batch of data"""
    random_indices = choice(key, len(data), shape=(batch_size,), replace=False)
    return data[random_indices], data_std[random_indices]


def sample_global_points(batch_size: int, key: Array) -> Array:
    """Get batch of global points"""
    return normal(key, (batch_size, 3))


def sample_local_points(data_points: Array, std: Array, batch_size: int, key: Array) -> Array:
    """Get batch of local points"""
    return data_points + normal(key, (batch_size, 3)) * std


def get_batch(data: Array, data_std: Array, data_batch_size: int, global_batch_size: int, key: Array) -> tuple[Array, Array]:
    """Get batch of data"""
    data_key, local_key, global_key = split(key, 3)

    boundary_points, std = sample_data(data, data_std, data_batch_size, data_key)
    local_points = sample_local_points(boundary_points, std, data_batch_size, local_key)
    global_points = sample_global_points(global_batch_size, global_key)

    return boundary_points, np.concatenate([local_points, global_points], axis=0)


def train(
    params: Params,
    data: Array,
    data_std: Array,
    optim: optax.GradientTransformation,
    data_batch_size: int,
    global_batch_size: int,
    num_steps: int,
    activation: Callable,
    F: Callable,
    skip_layers: list[int],
    loss_weights: Array,
    key: Array,
) -> tuple[Params, Array]:
    """Train the model"""

    def scan_fn(carry: tuple[Params, optax.OptState], key) -> tuple[tuple[Params, optax.OptState], Array]:
        params, opt_state = carry
        boundary_points, sample_points = get_batch(data, data_std, data_batch_size, global_batch_size, key)
        params, loss = step(
            params=params,
            boundary_points=boundary_points,
            sample_points=sample_points,
            opt_state=opt_state,
            optim=optim,
            activation=activation,
            F=F,
            skip_layers=skip_layers,
            loss_weights=loss_weights,
        )
        return (params, opt_state), loss

    (params, _), loss = scan(scan_fn, (params, optim.init(params)), split(key, num_steps))
    return params, loss


if __name__ == "__main__":
    init_key, data_key, train_key = split(key(0), 3)
    layer_sizes = [3] + [512] * 7 + [7]
    skip_layers = [4]
    num_steps, data_batch_size, global_batch_size = 100, 10, 10
    params = init_mlp_params(layer_sizes, key=init_key, skip_layers=skip_layers)
    loss_weights = np.array([1, 0.1, 1e-4, 5e-4, 0.1])
    optim = optax.adam(optax.piecewise_constant_schedule(1e-3, {2000 * i: 0.99 for i in range(1, num_steps // 2000 + 1)}))
    F = lambda x: x / 3

    data = normal(data_key, (100, 3))
    data = data / np.linalg.norm(data, axis=-1, keepdims=True)
    data_std = np.ones_like(data) * 0.1

    params, loss = train(
        params=params,
        data=data,
        data_std=data_std,
        optim=optim,
        data_batch_size=data_batch_size,
        global_batch_size=global_batch_size,
        num_steps=num_steps,
        activation=partial(beta_softplus, beta=100.0),
        F=F,
        skip_layers=skip_layers,
        loss_weights=loss_weights,
        key=train_key,
    )
    print(loss)

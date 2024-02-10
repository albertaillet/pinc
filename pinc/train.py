from functools import partial
from typing import Callable, Optional

import jax.numpy as jnp
import optax
from jax import Array, lax, value_and_grad, vmap
from jax.random import choice, key, normal, split, uniform

from pinc.experiment_logging import scan_eval_log
from pinc.model import Params, StaticLossArgs, beta_softplus, compute_loss, init_mlp_params

Losses = tuple[Array, tuple[Array, Array, Array]]


def step(
    params: Params,
    boundary_points: Array,
    sample_points: Array,
    opt_state: optax.OptState,
    optim: optax.GradientTransformation,
    static: StaticLossArgs,
) -> tuple[Params, optax.OptState, Losses]:
    """Compute loss and update parameters"""
    compute_loss_with_static = partial(compute_loss, static=static)

    def batch_loss(params: Params, boundary_points: Array, sample_points: Array) -> Losses:
        loss_sdf, boundary_loss_terms = vmap(partial(compute_loss_with_static, params))(boundary_points)
        _, sample_loss_terms = vmap(partial(compute_loss_with_static, params))(sample_points)
        loss_terms_sum = (boundary_loss_terms.sum() + sample_loss_terms.sum()) / (len(boundary_points) + len(sample_points))
        loss_sdf_mean = loss_sdf.mean()
        return loss_sdf_mean + loss_terms_sum, (loss_sdf_mean, boundary_loss_terms.mean(axis=0), sample_loss_terms.mean(axis=0))

    loss, grad = value_and_grad(batch_loss, has_aux=True)(params, boundary_points, sample_points)
    updates, opt_state = optim.update(grad, opt_state)
    params = optax.apply_updates(params, updates)  # type: ignore
    return params, opt_state, loss


def sample_data(data: Array, data_std: Array, batch_size: int, key: Array) -> tuple[Array, Array]:
    """Get batch of data"""
    random_indices = choice(key, len(data), shape=(batch_size,), replace=False)
    return data[random_indices], data_std[random_indices]


def sample_global_points(batch_size: int, key: Array, eta: float) -> Array:
    """Get batch of global points"""
    return uniform(key, (batch_size, 3), minval=-eta, maxval=eta)


def sample_local_points(data_points: Array, std: Array, batch_size: int, key: Array) -> Array:
    """Get batch of local points"""
    return data_points + normal(key, (batch_size, 3)) * std


def get_batch(
    data: Array, data_std: Array, data_batch_size: int, global_batch_size: int, eta: float, key: Array
) -> tuple[Array, Array]:
    """Get batch of data"""
    data_key, local_key, global_key = split(key, 3)

    boundary_points, std = sample_data(data, data_std, data_batch_size, data_key)
    local_points = sample_local_points(boundary_points, std, data_batch_size, local_key)
    global_points = sample_global_points(global_batch_size, global_key, eta)

    return boundary_points, jnp.concatenate([local_points, global_points], axis=0)


def train(
    params: Params,
    data: Array,
    data_std: Array,
    eta: float,
    optim: optax.GradientTransformation,
    data_batch_size: int,
    global_batch_size: int,
    num_steps: int,
    static: StaticLossArgs,
    log_model: Callable,
    log_loss: Callable,
    log_model_freq: Optional[int],
    log_loss_freq: Optional[int],
    key: Array,
) -> tuple[Params, Losses]:
    """Train the model"""

    @scan_eval_log(
        log_model_freq=log_model_freq,
        log_loss_freq=log_loss_freq,
        log_model=log_model,
        log_loss=log_loss,
    )
    def scan_fn(carry: tuple[Params, optax.OptState], it) -> tuple[tuple[Params, optax.OptState], Losses]:
        params, opt_state = carry
        _, key = it

        boundary_points, sample_points = get_batch(data, data_std, data_batch_size, global_batch_size, eta, key)
        params, opt_state, loss = step(
            params=params,
            boundary_points=boundary_points,
            sample_points=sample_points,
            opt_state=opt_state,
            optim=optim,
            static=static,
        )
        return (params, opt_state), loss

    (params, _), loss = lax.scan(scan_fn, (params, optim.init(params)), (jnp.arange(num_steps), split(key, num_steps)))
    return params, loss


if __name__ == "__main__":
    from pinc.data import create_sphere

    init_key, data_key, train_key = split(key(0), 3)
    layer_sizes = [3] + [512] * 7 + [7]
    skip_layers = [4]
    num_steps, data_batch_size, global_batch_size = 100, 10, 10
    params = init_mlp_params(layer_sizes, key=init_key, skip_layers=skip_layers)
    optim = optax.adam(optax.piecewise_constant_schedule(1e-3, {2000 * i: 0.99 for i in range(1, num_steps // 2000 + 1)}))

    # Dummy data
    points = create_sphere(1000, data_key)
    points = points / jnp.linalg.norm(points, axis=-1, keepdims=True)
    points_std = jnp.ones_like(points) * 0.1
    normals = jnp.copy(points)

    loss_weights = jnp.array([0.1, 1e-4, 5e-4, 0.1])
    static = StaticLossArgs(
        activation=partial(beta_softplus, beta=100.0),
        F=lambda x: x / 3,
        skip_layers=skip_layers,
        loss_weights=loss_weights,
        epsilon=0.1,
    )

    def log_model(params: Params, step: int):
        print(f"Log model run at step {step}")

    def log_loss(losses: Losses, step: int):
        loss, (loss_sdf, loss_boundary, loss_sample) = losses
        print(
            f"Step: {step}, Total loss: {loss:.4f}, SDF loss: {loss_sdf:.4f}, "
            f"Boundary loss: {loss_boundary.sum():.4f}, Sample loss: {loss_sample.sum():.4f}"
        )

    params, loss = train(
        params=params,
        data=points,
        data_std=points_std,
        eta=1.1,
        optim=optim,
        data_batch_size=data_batch_size,
        global_batch_size=global_batch_size,
        num_steps=num_steps,
        static=static,
        log_model=log_model,
        log_loss=log_loss,
        log_model_freq=33,
        log_loss_freq=10,
        key=train_key,
    )
    print("Training finished.")

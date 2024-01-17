import jax.numpy as np
from jax.random import KeyArray, PRNGKey, split, choice, normal
from jax.lax import scan
from jax import value_and_grad, Array, nn, vmap
from optax import adam, GradientTransformation, OptState, apply_updates, piecewise_constant_schedule
from mlp import init_mlp_params, compute_loss, Params
from functools import partial


def step(
    params: Params, x: Array, opt_state: OptState, optim: GradientTransformation, skip_layers: list[int], loss_weights: Array
) -> tuple[Params, Array]:
    """Compute loss and update parameters"""

    def batch_loss(params: Params, x: Array) -> Array:
        fun = partial(compute_loss, params=params, activation=nn.relu, F=F, skip_layers=skip_layers, loss_weights=loss_weights)
        return vmap(fun)(x=x).mean()

    loss, grad = value_and_grad(batch_loss)(params, x)
    updates, opt_state = optim.update(grad, opt_state)
    params = apply_updates(params, updates)  # type: ignore
    return params, loss


def sample_data(data: Array, batch_size: int, key: KeyArray) -> Array:
    """Get batch of data"""
    random_indices = choice(key, len(data), shape=(batch_size,), replace=False)
    return data[random_indices]


def sample_points(points: Array, batch_size: int, key: KeyArray) -> Array:
    """Get batch of points"""
    points = normal(key, shape=(batch_size,), dtype=np.int32)
    return points


def train(
    params: Params,
    data: Array,
    optim: GradientTransformation,
    batch_size: int,
    num_steps: int,
    skip_layers: list[int],
    loss_weights: Array,
    key: KeyArray,
) -> tuple[Params, Array]:
    """Train the model"""

    def scan_fn(carry: tuple[Params, OptState], key) -> tuple[tuple[Params, OptState], Array]:
        params, opt_state = carry
        x = sample_data(data, batch_size, key)
        params, loss = step(params, x, opt_state, optim, skip_layers, loss_weights)
        return (params, opt_state), loss

    (params, _), loss = scan(scan_fn, (params, optim.init(params)), split(key, num_steps))
    return params, loss


if __name__ == "__main__":
    init_key, data_key = split(PRNGKey(0))
    layer_sizes = [3] + [512] * 7 + [7]
    skip_layers = [4]
    num_steps, batch_size = 100, 10
    params = init_mlp_params(layer_sizes, key=init_key, skip_layers=skip_layers)
    loss_weights = np.array([1, 0.1, 0.0001, 0.0005, 0.1])
    optim = adam(piecewise_constant_schedule(1e-3, {2000 * i: 0.99 for i in range(1, num_steps // 2000 + 1)}))
    F = lambda x: x / 3
    data = np.arange(100 * 3, dtype=np.float32).reshape(100, 3)
    params, loss = train(
        params, data, optim, batch_size, num_steps, skip_layers=skip_layers, loss_weights=loss_weights, key=data_key
    )

from collections.abc import Callable
from functools import wraps
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax import Array, lax, vmap
from jax.experimental.host_callback import id_tap
from skimage.measure import marching_cubes


def get_grid(grid_range: float, resolution: int) -> Array:
    """Returns a grid of points in a cube of size 2*grid_range centered around [0, 0, 0]."""
    coords = jnp.linspace(-grid_range, grid_range, resolution)
    return jnp.stack(jnp.meshgrid(coords, coords, coords), axis=-1).reshape(-1, 3)


def mesh_from_sdf(sdf: Callable, grid_range: float, resolution: int, level: float) -> tuple[np.ndarray, ...]:
    """Extracts a mesh from an implicit function."""
    grid = get_grid(grid_range, resolution)
    sd_grid = vmap(sdf)(grid)
    sd_grid_numpy = np.array(sd_grid).reshape(resolution, resolution, resolution)
    try:
        verts, faces, _normals, _values = marching_cubes(sd_grid_numpy, level=level)  # type: ignore
        verts = verts / resolution * 2 * grid_range - grid_range
    except ValueError:  # no level set crossing
        verts, faces = np.zeros((1, 3)), np.zeros((1, 3))
    return verts, faces


def scan_eval_log(
    eval_freq: Optional[int],
    loss_freq: Optional[int],
    log_eval: Callable,
    log_loss: Callable,
) -> Callable:
    """Decorator that starts eval logging to `body_fun` used in `jax.lax.scan`."""

    def _scan_eval_log(func: Callable) -> Callable:
        @wraps(func)
        def wrapped_log(carry: tuple, x: tuple) -> tuple:
            iter_num, *_ = x
            params, *_ = carry

            lax.cond(
                eval_freq is not None and iter_num % eval_freq == 0,
                lambda params, iter_num: id_tap(log_eval, (params, iter_num)),
                lambda *args: args,
                params,
                iter_num,
            )
            out_carry, loss = func(carry, x)

            lax.cond(
                loss_freq is not None and iter_num % loss_freq == 0,
                lambda loss, iter_num: id_tap(log_loss, (loss, iter_num)),
                lambda *args: args,
                loss,
                iter_num,
            )

            return out_carry, loss

        return wrapped_log

    return _scan_eval_log


if __name__ == "__main__":
    import jax.numpy as jnp

    import wandb

    wandb.init(project="test", mode="offline")

    @scan_eval_log(
        eval_freq=None,
        loss_freq=10,
        log_eval=lambda x, _: wandb.log({"eval": x[0]}, step=x[1]),
        log_loss=lambda x, _: wandb.log({"loss": x[0]}, step=x[1]),
    )
    def scan_step(carry, x):
        return (carry[0] + 1,), x[0]

    lax.scan(scan_step, (0,), (jnp.arange(100), jnp.arange(100)))

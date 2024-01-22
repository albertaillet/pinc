from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax import Array, vmap
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
        return marching_cubes(sd_grid_numpy, level)
    except ValueError:  # no zero crossing
        return np.zeros((1, 3)), np.zeros((1, 3))

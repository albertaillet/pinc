from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jax import Array, lax, vmap
from skimage.measure import marching_cubes


def get_grid(grid_range: float, resolution: int) -> Array:
    """Returns a grid of points in a cube of size 2*grid_range centered around [0, 0, 0]."""
    coords = jnp.linspace(-grid_range, grid_range, resolution)
    return jnp.stack(jnp.meshgrid(coords, coords, coords), axis=-1).reshape(-1, 3)


def mesh_from_sdf(sdf: Callable, grid_range: float, resolution: int, level: float) -> tuple[np.ndarray, ...]:
    """Extracts a mesh from an implicit function."""
    grid = get_grid(grid_range, resolution)
    grid = grid.reshape(-1, resolution * resolution, 3)
    sd_grid = lax.map(vmap(sdf), grid)
    sd_grid_numpy = np.array(sd_grid).reshape(resolution, resolution, resolution)
    try:
        verts, faces, _normals, _values = marching_cubes(sd_grid_numpy, level=level)  # type: ignore
        verts = verts / resolution * 2 * grid_range - grid_range
    except ValueError:  # no level set crossing
        verts, faces = np.zeros((1, 3)), np.zeros((1, 3))
    return verts, faces

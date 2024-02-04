from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jax import Array, debug, vmap
from skimage.measure import marching_cubes


def get_grid(grid_range: float, resolution: int) -> Array:
    """Returns a grid of points in a cube of size 2*grid_range centered around [0, 0, 0]."""
    coords = jnp.linspace(-grid_range, grid_range, resolution)
    return jnp.stack(jnp.meshgrid(coords, coords, coords), axis=-1).reshape(-1, 3)


def sdf_grid_from_sdf(sdf: Callable, grid_range: float, resolution: int) -> Array:
    """Returns a grid of signed distances."""
    grid = get_grid(grid_range, resolution)
    return vmap(sdf)(grid).reshape(resolution, resolution, resolution)


def mesh_from_sdf_grid(sdf_grid: np.ndarray, grid_range: float, resolution: int, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Extracts a mesh from an implicit function."""
    try:
        verts, faces, _normals, _values = marching_cubes(sdf_grid, level=level)  # type: ignore
        verts = verts / resolution * 2 * grid_range - grid_range
    except ValueError as e:
        debug.print("exception: {e}", e=e)
        verts, faces = np.zeros((1, 3)), np.zeros((1, 3))
    return verts, faces

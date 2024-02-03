from matplotlib import pyplot as plt
from plotly import graph_objects as go


def figure(trace, title) -> go.Figure:
    return go.Figure(
        data=[trace],
        layout=go.Layout(
            title=title,
            scene=dict(
                xaxis=dict(title="X", range=[-2, 2], autorange=False),
                yaxis=dict(title="Y", range=[-2, 2], autorange=False),
                zaxis=dict(title="Z", range=[-2, 2], autorange=False),
                aspectratio=dict(x=1, y=1, z=1),
            ),
        ),
    )


def plot_points(points, title="") -> go.Figure:
    return figure(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=1, opacity=1, colorscale="Viridis", color=points[:, 2]),
        ),
        title,
    )


def plot_mesh(points, triangles, title="") -> go.Figure:
    return figure(
        go.Mesh3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            opacity=0.5,
        ),
        title,
    )


if __name__ == "__main__":
    from functools import partial

    import jax.numpy as jnp
    from jax import vmap

    from pinc.utils import Array, mesh_from_sdf  # noqa: F401

    def sdf_sphere(point: Array, radius: float) -> Array:
        return jnp.linalg.norm(point) - radius

    def sdf_torus(point: Array, radius: float, tube_radius: float) -> float:
        q = jnp.stack([jnp.linalg.norm(point[:2]) - radius, point[2]])
        return jnp.linalg.norm(q) - tube_radius

    sdf = partial(sdf_torus, radius=0.7, tube_radius=0.3)
    # verts, faces = mesh_from_sdf(sdf, grid_range=1.5, resolution=40, level=0.0)
    # plot_points(verts, title="Torus").show()
    # plot_mesh(verts, faces, title="Torus").show()

    def show_slice(sdf, z=0.0, w=200, r=1.1):
        y, x = jnp.mgrid[-r : r : w * 1j, -r : r : w * 1j].reshape(2, -1)
        z = z * jnp.ones_like(x)
        p = jnp.c_[x, y, z]
        d = vmap(sdf)(p).reshape(w, w)
        plt.figure(figsize=(5, 5))
        kw = dict(extent=(-r, r, -r, r), vmin=-r, vmax=r)
        plt.contourf(d, 16, cmap="bwr", **kw)
        plt.contour(d, levels=[0.0], colors="black", **kw)
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    show_slice(sdf)

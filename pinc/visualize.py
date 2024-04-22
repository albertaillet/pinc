from plotly import graph_objects as go


def figure(*traces, title: str) -> go.Figure:
    return go.Figure(
        data=traces,
        layout=go.Layout(
            title=title,
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
            ),
        ),
    )


def plot_points(points, **kwargs) -> go.Scatter3d:
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        **kwargs,
    )


def plot_mesh(points, triangles, **kwargs) -> go.Mesh3d:
    return go.Mesh3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        **kwargs,
    )


if __name__ == "__main__":
    from functools import partial

    import jax.numpy as jnp

    from pinc.utils import Array, mesh_from_sdf

    def sdf_sphere(point: Array, radius: float) -> Array:
        return jnp.linalg.norm(point) - radius

    def sdf_torus(point: Array, radius: float, tube_radius: float) -> float:
        q = jnp.stack([jnp.linalg.norm(point[:2]) - radius, point[2]])
        return jnp.linalg.norm(q) - tube_radius

    sdf = partial(sdf_torus, radius=0.2, tube_radius=0.1)
    verts, faces = mesh_from_sdf(sdf, grid_range=1.5, resolution=40, level=0.0)
    figure(plot_points(verts), plot_mesh(verts, faces), title="Torus").show()

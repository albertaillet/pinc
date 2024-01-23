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

    import wandb
    from pinc.utils import Array, mesh_from_sdf

    def sdf_sphere(point: Array, radius: float) -> Array:
        return jnp.linalg.norm(point) - radius

    def sdf_torus(point: Array, radius: float, tube_radius: float) -> float:
        q = jnp.stack([jnp.linalg.norm(point[:2]) - radius, point[2]])
        return jnp.linalg.norm(q) - tube_radius

    run = wandb.init(project="test-plotly-viz")

    # Log Table
    for i in range(1, 10):
        print(i)
        sdf = partial(sdf_sphere, radius=0.1 * i)
        verts, faces = mesh_from_sdf(sdf, grid_range=1.5, resolution=40, level=0)
        fig = plot_mesh(verts, faces, title="Sphere")
        wandb.log({"this-should-be-a-figure": fig}, step=i)
    wandb.finish()

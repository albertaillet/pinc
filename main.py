# %%
import numpy as np
import open3d as o3d
from pathlib import Path
from plotly import graph_objects as go
from skimage.measure import marching_cubes


repo_dir = Path(__file__).resolve().parent
data_dir = repo_dir / "data"
scans_dir = data_dir / "scans"
scan = o3d.io.read_point_cloud(str(scans_dir / "gargoyle.ply"))

# %%
points_array = np.asarray(scan.points)
print(points_array.shape)  # (95435, 3)


# %%
def figure(trace, title) -> go.Figure:
    return go.Figure(
        data=[trace],
        layout=go.Layout(
            title=title,
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
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
            marker=dict(size=1, opacity=0.8),
        ),
        title,
    )


plot_points(points_array, title="Gargoyle").show()


# %%
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

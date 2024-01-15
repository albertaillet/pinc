# %%
import numpy as np
import open3d as o3d
from pathlib import Path

repo_dir = Path(__file__).resolve().parent
data_dir = repo_dir / 'data'
scans_dir = data_dir / 'scans'
scan = o3d.io.read_point_cloud(str(scans_dir / 'gargoyle.ply'))

# %%
points_array = np.asarray(scan.points)
print(points_array.shape)  # (95435, 3)

# %%

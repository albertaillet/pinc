import numpy as np
import open3d as o3d


def process_point_cloud(point_cloud: o3d.geometry.PointCloud) -> tuple[np.ndarray, float, np.ndarray]:
    points = np.asarray(point_cloud.points).astype(np.float32)
    center_point = np.mean(points, axis=0)
    points = points - center_point
    max_coord = np.abs(points).max()
    points = points / max_coord
    assert points.shape[1] == 3
    return points, max_coord, center_point


if __name__ == "__main__":
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    ply_file = repo_root / "data/scenes/gargoyle.ply"
    point_cloud = o3d.io.read_point_cloud(str(ply_file))
    points, max_coord, center_point = process_point_cloud(point_cloud)
    print(points.shape, max_coord, center_point)

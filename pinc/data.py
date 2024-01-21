from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load point cloud from .ply file."""
    assert path.exists() and path.suffix == ".ply"
    mesh = trimesh.load(path)
    vertex_metadata = mesh.metadata["_ply_raw"]["vertex"]  # type: ignore
    expected_dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    assert vertex_metadata["data"].dtype == expected_dtype
    # vertex_metadata["data"] has to be converted from an array of tuples
    data = np.asarray(vertex_metadata["data"].tolist())
    assert data.shape == (vertex_metadata["length"], 6)
    return data[:, :3], data[:, 3:]  # points, normals


def process_points(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Center and normalize points."""
    center_point = np.mean(points, axis=0)
    points = points - center_point
    max_coord = np.abs(points).max()
    points = points / max_coord
    return points, max_coord, center_point


def get_sigma(points: np.ndarray, k: int = 50) -> np.ndarray:
    """Caculates the distance to the kth nearest neighbor of each point."""
    tree = cKDTree(points)
    d, _ = tree.query(points, [k + 1])  # k+1 to remove self
    return d  # shape (n, 1)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    for ply_file in repo_root.glob("data/scans/*.ply"):
        points, normals = load_ply(ply_file)
        print(ply_file, points.shape, normals.shape)
        points, normals = load_ply(ply_file)
        print(points.shape, normals.shape)
        points, max_coord, center_point = process_points(points)
        print(points.shape, max_coord, center_point)
        sigma = get_sigma(points)
        print(sigma.shape, type(sigma), sigma.dtype)
        break

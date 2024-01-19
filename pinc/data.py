from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


def load_ply(path: Path) -> np.ndarray:
    """Load point cloud from .ply file."""
    assert path.exists() and path.suffix == ".ply"
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
    return np.stack([plydata["vertex"][coord] for coord in "xyz"], axis=1)


def process_points(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Center and normalize points."""
    center_point = np.mean(points, axis=0)
    points = points - center_point
    max_coord = np.abs(points).max()
    points = points / max_coord
    return points, max_coord, center_point


def get_sigma(points: np.ndarray, k: int = 50):
    """Caculates the distance to the kth nearest neighbor of each point."""
    tree = cKDTree(points)
    d, _ = tree.query(points, [k + 1])  # k+1 to remove self
    return d  # shape (n, 1)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    ply_file = repo_root / "data/scans/gargoyle.ply"
    points = load_ply(ply_file)
    points, max_coord, center_point = process_points(points)
    print(points.shape, max_coord, center_point)

    sigma = get_sigma(points)
    print(sigma.shape, type(sigma), sigma.dtype)

import numpy as np
from pathlib import Path
from plyfile import PlyData


def process_points(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    assert points.ndim == 2 and points.shape[1] == 3
    center_point = np.mean(points, axis=0)
    points = points - center_point
    max_coord = np.abs(points).max()
    points = points / max_coord
    return points, max_coord, center_point


def load_ply(path: Path) -> np.ndarray:
    assert path.exists() and path.suffix == ".ply"
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
    return np.stack([plydata["vertex"][coord] for coord in "xyz"], axis=1)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    ply_file = repo_root / "data/scans/gargoyle.ply"
    points = load_ply(ply_file)
    points, max_coord, center_point = process_points(points)
    print(points.shape, max_coord, center_point)

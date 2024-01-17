import numpy as np
from pathlib import Path
from plyfile import PlyData


def process_points(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    center_point = np.mean(points, axis=0)
    points = points - center_point
    max_coord = np.abs(points).max()
    points = points / max_coord
    assert points.shape[1] == 3
    return points, max_coord, center_point


def load_ply(path: Path) -> np.ndarray:
    assert path.suffix == ".ply"
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
    points = np.stack([plydata["vertex"][coord] for coord in "xyz"], axis=1)
    return points


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    ply_file = repo_root / "data/scans/gargoyle.ply"
    point_cloud = load_ply(ply_file)

    points, max_coord, center_point = process_points(point_cloud)
    print(points.shape, max_coord, center_point)

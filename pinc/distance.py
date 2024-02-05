import time

import numpy as np
from scipy.spatial import cKDTree, distance
from trimesh import PointCloud, Trimesh, sample


# TODO remove all the old functions
def directed_chamfer_old(x: np.ndarray, y: np.ndarray) -> float:
    """Directed Chamfer distance: d_c(X, Y)= 1 / |X| sum_{x in X} min_{y in Y} |x-y|_2."""
    tree = cKDTree(y)
    d, _ = tree.query(x, k=1)
    return np.mean(d)


def chamfer_old(x: np.ndarray, y: np.ndarray) -> float:
    """Chamfer distance: d_C(X, Y)= 0.5 * (d_c(X, Y) + d_c(Y, X))."""
    return 0.5 * (directed_chamfer_old(x, y) + directed_chamfer_old(y, x))


def directed_hausdorff_old(x: np.ndarray, y: np.ndarray) -> float:
    """Directed Hausdorff distance: d_h(X, Y)= max_{x in X} min_{y in Y} |x-y|_2."""
    d, _, _ = distance.directed_hausdorff(x, y)
    return d


def hausdorff_old(x: np.ndarray, y: np.ndarray) -> float:
    """Hausdorff distance: d_H(X, Y)= max(d_h(X, Y), d_h(Y, X))."""
    return np.maximum(directed_hausdorff_old(x, y), directed_hausdorff_old(y, x))


def distances_old(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Computes the Chamfer and Hausdorff distances between two point clouds."""
    # TODO: fix the fact that the distances are calculated multiple times
    return {
        "chamfer": chamfer_old(x, y),
        "directed_chamfer": directed_chamfer_old(x, y),
        "hausdorff": hausdorff_old(x, y),
        "directed_hausdorff": directed_hausdorff_old(x, y),
    }


def distances(x: np.ndarray, y: np.ndarray, *, workers: int = 1) -> dict[str, float]:
    """Computes the Chamfer and Hausdorff distances between two point clouds."""
    # Code adapted from https://github.com/Chumbyte/DiGS/blob/main/surface_reconstruction/compute_metrics_srb.py#L38
    xy_distances, _ = cKDTree(x).query(y, k=1, workers=workers)
    yx_distances, _ = cKDTree(y).query(x, k=1, workers=workers)

    # Directed Chamfer distance: d_c(X, Y)= 1 / |X| sum_{x in X} min_{y in Y} |x-y|_2.
    xy_chamfer = np.mean(xy_distances)
    yx_chamfer = np.mean(yx_distances)

    # Chamfer distance: d_C(X, Y)= 0.5 * (d_c(X, Y) + d_c(Y, X)).
    chamfer_distance = 0.5 * (xy_chamfer + yx_chamfer)

    # Directed Hausdorff distance: d_h(X, Y)= max_{x in X} min_{y in Y} |x-y|_2.
    xy_hausdorff = np.max(xy_distances)
    yx_hausdorff = np.max(yx_distances)

    # Hausdorff distance: d_H(X, Y)= max(d_h(X, Y), d_h(Y, X)).
    hausdorff_distance = np.maximum(xy_hausdorff, yx_hausdorff)

    return {
        "chamfer": chamfer_distance,
        "directed_chamfer": xy_chamfer,
        "directed_chamfer_reversed": yx_chamfer,
        "hausdorff": hausdorff_distance,
        "directed_hausdorff": xy_hausdorff,
        "directed_hausdorff_reversed": yx_hausdorff,
    }


def mesh_distances(
    recon: Trimesh, ground_truth: PointCloud, scan: PointCloud, n_samples: int, seed: int
) -> dict[str, dict[str, float]]:
    """Computes the distance metrics between a the reconstruction and the ground truth and the scan."""
    # NOTE: it is unclear from the paper if the ground truth and scan are sampled or not
    # We suspect the authors used the code from https://github.com/Chumbyte/DiGS/
    # In compute_metrics_srb.py (1) and compute_metrics_shapenet.py (2), only the reconstruction is sampled
    # while in collectMetrics.py (3) and eval_shapespace.py (4), the reconstruction, ground truth and scan are sampled
    # (1) https://github.com/Chumbyte/DiGS/blob/main/surface_reconstruction/compute_metrics_srb.py#L72
    # (2) https://github.com/Chumbyte/DiGS/blob/main/surface_reconstruction/compute_metrics_shapenet.py#L143
    # (3) https://github.com/Chumbyte/DiGS/blob/main/shapespace/collectMetrics.py#L73
    # (4) https://github.com/Chumbyte/DiGS/blob/main/shapespace/eval_shapespace.py#L157
    # The most probable scenario is that the authors used the code from (1), which only samples the reconstruction

    # NOTE: reconstruction vertices must but multiplied by the scale factor and translated by the center
    reconstruction_points, *_ = sample.sample_surface(recon, n_samples, seed=seed)
    # gt_points, *_ = sample.sample_surface(ground_truth, n_samples)
    # scan_points, *_ = sample.sample_surface(scan, n_samples)

    ground_truth_points = ground_truth.vertices
    scan_points = scan.vertices

    # Calculate the distance metrics and return
    t = time.time()
    ground_truth_metrics = distances(reconstruction_points, ground_truth_points)
    scan_metrics = distances(reconstruction_points, scan_points)
    print("Time for new:", time.time() - t)

    t = time.time()
    ground_truth_metrics_old = distances_old(reconstruction_points, ground_truth_points)
    scan_metrics_old = distances_old(reconstruction_points, scan_points)
    print("Time for old:", time.time() - t)

    return {
        "ground_truth": ground_truth_metrics,
        "scan": scan_metrics,
        "ground_truth_old": ground_truth_metrics_old,
        "scan_old": scan_metrics_old,
    }


if __name__ == "__main__":
    from json import dumps
    from pathlib import Path

    from trimesh import load

    from pinc.data import SRB_FILES

    n_in_recon = 100
    n_samples = 10_000
    seed = 0
    random_state = np.random.RandomState(seed)
    vertices = random_state.rand(n_in_recon, 3)
    faces = random_state.randint(0, n_in_recon, size=(n_in_recon, 3))
    recon = Trimesh(vertices=vertices, faces=faces)

    repo_root = Path(__file__).resolve().parent.parent
    for name in SRB_FILES:
        print(name)

        scan = load(repo_root / f"data/scans/{name}.ply")
        assert isinstance(scan, PointCloud)

        gt = load(repo_root / f"data/ground_truth/{name}.xyz")
        assert isinstance(gt, PointCloud)

        print(scan.vertices.shape, gt.vertices.shape)

        d = mesh_distances(recon, gt, scan, n_samples, seed)

        assert d["ground_truth"]["chamfer"] == d["ground_truth_old"]["chamfer"]
        assert d["ground_truth"]["directed_chamfer_reversed"] == d["ground_truth_old"]["directed_chamfer"]
        assert d["ground_truth"]["hausdorff"] == d["ground_truth_old"]["hausdorff"]
        assert d["ground_truth"]["directed_hausdorff_reversed"] == d["ground_truth_old"]["directed_hausdorff"]

        print(dumps(d, indent=2))

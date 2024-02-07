import numpy as np
from scipy.spatial import KDTree
from trimesh import PointCloud, Trimesh, sample


def distances(x: np.ndarray, y: np.ndarray, *, workers: int) -> dict[str, float]:
    """Computes the Chamfer and Hausdorff distances between two point clouds."""
    # Code adapted from https://github.com/Chumbyte/DiGS/blob/main/surface_reconstruction/compute_metrics_srb.py#L38
    xy_distances, _ = KDTree(x).query(y, k=1, workers=workers)
    yx_distances, _ = KDTree(y).query(x, k=1, workers=workers)

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
    recon: Trimesh, ground_truth: PointCloud, scan: PointCloud, *, n_samples: int, seed: int, workers: int
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

    return {
        "ground_truth": distances(reconstruction_points, ground_truth_points, workers=workers),
        "scan": distances(reconstruction_points, scan_points, workers=workers),
    }


if __name__ == "__main__":
    from json import dumps
    from time import time

    from trimesh import load

    from pinc.data import REPO_ROOT, SRB_FILES

    workers = 5
    n_in_recon = 100
    n_samples = 10_000
    seed = 0
    random_state = np.random.RandomState(seed)
    vertices = random_state.rand(n_in_recon, 3)
    faces = random_state.randint(0, n_in_recon, size=(n_in_recon, 3))
    recon = Trimesh(vertices=vertices, faces=faces)
    total_time = 0
    for name in SRB_FILES:
        print(f"Processing {name}...")
        scan = load(REPO_ROOT / f"data/scans/{name}.ply")
        gt = load(REPO_ROOT / f"data/ground_truth/{name}.xyz")
        assert isinstance(gt, PointCloud) and isinstance(scan, PointCloud)
        print(f"{scan.vertices.shape=}, {gt.vertices.shape=}")
        t = time()
        dists = mesh_distances(recon, gt, scan, n_samples=n_samples, seed=seed, workers=workers)
        t = time() - t
        total_time += t
        print(f"Elapsed time: {t:.2f}s")
        print(dumps(dists, indent=2))
    print(f"Total time: {total_time:.2f}s")

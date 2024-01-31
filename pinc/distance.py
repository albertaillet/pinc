import numpy as np
from scipy.spatial import cKDTree, distance
from trimesh import PointCloud, Trimesh, sample


def directed_chamfer(x: np.ndarray, y: np.ndarray) -> float:
    """Directed Chamfer distance: d_c(X, Y)= 1 / |X| sum_{x in X} min_{y in Y} |x-y|_2."""
    tree = cKDTree(y)
    d, _ = tree.query(x, k=1)
    return np.mean(d)


def chamfer(x: np.ndarray, y: np.ndarray) -> float:
    """Chamfer distance: d_C(X, Y)= 0.5 * (d_c(X, Y) + d_c(Y, X))."""
    return 0.5 * (directed_chamfer(x, y) + directed_chamfer(y, x))


def directed_hausdorff(x: np.ndarray, y: np.ndarray) -> float:
    """Directed Hausdorff distance: d_h(X, Y)= max_{x in X} min_{y in Y} |x-y|_2."""
    d, _, _ = distance.directed_hausdorff(x, y)
    return d


def hausdorff(x: np.ndarray, y: np.ndarray) -> float:
    """Hausdorff distance: d_H(X, Y)= max(d_h(X, Y), d_h(Y, X))."""
    return np.maximum(directed_hausdorff(x, y), directed_hausdorff(y, x))


def distances(x: np.ndarray, y: np.ndarray) -> dict[str, np.float32]:
    """Computes the Chamfer and Hausdorff distances between two point clouds."""
    # TODO: fix the fact that the distances are calculated multiple times
    return {
        "chamfer": np.float32(chamfer(x, y)),
        "directed_chamfer": np.float32(directed_chamfer(x, y)),
        "hausdorff": np.float32(hausdorff(x, y)),
        "directed_hausdorff": np.float32(directed_hausdorff(x, y)),
    }


def mesh_distances(recon: Trimesh, gt: PointCloud, scan: PointCloud, n_samples: int) -> dict[str, dict[str, np.float32]]:
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
    reconstruction_points, *_ = sample.sample_surface(recon, n_samples)
    # gt_points, *_ = sample.sample_surface(ground_truth, n_samples)
    # scan_points, *_ = sample.sample_surface(scan, n_samples)

    gt_points = gt.vertices
    scan_points = scan.vertices

    # Calculate the distance metrics and return
    return {
        "reconstruction": distances(gt_points, reconstruction_points),
        "scan": distances(gt_points, scan_points),
    }


if __name__ == "__main__":
    from json import dumps
    from pathlib import Path

    from trimesh import load

    n_in_recon = 10
    n_samples = 10
    random_state = np.random.RandomState(0)
    vertices = random_state.rand(n_in_recon, 3)
    faces = random_state.randint(0, n_in_recon, size=(n_in_recon, 3))
    recon = Trimesh(vertices=vertices, faces=faces)

    repo_root = Path(__file__).resolve().parent.parent
    names = ["anchor", "daratech", "dc", "gargoyle", "lord_quas"]  # SRB dataset
    for name in names:
        print(name)

        scan = load(repo_root / f"data/scans/{name}.ply")
        assert isinstance(scan, PointCloud)

        gt = load(repo_root / f"data/ground_truth/{name}.xyz")
        assert isinstance(gt, PointCloud)

        print(scan.vertices.shape, gt.vertices.shape)

        print(dumps(mesh_distances(recon=recon, gt=gt, scan=scan, n_samples=n_samples), indent=2))

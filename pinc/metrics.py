import numpy as np
from scipy.spatial import cKDTree, distance


def directed_chamfer_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Directed Chamfer distance: d_c(X, Y)= 1 / |X| sum_{x in X} min_{y in Y} |x-y|_2."""
    tree = cKDTree(y)
    d, _ = tree.query(x, k=1)
    return np.mean(d)


def chamfer_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Chamfer distance: d_C(X, Y)= 0.5 * (d_c(X, Y) + d_c(Y, X))."""
    return 0.5 * (directed_chamfer_distance(x, y) + directed_chamfer_distance(y, x))


def directed_hausdorff_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Directed Hausdorff distance: d_h(X, Y)= max_{x in X} min_{y in Y} |x-y|_2."""
    d, _, _ = distance.directed_hausdorff(x, y)
    return d


def hausdorff_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Hausdorff distance: d_H(X, Y)= max(d_h(X, Y), d_h(Y, X))."""
    return np.maximum(directed_hausdorff_distance(x, y), directed_hausdorff_distance(y, x))


def normal_consistency(x: np.ndarray, y: np.ndarray) -> float:
    """Normal consistency: NC(G, n)= 1 / N sum_{i=1}^N |G(x_i)^T n_i|."""
    return np.mean(np.abs(np.sum(x * y, axis=1)))


if __name__ == "__main__":
    random_state = np.random.RandomState(0)
    x = random_state.rand(10_000, 3)
    y = random_state.rand(512, 3)
    print(chamfer_distance(x, y))
    print(hausdorff_distance(x, y))

    x = random_state.rand(10_000, 3)
    y = random_state.rand(10_000, 3)
    print(normal_consistency(x, y))

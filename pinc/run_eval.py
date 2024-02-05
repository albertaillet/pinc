import argparse
import json
import re
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import trimesh
from jax.nn import relu

from pinc.data import REPO_ROOT, SRB_FILES, load_data
from pinc.distance import mesh_distances
from pinc.model import Params, StaticLossArgs, beta_softplus, load_model, mlp_forward
from pinc.normal_consistency import compute_normal_consistency
from pinc.utils import mesh_from_sdf


def get_args() -> argparse.Namespace:
    """Parse command line arguments, if not specified, the default values are the same as in the paper."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--run-path", type=Path, help="Path to the run directory.")
    parser.add_argument("-n", "--n-models", type=int, default=5, help="Number of models to evaluate.")

    args = parser.parse_args()
    path: Path = args.run_path.resolve()
    assert path.is_dir(), f"Run directory {path} does not exist!"
    results = re.search("run-([0-9]{8})_([0-9]{6})-([a-z0-9]{8})", path.name)
    if results is None:
        raise ValueError(f"Invalid run directory name {path.name}!")
    _, _, run_id = results.groups()
    print(f"Loading run {run_id} from {path}...")
    path = path / "files"
    assert (path / "config.json").is_file(), f"Config file {path / 'config.json'} does not exist!"
    with (path / "config.json").open("r") as f:
        config = json.load(f)

    model_save_path = path / "saved_models"
    assert model_save_path.is_dir(), f"Model directory {path / 'saved_models'} does not exist!"
    models = list(reversed(sorted(model_save_path.glob("model_*.npz"))))[: args.n_models]
    assert models, f"No models found in {models}!"
    print(f"Found model {[m.name for m in models]}...")

    args = argparse.Namespace(**config)
    args.run_id = run_id
    args.model_save_path = model_save_path
    args.models = models
    return args


def compute_reconstructed_mesh(params: Params, static: StaticLossArgs, max_coord: float, center_point) -> trimesh.Trimesh:
    def sdf(x: jnp.ndarray) -> jnp.ndarray:
        return mlp_forward(params, x, activation=static.activation, skip_layers=static.skip_layers)[0]

    recon_vertices, recon_faces = mesh_from_sdf(sdf, grid_range=1.5, resolution=256, level=0)  # TODO: resolution 256 in paper
    recon_vertices = recon_vertices * max_coord + center_point
    return trimesh.Trimesh(vertices=recon_vertices, faces=recon_faces)


def main(args: argparse.Namespace):
    print("Evaluating model...")

    n_eval_samples = 100

    points, normals, _data_std, max_coord, center_point = load_data(args.data_filename)

    # softplus if defined for beta > 0 and approachs relu when beta approaches infinty
    # if beta < 0, then we set it to relu
    static = StaticLossArgs(
        activation=partial(beta_softplus, beta=args.beta) if args.beta > 0 else relu,
        F=lambda x: x / 3,
        skip_layers=args.mlp_skip_layers,
        loss_weights=jnp.array(args.loss_weights),
        epsilon=args.epsilon,
    )

    for model_path in args.models:
        model_path: Path
        print(f"Loading model from {model_path}...")
        params = load_model(model_path)
        assert not any(jnp.isnan(w).any() or jnp.isnan(b).any() for w, b in params), "NaNs in parameters!"

        normal_consistency = compute_normal_consistency(points=points, normals=normals, params=params, static=static)

        print(f"Model: {model_path.name}")
        print(f"Normal consistency: {normal_consistency:.4f}")

        recon_mesh = compute_reconstructed_mesh(params, static, max_coord, center_point)

        recon_mesh.export(args.model_save_path / f"{model_path.stem}.ply")

        data_filename: str = args.data_filename

        if data_filename in SRB_FILES:
            ground_truth_mesh = trimesh.load(REPO_ROOT / f"data/ground_truth/{data_filename}.xyz")
            scan_mesh = trimesh.load(REPO_ROOT / f"data/scans/{data_filename}.ply")
            assert isinstance(ground_truth_mesh, trimesh.PointCloud) and isinstance(scan_mesh, trimesh.PointCloud)

            distances = mesh_distances(recon_mesh, ground_truth_mesh, scan_mesh, n_samples=n_eval_samples)

            print(f"Distances: \n {json.dumps(distances, indent=2)}")

    print("Evaluation done!")


if __name__ == "__main__":
    main(get_args())

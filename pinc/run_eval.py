import argparse
import json
import re
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import trimesh
from jax import Array
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
    parser.add_argument("-n", "--n-models", type=int, default=None, help="Number of models to evaluate.")
    parser.add_argument("-r", "--grid-resolution", type=int, default=256, help="Resolution of the grid.")
    parser.add_argument("-nes", "--n-eval-samples", type=int, default=int(10e6), help="Number of samples for evaluation.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("-nw", "--n-workers", type=int, default=5, help="Number of workers for distance computation.")
    return parser.parse_args()


def check_run(run_path: Path, n_models: int) -> tuple[argparse.Namespace, list[Path]]:
    assert run_path.is_dir(), f"Run directory {run_path} does not exist!"
    results = re.search("run-([0-9]{8})_([0-9]{6})-([a-z0-9]{8})", run_path.name)
    if results is None:
        raise ValueError(f"Invalid run directory name {run_path.name}!")
    _, _, run_id = results.groups()
    print(f"Loading run {run_id} from {run_path}...")

    files_path = run_path / "files"
    assert (files_path / "config.json").is_file(), f"Config file {files_path / 'config.json'} does not exist!"
    with (files_path / "config.json").open("r") as f:
        config = json.load(f)
    train_args = argparse.Namespace(**config)

    model_save_path = files_path / "saved_models"
    assert model_save_path.is_dir(), f"Model directory {files_path / 'saved_models'} does not exist!"
    model_paths = list(reversed(sorted(model_save_path.glob("model_*.npz"))))
    if n_models is not None:
        model_paths = model_paths[:n_models]
    assert model_paths, f"No models found in {model_paths}!"
    print(f"Found models {[m.name for m in model_paths]}...")
    return train_args, model_paths


def compute_reconstructed_mesh(
    params: Params, static: StaticLossArgs, max_coord: float, center_point: Array, resolution: int
) -> trimesh.Trimesh:
    def sdf(x: Array) -> Array:
        return mlp_forward(params, x, activation=static.activation, skip_layers=static.skip_layers)[0]

    recon_vertices, recon_faces = mesh_from_sdf(sdf, grid_range=1.5, resolution=resolution, level=0)
    recon_vertices = recon_vertices * max_coord + center_point
    return trimesh.Trimesh(vertices=recon_vertices, faces=recon_faces)


def main(eval_args: argparse.Namespace) -> None:
    print("Evaluating model...")
    run_path: Path = eval_args.run_path.resolve()
    train_args, model_paths = check_run(run_path, eval_args.n_models)

    metrics_path = run_path / "files/metrics"
    metrics_path.mkdir(exist_ok=True)
    reconstructions_path = run_path / "files/reconstructions"
    reconstructions_path.mkdir(exist_ok=True)

    data_filename: str = train_args.data_filename
    points, normals, _data_std, max_coord, center_point = load_data(data_filename)
    static = StaticLossArgs(
        activation=partial(beta_softplus, beta=train_args.beta) if train_args.beta > 0 else relu,
        F=lambda x: x / 3,
        skip_layers=train_args.mlp_skip_layers,
        loss_weights=jnp.array(train_args.loss_weights),
        epsilon=train_args.epsilon,
    )

    if data_filename in SRB_FILES:
        ground_truth_mesh = trimesh.load(REPO_ROOT / f"data/ground_truth/{data_filename}.xyz")
        scan_mesh = trimesh.load(REPO_ROOT / f"data/scans/{data_filename}.ply")
        assert isinstance(ground_truth_mesh, trimesh.PointCloud) and isinstance(scan_mesh, trimesh.PointCloud)

    for model_path in model_paths:
        print(f"Model: {model_path.name}")

        params = load_model(model_path)
        if any(jnp.isnan(w).any() or jnp.isnan(b).any() for w, b in params):
            print("NaNs in parameters!")
            continue

        normal_consistency = compute_normal_consistency(points=points, normals=normals, params=params, static=static).item()

        print(f"Normal consistency: {normal_consistency:.4f}")

        recon_mesh = compute_reconstructed_mesh(params, static, max_coord, center_point, eval_args.grid_resolution)

        recon_mesh.export(reconstructions_path / f"{model_path.stem}.ply")

        if data_filename in SRB_FILES:
            distances = mesh_distances(
                recon=recon_mesh,
                ground_truth=ground_truth_mesh,  # type: ignore
                scan=scan_mesh,  # type: ignore
                n_samples=eval_args.n_eval_samples,
                seed=eval_args.seed,
                workers=eval_args.n_workers,
            )
            print(f"Distances: \n {json.dumps(distances, indent=2)}")
            metrics = {"normal_consistency": normal_consistency, "distances": distances}
        else:
            metrics = {"normal_consistency": normal_consistency}

        with (metrics_path / f"{model_path.stem}_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)

    print("Evaluation done!")


if __name__ == "__main__":
    main(get_args())

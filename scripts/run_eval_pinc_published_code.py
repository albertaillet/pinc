"""Evaluates runs from the published code and saves metrcs to tmp/metrics/"""

import json
from pathlib import Path

import numpy as np
import trimesh

from pinc.data import REPO_ROOT, SRB_FILES
from pinc.distance import mesh_distances

EXPERIMENTS_DIR = REPO_ROOT.parent / "exps" / "single_shape"
OUTPUT_DIR = REPO_ROOT / "tmp" / "metrics"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
N_SAMPLES = 10_000_000
RANDOM_SEED = 0

for experiment_path in EXPERIMENTS_DIR.iterdir():
    mesh_path = experiment_path / "plots" / "igr_100000_single_shape.ply"
    if not mesh_path.exists():
        continue
    recon_mesh: trimesh.Trimesh = trimesh.load(mesh_path)  # type: ignore

    config_path = experiment_path / "config.txt"
    config = config_path.read_text().splitlines()
    assert config[0].startswith("input_path: ")
    file_name = Path(config[0].split("input_path: ")[1]).stem
    assert file_name in SRB_FILES, f"File {file_name} not in SRB_FILES"

    output_file = OUTPUT_DIR / f"{file_name}_{experiment_path.name}.json"

    if output_file.exists():
        print(f"Skipping {file_name} in {experiment_path}")
        continue

    ground_truth: trimesh.PointCloud = trimesh.load(REPO_ROOT / "data" / "ground_truth" / f"{file_name}.xyz")  # type: ignore
    scan: trimesh.PointCloud = trimesh.load(REPO_ROOT / "data" / "scans" / f"{file_name}.ply")  # type: ignore

    center_point = np.mean(scan.vertices, axis=0)
    max_coord = np.abs(scan.vertices - center_point).max()
    rescaled_recon_mesh = recon_mesh.copy()
    rescaled_recon_mesh.vertices = recon_mesh.vertices * max_coord + center_point

    print(f"Computing metrics for {file_name} in {experiment_path}")
    distances = mesh_distances(rescaled_recon_mesh, ground_truth, scan, n_samples=N_SAMPLES, seed=RANDOM_SEED, workers=5)

    print(f"Metrics for {file_name}:")
    print(json.dumps(distances, indent=2))
    with (output_file).open("w") as f:
        json.dump({"distances": distances}, f, indent=2)

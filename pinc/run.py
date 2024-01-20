import optax
import argparse
import jax.numpy as np
from jax.random import key, split
from pathlib import Path

from pinc.train import train
from pinc.model import init_mlp_params
from pinc.data import load_ply, get_sigma


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-filename", type=str, default="gargoyle")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss-weights", type=float, nargs="+", default=[1, 0.1, 1e-4, 5e-4, 0.1])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--n-steps", type=int, default=100)  # set to 100_000 for full training
    parser.add_argument("--data-batch-size", type=int, default=128)  # set to 16384 for full training
    parser.add_argument("--global-batch-size", type=int, default=10)

    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-n-layers", type=int, default=7)
    parser.add_argument("--mlp-skip-layers", type=int, nargs="+", default=[4])

    args = parser.parse_args()
    assert len(args.loss_weights) == 5
    return args


def main(args: argparse.Namespace):
    print("Initializing...")
    repo_root = Path(__file__).resolve().parent.parent

    init_key, train_key = split(key(args.seed), 2)

    layer_sizes = [3] + [args.mlp_hidden_dim] * args.mlp_n_layers + [7]
    skip_layers = args.mlp_skip_layers
    params = init_mlp_params(layer_sizes, key=init_key, skip_layers=skip_layers)

    loss_weights = np.array(args.loss_weights)
    # F = lambda x: x / 3
    optim = optax.adam(optax.piecewise_constant_schedule(args.lr, {2000 * i: 0.99 for i in range(1, args.n_steps // 2000 + 1)}))

    points = load_ply(repo_root / f"data/scans/{args.data_filename}.ply")
    data_std = get_sigma(points)

    print("Starting training...")
    params, loss = train(
        params=params,
        data=np.array(points),
        data_std=np.array(data_std),
        optim=optim,
        data_batch_size=args.data_batch_size,
        global_batch_size=args.global_batch_size,
        num_steps=args.n_steps,
        skip_layers=skip_layers,
        loss_weights=loss_weights,
        key=train_key,
    )
    print(loss)


if __name__ == "__main__":
    main(get_args())

import argparse
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import optax
from jax.nn import relu
from jax.random import key, normal, split

from pinc.data import get_sigma, load_ply, process_points
from pinc.model import StaticLossArgs, beta_softplus, init_mlp_params
from pinc.train import train


def get_args() -> argparse.Namespace:
    """Parse command line arguments, if not specified, the default values are the same as in the paper."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--data-filename", type=str, default="gargoyle")

    parser.add_argument("-lr", type=float, default=1e-3, help="Learning rate for Adam optimizer.")
    parser.add_argument("-lw", "--loss-weights", type=float, nargs="+", default=[1, 0.1, 1e-4, 5e-4, 0.1])
    parser.add_argument("-b", "--beta", type=float, default=100.0, help="Beta parameter for beta softplus activation.")
    parser.add_argument("-e", "--epsilon", type=float, default=0.1, help="Epsilon parameter for delta_e.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed.")

    parser.add_argument("-n", "--n-steps", type=int, default=100, help="Number of training steps.")  # 100_000 in paper
    parser.add_argument("-bs", "--data-batch-size", type=int, default=128, help="Batch size.")  # 16384 in paper

    parser.add_argument("-hd", "--mlp-hidden-dim", type=int, default=512, help="Hidden dimension of MLP.")
    parser.add_argument("-nl", "--mlp-n-layers", type=int, default=7, help="Number of layers in MLP.")
    parser.add_argument("-sl", "--mlp-skip-layers", type=int, nargs="+", default=[4], help="Layers for skip connections.")

    args = parser.parse_args()
    assert len(args.loss_weights) == 5
    assert args.epsilon > 0  # epsilon must be positive
    args.global_batch_size = args.data_batch_size // 8  # global batch size is 1/8 of data batch size
    return args


def main(args: argparse.Namespace):
    print("Initializing...")
    repo_root = Path(__file__).resolve().parent.parent
    if args.data_filename in ["anchor", "daratech", "dc", "gargoyle", "lord_quas"]:
        points, _ = load_ply(repo_root / f"data/scans/{args.data_filename}.ply")
        points, _, _ = process_points(points)
    elif args.data_filename == "sphere":
        points = normal(key(21), (1000, 3))
        points = points / jnp.linalg.norm(points, axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unknown data filename: {args.data_filename}")
    data_std = get_sigma(points)

    init_key, train_key = split(key(args.seed), 2)

    layer_sizes = [3] + [args.mlp_hidden_dim] * args.mlp_n_layers + [7]
    skip_layers = args.mlp_skip_layers
    params = init_mlp_params(layer_sizes, key=init_key, skip_layers=skip_layers)

    optim = optax.adam(optax.piecewise_constant_schedule(args.lr, {2000 * i: 0.99 for i in range(1, args.n_steps // 2000 + 1)}))

    # softplus if defined for beta > 0 and approachs relu when beta approaches infinty
    # if beta < 0, then we set it to relu
    static = StaticLossArgs(
        activation=partial(beta_softplus, beta=args.beta) if args.beta > 0 else relu,
        F=lambda x: x / 3,
        skip_layers=skip_layers,
        loss_weights=jnp.array(args.loss_weights),
        epsilon=args.epsilon,
    )

    print("Starting training...")
    params, loss = train(
        params=params,
        data=jnp.array(points),
        data_std=jnp.array(data_std),
        optim=optim,
        data_batch_size=args.data_batch_size,
        global_batch_size=args.global_batch_size,
        num_steps=args.n_steps,
        static=static,
        key=train_key,
    )
    print(loss)


if __name__ == "__main__":
    main(get_args())

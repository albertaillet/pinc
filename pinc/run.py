import argparse
import datetime
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import optax
from jax.nn import relu
from jax.random import key, split

from pinc.data import load_SRB
from pinc.evaluation import init_wandb, log_loss
from pinc.model import Params, StaticLossArgs, beta_softplus, init_mlp_params, save_model
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

    parser.add_argument("-n", "--n-steps", type=int, default=100_000, help="Number of training steps.")
    parser.add_argument("-bs", "--data-batch-size", type=int, default=16384, help="Batch size.")

    parser.add_argument("-hd", "--mlp-hidden-dim", type=int, default=512, help="Hidden dimension of MLP.")
    parser.add_argument("-nl", "--mlp-n-layers", type=int, default=7, help="Number of layers in MLP.")
    parser.add_argument("-sl", "--mlp-skip-layers", type=int, nargs="+", default=[4], help="Layers for skip connections.")

    parser.add_argument("-ef", "--eval-freq", type=int, default=5000, help="Frequency of evaluation.")
    parser.add_argument("-lf", "--loss-freq", type=int, default=100, help="Frequency of logging loss.")
    parser.add_argument("-nes", "--n-eval-samples", type=int, default=100, help="Number of samples for evaluation.")  # 10^6 paper

    args = parser.parse_args()
    assert len(args.loss_weights) == 5
    assert args.epsilon > 0  # epsilon must be positive
    args.global_batch_size = args.data_batch_size // 8  # global batch size is 1/8 of data batch size in the original code
    return args


def main(args: argparse.Namespace):
    print("Initializing...")

    points, _normals, data_std, _max_coord, _center_point = load_SRB(args.data_filename)
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

    models_path = Path(__file__).resolve().parent.parent / "models"
    experiment_path = models_path / f"{args.data_filename}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment_path.mkdir(parents=True, exist_ok=False)

    def log_save_model(params: Params, step: int):
        assert not any(jnp.isnan(w).any() or jnp.isnan(b).any() for w, b in params), "NaNs in parameters!"
        print(f"Saving model at step {step}...")
        save_model(params, experiment_path / f"model_{step}.npz")
        print(f"Model saved at step {step}.")

    init_wandb(args)

    print("Starting training...")
    params, loss = train(
        params=params,
        data=points,
        data_std=data_std,
        optim=optim,
        data_batch_size=args.data_batch_size,
        global_batch_size=args.global_batch_size,
        num_steps=args.n_steps,
        static=static,
        log_model=log_save_model,
        log_loss=log_loss,
        eval_freq=args.eval_freq,
        loss_freq=args.loss_freq,
        key=train_key,
    )
    print(loss)


if __name__ == "__main__":
    main(get_args())

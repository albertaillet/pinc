from functools import wraps
from pathlib import Path
from typing import Callable, Optional

from jax import lax
from jax.experimental.host_callback import id_tap

import wandb
from pinc.data import REPO_ROOT
from pinc.evaluation import eval_step


def log_loss(loss, step):
    print(f"Loss: {loss[0]:.4f}, step: {step}")

    def losses_to_dict(losses):
        return {
            "loss_sdf": losses[0],
            "loss_grad": losses[1],
            "loss_G": losses[2],
            "loss_curl": losses[3],
            "loss_area": losses[4],
        }

    wandb.log(
        {"loss": loss[0], **{"boundary_loss": losses_to_dict(loss[1][0]), "sample_loss": losses_to_dict(loss[1][0])}}, step=step
    )


def log_eval(params, points, normals, static, max_coord, center_point, data_filename, n_eval_samples, step):
    metrics = eval_step(params, points, normals, static, max_coord, center_point, data_filename, n_eval_samples)
    wandb.log(metrics, step=step)


def init_experiment_logging(args, **kwargs) -> Path:
    print("Initializing experiment logging...")
    run = wandb.init(project="pinc", entity="reproducibility-challenge", config=vars(args), dir=REPO_ROOT, **kwargs)
    print("Experiment logging initialized.")
    return Path(run.dir)  # type: ignore


def scan_eval_log(
    eval_freq: Optional[int],
    loss_freq: Optional[int],
    log_model: Callable,
    log_loss: Callable,
) -> Callable:
    """Decorator that starts eval logging to `body_fun` used in `jax.lax.scan`."""

    def _scan_eval_log(func: Callable) -> Callable:
        @wraps(func)
        def wrapped_log(carry: tuple, x: tuple) -> tuple:
            iter_num, *_ = x
            params, *_ = carry

            lax.cond(
                eval_freq is not None and iter_num % eval_freq == 0,
                lambda params, iter_num: id_tap(log_model, (params, iter_num)),
                lambda *args: args,
                params,
                iter_num,
            )
            out_carry, loss = func(carry, x)

            lax.cond(
                loss_freq is not None and iter_num % loss_freq == 0,
                lambda loss, iter_num: id_tap(log_loss, (loss, iter_num)),
                lambda *args: args,
                loss,
                iter_num,
            )

            return out_carry, loss

        return wrapped_log

    return _scan_eval_log


if __name__ == "__main__":
    import jax.numpy as jnp

    @scan_eval_log(
        eval_freq=25,
        loss_freq=10,
        log_model=lambda x, _: print(f"Log model: {x[0]}, step: {x[1]}"),
        log_loss=lambda x, _: print(f"Log step: {x[0]}, step: {x[1]}"),
    )
    def scan_step(carry, x):
        return (carry[0] + 1,), x[0]

    n_steps = 100
    lax.scan(scan_step, (0,), (jnp.arange(n_steps), jnp.ones(n_steps)))

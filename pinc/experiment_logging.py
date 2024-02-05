import json
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

from jax import Array, lax
from jax.experimental.host_callback import id_tap

import wandb
from pinc.data import REPO_ROOT


def log_loss(losses, step: int) -> None:
    loss, (boundary_losses, sample_losses) = losses

    def loss_terms_dict(loss_terms: Array) -> dict[str, float]:
        loss_sdf, loss_grad, loss_G, loss_curl, loss_area = loss_terms
        return {
            "loss_sdf": float(loss_sdf),
            "loss_grad": float(loss_grad),
            "loss_G": float(loss_G),
            "loss_curl": float(loss_curl),
            "loss_area": float(loss_area),
        }

    loss = float(loss)
    loss_sdf, loss_grad, loss_G, loss_curl, loss_area = boundary_losses + sample_losses

    print(f"Losses: {loss:.4f}, {loss_sdf:.3f}, {loss_grad:.3f}, {loss_G:.3f}, {loss_curl:.3f}, {loss_area:.3f}, step: {step:4d}")
    data = {"loss": loss, "boundary_loss": loss_terms_dict(boundary_losses), "sample_loss": loss_terms_dict(sample_losses)}
    wandb.log(data, step=step)


def init_experiment_logging(args, **kwargs) -> Path:
    print("Initializing experiment logging...")
    config = vars(args)
    run = wandb.init(project="pinc", entity="reproducibility-challenge", config=config, dir=REPO_ROOT, **kwargs)
    experiment_dir = Path(run.dir)  # type: ignore
    assert experiment_dir.exists()
    with (experiment_dir / "config.json").open("w+") as f:
        json.dump(config, f, indent=4)
    print("Experiment logging initialized.")
    return experiment_dir


def scan_eval_log(
    log_model_freq: Optional[int],
    log_loss_freq: Optional[int],
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
                log_model_freq is not None and iter_num % log_model_freq == 0,
                lambda params, iter_num: id_tap(lambda args, _: log_model(*args), (params, iter_num)),
                lambda *args: args,
                params,
                iter_num,
            )
            out_carry, loss = func(carry, x)

            lax.cond(
                log_loss_freq is not None and iter_num % log_loss_freq == 0,
                lambda loss, iter_num: id_tap(lambda args, _: log_loss(*args), (loss, iter_num)),
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
        log_model_freq=25,
        log_loss_freq=10,
        log_model=lambda params, step: print(f"Log model: {params}, step: {step}"),
        log_loss=lambda losses, step: print(f"Log step: {losses}, step: {step}"),
    )
    def scan_step(carry, x):
        return (carry[0] + 7,), x[0]

    n_steps = 100
    lax.scan(scan_step, (0,), (jnp.arange(n_steps), jnp.ones(n_steps)))

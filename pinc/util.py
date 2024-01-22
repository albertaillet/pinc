from jax import lax, debug, experimental
from jax.experimental.host_callback import id_tap
from functools import wraps, partial
import threading

# typing
from collections.abc import Callable
from typing import Optional


def scan_eval_log(
    eval_freq: Optional[int],
    loss_freq: Optional[int],
    log_eval: Callable,
    log_loss: Callable,
) -> Callable:
    """Decorator that starts eval logging to `body_fun` used in `jax.lax.scan`."""

    def _scan_eval_log(func: Callable) -> Callable:
        @wraps(func)
        def wrapped_log(carry: tuple, x: tuple) -> tuple:
            iter_num, *_ = x
            model, *_ = carry

            lax.cond(
                eval_freq is not None and iter_num % eval_freq == 0,
                lambda model, iter_num: id_tap(log_eval, (model, iter_num)),
                lambda *args: args,
                model,
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
    import wandb

    wandb.init(project="test")

    @scan_eval_log(
        eval_freq=10,
        loss_freq=10,
        log_eval=lambda x, _: wandb.log({"eval": x[0]}, step=x[1]),
        log_loss=lambda x, _: wandb.log({"loss": x[0]}, step=x[1]),
    )
    def scan_step(carry, x):
        return (carry[0] + 1,), x[0]

    lax.scan(scan_step, (0,), (jnp.arange(100), jnp.arange(100)))

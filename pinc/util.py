from jax import lax, debug, jit
from functools import wraps

# typing
from collections.abc import Callable


def scan_eval_log(frequency: int, eval_: Callable, log: Callable) -> Callable:
    """Decorator that starts eval logging to `body_fun` used in `jax.lax.scan`."""

    def _scan_eval_log(func: Callable) -> Callable:
        @wraps(func)
        def wrapped_log(carry: tuple, x: tuple) -> tuple:
            iter_num, *_ = x
            model, *_ = carry

            lax.cond(
                iter_num % frequency == 0,
                lambda model: debug.callback(log, eval_(model), ordered=True),
                lambda _: None,
                operand=model,
            )
            return func(carry, x)

        return wrapped_log

    return _scan_eval_log


if __name__ == "__main__":
    import jax.numpy as jnp

    @scan_eval_log(frequency=10, eval_=jit(lambda model: model), log=print)
    def scan_step(carry, x):
        return (carry[0] + 1,), x

    lax.scan(scan_step, (0,), (jnp.arange(100), jnp.arange(100)))

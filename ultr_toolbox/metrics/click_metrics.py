from typing import Callable

import jax.numpy as jnp


def binary_cross_entropy(
    y_predict: jnp.ndarray,
    y_true: jnp.ndarray,
    aggregate: bool = True,
    eps: float = 1e-10,
    log: Callable = jnp.log,
):
    entropy_at_k = -(
        y_true * log(y_predict + eps) + (1 - y_true) * log(1 - y_predict + eps)
    ).mean(axis=0)

    return entropy_at_k.mean() if aggregate else entropy_at_k


def perplexity(
    y_predict: jnp.ndarray,
    y_true: jnp.ndarray,
    aggregate: bool = True,
):
    ppl_at_k = 2 ** binary_cross_entropy(
        y_predict,
        y_true,
        aggregate=False,
        log=jnp.log2,
    )
    return ppl_at_k.mean() if aggregate else ppl_at_k

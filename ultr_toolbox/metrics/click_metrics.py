from typing import Callable, Union

import jax.numpy as jnp


def binary_cross_entropy(
    y_predict: jnp.ndarray,
    y: jnp.ndarray,
    aggregate: bool = True,
    log: Callable = jnp.log,
    eps: float = 1e-10,
) -> Union[jnp.ndarray, float]:
    log_p = log(y_predict + eps)
    log_not_p = log(1 - y_predict + eps)
    cross_entropy = -y * log_p - (1 - y) * log_not_p
    return cross_entropy.mean() if aggregate else cross_entropy


def log_likelihood(
    y_predict: jnp.ndarray,
    y: jnp.ndarray,
) -> Union[jnp.ndarray, float]:
    return -binary_cross_entropy(y_predict, y)


def perplexity(
    y_predict: jnp.ndarray,
    y_true: jnp.ndarray,
    aggregate: bool = True,
) -> Union[jnp.ndarray, float]:
    perplexity_per_rank = 2 ** binary_cross_entropy(
        y_predict,
        y_true,
        aggregate=False,
        log=jnp.log2,
    ).mean(axis=0)

    return perplexity_per_rank.mean() if aggregate else perplexity_per_rank

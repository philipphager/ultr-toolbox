from typing import Callable

import jax.numpy as jnp
from jax import Array


def binary_cross_entropy(
    y_predict: Array,
    y: Array,
    log: Callable = jnp.log,
    eps: float = 1e-10,
) -> float:
    log_p = log(y_predict + eps)
    log_not_p = log(1 - y_predict + eps)
    cross_entropy = -y * log_p - (1 - y) * log_not_p

    return cross_entropy.mean()

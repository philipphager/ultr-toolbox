from typing import Callable, Union

import jax.numpy as jnp
import numpy as np


def binary_cross_entropy(
    y_predict: np.ndarray,
    y: np.ndarray,
    log: Callable = jnp.log,
    eps: float = 1e-10,
) -> Union[np.ndarray, float]:
    log_p = log(y_predict + eps)
    log_not_p = log(1 - y_predict + eps)
    cross_entropy = -y * log_p - (1 - y) * log_not_p

    return cross_entropy.mean()


class Perplexity:
    def __init__(self):
        self.entropy = None
        self.n_sessions = 0

    def update(self, y_predict: np.ndarray, y: np.ndarray, eps: float = 1e-10):
        y_predict = np.atleast_2d(y_predict)
        y = np.atleast_2d(y)

        log_p = np.log2(y_predict + eps)
        log_not_p = np.log2(1 - y_predict + eps)
        entropy = (-y * log_p - (1 - y) * log_not_p).sum(axis=0)

        self.entropy = entropy if self.entropy is None else self.entropy + entropy
        self.n_sessions += len(y)

    def compute(self, aggregate: bool = True) -> Union[float, np.ndarray]:
        perplexity_at_rank = 2 ** (self.entropy / self.n_sessions)
        return perplexity_at_rank.mean() if aggregate else perplexity_at_rank

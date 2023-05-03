from typing import Callable

import numpy as np


def binary_cross_entropy(
    y_predict: np.ndarray,
    y_true: np.ndarray,
    aggregate: bool = True,
    eps: float = 1e-10,
    log: Callable = np.log,
):
    entropy_at_k = -(
        y_true * log(y_predict + eps) + (1 - y_true) * log(1 - y_predict + eps)
    ).mean(axis=0)

    return entropy_at_k.mean() if aggregate else entropy_at_k


def perplexity(
    y_predict: np.ndarray,
    y_true: np.ndarray,
    aggregate: bool = True,
):
    ppl_at_k = 2 ** binary_cross_entropy(
        y_predict,
        y_true,
        aggregate=False,
        log=np.log2,
    )
    return ppl_at_k.mean() if aggregate else ppl_at_k

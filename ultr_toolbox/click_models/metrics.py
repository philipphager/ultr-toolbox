from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from sklearn.metrics import roc_auc_score


class Metric(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def update(self, y_predict: np.ndarray, y: np.ndarray, eps: float = 1e-10):
        pass

    @abstractmethod
    def compute(self) -> Union[float, np.ndarray]:
        pass


class Perplexity(Metric):
    def __init__(self, aggregate_ranks: bool = True):
        name = "perplexity" if aggregate_ranks else "perplexity_at_k"
        super().__init__(name)
        self.aggregate_ranks = aggregate_ranks
        self.entropy = None
        self.n_sessions = 0

    def update(self, y_predict: np.ndarray, y: np.ndarray, eps: float = 1e-10):
        y_predict = np.atleast_2d(y_predict)
        y = np.atleast_2d(y)

        log_p = np.log2(y_predict + eps)
        log_not_p = np.log2(1 - y_predict + eps)
        entropy = (-y * log_p - (1 - y) * log_not_p).sum(axis=0)

        if self.entropy is None:
            self.entropy = entropy
        else:
            self.entropy += entropy

        self.n_sessions += len(y)

    def compute(
        self,
    ) -> Union[float, np.ndarray]:
        perplexity_at_rank = 2 ** (self.entropy / self.n_sessions)
        return perplexity_at_rank.mean() if self.aggregate_ranks else perplexity_at_rank


class LogLikelihood(Metric):
    def __init__(self):
        super().__init__("log_likelihood")
        self.entropy = 0
        self.n_sessions = 0

    def update(self, y_predict: np.ndarray, y: np.ndarray, eps: float = 1e-10):
        y_predict = np.atleast_2d(y_predict)
        y = np.atleast_2d(y)
        n_batch, n_ranks = y.shape

        log_p = np.log(y_predict + eps)
        log_not_p = np.log(1 - y_predict + eps)
        self.entropy += (y * log_p + (1 - y) * log_not_p).sum()
        self.n_sessions += n_batch * n_ranks

    def compute(
        self,
    ) -> Union[float, np.ndarray]:
        return self.entropy / self.n_sessions


class RocAuc(Metric):
    def __init__(self, aggregate_ranks: bool = True):
        name = "roc_auc" if aggregate_ranks else "roc_auc_at_k"
        super().__init__(name)
        self.aggregate_ranks = aggregate_ranks
        self.scores = []

    def update(self, y_predict: np.ndarray, y: np.ndarray, eps: float = 1e-10):
        if self.aggregate_ranks:
            scores = roc_auc_score(y.ravel(), y_predict.ravel())
        else:
            n_ranks = y.shape[1]
            scores = [roc_auc_score(y[:, i], y_predict[:, i]) for i in range(n_ranks)]

        self.scores.append(scores)

    def compute(
        self,
    ) -> Union[float, np.ndarray]:
        scores = np.array(self.scores)
        return np.mean(scores) if self.aggregate_ranks else np.mean(scores, axis=0)

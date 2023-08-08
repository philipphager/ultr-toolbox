from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict

import jax.numpy as jnp
import numpy as np
from jax import jit, Array

from ultr_toolbox.click_models.data import ClickDataset


class StatsModel(ABC):
    @abstractmethod
    def params(self, dataset: ClickDataset) -> Dict:
        pass

    @abstractmethod
    def predict(self, params: Dict, x: Array) -> Array:
        pass


class RandomModel(StatsModel):
    def params(self, dataset: ClickDataset) -> Dict:
        return {"ctr": dataset.y.mean()}

    @staticmethod
    @jit
    def predict(params: Dict, x: Array) -> Array:
        return jnp.full_like(x, fill_value=params["ctr"], dtype=float)


class RankBasedModel(StatsModel):
    def params(self, dataset: ClickDataset) -> Dict:
        return {"ctr_per_rank": jnp.array(dataset.y.mean(axis=0))}

    @staticmethod
    @jit
    def predict(params: Dict, x: Array) -> Array:
        n_batch, _ = x.shape
        return jnp.tile(params["ctr_per_rank"], (n_batch, 1))


class DocumentBasedModel(StatsModel):
    prior_clicks = 1
    prior_impressions = 2

    def params(self, dataset: ClickDataset) -> Dict:
        clicks = np.bincount(dataset.x.ravel(), weights=dataset.y.ravel())
        impressions = np.bincount(dataset.x.ravel())

        clicks += self.prior_clicks
        impressions += self.prior_impressions
        y_predict = clicks / impressions
        prior_ctr = self.prior_clicks / self.prior_impressions

        return {
            "ctr_per_doc": jnp.array(y_predict),
            "prior_ctr": prior_ctr,
        }

    @staticmethod
    def predict(params: Dict, x: Array) -> Array:
        return jnp.take(params["ctr_per_doc"], x, fill_value=params["prior_ctr"])


class RankDocumentBasedModel(StatsModel):
    prior_clicks = 1
    prior_impressions = 2

    def params(self, dataset: ClickDataset) -> Dict:
        n_items = len(np.bincount(dataset.x.ravel()))
        n_ranks = dataset.x.shape[1]
        y_predict = np.zeros((n_items, n_ranks), dtype=float)

        for i in range(n_ranks):
            clicks = np.bincount(
                dataset.x[:, i],
                weights=dataset.y[:, i],
                minlength=n_items,
            )
            impressions = np.bincount(dataset.x[:, i], minlength=n_items)

            clicks = clicks + self.prior_clicks
            impressions = impressions + self.prior_impressions
            y_predict[:, i] = clicks / impressions

        return {"ctr_per_doc_rank": jnp.array(y_predict)}

    @staticmethod
    @jit
    def predict(params: Dict, x: Array) -> Array:
        n_batch, n_ranks = x.shape

        docs = x.ravel()
        ranks = jnp.tile(jnp.arange(n_ranks), (n_batch,))
        y_predict = params["ctr_per_doc_rank"][(docs, ranks)]

        return jnp.reshape(y_predict, x.shape)


class JointModel(StatsModel):
    prior_clicks = 1
    prior_impressions = 2

    def params(self, dataset: ClickDataset) -> Dict:
        n_ranks = dataset.x.shape[1]

        clicks = defaultdict(lambda: np.full(n_ranks, self.prior_clicks, dtype=float))
        impressions = defaultdict(lambda: self.prior_impressions)

        for i, (x, y) in enumerate(zip(dataset.x, dataset.y)):
            key = "_".join(map(str, x))
            clicks[key] += y
            impressions[key] += 1

        return {"clicks": clicks, "impressions": impressions}

    @staticmethod
    def predict(params: Dict, x: Array) -> Array:
        y_predict = np.zeros_like(x, dtype=float)

        for i, x in enumerate(x):
            key = "_".join(map(str, x))
            y_predict[i, :] = params["clicks"][key] / params["impressions"][key]

        return y_predict

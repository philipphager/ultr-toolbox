import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from ultr_toolbox.click_models.neural.base import NeuralModel


class UserBrowsingModel(NeuralModel):
    n_items: int
    n_ranks: int

    def setup(self):
        n_parameters = int(self.n_ranks * (self.n_ranks + 1) / 2)
        self.examination = nn.Sequential([nn.Embed(n_parameters, 1), nn.sigmoid])
        self.relevance = nn.Sequential([nn.Embed(self.n_items, 1), nn.sigmoid])

    def __call__(self, x: Array, y: Array) -> Array:
        ranks = jnp.arange(self.n_ranks)

        # The UBM uses one examination parameter per rank and last clicked position
        # before the current item. Thus, there is:
        # 1 param for rank 1, 2 params for rank 2, 3 params for rank 3, ...
        last_clicked_ranks = self._get_last_click_rank(y)
        examination_idx = ranks.cumsum() + last_clicked_ranks

        examination = self.examination(examination_idx).squeeze()
        relevance = self.relevance(x).squeeze()

        return examination * relevance

    @staticmethod
    def _get_last_click_rank(y: Array) -> Array:
        """
        Returns the last clicked rank for each item, ranks start at k = 1
        """
        n_batch, n_ranks = y.shape

        ranks = jnp.tile(jnp.arange(n_ranks), (n_batch, 1))
        last_clicked_k = jnp.zeros_like(ranks)

        for k in range(n_ranks - 1):
            # Set next rank to current k if clicked, otherwise use previous k
            last_clicked_k = last_clicked_k.at[:, k + 1].set(
                jnp.where(
                    y[:, k] == 1,
                    ranks[:, k] + 1,
                    last_clicked_k[:, k],
                )
            )

        return last_clicked_k

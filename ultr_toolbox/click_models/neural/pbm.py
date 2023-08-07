import jax.numpy as jnp
from flax import linen as nn
from jax import Array


class PositionBasedModel(nn.Module):
    n_items: int
    n_ranks: int

    def setup(self):
        self.examination = nn.Sequential([nn.Embed(self.n_ranks, 1), nn.sigmoid])
        self.relevance = nn.Sequential([nn.Embed(self.n_items, 1), nn.sigmoid])

    def __call__(self, x: Array, *args) -> Array:
        ranks = jnp.arange(self.n_ranks)
        examination = self.examination(ranks).squeeze()
        relevance = self.relevance(x).squeeze()

        return examination * relevance

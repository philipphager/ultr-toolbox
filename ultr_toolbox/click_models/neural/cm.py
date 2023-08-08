import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from ultr_toolbox.click_models.neural.base import NeuralModel


class CascadeModel(NeuralModel):
    n_items: int

    def setup(self) -> None:
        self.relevance = nn.Sequential([nn.Embed(self.n_items, 1), nn.sigmoid])

    def __call__(self, x: Array, *args) -> Array:
        relevance = self.relevance(x).squeeze()
        examination = self._get_examination(relevance)
        return examination * relevance

    def _get_examination(self, relevance: Array) -> Array:
        non_relevance = 1 - relevance
        examination = jnp.roll(non_relevance, shift=1, axis=-1)
        examination = examination.at[:, 0].set(1)
        return jnp.cumprod(examination, axis=-1)

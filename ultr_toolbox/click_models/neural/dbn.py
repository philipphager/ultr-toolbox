import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from ultr_toolbox.click_models.neural.base import NeuralModel


class DynamicBayesianNetwork(NeuralModel):
    n_items: int
    estimate_continuation: bool = True  # Switch between DBN and SBDN

    def setup(self):
        self.attractiveness = nn.Sequential([nn.Embed(self.n_items, 1), nn.sigmoid])
        self.satisfaction = nn.Sequential([nn.Embed(self.n_items, 1), nn.sigmoid])
        self.continuation = nn.Sequential([nn.Embed(1, 1), nn.sigmoid])

    def __call__(self, x: Array, y: Array) -> Array:
        attractiveness = self.attractiveness(x).squeeze()
        satisfaction = self.satisfaction(x).squeeze()
        continuation = (
            self.continuation(jnp.zeros_like(x)).squeeze()
            if self.estimate_continuation
            else jnp.ones_like(x)
        )
        examination = self._get_examination(continuation, satisfaction, y)

        return examination * attractiveness

    def _get_examination(
        self,
        continuation: Array,
        satisfaction: Array,
        clicks: Array,
    ) -> Array:
        examination = continuation * (1 - satisfaction * clicks)
        examination = jnp.roll(examination, shift=1, axis=1)
        examination = examination.at[:, 0].set(1)
        return jnp.cumprod(examination, axis=1)

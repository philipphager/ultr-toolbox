from abc import abstractmethod

from flax import linen as nn
from jax import Array


class NeuralModel(nn.Module):
    @abstractmethod
    def __call__(self, x: Array, y: Array) -> Array:
        pass

from typing import List, Callable, Optional

import jax.numpy as jnp
from flax import linen as nn


class DNN(nn.Module):
    hidden_layers: List[int]
    activation: Optional[Callable] = nn.relu

    @nn.compact
    def __call__(self, x):
        # Normalize data using log1p
        x = jnp.sign(x) * jnp.log1p(jnp.abs(x))

        for hidden_layer in self.hidden_layers:
            x = nn.Dense(hidden_layer)(x)
            x = self.activation(x)

        x = nn.Dense(1)(x)
        x = jnp.squeeze(x, -1)

        return x

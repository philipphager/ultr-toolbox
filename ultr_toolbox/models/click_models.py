from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from ultr_toolbox.metrics.click_metrics import binary_cross_entropy, perplexity


@jax.jit
def train_step(
        state: TrainState,
        batch: Tuple[jnp.DeviceArray, jnp.DeviceArray],
):
    def loss_fn(model, params, batch):
        x, y = batch
        y_predict, _ = model.apply_fn(params, x)
        return binary_cross_entropy(y_predict, y)

    loss, grads = jax.value_and_grad(
        loss_fn,
        argnums=1,  # Position of params in loss_fn
    )(state, state.params, batch)

    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(
        state: TrainState,
        batch: Tuple[jnp.DeviceArray, jnp.DeviceArray],
):
    x, y = batch
    y_predict, _ = state.apply_fn(state.params, x)

    return {
        "perplexity": perplexity(y_predict, y),
        "cross_entropy": binary_cross_entropy(y_predict, y),
    }


@dataclass
class DebugOutput:
    examination: jnp.ndarray
    relevance: jnp.ndarray


class PositionBasedModel(nn.Module):
    n_documents: int
    n_ranks: int

    def setup(self) -> None:
        self.examination = nn.Sequential([nn.Embed(self.n_ranks, 1), nn.sigmoid])
        self.relevance = nn.Sequential([nn.Embed(self.n_documents, 1), nn.sigmoid])

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, DebugOutput]:
        examination = self._get_examination()
        relevance = self._get_relevance(x)
        y_predict = examination * relevance
        debug = DebugOutput(
            examination=examination,
            relevance=relevance,
        )

        return y_predict, debug

    def _get_relevance(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.relevance(x).squeeze()

    def _get_examination(self) -> jnp.ndarray:
        ranks = jnp.arange(self.n_ranks)
        return self.examination(ranks).squeeze()


class CascadeModel(nn.Module):
    n_documents: int

    def setup(self) -> None:
        self.relevance = nn.Sequential([nn.Embed(self.n_documents, 1), nn.sigmoid])

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, DebugOutput]:
        relevance = self._get_relevance(x)
        examination = self._get_examination(relevance)
        y_predict = examination * relevance
        debug = DebugOutput(
            examination=examination,
            relevance=relevance,
        )
        return y_predict, debug

    def _get_relevance(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.relevance(x).squeeze()

    def _get_examination(self, relevance: jnp.ndarray) -> jnp.ndarray:
        non_relevance = 1 - relevance
        examination = jnp.roll(non_relevance, shift=1, axis=-1)
        examination = examination.at[:, 0].set(1)
        return jnp.cumprod(examination, axis=-1)

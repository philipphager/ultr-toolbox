from dataclasses import dataclass
from functools import partial
from typing import Tuple, Callable, Mapping, Dict

import jax
import jax.numpy as jnp
import optax
import pandas as pd
from flax import linen as nn
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from jax import jit, Array
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultr_toolbox.data import ClickDataset, np_collate
from ultr_toolbox.metrics.click_metrics import (
    binary_cross_entropy,
    Perplexity,
)
from ultr_toolbox.models.base import Trainer


class NeuralTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Callable = optax.adam,
        learning_rate: float = 0.001,
        n_epochs: int = 250,
        n_batch: int = 128,
        early_stopping_metric: str = "loss",
        n_patience: int = 3,
        n_workers: int = 4,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_batch = n_batch
        self.early_stopping_metric = early_stopping_metric
        self.n_patience = n_patience
        self.n_epochs = n_epochs
        self.n_workers = n_workers
        self.random_state = random_state
        self.verbose = verbose
        self.model_state = None

    def train(self, train_dataset: ClickDataset, val_dataset: ClickDataset):
        train_loader = self._get_dataloader(train_dataset, shuffle=True)
        val_loader = self._get_dataloader(val_dataset)

        key = jax.random.PRNGKey(self.random_state)
        optimizer = self.optimizer(learning_rate=self.learning_rate)
        early_stopping = EarlyStopping(min_delta=0.0001, patience=self.n_patience)

        x, y = next(iter(train_loader))
        params = self.model.init(key, x)
        model_state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer,
        )

        best_model_state = None

        for epoch in range(self.n_epochs):
            for batch in tqdm(
                train_loader, f"Epoch: {epoch} - Training", disable=not self.verbose
            ):
                model_state, loss = self._train_step(model_state, batch)

            metrics = []

            for batch in tqdm(
                val_loader, f"Epoch: {epoch} - Validation", disable=not self.verbose
            ):
                metrics.append(self._eval_step(model_state, batch))

            val_metric = pd.DataFrame(metrics).mean(axis=0).to_dict()
            has_improved, early_stopping = early_stopping.update(
                val_metric[self.early_stopping_metric]
            )

            if has_improved:
                best_model_state = model_state

            if early_stopping.should_stop:
                break

        self.model_state = best_model_state

    def test(self, dataset: ClickDataset) -> Dict:
        loader = self._get_dataloader(dataset, parallelize=False)
        perplexity = Perplexity()

        for batch in tqdm(loader, "Testing"):
            x, y = batch
            y_predict = self.model_state.apply_fn(self.model_state.params, x)
            perplexity.update(y_predict, y)

        return {"perplexity": perplexity.compute()}

    def _get_dataloader(
        self,
        dataset: ClickDataset,
        shuffle: bool = False,
        parallelize: bool = True,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.n_batch,
            collate_fn=np_collate,
            num_workers=self.n_workers if parallelize else 1,
            persistent_workers=parallelize,
            shuffle=shuffle,
        )

    @partial(jit, static_argnums=(0,))
    def _train_step(
        self,
        state: TrainState,
        batch: Tuple[Array, Array],
    ):
        def loss_fn(state: TrainState, params: Mapping, batch: Tuple[Array, Array]):
            x, y = batch
            y_predict = state.apply_fn(params, x)
            return binary_cross_entropy(y_predict, y)

        loss, grads = jax.value_and_grad(
            loss_fn,
            argnums=1,  # Position of params in loss_fn
        )(state, state.params, batch)

        state = state.apply_gradients(grads=grads)
        return state, loss

    @partial(jit, static_argnums=(0,))
    def _eval_step(
        self,
        state: TrainState,
        batch: Tuple[Array, Array],
    ):
        x, y = batch
        y_predict = state.apply_fn(state.params, x)

        return {
            "loss": binary_cross_entropy(y_predict, y),
        }


class PositionBasedModel(nn.Module):
    n_documents: int
    n_ranks: int

    def setup(self) -> None:
        self.examination = nn.Sequential([nn.Embed(self.n_ranks, 1), nn.sigmoid])
        self.relevance = nn.Sequential([nn.Embed(self.n_documents, 1), nn.sigmoid])

    def __call__(self, x: Array) -> Array:
        examination = self._get_examination()
        relevance = self._get_relevance(x)
        return examination * relevance

    def _get_relevance(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.relevance(x).squeeze()

    def _get_examination(self) -> jnp.ndarray:
        ranks = jnp.arange(self.n_ranks)
        return self.examination(ranks).squeeze()


class CascadeModel(nn.Module):
    n_documents: int

    def setup(self) -> None:
        self.relevance = nn.Sequential([nn.Embed(self.n_documents, 1), nn.sigmoid])

    def __call__(self, x: Array) -> Array:
        relevance = self._get_relevance(x)
        examination = self._get_examination(relevance)
        return examination * relevance

    def _get_relevance(self, x: Array) -> Array:
        return self.relevance(x).squeeze()

    def _get_examination(self, relevance: Array) -> Array:
        non_relevance = 1 - relevance
        examination = jnp.roll(non_relevance, shift=1, axis=-1)
        examination = examination.at[:, 0].set(1)
        return jnp.cumprod(examination, axis=-1)

from functools import partial
from typing import Callable, Tuple, Dict, Mapping, List

import jax
import optax
from flax import linen as nn
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from jax import jit, Array
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultr_toolbox.click_models.data import ClickDataset, np_collate
from ultr_toolbox.click_models.metrics import Metric
from ultr_toolbox.click_models.neural.loss import binary_cross_entropy


class NeuralTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Callable = optax.adam(learning_rate=0.001),
        early_stopping: EarlyStopping = EarlyStopping(min_delta=0.0001, patience=3),
        max_epochs: int = 250,
        n_batch: int = 512,
        n_workers: int = 4,
        random_state: int = 42,
    ):
        self.model = model
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.max_epochs = max_epochs
        self.n_batch = n_batch
        self.n_workers = n_workers
        self.random_state = random_state
        self.model_state = None

    def fit(self, train_dataset: ClickDataset, val_dataset: ClickDataset):
        train_loader = self._get_dataloader(train_dataset, shuffle=True)
        val_loader = self._get_dataloader(val_dataset)

        # Initialize model
        key = jax.random.PRNGKey(self.random_state)
        x, y = next(iter(train_loader))
        params = self.model.init(key, x, y)
        model_state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
        )

        best_model_state = None

        for epoch in range(self.max_epochs):
            for batch in tqdm(train_loader, f"Epoch: {epoch} - Training"):
                model_state, train_loss = self._train_step(model_state, batch)

            val_loss = 0

            for batch in tqdm(val_loader, f"Epoch: {epoch} - Validation"):
                val_loss += self._validation_step(model_state, batch)

            val_loss = val_loss / len(val_loader)
            has_improved, self.early_stopping = self.early_stopping.update(val_loss)

            if has_improved:
                best_model_state = model_state

            if self.early_stopping.should_stop:
                break

        self.model_state = best_model_state

    def test(self, dataset: ClickDataset, metrics: List[Metric]) -> Dict:
        loader = self._get_dataloader(dataset, parallelize=False)

        for batch in tqdm(loader, "Testing"):
            x, y = batch
            y_predict = self._test_step(self.model_state, batch)

            for metric in metrics:
                metric.update(y_predict, y)

        return {metric.name: metric.compute() for metric in metrics}

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
            y_predict = state.apply_fn(params, x, y)
            return binary_cross_entropy(y_predict, y)

        # Differentiate w.r.t to parameters: argnums=1
        loss, grads = jax.value_and_grad(loss_fn, argnums=1)(state, state.params, batch)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @partial(jit, static_argnums=(0,))
    def _validation_step(
        self,
        state: TrainState,
        batch: Tuple[Array, Array],
    ):
        x, y = batch
        y_predict = state.apply_fn(state.params, x, y)
        loss = binary_cross_entropy(y_predict, y)

        return loss

    @partial(jit, static_argnums=(0,))
    def _test_step(
        self,
        state: TrainState,
        batch: Tuple[Array, Array],
    ):
        x, y = batch

        return state.apply_fn(state.params, x, y)

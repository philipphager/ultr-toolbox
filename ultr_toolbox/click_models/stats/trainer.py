from typing import Dict, List

from jax import Array
import jax.numpy as jnp
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultr_toolbox.click_models.data import ClickDataset, np_collate
from ultr_toolbox.click_models.metrics import Metric
from ultr_toolbox.click_models.stats.models import StatsModel


class StatsTrainer:
    def __init__(self, model: StatsModel, n_batch: int = 10_000):
        self.model = model
        self.n_batch = n_batch
        self.model_state = None

    def fit(self, train_dataset: ClickDataset, val_dataset: ClickDataset):
        self.model_state = self.model.params(train_dataset)

    def test(self, dataset: ClickDataset, metrics: List[Metric]) -> Dict:
        loader = DataLoader(dataset, batch_size=10_000, collate_fn=np_collate)

        for batch in tqdm(loader, "Testing"):
            x, y = batch
            y_predict = self.model.predict(self.model_state, x)

            for metric in metrics:
                metric.update(y_predict, y)

        return {metric.name: metric.compute() for metric in metrics}

    def predict(self, dataset: ClickDataset) -> Array:
        loader = DataLoader(dataset, batch_size=10_000, collate_fn=np_collate)
        y_predicts = []

        for batch in tqdm(loader, "Predicting"):
            x, y = batch
            y_predict = self.model.predict(self.model_state, x)
            y_predicts.append(y_predict)

        return jnp.vstack(y_predicts)

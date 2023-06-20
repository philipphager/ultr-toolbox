from abc import ABC, abstractmethod

from jax import Array

from ultr_toolbox.data import ClickDataset


class Trainer(ABC):
    @abstractmethod
    def train(self, train_dataset: ClickDataset, val_dataset: ClickDataset) -> Array:
        pass

    @abstractmethod
    def test(self, test_dataset: ClickDataset) -> Array:
        pass

from abc import ABC, abstractmethod
from typing import Dict

from ultr_toolbox.data import ClickDataset


class Trainer(ABC):
    @abstractmethod
    def train(self, train_dataset: ClickDataset, val_dataset: ClickDataset):
        pass

    @abstractmethod
    def test(self, test_dataset: ClickDataset) -> Dict:
        pass

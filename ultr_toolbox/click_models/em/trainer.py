from typing import Dict, List

import numpy as np
from pyclick.click_models import ClickModel
from pyclick.click_models.task_centric.TaskCentricSearchSession import (
    TaskCentricSearchSession,
)
from pyclick.search_session import SearchResult
from tqdm import tqdm

from ultr_toolbox.click_models.data import ClickDataset
from ultr_toolbox.click_models.metrics import Metric


class PyClickTrainer:
    def __init__(self, model: ClickModel):
        self.model = model

    def fit(self, train_dataset: ClickDataset, val_dataset: ClickDataset):
        sessions = [self._to_session(x, y) for x, y in train_dataset]
        self.model.train(sessions)

    def test(self, dataset: ClickDataset, metrics: List[Metric]) -> Dict:
        for x, y in tqdm(dataset, "Testing"):
            session = self._to_session(x, y)
            y_predict = np.array(self.model.get_full_click_probs(session))

            for metric in metrics:
                metric.update(y_predict, y)

        return {metric.name: metric.compute() for metric in metrics}

    @staticmethod
    def _to_session(x: np.ndarray, y: np.ndarray) -> TaskCentricSearchSession:
        session = TaskCentricSearchSession("0", "0")
        results = [SearchResult(doc_id, click) for doc_id, click in zip(x, y)]
        session.web_results = results
        return session

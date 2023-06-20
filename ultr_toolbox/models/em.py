import numpy as np
from typing import List, Dict

import pandas as pd
from jax import Array
from pyclick.click_models import ClickModel
from pyclick.click_models.task_centric.TaskCentricSearchSession import (
    TaskCentricSearchSession,
)
from pyclick.search_session import SearchResult
from tqdm import tqdm

from ultr_toolbox.data import ClickDataset
from ultr_toolbox.metrics.click_metrics import perplexity, binary_cross_entropy
from ultr_toolbox.models.base import Trainer


class EMTrainer(Trainer):
    def __init__(self, model: ClickModel):
        self.model = model

    def train(self, train_dataset: ClickDataset, val_dataset: ClickDataset):
        print("Training")
        sessions = [self._to_session(x, y) for x, y in train_dataset]
        self.model.train(sessions)

    def test(self, test_dataset: ClickDataset) -> Dict:
        metrics = []

        for x, y in tqdm(test_dataset, "Testing"):
            session = self._to_session(x, y)
            y_predict = np.array(self.model.get_full_click_probs(session))
            metric = {
                "perplexity": perplexity(y_predict, y),
                "cross_entropy": binary_cross_entropy(y_predict, y),
            }
            metrics.append(metric)

        return pd.DataFrame(metrics).mean(axis=0).to_dict()

    @staticmethod
    def _to_session(x: np.ndarray, y: np.ndarray) -> TaskCentricSearchSession:
        session = TaskCentricSearchSession("0", "0")
        results = [SearchResult(doc_id, click) for doc_id, click in zip(x, y)]
        session.web_results = results
        return session

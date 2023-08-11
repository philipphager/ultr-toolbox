import numpy as np
import pandas as pd

from ultr_toolbox.click_models.data import ClickDataset
from ultr_toolbox.click_models.metrics import Perplexity, LogLikelihood, RocAuc
from ultr_toolbox.click_models.neural import (
    PositionBasedModel,
    UserBrowsingModel,
    CascadeModel,
    DynamicBayesianNetwork,
    NeuralTrainer,
    DependentClickModel,
)
from ultr_toolbox.click_models.neural.base import NeuralModel
from ultr_toolbox.click_models.stats import (
    RandomModel,
    RankBasedModel,
    JointModel,
    RankDocumentBasedModel,
    DocumentBasedModel,
    StatsTrainer,
)
from ultr_toolbox.click_models.stats.models import StatsModel


def main():
    train_df = pd.read_pickle("data/train-eps.pckl")
    val_df = pd.read_pickle("data/val-eps.pckl")
    test_df = pd.read_pickle("data/test-eps.pckl")

    train_dataset = ClickDataset(train_df)
    val_dataset = ClickDataset(val_df)
    test_dataset = ClickDataset(test_df)

    n_items = train_dataset.n_items()
    n_ranks = train_dataset.n_ranks()

    models = {
        "GCTR": RandomModel(),
        "RCTR": RankBasedModel(),
        "DCTR": DocumentBasedModel(),
        "RDCTR": RankDocumentBasedModel(),
        "JCTR": JointModel(),
        "PBM": PositionBasedModel(n_items=n_items, n_ranks=n_ranks),
        "CM": CascadeModel(n_items=n_items),
        "DCM": DependentClickModel(n_items=n_items),
        "UBM": UserBrowsingModel(n_items=n_items, n_ranks=n_ranks),
        "DBN": DynamicBayesianNetwork(n_items=n_items),
        "SDBN": DynamicBayesianNetwork(n_items=n_items, estimate_continuation=False),
    }

    results = []
    predictions = []

    for name, model in models.items():
        metrics = [
            Perplexity(),
            Perplexity(aggregate_ranks=False),
            LogLikelihood(),
            RocAuc(),
            RocAuc(aggregate_ranks=False),
        ]

        if isinstance(model, StatsModel):
            trainer = StatsTrainer(model)
        elif isinstance(model, NeuralModel):
            trainer = NeuralTrainer(model)

        trainer.fit(train_dataset, val_dataset)

        metrics = trainer.test(test_dataset, metrics)
        metrics["model"] = name

        results.append(metrics)
        df = pd.DataFrame(results)
        df.to_parquet("results.parquet")

        y_predict = trainer.predict(test_dataset)
        prediction_df = pd.DataFrame(
            {
                "model": name,
                "x": list(np.array(test_dataset.x)),
                "y": list(np.array(test_dataset.y)),
                "y_predict": list(np.array(y_predict)),
            }
        )

        predictions.append(prediction_df)
        prediction_df = pd.concat(predictions)
        prediction_df.to_parquet("predictions.parquet")

    print(df.to_string())


if __name__ == "__main__":
    main()

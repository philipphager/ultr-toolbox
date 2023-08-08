import pandas as pd
from sklearn.model_selection import train_test_split

from ultr_toolbox.click_models.data import ClickDataset
from ultr_toolbox.click_models.metrics import Perplexity, LogLikelihood
from ultr_toolbox.click_models.neural import (
    PositionBasedModel,
    UserBrowsingModel,
    CascadeModel,
    DynamicBayesianNetwork,
    NeuralTrainer,
)
from ultr_toolbox.click_models.neural.base import NeuralModel
from ultr_toolbox.click_models.stats import (
    RandomModel,
    RankBasedModel,
    JointModel,
    RankDocumentBasedModel,
    DocumentBasedModel, StatsTrainer,
)
from ultr_toolbox.click_models.stats.models import StatsModel


def main():
    df = pd.read_parquet("data/yandex-sample.parquet")

    train_df, test_df = train_test_split(df, random_state=0)
    train_df, val_df = train_test_split(train_df, random_state=0)

    train_dataset = ClickDataset(train_df, "doc_ids", "click")
    val_dataset = ClickDataset(val_df, "doc_ids", "click")
    test_dataset = ClickDataset(test_df, "doc_ids", "click")

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
        "UBM": UserBrowsingModel(n_items=n_items, n_ranks=n_ranks),
        "DBN": DynamicBayesianNetwork(n_items=n_items),
        "SDBN": DynamicBayesianNetwork(n_items=n_items, estimate_continuation=False),
    }

    results = []

    for name, model in models.items():
        metrics = [Perplexity(), Perplexity(aggregate_ranks=False), LogLikelihood()]

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

    print(df.to_string())


if __name__ == "__main__":
    main()

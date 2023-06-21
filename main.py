import pandas as pd
from pyclick.click_models import PBM
from sklearn.model_selection import train_test_split

from ultr_toolbox.data import ClickDataset
from ultr_toolbox.models.ctr import CTRTrainer, RankBasedModel
from ultr_toolbox.models.em import EMTrainer

if __name__ == "__main__":
    path = "data/clicks.parquet"
    df = pd.read_parquet(path)

    df = df.head(100_000)
    df.to_parquet("data/yandex-sample.parquet")

    train_df, test_df = train_test_split(df, random_state=0)
    train_df, val_df = train_test_split(train_df, random_state=0)

    train_dataset = ClickDataset(train_df)
    val_dataset = ClickDataset(val_df)
    test_dataset = ClickDataset(test_df)

    trainer = CTRTrainer(RankBasedModel())
    trainer = EMTrainer(PBM())
    trainer.train(train_dataset, val_dataset)
    metrics = trainer.test(test_dataset)

    print(metrics)
